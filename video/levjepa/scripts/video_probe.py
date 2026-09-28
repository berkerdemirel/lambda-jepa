"""SSv2 (attentive) and K400 (linear on the token mean) frozen probes on clip-file stores, V-JEPA protocol."""
import argparse, csv, io, math, os, random, struct, sys, time, contextlib
import numpy as np, torch, torch.distributed as dist, torch.nn as nn, torch.nn.functional as F
from PIL import Image
from timm.data.auto_augment import rand_augment_ops, _RAND_INCREASING_TRANSFORMS

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from attentive_probe import AttentiveClassifier, WarmupCosineSchedule, CosineWDSchedule, load_encoder

MEAN, STD = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
PRESETS = {"ssv2": dict(frame_step=4, segments=2, views=3), "k400": dict(frame_step=2, segments=8, views=3)}

def segment_indices(n, fpc, fstp, num_clips, random_clip_sampling):
    clip_len = int(fpc * fstp); partition_len = n // num_clips; out = []
    for i in range(num_clips):
        if partition_len > clip_len:
            end = np.random.randint(clip_len, partition_len) if random_clip_sampling else clip_len
            start = end - clip_len
            idx = np.clip(np.linspace(start, end, num=fpc), start, end - 1).astype(np.int64) + i * partition_len
        else:
            sample_len = min(clip_len, n) - 1
            idx = np.linspace(0, sample_len, num=sample_len // fstp)
            idx = np.concatenate((idx, np.ones(fpc - sample_len // fstp) * sample_len))
            idx = np.clip(idx, 0, sample_len - 1).astype(np.int64)
            clip_step = (n - clip_len) // (num_clips - 1) if (n > clip_len and num_clips > 1) else 0
            idx = idx + i * clip_step
        out.append(idx)
    return out

class ClipRandAugment:
    def __init__(self, crop=224, magnitude=7, num_layers=4, mstd=0.5):
        hp = {"translate_const": int(crop * 0.45), "interpolation": Image.BICUBIC, "magnitude_std": mstd}
        self.ops, self.n = rand_augment_ops(magnitude=magnitude, prob=0.5, hparams=hp, transforms=_RAND_INCREASING_TRANSFORMS), num_layers

    def __call__(self, frames):
        for op in np.random.choice(self.ops, self.n, replace=True):
            if op.prob < 1.0 and random.random() > op.prob:
                continue
            m = op.magnitude
            if op.magnitude_std > 0:
                m = random.gauss(m, op.magnitude_std)
            m = min(10.0, max(0.0, m))
            args = op.level_fn(m, op.hparams) if op.level_fn is not None else ()
            frames = [op.aug_fn(f, *args, **op.kwargs) for f in frames]
        return frames

def crop_params(scale, ratio, h, w):
    for _ in range(10):
        area = random.uniform(*scale) * h * w
        ar = math.exp(random.uniform(math.log(ratio[0]), math.log(ratio[1])))
        cw, ch = int(round(math.sqrt(area * ar))), int(round(math.sqrt(area / ar)))
        if 0 < cw <= w and 0 < ch <= h:
            return random.randint(0, h - ch), random.randint(0, w - cw), ch, cw
    in_ratio = w / h
    if in_ratio < min(ratio): cw, ch = w, int(round(w / min(ratio)))
    elif in_ratio > max(ratio): ch, cw = h, int(round(h * max(ratio)))
    else: cw, ch = w, h
    return (h - ch) // 2, (w - cw) // 2, ch, cw

def erase_cube(x, p=0.25, min_area=0.02, max_area=1 / 3, min_aspect=0.3):
    if random.random() > p:
        return x
    T, C, H, W = x.shape
    for _ in range(100):
        area = random.uniform(min_area, max_area) * H * W
        ar = math.exp(random.uniform(math.log(min_aspect), math.log(1 / min_aspect)))
        h, w = int(round(math.sqrt(area * ar))), int(round(math.sqrt(area / ar)))
        if w < W and h < H:
            top, left = random.randint(0, H - h), random.randint(0, W - w)
            for t in range(T):
                x[t, :, top:top + h, left:left + w] = torch.empty((C, h, w), dtype=x.dtype).normal_()
            break
    return x

class TrainTransform:
    def __init__(self, crop=224):
        self.crop, self.ra = crop, ClipRandAugment(crop)
        self.mean, self.std = torch.tensor(MEAN), torch.tensor(STD)

    def __call__(self, frames):
        frames = self.ra([Image.fromarray(f) for f in frames])
        x = torch.stack([torch.from_numpy(np.array(f, dtype=np.uint8)) for f in frames]).float() / 255.0
        x = ((x - self.mean) / self.std).permute(3, 0, 1, 2)
        i, j, h, w = crop_params((0.08, 1.0), (0.75, 4 / 3), x.shape[2], x.shape[3])
        x = F.interpolate(x[:, :, i:i + h, j:j + w], size=(self.crop, self.crop), mode="bilinear", align_corners=False)
        return erase_cube(x.permute(1, 0, 2, 3)).permute(1, 0, 2, 3).contiguous()

class EvalTransform:
    def __init__(self, crop=224, views=3):
        self.crop, self.views = crop, views
        self.mean, self.std = torch.tensor(MEAN).view(3, 1, 1, 1), torch.tensor(STD).view(3, 1, 1, 1)

    def __call__(self, frames):
        import cv2
        h, w = frames[0].shape[:2]
        if not ((w <= h and w == self.crop) or (h <= w and h == self.crop)):
            oh, ow = (int(self.crop * h / w), self.crop) if w < h else (self.crop, int(self.crop * w / h))
            frames = [cv2.resize(f, (ow, oh), interpolation=cv2.INTER_LINEAR) for f in frames]
        buf = np.stack(frames); T, H, W, _ = buf.shape
        step = (max(H, W) - self.crop) // max(1, self.views - 1)
        out = []
        for v in range(self.views):
            s = v * step
            view = buf[:, s:s + self.crop, :, :] if H > W else buf[:, :, s:s + self.crop, :]
            x = torch.from_numpy(np.ascontiguousarray(view)).permute(3, 0, 1, 2).float() / 255.0
            out.append((x - self.mean) / self.std)
        return out

def list_store(root, classes_file, limit=0, seed=0):
    classes = [c for c in open(classes_file).read().split("\n") if c]
    index = {c.replace("/", "_"): i for i, c in enumerate(classes)}
    files, labels = [], []
    for c in sorted(d for d in os.listdir(root) if os.path.isdir(os.path.join(root, d))):
        for f in sorted(os.listdir(os.path.join(root, c))):
            if f.endswith(".clip"):
                files.append(os.path.join(root, c, f)); labels.append(index[c])
    if limit:
        g = random.Random(seed); keep = sorted(g.sample(range(len(files)), min(limit, len(files))))
        files, labels = [files[i] for i in keep], [labels[i] for i in keep]
    return files, labels, len(classes)

def stage_shard(files, src_root, dst_root, rank):
    import shutil
    from concurrent.futures import ThreadPoolExecutor
    t0 = time.time(); dst = [os.path.join(dst_root, os.path.relpath(f, src_root)) for f in files]
    for d in {os.path.dirname(x) for x in dst}:
        os.makedirs(d, exist_ok=True)
    def cp(pair):
        if not (os.path.exists(pair[1]) and os.path.getsize(pair[1]) == os.path.getsize(pair[0])):
            shutil.copyfile(pair[0], pair[1])
        return os.path.getsize(pair[1])
    with ThreadPoolExecutor(16) as ex:
        nbytes = sum(ex.map(cp, zip(files, dst)))
    print(f"[video-probe] rank {rank}: staged {len(files)} files, {nbytes / 1e9:.1f} GB to {dst_root} in {(time.time() - t0) / 60:.1f} min", flush=True)
    return dst

class ProbeClipDataset(torch.utils.data.Dataset):
    def __init__(self, files, labels, num_classes, fpc, fstp, segments, training, views):
        self.files, self.labels, self.num_classes = files, labels, num_classes
        self.fpc, self.fstp, self.segments, self.training = fpc, fstp, segments, training
        self.tf = TrainTransform() if training else EvalTransform(views=views)

    def __len__(self): return len(self.files)

    def __getitem__(self, i):
        with open(self.files[i], "rb") as f:
            n = struct.unpack("<I", f.read(4))[0]
            offs = np.frombuffer(f.read(8 * (n + 1)), dtype=np.int64); blob = f.read()
        segs = segment_indices(n, self.fpc, self.fstp, self.segments, random_clip_sampling=True)
        cache = {}
        def frame(k):
            if k not in cache:
                cache[k] = np.asarray(Image.open(io.BytesIO(blob[int(offs[k]):int(offs[k + 1])])).convert("RGB"))
            return cache[k]
        clips = []
        for idx in segs:
            frames = [frame(int(k)) for k in idx]
            clips.append([self.tf(frames)] if self.training else self.tf(frames))
        return clips, self.labels[i]

class LinearMean(nn.Module):
    def __init__(self, dim, num_classes):
        super().__init__(); self.linear = nn.Linear(dim, num_classes)
    def forward(self, x): return self.linear(x.mean(1))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt"); ap.add_argument("--weights", default="ema", choices=["ema", "student"])
    ap.add_argument("--data", required=True, help="store root with train/ and val|validation/ + classes.txt")
    ap.add_argument("--dataset", required=True, choices=list(PRESETS)); ap.add_argument("--head", default="attentive", choices=["attentive", "linear_mean"])
    ap.add_argument("--out", required=True); ap.add_argument("--epochs", type=int, default=20); ap.add_argument("--bs", type=int, default=4)
    ap.add_argument("--lr", type=float, default=1e-3); ap.add_argument("--wd", type=float, default=0.01); ap.add_argument("--final-wd", type=float, default=1e-6)
    ap.add_argument("--workers", type=int, default=8); ap.add_argument("--limit", type=int, default=0, help="smoke: clips per split")
    ap.add_argument("--frame-step", type=int); ap.add_argument("--segments", type=int); ap.add_argument("--views", type=int)
    ap.add_argument("--smoke", action="store_true", help="random vit_tiny, CPU, synthetic store")
    ap.add_argument("--stage-local", default="", help="node-local dir: this rank's shard of both splits is copied there once at start")
    a = ap.parse_args()
    P = PRESETS[a.dataset]; fstp, segs, views = a.frame_step or P["frame_step"], a.segments or P["segments"], a.views or P["views"]
    ddp = "RANK" in os.environ
    if ddp:
        dist.init_process_group("nccl"); rank, world = dist.get_rank(), dist.get_world_size()
        torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    else:
        rank, world = 0, 1
    dev = torch.device("cpu" if a.smoke else "cuda")
    if a.smoke:
        import module as vit_models
        enc = vit_models.vit_tiny(img_size=224, patch_size=16, num_frames=16, tubelet_size=1, token_drop_rate=0.0, attn_mode="block_causal")
    else:
        enc, _ = load_encoder(a.ckpt, a.weights)
    enc.to(dev).eval()
    for p in enc.parameters():
        p.requires_grad_(False)
    val_dir = next(d for d in ("val", "validation") if os.path.isdir(os.path.join(a.data, d)))
    classes = os.path.join(a.data, "classes.txt")
    random.seed(rank); np.random.seed(rank); torch.manual_seed(1000 + rank)
    splits = {}
    for name, sub in (("train", "train"), ("val", val_dir)):
        files, labels, nc = list_store(os.path.join(a.data, sub), classes, a.limit)
        files, labels = files[rank::world], labels[rank::world]
        if name == "train" and world > 1:
            n = (len(list_store(os.path.join(a.data, sub), classes, a.limit)[0]) // world); files, labels = files[:n], labels[:n]
        if a.stage_local:
            files = stage_shard(files, a.data, a.stage_local, rank)
        splits[name] = (files, labels, nc)
    train_ds = ProbeClipDataset(*splits["train"], 16, fstp, segs, True, views)
    val_ds = ProbeClipDataset(*splits["val"], 16, fstp, segs, False, views)
    head = AttentiveClassifier(enc.embed_dim, enc.num_heads, train_ds.num_classes) if a.head == "attentive" else LinearMean(enc.embed_dim, train_ds.num_classes)
    head = head.to(dev)
    if ddp:
        head = nn.parallel.DistributedDataParallel(head, device_ids=[dev.index])
    mk = lambda ds, sh: torch.utils.data.DataLoader(ds, batch_size=a.bs, shuffle=sh, num_workers=a.workers,
                                                    pin_memory=not a.smoke, persistent_workers=a.workers > 0, drop_last=False)
    train_loader, val_loader = mk(train_ds, True), mk(val_ds, False)
    params = list((head.module if ddp else head).named_parameters())
    groups = [{"params": [p for n, p in params if "bias" not in n and p.ndim != 1]},
              {"params": [p for n, p in params if "bias" in n or p.ndim == 1], "WD_exclude": True, "weight_decay": 0}]
    opt = torch.optim.AdamW(groups); ipe = len(train_loader)
    lr_s = WarmupCosineSchedule(opt, 0, a.lr, a.lr, a.epochs * ipe, final_lr=0.0); wd_s = CosineWDSchedule(opt, a.wd, a.epochs * ipe, final_wd=a.final_wd)
    if rank == 0:
        print(f"[video-probe] {a.dataset} head={a.head} ckpt={a.ckpt} weights={a.weights} dim={enc.embed_dim} | train {len(train_ds)}x{world} val {len(val_ds)}x{world} "
              f"({train_ds.num_classes} classes; per-rank shards{', staged to ' + a.stage_local if a.stage_local else ''}) | frame_step {fstp} segments {segs} views {views} | "
              f"bs {a.bs}x{world}={a.bs * world} ipe {ipe} epochs {a.epochs}", flush=True)
        os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    amp = contextlib.nullcontext() if a.smoke else torch.autocast("cuda", dtype=torch.bfloat16)

    def encode_views(clips):
        S, V = len(clips), len(clips[0]); B = clips[0][0].shape[0]
        x = torch.cat([clips[s][v] for v in range(V) for s in range(S)], 0).to(dev, non_blocking=True)
        with torch.no_grad(), amp:
            tok = enc(x)
        tok = tok.view(V, S, B, tok.shape[1], tok.shape[2])
        return [tok[v].permute(1, 0, 2, 3).reshape(B, S * tok.shape[3], tok.shape[4]) for v in range(V)]

    def run(loader, training, epoch):
        head.train(training)
        correct = torch.zeros((), device=dev); seen = torch.zeros((), device=dev)
        for clips, labels in loader:
            labels = labels.to(dev, non_blocking=True)
            if training:
                lr_s.step(); wd_s.step()
            views = encode_views(clips)
            with amp, (contextlib.nullcontext() if training else torch.no_grad()):
                logits = [head(v) for v in views]
            loss = sum(F.cross_entropy(l.float(), labels) for l in logits) / len(logits)
            if training:
                opt.zero_grad(set_to_none=True); loss.backward()
                torch.nn.utils.clip_grad_norm_(head.parameters(), 1.0); opt.step()
            with torch.no_grad():
                prob = sum(F.softmax(l.float(), 1) for l in logits) / len(logits)
                correct += (prob.argmax(1) == labels).sum(); seen += len(labels)
        if ddp:
            dist.all_reduce(correct); dist.all_reduce(seen)
        return 100.0 * correct.item() / seen.item()

    rows = []
    for ep in range(a.epochs):
        t0 = time.time(); tr = run(train_loader, True, ep); va = run(val_loader, False, ep)
        if rank == 0:
            print(f"[video-probe] ep{ep + 1}/{a.epochs} train_top1={tr:.2f} val_top1={va:.2f} lr={opt.param_groups[0]['lr']:.2e} ({(time.time() - t0) / 60:.1f} min)", flush=True)
            rows.append({"ckpt": a.ckpt, "weights": a.weights, "dataset": a.dataset, "head": a.head, "epoch": ep + 1, "train_top1": tr, "val_top1": va})
            with open(a.out, "w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    if rank == 0:
        print(f"[video-probe] final val_top1={rows[-1]['val_top1']:.2f} best={max(r['val_top1'] for r in rows):.2f} -> {a.out}", flush=True)
    if ddp:
        dist.destroy_process_group()

if __name__ == "__main__":
    main()
