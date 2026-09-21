"""Cache the frozen clip-level features that the K400 protocol reads, for several encoder states in ONE pass (E34, Berker
2026-09-12: "cache the feats and do your thing"). Why: the K400 column of tab:video is a 20-epoch AdamW linear head on the mean of
all output tokens (D-119), and it moved 0.20 across the 240 -> 1085 B ladder while the attentive columns moved ~4.2; the cached
features let optimizer-free readers (kNN, a convex L-BFGS fit) and the geometry of the pooled vector be read on the SAME clips for
every state (analysis: experiments/e34_k400_features.py). Sampling = the protocol's val sampling (video_probe.py: 16 frames at
frame_step 2 on the 15-fps store, 8 segments, allow_clip_overlap -> deterministic for K400 clip lengths), ONE spatial view per
segment = the centre crop of the protocol's three (short side 224); no augmentation. Per clip and state, averaged over the 8 segments:
  pool  mean over ALL tokens of a segment (CLS + 16 x 196 patches) = the protocol's pooled vector (LinearMean in video_probe.py)
  gap   mean over the patch tokens only        cls   the CLS token
  scalars: tok_norm (mean patch-token norm), cls_norm, within_var (mean ||patch - segment mean||^2 = token variance inside a
  segment), seg_var (mean ||segment gap - clip gap||^2 = variation across the 8 segments); the between-clip part is read from the
  cached vectors. Tokens are the encoder's LayerNorm-ed output, as both probes see them. Features fp16, scalars fp32.
Each rank owns a fixed shard (files sorted by class then name, rank::world; the same split rule as video_probe.py) and writes
<out_dir>/<split>_rank<r>.pt; --stage-local copies the shard to node-local disk once (the BeeGFS per-node read cap). No DDP: ranks
are independent (RANK / WORLD_SIZE / LOCAL_RANK from the launcher). --limit N caches a seeded N-clip subset per split (smoke)."""
import argparse, io, os, struct, sys, time
import numpy as np, torch
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from attentive_probe import load_encoder  # noqa: E402
from video_probe import EvalTransform, PRESETS, list_store, segment_indices, stage_shard  # noqa: E402


class CenterClips(torch.utils.data.Dataset):
    """clip file -> list over segments of (C, T, 224, 224) centre views (the protocol's middle crop), label."""
    def __init__(self, files, labels, fpc=16, fstp=2, segments=8):
        self.files, self.labels, self.fpc, self.fstp, self.segments = files, labels, fpc, fstp, segments
        self.tf = EvalTransform(views=3)

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
        return [self.tf([frame(int(k)) for k in idx])[1] for idx in segs], self.labels[i]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpts", nargs="+", required=True, help="name=path, e.g. b240=/…/epoch-0239.ckpt")
    ap.add_argument("--weights", default="ema", choices=["ema", "student"])
    ap.add_argument("--data", required=True, help="store root with train/ and val/ + classes.txt")
    ap.add_argument("--out-dir", required=True); ap.add_argument("--bs", type=int, default=8); ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--limit", type=int, default=0); ap.add_argument("--stage-local", default="")
    a = ap.parse_args()
    P = PRESETS["k400"]
    rank, world = int(os.environ.get("RANK", 0)), int(os.environ.get("WORLD_SIZE", 1))
    dev = torch.device("cuda", int(os.environ.get("LOCAL_RANK", 0))); torch.cuda.set_device(dev)
    encs = {}
    for spec in a.ckpts:
        name, path = spec.split("=", 1)
        enc, hp = load_encoder(path, a.weights); enc.to(dev).eval()
        for p in enc.parameters():
            p.requires_grad_(False)
        encs[name] = enc
        if rank == 0:
            print(f"[k400-cache] {name}: {path} ({hp['model']['name']}, {a.weights})", flush=True)
    os.makedirs(a.out_dir, exist_ok=True)
    classes = os.path.join(a.data, "classes.txt")
    for split in ("train", "val"):
        files, labels, nc = list_store(os.path.join(a.data, split), classes, a.limit)
        files, labels = files[rank::world], labels[rank::world]
        rel = [os.path.relpath(f, a.data) for f in files]
        if a.stage_local:
            files = stage_shard(files, a.data, a.stage_local, rank)
        ds = CenterClips(files, labels, 16, P["frame_step"], P["segments"])
        loader = torch.utils.data.DataLoader(ds, batch_size=a.bs, shuffle=False, num_workers=a.workers, pin_memory=True,
                                             persistent_workers=a.workers > 0)
        out = {n: {k: [] for k in ("cls", "gap", "pool", "tok_norm", "cls_norm", "within_var", "seg_var")} for n in encs}
        t0 = time.time(); seen = 0
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            for bi, (clips, y) in enumerate(loader):
                S, B = len(clips), clips[0].shape[0]
                x = torch.cat(clips, 0).to(dev, non_blocking=True)            # (S*B, C, T, H, W), segment-major
                for n, enc in encs.items():
                    tok = enc(x).float()                                        # (S*B, L, D), LayerNorm-ed, index 0 = CLS
                    tok = tok.view(S, B, tok.shape[1], tok.shape[2])
                    cls, patches, pool = tok[:, :, 0], tok[:, :, 1:], tok.mean(2)
                    gap = patches.mean(2)                                       # (S, B, D)
                    within = ((patches - gap[:, :, None]) ** 2).sum(-1).mean(-1)  # (S, B)
                    gap_clip = gap.mean(0)
                    o = out[n]
                    o["cls"].append(cls.mean(0).half().cpu()); o["gap"].append(gap_clip.half().cpu()); o["pool"].append(pool.mean(0).half().cpu())
                    o["tok_norm"].append(patches.norm(dim=-1).mean(-1).mean(0).cpu()); o["cls_norm"].append(cls.norm(dim=-1).mean(0).cpu())
                    o["within_var"].append(within.mean(0).cpu()); o["seg_var"].append(((gap - gap_clip[None]) ** 2).sum(-1).mean(0).cpu())
                seen += B
                if rank == 0 and bi % 50 == 0:
                    print(f"[k400-cache] {split} {seen}/{len(ds)} clips in {time.time() - t0:.0f}s", flush=True)
        blob = {"split": split, "rank": rank, "world": world, "files": rel, "labels": torch.tensor(labels), "num_classes": nc,
                "weights": a.weights, "states": {n: {k: torch.cat(v) for k, v in o.items()} for n, o in out.items()}}
        torch.save(blob, os.path.join(a.out_dir, f"{split}_rank{rank}.pt"))
        print(f"[k400-cache] rank {rank}: {split} {len(ds)} clips cached in {(time.time() - t0) / 60:.1f} min", flush=True)
    print("K400_FEATURES_DONE", flush=True)


if __name__ == "__main__":
    main()
