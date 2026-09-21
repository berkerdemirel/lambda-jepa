"""K710 clip probe on saved checkpoints (E34 peek; Berker 2026-09-04: "some sort of a positive curve ... see if this
video ssl training works at all"). One CLEAN view per clip — the center 31-frame span at stride 2, short side
resized to 224, center crop, no augmentation — and the encoder's CLS (h.cls, the audit's backbone tap), the mean of its
patch tokens (h.gap; LeVJEPA never scores CLS alone: their attentive probe attends over all tokens) plus the student
projector's output (z) on a fixed class-balanced split of the pretraining clips (up to 20 train + 8 eval clips per
class, seed 0; the SSL loss never saw the labels), then k-NN (k 20, cosine, DINO's weighted vote) and a linear
classifier (multinomial logistic regression, Adam) on the frozen features. Every checkpoint's student and EMA
encoders are applied in ONE pass over the clips. Not the landing protocol (that is the IN-1k attentive probe): an
in-domain read of whether the representation organizes the K710 classes as training proceeds.
Usage: k710_probe.py --ckpts <ckpt ...> --out results/e34/k710_probe.csv"""
import argparse, csv, io, os, struct, sys, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import multiprocessing as mp
import numpy as np, torch, torch.nn.functional as F
from PIL import Image
from torchvision.transforms import v2
import module as vit_models
from attentive_probe import load_encoder
from data.clipfile_loader import ClipFileDataset
from main import to_float_normalized

ROOT = "/mnt/beegfs/locatgrp/shared/datasets/kinetics"
SETS = ("k710_clipfiles_k700_2020", "k710_clipfiles_k600", "k710_clipfiles_k400")
CLASSES = f"{ROOT}/k710_20pct_k400.lance.classes.txt"


class CenterClips(torch.utils.data.Dataset):
    """Fixed clip list -> {"frame": uint8 (T, 3, 224, 224), "label"}: center span, clean spatial view."""

    def __init__(self, files, labels, num_frames=16, frame_stride=2):
        self.files, self.labels, self.T, self.S = files, labels, num_frames, frame_stride
        self.tf = v2.Compose([v2.Resize(224, antialias=True), v2.CenterCrop(224)])

    def __len__(self): return len(self.files)

    def __getitem__(self, i):
        with open(self.files[i], "rb") as f:
            n = struct.unpack("<I", f.read(4))[0]
            offs = np.frombuffer(f.read(8 * (n + 1)), dtype=np.int64); base = 4 + 8 * (n + 1)
            span = self.S * (self.T - 1) + 1; start = max(n - span, 0) // 2; last = min(start + span - 1, n - 1)
            f.seek(base + int(offs[start])); blob = f.read(int(offs[last + 1] - offs[start]))
        idx = [min(start + k * self.S, n - 1) for k in range(self.T)]
        fr = [np.asarray(Image.open(io.BytesIO(blob[int(offs[k] - offs[start]): int(offs[k + 1] - offs[start])])).convert("RGB")) for k in idx]
        return {"frame": self.tf(torch.from_numpy(np.stack(fr)).permute(0, 3, 1, 2)), "label": self.labels[i]}


def split(per_class_train, per_class_eval, seed=0):
    """Class-balanced fixed split of the pretraining clips: files, labels, and a train/eval mask."""
    ds = ClipFileDataset([f"{ROOT}/{s}" for s in SETS], CLASSES, transform=None)
    g = torch.Generator().manual_seed(seed); files, labels, is_train = [], [], []
    by = {}
    for f, l in zip(ds.files, ds.labels): by.setdefault(l, []).append(f)
    for l in sorted(by):
        fs = [by[l][j] for j in torch.randperm(len(by[l]), generator=g).tolist()]
        for j, f in enumerate(fs[:per_class_train + per_class_eval]):
            files.append(f); labels.append(l); is_train.append(j < per_class_train)
    return files, labels, torch.tensor(is_train), len(ds.classes)


def knn_top1(ftr, ytr, fev, yev, k=20, T=0.07, C=724):
    a, b = F.normalize(ftr, dim=1), F.normalize(fev, dim=1); correct = 0
    for i in range(0, len(b), 1024):
        sim = b[i:i + 1024] @ a.T; w, j = sim.topk(k, dim=1); w = (w / T).exp()
        votes = torch.zeros(len(j), C, device=a.device).scatter_add_(1, ytr[j], w)
        correct += (votes.argmax(1) == yev[i:i + 1024]).sum().item()
    return 100.0 * correct / len(b)


def linear_top1(ftr, ytr, fev, yev, C=724, epochs=100, lr=1e-3, wd=1e-4):
    mu, sd = ftr.mean(0, keepdim=True), ftr.std(0, keepdim=True) + 1e-6
    ftr, fev = (ftr - mu) / sd, (fev - mu) / sd
    clf = torch.nn.Linear(ftr.shape[1], C).to(ftr.device); opt = torch.optim.Adam(clf.parameters(), lr=lr, weight_decay=wd)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs * ((len(ftr) + 1023) // 1024))
    for _ in range(epochs):
        perm = torch.randperm(len(ftr), device=ftr.device)
        for i in range(0, len(ftr), 1024):
            idx = perm[i:i + 1024]; loss = F.cross_entropy(clf(ftr[idx]), ytr[idx]); opt.zero_grad(); loss.backward(); opt.step(); sched.step()
    with torch.no_grad():
        return 100.0 * (clf(fev).argmax(1) == yev).float().mean().item(), 100.0 * (clf(ftr).argmax(1) == ytr).float().mean().item()


def read_safetensors(path):
    """Minimal safetensors reader (header JSON + raw little-endian tensors); avoids a new dependency."""
    import json
    dt = {"F32": torch.float32, "F16": torch.float16, "BF16": torch.bfloat16, "I64": torch.int64}
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]; hdr = json.loads(f.read(n)); base = 8 + n
        out = {}
        for k, v in hdr.items():
            if k == "__metadata__": continue
            a, b = v["data_offsets"]; f.seek(base + a); buf = f.read(b - a)
            out[k] = torch.frombuffer(bytearray(buf), dtype=dt[v["dtype"]]).reshape(v["shape"]).clone()
    return out


def load_reference(spec):
    """--ref name=path:arch — a released encoder (e.g. LeVJEPA's HF checkpoint, keys encoder.*) built with our module."""
    name, rest = spec.split("="); path, arch = rest.rsplit(":", 1)
    enc = getattr(vit_models, arch)(img_size=224, patch_size=16, num_frames=16, tubelet_size=1, use_rope=True, token_drop_rate=0.0, attn_mode="block_causal")
    sd = {k[len("encoder."):]: v for k, v in read_safetensors(path).items() if k.startswith("encoder.")}
    missing, unexpected = enc.load_state_dict(sd, strict=False)
    assert not unexpected and all(k == "pos_embed" for k in missing), (missing, unexpected)
    return name, enc


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--ckpts", nargs="*", default=[]); ap.add_argument("--out", required=True)
    ap.add_argument("--ref", nargs="*", default=[], help="name=path.safetensors:arch, e.g. levjepa_L_videomix=/x/model.safetensors:vit_large")
    ap.add_argument("--per-class-train", type=int, default=20); ap.add_argument("--per-class-eval", type=int, default=8)
    ap.add_argument("--bs", type=int, default=64); ap.add_argument("--workers", type=int, default=20); a = ap.parse_args()
    dev = torch.device("cuda")
    files, labels, is_train, C = split(a.per_class_train, a.per_class_eval)
    print(f"[k710-probe] {len(files)} clips ({int(is_train.sum())} train / {int((~is_train).sum())} eval), {C} classes", flush=True)
    models = []                                                       # (tag, epoch, weights, encoder, projector or None)
    for p in a.ckpts:
        ck = torch.load(p, map_location="cpu", weights_only=False); ep = int(ck["epoch"]); hp = ck["hyper_parameters"]
        for w in ("student", "ema"):
            enc, _ = load_encoder(p, w); enc.to(dev).eval(); proj = None
            if w == "student":
                proj = vit_models.Projector(input_dim=enc.embed_dim, hidden_dim=hp["projector"]["hidden_dim"], output_dim=hp["projector"]["output_dim"], norm_layer=torch.nn.BatchNorm1d)
                proj.load_state_dict({k[len("projector."):]: v for k, v in ck["state_dict"].items() if k.startswith("projector.")}); proj.to(dev).eval()
            models.append((os.path.basename(p), ep, w, enc, proj))
        del ck
    for spec in a.ref:
        name, enc = load_reference(spec); enc.to(dev).eval(); models.append((name, -1, "ref", enc, None))
    print(f"[k710-probe] {len(models)} encoders loaded from {len(a.ckpts)} checkpoints + {len(a.ref)} references", flush=True)
    loader = torch.utils.data.DataLoader(CenterClips(files, labels), batch_size=a.bs, shuffle=False, num_workers=a.workers, multiprocessing_context=mp.get_context("spawn"), pin_memory=True)
    feats = {(i, s): [] for i in range(len(models)) for s in ("h.cls", "h.gap", "z")}; ys = []; t0 = time.time(); n = 0
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        for b in loader:
            x = to_float_normalized(b["frame"].to(dev, non_blocking=True)).permute(0, 2, 1, 3, 4)   # (B, 3, T, H, W)
            ys.append(b["label"])
            for i, (_, _, _, enc, proj) in enumerate(models):
                tok = enc(x); h = tok[:, 0].float(); feats[(i, "h.cls")].append(h.cpu()); feats[(i, "h.gap")].append(tok[:, 1:].float().mean(1).cpu())
                if proj is not None: feats[(i, "z")].append(proj(h).float().cpu())
            n += len(x)
            if n % (a.bs * 25) == 0: print(f"[k710-probe] {n}/{len(files)} clips in {time.time() - t0:.0f}s", flush=True)
    y = torch.cat(ys).to(dev); tr, ev = is_train.to(dev), (~is_train).to(dev)
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True); new = not os.path.exists(a.out)
    with open(a.out, "a", newline="") as fh:
        wr = csv.writer(fh)
        if new: wr.writerow(["ckpt", "epoch", "weights", "space", "n_train", "n_eval", "knn_top1", "linear_top1", "linear_train_top1"])
        for i, (name, ep, w, _, proj) in enumerate(models):
            for s in ("h.cls", "h.gap", "z"):
                if s == "z" and proj is None: continue
                f = torch.cat(feats[(i, s)]).to(dev)
                knn = knn_top1(f[tr], y[tr], f[ev], y[ev], C=C); lin, lin_tr = linear_top1(f[tr], y[tr], f[ev], y[ev], C=C)
                print(f"[k710-probe] epoch {ep:3d} {w:7s} {s}: knn {knn:5.2f}  linear {lin:5.2f} (train {lin_tr:5.2f})", flush=True)
                wr.writerow([name, ep, w, s, int(tr.sum()), int(ev.sum()), f"{knn:.2f}", f"{lin:.2f}", f"{lin_tr:.2f}"])
    print("K710_PROBE_DONE", flush=True)


if __name__ == "__main__":
    main()
