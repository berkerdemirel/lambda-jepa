"""ImageNet-1k linear probe and kNN in the Lightly benchmark recipe (BatchNorm + linear head, 90 epochs; kNN k = 200)."""
import math
import os
import sys

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.amp import autocast
from torch.utils.data import DataLoader
from torchvision import datasets as tvd
from torchvision import transforms as T

from sslgap.ckpt import adapters
from sslgap.models.backbones import trunk_features
from sslgap.probes import knn_topk_acc

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.expanduser("~/data/imagenet")
NORM = T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])

def knn_rider(run_id, device, out_csv):
    run_dir = os.path.join(ROOT, "features", run_id)
    if not os.path.isdir(run_dir):
        return
    mans = [m for m in sorted(os.listdir(run_dir)) if "@" not in m]
    man_tr = next((m for m in mans if "train" in m), None)
    man_va = next((m for m in mans if "val" in m), None)
    if not (man_tr and man_va):
        return
    load = lambda man, name: np.load(os.path.join(run_dir, man, name))
    Xtr, ytr = load(man_tr, "student.h.cls.npy"), load(man_tr, "labels.npy")
    Xva, yva = load(man_va, "student.h.cls.npy"), load(man_va, "labels.npy")
    rows = [{"run_id": run_id, "probe": f"knn_k200_t{t}", "space": "student.h.cls",
             "val_acc": knn_topk_acc(Xtr, ytr, Xva, yva, int(ytr.max()) + 1,
                                     knn_k=200, knn_t=t, device=device)}
            for t in (0.1, 0.07)]
    for r in rows:
        print(f"[bench] {run_id} {r['probe']}={r['val_acc']:.4f}", flush=True)
    pd.DataFrame(rows).to_csv(out_csv, index=False)

def main():
    if sys.argv[1] == "--knn-only":
        out_csv = os.path.join(ROOT, "results/probes", f"{sys.argv[2]}.bench.csv")
        knn_rider(sys.argv[2], "cuda", out_csv.replace(".bench.csv", ".bench_knn.csv"))
        return
    ckpt, run_id = sys.argv[1], sys.argv[2]
    epochs = int(sys.argv[3]) if len(sys.argv) > 3 else 90
    adapter = sys.argv[4] if len(sys.argv) > 4 else "native"
    bs, device = 1024, "cuda"
    torch.backends.cudnn.benchmark = True
    workers = int(os.environ.get("SLURM_CPUS_PER_TASK", 12))

    loaded = adapters.load(adapter, os.path.expanduser(ckpt), run_id).eval_(device)
    assert loaded.frame["dataset"] == "imagenet1k", loaded.frame["dataset"]
    trunk = loaded.branches[loaded.probed_branch].trunk
    dim = trunk.num_features

    train_ds = tvd.ImageFolder(os.path.join(DATA, "train"), T.Compose(
        [T.RandomResizedCrop(224), T.RandomHorizontalFlip(), T.ToTensor(), NORM]))
    val_ds = tvd.ImageFolder(os.path.join(DATA, "val"), T.Compose(
        [T.Resize(256), T.CenterCrop(224), T.ToTensor(), NORM]))
    train_dl = DataLoader(train_ds, batch_size=bs, shuffle=True, num_workers=workers,
                          pin_memory=True, persistent_workers=True, drop_last=True,
                          prefetch_factor=4)
    val_dl = DataLoader(val_ds, batch_size=bs, shuffle=False, num_workers=workers,
                        pin_memory=True, persistent_workers=True)

    head = nn.Sequential(nn.BatchNorm1d(dim, affine=False, eps=1e-6),
                         nn.Linear(dim, len(train_ds.classes))).to(device)
    peak_lr = 0.1 * bs / 256
    opt = torch.optim.SGD(head.parameters(), lr=peak_lr, momentum=0.9, weight_decay=0.0)
    total = epochs * len(train_dl)
    warm = min(10, epochs) * len(train_dl)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: (
        0.01 + 0.99 * s / warm if s < warm
        else 0.001 + 0.999 * 0.5 * (1 + math.cos(math.pi * (s - warm) / max(1, total - warm)))))

    def cls_feats(x):
        with torch.no_grad(), autocast("cuda", dtype=torch.bfloat16):
            return trunk_features(trunk, x)["cls"].float()

    out_csv = os.path.join(ROOT, "results/probes", f"{run_id}.bench.csv")
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    state_pt = os.path.join(ROOT, "outputs", f"{run_id}.bench_head.pt")
    rows, best, start_ep = [], {"val_top1": 0.0, "val_top5": 0.0, "epoch": -1}, 0
    if os.path.exists(state_pt):
        st = torch.load(state_pt, map_location=device, weights_only=False)
        head.load_state_dict(st["head"])
        opt.load_state_dict(st["opt"])
        sched.load_state_dict(st["sched"])
        rows, best, start_ep = st["rows"], st["best"], st["epoch"]
        print(f"[bench] resumed {run_id} at epoch {start_ep}", flush=True)
    for ep in range(start_ep, epochs):
        head.train()
        tl, nb = 0.0, 0
        for x, y in train_dl:
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            loss = F.cross_entropy(head(cls_feats(x)), y)
            opt.zero_grad()
            loss.backward()
            opt.step()
            sched.step()
            tl, nb = tl + loss.item(), nb + 1
        head.eval()
        c1 = c5 = n = 0
        with torch.no_grad():
            for x, y in val_dl:
                x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
                logits = head(cls_feats(x))
                top5 = logits.topk(5, dim=1).indices
                c1 += (top5[:, 0] == y).sum().item()
                c5 += (top5 == y.unsqueeze(1)).any(1).sum().item()
                n += y.numel()
        row = {"run_id": run_id, "protocol": "lightly_mae_linear", "epoch": ep + 1,
               "lr": sched.get_last_lr()[0], "train_loss": tl / nb,
               "val_top1": c1 / n, "val_top5": c5 / n}
        if row["val_top1"] > best["val_top1"]:
            best = {"val_top1": row["val_top1"], "val_top5": row["val_top5"], "epoch": ep + 1}
        rows.append(row)
        pd.DataFrame(rows).to_csv(out_csv, index=False)
        torch.save({"head": head.state_dict(), "opt": opt.state_dict(),
                    "sched": sched.state_dict(), "rows": rows, "best": best, "epoch": ep + 1},
                   state_pt)
        print(f"[bench] {run_id} ep{ep + 1}/{epochs} lr={row['lr']:.4f} "
              f"train_loss={row['train_loss']:.3f} val_top1={row['val_top1']:.4f} "
              f"val_top5={row['val_top5']:.4f} best={best['val_top1']:.4f}", flush=True)
    print(f"[bench] done {run_id}: bench_linear_top1={best['val_top1']:.4f} "
          f"top5={best['val_top5']:.4f} (ep{best['epoch']}) -> {out_csv}", flush=True)
    knn_rider(run_id, device, out_csv.replace(".bench.csv", ".bench_knn.csv"))

if __name__ == "__main__":
    main()
