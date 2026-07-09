"""Probe-convergence diagnostic (2026-07-09, Berker's question on the M2 seed-0 probes).

Is the linear_raw_v1 deficit vs linear_house_v1 at GAP spaces (up to 4.4 pt) a property of the
features, or an optimization artifact of fitting a plain Linear on unnormalized features under
the fixed 30-ep AdamW budget? Arms vary ONLY the input transform and the epoch budget:

  raw    — protocol linear_raw_v1 (plain Linear)                @ 30 and 300 ep
  std    — (x-mu)/sigma with TRAIN stats, then Linear. Affine => capacity-neutral: any gap it
           closes is conditioning, not linear separability      @ 30 and 300 ep
  bn     — BatchNorm1d(affine=False) + Linear (MAE-paper probe-BN; F3/E11 arm)  @ 30 ep
  house  — protocol linear_house_v1 (LayerNorm + Linear; per-sample, geometry-changing)
                                                                @ 30 (reproduction) and 300 ep

Cells: the four big raw-vs-house gaps (simclr/byol/vicreg/mae at h.gap) + two contrasts where
raw >= house (dino teacher CLS, lejepa embed). Everything else per PROTOCOL: seed 0, AdamW
1e-3/1e-7, bs 256, best-val selection. Diagnostic only — not a protocol change (D-006v2 stands
unless a DECISIONS row says otherwise). Outputs: results/diag/probe_conv{,_curves}.csv.
"""
import csv
import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TRAIN_MAN, VAL_MAN = "in100.train500.v1", "in100.val.v1"
CELLS = [
    ("in100.simclr.s0.ext", "student.h.gap"),
    ("in100.byol.s0.ext", "student.h.gap"),
    ("in100.vicreg.s0.ext", "student.h.gap"),
    ("in100.mae.s0.ext", "student.h.gap"),
    ("in100.dino.s0.ext", "teacher.h.cls"),
    ("in100.lejepa.s0.ext", "student.z.embed"),
]
ARMS = [("raw", 30), ("raw", 300), ("std", 30), ("std", 300),
        ("bn", 30), ("house", 30), ("house", 300)]
LR, WD, BS, SEED = 1e-3, 1e-7, 256, 0


def load(run_id, man, space, device):
    d = os.path.join(ROOT, "features", run_id, man)
    X = torch.as_tensor(np.load(os.path.join(d, f"{space}.npy")), dtype=torch.float32,
                        device=device)
    y = torch.as_tensor(np.load(os.path.join(d, "labels.npy")), dtype=torch.long, device=device)
    return X, y


def build(arm, d, num_classes, mu=None, sigma=None):
    torch.manual_seed(SEED)
    if arm == "house":
        return nn.Sequential(nn.LayerNorm(d), nn.Linear(d, num_classes))
    if arm == "bn":
        return nn.Sequential(nn.BatchNorm1d(d, affine=False), nn.Linear(d, num_classes))
    return nn.Linear(d, num_classes)


def run_arm(arm, epochs, Xtr, ytr, Xva, yva, num_classes, device):
    if arm == "std":
        mu, sigma = Xtr.mean(0), Xtr.std(0)
        sigma = torch.where(sigma < 1e-6, torch.ones_like(sigma), sigma)
        Xtr, Xva = (Xtr - mu) / sigma, (Xva - mu) / sigma
    probe = build(arm, Xtr.shape[1], num_classes).to(device)
    opt = torch.optim.AdamW(probe.parameters(), lr=LR, weight_decay=WD)

    @torch.no_grad()
    def acc(X, y):
        return (probe(X).argmax(1) == y).float().mean().item()

    n, best, curve = Xtr.shape[0], None, []
    for ep in range(epochs):
        probe.train()
        perm = torch.randperm(n, device=device)
        for i in range(0, n, BS):
            idx = perm[i:i + BS]
            loss = F.cross_entropy(probe(Xtr[idx]), ytr[idx])
            opt.zero_grad()
            loss.backward()
            opt.step()
        probe.eval()
        va = acc(Xva, yva)
        curve.append(va)
        if best is None or va > best["val_acc"]:
            best = {"val_acc": va, "ep": ep, "train_acc": acc(Xtr, ytr)}
    return best, curve


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    out_dir = os.path.join(ROOT, "results", "diag")
    os.makedirs(out_dir, exist_ok=True)
    rows, curves = [], []
    for run_id, space in CELLS:
        Xtr, ytr = load(run_id, TRAIN_MAN, space, device)
        Xva, yva = load(run_id, VAL_MAN, space, device)
        C = int(ytr.max().item()) + 1
        for arm, epochs in ARMS:
            best, curve = run_arm(arm, epochs, Xtr, ytr, Xva, yva, C, device)
            rows.append({"run_id": run_id, "space": space, "arm": arm, "epochs": epochs,
                         "best_val": round(best["val_acc"], 4), "best_ep": best["ep"],
                         "train_acc_at_best": round(best["train_acc"], 4),
                         "final_val": round(curve[-1], 4)})
            curves += [{"run_id": run_id, "space": space, "arm": arm, "epochs": epochs,
                        "ep": i, "val_acc": round(v, 4)} for i, v in enumerate(curve)]
            print(f"[conv] {run_id} {space} {arm}@{epochs}: best={best['val_acc']:.4f} "
                  f"(ep{best['ep']}) train_at_best={best['train_acc']:.4f} "
                  f"final={curve[-1]:.4f}", flush=True)
    for name, data, fields in [("probe_conv.csv", rows, list(rows[0])),
                               ("probe_conv_curves.csv", curves, list(curves[0]))]:
        with open(os.path.join(out_dir, name), "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fields)
            w.writeheader()
            w.writerows(data)
    print(f"[conv] done -> {out_dir}/probe_conv.csv")


if __name__ == "__main__":
    main()
