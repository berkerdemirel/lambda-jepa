"""E17 discussion diagnostic (2026-07-16): decompose probe/geometry effects into mean-cone vs
shape. Per run @ declared h: B/T class-variance share (e12_class_align convention), cosine-kNN
k200+k20 raw vs TRAIN-MEAN-CENTERED (eval-time mean removal; knn_v1 is cosine => mean-sensitive),
centered orbit triple (pos_c/rand_c). Plus B/T + cone decomposition @ z.final. Includes the E12-lane
lejepa pair (C1/f2) for T7 continuity. Writes results/diag/e17_centered.csv. RAW numbers."""
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, "/nfs/scistore19/locatgrp/bdemirel/ssl_project")
torch.set_num_threads(2)
from sslgap.probes.knn import knn_topk_acc

ROOT = "/nfs/scistore19/locatgrp/bdemirel/ssl_project"
FEAT = f"{ROOT}/features"
CFG = {
    "lejepa": ("student.z.embed", "student.z.proj.out",
               [("ctrl", "in100.lejepa.s0.e17c.ext"), ("sigreg", "in100.lejepa.s0.hpull_sigreg.ext"),
                ("sigreg_inv", "in100.lejepa.s0.hpull_sigreg_inv.ext")]),
    "simclr": ("student.h.cls", "student.z.proj.out",
               [("ctrl", "in100.simclr.s0.e17c.ext"), ("uniform", "in100.simclr.s0.hpull_uniform.ext"),
                ("uniform_align", "in100.simclr.s0.hpull_uniform_align.ext")]),
    "byol": ("student.h.cls", "student.z.pred.out",
             [("ctrl", "in100.byol.s0.e17c.ext"), ("align", "in100.byol.s0.hpull_align.ext")]),
    "dino": ("teacher.h.cls", "teacher.z.dino.bottleneck",
             [("ctrl", "in100.dino.s0.e17c.ext"), ("protoce", "in100.dino.s0.hpull_protoce.ext")]),
    "vicreg": ("student.h.cls", "student.z.proj.out", [("ctrl", "in100.vicreg.s0.e17c.ext")]),
    "lejepa_e12lane": ("student.z.embed", "student.z.proj.out",
                       [("C1", "in100.lejepa.s0.e12c1.ext"), ("f2", "in100.lejepa.s0.e12f2.ext")]),
}


def bt(X, y):
    mu = X.mean(0)
    tot = ((X - mu) ** 2).sum(1).mean()
    b = sum((y == c).mean() * ((X[y == c].mean(0) - mu) ** 2).sum() for c in np.unique(y))
    return float(b / tot)


def cone(X):
    U = X / np.linalg.norm(X, axis=1, keepdims=True)
    n = len(U)
    return float((np.linalg.norm(U.sum(0)) ** 2 - n) / (n * (n - 1)))


def knn(tr, ytr, va, yva):
    return knn_topk_acc(torch.from_numpy(tr).float(), torch.from_numpy(ytr).long(),
                        torch.from_numpy(va).float(), torch.from_numpy(yva).long(),
                        num_classes=100, knn_k=200, knn_t=0.1)


def orbit_both(run, sp):
    """(pos_raw, rand_raw, pos_c, rand_c) — the o8 triple raw and after pooled-mean centering."""
    d = f"{FEAT}/{run}/in100.pairs100.v1@audit_v1.o8"
    V = json.load(open(f"{d}/meta.json"))["v"]
    raw = [np.load(f"{d}/{sp}.view{k}.npy").astype(np.float64) for k in range(V)]
    mu = np.concatenate(raw).mean(0)
    n = len(raw[0])
    p = np.random.default_rng(0).permutation(n)
    q = np.roll(p, 1)
    out = []
    for vs in (raw, [x - mu for x in raw]):
        vs = [x / (np.linalg.norm(x, axis=1, keepdims=True) + 1e-12) for x in vs]
        pc, rc = [], []
        for u in range(V):
            for w in range(u + 1, V):
                pc.append((vs[u] * vs[w]).sum(1).mean())
                rc.append((vs[u][p] * vs[w][q]).sum(1).mean())
        out += [float(np.mean(pc)), float(np.mean(rc))]
    return out


def knn_k(tr, ytr, va, yva, k):
    return knn_topk_acc(torch.from_numpy(tr).float(), torch.from_numpy(ytr).long(),
                        torch.from_numpy(va).float(), torch.from_numpy(yva).long(),
                        num_classes=100, knn_k=k, knn_t=0.1)


rows = []
print(f"{'run':24s} {'B/T@h':>6s} {'B/T@z':>6s} {'cone@z':>7s} {'knn200':>7s} {'k200ctr':>8s} "
      f"{'knn20':>6s} {'k20ctr':>7s} {'pos_c':>6s} {'rand_c':>7s}")
for m, (h, z, runs) in CFG.items():
    for tag, r in runs:
        d = f"{FEAT}/{r}/in100.train500.v1L"
        dv = f"{FEAT}/{r}/in100.val.v1L"
        Xtr = np.load(f"{d}/{h}.npy").astype(np.float64)
        ytr = np.load(f"{d}/labels.npy")
        Xva = np.load(f"{dv}/{h}.npy").astype(np.float64)
        yva = np.load(f"{dv}/labels.npy")
        Z = np.load(f"{d}/{z}.npy").astype(np.float64)
        mu = Xtr.mean(0)
        rows.append(dict(method=m, tag=tag, run=r, space=h,
                         bt_h=round(bt(Xtr, ytr), 4), bt_z=round(bt(Z, ytr), 4),
                         cone_z=round(cone(Z), 4),
                         knn200=round(100 * knn_k(Xtr, ytr, Xva, yva, 200), 2),
                         knn200_ctr=round(100 * knn_k(Xtr - mu, ytr, Xva - mu, yva, 200), 2),
                         knn20=round(100 * knn_k(Xtr, ytr, Xva, yva, 20), 2),
                         knn20_ctr=round(100 * knn_k(Xtr - mu, ytr, Xva - mu, yva, 20), 2)))
        pr, rr, pc, rc = orbit_both(r, h)
        rows[-1].update(pos_raw=round(pr, 4), rand_raw=round(rr, 4),
                        pos_c=round(pc, 4), rand_c=round(rc, 4))
        x = rows[-1]
        print(f"{m + '/' + tag:24s} {x['bt_h']:6.3f} {x['bt_z']:6.3f} {x['cone_z']:7.3f} "
              f"{x['knn200']:7.1f} {x['knn200_ctr']:8.1f} {x['knn20']:6.1f} {x['knn20_ctr']:7.1f} "
              f"{x['pos_c']:6.3f} {x['rand_c']:7.3f}", flush=True)

import csv

os.makedirs(f"{ROOT}/results/diag", exist_ok=True)
with open(f"{ROOT}/results/diag/e17_centered.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
    w.writeheader()
    w.writerows(rows)
print(f"wrote {ROOT}/results/diag/e17_centered.csv")
