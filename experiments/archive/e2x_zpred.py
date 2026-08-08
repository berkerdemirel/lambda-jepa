"""Linear predictability of z from h (Berker 2026-07-20: "predictability of z from h
linearly. make it scale invariant across methods. (will show how much of a beast the
corresponding model's MLP is)"). OLS h -> z fit on the train store, scored on val:
r2_total = 1 - ||Z - HW||_F^2 / ||Z - Zbar||_F^2 (z centered with TRAIN means) — invariant
to any invertible linear map of h and to rotation/global scale of z, so cross-method
comparable. 1 = the head is affine in effect; low = the MLP's output is linearly
unreachable from h. Pair = the TRAINING-TIME head: student h.cls -> loss-terminal z
(dino: student bottleneck — the head lives on the student branch; probe declarations are
a separate convention). r2_meandim = per-dim R^2 averaged (equal-weight secondary column;
near-dead dims make it noisier).

DIMENSION-CONFOUND CONTROLS (Berker's same-day flag: "this linearity thing could be
confounded by the dimension difference"): the estimator is d_z-clean (384 regressors on 50k
per output dim, identical across methods), but the LOSS's demand scales with d_z — a Σ=I
floor over more dims can force the head deeper into its nonlinear repertoire. Two controls:
(a) the method's OWN dim bracket d16..d512 (same architecture family + loss, only d_z
varies) — the dimension-response of the metric under one head family; (b) a RANDOM-HEAD
null — seed-0 norm-free MLP (Linear-ReLU x2 + Linear, hidden 2048; BN omitted so the null
has no normalization-state ambiguity) applied to ONE fixed h (the d256 run's), d_z in
{16, 64, 256, 2048}: random functionals of a shared hidden have d-independent linear share
in expectation, so a flat null acquits dimension per se and attributes any bracket gradient
to training. Overwrites results/diag/e2x_zpred.csv. RAW."""
import csv
import os

import numpy as np

ROOT = "/nfs/scistore19/locatgrp/bdemirel/ssl_project"
FEAT = f"{ROOT}/features"
OUT = f"{ROOT}/results/diag/e2x_zpred.csv"
H = "student.h.cls"
PAIRS = [(f"in100.floorssl.s0.d{d}.ext", "student.z.proj.out")
         for d in (16, 32, 64, 128, 256, 512)] + \
        [("in100.lejepa.s0.e20f.ext", "student.z.proj.out"),
         ("in100.lejepa.s0.ext", "student.z.proj.out"),
         ("in100.vicreg.s0.ext", "student.z.proj.out"),
         ("in100.dino.s0.e20f_lam008.ext", "student.z.dino.bottleneck"),
         ("in100.dino.s0.ext", "student.z.dino.bottleneck")]
NULL_H_RUN = "in100.floorssl.s0.d256.ext"
NULL_DIMS = (16, 64, 256, 2048)


def load(run, sp, man):
    for flavor in ("v1L", "v1"):
        p = f"{FEAT}/{run}/{man}.{flavor}/{sp}.npy"
        if os.path.exists(p):
            return np.load(p).astype(np.float64)
    return None


rows = []
print(f"{'run':38s} {'z_space':26s} {'d_z':>4s} {'r2_total':>8s} {'r2_meandim':>10s}")
for run, zsp in PAIRS:
    Ht, Zt = load(run, H, "in100.train500"), load(run, zsp, "in100.train500")
    Hv, Zv = load(run, H, "in100.val"), load(run, zsp, "in100.val")
    if any(x is None for x in (Ht, Zt, Hv, Zv)):
        print(f"{run:38s} {zsp:26s}   (store absent — skipped)")
        continue
    hm, zm = Ht.mean(0), Zt.mean(0)
    W, *_ = np.linalg.lstsq(Ht - hm, Zt - zm, rcond=None)
    res = (Zv - zm) - (Hv - hm) @ W
    tot = Zv - zm
    r2_total = 1.0 - (res ** 2).sum() / (tot ** 2).sum()
    r2_dim = 1.0 - (res ** 2).sum(0) / np.maximum((tot ** 2).sum(0), 1e-12)
    rows.append({"run": run, "h_space": H, "z_space": zsp, "d_z": Zt.shape[1],
                 "r2_total": round(float(r2_total), 4),
                 "r2_meandim": round(float(r2_dim.mean()), 4),
                 "n_train": len(Ht), "n_val": len(Hv)})
    print(f"{run:38s} {zsp:26s} {Zt.shape[1]:4d} {r2_total:8.4f} {r2_dim.mean():10.4f}",
          flush=True)
# (b) random-head null on one fixed h — pure-dimension response with no training
import torch
torch.set_num_threads(8)
Ht, Hv = load(NULL_H_RUN, H, "in100.train500"), load(NULL_H_RUN, H, "in100.val")
tt, tv = torch.from_numpy(Ht).float(), torch.from_numpy(Hv).float()
for d in NULL_DIMS:
    torch.manual_seed(0)
    head = torch.nn.Sequential(
        torch.nn.Linear(384, 2048), torch.nn.ReLU(),
        torch.nn.Linear(2048, 2048), torch.nn.ReLU(), torch.nn.Linear(2048, d))
    with torch.no_grad():
        Zt = torch.cat([head(b) for b in tt.split(4096)]).double().numpy()
        Zv = torch.cat([head(b) for b in tv.split(4096)]).double().numpy()
    hm, zm = Ht.mean(0), Zt.mean(0)
    W, *_ = np.linalg.lstsq(Ht - hm, Zt - zm, rcond=None)
    res, tot = (Zv - zm) - (Hv - hm) @ W, Zv - zm
    r2 = 1.0 - (res ** 2).sum() / (tot ** 2).sum()
    r2d = (1.0 - (res ** 2).sum(0) / np.maximum((tot ** 2).sum(0), 1e-12)).mean()
    rows.append({"run": f"null.randhead.d{d}", "h_space": H, "z_space": "randMLP(h)",
                 "d_z": d, "r2_total": round(float(r2), 4),
                 "r2_meandim": round(float(r2d), 4), "n_train": len(Ht), "n_val": len(Hv)})
    print(f"{'null.randhead.d%d' % d:38s} {'randMLP(h)':26s} {d:4d} {r2:8.4f} {r2d:10.4f}",
          flush=True)
with open(OUT, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0]))
    w.writeheader()
    w.writerows(rows)
print("wrote", OUT)
