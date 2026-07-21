"""Label-conditioned cosine structure at declared h (Berker 2026-07-20: "E[cos(hi,hj)|yi=yj]
vs E[cos(hi,hj)|yi!=yj], same class expected cos similarity vs diff class expected cos
similarity"). Exact over ALL pairs via the class-sum identity on L2-normalized features:
sum_same = Σ_c (||S_c||² − n_c), pairs_same = Σ_c n_c(n_c−1); sum_diff = ||S||² − Σ_c ||S_c||²
(S = Σu, S_c = per-class Σu) — no sampling, O(N·d). Train split (500/class). Scale-invariant
(cosines). Space = each run's DECLARED h (probe convention; teacher for dino) — this is a
representation read; the head read lives in e2x_zpred. Overwrites
results/diag/e2x_classcos.csv. RAW."""
import csv
import os

import numpy as np

ROOT = "/nfs/scistore19/locatgrp/bdemirel/ssl_project"
FEAT = f"{ROOT}/features"
OUT = f"{ROOT}/results/diag/e2x_classcos.csv"
RUNS = [("in100.floorssl.s0.d64.ext", "student.h.cls"),
        ("in100.floorssl.s0.d256.ext", "student.h.cls"),
        ("in100.floorssl.s0.d256vm2.ext", "student.h.cls"),
        ("in100.floorssl.s0.d256e200.ext", "student.h.cls"),
        ("in100.lejepa.s0.e20f.ext", "student.h.cls"),
        ("in100.lejepa.s0.ext", "student.h.cls"),
        ("in100.dino.s0.e20f_lam008.ext", "teacher.h.cls"),
        ("in100.dino.s0.ext", "teacher.h.cls")]


def load(run, sp):
    for flavor in ("v1L", "v1"):
        d = f"{FEAT}/{run}/in100.train500.{flavor}"
        if os.path.exists(f"{d}/{sp}.npy"):
            return (np.load(f"{d}/{sp}.npy").astype(np.float64),
                    np.load(f"{d}/labels.npy"))
    return None


rows = []
print(f"{'run':38s} {'space':16s} {'cos_same':>8s} {'cos_diff':>9s} {'gap':>7s}")
for run, sp in RUNS:
    got = load(run, sp)
    if got is None:
        print(f"{run:38s} {sp:16s}   (store absent — skipped)")
        continue
    X, y = got
    U = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-12)
    S = U.sum(0)
    sum_same, pairs_same, sumsq_c = 0.0, 0.0, 0.0
    for c in np.unique(y):
        Sc = U[y == c].sum(0)
        nc = int((y == c).sum())
        sum_same += Sc @ Sc - nc
        pairs_same += nc * (nc - 1)
        sumsq_c += Sc @ Sc
    n = len(U)
    cos_same = sum_same / pairs_same
    cos_diff = (S @ S - sumsq_c) / (n * n - sum((int((y == c).sum())) ** 2
                                                for c in np.unique(y)))
    rows.append({"run": run, "space": sp, "cos_same": round(float(cos_same), 4),
                 "cos_diff": round(float(cos_diff), 4),
                 "gap": round(float(cos_same - cos_diff), 4), "n": n})
    print(f"{run:38s} {sp:16s} {cos_same:8.4f} {cos_diff:9.4f} {cos_same-cos_diff:7.4f}",
          flush=True)
with open(OUT, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0]))
    w.writeheader()
    w.writerows(rows)
print("wrote", OUT)
