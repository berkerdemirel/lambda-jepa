"""E12 P5 — the framework §6.2 fork-gate symptom read (D-019/D-026; ν=∞ pilot double-duty).
On stored h (z.embed, train500) for A3 and C1 (+f2 for context): ridge-fit the class-probe
weight matrix W (100 one-hot tasks — the declared-F proxy at IN-100; adaptation noted), take an
orthonormal basis Q of the probe subspace, Varimax-rotate (R ∈ O(100)), report (i) sparsity gain
of QR over random-rotation nulls and (ii) coordinatewise excess kurtosis of h @ QR. Numbers land
raw; the fork-gate reading (symptoms present/absent) is discussion material.
"""
import csv
import os

import numpy as np

ROOT = "/nfs/scistore19/locatgrp/bdemirel/ssl_project"
RUNS = {"e12a3": "in100.lejepa.s0.e12a3.ext", "e12c1": "in100.lejepa.s0.e12c1.ext",
        "e12f2": "in100.lejepa.s0.e12f2.ext"}


def varimax(L, iters=100, tol=1e-7):
    p, k = L.shape
    R = np.eye(k)
    var_old = 0.0
    for _ in range(iters):
        LR = L @ R
        U, s, Vt = np.linalg.svd(L.T @ (LR ** 3 - LR @ np.diag((LR ** 2).sum(0)) / p))
        R = U @ Vt
        var_new = s.sum()
        if var_new - var_old < tol:
            break
        var_old = var_new
    return R


def vcrit(M):
    # varimax criterion: sum of per-column variance of squared loadings
    return float(((M ** 2) - (M ** 2).mean(0)).var(0).sum())


def main():
    rng = np.random.default_rng(0)
    rows = []
    for tag, rid in RUNS.items():
        d = os.path.join(ROOT, "features", rid, "in100.train500.v1")
        X = np.load(os.path.join(d, "student.z.embed.npy")).astype(np.float64)
        y = np.load(os.path.join(d, "labels.npy"))
        Xc = X - X.mean(0)
        Y = np.eye(100)[y] - 1.0 / 100
        W = np.linalg.solve(Xc.T @ Xc + 1e-2 * len(X) * np.eye(X.shape[1]), Xc.T @ Y)  # [512,100]
        Q, _ = np.linalg.qr(W)
        R = varimax(Q)
        crit_vm = vcrit(Q @ R)
        nulls = []
        for _ in range(20):
            A = rng.standard_normal((Q.shape[1], Q.shape[1]))
            Rn, _ = np.linalg.qr(A)
            nulls.append(vcrit(Q @ Rn))
        nulls = np.array(nulls)
        P = Xc @ (Q @ R)
        Pz = (P - P.mean(0)) / (P.std(0) + 1e-12)
        kurt = (Pz ** 4).mean(0) - 3.0
        rows.append({"arm": tag, "varimax_crit": round(crit_vm, 5),
                     "null_crit_mean": round(float(nulls.mean()), 5),
                     "null_crit_max": round(float(nulls.max()), 5),
                     "sparsity_gain_x": round(crit_vm / nulls.mean(), 2),
                     "kurt_mean": round(float(kurt.mean()), 3),
                     "kurt_median": round(float(np.median(kurt)), 3),
                     "kurt_frac_pos": round(float((kurt > 0).mean()), 3),
                     "kurt_p90": round(float(np.quantile(kurt, 0.9)), 3)})
        print(rows[-1], flush=True)
    out = os.path.join(ROOT, "results", "diag", "e12_varimax_p5.csv")
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print("wrote", out)


if __name__ == "__main__":
    main()
