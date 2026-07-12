"""E12: how well is the floor actually satisfied at h? (Berker 2026-07-12: probes-vs-dose is
not enough — show the corresponding LOSS at h.) Computes, on stored train500 features at
h = z.embed for every arm + references, the DECLARED estimator (MomentFloor: sliced d'=128,
eps 1e-4) averaged over 32 fresh draws at n=4096/draw, PLUS the full-dim diagonal read
(mean^2 and per-dim var-KL) and the full-cov KL with shrinkage — so constraint satisfaction is
comparable across arms including those that never trained the term (C1, lane, null). Raw numbers.
"""
import csv
import os

import numpy as np

ROOT = "/nfs/scistore19/locatgrp/bdemirel/ssl_project"
RUNS = ["e12c1", "e12f2", "e12f1", "e12a3", "e12f6", "e12f5", "e12f3", "e12f4", "e12a2", "e12a1"]
REFS = {"lane": "in100.lejepa.s0.ext", "null": "in100.lejepa.s0.null.ext"}


def sliced_kl(X, draws=32, d=128, eps=1e-4, n=4096, seed=0):
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(draws):
        idx = rng.choice(len(X), size=n, replace=False)
        Q, _ = np.linalg.qr(rng.standard_normal((X.shape[1], d)))
        p = X[idx] @ Q
        mu = p.mean(0)
        pc = p - mu
        cov = pc.T @ pc / (n - 1) + eps * np.eye(d)
        sign, logdet = np.linalg.slogdet(cov)
        vals.append(0.5 * (np.trace(cov) + mu @ mu - d - logdet) / d)
    return float(np.mean(vals)), float(np.std(vals))


def diag_read(X):
    mu = X.mean(0)
    var = X.var(0).clip(1e-8)
    return float(0.5 * (np.mean(mu ** 2) + np.mean(var - 1 - np.log(var))))


def main():
    rows = []
    for label, rid in [(a, f"in100.lejepa.s0.{a}.ext") for a in RUNS] + list(REFS.items()):
        X = np.load(f"{ROOT}/features/{rid}/in100.train500.v1/student.z.embed.npy").astype(np.float64)
        m, s = sliced_kl(X)
        rows.append({"arm": label, "moment_kl_sliced(mean±sd,32draws)": f"{m:.4f}±{s:.4f}",
                     "moment_kl_sliced": round(m, 4), "diag_kl": round(diag_read(X), 4),
                     "feat_mean_norm2/d": round(float((X.mean(0) ** 2).mean()), 4),
                     "feat_var_mean": round(float(X.var(0).mean()), 4)})
        print(rows[-1], flush=True)
    out = f"{ROOT}/results/diag/e12_floor_values.csv"
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print("wrote", out)


if __name__ == "__main__":
    main()
