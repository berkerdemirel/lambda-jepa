"""E18 (finite-ν declared-prior arm) design datum — fit ν for the t_ν CF target.

Re-fit of HANDOVER item (iv)'s rough ν ≈ 9.7 (that number came from e12c1's P5 Varimax kurtosis,
the E12 lane). The E18 arm rides the SHIPPED lane against e17c, so the fit is recomputed there,
with the exact P5 procedure (probe-subspace Varimax coordinate kurtosis, e12_varimax_p5.py) on
e17c's declared h (student.z.embed). Moment-match: a 1-d t_ν marginal has excess kurtosis
6/(ν−4) ⇒ ν = 4 + 6/κ.

Alongside, two context data the design discussion needs:
- RANDOM-SLICE kurtosis (256 unit directions, z-scored): what the sliced CF term actually
  compares against its target. If slices are CLT-Gaussianized (κ_slice ≈ 0) while coordinates
  are leptokurtic, a coordinate-fit t_ν target demands heavier slice tails than the data has —
  the term's initial shape gradient then points TOWARD heavy tails, not away.
- raw per-coordinate kurtosis (512 dims, no Varimax), for basis sensitivity.

e12c1 is re-run as a reproduction check of the recorded 1.063. Numbers land raw
(results/diag/e18_nu_fit.csv); the ν declaration is made in discussion / on the card.
"""
import csv
import os

import numpy as np

ROOT = "/nfs/scistore19/locatgrp/bdemirel/ssl_project"
RUNS = {"e17c": ("in100.lejepa.s0.e17c.ext", "in100.train500.v1L"),
        "e12c1": ("in100.lejepa.s0.e12c1.ext", "in100.train500.v1")}


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


def kurt(P):
    Pz = (P - P.mean(0)) / (P.std(0) + 1e-12)
    return (Pz ** 4).mean(0) - 3.0


def nu_of(k):
    return round(4 + 6 / k, 2) if k > 0 else float("inf")


def main():
    rng = np.random.default_rng(0)
    rows = []
    for tag, (rid, split) in RUNS.items():
        d = os.path.join(ROOT, "features", rid, split)
        X = np.load(os.path.join(d, "student.z.embed.npy")).astype(np.float64)
        y = np.load(os.path.join(d, "labels.npy"))
        Xc = X - X.mean(0)
        Y = np.eye(100)[y] - 1.0 / 100
        W = np.linalg.solve(Xc.T @ Xc + 1e-2 * len(X) * np.eye(X.shape[1]), Xc.T @ Y)
        Q, _ = np.linalg.qr(W)
        kv = kurt(Xc @ (Q @ varimax(Q)))                       # P5 statistic
        A = rng.standard_normal((X.shape[1], 256))
        A /= np.linalg.norm(A, axis=0)
        ks = kurt(Xc @ A)                                      # slice statistic
        kc = kurt(Xc)                                          # raw coordinates
        rows.append({"arm": tag,
                     "kurt_varimax_mean": round(float(kv.mean()), 3),
                     "kurt_varimax_median": round(float(np.median(kv)), 3),
                     "kurt_slice_mean": round(float(ks.mean()), 3),
                     "kurt_slice_median": round(float(np.median(ks)), 3),
                     "kurt_coord_mean": round(float(kc.mean()), 3),
                     "kurt_coord_median": round(float(np.median(kc)), 3),
                     "nu_varimax_mean": nu_of(float(kv.mean())),
                     "nu_varimax_median": nu_of(float(np.median(kv))),
                     "nu_slice_mean": nu_of(float(ks.mean()))})
        print(rows[-1], flush=True)
    out = os.path.join(ROOT, "results", "diag", "e18_nu_fit.csv")
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print("wrote", out)


if __name__ == "__main__":
    main()
