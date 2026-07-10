"""Committed ground-truth validation for the k-hat suite (D-019; replaces the ad-hoc pre-commit
pass whose outcomes were only recorded in HANDOVER). Planted-defect clouds at toy geometry
(n=9469, d=256, frame m=64): r kurtotic directions get top variance (power lives in the
variance-weighted frame), ambient basis randomly rotated so nothing is axis-aligned in feature
coords. Cases mirror the claims we need to trust: same-sign spiky r=1/4/8/16, mixed-sign,
10-cluster (spans 9 dims), defects mixed pairwise 45-deg WITHIN the top-variance block (eig
attenuates by design, pursuit should not), and a pure-Gaussian negative control. Also lands the
khat_A before/after column for the inverted dimension-correction found in review 2026-07-10.

  python experiments/defect_rank_validate.py   -> results/diag/defect_rank_validate.csv
"""
import os
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
from sslgap.metrics.defect_rank import khat_eig, khat_pursuit, khat_spectrum, pursuit_null_band

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
N, D, M = 9469, 256, 64
rng = np.random.default_rng(7)


def laplace(n):
    return rng.laplace(0.0, 1.0 / np.sqrt(2.0), n)          # unit var, excess kurt +3


def bimodal(n):
    return np.sqrt(0.91) * rng.choice([-1.0, 1.0], n) + rng.normal(0, 0.3, n)  # kurt -1.66


def assemble(defect_cols, r):
    """Defect coords (std 2, top variance) + decaying Gaussian background, rotated in ambient."""
    bg = rng.standard_normal((N, D - r)) * np.linspace(1.0, 0.5, D - r)
    X = np.concatenate([2.0 * defect_cols, bg], axis=1)
    Q, _ = np.linalg.qr(rng.standard_normal((D, D)))
    return X @ Q


def clusters10():
    centers = rng.standard_normal((10, 9)) * 2.0
    z = rng.integers(0, 10, N)
    cols = centers[z] + rng.normal(0, 0.3, (N, 9))
    return (cols - cols.mean(0)) / cols.std(0)


def rot45_within(cols):
    for i in range(0, cols.shape[1] - 1, 2):
        a, b = cols[:, i].copy(), cols[:, i + 1].copy()
        cols[:, i], cols[:, i + 1] = (a + b) / np.sqrt(2), (a - b) / np.sqrt(2)
    return cols


CASES = [
    ("gaussian control", 0, lambda: assemble(np.empty((N, 0)), 0)),
    ("spiky r=1", 1, lambda: assemble(laplace((N, 1)), 1)),
    ("spiky r=4", 4, lambda: assemble(laplace((N, 4)), 4)),
    ("spiky r=8", 8, lambda: assemble(laplace((N, 8)), 8)),
    ("spiky r=16", 16, lambda: assemble(laplace((N, 16)), 16)),
    ("mixed-sign r=8 (4+4)", 8,
     lambda: assemble(np.concatenate([laplace((N, 4)),
                                      np.stack([bimodal(N) for _ in range(4)], 1)], 1), 8)),
    ("10-cluster (9-dim)", 9, lambda: assemble(clusters10(), 9)),
    ("spiky r=8 rot45-within", 8, lambda: assemble(rot45_within(laplace((N, 8))), 8)),
]


def invert_old(m1, m2, meff):
    A = (m2 / m1**2) * meff * (meff + 2) / ((meff + 4) * (meff + 6))
    return float(meff) if A <= 1 else min(float(32 / (3 * (A - 1))), float(meff))


def main():
    rows, bands = [], {}
    for name, r_true, build in CASES:
        X = build()
        e = khat_eig(X, m=M)
        meff = len(e["eig_kurts"])
        nd = min(20, meff)
        key = (N, meff, nd)
        if key not in bands:
            bands[key] = pursuit_null_band(N, meff, nd)
        p = khat_pursuit(X, m=M, n_dirs=nd, seed=3, band=bands[key])
        sp = khat_spectrum(X, m=M)
        rows.append({"case": name, "r_true": r_true, "khat_eig": e["khat_eig"],
                     "khat_pp": p["khat_pp"],
                     "khat_A": round(sp["khat_A"], 1) if sp["detect_A"] else float("nan"),
                     "khat_A_oldformula": round(invert_old(sp["m1"], sp["m2"], sp["m_frame"]), 1)
                     if sp["detect_A"] else float("nan"),
                     "detect_A": sp["detect_A"],
                     "eig_top|k|": round(float(np.abs(e["eig_kurts"]).max()), 2),
                     "pp_band": round(p["pp_null_band"], 2)})
        print(f"[validate] {rows[-1]}")
    df = pd.DataFrame(rows)
    out = os.path.join(ROOT, "results", "diag", "defect_rank_validate.csv")
    df.to_csv(out, index=False)
    print(df.to_string(index=False))
    print(f"[validate] wrote {out}")


if __name__ == "__main__":
    main()
