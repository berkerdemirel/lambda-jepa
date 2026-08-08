"""Eigenvalue-spectrum figure for the quad roster (Berker 2026-07-21: "a figure for each
method in quad, where we see the distribution of cov eigenvalues … i wanna identify the sort
of discrepancy between rankme and effective rank"). Per run × per space, three normalized
spectra on one log-log panel: CENTERED covariance eigenvalue shares (lambda_i/sum — what
effective_rank exp-entropies; solid) and UNCENTERED singular-value shares (sigma_i/sum — what
rankme exp-entropies; dashed), ranks on x. The two effective counts are marked as ticks on
the rank axis. The curve gap decomposes the discrepancy by eye: a dashed head bump at rank 1
= the mean/cone component (uncentered only); the solid curve's steeper tail = the sigma->
sigma^2 squaring. Tail shape reads regularization: a smooth power-law decay = content-
flavored anisotropy, a cliff = dead-direction/collapse-flavored concentration.
Layout 2 rows (declared h / loss-terminal z) x 8 runs. Numbers recomputed from the train500
stores and cross-checkable against the battery CSVs. RAW, no takeaway."""
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, "/nfs/scistore19/locatgrp/bdemirel/ssl_project")
from sslgap.extract.store import FeatureStore

ROOT = "/nfs/scistore19/locatgrp/bdemirel/ssl_project"
MAN = "in100.train500.v1"
S = FeatureStore(f"{ROOT}/features")
RUNS = [("d64", "in100.floorssl.s0.d64.ext", "#3d65d0", "student.h.cls", "student.z.proj.out"),
        ("d256", "in100.floorssl.s0.d256.ext", "#1b2c60", "student.h.cls", "student.z.proj.out"),
        ("d256vm2", "in100.floorssl.s0.d256vm2.ext", "#6a3fb5", "student.h.cls", "student.z.proj.out"),
        ("d256e200", "in100.floorssl.s0.d256e200.ext", "#0f7b8a", "student.h.cls", "student.z.proj.out"),
        ("e20f", "in100.lejepa.s0.e20f.ext", "#2e8b57", "student.h.cls", "student.z.proj.out"),
        ("ctrl", "in100.lejepa.s0.ext", "#8c8c8c", "student.h.cls", "student.z.proj.out"),
        ("dino λ.008", "in100.dino.s0.e20f_lam008.ext", "#d4820a", "teacher.h.cls", "student.z.dino.bottleneck"),
        ("dino ctrl", "in100.dino.s0.ext", "#b39b77", "teacher.h.cls", "student.z.dino.bottleneck")]


def spectra(X):
    """(centered-eig shares, uncentered-sv shares, effrank, rankme) from an N×d store array."""
    X = np.asarray(X, dtype=np.float64)
    n = X.shape[0]
    Xc = X - X.mean(0)
    lam = np.linalg.eigvalsh(Xc.T @ Xc / (n - 1))[::-1]
    lam = np.clip(lam, 0, None)
    sig = np.sqrt(np.clip(np.linalg.eigvalsh(X.T @ X)[::-1], 0, None))   # sv of X (uncentered)
    sig_c = np.sqrt(lam)                     # centered SVs (= sqrt of cov eigenvalues)
    p_l = lam / lam.sum()
    p_s = sig / sig.sum()
    p_c = sig_c / sig_c.sum()

    def expent(p):
        p = p[p > 1e-300]
        return float(np.exp(-(p * np.log(p)).sum()))
    return p_l, p_s, p_c, expent(p_l), expent(p_s), expent(p_c)


fig, axes = plt.subplots(2, 8, figsize=(21, 6.2), facecolor="white")
for col, (lab, run, c, hsp, zsp) in enumerate(RUNS):
    for row, sp in ((0, hsp), (1, zsp)):
        ax = axes[row, col]
        try:
            X = S.get(run, MAN, sp)
        except Exception:
            ax.set_axis_off()
            continue
        p_l, p_s, p_c, er, rm, rmc = spectra(X)
        r = np.arange(1, len(p_l) + 1)
        ax.plot(r, p_l, "-", color=c, lw=1.4, label="centered eig (effrank)")
        ax.plot(r, p_c, ":", color=c, lw=1.2, label="centered SV (rankme_c)")
        ax.plot(np.arange(1, len(p_s) + 1), p_s, "--", color=c, lw=1.1, alpha=0.6,
                label="uncentered SV (rankme)")
        ymin = max(min(p_l[p_l > 0].min(), p_s[p_s > 0].min()), 1e-12)
        ax.plot([er], [ymin * 3], marker="|", ms=9, color=c, mew=1.6)
        ax.plot([rmc], [ymin * 3], marker="|", ms=9, color=c, mew=1.4, alpha=0.75)
        ax.plot([rm], [ymin * 3], marker="|", ms=9, color=c, mew=1.2, alpha=0.5)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_title(f"{lab} — {'h' if row == 0 else 'z'} (d={len(p_l)})\n"
                     f"effrank {er:.1f} · rankme_c {rmc:.1f} · rankme {rm:.1f}", fontsize=7)
        ax.tick_params(length=0, labelsize=6)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        if col == 0:
            ax.set_ylabel("spectral share (log)", fontsize=7)
        if row == 1:
            ax.set_xlabel("rank (log)", fontsize=7)
axes[0, 0].legend(fontsize=5.6, frameon=False, loc="lower left")
fig.suptitle("E21 spectra: centered covariance eigenvalue shares (solid = what effrank integrates) vs uncentered "
             "singular-value shares (dashed = what rankme integrates) — ticks on the rank axis = the two effective counts (RAW)",
             fontsize=9.5, y=0.995)
fig.tight_layout(rect=(0, 0.035, 1, 0.96))
fig.text(0.01, 0.005, "train500 clean features (no augs) · top = declared h, bottom = each run's loss-terminal z · "
         "dashed-above-solid at rank 1 = mean/cone component (uncentered only) · solid tail steeper than dashed = the σ→σ² "
         "squaring; rankme−rankme_c isolates centering, rankme_c−effrank isolates squaring · tail cliff = dead-direction concentration, smooth power law = "
         "content-flavored anisotropy", fontsize=6, color="#555555")
out = f"{ROOT}/results/figures/e21/e21_eig_spectra.png"
os.makedirs(os.path.dirname(out), exist_ok=True)
fig.savefig(out, dpi=160)
print("wrote", out)
