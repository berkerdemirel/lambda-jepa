"""Forensics for the pursuit-saturation observation (D-019 k-hat; Berker's evidence request
2026-07-10): re-derive the deflated-pursuit directions for every toy cell EXACTLY as the first
pass (same seed/flow -> dir-1 |kurt| must reproduce results/diag/defect_rank_toy.csv
top|k|_pp), then quantify how much of each direction's excess kurtosis a few extreme images
carry: trimmed kurtosis (drop top-q |s|), tail counts, fourth-moment share of the top-10, and
the drivers' identity (image index, class). The same stats for each cell's top-|kurt| PCA axes
land beside them, so "outlier-driven vs distributional" is a read-off, not a claim. Raw numbers.

  python experiments/defect_rank_outliers.py
    -> results/diag/defect_rank_outliers.csv          per-direction forensics (pp + eig)
    -> results/diag/defect_rank_outlier_drivers.csv   recurring driver images across cells
    -> results/diag/defect_rank_outlier_dirs.npz      directions (frame + original coords)
    -> results/figures/diag/defect_rank_outliers.png  showcase histograms, drivers marked
"""
import os
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from defect_rank_toy import CELLS, FEAT, MAN
from sslgap.metrics.defect_rank import _kurt_pursuit_dir

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SHOWCASE = ["lejepa embed", "lejepa h.cls", "A embed (free)", "Dlr embed (constrained)",
            "simclr z.proj", "dino z.bottleneck"]
TRIMS = (5, 10, 20)


def frame_with_basis(X, m=64, lam_ratio_floor=1e-4):
    """top_pca_frame's exact math, additionally returning W with P = (X - mean) @ W."""
    X = np.asarray(X, dtype=np.float64)
    X = X - X.mean(0)
    C = np.cov(X, rowvar=False)
    w, V = np.linalg.eigh(C)
    idx = np.argsort(w)[::-1]
    idx = idx[: min(m, int((w[idx] > lam_ratio_floor * w[idx[0]]).sum()))]
    P = X @ V[:, idx]
    s = P.std(0)
    return P / s, V[:, idx] / s


def pursuit_dirs(P, n_dirs, seed):
    """_pursuit_profile's exact flow (same rng consumption), keeping signed kurt + frame dirs."""
    rng = np.random.default_rng(seed)
    comp = np.eye(P.shape[1])
    ks, dirs = [], []
    for _ in range(n_dirs):
        Q = P @ comp
        u_local, k = _kurt_pursuit_dir(Q, rng)
        ks.append(k)
        dirs.append(comp @ u_local)
        M = np.eye(comp.shape[1]) - np.outer(u_local, u_local)
        q, r = np.linalg.qr(M)
        comp = comp @ q[:, np.abs(np.diag(r)) > 1e-9][:, : comp.shape[1] - 1]
        if comp.shape[1] == 0:
            break
    return np.array(ks), np.array(dirs)


def dir_stats(s):
    """s: standardized 1-d projection. Trim = drop the q largest |s|, restandardize, remeasure."""
    out = {"kurt": float((s**4).mean() - 3.0)}
    a = np.abs(s)
    order = np.argsort(a)
    for q in TRIMS:
        t = s[order[: len(s) - q]]
        t = (t - t.mean()) / t.std()
        out[f"kurt_trim{q}"] = float((t**4).mean() - 3.0)
    top10 = order[::-1][:10]
    out["share4_top10"] = float((s[top10] ** 4).sum() / (s**4).sum())
    out["n_gt4sd"] = int((a > 4).sum())
    out["n_gt6sd"] = int((a > 6).sum())
    return out, top10


def main():
    toy = pd.read_csv(os.path.join(ROOT, "results", "diag", "defect_rank_toy.csv"))
    ref_pp = dict(zip(toy["cell"], toy["top|k|_pp"]))
    rows, driver_slots, npz, fig_data = [], [], {}, {}
    labels = None
    for ci, (label, rid, space, known) in enumerate(CELLS):
        X = np.load(f"{FEAT}/{rid}/{MAN}/{space}.npy").astype(np.float64)
        if labels is None:
            labels = np.load(f"{FEAT}/{rid}/{MAN}/labels.npy")
        P, W = frame_with_basis(X, m=64)
        n, meff = P.shape
        nd = min(20, meff)
        ks, dirs = pursuit_dirs(P, nd, seed=3)
        match = abs(abs(ks[0]) - ref_pp[label])
        print(f"[forensics] {label}: dir1 |kurt| {abs(ks[0]):.2f} vs toy {ref_pp[label]:.2f} "
              f"({'OK' if match < 0.05 else 'MISMATCH'})")
        npz[f"c{ci:02d}_dirs_frame"] = dirs
        npz[f"c{ci:02d}_dirs_orig"] = dirs @ W.T
        npz[f"c{ci:02d}_kurt"] = ks
        for di in range(nd):
            s = P @ dirs[di]
            s = (s - s.mean()) / s.std()
            st, top10 = dir_stats(s)
            rows.append({"cell": label, "kind": "pp", "rank": di + 1, **st,
                         "top10_idx": ";".join(map(str, top10)),
                         "top10_lab": ";".join(map(str, labels[top10]))})
            if di < 10:
                driver_slots += [(label, int(i), int(labels[i])) for i in top10]
            if di == 0 and label in SHOWCASE:
                fig_data[label] = (s, top10, st)
        ax_kurt = (P**4).mean(0) - 3.0
        for j in np.argsort(-np.abs(ax_kurt))[:5]:
            s = P[:, j]
            s = (s - s.mean()) / s.std()
            st, top10 = dir_stats(s)
            rows.append({"cell": label, "kind": "eig", "rank": int(j) + 1, **st,
                         "top10_idx": ";".join(map(str, top10)),
                         "top10_lab": ";".join(map(str, labels[top10]))})
    diag = os.path.join(ROOT, "results", "diag")
    pd.DataFrame(rows).to_csv(os.path.join(diag, "defect_rank_outliers.csv"), index=False)
    np.savez(os.path.join(diag, "defect_rank_outlier_dirs.npz"),
             cells=np.array([c[0] for c in CELLS]), **npz)

    dr = pd.DataFrame(driver_slots, columns=["cell", "idx", "label"])
    rec = (dr.groupby("idx").agg(label=("label", "first"), slots=("cell", "size"),
                                 n_cells=("cell", "nunique"),
                                 cells=("cell", lambda c: ";".join(sorted(set(c))[:4])))
           .sort_values("slots", ascending=False).reset_index())
    rec.head(50).to_csv(os.path.join(diag, "defect_rank_outlier_drivers.csv"), index=False)
    print(f"[forensics] driver recurrence: {len(dr)} slots, {dr['idx'].nunique()} distinct "
          f"images; top: {rec.head(8)[['idx', 'label', 'slots', 'n_cells']].to_dict('records')}")

    figdir = os.path.join(ROOT, "results", "figures", "diag")
    os.makedirs(figdir, exist_ok=True)
    fig, axes = plt.subplots(2, 3, figsize=(13, 6.5), constrained_layout=True)
    xg = np.linspace(-5, 5, 200)
    for ax, label in zip(axes.ravel(), SHOWCASE):
        s, top10, st = fig_data[label]
        ax.hist(s, bins=120, density=True, color="#7a93b5", alpha=0.9)
        ax.plot(xg, np.exp(-xg**2 / 2) / np.sqrt(2 * np.pi), color="#333333", ls="--", lw=1.2)
        ax.plot(s[top10], np.full(10, 2e-5), "|", color="#c23b22", ms=14, mew=1.6)
        ax.set_yscale("log")
        ax.set_ylim(bottom=1e-5)
        ax.set_title(f"{label} — pursuit dir 1\nkurt {st['kurt']:+.1f} -> trim10 "
                     f"{st['kurt_trim10']:+.2f}, top-10 share {st['share4_top10']:.0%}",
                     fontsize=9)
        ax.grid(alpha=0.25)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
    axes[0, 0].text(0.02, 0.95, "dashed: N(0,1) reference\nred ticks: top-10 driver images",
                    transform=axes[0, 0].transAxes, fontsize=8, va="top", color="#333333")
    fig.suptitle("Pursuit direction 1 per cell: standardized projection (log density)", fontsize=11)
    fig.savefig(os.path.join(figdir, "defect_rank_outliers.png"), dpi=150)
    print(f"[forensics] wrote {figdir}/defect_rank_outliers.png")


if __name__ == "__main__":
    main()
