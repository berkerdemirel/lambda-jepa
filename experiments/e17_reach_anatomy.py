"""Distance anatomy behind the reach instrument (Berker 2026-07-17: "if positives and negatives
have the same margins then how does knn work better for f2 … my expectation would be invariances
having very small distances"). Per pilot run, ECDFs over all 10k orbit instances of three
distances at declared h (L2-normalized):
  r        = view-cloud radius (mean view→centroid),
  d_intra  = distance to the NEAREST same-class centroid,
  d_inter  = distance to the NEAREST other-class centroid.
x is normalized by the run's median d_inter (scale-free across spaces; raw medians annotated).
What it separates: kNN reads the ORDERING d_intra < d_inter (the blue-vs-orange horizontal gap);
zero-cost percolation reads the THRESHOLD r ≳ d_intra/2 (gray vs blue) — a space can order
perfectly while touching universally (f2), which is exactly how the raw margin saturates while
kNN thrives. → results/figures/e17/e17_reach_anatomy.png"""
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = "/nfs/scistore19/locatgrp/bdemirel/ssl_project"
BLUE, ORANGE, GRAY = "#3d65d0", "#e07b39", "#8a8a8a"
RUNS = [("lejepa/ctrl", "in100.lejepa.s0.e17c.ext", "student.z.embed"),
        ("lejepa/sigreg", "in100.lejepa.s0.hpull_sigreg.ext", "student.z.embed"),
        ("lejepa/sigreg_inv", "in100.lejepa.s0.hpull_sigreg_inv.ext", "student.z.embed"),
        ("e12/C1", "in100.lejepa.s0.e12c1.ext", "student.z.embed"),
        ("e12/f2", "in100.lejepa.s0.e12f2.ext", "student.z.embed")]
KNN = {"lejepa/ctrl": 52.4, "lejepa/sigreg": 48.1, "lejepa/sigreg_inv": 40.8,
       "e12/C1": 53.1, "e12/f2": 60.5}


def anatomy(run_id, space):
    d = f"{ROOT}/features/{run_id}/in100.pairs100.v1@audit_v1.o8"
    V = json.load(open(f"{d}/meta.json"))["v"]
    views = np.stack([np.load(f"{d}/{space}.view{k}.npy") for k in range(V)], 1).astype(np.float64)
    views /= np.linalg.norm(views, axis=2, keepdims=True) + 1e-12
    xb = views.mean(1)
    r = np.linalg.norm(views - xb[:, None], axis=2).mean(1)
    y = np.load(f"{d}/labels.npy")
    n = len(xb)
    sq = (xb ** 2).sum(1)
    d_in, d_out = np.empty(n), np.empty(n)
    for a in range(0, n, 500):
        b = min(a + 500, n)
        D = np.sqrt(np.maximum(0.0, sq[a:b, None] + sq[None] - 2 * xb[a:b] @ xb.T))
        D[np.arange(b - a), np.arange(a, b)] = np.inf
        same = y[a:b, None] == y[None]
        d_in[a:b] = np.where(same, D, np.inf).min(1)
        d_out[a:b] = np.where(same, np.inf, D).min(1)
    return r, d_in, d_out


def ecdf(ax, x, scale, color, lw=2):
    xs = np.sort(x) / scale
    ax.plot(xs, np.arange(1, len(xs) + 1) / len(xs), color=color, lw=lw)


def main():
    fig, axes = plt.subplots(1, len(RUNS), figsize=(3.1 * len(RUNS), 3.6), sharey=True)
    for ax, (tag, rid, space) in zip(axes, RUNS):
        r, d_in, d_out = anatomy(rid, space)
        s = np.median(d_out)
        ecdf(ax, d_in, s, BLUE)
        ecdf(ax, d_out, s, ORANGE)
        ecdf(ax, 2 * r, s, GRAY)
        touch = float((2 * r >= d_in).mean())
        ax.set_title(f"{tag}\nknn {KNN[tag]:.1f} · touch {100 * touch:.0f}%", fontsize=10)
        ax.annotate(f"med d_inter {s:.2f}\nmed d_intra {np.median(d_in):.2f}\n"
                    f"med 2r {np.median(2 * r):.2f}", (0.03, 0.72),
                    xycoords="axes fraction", fontsize=8, color="#444444")
        ax.axvline(1.0, color="#bbbbbb", lw=0.7, ls=":")
        ax.set_xlim(0, 2.2)
        ax.grid(alpha=0.15)
        ax.set_xlabel("distance / med d_inter")
    axes[0].set_ylabel("ECDF")
    axes[0].annotate("d_intra (nearest same-class)", (0.30, 0.42), xycoords="axes fraction",
                     fontsize=9, color=BLUE)
    axes[0].annotate("d_inter (nearest other-class)", (0.30, 0.30), xycoords="axes fraction",
                     fontsize=9, color=ORANGE)
    axes[0].annotate("2r (cloud diameter)", (0.30, 0.18), xycoords="axes fraction",
                     fontsize=9, color=GRAY)
    fig.suptitle("kNN reads the blue–orange ORDERING gap; percolation reads the gray-vs-blue THRESHOLD"
                 " — they can disagree", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(f"{ROOT}/results/figures/e17/e17_reach_anatomy.png", dpi=160)
    print("wrote e17_reach_anatomy.png")


if __name__ == "__main__":
    main()
