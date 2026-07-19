"""Reach v3 presentation (session 2026-07-17b, HANDOVER agenda 3): the 3-axis view of
results/diag/e17_reach.csv (16 runs x 100 classes) joined with e17_centered.csv — is
margin_max@α* a standing connectivity readout? Panels: (1) margin_max vs pos_c vs kNN —
does null-calibrated class-connectivity add an axis the pair/probe numbers don't already
carry; (2) margin_max vs the fixed-radius perc_margin — what the α-sweep rescues from
saturation; (3) the median margin(α) profiles — where each space's discriminative operating
point sits. FS-only figure -> results/figures/e17/e17_reach_3axis.png."""
import csv

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = "/nfs/scistore19/locatgrp/bdemirel/ssl_project"
ALPHAS = [50, 62, 75, 87, 100]
CMAP = {"e12/C1": ("lejepa_e12lane", "C1"), "e12/f2": ("lejepa_e12lane", "f2")}


def main():
    reach = {}
    for r in csv.DictReader(open(f"{ROOT}/results/diag/e17_reach.csv")):
        reach.setdefault(r["tag"], []).append(r)
    cent = {(r["method"], r["tag"]): r
            for r in csv.DictReader(open(f"{ROOT}/results/diag/e17_centered.csv"))}
    agg = {}
    for tag, rows in reach.items():
        mm = np.array([float(r["margin_max"]) for r in rows])
        agg[tag] = {"mm_med": np.median(mm), "mm_q1": np.quantile(mm, .25),
                    "mm_q3": np.quantile(mm, .75),
                    "astar_med": np.median([float(r["alpha_star"]) for r in rows]),
                    "pm_med": np.median([float(r["perc_margin_r_mean"]) for r in rows]),
                    "prof": [np.median([float(r[f"margin_a{a}"]) for r in rows])
                             for a in ALPHAS],
                    "cent": cent.get(CMAP.get(tag, tuple(tag.split("/", 1))))}
    fig, axes = plt.subplots(1, 3, figsize=(18.5, 5.2))

    ax = axes[0]
    xs, ys, cs = [], [], []
    pts = sorted(((t, a) for t, a in agg.items() if a["cent"] is not None),
                 key=lambda p: float(p[1]["cent"]["pos_c"]))
    prev_x, stack = -9, 0
    for tag, a in pts:
        x, y, k = float(a["cent"]["pos_c"]), a["mm_med"], float(a["cent"]["knn200"])
        stack = stack + 1 if x - prev_x < 0.02 else 0
        prev_x = x
        xs.append(x), ys.append(y), cs.append(k)
        ax.plot([x, x], [a["mm_q1"], a["mm_q3"]], color="#bbbbbb", lw=1.2, zorder=1)
        ax.annotate(f"{tag}  α*={a['astar_med']:.2f}", (x, y), textcoords="offset points",
                    xytext=(7, 5 - 11 * stack), fontsize=7)
    sc = ax.scatter(xs, ys, c=cs, cmap="viridis", s=70, zorder=3, edgecolors="white",
                    linewidths=1.2)
    fig.colorbar(sc, ax=ax, label="knn200 @ declared h (centered)", shrink=0.85)
    ax.set_xlabel("pos_c (centered positive-pair cosine — orbit tightness)")
    ax.set_ylabel("median margin_max over classes")
    ax.set_title("class-connectivity margin vs orbit tightness vs kNN\n"
                 "(13/16 runs with centered probes; whiskers = per-class IQR)", fontsize=10)
    ax.grid(alpha=0.15)

    ax = axes[1]
    for tag, a in agg.items():
        ax.plot(a["pm_med"], a["mm_med"], "o", color="#3d65d0", ms=6)
        ax.annotate(tag, (a["pm_med"], a["mm_med"]), textcoords="offset points",
                    xytext=(5, 3), fontsize=7)
    lim = [-0.02, max(a["mm_med"] for a in agg.values()) + 0.1]
    ax.plot(lim, lim, color="#8a8a8a", lw=1, ls=":")
    ax.set_xlabel("median perc_margin @ fixed α=1 (r_mean)")
    ax.set_ylabel("median margin_max @ α*")
    ax.set_title("what the α-sweep adds: points far above the diagonal are spaces\n"
                 "whose full-radius balls saturate (class ≈ null) but separate at tighter α",
                 fontsize=10)
    ax.grid(alpha=0.15)

    ax = axes[2]
    hi = {"lejepa/ctrl": "#8a8a8a", "e12/f2": "#3d65d0", "lejepa/sigreg": "#c04f4f",
          "vicreg/ctrl": "#e07b39", "dino/ctrl": "#3a8c5c", "simclr/uniform": "#8a5cb8"}
    for tag, a in agg.items():
        if tag in hi:
            ax.plot([a / 100 for a in ALPHAS], a["prof"], "-o", color=hi[tag], lw=2,
                    ms=4, label=tag)
        else:
            ax.plot([a / 100 for a in ALPHAS], a["prof"], "-", color="#cccccc", lw=0.9,
                    zorder=1)
    ax.set_xlabel("α (ball radius = α · r_mean)")
    ax.set_ylabel("median margin(α)")
    ax.set_title("margin profiles: the operating point is a property of the space\n"
                 "(gray = remaining runs)", fontsize=10)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.15)

    fig.tight_layout()
    fig.savefig(f"{ROOT}/results/figures/e17/e17_reach_3axis.png", dpi=160)
    print("wrote e17_reach_3axis.png", flush=True)


if __name__ == "__main__":
    main()
