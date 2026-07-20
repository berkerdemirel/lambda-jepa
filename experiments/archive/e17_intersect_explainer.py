"""The intersection instrument DRAWN (Berker 2026-07-17b: "explain it better … what you
measure, w same w negs; do same-class views create a path or not?"). Three panels: (1) the
objects and omega, with the high-d fact the v1/v2 nulls established (views live on their own
shell — BALLS overlap, view POINTS never mix); (2) the three comparison sets measured per
anchor (same top-20 / pool-matched negs top-20-of-99 / global-tail negs top-10-of-9900) and
the two AUCs; (3) the path answer from the data: touch_same = fraction of your 20 nearest
same-class clouds reachable gap-free (omega>0 ⇔ free percolation edge), per run, with
omega_same levels. Companion to results/diag/e17_intersect.csv. FS-only figure."""
import csv

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle, FancyArrowPatch

ROOT = "/nfs/scistore19/locatgrp/bdemirel/ssl_project"
BLUE, ORANGE, GREEN, RED, GRAY = "#3d65d0", "#e07b39", "#3a8c5c", "#c04f4f", "#8a8a8a"
rng = np.random.default_rng(5)


def cloud(ax, c, r, color, n=10, label=None, dy=None, shell=True):
    """views drawn ON the shell (the measured high-d reality), ball dashed."""
    th = rng.uniform(0, 2 * np.pi, n)
    rad = r * (0.86 + 0.14 * rng.random(n)) if shell else r * 0.5 * rng.random(n)
    ax.scatter(c[0] + rad * np.cos(th), c[1] + rad * np.sin(th), s=11, color=color,
               alpha=0.75, zorder=3)
    ax.add_patch(Circle(c, r, fill=False, ls="--", lw=1.2, ec=color, alpha=0.85))
    ax.plot(*c, marker="x", ms=7, color=color, mew=2, zorder=4)
    if label:
        ax.annotate(label, (c[0], c[1] + (dy if dy is not None else -r - 0.13)), ha="center",
                    fontsize=8.5, color=color)
    return np.array(c), r


def p1(ax):
    ax.set_title("the objects, and ω — one number per PAIR of images\n"
                 "ω(i,j) = (rᵢ + rⱼ − d) / (rᵢ + rⱼ)", fontsize=10)
    a, ra = cloud(ax, (0.0, 1.9), 0.55, BLUE, label="image i: its V augmentation\nviews (dots) + centroid ×", dy=0.62)
    b, rb = cloud(ax, (0.78, 1.9), 0.5, BLUE)
    ax.plot(*zip(a, b), color=GRAY, lw=0.8)
    ax.annotate("ω > 0 : the r_mean BALLS overlap\n= you can step cloud→cloud without\ncrossing empty space (a free edge)",
                (1.55, 1.85), fontsize=8.5, color=BLUE)
    ax.annotate("d < rᵢ+rⱼ", ((a[0] + b[0]) / 2, 1.62), ha="center", fontsize=8, color=GRAY)

    c1, r1 = cloud(ax, (0.0, 0.55), 0.5, GRAY)
    c2, r2 = cloud(ax, (1.35, 0.55), 0.45, GRAY)
    ax.plot([c1[0] + r1, c2[0] - r2], [0.55, 0.55], color=RED, lw=2.5)
    ax.annotate("ω < 0 : a GAP — |ω| = gap size in units of the\nsummed radii (ω=0: balls exactly kiss)",
                (1.62, 0.5), fontsize=8.5, color=RED)

    ax.annotate("measured high-d fact (v1/v2 nulls): views concentrate ON their own shell —\n"
                "balls overlap, but a view's nearest neighbor is ALWAYS an own-cloud sibling;\n"
                "view points never mix. \"Intersection\" = ball overlap, not point mixing.",
                (0.5, -0.62), ha="center", fontsize=8.5, color="#333333",
                bbox=dict(fc="#f5f5f5", ec="#cccccc", lw=0.7))
    ax.set_xlim(-0.85, 3.55)
    ax.set_ylim(-1.05, 2.75)


def p2(ax):
    ax.set_title("what we measure per anchor image: ω against THREE sets", fontsize=10)
    an, ra = cloud(ax, (0.0, 0.0), 0.52, "#222222", label="anchor", dy=0.6)
    for c, r in [((0.85, 0.45), 0.48), ((0.55, -0.85), 0.45), ((1.25, -0.35), 0.42)]:
        cloud(ax, c, r, BLUE, n=7)
    ax.annotate("ω_same: its 20 nearest\nSAME-CLASS clouds (of 99)", (1.15, 0.95), fontsize=8.5, color=BLUE)
    for c, r in [((-1.45, 0.75), 0.42), ((-1.7, -0.35), 0.4), ((-1.2, -1.05), 0.4)]:
        cloud(ax, c, r, ORANGE, n=7)
    ax.annotate("ω_pool: 20 nearest of a RANDOM\n99-foreign sample (pool-MATCHED\n— the fair negs)", (-2.75, 1.25), fontsize=8.5, color=ORANGE)
    cloud(ax, (0.42, 0.62), 0.4, RED, n=7)
    ax.annotate("ω_tail: the 10 nearest of ALL ~9900\nforeign clouds — the extreme tail kNN\nactually meets (wedged deepest)", (0.62, 1.55), fontsize=8.5, color=RED)
    ax.annotate("AUC_pool = P(ω_same > ω_pool)   ← \"same-class intersection better than the negs\", pool-matched\n"
                "AUC_tail = P(ω_same > ω_tail)   ← same question vs the order-statistics tail (100× larger pool)",
                (-0.55, -1.9), ha="center", fontsize=8.5, color="#333333",
                bbox=dict(fc="#f5f5f5", ec="#cccccc", lw=0.7))
    ax.set_xlim(-3.0, 2.6)
    ax.set_ylim(-2.35, 2.1)


def p3(ax):
    rows = list(csv.DictReader(open(f"{ROOT}/results/diag/e17_intersect.csv")))
    agg = {}
    for r in rows:
        agg.setdefault(r["tag"], []).append(r)
    stats = {t: {k: float(np.median([float(r[k]) for r in rs]))
                 for k in ["touch_same", "touch_tail", "w_same", "auc_pool"]}
             for t, rs in agg.items()}
    order = sorted(stats, key=lambda t: stats[t]["touch_same"])
    ypos = np.arange(len(order))
    ax.barh(ypos, [stats[t]["touch_same"] for t in order], height=0.62, color=BLUE, alpha=0.85)
    for yy, t in zip(ypos, order):
        s = stats[t]
        ax.plot(s["touch_tail"], yy, "o", ms=6, color=RED, zorder=4)
        ax.annotate(f"ω_same {s['w_same']:+.2f}", (0.02, yy), va="center", fontsize=7,
                    color="white" if s["touch_same"] > 0.25 else "#444444", zorder=5)
    ax.set_yticks(ypos, order, fontsize=7.5)
    ax.axvline(0, color="#444444", lw=0.8)
    ax.set_xlabel("bar: fraction of the 20 nearest same-class clouds reachable gap-free\n"
                  "red dot: same fraction for the 10 global-tail negs")
    ax.set_title("the path answer: do same-class clouds chain?\n"
                 "top = touch regime · bottom = gap regime (~half link ⇒ sparse chains)",
                 fontsize=10)
    ax.set_xlim(-0.03, 1.09)
    ax.grid(alpha=0.15, axis="x")


def main():
    fig, axes = plt.subplots(1, 3, figsize=(17.5, 5.0))
    for ax in axes[:2]:
        ax.set_aspect("equal")
        ax.axis("off")
    p1(axes[0])
    p2(axes[1])
    p3(axes[2])
    fig.tight_layout()
    fig.savefig(f"{ROOT}/results/figures/e17/e17_intersect_explainer.png", dpi=160)
    print("wrote e17_intersect_explainer.png", flush=True)


if __name__ == "__main__":
    main()
