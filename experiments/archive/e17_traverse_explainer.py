"""The traversal instrument DRAWN (Berker 2026-07-17b asked for the hop-cost/purity walk
explainer, e17_reach_explainer-style). Four panels: (1) the edge cost — support distance =
min view-pair between two clouds (TRUE point-cloud gap; no ball/radius assumption, unlike ω);
(2) the walk — Prim from the class medoid: every hop is the cheapest support-distance step
from the WHOLE visited set to any unvisited same-class cloud (walks branch); hop cost = the
insertion cost, profile = the 99 sorted costs / med d_inter; (3) purity — per hop, ORDINAL:
the new cloud's hop cost vs ITS OWN support distance to the nearest foreign cloud (min
view-pair over its 32 nearest foreign centroids) — pure ⇔ the walk never had a negative
closer than the positive it took; (4) the readout on record (job 62396106, hard-coded
numbers): med hop cost ⊥ purity — sigreg_inv hops cheapest and impure, f2 hops dearest and
purest. Companion data figure: e17_traverse.png. FS-only figure."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle, FancyArrowPatch

ROOT = "/nfs/scistore19/locatgrp/bdemirel/ssl_project"
BLUE, ORANGE, GREEN, RED, GRAY = "#3d65d0", "#e07b39", "#3a8c5c", "#c04f4f", "#8a8a8a"
rng = np.random.default_rng(7)

# job 62396106 printed record (per run x class): med hop (x med d_inter), purity %, store V
DATA = [("lejepa/ctrl", BLUE, 32, [(0.75, 54), (0.78, 39)]),
        ("lejepa/sigreg", GREEN, 8, [(0.92, 38), (1.07, 36)]),
        ("lejepa/sigreg_inv", RED, 32, [(0.51, 48), (0.63, 37)]),
        ("e12/C1", GRAY, 8, [(0.91, 62), (0.94, 40)]),
        ("e12/f2", ORANGE, 32, [(1.12, 78), (1.21, 58)])]


def cloud(ax, c, r, color, n=9, label=None, dy=None):
    """views on their own shell (the measured high-d reality); NO ball drawn — this
    instrument never uses a radius."""
    th = rng.uniform(0, 2 * np.pi, n)
    rad = r * (0.82 + 0.18 * rng.random(n))
    x, y = c[0] + rad * np.cos(th), c[1] + rad * np.sin(th)
    ax.scatter(x, y, s=10, color=color, alpha=0.8, zorder=3)
    ax.plot(*c, marker="x", ms=6, color=color, mew=1.8, zorder=4, alpha=0.7)
    if label:
        ax.annotate(label, (c[0], c[1] + (dy if dy is not None else -r - 0.14)), ha="center",
                    fontsize=8.5, color=color)
    return np.stack([x, y], 1), np.array(c)


def closest_pair(Pa, Pb):
    d = ((Pa[:, None] - Pb[None]) ** 2).sum(-1)
    i, j = np.unravel_index(d.argmin(), d.shape)
    return Pa[i], Pb[j], float(np.sqrt(d[i, j]))


def p1(ax):
    ax.set_title("the edge cost: SUPPORT distance between two clouds\n"
                 "S(i,j) = min over view pairs ‖v_ik − v_jl‖", fontsize=10)
    Pa, ca = cloud(ax, (0.0, 0.6), 0.55, BLUE, n=10,
                   label="image i: V augmentation\nviews (dots), centroid ×", dy=0.66)
    Pb, cb = cloud(ax, (1.85, 0.45), 0.5, BLUE, n=10, label="image j", dy=0.62)
    va, vb, _ = closest_pair(Pa, Pb)
    ax.plot([ca[0], cb[0]], [ca[1], cb[1]], color=GRAY, lw=0.9, ls=":")
    ax.annotate("centroid–centroid\n(NOT what we use)", ((ca[0] + cb[0]) / 2, 0.95),
                ha="center", fontsize=7.5, color=GRAY)
    ax.plot([va[0], vb[0]], [va[1], vb[1]], color=RED, lw=2.4, zorder=5)
    ax.annotate("S(i,j): the closest actual\nVIEW PAIR — the true gap\nbetween the point clouds",
                (0.6, -0.72), ha="center", fontsize=8.5, color=RED)
    ax.annotate("no ball, no radius, no threshold anywhere in this instrument\n"
                "(contrast ω: r_mean balls). Few views UNDERSAMPLE the orbit ⇒ S\n"
                "overstates true gaps — o8 walks are upper bounds, o32 the corrected read.",
                (0.88, -1.45), ha="center", fontsize=8, color="#333333",
                bbox=dict(fc="#f5f5f5", ec="#cccccc", lw=0.7))
    ax.set_xlim(-0.95, 2.85)
    ax.set_ylim(-1.85, 1.75)


def p2(ax):
    ax.set_title("the walk: Prim over the class's 100 clouds\n"
                 "always the CHEAPEST support-distance hop from the visited set", fontsize=10)
    pos = [(0.0, 0.0), (0.95, 0.35), (1.75, -0.1), (0.55, -1.05), (2.6, 0.5),
           (1.45, 1.25), (-0.9, -0.75)]
    pts = []
    for k, c in enumerate(pos):
        P, _ = cloud(ax, c, 0.34, BLUE, n=7)
        pts.append(P)
    ax.annotate("start: the class MEDOID\n(min total support distance)", (0.0, 0.52),
                ha="center", fontsize=8, color="#222222")
    ax.plot(*pos[0], marker="*", ms=15, color="#222222", zorder=6)
    # insertion order with a BRANCH: hop 4 leaves from cloud 1, not from the last-added 3
    edges = [(0, 1), (1, 2), (0, 3), (1, 5), (2, 4), (0, 6)]
    for k, (a, b) in enumerate(edges):
        va, vb, _ = closest_pair(pts[a], pts[b])
        ax.add_patch(FancyArrowPatch(va, vb, arrowstyle="-|>", mutation_scale=13,
                                     color=GREEN, lw=1.9, zorder=5))
        ax.annotate(f"{k + 1}", ((va[0] + vb[0]) / 2 + 0.1, (va[1] + vb[1]) / 2 + 0.12),
                    fontsize=9.5, color="#1f6b38", fontweight="bold", zorder=7,
                    bbox=dict(fc="white", ec="none", alpha=0.75, pad=0.5))
    ax.annotate("hops BRANCH: the next hop can grow from ANY\nvisited cloud, not the last-added"
                " — cheapest edge\nfrom the whole frontier (this walk IS Prim's MST)",
                (2.45, -1.15), ha="center", fontsize=8, color=GREEN)
    ax.annotate("hop cost = each insertion's S · profile = the 99 sorted costs\n"
                "scale: / median inter-class NN centroid distance (per-run)",
                (0.85, -1.85), ha="center", fontsize=8, color="#333333",
                bbox=dict(fc="#f5f5f5", ec="#cccccc", lw=0.7))
    ax.set_xlim(-1.75, 3.45)
    ax.set_ylim(-2.25, 1.95)


def p3(ax):
    ax.set_title("purity, per hop and ORDINAL: was a negative closer\nthan the positive the walk"
                 " just took?", fontsize=10)
    # pure case
    Pv, _ = cloud(ax, (-1.5, 0.55), 0.36, BLUE, n=7, label="visited", dy=0.44)
    Pn, cn = cloud(ax, (-0.45, 0.25), 0.36, BLUE, n=7, label="new cloud\n(this hop)", dy=-0.5)
    Pf, _ = cloud(ax, (0.75, 0.85), 0.36, ORANGE, n=7, label="nearest FOREIGN\ncloud", dy=-0.62)
    va, vb, _ = closest_pair(Pv, Pn)
    ax.add_patch(FancyArrowPatch(va, vb, arrowstyle="-|>", mutation_scale=12, color=GREEN, lw=2.2, zorder=5))
    fa, fb, _ = closest_pair(Pn, Pf)
    ax.plot([fa[0], fb[0]], [fa[1], fb[1]], color=RED, lw=1.6, ls="--", zorder=4)
    ax.annotate("hop ≤ foreign escape ⇒ PURE ✓", (-1.35, 1.62), ha="left", fontsize=9, color=GREEN)
    # impure case
    Pv2, _ = cloud(ax, (-1.35, -1.75), 0.36, BLUE, n=7)
    Pn2, _ = cloud(ax, (0.1, -1.95), 0.36, BLUE, n=7)
    Pf2, _ = cloud(ax, (0.55, -1.15), 0.36, ORANGE, n=7)
    va, vb, _ = closest_pair(Pv2, Pn2)
    ax.add_patch(FancyArrowPatch(va, vb, arrowstyle="-|>", mutation_scale=12, color=GREEN, lw=2.2, zorder=5))
    fa, fb, _ = closest_pair(Pn2, Pf2)
    ax.plot([fa[0], fb[0]], [fa[1], fb[1]], color=RED, lw=1.6, ls="--", zorder=4)
    ax.annotate("foreign closer than the hop ⇒ IMPURE ✗", (-0.3, -2.65), ha="center",
                fontsize=9, color=RED)
    ax.annotate("escape = the NEW cloud's own support distance to its nearest foreign\n"
                "cloud (min view-pair over its 32 nearest foreign centroids).\n"
                "Per-hop rank comparison — the ordinal family the Q3 prescription keeps.",
                (-0.3, -3.55), ha="center", fontsize=8, color="#333333",
                bbox=dict(fc="#f5f5f5", ec="#cccccc", lw=0.7))
    ax.set_xlim(-2.3, 1.85)
    ax.set_ylim(-3.95, 1.95)


def p4(ax):
    ax.set_title("the readout on record (job 62396106; classes 0 ● and 42 ○):\n"
                 "hop cost and purity are separate axes", fontsize=10)
    OFF = {"lejepa/ctrl": (0, 4.5), "lejepa/sigreg": (0, -6.5), "lejepa/sigreg_inv": (0, 6.5),
           "e12/C1": (0, 4.5), "e12/f2": (-0.045, 4.5)}
    for tag, col, V, pts in DATA:
        for k, (h, p) in enumerate(pts):
            ax.plot(h, p, "o", ms=9, color=col, mfc=col if k == 0 else "white", mew=1.6)
        dx, dy = OFF[tag]
        ax.annotate(tag + ("" if V == 32 else " (o8)"), (np.mean([h for h, _ in pts]) + dx,
                    np.max([p for _, p in pts]) + dy), ha="center", fontsize=8, color=col)
    ax.annotate("cheap hops,\nimpure walk", (0.545, 27), fontsize=8.5, color=RED, ha="center")
    ax.annotate("dearest hops,\npurest walk", (1.21, 64), fontsize=8.5, color=ORANGE, ha="center")
    ax.annotate("o8 runs: support gaps overstated (fewer views)\n— compare shapes, not levels, across store V",
                (0.85, 21.5), ha="center", fontsize=7.5, color="#555555")
    ax.set_xlabel("median hop cost (× med d_inter)")
    ax.set_ylabel("walk purity (%)")
    ax.set_xlim(0.42, 1.34)
    ax.set_ylim(17, 85)
    ax.grid(alpha=0.15)


def main():
    fig, axes = plt.subplots(1, 4, figsize=(21.5, 5.2))
    for ax in axes[:3]:
        ax.set_aspect("equal")
        ax.axis("off")
    p1(axes[0])
    p2(axes[1])
    p3(axes[2])
    p4(axes[3])
    fig.tight_layout()
    fig.savefig(f"{ROOT}/results/figures/e17/e17_traverse_explainer.png", dpi=160)
    print("wrote e17_traverse_explainer.png", flush=True)


if __name__ == "__main__":
    main()
