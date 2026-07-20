"""Berker's traversal probe (2026-07-17): "go one by one (through unvisited clouds), check the
closest augmented cloud for each cloud — would we be able to travel through the positives?"
That walk IS Prim's algorithm on SUPPORT distances: grow the visited set by always taking the
cheapest support-distance hop to an unvisited positive. Instance-level, not distributional —
every plotted point is one actual hop.

Per class, on the o8 orbit (L2-normalized):
  S[i,j] = min over view pairs ‖v_ik − v_jl‖  (true point-cloud gap; NO ball assumption)
  walk   = Prim from the medoid; hop profile = the 99 insertion costs
  purity = per hop, is the new positive's hop cost ≤ its support distance to the nearest
           FOREIGN cloud (exact min-view-pair to its 32 nearest foreign centroids)? A pure hop
           means the walk never had a negative closer than the positive it took.
Anisotropy datum (Berker's "clusters align in a few axes, not balls"):
  per-cloud top-eigenvalue share of the 8-view covariance vs the SAME statistic on 8 iid
  Gaussian points (the 8-sample null — 8 points in 512-d look elongated by chance; only the
  EXCESS is real), + |cos| between top axes of MST-adjacent clouds vs random same-class pairs
  (alignment above random = the filament story).
DECLARED CAVEAT: 8 views undersample the orbit support (cloud diameters ~1.3 with 8 samples ⇒
min-view-pair OVERSTATES true support gaps). This walk is an upper bound on travel cost; the
clean version needs an o32 re-extract of selected runs.

Pilot: python experiments/e17_traverse.py [cls ...]  (defaults 0 42; 5 lejepa-family runs)
Figure: results/figures/e17/e17_traverse.png + prints RAW per-run numbers.
"""
import json
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = "/nfs/scistore19/locatgrp/bdemirel/ssl_project"
BLUE, ORANGE, GREEN, RED, GRAY = "#3d65d0", "#e07b39", "#3a8c5c", "#c04f4f", "#8a8a8a"
RUNS = [("lejepa/ctrl", "in100.lejepa.s0.e17c.ext", "student.z.embed", BLUE),
        ("lejepa/sigreg", "in100.lejepa.s0.hpull_sigreg.ext", "student.z.embed", GREEN),
        ("lejepa/sigreg_inv", "in100.lejepa.s0.hpull_sigreg_inv.ext", "student.z.embed", RED),
        ("e12/C1", "in100.lejepa.s0.e12c1.ext", "student.z.embed", GRAY),
        ("e12/f2", "in100.lejepa.s0.e12f2.ext", "student.z.embed", ORANGE)]
N_FOREIGN = 32


O32 = {"in100.lejepa.s0.e17c.ext": "in100.lejepa.s0.o32",       # ctrl's o32 rides the tag-less ckpt name
       "in100.lejepa.s0.e12f2.ext": "in100.lejepa.s0.e12f2.o32",
       "in100.lejepa.s0.hpull_sigreg_inv.ext": "in100.lejepa.s0.hpull_sigreg_inv.o32"}


def load(run_id, space):
    """Prefer the o32 store when it exists; fall back to o8. Prints which store was used."""
    import os
    d32 = f"{ROOT}/features/{O32.get(run_id, '_none_')}/in100.pairs100.v1@audit_v1.o32"
    d = d32 if os.path.isdir(d32) else f"{ROOT}/features/{run_id}/in100.pairs100.v1@audit_v1.o8"
    print(f"[store] {run_id} <- {d.split('features/')[-1]}", flush=True)
    V = json.load(open(f"{d}/meta.json"))["v"]
    views = np.stack([np.load(f"{d}/{space}.view{k}.npy") for k in range(V)], 1).astype(np.float64)
    views /= np.linalg.norm(views, axis=2, keepdims=True) + 1e-12
    return views, np.load(f"{d}/labels.npy")


def support_dists(Va, Vb):
    """min view-pair distance matrix between two cloud sets [na,8,D],[nb,8,D] -> [na,nb]."""
    na, v, dd = Va.shape
    nb = Vb.shape[0]
    A, B = Va.reshape(na * v, dd), Vb.reshape(nb * v, dd)
    sq = ((A ** 2).sum(1)[:, None] + (B ** 2).sum(1)[None]) - 2 * A @ B.T
    return np.sqrt(np.maximum(0.0, sq)).reshape(na, v, nb, v).min(axis=(1, 3))


def prim_walk(S, seed):
    """Berker's walk: repeatedly hop to the cheapest unvisited cloud. Returns hop costs +
    the visited node per hop (insertion order)."""
    n = len(S)
    visited = np.zeros(n, bool)
    visited[seed] = True
    d = S[seed].copy()
    hops, nodes = [], []
    for _ in range(n - 1):
        d[visited] = np.inf
        j = int(np.argmin(d))
        hops.append(float(d[j]))
        nodes.append(j)
        visited[j] = True
        d = np.minimum(d, S[j])
    return np.array(hops), np.array(nodes)


def top_share(V):
    """Per-cloud top-eigenvalue share of the view covariance."""
    Vc = V - V.mean(1, keepdims=True)
    s = np.linalg.svd(Vc, compute_uv=False) ** 2
    return s[:, 0] / s.sum(1)


def top_axes(V):
    Vc = V - V.mean(1, keepdims=True)
    _, _, Wt = np.linalg.svd(Vc)
    return Wt[:, 0]


def main():
    cls_list = [int(a) for a in sys.argv[1:]] or [0, 42]
    rng = np.random.default_rng(0)
    # null is per (n_views, D) of the store actually loaded — o8 and o32 stores mix in RUNS
    # (an 8-point cloud looks elongated by chance far more than a 32-point one)
    null_cache = {}

    def gauss_null(V, D):
        if (V, D) not in null_cache:
            null_cache[V, D] = float(np.mean(
                [np.linalg.svd(x - x.mean(0), compute_uv=False)[0] ** 2 /
                 (np.linalg.svd(x - x.mean(0), compute_uv=False) ** 2).sum()
                 for x in rng.standard_normal((400, V, D))]))
        return null_cache[V, D]
    fig, axes = plt.subplots(1, len(cls_list) + 1, figsize=(5.2 * (len(cls_list) + 1), 4.0))
    aniso_rows = []
    for tag, rid, space, col in RUNS:
        views, y = load(rid, space)
        xb = views.mean(1)
        sq = (xb ** 2).sum(1)
        scale = None
        for pi, c in enumerate(cls_list):
            idx = np.flatnonzero(y == c)
            Vc = views[idx]
            S = support_dists(Vc, Vc)
            np.fill_diagonal(S, np.inf)
            if scale is None:                                   # per-run scale: median inter-NN
                Dc = np.sqrt(np.maximum(0.0, sq[:, None][idx] + sq[None] - 2 * xb[idx] @ xb.T))
                scale = float(np.median(np.where(y[None] != c, Dc, np.inf).min(1)))
            seed = int(np.argmin(np.where(np.isinf(S), 0, S).sum(1)))
            hops, nodes = prim_walk(S, seed)
            D_all = np.sqrt(np.maximum(0.0, sq[idx][:, None] + sq[None] - 2 * xb[idx] @ xb.T))
            D_all[:, idx] = np.inf
            near_f = np.argsort(D_all, axis=1)[:, :N_FOREIGN]
            fmin = np.array([support_dists(Vc[i:i + 1], views[near_f[i]])[0].min()
                             for i in range(len(idx))])
            purity = float((hops <= fmin[nodes]).mean())
            ax = axes[pi]
            ax.plot(np.arange(1, len(hops) + 1), np.sort(hops) / scale, color=col, lw=2)
            ax.annotate(f"{tag}  purity {100 * purity:.0f}%", (0.03, 0.94 - 0.07 * RUNS.index((tag, rid, space, col))),
                        xycoords="axes fraction", fontsize=9, color=col)
            print(f"{tag:20s} cls {c:3d}: med hop {np.median(hops) / scale:.2f} "
                  f"p90 {np.quantile(hops, .9) / scale:.2f} max {hops.max() / scale:.2f} "
                  f"(x med d_inter) | purity {100 * purity:.0f}% | seed medoid {seed}", flush=True)
        share = top_share(views[::7])                                   # full clouds, ~1400, cheap
        null_run = gauss_null(views.shape[1], views.shape[2])
        ax_al = None
        # axis alignment on the pilot classes' MST edges vs random same-class pairs
        aligns, rands = [], []
        for c in cls_list:
            idx = np.flatnonzero(y == c)
            Vc = views[idx]
            S = support_dists(Vc, Vc)
            np.fill_diagonal(S, np.inf)
            seed = int(np.argmin(np.where(np.isinf(S), 0, S).sum(1)))
            hops, nodes = prim_walk(S, seed)
            U = top_axes(Vc)
            prev = seed
            order = [seed] + list(nodes)
            aligns += [abs(float(U[a] @ U[b])) for a, b in zip(order[:-1], order[1:])]
            pr = rng.permutation(len(idx))
            rands += [abs(float(U[a] @ U[b])) for a, b in zip(pr[:-1], pr[1:])]
        aniso_rows.append((tag, col, float(np.median(share)),
                           float(np.median(aligns)), float(np.median(rands)), null_run,
                           views.shape[1]))
        print(f"{tag:20s} aniso: med top-eig share {aniso_rows[-1][2]:.3f} (gauss null "
              f"{null_run:.3f} @V={views.shape[1]}) | axis |cos| walk-adjacent "
              f"{aniso_rows[-1][3]:.3f} vs random-pair {aniso_rows[-1][4]:.3f}", flush=True)
    for pi, c in enumerate(cls_list):
        axes[pi].set_title(f"class {c}: sorted hop costs of the walk", fontsize=10)
        axes[pi].set_xlabel("hop rank (1..99)")
        axes[pi].set_ylabel("hop cost / med d_inter" if pi == 0 else "")
        axes[pi].grid(alpha=0.15)
    ax = axes[-1]
    for k, (tag, col, sh, al, rd, nl, V) in enumerate(aniso_rows):
        ax.plot([k - 0.15, k + 0.15], [al, rd], color=col, lw=1.2)
        ax.plot(k - 0.15, al, "o", ms=8, color=col)
        ax.plot(k + 0.15, rd, "o", ms=8, mfc="white", color=col)
        ax.annotate(f"share {sh:.2f}\nnull {nl:.2f} V{V}", (k, 0.02), ha="center",
                    fontsize=7.5, color=col)
        ax.plot([k - 0.28, k + 0.28], [nl, nl], color="#444444", lw=0.9, ls=":")
    ax.annotate("filled = walk-adjacent |cos| · open = random pair\ndotted = per-run gauss null (top-eig share, matched V,D)",
                (0.02, 0.86), xycoords="axes fraction", fontsize=8.5, color="#444444")
    ax.set_xticks(range(len(aniso_rows)))
    ax.set_xticklabels([t.split("/")[-1] for t, *_ in aniso_rows], fontsize=8)
    ax.set_title("cloud anisotropy + axis alignment", fontsize=10)
    ax.grid(alpha=0.15)
    fig.tight_layout()
    fig.savefig(f"{ROOT}/results/figures/e17/e17_traverse.png", dpi=160)
    print("wrote e17_traverse.png")


if __name__ == "__main__":
    main()
