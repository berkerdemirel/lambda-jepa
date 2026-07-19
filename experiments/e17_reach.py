"""E17 follow-up (agenda i): reachability-with-cost v3 — per-class connectivity, NULL-CALIBRATED.

v1 (2026-07-17 pilot) measured within-class connectivity only; Berker's catch: with no negative
reference, a fully COLLAPSED space scores perfectly. v3 (design delegated to Claude same day)
adopts the battery's data-vs-null convention (D-040 precedent) + Berker's negative-relay idea:

Geometry (L2-normalized features; chordal metric; block-(m) conventions): per image, the 8-view
cloud has centroid x̄ and radius r. Edge cost between two clouds w = max(0, ‖x̄_i−x̄_j‖ −
(r_i+r_j)): zero iff the balls touch. Radius convention (Berker 2026-07-17, second pilot):
**r_mean PRIMARY** — the r90 (~furthest-augmentation) offset inflates clouds until class AND
null both saturate at full percolation (margin 0 vs 0 = unreadable); the tighter mean-radius
ball keeps the detector in range. Both variants are computed end-to-end (within + null + relay)
so saturation is diagnosable in the data; if a space saturates at r_mean too, the queued
fallback is an α·r_mean scale sweep (α < 1 = require overlap depth).

Per class c (n=100 instances), three rows:
  WITHIN  — complete graph over the class's own clouds: perc_seed = Σ|C|²/n² over zero-cost
            components (= P[two uniformly drawn instances are connected free]), perc_lcc,
            n_comp, mst_cost (total bridge gap to visit all instances), mst_max (worst bridge).
  NULL    — the SAME construction on M=3 draws of n instances sampled uniformly from OTHER
            classes. Collapse ⇒ class and null both saturate ⇒ margins → 0: the instrument
            calls trivial connectivity trivial. Margins: perc_margin = perc_seed − null_perc_seed;
            costs reported as the raw pair (ratios of two →0 numbers are unstable).
  RELAY   — Berker's interleaving axis: span the class through ANY territory. Full 10k-cloud
            graph, sparsified to each node's 64 lowest-cost edges (+ every class's complete
            subgraph, so within-class routing is never truncated); zero-cost components of that
            graph give perc_free_seed (connected-through-anything probability); Dijkstra metric
            closure over the class's nodes gives mst_free; relay_ratio = mst_free / mst_cost
            ∈ (0,1] (1 = the class is self-connected; small = its parts are joined only through
            foreign clouds — the geometry that manufactures wrong kNN neighbors).
            perc_borrow = perc_free_seed − perc_seed = connectivity borrowed from negatives.

Dropped from v1: gap_min (support variant) — the under/over-sampling concern it hedged is
second-order once margins are null-calibrated; one estimator family, declared.

Pilot: python experiments/e17_reach.py pilot [cls ...]   (lejepa families, verbose)
Sweep:  python experiments/e17_reach.py                  (all runs -> results/diag/e17_reach.csv)
"""
import csv
import json
import sys

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components, dijkstra

ROOT = "/nfs/scistore19/locatgrp/bdemirel/ssl_project"
RUNS = {
    "lejepa/ctrl": ("in100.lejepa.s0.e17c.ext", "student.z.embed"),
    "lejepa/sigreg": ("in100.lejepa.s0.hpull_sigreg.ext", "student.z.embed"),
    "lejepa/sigreg_inv": ("in100.lejepa.s0.hpull_sigreg_inv.ext", "student.z.embed"),
    "e12/C1": ("in100.lejepa.s0.e12c1.ext", "student.z.embed"),
    "e12/f2": ("in100.lejepa.s0.e12f2.ext", "student.z.embed"),
    "e12/f8": ("in100.lejepa.s0.e12f8.ext", "student.z.embed"),
    "e12/f7": ("in100.lejepa.s0.e12f7.ext", "student.z.embed"),
    "simclr/ctrl": ("in100.simclr.s0.e17c.ext", "student.h.cls"),
    "simclr/uniform": ("in100.simclr.s0.hpull_uniform.ext", "student.h.cls"),
    "simclr/uniform_align": ("in100.simclr.s0.hpull_uniform_align.ext", "student.h.cls"),
    "byol/ctrl": ("in100.byol.s0.e17c.ext", "student.h.cls"),
    "byol/align": ("in100.byol.s0.hpull_align.ext", "student.h.cls"),
    "dino/ctrl": ("in100.dino.s0.e17c.ext", "teacher.h.cls"),
    "dino/protoce": ("in100.dino.s0.hpull_protoce.ext", "teacher.h.cls"),
    "vicreg/ctrl": ("in100.vicreg.s0.e17c.ext", "student.h.cls"),
    "vicreg/varcov_c015": ("in100.vicreg.s0.hpull_varcov.c015.ext", "student.h.cls"),
}
PILOT_RUNS = ["lejepa/ctrl", "lejepa/sigreg", "lejepa/sigreg_inv", "e12/C1", "e12/f2"]
KNN_EDGES, NULL_DRAWS = 64, 3
# Second pilot: f2's space touches universally at ANY fixed radius (null saturates even at
# r_mean) — so the margin is additionally swept over shrunken balls α·r_mean; margin_max /
# alpha_star = class-specificity at the space's own discriminative operating point (a detector
# with unknown gain calibrates its own threshold). Berker's "margins are not tight enough".
ALPHAS = (0.5, 0.625, 0.75, 0.875, 1.0)   # pilot-3: α ≤ .5 dead everywhere; peaks live in [.75, 1]


def load_clouds(run_id, space):
    d = f"{ROOT}/features/{run_id}/in100.pairs100.v1@audit_v1.o8"
    V = json.load(open(f"{d}/meta.json"))["v"]
    views = np.stack([np.load(f"{d}/{space}.view{k}.npy") for k in range(V)], 1).astype(np.float64)
    views /= np.linalg.norm(views, axis=2, keepdims=True) + 1e-12
    xb = views.mean(1)
    dv = np.linalg.norm(views - xb[:, None], axis=2)
    return xb, {"r90": np.quantile(dv, 0.9, axis=1), "r_mean": dv.mean(1)}, \
        np.load(f"{d}/labels.npy")


def pair_costs(xb, r, idx):
    D = np.linalg.norm(xb[idx][:, None] - xb[idx][None], axis=2)
    W = np.maximum(0.0, D - (r[idx][:, None] + r[idx][None]))
    np.fill_diagonal(W, 0.0)
    return W


def mst_prim(W):
    n = len(W)
    in_tree = np.zeros(n, bool)
    d = np.full(n, np.inf)
    d[0], total, mx = 0.0, 0.0, 0.0
    for _ in range(n):
        i = np.argmin(np.where(in_tree, np.inf, d))
        in_tree[i], total, mx = True, total + d[i], max(mx, d[i])
        d = np.minimum(d, W[i])
    return total, mx


def graph_stats(W):
    n = len(W)
    adj = (W == 0) & ~np.eye(n, dtype=bool)
    lab = connected_components(coo_matrix(adj), directed=False)[1]
    sizes = np.bincount(lab)
    mst_cost, mst_max = mst_prim(W)
    return {"perc_lcc": sizes.max() / n, "perc_seed": float((sizes ** 2).sum()) / n ** 2,
            "n_comp": len(sizes), "mst_cost": mst_cost, "mst_max": mst_max}


def full_graph(xb, r, y):
    """64-lowest-cost edges per node + every class's complete subgraph; returns the sparse
    cost graph and the zero-cost component labels of the full space."""
    n = len(xb)
    sq = (xb ** 2).sum(1)
    src, dst, wgt = [], [], []
    for a in range(0, n, 500):
        b = min(a + 500, n)
        D = np.sqrt(np.maximum(0.0, sq[a:b, None] + sq[None] - 2 * xb[a:b] @ xb.T))
        W = np.maximum(0.0, D - (r[a:b, None] + r[None]))
        W[np.arange(b - a), np.arange(a, b)] = np.inf
        near = np.argpartition(W, KNN_EDGES, axis=1)[:, :KNN_EDGES]
        src.append(np.repeat(np.arange(a, b), KNN_EDGES))
        dst.append(near.ravel())
        wgt.append(np.take_along_axis(W, near, 1).ravel())
    for c in np.unique(y):
        idx = np.flatnonzero(y == c)
        Wc = pair_costs(xb, r, idx)
        iu, ju = np.triu_indices(len(idx), 1)
        src.append(idx[iu])
        dst.append(idx[ju])
        wgt.append(Wc[iu, ju])
    src, dst, wgt = map(np.concatenate, (src, dst, wgt))
    # canonical (lo,hi) + dedup: coo->csr SUMS duplicates (a pair present in both the kNN list
    # and a class-complete block would double its cost). Upper-tri storage suffices: csgraph
    # directed=False walks stored edges both ways; explicit zeros are true zero-weight edges.
    lo, hi = np.minimum(src, dst), np.maximum(src, dst)
    _, first = np.unique(lo * n + hi, return_index=True)
    g = coo_matrix((wgt[first], (lo[first], hi[first])), shape=(n, n)).tocsr()
    zero = g.copy()
    zero.data = (zero.data == 0).astype(float)
    zero.eliminate_zeros()
    lab_free = connected_components(zero, directed=False)[1]
    return g, lab_free


def perc_seed_at(D, rsum, a):
    W = np.maximum(0.0, D - a * rsum)
    lab = connected_components(coo_matrix((W == 0) & ~np.eye(len(W), dtype=bool)),
                               directed=False)[1]
    s = np.bincount(lab)
    return float((s ** 2).sum()) / len(W) ** 2


def margin_profile(xb, r, y, c, rng):
    """perc-margin (class − null) at shrunken balls α·r; returns per-α margins + max/argmax."""
    idx = np.flatnonzero(y == c)
    sets = [idx] + [rng.choice(np.flatnonzero(y != c), size=len(idx), replace=False)
                    for _ in range(NULL_DRAWS)]
    D = [np.linalg.norm(xb[s][:, None] - xb[s][None], axis=2) for s in sets]
    R = [r[s][:, None] + r[s][None] for s in sets]
    out, best_m, best_a = {}, -2.0, 1.0
    for a in ALPHAS:
        m = perc_seed_at(D[0], R[0], a) - np.mean([perc_seed_at(D[k], R[k], a)
                                                   for k in range(1, len(sets))])
        out[f"margin_a{int(a * 100)}"] = round(float(m), 4)
        if m > best_m:
            best_m, best_a = float(m), a
    out["margin_max"], out["alpha_star"] = round(best_m, 4), best_a
    return out


def variant_stats(xb, r, y, lab_free, g, c, rng):
    """One radius variant's full readout block for one class (unsuffixed keys)."""
    idx = np.flatnonzero(y == c)
    row = {f"{k}": round(float(v), 4) for k, v in graph_stats(pair_costs(xb, r, idx)).items()}
    nulls = []
    for _ in range(NULL_DRAWS):
        nidx = rng.choice(np.flatnonzero(y != c), size=len(idx), replace=False)
        nulls.append(graph_stats(pair_costs(xb, r, nidx)))
    row["null_perc_seed"] = round(float(np.mean([s["perc_seed"] for s in nulls])), 4)
    row["null_mst_cost"] = round(float(np.mean([s["mst_cost"] for s in nulls])), 4)
    row["perc_margin"] = round(row["perc_seed"] - row["null_perc_seed"], 4)
    cnt = np.bincount(lab_free[idx])
    row["perc_free_seed"] = round(float((cnt ** 2).sum()) / len(idx) ** 2, 4)
    row["perc_borrow"] = round(row["perc_free_seed"] - row["perc_seed"], 4)
    dm = dijkstra(g, directed=False, indices=idx)[:, idx]
    mst_free, _ = mst_prim(np.minimum(dm, dm.T))
    row["mst_free"] = round(float(mst_free), 4)
    row["relay_ratio"] = round(min(1.0, mst_free / row["mst_cost"]), 4) \
        if row["mst_cost"] > 1e-9 else 1.0
    return row


def main():
    pilot = len(sys.argv) > 1 and sys.argv[1] == "pilot"
    cls_list = [int(a) for a in sys.argv[2:]] if len(sys.argv) > 2 else [0, 42]
    out_rows = []
    for tag in (PILOT_RUNS if pilot else list(RUNS)):
        rid, space = RUNS[tag]
        xb, rads, y = load_clouds(rid, space)
        classes = cls_list if pilot else sorted(set(y.tolist()))
        rows = {c: {"class": int(c), "tag": tag, "run": rid} for c in classes}
        for rv in ("r_mean", "r90"):
            g, lab_free = full_graph(xb, rads[rv], y)
            rng = np.random.default_rng(0)
            for c in classes:
                blk = variant_stats(xb, rads[rv], y, lab_free, g, c, rng)
                rows[c].update({f"{k}_{rv}": v for k, v in blk.items()})
                if pilot:
                    print(f"{tag:20s} cls {c:3d} {rv:6s} | within: comp {blk['n_comp']:3.0f} "
                          f"perc {blk['perc_seed']:.2f} mst {blk['mst_cost']:7.3f} | "
                          f"null: perc {blk['null_perc_seed']:.2f} mst {blk['null_mst_cost']:7.3f} | "
                          f"MARGIN {blk['perc_margin']:+.2f} | free: perc {blk['perc_free_seed']:.2f} "
                          f"borrow {blk['perc_borrow']:+.2f}", flush=True)
        rng = np.random.default_rng(1)
        for c in classes:
            prof = margin_profile(xb, rads["r_mean"], y, c, rng)
            rows[c].update(prof)
            if pilot:
                print(f"{tag:20s} cls {c:3d} alpha  | " +
                      " ".join(f"m@{a:.2f} {prof[f'margin_a{int(a*100)}']:+.2f}" for a in ALPHAS) +
                      f" | MAX {prof['margin_max']:+.2f} @a={prof['alpha_star']:.2f}", flush=True)
        out_rows += [rows[c] for c in classes]
        if not pilot:
            med = {k: round(float(np.median([rows[c][k] for c in classes])), 3)
                   for k in ("margin_max", "alpha_star", "perc_margin_r_mean", "perc_seed_r_mean",
                             "null_perc_seed_r_mean", "mst_cost_r_mean", "null_mst_cost_r_mean",
                             "perc_borrow_r_mean")}
            print(tag, med, flush=True)
    if not pilot:
        out = f"{ROOT}/results/diag/e17_reach.csv"
        with open(out, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(out_rows[0]))
            w.writeheader()
            w.writerows(out_rows)
        print("wrote", out)


if __name__ == "__main__":
    main()
