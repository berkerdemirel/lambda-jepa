"""Berker's touch census as a battery-grade pure function (D-068 revival of the archived
e17_touch_census.py, spec verbatim 2026-07-17b): for EVERY instance as anchor over ALL
candidates, a candidate j touches anchor i iff the r_mean balls overlap on the centroid
axis — ω(i,j) = (r_i + r_j − d(x̄_i, x̄_j)) / (r_i + r_j) > 0 (the e17_intersect v3 house
definition; signed fractional overlap depth, negative = gap in radius units). Counts are
normalized to pool-size-free probabilities p_pos (touch-same) and p_neg (touch-diff),
compared ratio-like (enrichment E = p_pos/p_neg, purity vs the base rate), α=.75 columns
ride along as the operating-point robustness check, and the ω LEVELS themselves are
class-conditioned (per-anchor median ω to same-class vs foreign clouds). Per-class rows
(enrichment as ratio-of-means — per-anchor ratios blow up at deg_neg=0); run summary =
median over classes (E17 convention)."""
import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components


def touch_census(views, labels, alphas=(1.0, 0.75)):
    """views: list of V arrays (N, D), same image order; labels (N,).
    Returns (per_class_rows, summary) — rows lack the store/space id (caller adds)."""
    X = np.stack([v.astype(np.float32) for v in views])          # (V, N, D)
    y = np.asarray(labels)
    xb = X.mean(0)
    rr = np.linalg.norm(X - xb, axis=2).mean(0)                  # r_mean per anchor
    sq = (xb ** 2).sum(1)
    d = np.sqrt(np.maximum(0, sq[:, None] + sq[None] - 2 * xb @ xb.T))
    rsum = rr[:, None] + rr[None]
    same = y[:, None] == y[None]
    np.fill_diagonal(same, False)
    omega = (rsum - d) / rsum
    np.fill_diagonal(omega, np.nan)
    om_same = np.nanmedian(np.where(same, omega, np.nan), axis=1)
    om_diff = np.nanmedian(np.where(~same & ~np.isnan(omega), omega, np.nan), axis=1)
    stats = {}
    for a in alphas:
        T = (a * rsum - d) > 0
        np.fill_diagonal(T, False)
        deg_pos = (T & same).sum(1)
        deg_neg = (T & ~same).sum(1)
        stats[a] = (deg_pos / same.sum(1), deg_neg / ((~same).sum(1) - 1), deg_pos, deg_neg)
    rows = []
    for c in np.unique(y):
        m = y == c
        row = {"class": int(c),
               "omega_same_med": round(float(om_same[m].mean()), 5),
               "omega_diff_med": round(float(om_diff[m].mean()), 5)}
        for a in alphas:
            p_pos, p_neg, dp, dn = stats[a]
            pp, pn = float(p_pos[m].mean()), float(p_neg[m].mean())
            suf = "" if a == 1.0 else f"_a{int(a * 100)}"
            row[f"p_pos{suf}"] = round(pp, 5)
            row[f"p_neg{suf}"] = round(pn, 5)
            row[f"enrich{suf}"] = round(pp / pn, 2) if pn > 0 else float("inf")
            tot = dp[m].sum() + dn[m].sum()
            row[f"purity{suf}"] = round(float(dp[m].sum() / tot), 4) if tot else float("nan")
        rows.append(row)
    summary = {k: float(np.median([r[k] for r in rows if np.isfinite(r[k])]))
               for k in rows[0] if k != "class"}
    return rows, summary


def touch_law_stats(views, k_local=20):
    """Label-free touch-law aggregates (theory doc §2/§9 R1–R3; native recomputation of
    the reconstructed aggregates flagged in §10). Same geometric conventions as
    touch_census (r_mean radii, sample centers): M = median signed overlap ω over ALL
    unordered pairs; T = touching fraction (r_i + r_j > d_ij). Ω_local(k): per anchor,
    own-cloud pair energy W_i = E_{u<v}||x_iu − x_iv||² over the mean debiased center
    energy to its k nearest centers (pair debias (W_i+W_j)/(2V), the exact view-noise
    term) — cloud-local thickness; medians/means over anchors. Touch graph at α=1:
    connected components of the touch relation — component count, giant share,
    singleton count. Returns a flat dict."""
    X = np.stack([v.astype(np.float32) for v in views])          # (V, N, D)
    V, N, _ = X.shape
    xb = X.mean(0)
    rr = np.linalg.norm(X - xb, axis=2).mean(0)
    W_i = np.zeros(N)
    for u in range(V):
        for w in range(u + 1, V):
            W_i += ((X[u] - X[w]) ** 2).sum(1)
    W_i /= V * (V - 1) / 2
    sq = (xb ** 2).sum(1)
    d2 = np.maximum(0, sq[:, None] + sq[None] - 2 * xb @ xb.T)
    d = np.sqrt(d2)
    rsum = rr[:, None] + rr[None]
    iu = np.triu_indices(N, 1)
    omega = (rsum[iu] - d[iu]) / rsum[iu]
    adj = rsum > d
    np.fill_diagonal(adj, False)
    ncomp, lab = connected_components(csr_matrix(adj), directed=False)
    sizes = np.bincount(lab)
    b_loc = d2 - (W_i[:, None] + W_i[None]) / (2 * V)            # debiased center energies
    np.fill_diagonal(b_loc, np.inf)
    idx = np.argpartition(d, k_local, axis=1)[:, :k_local + 1]
    near = np.take_along_axis(b_loc, idx, axis=1)
    near[~np.isfinite(near)] = np.nan
    B_loc = np.nanmean(np.sort(near, axis=1)[:, :k_local], axis=1)
    om_loc = W_i / np.maximum(B_loc, 1e-12)
    return {"N": int(N), "V": int(V), "M_med": float(np.median(omega)),
            "T_frac": float(adj[iu[0], iu[1]].mean()),
            "omega_local_med": float(np.median(om_loc)),
            "omega_local_mean": float(om_loc.mean()),
            "n_comp": int(ncomp), "giant_share": float(sizes.max() / N),
            "n_singletons": int((sizes == 1).sum())}


def touch_graph_profile(views, alphas=(0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.15, 1.3, 1.5)):
    """Berker's tunable-inclusion cloud-count (2026-08-06: 'define a trajectory from
    each image until no img is unvisited … graphs with potentially tunable inclusion
    regime'): sweep the inclusion dial α — clouds i,j linked iff α·(r_i + r_j) > d_ij
    (α = 1 is the physical touch relation; α < 1 demands overlap depth, α > 1 admits
    near-misses) — and report the component census per α: how many separate cloud
    groups the space splits into, the giant group's share, and the singleton count.
    The α-profile is the connectivity fingerprint; its transition point is the
    informative number where the α=1 graph saturates (dense frames). Same geometric
    conventions as touch_census. Returns (rows, M_med) — one row per α."""
    X = np.stack([v.astype(np.float32) for v in views])
    V, N, _ = X.shape
    xb = X.mean(0)
    rr = np.linalg.norm(X - xb, axis=2).mean(0)
    sq = (xb ** 2).sum(1)
    d = np.sqrt(np.maximum(0, sq[:, None] + sq[None] - 2 * xb @ xb.T))
    rsum = rr[:, None] + rr[None]
    iu = np.triu_indices(N, 1)
    M_med = float(np.median((rsum[iu] - d[iu]) / rsum[iu]))
    rows = []
    for a in alphas:
        adj = (a * rsum) > d
        np.fill_diagonal(adj, False)
        ncomp, lab = connected_components(csr_matrix(adj), directed=False)
        sizes = np.bincount(lab)
        rows.append({"alpha": a, "T_frac": float(adj[iu[0], iu[1]].mean()),
                     "n_comp": int(ncomp), "giant_share": float(sizes.max() / N),
                     "n_singletons": int((sizes == 1).sum())})
    return rows, M_med
