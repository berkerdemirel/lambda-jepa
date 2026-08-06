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
