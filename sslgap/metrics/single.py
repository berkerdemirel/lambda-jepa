"""Metrics on a single feature matrix X [N, d] (float64 numpy). Multi-valued metrics return a
dict of sub-metrics; the battery flattens them into rows."""
import numpy as np


def uniformity(X, n_sub=4096, seed=0):
    """Wang–Isola uniformity: log E exp(-2 ||u_i - u_j||^2) on L2-NORMALIZED features (the metric
    is defined on the sphere — normalization happens here, not in the battery variant machinery).
    Lower = more spread. Subsampled pairs, matching sslx eval_inet.align_unif."""
    rng = np.random.default_rng(seed)
    x = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-12)
    if x.shape[0] > n_sub:
        x = x[rng.choice(x.shape[0], n_sub, replace=False)]
    sq = ((x[:, None, :] - x[None, :, :]) ** 2).sum(-1) if x.shape[0] <= 2048 else None
    if sq is None:                                    # blockwise pdist to bound memory
        n = x.shape[0]
        acc, cnt = 0.0, 0
        for i in range(0, n, 1024):
            d = ((x[i:i + 1024, None, :] - x[None, :, :]) ** 2).sum(-1)
            acc += np.exp(-2.0 * d).sum() - np.exp(0.0) * d.shape[0]   # drop self-pairs
            cnt += d.shape[0] * (n - 1)
        return float(np.log(acc / cnt))
    iu = np.triu_indices_from(sq, k=1)
    return float(np.log(np.exp(-2.0 * sq[iu]).mean()))


def variance_floor(X, gamma=1.0):
    """VICReg's variance criterion read as a meter: per-dim std profile. hinge = the VICReg term
    value at gamma (exactly satisfied at z by VICReg training); floor fractions are
    scale-relative so they compare across spaces."""
    s = X.std(0)
    m = s.mean() + 1e-12
    return {"hinge": float(np.maximum(0.0, gamma - s).mean()),
            "frac_below_half_mean": float((s < 0.5 * m).mean()),
            "frac_below_tenth_mean": float((s < 0.1 * m).mean()),
            "min_over_mean_std": float(s.min() / m)}


def offdiag_redundancy(X):
    """Barlow/VICReg decorrelation read as a meter, on standardized features:
    off-diagonal mean square of the correlation matrix + mean |corr|."""
    x = (X - X.mean(0)) / (X.std(0) + 1e-12)
    C = (x.T @ x) / (x.shape[0] - 1)
    d = C.shape[0]
    off = C[~np.eye(d, dtype=bool)]
    return {"offdiag_msq": float((off ** 2).mean()), "mean_abs_corr": float(np.abs(off).mean())}


def collapse_margin(X, n_sub=2048, seed=0):
    """Sanity tier: distance from complete collapse."""
    rng = np.random.default_rng(seed)
    s = X.std(0)
    x = X[rng.choice(X.shape[0], min(n_sub, X.shape[0]), replace=False)]
    d2 = ((x[:, None, :] - x[None, :, :]) ** 2).sum(-1)
    np.fill_diagonal(d2, np.inf)
    return {"trace_cov": float((X.var(0)).sum()), "min_std": float(s.min()),
            "nn_dist_p5": float(np.percentile(np.sqrt(d2.min(1)), 5))}
