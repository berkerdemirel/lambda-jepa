"""Spectral / geometric diagnostics. Ported from ssl_explore/sslx/geometry.py (float64 numpy,
verbatim numerics) + a gram-trick RankMe port of sslx/meters.rankme.

The kurtosis pair implements the house rule: random-slice kurtosis is a WEAK probe
(Diaconis–Freedman: random 1-D projections of almost anything look Gaussian) — non-Gaussianity
claims lead with the top-eigenvector / worst-direction statistics."""
import numpy as np


def _to_np(feats):
    return np.asarray(feats, dtype=np.float64)


def covariance_eigs(feats, standardize=False):
    x = _to_np(feats)
    x = x - x.mean(0, keepdims=True)
    if standardize:
        x = x / (x.std(0, keepdims=True) + 1e-8)
    cov = (x.T @ x) / (x.shape[0] - 1)
    eigs = np.linalg.eigvalsh(cov)[::-1]
    return np.clip(eigs, 0.0, None)


def effective_rank(eigs):
    """exp(entropy of the normalized spectrum). Isotropy -> ~d; collapse -> ~1."""
    p = eigs / (eigs.sum() + 1e-12)
    p = p[p > 0]
    return float(np.exp(-(p * np.log(p)).sum()))


def participation_ratio(eigs):
    return float((eigs.sum() ** 2) / (np.square(eigs).sum() + 1e-12))


def power_law_alpha(eigs, kmax=200):
    """alpha = -slope of log(eig) vs log(rank) over the top-k (α-ReQ convention).
    ~0 flat/isotropic; larger = steeper decay."""
    k = int(min(kmax, (eigs > 1e-12).sum()))
    if k < 5:
        return float("nan")
    x = np.log(np.arange(1, k + 1))
    y = np.log(eigs[:k] + 1e-12)
    A = np.vstack([x, np.ones_like(x)]).T
    slope = np.linalg.lstsq(A, y, rcond=None)[0][0]
    return float(-slope)


def rankme(feats, eps=1e-7):
    """RankMe (Garrido et al. 2023): exp-entropy of the UNCENTERED singular values.
    Computed via the d x d gram (sigma = sqrt(eig(X^T X))) — identical values, feasible at N=50k."""
    x = _to_np(feats)
    lam = np.clip(np.linalg.eigvalsh(x.T @ x)[::-1], 0.0, None)
    s = np.sqrt(lam)
    p = s / (s.sum() + 1e-12) + eps
    return float(np.exp(-(p * np.log(p)).sum()))


def random_slice_excess_kurtosis(feats, n_slices=256, seed=0):
    """WEAK global probe (see module docstring). Returns per-slice excess kurtosis array."""
    x = _to_np(feats)
    x = x - x.mean(0, keepdims=True)
    rng = np.random.default_rng(seed)
    A = rng.standard_normal((x.shape[1], n_slices))
    A /= np.linalg.norm(A, axis=0, keepdims=True)
    p = x @ A
    p = (p - p.mean(0)) / (p.std(0) + 1e-12)
    return (p ** 4).mean(0) - 3.0


def top_eigvec_excess_kurtosis(feats, n_top=10):
    """Excess kurtosis along the top-variance eigenvectors — where non-Gaussianity shows."""
    x = _to_np(feats)
    x = x - x.mean(0, keepdims=True)
    cov = (x.T @ x) / (x.shape[0] - 1)
    w, V = np.linalg.eigh(cov)
    V = V[:, ::-1][:, :n_top]
    p = x @ V
    p = (p - p.mean(0)) / (p.std(0) + 1e-12)
    return (p ** 4).mean(0) - 3.0


def two_nn_intrinsic_dim(feats, fraction=0.9, max_n=20000, seed=0):
    """TwoNN (Facco et al. 2017); subsampled for tractability at N=50k."""
    from sklearn.neighbors import NearestNeighbors

    x = _to_np(feats)
    if x.shape[0] > max_n:
        rng = np.random.default_rng(seed)
        x = x[rng.choice(x.shape[0], max_n, replace=False)]
    d, _ = NearestNeighbors(n_neighbors=3).fit(x).kneighbors(x)
    r1, r2 = d[:, 1], d[:, 2]
    mask = r1 > 1e-12
    mu = np.sort(r2[mask] / r1[mask])
    keep = int(fraction * len(mu))
    mu = mu[:keep]
    Femp = np.arange(1, keep + 1) / len(mu)
    xl = np.log(mu)
    yl = -np.log(1.0 - Femp + 1e-12)
    return float((xl @ yl) / (xl @ xl + 1e-12))
