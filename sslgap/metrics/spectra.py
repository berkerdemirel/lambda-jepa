"""Spectral diagnostics: covariance eigenvalues, effective rank, participation ratio, spectral-tail exponent, RankMe."""
import math

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

def pca_frame(X):
    """Variance-ordered PCA frame: (mean [d], eigenvalues [d] high->low, eigenvectors [d, d] as
    COLUMNS in that order). covariance_eigs
    returns the same spectrum without the basis."""
    x = _to_np(X)
    mu = x.mean(0)
    xc = x - mu
    lam, V = np.linalg.eigh(xc.T @ xc / (x.shape[0] - 1))
    return mu, np.clip(lam[::-1], 0.0, None), np.ascontiguousarray(V[:, ::-1])

def random_basis(d, k, seed=0):
    """Haar-random orthonormal k-frame in R^d — the count-matched control for a direction
    ablation: the same number of kept directions, no variance ordering."""
    Q, _ = np.linalg.qr(np.random.default_rng(seed).standard_normal((d, k)))
    return Q

def keep_directions(X, mu, B):
    """X projected onto the affine subspace mu + span(B), in the SAME ambient coordinates: the
    rank drops to B.shape[1], the dimension does not — so a frozen head or probe downstream sees
    an unchanged input shape and nothing is re-parameterized. B: [d, k] orthonormal columns."""
    x = _to_np(X)
    return mu + (x - mu) @ B @ B.T

def effective_rank(eigs):
    """exp(entropy of the normalized spectrum). Isotropy -> ~d; collapse -> ~1."""
    p = eigs / (eigs.sum() + 1e-12)
    p = p[p > 0]
    return float(np.exp(-(p * np.log(p)).sum()))

def spectrum_row(X):
    """One matrix's spectral summary as a flat row: scale, spectrum edges, stable/effective rank,
    participation ratio, RankMe, power-law slope, top-10 mass. Returns (summary dict,
    eigenvalues descending)."""
    X = np.asarray(X, dtype=np.float64)
    eigs = covariance_eigs(X)
    tr, lam1, d = float(eigs.sum()), float(eigs[0]), X.shape[1]
    er = effective_rank(eigs)
    return {"n": X.shape[0], "d": d,
            "rms": math.sqrt(tr / d),
            "mean_norm": float(np.linalg.norm(X.mean(0))),
            "trace": tr, "lam1": lam1, "lam_min": float(eigs[-1]),
            "stable_rank": tr / (lam1 + 1e-300),
            "effrank": er,
            "effrank_frac": er / d,
            "pr": participation_ratio(eigs),
            "rankme": rankme(X),
            "alpha": power_law_alpha(eigs),
            "top10_frac": float(eigs[:10].sum() / (tr + 1e-300)),
            "n_eig_1e3": int((eigs > 1e-3 * lam1).sum())}, eigs

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

def whiten_frame(X, alpha=0.0, rank=None):
    """Shrinkage whitening frame fit on X: (mean [d], W [d, r]), W = V_r diag(lam_r^-1/2) with the
    spectrum shrunk toward its isotropic target tr(Sigma)/d by `alpha` (alpha=1 -> isotropic
    rescale only, i.e. raw up to one global scale).

    EQUIVARIANCE — why alpha and rank ride with every number computed through this frame: at
    alpha=0 and full rank, whitening is exactly equivariant under an invertible linear map. If
    X_T = X_U A' then W_T'(x_T - mu_T) = R W_U'(x_U - mu_U) with R = Sigma_T^-1/2 A Sigma_U^1/2,
    and R R' = Sigma_T^-1/2 A Sigma_U A' Sigma_T^-1/2 = I identically, so R is orthogonal. Any
    cosine taken INSIDE the whitened space therefore cancels A exactly — that is what lets one
    test "is the map between the two models affine?" with nothing estimated. Shrinkage toward the
    isotropic target and rank truncation BOTH break that equivariance, so the affine null holds
    exactly only at the (alpha=0, rank=d) corner; away from it the null is whatever the sham arm
    reads at the SAME setting, never zero."""
    x = _to_np(X)
    mu = x.mean(0)
    xc = x - mu
    lam, V = np.linalg.eigh(xc.T @ xc / (x.shape[0] - 1))
    lam, V = np.clip(lam[::-1], 0.0, None), np.ascontiguousarray(V[:, ::-1])
    r = int(rank or len(lam))
    lam_s = (1.0 - alpha) * lam[:r] + alpha * lam.sum() / len(lam)
    return mu, (V[:, :r] / np.sqrt(lam_s + 1e-30)).astype(np.float32)

def apply_whiten(X, mu, W):
    """(X - mu) @ W -> [N, r], the frame's own coordinates."""
    return (_to_np(X) - mu) @ W

def random_linear_map(d, cond=1e4, seed=0):
    """A random INVERTIBLE linear map with a prescribed condition number: G = Q1 diag(s) Q2',
    s log-spaced over [cond^-1/2, cond^1/2]. The sham arm's recoding — G h is the same content in
    different coordinates by construction, so every statistic must read null on (h, Gh), and
    whatever it reads instead at a given (alpha, rank) IS that setting's noise floor."""
    rng = np.random.default_rng(seed)
    Q1, _ = np.linalg.qr(rng.standard_normal((d, d)))
    Q2, _ = np.linalg.qr(rng.standard_normal((d, d)))
    s = np.logspace(-0.5 * np.log10(cond), 0.5 * np.log10(cond), d)
    return (Q1 * s) @ Q2.T
