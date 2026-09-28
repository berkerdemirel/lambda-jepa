"""Null references (random-init backbone, Gaussian match)."""
import numpy as np

def gaussian_match(X, seed=0):
    """Sample N(mean(X), cov(X)) with X's own (N, d) — full-covariance moment match, so spectral
    metrics are preserved by construction and deviations isolate non-Gaussian structure."""
    rng = np.random.default_rng(seed)
    mu = X.mean(0)
    Xc = X - mu
    cov = (Xc.T @ Xc) / (X.shape[0] - 1)
    w, V = np.linalg.eigh(cov)
    w = np.clip(w, 0.0, None)
    A = V * np.sqrt(w)
    return rng.standard_normal(X.shape) @ A.T + mu
