"""Isotropy / Gaussianity. Epps–Pulley sliced statistic ported from ssl_explore/sslx/sigreg.py
(itself the lejepa-minimal SIGReg with seeded slices). HOUSE RULE (CLAUDE.md): the sliced statistic
is foolable — the battery always pairs it with the top-eigenvector kurtosis from spectra.py, and
isotropy claims lead with the worst-direction numbers."""
import numpy as np
import torch


def gauss_kl_full(X, shrink=1e-3):
    """Full-covariance Gaussian moment-KL to N(0,I), per dimension (D-040). The exact MOMENT
    component of KL(P‖N(0,I)) via the Pythagorean split KL(P‖N(0,I)) = KL(P‖P_G) + KL(P_G‖N(0,I))
    — the log-ratio of two Gaussians is quadratic, so only moments 1–2 enter; no shape is read.
    Decomposed: location = ‖μ‖²/2d (the cone); spectrum = Stein/Burg divergence Σ(λ−1−logλ)/2d on
    shrunk eigenvalues — the logdet barrier is the anti-collapse content the DIAGONAL moment-KL
    lacks (E17: sigreg_inv rank-15 collapse read diagKL .041). Deliberately not scale-free
    (calibration instrument)."""
    X = np.asarray(X, dtype=np.float64)
    mu = X.mean(0)
    lam = np.linalg.eigvalsh(np.cov(X, rowvar=False))
    lam = (1 - shrink) * np.clip(lam, 0.0, None) + shrink * lam.mean()
    d = X.shape[1]
    loc = float(mu @ mu / (2 * d))
    spec = float(np.sum(lam - 1 - np.log(lam)) / (2 * d))
    return {"total": loc + spec, "location": loc, "spectrum": spec}


def radial_gauss(X, seed=0, shrink=1e-3):
    """Normalized radial law vs the isotropic Gaussian (Berker 2026-07-16; D-040). Cross-fit:
    whitening moments (μ, Σ^{-1/2}) from one half, radii r² = ‖W(x−μ)‖² on the held-out half
    (in-sample whitening over-Gaussianizes). Rotation-invariant and CLT-immune — reads shells vs
    balls vs clumped mixtures, the axis sliced tests are blind to at high effective rank.
    var_ratio = Var(r²)/2d (Gaussian ≈ 1: shell < 1 < clumped/heavy); mean_ratio = mean(r²)/d
    (cross-fit consistency check). Finite-sample whitening bias is systematic — read BOTH against
    the battery's moment-matched gauss_null row (EP precedent), not against the analytic χ²_d."""
    X = np.asarray(X, dtype=np.float64)
    idx = np.random.default_rng(seed).permutation(len(X))
    half = len(X) // 2
    A, B = X[idx[:half]], X[idx[half:]]
    mu = A.mean(0)
    lam, V = np.linalg.eigh(np.cov(A, rowvar=False))
    lam = (1 - shrink) * np.clip(lam, 1e-12, None) + shrink * lam.mean()
    r2 = (((B - mu) @ (V / np.sqrt(lam))) ** 2).sum(1)
    d = X.shape[1]
    return {"var_ratio": float(r2.var() / (2 * d)), "mean_ratio": float(r2.mean() / d)}


def epps_pulley(X, n_slices=256, seed=0, standardize=True, knots=17, t_max=3.0):
    """Sliced Epps–Pulley distance to N(0, I). Lower = more isotropic-Gaussian.
    standardize=True z-scores dims first (isolates shape from wrong moments — raw features are
    not zero-mean/unit-var, which inflates the statistic on its own)."""
    x = torch.as_tensor(np.asarray(X, dtype=np.float32))
    if standardize:
        x = (x - x.mean(0)) / (x.std(0) + 1e-6)
    t = torch.linspace(0, t_max, knots)
    dt = t_max / (knots - 1)
    w = torch.full((knots,), 2 * dt)
    w[0] = w[-1] = dt
    phi = torch.exp(-t.square() / 2.0)
    w = w * phi
    g = torch.Generator().manual_seed(seed)
    A = torch.randn(x.size(-1), n_slices, generator=g)
    A = A / A.norm(p=2, dim=0)
    x_t = (x @ A).unsqueeze(-1) * t
    err = (x_t.cos().mean(-3) - phi).square() + x_t.sin().mean(-3).square()
    return float(((err @ w) * x.size(-2)).mean())
