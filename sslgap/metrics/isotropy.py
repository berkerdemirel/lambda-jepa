"""Isotropy / Gaussianity. Epps–Pulley sliced statistic ported from ssl_explore/sslx/sigreg.py
(itself the lejepa-minimal SIGReg with seeded slices). HOUSE RULE (CLAUDE.md): the sliced statistic
is foolable — the battery always pairs it with the top-eigenvector kurtosis from spectra.py, and
isotropy claims lead with the worst-direction numbers."""
import numpy as np
import torch


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
