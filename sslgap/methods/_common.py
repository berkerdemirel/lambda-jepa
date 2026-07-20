"""Shared recipe utilities: the house toy-rung optimizer schedule (D-012), arch helpers, and the
E12 moment floor (promoted from lejepa.py for the D5 cross-method arms, Berker 2026-07-12)."""
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR


class MomentFloor(nn.Module):
    """E12 (D-026/D-027): the Gaussian-moment calibration floor at the probed space.
    KL(N(mu_Q, Sigma_Q) || N(0, I_d')) / d' on a fresh random d'-dim orthonormal subspace Q per
    step (batch covariance is rank-deficient at full D; the subspace keeps the logdet barrier
    meaningful — fresh-slice coverage logic, unseeded like the official SIGReg). A pure function
    of batch mean+covariance: blind to clusters and all higher-order shape BY CONSTRUCTION;
    the logdet is the anti-degeneracy barrier. Estimator settings (d', eps) declared on the E12
    card and fixed. fp32 with autocast disabled (Cholesky). Measured dose lesson (E12 F-wave):
    conditioner with an interior optimum near lambda~.02 at h; destination weights (~.5) scrub
    class structure — see the card before reusing at other weights."""

    def __init__(self, d_slice=128, eps=1e-4, d_draw=None):
        # d_draw: RNG-stream-parity discipline (the hinge-arm precedent — "per-step RNG
        # streams stay aligned across floor variants"). A narrower slice (estimator co-design
        # for small-n inputs, D-051 view-mean payment) still DRAWS the family's canonical
        # frame shape and uses its first d_slice columns: Householder QR's leading k columns
        # depend only on the first k input columns, so the sub-frame IS the fresh k-frame
        # draw — same distribution, and the shared draw keeps h-floor slices matched across
        # arms at matched steps. Default d_draw=None (== d_slice) is byte-identical to the
        # historical behavior.
        super().__init__()
        self.d_slice, self.eps, self.d_draw = d_slice, eps, d_draw or d_slice

    def forward(self, x):
        with torch.autocast(x.device.type, enabled=False):
            x = x.reshape(-1, x.size(-1)).float()
            Q, _ = torch.linalg.qr(torch.randn(x.size(1), self.d_draw, device=x.device))
            p = x @ Q[:, :self.d_slice]
            mu = p.mean(0)
            pc = p - mu
            cov = pc.T @ pc / (p.size(0) - 1) + self.eps * torch.eye(self.d_slice, device=x.device)
            logdet = 2 * torch.linalg.cholesky(cov).diagonal().log().sum()
            return 0.5 * (cov.diagonal().sum() + mu.square().sum() - self.d_slice - logdet) / self.d_slice


class HingeFloor(nn.Module):
    """E21 one-sided floor (D-049): vicreg's var-hinge generalized to fresh random slices —
    per slice direction q, relu(1 - std(x @ q)), plus MomentFloor's cone term. Population
    target set: Sigma >= I in the PSD order (every direction's variance >= 1) — a literal
    floor: one-sided (nothing above it is ever penalized; anisotropy and content scale are
    FREE), rotation-invariant in distribution (fresh slices — no exploitable gauge, unlike
    vicreg's per-original-dim hinge). Slice mixing makes rank collapse visible without a cov
    term: low-rank Sigma spreads direction-variances chi^2-like and the below-1 mass fires
    the hinge. Diagonal (not eigenvalue) hinge BY CONSTRUCTION: at n/d'=4 the MP eigenvalue
    spread [.26, 2.18] would give an eigen-hinge a ~10x estimator phantom; the diagonal
    hinge's is ~.01 (E21 estimator rider). Same slice mechanics and per-forward RNG draw as
    MomentFloor (randn -> QR, d'=128) — per-step RNG streams stay aligned across floor
    variants. Bounded gradients (no logdet)."""

    def __init__(self, d_slice=128, eps=1e-4):
        super().__init__()
        self.d_slice, self.eps = d_slice, eps

    def forward(self, x):
        with torch.autocast(x.device.type, enabled=False):
            x = x.reshape(-1, x.size(-1)).float()
            Q, _ = torch.linalg.qr(torch.randn(x.size(1), self.d_slice, device=x.device))
            p = x @ Q
            mu = p.mean(0)
            std = ((p - mu).square().sum(0) / (p.size(0) - 1) + self.eps).sqrt()
            return torch.relu(1 - std).mean() + 0.5 * mu.square().sum() / self.d_slice


def house_scheduler(optimizer, steps_per_epoch, total_steps, warmup_ep, eta_min):
    """Toy-rung schedule: linear warmup (start 0.01x) then cosine to eta_min (house rule:
    eta_min <= lr/20)."""
    warmup = steps_per_epoch * warmup_ep
    s1 = LinearLR(optimizer, start_factor=0.01, total_iters=warmup)
    s2 = CosineAnnealingLR(optimizer, T_max=total_steps - warmup, eta_min=eta_min)
    return SequentialLR(optimizer, schedulers=[s1, s2], milestones=[warmup])


def trunk_arch(frame, drop_path):
    return {"class": "sslgap.models.backbones.build_vit_trunk",
            "kwargs": {"model_name": frame.model_name, "img_size": frame.img_size,
                       "dynamic_img_size": False, "drop_path_rate": drop_path}}


def ema_momentum(step, total_steps, base, end=1.0):
    """Cosine EMA momentum schedule base -> end (BYOL/DINO convention)."""
    import math
    return end - (end - base) * (math.cos(math.pi * step / total_steps) + 1) / 2
