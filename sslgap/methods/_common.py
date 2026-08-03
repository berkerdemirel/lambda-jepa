"""Shared recipe utilities: the house toy-rung optimizer schedule (D-012), arch helpers, and the
E12 moment floor (promoted from lejepa.py for the D5 cross-method arms, Berker 2026-07-12)."""
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR


class SpectralConditioner(nn.Module):
    """E12 lineage (D-026/D-027; renamed from SpectralConditioner per D-059 — Berker: "there is
    nothing floored in there. it is two sided spectral conditioner"): the TWO-SIDED
    Gaussian-moment conditioner. KL penalizes deviation from (0, I) in BOTH directions —
    variance above 1 is taxed exactly like variance below it (the E21 2048-d collapse lived
    on that cap side), plus the mean/cone term. The one-sided object is HingeFloor (S>=I),
    which IS a floor — and is vetoed as method identity (D-049).
    KL(N(mu_Q, Sigma_Q) || N(0, I_d')) / d' on a fresh random d'-dim orthonormal subspace Q per
    step (batch covariance is rank-deficient at full D; the subspace keeps the logdet barrier
    meaningful — fresh-slice coverage logic, unseeded like the official SIGReg). A pure function
    of batch mean+covariance: blind to clusters and all higher-order shape BY CONSTRUCTION;
    the logdet is the anti-degeneracy barrier. Estimator settings (d', eps) declared on the E12
    card and fixed. fp32 with autocast disabled (Cholesky). Measured dose lesson (E12 F-wave):
    conditioner with an interior optimum near lambda~.02 at h; destination weights (~.5) scrub
    class structure — see the card before reusing at other weights."""

    def __init__(self, d_slice=128, eps=1e-4, d_draw=None, shrink=None):
        # d_draw: RNG-stream-parity discipline (the hinge-arm precedent — "per-step RNG
        # streams stay aligned across floor variants"). A narrower slice (estimator co-design
        # for small-n inputs, D-051 view-mean payment) still DRAWS the family's canonical
        # frame shape and uses its first d_slice columns: Householder QR's leading k columns
        # depend only on the first k input columns, so the sub-frame IS the fresh k-frame
        # draw — same distribution, and the shared draw keeps h-floor slices matched across
        # arms at matched steps. Default d_draw=None (== d_slice) is byte-identical to the
        # historical behavior.
        # shrink="oas" (D-073, Berker's prescription): fresh-batch Oracle Approximating
        # Shrinkage of the slice scatter toward its own scalar mean mI before the KL —
        # the no-ring answer to n = d' rows/step (the ring's temporal rows carry moving-
        # model bias; OAS trades it for a known statistical one). rho from S.detach()
        # (estimator parameter, not a loss path); m stays LIVE and the target is mI,
        # never I (the desired answer must not be the estimator target — trace is
        # preserved exactly, so scale error stays fully supervised at every rho).
        # lambda_min(Sigma) >= rho*m + eps bounds the null-direction inverse gain; at
        # rho->1 only the isotropic scale force survives; at rho->0 the legacy
        # conditioner returns. None = byte-identical legacy path.
        super().__init__()
        self.d_slice, self.eps, self.d_draw = d_slice, eps, d_draw or d_slice
        self.shrink, self.rho_last = shrink, None

    def forward(self, x):
        with torch.autocast(x.device.type, enabled=False):
            x = x.reshape(-1, x.size(-1)).float()
            Q, _ = torch.linalg.qr(torch.randn(x.size(1), self.d_draw, device=x.device))
            p = x @ Q[:, :self.d_slice]
            mu = p.mean(0)
            pc = p - mu
            S = pc.T @ pc / (p.size(0) - 1)
            if self.shrink == "oas":
                d, nu = float(self.d_slice), float(p.size(0) - 1)
                Sd = S.detach()
                trS, trS2 = Sd.diagonal().sum(), (Sd * Sd).sum()
                num = (1 - 2 / d) * trS2 + trS.square()
                den = ((nu + 1 - 2 / d) * (trS2 - trS.square() / d)).clamp_min(1e-12)
                rho = (num / den).clamp(max=1.0)
                self.rho_last = float(rho)
                m = S.diagonal().sum() / d
                S = (1 - rho) * S + rho * m * torch.eye(self.d_slice, device=x.device)
            cov = S + self.eps * torch.eye(self.d_slice, device=x.device)
            logdet = 2 * torch.linalg.cholesky(cov).diagonal().log().sum()
            return 0.5 * (cov.diagonal().sum() + mu.square().sum() - self.d_slice - logdet) / self.d_slice


class HingeFloor(nn.Module):
    """E21 one-sided floor (D-049): vicreg's var-hinge generalized to fresh random slices —
    per slice direction q, relu(1 - std(x @ q)), plus SpectralConditioner's cone term. Population
    target set: Sigma >= I in the PSD order (every direction's variance >= 1) — a literal
    floor: one-sided (nothing above it is ever penalized; anisotropy and content scale are
    FREE), rotation-invariant in distribution (fresh slices — no exploitable gauge, unlike
    vicreg's per-original-dim hinge). Slice mixing makes rank collapse visible without a cov
    term: low-rank Sigma spreads direction-variances chi^2-like and the below-1 mass fires
    the hinge. Diagonal (not eigenvalue) hinge BY CONSTRUCTION: at n/d'=4 the MP eigenvalue
    spread [.26, 2.18] would give an eigen-hinge a ~10x estimator phantom; the diagonal
    hinge's is ~.01 (E21 estimator rider). Same slice mechanics and per-forward RNG draw as
    SpectralConditioner (randn -> QR, d'=128) — per-step RNG streams stay aligned across floor
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
