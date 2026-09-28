"""Shared pieces: SACReg, the warmup + cosine schedule, trunk construction, the EMA momentum schedule."""
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR

class SACReg(nn.Module):
    """KL(N(mu, Sigma) || N(0, I)) / d' of the batch mean and covariance on a fresh random d'-dimensional orthonormal slice per step (the batch covariance is rank-deficient at full width); eps stabilizes the log-determinant, shrink="oas" replaces the slice covariance by its OAS-shrunk estimate; fp32."""

    def __init__(self, d_slice=128, eps=1e-4, d_draw=None, shrink=None):
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

def house_scheduler(optimizer, steps_per_epoch, total_steps, warmup_ep, eta_min):
    warmup = steps_per_epoch * warmup_ep
    s1 = LinearLR(optimizer, start_factor=0.01, total_iters=warmup)
    s2 = CosineAnnealingLR(optimizer, T_max=total_steps - warmup, eta_min=eta_min)
    return SequentialLR(optimizer, schedulers=[s1, s2], milestones=[warmup])

def trunk_arch(frame, drop_path, dynamic_img_size=False):
    return {"class": "sslgap.models.backbones.build_vit_trunk",
            "kwargs": {"model_name": frame.model_name, "img_size": frame.img_size,
                       "dynamic_img_size": dynamic_img_size, "drop_path_rate": drop_path}}

def ema_momentum(step, total_steps, base, end=1.0):
    import math
    return end - (end - base) * (math.cos(math.pi * step / total_steps) + 1) / 2
