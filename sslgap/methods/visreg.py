"""VISReg (arXiv 2606.02572) — house port for the IN-100 treatment zoo (Berker 2026-08-25:
"it is a combination of lejepa and vicreg i agree. we should not go out of the house with
augmentation etc. we follow in100 in house way. just updating the loss as visreg").

Anatomy, frame, aug = the house lejepa lane (trunk -> 512-d embed -> projector, V=4 house
aug family, house lr 1e-3/warmup 10/eta_min 1e-5 hygiene) registered in the CANONICAL roles
(backbone/embed/projector per SSLMethod.build_modules) — NOT lejepa's "encoder" monolith:
that monolith exists only for lejepa's seed-faithful port (construction-order parity with
the official init stream), which visreg has no donor to match, and the canonical split is
what the trainer instruments (share/Ω logger) and the ckpt assembly consume. 2026-08-25
incident: the first build reused the monolith and crash-looped every launch at the share
logger's `modules["backbone"]` lookup (e-visreg 63677740-42 + pilot 63677743). ONLY the
loss follows the donor: loss = w_reg * VISReg(z) + w_inv * inv, the donor's lamb=.9 convex
mix carried as EXPLICIT weights (w_reg=.9, w_inv=.1; numerically identical, PULL_W-clean —
no derived (1-lamb) weight), inv = view-to-mean MSE at proj.out (their global-mean anchor;
the house V=4 views are all global).

PORT_NOTES (donor ~/visreg_repro @ the D-102-arc pin, visreg/losses/visreg.py + train.py):
- reg ported verbatim: center = mean(mu^2) over dims + scale = (per-dim std - 1)^2 mean
  + shape = sorted projections over 256 unit slices vs erfinv Gaussian quantile targets
  (cached per batch size); std DETACHED in the shape normalizer, exactly as the donor.
- their trainer composes reg*lamb + inv*(1-lamb) with lamb=.9 (configs/default.yaml) and
  anchors inv to the mean over GLOBAL views (train.py:230); at house V=4 all-global views
  the all-view mean IS that anchor.
- deviations (declared): house aug/frame/anatomy instead of their torchvision stack +
  ViT-B (the "in house way" ruling); proj_dim = 128 (the house z-slice width) instead of
  their embed_dim//3 rule, which gives the nonstandard 170 at our 512-d embedding; house
  lr/warmup/eta_min hygiene instead of their 9e-4 @ bs 16x8.
- h_reg="moment" hook mirrors the zoo floor at the declared h (the 512-d embedding,
  lejepa-family D-036 convention; conditioner input pools the views, n = bs*V), weight
  h_lamb — the E20-style treated arm.
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from sslgap.data import ViewsDataset
from sslgap.methods._common import SpectralConditioner, house_scheduler, trunk_arch
from sslgap.methods.base import SSLMethod
from sslgap.methods.lejepa import lejepa_projector
from sslgap.models.backbones import build_vit_trunk


class VISRegLoss(nn.Module):
    """Donor visreg/losses/visreg.py verbatim (input [V, B, D])."""

    def __init__(self, num_projections=256, scale_weight=1.0, shape_weight=1.0,
                 center_weight=1.0):
        super().__init__()
        self.K = num_projections
        self._cached_B = -1
        self._cached_target = None
        self.scale_weight = scale_weight
        self.shape_weight = shape_weight
        self.center_weight = center_weight

    def _get_target(self, B, device):
        if self._cached_B != B:
            q = torch.linspace(1, B, B, device=device, dtype=torch.float32) / (B + 1)
            self._cached_target = torch.erfinv(2 * q - 1).mul_(math.sqrt(2))
            self._cached_B = B
        return self._cached_target.to(device=device)

    def forward(self, z):
        _, B, D = z.shape
        mu = z.mean(dim=1, keepdim=True)
        center_loss = mu.pow(2).mean()
        z_centered = z - mu
        std = z_centered.norm(dim=1).div(math.sqrt(B)).clamp_min(1e-6)
        scale_loss = (std - 1.0).pow(2).mean()
        z_norm = z_centered / std.detach().unsqueeze(1)
        W = F.normalize(torch.randn(D, self.K, device=z.device, dtype=z.dtype), dim=0)
        p_sorted = (z_norm @ W).sort(dim=1).values
        target = self._get_target(B, z.device).view(1, B, 1)
        shape_loss = (p_sorted - target).pow(2).mean()
        return (self.scale_weight * scale_loss + self.shape_weight * shape_loss
                + self.center_weight * center_loss)


class VISReg(SSLMethod):
    name = "visreg"
    PULL_W = {"visreg": "w_reg", "inv": "w_inv", "h_moment_kl": "h_lamb"}

    def __init__(self, cfg, frame):
        super().__init__(cfg, frame)
        self.reg = VISRegLoss(num_projections=cfg.get("num_projections", 256))
        if self.cfg.get("h_reg") == "moment":     # zoo floor hook (no RNG at construction)
            self.floor = SpectralConditioner()

    def build_modules(self):
        trunk = build_vit_trunk(self.frame.model_name, self.frame.img_size,
                                drop_path_rate=self.cfg.drop_path)
        self._h_dim = trunk.num_features
        embed = nn.Linear(self._h_dim, self.cfg.emb_dim)
        proj = lejepa_projector(self.cfg.proj_dim, emb_dim=self.cfg.emb_dim,
                                depth=self.cfg.get("proj_depth", 3))
        return nn.ModuleDict({"backbone": trunk, "embed": embed, "projector": proj})

    def build_train_dataset(self):
        return ViewsDataset(self.frame.dataset, "train", V=self.cfg.get("V", 4),
                            img_size=self.frame.img_size,
                            data_root=self.frame.data_root)

    def param_groups(self, modules):
        return [{"params": [p for m in modules.values() for p in m.parameters()],
                 "lr": self.cfg.lr, "weight_decay": self.cfg.wd}]

    def build_scheduler(self, optimizer, steps_per_epoch, total_steps):
        return house_scheduler(optimizer, steps_per_epoch, total_steps,
                               self.cfg.warmup_ep, self.cfg.eta_min)

    def training_step(self, modules, views, device, y=None):
        N, V = views.shape[:2]
        cls = modules["backbone"].forward_features(views.flatten(0, 1))[:, 0]
        emb = modules["embed"](cls)
        z = modules["projector"](emb).reshape(N, V, -1)
        z_vbd = z.transpose(0, 1)                              # [V, B, D], donor convention
        inv = (z_vbd - z_vbd.mean(dim=0, keepdim=True)).square().mean()
        reg = self.reg(z_vbd)
        loss = self.cfg.w_reg * reg + self.cfg.w_inv * inv
        terms = {"visreg": reg, "inv": inv}
        if self.cfg.get("h_reg") == "moment":                  # the E20-style treated arm
            emb_v = emb.reshape(N, V, -1).transpose(0, 1)
            h_loss = self.floor(emb_v)
            loss = loss + self.cfg.h_lamb * h_loss
            terms["h_moment_kl"] = h_loss
        probe_feats = emb.detach()
        return {"loss": loss, **terms}, probe_feats, V

    def arch(self):
        return {"backbone": trunk_arch(self.frame, self.cfg.drop_path),
                "embed": {"class": "torch.nn.Linear",
                          "kwargs": {"in_features": self._h_dim,
                                     "out_features": self.cfg.emb_dim}},
                "projector": {"class": "sslgap.methods.lejepa.lejepa_projector",
                              "kwargs": {"proj_dim": self.cfg.proj_dim,
                                         "emb_dim": self.cfg.emb_dim,
                                         "depth": self.cfg.get("proj_depth", 3)}}}

    @torch.inference_mode()
    def eval_features(self, modules, x, device):
        return modules["embed"](modules["backbone"].forward_features(x)[:, 0])

    def probe_dim(self):
        return self.cfg.emb_dim
