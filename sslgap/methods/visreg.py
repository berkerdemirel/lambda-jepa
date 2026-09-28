"""VISReg as released, with the optional SACReg term at the 512-d embedding (h_reg=sacreg)."""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from sslgap.data import ViewsDataset
from sslgap.methods._common import SACReg, house_scheduler, trunk_arch
from sslgap.methods.base import SSLMethod
from sslgap.methods.lejepa import lejepa_projector
from sslgap.models.backbones import build_vit_trunk

class VISRegLoss(nn.Module):

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
    PULL_W = {"visreg": "w_reg", "inv": "w_inv", "h_moment_kl": "h_lamb",
              "h_visreg": "h_lamb"}

    def __init__(self, cfg, frame):
        super().__init__(cfg, frame)
        self.reg = VISRegLoss(num_projections=cfg.get("num_projections", 256))
        if self.cfg.get("h_reg") == "sacreg":
            self.floor = SACReg()
        elif self.cfg.get("h_reg") == "visreg":
            self.h_reg = VISRegLoss(num_projections=cfg.get("num_projections", 256))

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
        z_vbd = z.transpose(0, 1)
        inv = (z_vbd - z_vbd.mean(dim=0, keepdim=True)).square().mean()
        reg = self.reg(z_vbd)
        loss = self.cfg.w_reg * reg + self.cfg.w_inv * inv
        terms = {"visreg": reg, "inv": inv}
        if self.cfg.get("h_reg") == "sacreg":
            emb_v = emb.reshape(N, V, -1).transpose(0, 1)
            h_loss = self.floor(emb_v)
            loss = loss + self.cfg.h_lamb * h_loss
            terms["h_moment_kl"] = h_loss
        elif self.cfg.get("h_reg") == "visreg":
            emb_v = emb.reshape(N, V, -1).transpose(0, 1)
            h_loss = self.h_reg(emb_v)
            loss = loss + self.cfg.h_lamb * h_loss
            terms["h_visreg"] = h_loss
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
