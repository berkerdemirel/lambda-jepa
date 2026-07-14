"""VICReg (Bardes et al. 2022) — toy-rung instance per D-012: canonical loss (var/inv/cov with
paper weights 25/25/1, computed at the expander output), canonical expander shape
(Linear-BN-ReLU x2 + Linear), BYOL-style asymmetric view pair (paper follows BYOL augs), house
AdamW schedule (LARS at M1.5). Donor cross-check: solo-learn methods/vicreg.py + paper App. C.
Expander 2048-d for the 384-d trunk (paper ratio ~4x; solo-learn RN18-512 used 2048)."""
import torch
import torch.nn as nn
import torch.nn.functional as F

from sslgap.data import ViewsDataset, byol_pair
from sslgap.methods._common import MomentFloor, house_scheduler, trunk_arch
from sslgap.methods.base import SSLMethod
from sslgap.models.backbones import build_vit_trunk


def vicreg_expander(in_dim=384, hidden=2048, out_dim=2048):
    return nn.Sequential(nn.Linear(in_dim, hidden), nn.BatchNorm1d(hidden), nn.ReLU(inplace=True),
                         nn.Linear(hidden, hidden), nn.BatchNorm1d(hidden), nn.ReLU(inplace=True),
                         nn.Linear(hidden, out_dim))


def variance_term(z, gamma=1.0, eps=1e-4):
    std = (z.var(0) + eps).sqrt()
    return F.relu(gamma - std).mean()


def covariance_term(z):
    zc = z - z.mean(0)
    C = (zc.T @ zc) / (z.shape[0] - 1)
    d = C.shape[0]
    return (C.pow(2).sum() - C.diagonal().pow(2).sum()) / d


class VICReg(SSLMethod):
    name = "vicreg"

    def build_modules(self):
        trunk = build_vit_trunk(self.frame.model_name, self.frame.img_size,
                                drop_path_rate=self.cfg.drop_path)
        proj = vicreg_expander(384, self.cfg.expander_hidden, self.cfg.expander_dim)
        if self.cfg.get("h_reg") == "moment":     # E12 cross-method arm (no RNG at construction)
            self.floor = MomentFloor()
        return nn.ModuleDict({"backbone": trunk, "projector": proj})

    def arch(self):
        return {"backbone": trunk_arch(self.frame, self.cfg.drop_path),
                "projector": {"class": "sslgap.methods.vicreg.vicreg_expander",
                              "kwargs": {"in_dim": 384, "hidden": self.cfg.expander_hidden,
                                         "out_dim": self.cfg.expander_dim}}}

    def build_train_dataset(self):
        return ViewsDataset(self.frame.dataset, "train", V=2, img_size=self.frame.img_size,
                            data_root=self.frame.data_root,
                            transforms=byol_pair(self.frame.img_size))

    def param_groups(self, modules):
        return [{"params": [p for m in modules.values() for p in m.parameters()],
                 "lr": self.cfg.lr, "weight_decay": self.cfg.wd}]

    def build_scheduler(self, optimizer, steps_per_epoch, total_steps):
        return house_scheduler(optimizer, steps_per_epoch, total_steps,
                               self.cfg.warmup_ep, self.cfg.eta_min)

    def training_step(self, modules, views, device, y=None):
        N, V = views.shape[:2]
        tok = modules["backbone"].forward_features(views.flatten(0, 1))
        z = modules["projector"](tok[:, 0]).reshape(N, V, -1)              # loss input: CLS (trained)
        za, zb = z[:, 0], z[:, 1]
        inv = F.mse_loss(za, zb)
        var = variance_term(za) + variance_term(zb)
        cov = covariance_term(za) + covariance_term(zb)
        loss = self.cfg.w_inv * inv + self.cfg.w_var * var + self.cfg.w_cov * cov
        terms = {"inv": inv, "var": var, "cov": cov}
        # E12 cross-method arm: additive moment floor at h — h_tap picks the placement:
        # "gap" (default; the audited h, off the head's input path) or "cls" (the head's actual
        # input — e12gvcls, Berker 2026-07-14: "both the inv and kl on cls if thats the feature
        # space"). vicreg's own var/cov terms stay at z untouched.
        gap = tok[:, 1:].mean(1)
        h_feat = tok[:, 0] if self.cfg.get("h_tap", "gap") == "cls" else gap
        if self.cfg.get("h_reg") == "moment":
            h_loss = self.floor(h_feat)
            loss = loss + self.cfg.h_lamb * h_loss
            terms["h_moment_kl"] = h_loss
        # H-wave (D-035): tiny ADDITIVE view-invariance pull at the same h tap — functional form
        # and 10%-pull rule as lejepa's h_inv. gvi = gap placement (off-path), gvcls = cls.
        h_inv = self.cfg.get("h_inv", 0.0)
        if h_inv:
            g_v = h_feat.reshape(N, V, -1).transpose(0, 1)
            hi_loss = (g_v.mean(0) - g_v).square().mean()
            loss = loss + h_inv * hi_loss
            terms["h_inv"] = hi_loss
        probe_feats = gap.detach()                   # monitor stays = audited h (trunk-GAP, F1)
        return ({"loss": loss, **terms}, probe_feats, V)

    @torch.inference_mode()
    def eval_features(self, modules, x, device):
        return modules["backbone"].forward_features(x)[:, 1:].mean(1)     # audited h (GAP)

    def probe_dim(self):
        return 384
