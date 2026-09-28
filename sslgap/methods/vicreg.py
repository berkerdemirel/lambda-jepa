"""VICReg as released, with the optional SACReg term at the backbone (h_reg=sacreg, h_tap=cls)."""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.parametrizations import spectral_norm

from sslgap.data import ViewsDataset, byol_pair
from sslgap.methods._common import SACReg, house_scheduler, trunk_arch
from sslgap.methods.base import SSLMethod
from sslgap.models.backbones import build_vit_trunk

def vicreg_expander(in_dim=384, hidden=2048, out_dim=2048, spec_norm=False):
    mlp = nn.Sequential(nn.Linear(in_dim, hidden), nn.BatchNorm1d(hidden), nn.ReLU(inplace=True),
                        nn.Linear(hidden, hidden), nn.BatchNorm1d(hidden), nn.ReLU(inplace=True),
                        nn.Linear(hidden, out_dim))
    if spec_norm:
        for m in mlp.modules():
            if isinstance(m, nn.Linear):
                spectral_norm(m)
    return mlp

def variance_term(z, gamma=1.0, eps=1e-4):
    std = (z.var(0) + eps).sqrt()
    return F.relu(gamma - std).mean()

def covariance_term(z):
    zc = z - z.mean(0)
    C = (zc.T @ zc) / (z.shape[0] - 1)
    d = C.shape[0]
    return (C.pow(2).sum() - C.diagonal().pow(2).sum()) / d

class VICReg(SSLMethod):
    PULL_W = {"inv": "w_inv", "var": "w_var", "cov": "w_cov", "moment_kl": "w_floor",
              "h_moment_kl": "h_lamb", "h_var": "h_lamb", "h_cov": "h_lamb",
              "h_inv": "h_inv"}
    name = "vicreg"

    def build_modules(self):
        trunk = build_vit_trunk(self.frame.model_name, self.frame.img_size,
                                drop_path_rate=self.cfg.drop_path)
        proj = vicreg_expander(384, self.cfg.expander_hidden, self.cfg.expander_dim,
                               spec_norm=self.cfg.get("spec_norm", False))
        if self.cfg.get("h_reg") == "sacreg":
            self.floor = SACReg()
        if self.cfg.get("anticollapse", "varcov") == "floor":
            self.floor_z = SACReg()
        mods = {"backbone": trunk, "projector": proj}
        if self.cfg.get("emb_dim", 0):
            mods["embed"] = nn.Linear(384, self.cfg.emb_dim)
        return nn.ModuleDict(mods)

    def arch(self):
        proj_kwargs = {"in_dim": 384, "hidden": self.cfg.expander_hidden,
                       "out_dim": self.cfg.expander_dim}
        if self.cfg.get("spec_norm", False):
            proj_kwargs["spec_norm"] = True
        a = {"backbone": trunk_arch(self.frame, self.cfg.drop_path),
             "projector": {"class": "sslgap.methods.vicreg.vicreg_expander",
                           "kwargs": proj_kwargs}}
        if self.cfg.get("emb_dim", 0):
            a["embed"] = {"class": "torch.nn.Linear",
                          "kwargs": {"in_features": 384, "out_features": self.cfg.emb_dim}}
        return a

    def build_train_dataset(self):
        return ViewsDataset(self.frame.dataset, "train", V=2, img_size=self.frame.img_size,
                            data_root=self.frame.data_root,
                            transforms=byol_pair(self.frame.img_size))

    def param_groups(self, modules):
        if self.cfg.get("emb_wd0", False):
            print("[vicreg] emb_wd0 active: embed param group weight_decay=0", flush=True)
            return [{"params": [p for k, m in modules.items() if k != "embed"
                                for p in m.parameters()],
                     "lr": self.cfg.lr, "weight_decay": self.cfg.wd},
                    {"params": list(modules["embed"].parameters()),
                     "lr": self.cfg.lr, "weight_decay": 0.0}]
        return [{"params": [p for m in modules.values() for p in m.parameters()],
                 "lr": self.cfg.lr, "weight_decay": self.cfg.wd}]

    def build_scheduler(self, optimizer, steps_per_epoch, total_steps):
        return house_scheduler(optimizer, steps_per_epoch, total_steps,
                               self.cfg.warmup_ep, self.cfg.eta_min)

    def training_step(self, modules, views, device, y=None):
        N, V = views.shape[:2]
        tok = modules["backbone"].forward_features(views.flatten(0, 1))
        if self.cfg.get("emb_calib", False) and not getattr(self, "_emb_calibrated", False):
            with torch.no_grad():
                e0 = modules["embed"](tok[:, 0]).float()
                mu, sd = e0.mean(0), e0.std(0).clamp_min(1e-6)
                modules["embed"].weight.div_(sd.unsqueeze(1))
                modules["embed"].bias.copy_((modules["embed"].bias - mu) / sd)
            self._emb_calibrated = True
        z_in = modules["embed"](tok[:, 0]) if "embed" in modules else tok[:, 0]
        z = modules["projector"](z_in).reshape(N, V, -1)
        za, zb = z[:, 0], z[:, 1]
        inv = F.mse_loss(za, zb)
        if self.cfg.get("anticollapse", "varcov") == "floor":
            reg = self.floor_z(z.reshape(N * V, -1))
            loss = self.cfg.w_inv * inv + self.cfg.w_floor * reg
            terms = {"inv": inv, "moment_kl": reg}
        else:
            var = variance_term(za) + variance_term(zb)
            cov = covariance_term(za) + covariance_term(zb)
            loss = self.cfg.w_inv * inv + self.cfg.w_var * var + self.cfg.w_cov * cov
            terms = {"inv": inv, "var": var, "cov": cov}
        gap = tok[:, 1:].mean(1)
        h_feat = {"cls": tok[:, 0], "gap": gap,
                  "emb": z_in}[self.cfg.get("h_tap", "gap")]
        if self.cfg.get("h_reg") == "sacreg":
            h_loss = self.floor(h_feat)
            loss = loss + self.cfg.h_lamb * h_loss
            terms["h_moment_kl"] = h_loss
        elif self.cfg.get("h_reg") == "varcov":
            hv = h_feat.reshape(N, V, -1)
            h_var = variance_term(hv[:, 0]) + variance_term(hv[:, 1])
            h_cov = covariance_term(hv[:, 0]) + covariance_term(hv[:, 1])
            loss = loss + self.cfg.h_lamb * (self.cfg.w_var * h_var + self.cfg.w_cov * h_cov)
            terms["h_var"], terms["h_cov"] = h_var, h_cov
        h_inv = self.cfg.get("h_inv", 0.0)
        if h_inv:
            g_v = h_feat.reshape(N, V, -1).transpose(0, 1)
            hi_loss = (g_v.mean(0) - g_v).square().mean()
            loss = loss + h_inv * hi_loss
            terms["h_inv"] = hi_loss
        probe_feats = tok[:, 0].detach()
        return ({"loss": loss, **terms}, probe_feats, V)

    def extras(self):
        return {"emb_calibrated": getattr(self, "_emb_calibrated", False)}

    def load_extras(self, extras):
        self._emb_calibrated = extras.get("emb_calibrated", False)

    @torch.inference_mode()
    def eval_features(self, modules, x, device):
        return modules["backbone"].forward_features(x)[:, 0]

    def probe_dim(self):
        return 384
