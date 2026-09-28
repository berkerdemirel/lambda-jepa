"""BYOL as released, with the optional SACReg term at the backbone (h_reg=sacreg)."""
import copy

import torch
import torch.nn as nn
import torch.nn.functional as F

from sslgap.data import ViewsDataset, byol_pair
from sslgap.methods._common import SACReg, ema_momentum, house_scheduler, trunk_arch
from sslgap.methods.base import SSLMethod
from sslgap.models.backbones import build_vit_trunk
from sslgap.models.vitops import ema_update

def byol_mlp(in_dim, hidden=4096, out_dim=256):
    return nn.Sequential(nn.Linear(in_dim, hidden), nn.BatchNorm1d(hidden), nn.ReLU(inplace=True),
                         nn.Linear(hidden, out_dim))

def byol_h_predictor(dim=384):
    return nn.Linear(dim, dim)

class BYOL(SSLMethod):
    PULL_W = {"regress": 1.0, "h_align": "h_align", "h_moment_kl": "h_lamb"}
    name = "byol"

    def build_modules(self):
        trunk = build_vit_trunk(self.frame.model_name, self.frame.img_size,
                                drop_path_rate=self.cfg.drop_path)
        proj = byol_mlp(384, self.cfg.proj_hidden, self.cfg.out_dim)
        pred = byol_mlp(self.cfg.out_dim, self.cfg.pred_hidden, self.cfg.out_dim)
        t_trunk = copy.deepcopy(trunk).requires_grad_(False)
        t_proj = copy.deepcopy(proj).requires_grad_(False)
        mods = {"backbone": trunk, "projector": proj, "predictor": pred,
                "teacher_backbone": t_trunk, "teacher_projector": t_proj}
        if self.cfg.get("h_align", 0.0):
            mods["h_predictor"] = byol_h_predictor(384)
        if self.cfg.get("h_reg") == "sacreg":
            self.floor = SACReg()
        return nn.ModuleDict(mods)

    def arch(self):
        mlp = lambda i, h: {"class": "sslgap.methods.byol.byol_mlp",
                            "kwargs": {"in_dim": i, "hidden": h, "out_dim": self.cfg.out_dim}}
        a = {"backbone": trunk_arch(self.frame, self.cfg.drop_path),
             "projector": mlp(384, self.cfg.proj_hidden),
             "predictor": mlp(self.cfg.out_dim, self.cfg.pred_hidden),
             "teacher_backbone": trunk_arch(self.frame, self.cfg.drop_path),
             "teacher_projector": mlp(384, self.cfg.proj_hidden)}
        if self.cfg.get("h_align", 0.0):
            a["h_predictor"] = {"class": "sslgap.methods.byol.byol_h_predictor", "kwargs": {"dim": 384}}
        return a

    def build_train_dataset(self):
        return ViewsDataset(self.frame.dataset, "train", V=2, img_size=self.frame.img_size,
                            data_root=self.frame.data_root,
                            transforms=byol_pair(self.frame.img_size))

    def param_groups(self, modules):
        student = ("backbone", "projector", "predictor")
        params = [p for k in student for p in modules[k].parameters()]
        if "h_predictor" in modules:
            params += list(modules["h_predictor"].parameters())
        return [{"params": params, "lr": self.cfg.lr, "weight_decay": self.cfg.wd}]

    def build_scheduler(self, optimizer, steps_per_epoch, total_steps):
        return house_scheduler(optimizer, steps_per_epoch, total_steps,
                               self.cfg.warmup_ep, self.cfg.eta_min)

    @staticmethod
    def _regress(q, t):
        return 2 - 2 * (F.normalize(q, dim=1) * F.normalize(t, dim=1)).sum(1).mean()

    def training_step(self, modules, views, device, y=None):
        N, V = views.shape[:2]
        tok = modules["backbone"].forward_features(views.flatten(0, 1))
        q = modules["predictor"](modules["projector"](tok[:, 0])).reshape(N, V, -1)
        with torch.no_grad():
            ht = modules["teacher_backbone"].forward_features(views.flatten(0, 1))[:, 0]
            zt = modules["teacher_projector"](ht).reshape(N, V, -1)
            self._t_std = zt.reshape(-1, zt.shape[-1]).float().std(0).mean().item()
        regress = self._regress(q[:, 0], zt[:, 1]) + self._regress(q[:, 1], zt[:, 0])
        loss, terms = regress, {"regress": regress}
        h_align = self.cfg.get("h_align", 0.0)
        if h_align:
            ph = modules["h_predictor"](tok[:, 0]).reshape(N, V, -1)
            th = ht.reshape(N, V, -1)
            ha = self._regress(ph[:, 0], th[:, 1]) + self._regress(ph[:, 1], th[:, 0])
            loss = loss + h_align * ha
            terms["h_align"] = ha
        if self.cfg.get("h_reg") == "sacreg":
            h_loss = self.floor(tok[:, 0])
            loss = loss + self.cfg.h_lamb * h_loss
            terms["h_moment_kl"] = h_loss
        probe_feats = tok[:, 0].detach()
        return ({"loss": loss, **terms}, probe_feats, V)

    def post_step(self, modules, step, total_steps):
        m = ema_momentum(step, total_steps, self.cfg.ema_base)
        ema_update(modules["teacher_backbone"], modules["backbone"], m)
        ema_update(modules["teacher_projector"], modules["projector"], m)
        return {"ema_m": m, "teacher_proj_std": self._t_std}

    @torch.inference_mode()
    def eval_features(self, modules, x, device):
        return modules["backbone"].forward_features(x)[:, 0]

    def probe_dim(self):
        return 384
