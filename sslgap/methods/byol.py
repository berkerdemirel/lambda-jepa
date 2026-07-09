"""BYOL (Grill et al. 2020) — toy-rung instance per D-012: canonical machinery (projector +
predictor + EMA teacher + stop-grad, normalized-MSE symmetrized), canonical asymmetric view pair,
house AdamW (LARS at M1.5). Donor cross-check: solo-learn methods/byol.py (proj hidden 4096, pred hidden 8192, out 256 — their IN-100 config). EMA base 0.99 (NOT the paper's 0.996): at 37 steps/epoch on Imagenette the 0.996 teacher
is near-frozen — the ssl_explore DINO-control incident, avoided by design here (documented).
Collapse canary (paper Tab.5: no-predictor -> 0.3%): teacher-projection std is monitored per step."""
import copy

import torch
import torch.nn as nn
import torch.nn.functional as F

from sslgap.data import ViewsDataset, byol_pair
from sslgap.methods._common import ema_momentum, house_scheduler, trunk_arch
from sslgap.methods.base import SSLMethod
from sslgap.models.backbones import build_vit_trunk
from sslgap.models.vitops import ema_update


def byol_mlp(in_dim, hidden=4096, out_dim=256):
    return nn.Sequential(nn.Linear(in_dim, hidden), nn.BatchNorm1d(hidden), nn.ReLU(inplace=True),
                         nn.Linear(hidden, out_dim))


class BYOL(SSLMethod):
    name = "byol"

    def build_modules(self):
        trunk = build_vit_trunk(self.frame.model_name, self.frame.img_size,
                                drop_path_rate=self.cfg.drop_path)
        proj = byol_mlp(384, self.cfg.proj_hidden, self.cfg.out_dim)
        pred = byol_mlp(self.cfg.out_dim, self.cfg.pred_hidden, self.cfg.out_dim)
        t_trunk = copy.deepcopy(trunk).requires_grad_(False)
        t_proj = copy.deepcopy(proj).requires_grad_(False)
        return nn.ModuleDict({"backbone": trunk, "projector": proj, "predictor": pred,
                              "teacher_backbone": t_trunk, "teacher_projector": t_proj})

    def arch(self):
        mlp = lambda i, h: {"class": "sslgap.methods.byol.byol_mlp",
                            "kwargs": {"in_dim": i, "hidden": h, "out_dim": self.cfg.out_dim}}
        return {"backbone": trunk_arch(self.frame, self.cfg.drop_path),
                "projector": mlp(384, self.cfg.proj_hidden),
                "predictor": mlp(self.cfg.out_dim, self.cfg.pred_hidden),
                "teacher_backbone": trunk_arch(self.frame, self.cfg.drop_path),
                "teacher_projector": mlp(384, self.cfg.proj_hidden)}

    def build_train_dataset(self):
        return ViewsDataset(self.frame.dataset, "train", V=2, img_size=self.frame.img_size,
                            data_root=self.frame.data_root,
                            transforms=byol_pair(self.frame.img_size))

    def param_groups(self, modules):
        student = ("backbone", "projector", "predictor")
        return [{"params": [p for k in student for p in modules[k].parameters()],
                 "lr": self.cfg.lr, "weight_decay": self.cfg.wd}]

    def build_scheduler(self, optimizer, steps_per_epoch, total_steps):
        return house_scheduler(optimizer, steps_per_epoch, total_steps,
                               self.cfg.warmup_ep, self.cfg.eta_min)

    @staticmethod
    def _regress(q, t):
        return 2 - 2 * (F.normalize(q, dim=1) * F.normalize(t, dim=1)).sum(1).mean()

    def training_step(self, modules, views, device, y=None):
        N, V = views.shape[:2]
        tok = modules["backbone"].forward_features(views.flatten(0, 1))
        q = modules["predictor"](modules["projector"](tok[:, 0])).reshape(N, V, -1)  # loss: CLS
        with torch.no_grad():
            ht = modules["teacher_backbone"].forward_features(views.flatten(0, 1))[:, 0]
            zt = modules["teacher_projector"](ht).reshape(N, V, -1)
            self._t_std = zt.reshape(-1, zt.shape[-1]).float().std(0).mean().item()
        loss = self._regress(q[:, 0], zt[:, 1]) + self._regress(q[:, 1], zt[:, 0])
        probe_feats = tok[:, 1:].mean(1).detach()    # monitor = audited h (trunk-GAP, F1)
        return ({"loss": loss}, probe_feats, V)

    def post_step(self, modules, step, total_steps):
        m = ema_momentum(step, total_steps, self.cfg.ema_base)
        ema_update(modules["teacher_backbone"], modules["backbone"], m)
        ema_update(modules["teacher_projector"], modules["projector"], m)
        return {"ema_m": m, "teacher_proj_std": self._t_std}

    @torch.inference_mode()
    def eval_features(self, modules, x, device):
        return modules["backbone"].forward_features(x)[:, 1:].mean(1)     # audited h (GAP)

    def probe_dim(self):
        return 384
