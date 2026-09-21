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
from sslgap.methods._common import SACReg, ema_momentum, house_scheduler, trunk_arch
from sslgap.methods.base import SSLMethod
from sslgap.models.backbones import build_vit_trunk
from sslgap.models.vitops import ema_update


def byol_mlp(in_dim, hidden=4096, out_dim=256):
    return nn.Sequential(nn.Linear(in_dim, hidden), nn.BatchNorm1d(hidden), nn.ReLU(inplace=True),
                         nn.Linear(hidden, out_dim))


def byol_h_predictor(dim=384):                # E17: byol's predictor at h, minimal (a linear map)
    return nn.Linear(dim, dim)


class BYOL(SSLMethod):
    # pull-instrument weight map (added 2026-08-25, E20 dino re-dose program: byol =
    # the winner-share reference lane); w_regress defaults to 1.0 in the method config.
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
        if self.cfg.get("h_align", 0.0):      # E17: linear predictor at h (byol's own mechanism)
            mods["h_predictor"] = byol_h_predictor(384)
        if self.cfg.get("h_reg") == "sacreg":     # E20 calibrated zoo floor (no RNG at construction)
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
        if "h_predictor" in modules:          # E17: the h-predictor is student-side, trainable
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
        q = modules["predictor"](modules["projector"](tok[:, 0])).reshape(N, V, -1)  # loss: CLS
        with torch.no_grad():
            ht = modules["teacher_backbone"].forward_features(views.flatten(0, 1))[:, 0]
            zt = modules["teacher_projector"](ht).reshape(N, V, -1)
            self._t_std = zt.reshape(-1, zt.shape[-1]).float().std(0).mean().item()
        regress = self._regress(q[:, 0], zt[:, 1]) + self._regress(q[:, 1], zt[:, 0])
        loss, terms = regress, {"regress": regress}
        # E17 h-pull (D-039): byol's OWN mechanism at h — a small LINEAR predictor maps student
        # trunk-CLS to predict the stop-grad EMA-teacher trunk-CLS, cross-view (Berker 2026-07-15:
        # "add a linear to its h and do the pull there"). The direct/no-predictor version was inert
        # (student-CLS ~= teacher-CLS at h, 0.999 same-view); the predictor is byol's actual
        # asymmetry, so the pull is measured/applied through it. Additive small dose on the z-loss.
        h_align = self.cfg.get("h_align", 0.0)
        if h_align:
            ph = modules["h_predictor"](tok[:, 0]).reshape(N, V, -1)
            th = ht.reshape(N, V, -1)
            ha = self._regress(ph[:, 0], th[:, 1]) + self._regress(ph[:, 1], th[:, 0])
            loss = loss + h_align * ha
            terms["h_align"] = ha
        # E20 (calibrated zoo floor): additive moment floor at declared h (student CLS, pooled
        # views); regress/EMA untouched. Dose = per-method calibrated share (E20 card).
        if self.cfg.get("h_reg") == "sacreg":
            h_loss = self.floor(tok[:, 0])
            loss = loss + self.cfg.h_lamb * h_loss
            terms["h_moment_kl"] = h_loss
        probe_feats = tok[:, 0].detach()             # monitor = declared h (projector-input CLS, D-036); _best aligns to audit
        return ({"loss": loss, **terms}, probe_feats, V)

    def post_step(self, modules, step, total_steps):
        m = ema_momentum(step, total_steps, self.cfg.ema_base)
        ema_update(modules["teacher_backbone"], modules["backbone"], m)
        ema_update(modules["teacher_projector"], modules["projector"], m)
        return {"ema_m": m, "teacher_proj_std": self._t_std}

    @torch.inference_mode()
    def eval_features(self, modules, x, device):
        return modules["backbone"].forward_features(x)[:, 0]              # declared h (projector-input CLS, D-036)

    def probe_dim(self):
        return 384
