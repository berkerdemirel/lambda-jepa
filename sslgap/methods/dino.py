"""DINO, classic (Caron et al. 2021) — restructured port of ssl_explore/experiments/train_dinov2.py
(w_ibot=0, w_koleo=0 branch) + sslx/dinov2.py terms. Toy scale per the sslx precedent + house
incident ledger: K=4096 prototypes (Imagenette has 9469 train images), teacher temp CONSTANT 0.04
(no ramp) and EMA base 0.99 — at ~37 steps/epoch the paper schedule (0.996, temp ramp) hit the
uniform fixed point in the predecessor project (run 61018543). grad_clip=3.0 (DINO paper / control).
Views: Lightly/DINO-faithful 2 globals + n_local 64px locals. Prototypes frozen during epoch 0
(paper). Monitors: teacher entropy (batch-mean = centering pressure, per-sample must FALL) +
distinct argmax prototypes."""
import copy
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from sslgap.data import MultiCropDataset
from sslgap.methods._common import MomentFloor, ema_momentum, house_scheduler, trunk_arch
from sslgap.methods.base import SSLMethod
from sslgap.models.backbones import build_vit_trunk
from sslgap.models.heads import DINOHead
from sslgap.models.vitops import ema_update, vit_tokens, vit_tokens_lowres


def dino_head(in_dim=384, hidden=2048, bottleneck=256, K=4096, norm_last_layer=True):
    return DINOHead(in_dim, hidden, bottleneck, K, norm_last_layer)


def dino_ce(t_logits, s_logits, center, t_temp, s_temp):
    t = F.softmax((t_logits.float() - center) / t_temp, dim=-1)
    return -(t * F.log_softmax(s_logits.float() / s_temp, dim=-1)).sum(-1).mean()


class DINO(SSLMethod):
    name = "dino"

    def build_modules(self):
        trunk = build_vit_trunk(self.frame.model_name, self.frame.img_size,
                                drop_path_rate=self.cfg.drop_path)
        head = dino_head(384, self.cfg.head_hidden, self.cfg.bottleneck, self.cfg.K,
                         self.cfg.norm_last_layer)
        t_trunk = copy.deepcopy(trunk).requires_grad_(False)
        t_head = copy.deepcopy(head).requires_grad_(False)
        self.center = torch.zeros(self.cfg.K)
        if self.cfg.get("h_reg") == "moment":     # E12 cross-method arm (no RNG at construction)
            self.floor = MomentFloor()
        return nn.ModuleDict({"backbone": trunk, "projector": head,
                              "teacher_backbone": t_trunk, "teacher_projector": t_head})

    def arch(self):
        head = {"class": "sslgap.methods.dino.dino_head",
                "kwargs": {"in_dim": 384, "hidden": self.cfg.head_hidden,
                           "bottleneck": self.cfg.bottleneck, "K": self.cfg.K,
                           "norm_last_layer": self.cfg.norm_last_layer}}
        return {"backbone": trunk_arch(self.frame, self.cfg.drop_path), "projector": head,
                "teacher_backbone": trunk_arch(self.frame, self.cfg.drop_path),
                "teacher_projector": head}

    def build_train_dataset(self):
        return MultiCropDataset(self.frame.dataset, "train", img_size=self.frame.img_size,
                                local_size=self.cfg.local_size, n_local=self.cfg.n_local,
                                data_root=self.frame.data_root)

    def param_groups(self, modules):
        return [{"params": list(modules["backbone"].parameters())
                 + list(modules["projector"].parameters()),
                 "lr": self.cfg.lr, "weight_decay": self.cfg.wd}]

    def build_scheduler(self, optimizer, steps_per_epoch, total_steps):
        return house_scheduler(optimizer, steps_per_epoch, total_steps,
                               self.cfg.warmup_ep, self.cfg.eta_min)

    def train_mode(self, modules):
        modules["backbone"].train()
        modules["projector"].train()
        modules["teacher_backbone"].eval()      # sslx control: teachers eval (kills drop_path
        modules["teacher_projector"].eval()     # stochasticity in targets; official DINO equiv.)

    def on_epoch_start(self, modules, epoch):
        # paper: prototype layer frozen during epoch 0; the weight-norm gain (original0) stays
        # frozen FOREVER when norm_last_layer=True (DINOHead init semantics).
        for name, p in modules["projector"].last.named_parameters():
            frozen_gain = self.cfg.norm_last_layer and name.endswith("original0")
            p.requires_grad_(epoch > 0 and not frozen_gain)

    def training_step(self, modules, batch_x, device, y=None):
        g, l = batch_x                                            # [N,2,C,G,G], [N,nl,C,L,L]
        N, nl = g.shape[0], l.shape[1]
        gx = g.transpose(0, 1).flatten(0, 1)                      # view-major [2N]
        lx = l.transpose(0, 1).flatten(0, 1)
        with torch.no_grad():
            t_tok = modules["teacher_backbone"].forward_features(gx)
            t_cls = modules["teacher_projector"](t_tok[:, 0]).reshape(2, N, -1)
        s_tok = vit_tokens(modules["backbone"], gx)
        s_cls_g = modules["projector"](s_tok[:, 0]).reshape(2, N, -1)
        s_cls_l = modules["projector"](
            vit_tokens_lowres(modules["backbone"], lx)[:, 0]).reshape(nl, N, -1)
        center = self.center.to(device)
        pairs = []
        for t in range(2):
            pairs += [dino_ce(t_cls[t], s_cls_g[s], center, self.cfg.t_temp, self.cfg.s_temp)
                      for s in range(2) if s != t]
            pairs += [dino_ce(t_cls[t], s_cls_l[s], center, self.cfg.t_temp, self.cfg.s_temp)
                      for s in range(nl)]
        dino_loss = torch.stack(pairs).mean()
        loss = dino_loss
        terms = {"dino": dino_loss}
        # E12 cross-method arm: additive moment floor at the STUDENT's h (global-crop trunk CLS —
        # gradients flow only through the student; the audited teacher h follows by EMA).
        if self.cfg.get("h_reg") == "moment":
            h_loss = self.floor(s_tok[:, 0])
            terms["h_moment_kl"] = h_loss
            loss = loss + self.cfg.h_lamb * h_loss
            if self.cfg.get("h_taps", "cls") == "clsgap":   # H-wave (D-035): floor BOTH trunk
                g_loss = self.floor(s_tok[:, 1:].mean(1))   # readouts; gap dose = h_lamb_gap
                terms["h_moment_kl_gap"] = g_loss           # (defaults to h_lamb)
                loss = loss + self.cfg.get("h_lamb_gap", self.cfg.h_lamb) * g_loss
        self._t_cls = t_cls.detach()
        # probe monitors the AUDITED branch (teacher CLS, PROTOCOL §3) so _best selection aligns
        # with what the audit evaluates; view-major [2N] -> image-major per the base.py contract
        # (the M1 toy.dino.s0 label-misalignment incident — HISTORY 2026-07-02).
        probe_feats = t_tok[:, 0].reshape(2, N, -1).transpose(0, 1).flatten(0, 1).detach()
        return ({"loss": loss, **terms}, probe_feats, 2)

    def post_step(self, modules, step, total_steps):
        m = ema_momentum(step, total_steps, self.cfg.ema_base)
        ema_update(modules["teacher_backbone"], modules["backbone"], m)
        ema_update(modules["teacher_projector"], modules["projector"], m)
        t = self._t_cls.reshape(-1, self.cfg.K).float()
        self.center = (self.cfg.center_m * self.center.to(t.device)
                       + (1 - self.cfg.center_m) * t.mean(0)).detach()
        with torch.no_grad():
            p = F.softmax((t - self.center) / self.cfg.t_temp, -1)
            ent_sample = -(p * p.clamp_min(1e-9).log()).sum(-1).mean() / math.log(self.cfg.K)
            protos = int(p.argmax(-1).unique().numel())
        return {"ema_m": m, "teacher_entropy_sample": ent_sample.item(),
                "proto_used": protos}

    def extras(self):
        return {"center_cls": self.center.cpu()}

    def load_extras(self, extras):
        if "center_cls" in extras:
            self.center = extras["center_cls"]

    @torch.inference_mode()
    def eval_features(self, modules, x, device):
        return modules["teacher_backbone"].forward_features(x)[:, 0]   # audited branch (teacher CLS)

    def probe_dim(self):
        return 384
