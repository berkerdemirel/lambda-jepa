"""DINO as released, with the optional SACReg term at the backbone (h_reg=sacreg)."""
import copy
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from sslgap.data import MultiCropDataset
from sslgap.methods._common import SACReg, ema_momentum, house_scheduler, trunk_arch
from sslgap.methods.base import SSLMethod
from sslgap.models.backbones import build_vit_trunk
from sslgap.models.heads import DINOHead, DINOLinearHead
from sslgap.models.vitops import ema_update, vit_tokens, vit_tokens_lowres

def dino_head(in_dim=384, hidden=2048, bottleneck=256, K=4096, norm_last_layer=True):
    return DINOHead(in_dim, hidden, bottleneck, K, norm_last_layer)

def dino_linear_head(in_dim=384, K=512, norm_last_layer=True):
    return DINOLinearHead(in_dim, K, norm_last_layer)

def dino_ce(t_logits, s_logits, center, t_temp, s_temp):
    t = F.softmax((t_logits.float() - center) / t_temp, dim=-1)
    return -(t * F.log_softmax(s_logits.float() / s_temp, dim=-1)).sum(-1).mean()

class DINO(SSLMethod):
    PULL_W = {"dino": 1.0, "h_moment_kl": "h_lamb", "h_protoce": "h_protoce"}
    name = "dino"

    def build_modules(self):
        trunk = build_vit_trunk(self.frame.model_name, self.frame.img_size,
                                drop_path_rate=self.cfg.drop_path)
        head = dino_head(384, self.cfg.head_hidden, self.cfg.bottleneck, self.cfg.K,
                         self.cfg.norm_last_layer)
        t_trunk = copy.deepcopy(trunk).requires_grad_(False)
        t_head = copy.deepcopy(head).requires_grad_(False)
        self.center = torch.zeros(self.cfg.K)
        if self.cfg.get("h_reg") == "sacreg":
            self.floor = SACReg()
        mods = {"backbone": trunk, "projector": head,
                "teacher_backbone": t_trunk, "teacher_projector": t_head}
        if self.cfg.get("h_protoce", 0.0):
            hK = self.cfg.get("h_K", 512)
            hh = dino_linear_head(384, hK, self.cfg.norm_last_layer)
            mods["h_head"] = hh
            mods["teacher_h_head"] = copy.deepcopy(hh).requires_grad_(False)
            self.h_center = torch.zeros(hK)
        return nn.ModuleDict(mods)

    def arch(self):
        head = {"class": "sslgap.methods.dino.dino_head",
                "kwargs": {"in_dim": 384, "hidden": self.cfg.head_hidden,
                           "bottleneck": self.cfg.bottleneck, "K": self.cfg.K,
                           "norm_last_layer": self.cfg.norm_last_layer}}
        a = {"backbone": trunk_arch(self.frame, self.cfg.drop_path), "projector": head,
             "teacher_backbone": trunk_arch(self.frame, self.cfg.drop_path),
             "teacher_projector": head}
        if self.cfg.get("h_protoce", 0.0):
            hh = {"class": "sslgap.methods.dino.dino_linear_head",
                  "kwargs": {"in_dim": 384, "K": self.cfg.get("h_K", 512),
                             "norm_last_layer": self.cfg.norm_last_layer}}
            a["h_head"] = hh
            a["teacher_h_head"] = hh
        return a

    def build_train_dataset(self):
        return MultiCropDataset(self.frame.dataset, "train", img_size=self.frame.img_size,
                                local_size=self.cfg.local_size, n_local=self.cfg.n_local,
                                data_root=self.frame.data_root)

    def param_groups(self, modules):
        params = list(modules["backbone"].parameters()) + list(modules["projector"].parameters())
        if "h_head" in modules:
            params += list(modules["h_head"].parameters())
        return [{"params": params,
                 "lr": self.cfg.lr, "weight_decay": self.cfg.wd}]

    def build_scheduler(self, optimizer, steps_per_epoch, total_steps):
        return house_scheduler(optimizer, steps_per_epoch, total_steps,
                               self.cfg.warmup_ep, self.cfg.eta_min)

    def train_mode(self, modules):
        modules["backbone"].train()
        modules["projector"].train()
        modules["teacher_backbone"].eval()
        modules["teacher_projector"].eval()
        if "h_head" in modules:
            modules["h_head"].train()
            modules["teacher_h_head"].eval()

    def on_epoch_start(self, modules, epoch):
        heads = [modules["projector"]] + ([modules["h_head"]] if "h_head" in modules else [])
        for hd in heads:
            for name, p in hd.last.named_parameters():
                frozen_gain = self.cfg.norm_last_layer and name.endswith("original0")
                p.requires_grad_(epoch > 0 and not frozen_gain)

    def training_step(self, modules, batch_x, device, y=None):
        g, l = batch_x
        N, nl = g.shape[0], l.shape[1]
        gx = g.transpose(0, 1).flatten(0, 1)
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
        if self.cfg.get("h_reg") == "sacreg":
            h_loss = self.floor(s_tok[:, 0])
            terms["h_moment_kl"] = h_loss
            loss = loss + self.cfg.h_lamb * h_loss
            if self.cfg.get("h_taps", "cls") == "clsgap":
                g_loss = self.floor(s_tok[:, 1:].mean(1))
                terms["h_moment_kl_gap"] = g_loss
                loss = loss + self.cfg.get("h_lamb_gap", self.cfg.h_lamb) * g_loss
        if self.cfg.get("h_protoce", 0.0):
            with torch.no_grad():
                t_h = modules["teacher_h_head"](t_tok[:, 0]).reshape(2, N, -1)
            s_h = modules["h_head"](s_tok[:, 0]).reshape(2, N, -1)
            hc = self.h_center.to(device)
            h_pairs = [dino_ce(t_h[t], s_h[s], hc, self.cfg.t_temp, self.cfg.s_temp)
                       for t in range(2) for s in range(2) if s != t]
            h_ce = torch.stack(h_pairs).mean()
            terms["h_protoce"] = h_ce
            loss = loss + self.cfg.h_protoce * h_ce
            self._t_h = t_h.detach()
        self._t_cls = t_cls.detach()
        probe_feats = t_tok[:, 0].reshape(2, N, -1).transpose(0, 1).flatten(0, 1).detach()
        return ({"loss": loss, **terms}, probe_feats, 2)

    def post_step(self, modules, step, total_steps):
        m = ema_momentum(step, total_steps, self.cfg.ema_base)
        ema_update(modules["teacher_backbone"], modules["backbone"], m)
        ema_update(modules["teacher_projector"], modules["projector"], m)
        t = self._t_cls.reshape(-1, self.cfg.K).float()
        self.center = (self.cfg.center_m * self.center.to(t.device)
                       + (1 - self.cfg.center_m) * t.mean(0)).detach()
        if "h_head" in modules:
            ema_update(modules["teacher_h_head"], modules["h_head"], m)
            th = self._t_h.reshape(-1, self._t_h.shape[-1]).float()
            self.h_center = (self.cfg.center_m * self.h_center.to(th.device)
                             + (1 - self.cfg.center_m) * th.mean(0)).detach()
        with torch.no_grad():
            p = F.softmax((t - self.center) / self.cfg.t_temp, -1)
            ent_sample = -(p * p.clamp_min(1e-9).log()).sum(-1).mean() / math.log(self.cfg.K)
            protos = int(p.argmax(-1).unique().numel())
        return {"ema_m": m, "teacher_entropy_sample": ent_sample.item(),
                "proto_used": protos}

    def extras(self):
        e = {"center_cls": self.center.cpu()}
        if hasattr(self, "h_center"):
            e["h_center"] = self.h_center.cpu()
        return e

    def load_extras(self, extras):
        if "center_cls" in extras:
            self.center = extras["center_cls"]
        if "h_center" in extras:
            self.h_center = extras["h_center"]

    @torch.inference_mode()
    def eval_features(self, modules, x, device):
        return modules["teacher_backbone"].forward_features(x)[:, 0]

    def probe_dim(self):
        return 384
