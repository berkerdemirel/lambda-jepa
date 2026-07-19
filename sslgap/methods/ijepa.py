"""I-JEPA (Assran et al. 2023) — port of the ssl_explore/sslx/ijepa.py building blocks (verbatim
MaskSampler and Predictor; the sslx trainer was deleted) on the frame. Faithful pieces: multi-block
target sampling (shapes per batch, locations per image; official MaskCollator behaviour), context =
block minus target-union truncated to batch-min, narrow ViT predictor (dim 384 depth 6; FIXED
sincos pos, verified vs official), smooth-L1 on layer-normed EMA-teacher tokens (official code; paper text
says L2), EMA momentum 0.996->1 LINEAR (paper). Minimal aug (RRC 0.3-1.0 + flip). Toy adaptations
per D-012: house AdamW schedule; drop_path 0. Known fragility (C-JEPA): teacher-token std is
monitored per step from birth."""
import copy
import math
import random

import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.layers import trunc_normal_
from timm.models.vision_transformer import Block

from sslgap.models.posembed import get_2d_sincos_pos_embed

from sslgap.data import ViewsDataset, minaug_stack
from sslgap.methods._common import MomentFloor, house_scheduler, trunk_arch
from sslgap.methods.base import SSLMethod
from sslgap.models.backbones import build_vit_trunk
from sslgap.models.vitops import ema_update, vit_tokens


class MaskSampler:
    """Port of sslx/ijepa.MaskSampler (shapes per batch, locations per image)."""

    def __init__(self, grid=16, n_targets=4, tgt_scale=(0.15, 0.2), tgt_aspect=(0.75, 1.5),
                 ctx_scale=(0.85, 1.0)):
        self.g, self.n = grid, n_targets
        self.tgt_scale, self.tgt_aspect, self.ctx_scale = tgt_scale, tgt_aspect, ctx_scale

    def _shape(self, scale, aspect):
        s = random.uniform(*scale) * self.g ** 2
        a = math.exp(random.uniform(math.log(aspect[0]), math.log(aspect[1]))) if aspect else 1.0
        h = max(1, min(self.g, round(math.sqrt(s / a))))
        w = max(1, min(self.g, round(math.sqrt(s * a))))
        return h, w

    def _block(self, h, w):
        t, l = random.randint(0, self.g - h), random.randint(0, self.g - w)
        rows = torch.arange(t, t + h)[:, None] * self.g + torch.arange(l, l + w)[None, :]
        return rows.flatten()

    def __call__(self, B):
        th, tw = self._shape(self.tgt_scale, self.tgt_aspect)
        ch, cw = self._shape(self.ctx_scale, None)
        tgt = torch.stack([torch.stack([self._block(th, tw) for _ in range(self.n)])
                           for _ in range(B)])
        ctx = []
        for b in range(B):
            keep = torch.ones(self.g ** 2, dtype=torch.bool)
            keep[tgt[b].flatten()] = False
            blk = self._block(ch, cw)
            ctx.append(blk[keep[blk]])
        kc = min(len(c) for c in ctx)
        if kc < 10:
            raise RuntimeError(f"context {kc} < official min_keep=10 — scales misconfigured")
        ctx = torch.stack([c[torch.randperm(len(c))[:kc]] for c in ctx])
        return ctx, tgt


class Predictor(nn.Module):
    """Port of sslx/ijepa.Predictor (narrow ViT; one pass per target block via batch repeat)."""

    def __init__(self, dim=384, depth=6, heads=6, n_patches=256):
        super().__init__()
        self.embed = nn.Linear(dim, dim)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, dim))
        grid = int(n_patches ** 0.5)                      # FIXED sincos (official predictor pos,
        self.register_buffer("pos", get_2d_sincos_pos_embed(dim, grid)[None])  # ijepa@52c1ae9)
        self.blocks = nn.Sequential(*[Block(dim, heads, qkv_bias=True) for _ in range(depth)])
        self.norm = nn.LayerNorm(dim)
        self.out = nn.Linear(dim, dim)
        trunc_normal_(self.mask_token, std=0.02)

    def _pos_at(self, idx):
        pos = self.pos.expand(idx.shape[0], -1, -1)
        return torch.gather(pos, 1, idx[..., None].expand(-1, -1, pos.shape[-1]))

    def forward(self, ctx, ctx_idx, tgt_idx):
        B, T, Kt = tgt_idx.shape
        z = (self.embed(ctx) + self._pos_at(ctx_idx)).repeat_interleave(T, 0)
        mt = self.mask_token + self._pos_at(tgt_idx.reshape(B * T, Kt))
        h = self.norm(self.blocks(torch.cat([z, mt], 1)))[:, -Kt:]
        return self.out(h).reshape(B, T, Kt, -1)


def ijepa_predictor(dim=384, depth=6, heads=6, n_patches=256):
    return Predictor(dim, depth, heads, n_patches)


class IJEPA(SSLMethod):
    name = "ijepa"

    def __init__(self, cfg, frame):
        super().__init__(cfg, frame)
        grid = frame.img_size // self.cfg.patch
        self.n_patches = grid * grid
        self.masks = MaskSampler(grid=grid, n_targets=self.cfg.n_targets)

    def build_modules(self):
        trunk = build_vit_trunk(self.frame.model_name, self.frame.img_size, drop_path_rate=0.0)
        pred = ijepa_predictor(384, self.cfg.pred_depth, 6, self.n_patches)
        t_trunk = copy.deepcopy(trunk).requires_grad_(False)
        if self.cfg.get("h_reg") == "moment":     # E20 calibrated zoo floor (no RNG at construction)
            self.floor = MomentFloor()
        return nn.ModuleDict({"backbone": trunk, "predictor": pred, "teacher_backbone": t_trunk})

    def arch(self):
        return {"backbone": trunk_arch(self.frame, 0.0),
                "predictor": {"class": "sslgap.methods.ijepa.ijepa_predictor",
                              "kwargs": {"dim": 384, "depth": self.cfg.pred_depth, "heads": 6,
                                         "n_patches": self.n_patches}},
                "teacher_backbone": trunk_arch(self.frame, 0.0)}

    def build_train_dataset(self):
        return ViewsDataset(self.frame.dataset, "train", V=1, img_size=self.frame.img_size,
                            data_root=self.frame.data_root,
                            transforms=[minaug_stack(self.frame.img_size, scale=(0.3, 1.0))])

    def param_groups(self, modules):
        return [{"params": list(modules["backbone"].parameters())
                 + list(modules["predictor"].parameters()),
                 "lr": self.cfg.lr, "weight_decay": self.cfg.wd}]

    def build_scheduler(self, optimizer, steps_per_epoch, total_steps):
        return house_scheduler(optimizer, steps_per_epoch, total_steps,
                               self.cfg.warmup_ep, self.cfg.eta_min)

    def training_step(self, modules, views, device, y=None):
        x = views[:, 0]
        B = x.shape[0]
        ctx_keep, tgt_idx = self.masks(B)
        ctx_keep, tgt_idx = ctx_keep.to(device), tgt_idx.to(device)
        ctx = vit_tokens(modules["backbone"], x, keep=ctx_keep)[:, 1:]
        with torch.no_grad():
            tt = modules["teacher_backbone"].forward_features(x)[:, 1:]
            tgt = F.layer_norm(tt, (tt.shape[-1],))
            tgt = torch.gather(tgt, 1, tgt_idx.reshape(B, -1)[..., None]
                               .expand(-1, -1, tgt.shape[-1])).reshape(*tgt_idx.shape, -1)
            self._t_std = tt.float().std(-1).mean().item()
        pred = modules["predictor"](ctx, ctx_keep, tgt_idx)
        jepa = F.smooth_l1_loss(pred.float(), tgt.float())
        loss, terms = jepa, {"jepa": jepa}
        # E20 (calibrated zoo floor): moment floor at ijepa's TRAINING-TIME student h — GAP
        # over the context-encoder tokens (the grad branch; audited h is the EMA teacher GAP —
        # deviation declared on the E20 card; the teacher inherits conditioning via EMA).
        if self.cfg.get("h_reg") == "moment":
            h_loss = self.floor(ctx.mean(1))
            loss = loss + self.cfg.h_lamb * h_loss
            terms["h_moment_kl"] = h_loss
        probe_feats = tt.mean(1).detach()      # teacher GAP = the audited branch (PROTOCOL §3),
        return ({"loss": loss, **terms}, probe_feats, 1)  # free: tt already computed

    def train_mode(self, modules):
        modules["backbone"].train()
        modules["predictor"].train()
        modules["teacher_backbone"].eval()      # EMA target encoder stays eval (sslx convention)

    def post_step(self, modules, step, total_steps):
        m = self.cfg.ema_base + (1.0 - self.cfg.ema_base) * step / total_steps   # linear (paper)
        ema_update(modules["teacher_backbone"], modules["backbone"], m)
        return {"ema_m": m, "teacher_tok_std": self._t_std}

    @torch.inference_mode()
    def eval_features(self, modules, x, device):
        feats = modules["teacher_backbone"].forward_features(x)
        return feats[:, 1:].mean(1)                            # audited branch (teacher GAP)

    def probe_dim(self):
        return 384
