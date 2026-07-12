"""PIVOT MVI (E15, D-032) — proposal §5.7/§6.1 at the M2 frame, head-less per §5.1: the trunk's
h.gap (384) DIRECTLY regresses a 384-d two-bandwidth RFF sketch of a frozen mae tokenizer's
descriptor of withheld sharp context (stopgrad), plus same-event view consistency and a
variance floor (inactive at init by construction). Optional hflip token-reindex transport
(arm B; crop/warp transport deferred — declared deviation). No projector, no predictor, no EMA.
Target stats/W/b/gamma frozen from stored E14 ctx features (experiments/e15_target_prep.py);
lambdas by equal-pull-at-init (experiments/e15_pull.py), recorded on the E15 card."""
import hashlib

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from sslgap.data import PivotTrainDataset
from sslgap.methods._common import house_scheduler, trunk_arch
from sslgap.methods.base import SSLMethod
from sslgap.models.backbones import build_vit_trunk


class TargetSketch(nn.Module):
    """Frozen tokenizer trunk + standardization + fixed RFF blocks -> z [B, 384]. Everything is
    buffers/frozen params; forward runs under no_grad (stopgrad is structural, §5.2 step 4)."""

    def __init__(self, tokenizer_trunk, prep_path):
        super().__init__()
        self.trunk = tokenizer_trunk.eval()
        for p in self.trunk.parameters():
            p.requires_grad_(False)
        d = np.load(prep_path)
        self.prep_sha = hashlib.sha256(open(prep_path, "rb").read()).hexdigest()[:16]
        for k in ("mu", "sd", "W", "b"):
            self.register_buffer(k, torch.from_numpy(d[k].astype(np.float32)))
        self.block = int(d["block"])                     # 192: sqrt(2/block) per-block scale
        self.gamma = float(d["gamma"])                   # var-floor level (0.5x target dim-std)
        self.var_z = float(d["var_z"])                   # constant-h floor of L_pred (audit ref)

    @torch.no_grad()
    def forward(self, ctx):
        h = self.trunk.forward_features(ctx)[:, 1:].mean(1)
        with torch.autocast(ctx.device.type, enabled=False):     # RFF in fp32 (target fidelity)
            y = (h.float() - self.mu) / self.sd
            return np.sqrt(2.0 / self.block) * torch.cos(y @ self.W + self.b)


class Pivot(SSLMethod):
    name = "pivot"

    def build_modules(self):
        trunk = build_vit_trunk(self.frame.model_name, self.frame.img_size,
                                drop_path_rate=self.cfg.drop_path)
        ck = torch.load(self.cfg.tokenizer_ckpt, map_location="cpu", weights_only=False)
        assert ck["format"] == "sslgap/ckpt/v1" and ck["method"] == "mae", ck.get("method")
        spec = ck["arch"]["backbone"]
        from sslgap.ckpt.adapters import _resolve
        tok = _resolve(spec["class"])(**spec["kwargs"])
        tok.load_state_dict(ck["modules"]["backbone"])
        self.sketch = TargetSketch(tok, self.cfg.target_prep)   # device-moved lazily in step
        return nn.ModuleDict({"backbone": trunk})

    def arch(self):
        return {"backbone": trunk_arch(self.frame, self.cfg.drop_path)}

    def build_train_dataset(self):
        return PivotTrainDataset(self.frame.dataset, "train", self.frame.img_size,
                                 data_root=self.frame.data_root, f=self.cfg.fovea)

    def param_groups(self, modules):
        return [{"params": [p for p in modules["backbone"].parameters()],
                 "lr": self.cfg.lr, "weight_decay": self.cfg.wd}]

    def build_scheduler(self, optimizer, steps_per_epoch, total_steps):
        return house_scheduler(optimizer, steps_per_epoch, total_steps,
                               self.cfg.warmup_ep, self.cfg.eta_min)

    def _gap(self, tok):
        return tok[:, 1:].mean(1)

    def training_step(self, modules, batch_x, device, y=None):
        v1, v2, ctx = batch_x
        if next(self.sketch.parameters()).device != v1.device:
            self.sketch.to(v1.device).eval()
        tok1 = modules["backbone"].forward_features(v1)
        tok2 = modules["backbone"].forward_features(v2)
        h1, h2 = self._gap(tok1), self._gap(tok2)
        z = self.sketch(ctx)
        pred = F.smooth_l1_loss(h1, z) + F.smooth_l1_loss(h2, z)
        view = F.mse_loss(h1, h2)
        hh = torch.cat([h1, h2]).float()
        std = (hh.var(0) + 1e-6).sqrt()
        var = F.relu(self.sketch.gamma - std).square().mean()
        loss = pred + self.cfg.lamb_view * view + self.cfg.lamb_var * var
        terms = {"pred": pred, "view": view, "var": var,
                 "h_std_mean": std.mean().detach(),
                 "pred_over_varz": pred.detach() / self.sketch.var_z}   # 1.0 = constant-h floor
        if self.cfg.get("transport"):
            tokf = modules["backbone"].forward_features(torch.flip(v1, dims=[3]))
            g = v1.shape[-1] // 16
            patches = tokf[:, 1:].reshape(-1, g, g, tok1.shape[-1]).flip(2).reshape(tok1[:, 1:].shape)
            tr = F.smooth_l1_loss(patches, tok1[:, 1:])
            loss = loss + self.cfg.lamb_t * tr
            terms["transport"] = tr
        probe_feats = torch.stack([h1, h2], 1).flatten(0, 1).detach()   # image-major, k=2
        return ({"loss": loss, **terms}, probe_feats, 2)

    def post_step(self, modules, step, total_steps):
        return {}

    @torch.inference_mode()
    def eval_features(self, modules, x, device):
        return modules["backbone"].forward_features(x)[:, 1:].mean(1)   # audited h (GAP)

    def probe_dim(self):
        return 384
