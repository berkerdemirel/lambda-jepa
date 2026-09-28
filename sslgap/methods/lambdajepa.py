"""lambda-JEPA: view-to-mean invariance at the projector output, SACReg on the per-image view centers at the backbone (h) and at the projector output (z), with ring buffers and random slices."""
import copy
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from sslgap.data import (LejepaMultiCropDataset, LightlyLejepaMultiCropDataset,
                         ViewsDataset, byol_pair)
from sslgap.methods._common import SACReg, house_scheduler, trunk_arch
from sslgap.methods.base import SSLMethod
from sslgap.models.backbones import build_vit_trunk

def lambdajepa_head(in_dim=384, hidden=2048, out_dim=2048, norm="bn"):
    if norm == "bn":
        return nn.Sequential(nn.Linear(in_dim, hidden), nn.BatchNorm1d(hidden),
                             nn.ReLU(inplace=True),
                             nn.Linear(hidden, hidden), nn.BatchNorm1d(hidden),
                             nn.ReLU(inplace=True),
                             nn.Linear(hidden, out_dim))
    return nn.Sequential(nn.Linear(in_dim, hidden), nn.ReLU(inplace=True),
                         nn.Linear(hidden, hidden), nn.ReLU(inplace=True),
                         nn.Linear(hidden, out_dim))

def lambdajepa_ladder_head(in_dim=384, hidden=2048, out_dim=256, depth=2, width=None):
    w = width or hidden
    if depth == 0:
        return nn.Sequential(nn.Linear(in_dim, hidden), nn.Linear(hidden, out_dim))
    layers = [nn.Linear(in_dim, hidden), nn.BatchNorm1d(hidden), nn.ReLU(inplace=True)]
    for i in range(depth - 1):
        layers += [nn.Linear(hidden if i == 0 else w, w), nn.BatchNorm1d(w),
                   nn.ReLU(inplace=True)]
    layers.append(nn.Linear(hidden if depth == 1 else w, out_dim))
    return nn.Sequential(*layers)

class LambdaJEPA(SSLMethod):
    name = "lambdajepa"
    PULL_W = {"inv": "w_inv", "moment_kl": "w_floor", "h_moment_kl": "h_lamb"}

    def build_modules(self):
        self._mc = self.cfg.get("aug") in ("lejepa_mc", "lightly_mc")
        trunk = build_vit_trunk(self.frame.model_name, self.frame.img_size,
                                dynamic_img_size=self._mc,
                                drop_path_rate=self.cfg.drop_path)
        self._dim = trunk.num_features
        if self.cfg.get("grad_ckpt"):
            trunk.set_grad_checkpointing()
        k = self.cfg.get("head_layers")
        head = (lambdajepa_ladder_head(self._dim, self.cfg.expander_hidden,
                                     self.cfg.expander_dim,
                                     depth=k, width=self.cfg.get("head_width"))
                if k is not None
                else lambdajepa_head(self._dim, self.cfg.expander_hidden,
                                   self.cfg.expander_dim, norm=self.cfg.head_norm))
        shr = self.cfg.get("floor_shrink")
        d_h = self.cfg.get("h_d_slice") or 128
        self.cond_h = SACReg(d_slice=d_h, d_draw=max(128, d_h), shrink=shr)
        d_canon = min(128, self.cfg.expander_dim)
        d_z = self.cfg.get("z_d_slice") or d_canon
        self.cond_z = SACReg(d_slice=d_z, d_draw=d_canon, shrink=shr)
        mods = nn.ModuleDict({"backbone": trunk, "projector": head})
        if self.cfg.get("swa"):
            mods["teacher_backbone"] = copy.deepcopy(trunk).requires_grad_(False)
            mods["teacher_projector"] = copy.deepcopy(head).requires_grad_(False)
            self._swa_k = 0
        return mods

    def arch(self):
        k = self.cfg.get("head_layers")
        dim = getattr(self, "_dim", 384)
        proj = ({"class": "sslgap.methods.lambdajepa.lambdajepa_ladder_head",
                 "kwargs": {"in_dim": dim, "hidden": self.cfg.expander_hidden,
                            "out_dim": self.cfg.expander_dim, "depth": k,
                            "width": self.cfg.get("head_width")}} if k is not None
                else {"class": "sslgap.methods.lambdajepa.lambdajepa_head",
                      "kwargs": {"in_dim": dim, "hidden": self.cfg.expander_hidden,
                                 "out_dim": self.cfg.expander_dim,
                                 "norm": self.cfg.head_norm}})
        a = {"backbone": trunk_arch(self.frame, self.cfg.drop_path,
                                    dynamic_img_size=getattr(self, "_mc", False)),
             "projector": proj}
        if self.cfg.get("swa"):
            a["teacher_backbone"], a["teacher_projector"] = a["backbone"], proj
        return a

    def build_train_dataset(self):
        if self.cfg.get("aug", "byol") == "lightly_mc":
            return LightlyLejepaMultiCropDataset(
                self.frame.dataset, "train", img_size=self.frame.img_size,
                data_root=self.frame.data_root, n_g=self.cfg.get("Vg", 2),
                n_l=self.cfg.get("Vl", 6),
                local_size=self.cfg.get("local_size", 96),
                global_scale=tuple(self.cfg.get("global_scale", (0.3, 1.0))),
                local_scale=tuple(self.cfg.get("local_scale", (0.05, 0.3))))
        if self.cfg.get("aug", "byol") == "lejepa_mc":
            return LejepaMultiCropDataset(
                self.frame.dataset, "train", img_size=self.frame.img_size,
                data_root=self.frame.data_root, n_g=self.cfg.get("Vg", 2),
                n_l=self.cfg.get("Vl", 8), local_size=self.cfg.get("local_size", 96),
                global_scale=tuple(self.cfg.get("global_scale", (0.4, 1.0))),
                local_scale=tuple(self.cfg.get("local_scale", (0.05, 0.4))))
        if self.cfg.get("aug", "byol") == "lejepa":
            return ViewsDataset(self.frame.dataset, "train", V=self.cfg.get("V", 4),
                                img_size=self.frame.img_size, data_root=self.frame.data_root)
        return ViewsDataset(self.frame.dataset, "train", V=2, img_size=self.frame.img_size,
                            data_root=self.frame.data_root,
                            transforms=byol_pair(self.frame.img_size))

    def param_groups(self, modules):
        mwd = self.cfg.get("mlp_wd")
        if mwd is None:
            return [{"params": [p for k in ("backbone", "projector")
                                for p in modules[k].parameters()],
                     "lr": self.cfg.lr, "weight_decay": self.cfg.wd}]
        return [{"params": list(modules["backbone"].parameters()),
                 "lr": self.cfg.lr, "weight_decay": self.cfg.wd},
                {"params": list(modules["projector"].parameters()),
                 "lr": self.cfg.lr, "weight_decay": mwd}]

    def build_scheduler(self, optimizer, steps_per_epoch, total_steps):
        return house_scheduler(optimizer, steps_per_epoch, total_steps,
                               self.cfg.warmup_ep, self.cfg.eta_min)

    def _ring(self, attr, x, q):
        if not q:
            return x
        buf = getattr(self, attr, [])
        out = torch.cat([x] + buf) if buf else x
        setattr(self, attr, [x.detach()] + buf[:q - 1])
        return out

    def _swa_mu(self, modules, views):
        tb, tp = modules["teacher_backbone"], modules["teacher_projector"]
        with torch.no_grad():
            if isinstance(views, (list, tuple)):
                g, l = views
                N, Vg = g.shape[:2]
                cls = torch.cat(
                    [tb.forward_features(g.flatten(0, 1))[:, 0].reshape(N, Vg, -1),
                     tb.forward_features(l.flatten(0, 1))[:, 0]
                     .reshape(N, l.shape[1], -1)], 1)
            else:
                N, V = views.shape[:2]
                cls = tb.forward_features(views.flatten(0, 1))[:, 0].reshape(N, V, -1)
            return tp(cls.flatten(0, 1)).reshape(N, cls.shape[1], -1).mean(1, keepdim=True)

    def post_step(self, modules, step, total_steps):
        if "teacher_backbone" not in modules:
            return {}
        k = self._swa_k
        if self.cfg.get("swa") == "ema":
            t0 = self.cfg.get("swa_tau", 0.996)
            tau = 1 - (1 - t0) * (math.cos(math.pi * step / max(1, total_steps)) + 1) / 2
            w_old, w_new = tau, 1 - tau
        else:
            w_old, w_new = k / (k + 1), 1.0 / (k + 1)
        with torch.no_grad():
            for role in ("backbone", "projector"):
                t, s = modules[f"teacher_{role}"], modules[role]
                for pt, ps in zip(t.parameters(), s.parameters()):
                    pt.mul_(w_old).add_(ps, alpha=w_new)
                for bt, bs in zip(t.buffers(), s.buffers()):
                    bt.copy_(bs)
        self._swa_k = k + 1
        return {}

    def train_mode(self, modules):
        for name, m in modules.items():
            m.eval() if name.startswith("teacher_") else m.train()

    def extras(self):
        return {"swa_k": self._swa_k} if hasattr(self, "_swa_k") else {}

    def load_extras(self, extras):
        if extras and "swa_k" in extras:
            self._swa_k = extras["swa_k"]

    def training_step(self, modules, views, device, y=None):
        if isinstance(views, (list, tuple)):
            g, l = views
            N, Vg = g.shape[:2]
            V = Vg + l.shape[1]
            cls = torch.cat(
                [modules["backbone"].forward_features(g.flatten(0, 1))[:, 0]
                 .reshape(N, Vg, -1),
                 modules["backbone"].forward_features(l.flatten(0, 1))[:, 0]
                 .reshape(N, V - Vg, -1)], 1)
            z = modules["projector"](cls.flatten(0, 1)).reshape(N, V, -1)
            inv = ((z - self._swa_mu(modules, views)).square().mean()
                   if "teacher_backbone" in modules
                   else (z - z.mean(1, keepdim=True)).square().mean())
            cs = self.cfg.get("cond_stream")
            if cs == "grouped":
                zin = (z[:, :Vg].mean(1), z[:, Vg:].mean(1))
                hin = (cls[:, :Vg].mean(1), cls[:, Vg:].mean(1))
            elif cs == "perview":
                zin = tuple(z[:, v] for v in range(V))
                hin = tuple(cls[:, v] for v in range(V))
            else:
                ns = Vg if cs == "globals" else V
                zin, hin = z[:, :ns].mean(1), cls[:, :ns].mean(1)
            probe_feats = cls.flatten(0, 1).detach()
        else:
            N, V = views.shape[:2]
            tok = modules["backbone"].forward_features(views.flatten(0, 1))
            cls = tok[:, 0]
            z = modules["projector"](cls).reshape(N, V, -1)
            inv = ((z - self._swa_mu(modules, views)).square().mean() * (2 * V / (V - 1))
                   if "teacher_backbone" in modules
                   else sum(F.mse_loss(z[:, u], z[:, w])
                            for u in range(V) for w in range(u + 1, V)) / (V * (V - 1) / 2))
            zin = (z.mean(1) if self.cfg.get("z_floor_batch", "pooled") == "view_mean"
                   else z.reshape(N * V, -1))
            hin = (cls.reshape(N, V, -1).mean(1)
                   if self.cfg.get("h_floor_batch", "pooled") == "view_mean" else cls)
            probe_feats = cls.detach()
        q = int(self.cfg.get("queue_steps", 0) or 0)
        qh = self.cfg.get("h_queue_steps")
        qh = q if qh is None else int(qh)

        def _cond(cond, attr, x, qn):
            if isinstance(x, tuple):
                return sum(cond(self._ring(f"{attr}{i}", xi, qn))
                           for i, xi in enumerate(x)) / len(x)
            return cond(self._ring(attr, x, qn))

        reg_z = _cond(self.cond_z, "_zq", zin, q)
        reg_h = _cond(self.cond_h, "_hq", hin, qh)
        loss = self.cfg.w_inv * inv + self.cfg.w_floor * reg_z + self.cfg.h_lamb * reg_h
        terms = {"inv": inv, "moment_kl": reg_z, "h_moment_kl": reg_h}
        return ({"loss": loss, **terms}, probe_feats, V)

    @torch.inference_mode()
    def eval_features(self, modules, x, device):
        return modules["backbone"].forward_features(x)[:, 0]

    def probe_dim(self):
        return getattr(self, "_dim", 384)
