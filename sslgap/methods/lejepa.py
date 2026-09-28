"""LeJEPA (SIGReg + invariance at the projector output) with the optional backbone term (h_reg=sacreg | sigreg)."""
import timm
import torch
import torch.nn as nn
from torch.nn.utils.parametrizations import spectral_norm
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from torchvision.ops import MLP

from sslgap.data import LejepaMultiCropDataset, LightlyLejepaMultiCropDataset, ViewsDataset
from sslgap.methods._common import SACReg
from sslgap.methods.base import SSLMethod

def t_nu_cf(t, nu):
    import numpy as np
    from scipy.special import gammaln, kv
    u = t.double().numpy() * (nu - 2) ** 0.5
    pos = u > 0
    logphi = np.zeros_like(u)
    logphi[pos] = (np.log(kv(nu / 2, u[pos])) + (nu / 2) * np.log(u[pos])
                   - gammaln(nu / 2) - (nu / 2 - 1) * np.log(2.0))
    return torch.from_numpy(np.exp(logphi)).float()

class SIGReg(nn.Module):

    def __init__(self, knots=17, n_slices=256, t_max=3.0, standardize=False, nu=None):
        super().__init__()
        self.n_slices = n_slices
        self.standardize = standardize
        t = torch.linspace(0, t_max, knots, dtype=torch.float32)
        dt = t_max / (knots - 1)
        weights = torch.full((knots,), 2 * dt, dtype=torch.float32)
        weights[[0, -1]] = dt
        window = torch.exp(-t.square() / 2.0)
        self.register_buffer("t", t)
        self.register_buffer("phi", window if nu is None else t_nu_cf(t, nu))
        self.register_buffer("weights", weights * window)

    def forward(self, proj):
        A = torch.randn(proj.size(-1), self.n_slices, device=proj.device)
        A = A.div_(A.norm(p=2, dim=0))
        s = proj @ A
        if self.standardize:
            s = (s - s.mean(-2, keepdim=True)) / s.std(-2, keepdim=True).clamp_min(1e-4)
        x_t = s.unsqueeze(-1) * self.t
        err = (x_t.cos().mean(-3) - self.phi).square() + x_t.sin().mean(-3).square()
        statistic = (err @ self.weights) * proj.size(-2)
        return statistic.mean()

def lejepa_encoder(model_name, img_size, emb_dim=512, drop_path=0.1, dynamic_img_size=False):
    return timm.create_model(model_name, pretrained=False, num_classes=emb_dim,
                             drop_path_rate=drop_path, img_size=img_size,
                             dynamic_img_size=dynamic_img_size)

def lejepa_projector(proj_dim, emb_dim=512, hidden=2048, depth=3, spec_norm=False):
    if depth == 0:
        return nn.Identity()
    mlp = MLP(emb_dim, [hidden] * (depth - 1) + [proj_dim], norm_layer=nn.BatchNorm1d)
    if spec_norm:
        for m in mlp.modules():
            if isinstance(m, nn.Linear):
                spectral_norm(m)
    return mlp

H_KEYS = {"sigreg": "h_sigreg", "sacreg": "h_moment_kl", "sigreg_std": "h_sigreg_std",
          "sigreg_t": "h_sigreg_t"}

class LeJEPA(SSLMethod):
    name = "lejepa"
    _epoch = 0

    def on_epoch_start(self, modules, epoch):
        self._epoch = epoch

    def build_modules(self):
        self._mc = self.cfg.get("aug") in ("lejepa_mc", "lightly_mc")
        enc = lejepa_encoder(self.frame.model_name, self.frame.img_size,
                             emb_dim=self.cfg.emb_dim, drop_path=self.cfg.drop_path,
                             dynamic_img_size=self._mc)
        if self.cfg.get("grad_ckpt", False):
            enc.set_grad_checkpointing()
        self._probe_dim = self.cfg.emb_dim or enc.num_features
        proj = lejepa_projector(self.cfg.proj_dim, emb_dim=self._probe_dim,
                                depth=self.cfg.get("proj_depth", 3),
                                spec_norm=self.cfg.get("spec_norm", False))
        self.sigreg = SIGReg(n_slices=self.cfg.get("n_slices", 256))
        if self.cfg.get("floor", "sigreg") == "sacreg" or self.cfg.get("h_reg") == "sacreg":
            self.floor = SACReg()
        h_reg = self.cfg.get("h_reg")
        if h_reg == "sigreg_std":
            self.sigreg_std = SIGReg(standardize=True)
        elif h_reg == "sigreg_t":
            self.sigreg_t = SIGReg(nu=self.cfg.sigreg_nu)
        return nn.ModuleDict({"encoder": enc, "projector": proj})

    def arch(self):
        proj_kwargs = {"proj_dim": self.cfg.proj_dim,
                       "emb_dim": self.cfg.emb_dim or self._probe_dim}
        if self.cfg.get("proj_depth", 3) != 3:
            proj_kwargs["depth"] = self.cfg.proj_depth
        if self.cfg.get("spec_norm", False):
            proj_kwargs["spec_norm"] = True
        enc_kwargs = {"model_name": self.frame.model_name,
                      "img_size": self.frame.img_size,
                      "emb_dim": self.cfg.emb_dim,
                      "drop_path": self.cfg.drop_path}
        if getattr(self, "_mc", False):
            enc_kwargs["dynamic_img_size"] = True
        return {"encoder": {"class": "sslgap.methods.lejepa.lejepa_encoder",
                            "kwargs": enc_kwargs},
                "projector": {"class": "sslgap.methods.lejepa.lejepa_projector",
                              "kwargs": proj_kwargs}}

    def build_train_dataset(self):
        if self.cfg.get("aug") == "lightly_mc":
            return LightlyLejepaMultiCropDataset(
                self.frame.dataset, "train", img_size=self.frame.img_size,
                data_root=self.frame.data_root, n_l=self.cfg.get("Vl", 6),
                local_size=self.cfg.get("local_size", 96),
                global_scale=tuple(self.cfg.get("global_scale", (0.3, 1.0))),
                local_scale=tuple(self.cfg.get("local_scale", (0.05, 0.3))))
        if self.cfg.get("aug") == "lejepa_mc":
            return LejepaMultiCropDataset(
                self.frame.dataset, "train", img_size=self.frame.img_size,
                data_root=self.frame.data_root, n_g=self.cfg.get("Vg", 2),
                n_l=self.cfg.get("Vl", 8), local_size=self.cfg.get("local_size", 96),
                global_scale=tuple(self.cfg.get("global_scale", (0.3, 1.0))),
                local_scale=tuple(self.cfg.get("local_scale", (0.05, 0.3))))
        return ViewsDataset(self.frame.dataset, "train", V=self.cfg.V,
                            img_size=self.frame.img_size, data_root=self.frame.data_root)

    def param_groups(self, modules):
        return [{"params": list(modules["encoder"].parameters())
                 + list(modules["projector"].parameters()),
                 "lr": self.cfg.lr, "weight_decay": self.cfg.wd}]

    def build_scheduler(self, optimizer, steps_per_epoch, total_steps):
        warmup = steps_per_epoch * self.cfg.warmup_ep
        s1 = LinearLR(optimizer, start_factor=0.01, total_iters=warmup)
        s2 = CosineAnnealingLR(optimizer, T_max=total_steps - warmup, eta_min=self.cfg.eta_min)
        return SequentialLR(optimizer, schedulers=[s1, s2], milestones=[warmup])

    def training_step(self, modules, views, device, y=None):
        if isinstance(views, (list, tuple)):
            assert self.cfg.get("sigreg_at", "proj") == "proj" and \
                not self.cfg.get("embed_calib", False), "mc control supports the default path only"
            g, l = views
            N, Vg = g.shape[:2]
            V = Vg + l.shape[1]
            sig_at = "proj"
            emb = torch.cat(
                [modules["encoder"](g.flatten(0, 1)).reshape(N, Vg, -1),
                 modules["encoder"](l.flatten(0, 1)).reshape(N, V - Vg, -1)],
                1).flatten(0, 1)
        else:
            if self.cfg.get("embed_calib", False) and not getattr(self, "_calibrated", False):
                with torch.no_grad():
                    emb0 = modules["encoder"](views.flatten(0, 1)).float()
                    mu, sd = emb0.mean(0), emb0.std(0).clamp_min(1e-6)
                    head = modules["encoder"].head
                    head.weight.div_(sd.unsqueeze(1))
                    head.bias.copy_((head.bias - mu) / sd)
                self._calibrated = True
            N, V = views.shape[:2]
            sig_at = self.cfg.get("sigreg_at", "proj")
            x = views.flatten(0, 1)
            if sig_at == "cls":
                feats = modules["encoder"].forward_features(x)
                cls = feats[:, 0]
                emb = modules["encoder"].forward_head(feats)
            else:
                emb = modules["encoder"](x)
        proj = modules["projector"](emb).reshape(N, V, -1).transpose(0, 1)
        if self.cfg.get("mc_form") == "lightly":
            inv_loss = (proj[:Vg].mean(0) - proj[Vg:]).square().mean()
            proj = proj[Vg:]
        else:
            inv_loss = (proj.mean(0) - proj).square().mean()
        sig_in = {"proj": proj,
                  "embed": emb.reshape(N, V, -1).transpose(0, 1),
                  "cls": cls.reshape(N, V, -1).transpose(0, 1) if sig_at == "cls" else None}[sig_at]
        if self.cfg.get("floor", "sigreg") == "sacreg":
            reg_key, reg_loss = "moment_kl", self.floor(sig_in)
        else:
            reg_key, reg_loss = "sigreg", self.sigreg.to(device)(sig_in)
        loss = reg_loss * self.cfg.lamb + inv_loss * (1 - self.cfg.lamb)
        terms = {reg_key: reg_loss, "inv": inv_loss}
        h_reg = self.cfg.get("h_reg")
        if h_reg and self._epoch >= self.cfg.get("h_start_ep", 0):
            emb_in = emb.reshape(N, V, -1).transpose(0, 1)
            h_loss = {"sigreg": lambda: self.sigreg.to(device)(emb_in),
                      "sacreg": lambda: self.floor(emb_in),
                      "sigreg_std": lambda: self.sigreg_std.to(device)(emb_in),
                      "sigreg_t": lambda: self.sigreg_t.to(device)(emb_in)}[h_reg]()
            loss = loss + self.cfg.h_lamb * h_loss
            terms[H_KEYS[h_reg]] = h_loss
        h_inv = self.cfg.get("h_inv", 0.0)
        if h_inv:
            emb_v = emb.reshape(N, V, -1).transpose(0, 1)
            hi_loss = (emb_v.mean(0) - emb_v).square().mean()
            loss = loss + h_inv * hi_loss
            terms["h_inv"] = hi_loss
        return ({"loss": loss, **terms}, emb.detach(), V)

    @torch.inference_mode()
    def eval_features(self, modules, x, device):
        return modules["encoder"](x)

    def probe_dim(self):
        return self.cfg.emb_dim or self._probe_dim

    def extras(self):
        return {"embed_calibrated": getattr(self, "_calibrated", False)}

    def load_extras(self, extras):
        self._calibrated = extras.get("embed_calibrated", False)
