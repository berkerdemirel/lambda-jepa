"""SimCLR as released, with the optional SACReg term at the backbone (h_reg=sacreg)."""
import torch
import torch.nn as nn
import torch.nn.functional as F

from sslgap.data import ViewsDataset, simclr_stack
from sslgap.methods._common import SACReg, house_scheduler, trunk_arch
from sslgap.methods.base import SSLMethod
from sslgap.models.backbones import build_vit_trunk

def simclr_projector(in_dim=384, hidden=2048, out_dim=256):
    return nn.Sequential(nn.Linear(in_dim, hidden), nn.ReLU(inplace=True),
                         nn.Linear(hidden, out_dim))

def nt_xent(z, temp):
    z = F.normalize(z, dim=1)
    n2 = z.shape[0]
    sim = (z @ z.T) / temp
    sim.fill_diagonal_(float("-inf"))
    pos = torch.arange(n2, device=z.device).roll(n2 // 2)
    return F.cross_entropy(sim, pos)

def uniformity(x, t=2.0):
    x = F.normalize(x, dim=1)
    n = x.size(0)
    sq = (x.pow(2).sum(1, keepdim=True) + x.pow(2).sum(1) - 2 * x @ x.T).clamp_min(0)
    w = torch.exp(-t * sq) * (1 - torch.eye(n, device=x.device))
    return (w.sum() / (n * (n - 1))).log()

def alignment(xa, xb):
    return (F.normalize(xa, dim=1) - F.normalize(xb, dim=1)).pow(2).sum(1).mean()

class SimCLR(SSLMethod):
    name = "simclr"

    def build_modules(self):
        trunk = build_vit_trunk(self.frame.model_name, self.frame.img_size,
                                drop_path_rate=self.cfg.drop_path)
        proj = simclr_projector(384, self.cfg.proj_hidden, self.cfg.proj_dim)
        if self.cfg.get("h_reg") == "sacreg":
            self.floor = SACReg()
        return nn.ModuleDict({"backbone": trunk, "projector": proj})

    def arch(self):
        return {"backbone": trunk_arch(self.frame, self.cfg.drop_path),
                "projector": {"class": "sslgap.methods.simclr.simclr_projector",
                              "kwargs": {"in_dim": 384, "hidden": self.cfg.proj_hidden,
                                         "out_dim": self.cfg.proj_dim}}}

    def build_train_dataset(self):
        return ViewsDataset(self.frame.dataset, "train", V=2, img_size=self.frame.img_size,
                            data_root=self.frame.data_root, aug=simclr_stack(self.frame.img_size))

    def param_groups(self, modules):
        return [{"params": [p for m in modules.values() for p in m.parameters()],
                 "lr": self.cfg.lr, "weight_decay": self.cfg.wd}]

    def build_scheduler(self, optimizer, steps_per_epoch, total_steps):
        return house_scheduler(optimizer, steps_per_epoch, total_steps,
                               self.cfg.warmup_ep, self.cfg.eta_min)

    def training_step(self, modules, views, device, y=None):
        N, V = views.shape[:2]
        h = modules["backbone"].forward_features(views.flatten(0, 1))
        z = modules["projector"](h[:, 0]).reshape(N, V, -1)
        z = torch.cat([z[:, 0], z[:, 1]])
        loss = nt_xent(z, self.cfg.temp)
        terms = {"nt_xent": loss}
        hv = h[:, 0].reshape(N, V, -1)
        if self.cfg.get("h_uniform", 0.0):
            u = uniformity(hv.reshape(N * V, -1), t=self.cfg.get("h_unif_t", 2.0))
            loss = loss + self.cfg.h_uniform * u
            terms["h_uniform"] = u
        if self.cfg.get("h_align", 0.0):
            a = alignment(hv[:, 0], hv[:, 1])
            loss = loss + self.cfg.h_align * a
            terms["h_align"] = a
        if self.cfg.get("h_reg") == "sacreg":
            h_loss = self.floor(h[:, 0])
            loss = loss + self.cfg.h_lamb * h_loss
            terms["h_moment_kl"] = h_loss
        probe_feats = h[:, 0].detach()
        return ({"loss": loss, **terms}, probe_feats, V)

    @torch.inference_mode()
    def eval_features(self, modules, x, device):
        return modules["backbone"].forward_features(x)[:, 0]

    def probe_dim(self):
        return 384
