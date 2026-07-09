"""SimCLR (Chen et al. 2020) — toy-rung instance per D-012: canonical loss (NT-Xent, symmetric,
in-batch negatives), canonical head shape (2-layer MLP projector), canonical augs (paper stack),
house AdamW schedule (toy adaptation — LARS enters with the verbatim donor recipes at M1.5).
Donor cross-check: third_party/solo-learn @9187ea3 (methods/simclr.py; temp 0.2 = their IN-100)."""
import torch
import torch.nn as nn
import torch.nn.functional as F

from sslgap.data import ViewsDataset, simclr_stack
from sslgap.methods._common import house_scheduler, trunk_arch
from sslgap.methods.base import SSLMethod
from sslgap.models.backbones import build_vit_trunk


def simclr_projector(in_dim=384, hidden=2048, out_dim=256):
    return nn.Sequential(nn.Linear(in_dim, hidden), nn.ReLU(inplace=True),
                         nn.Linear(hidden, out_dim))


def nt_xent(z, temp):
    """z [2N, d] view-major pairs (i <-> i+N). Symmetric InfoNCE with in-batch negatives."""
    z = F.normalize(z, dim=1)
    n2 = z.shape[0]
    sim = (z @ z.T) / temp
    sim.fill_diagonal_(float("-inf"))
    pos = torch.arange(n2, device=z.device).roll(n2 // 2)
    return F.cross_entropy(sim, pos)


class SimCLR(SSLMethod):
    name = "simclr"

    def build_modules(self):
        trunk = build_vit_trunk(self.frame.model_name, self.frame.img_size,
                                drop_path_rate=self.cfg.drop_path)
        proj = simclr_projector(384, self.cfg.proj_hidden, self.cfg.proj_dim)
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
        z = modules["projector"](h[:, 0]).reshape(N, V, -1)               # loss input: CLS (trained)
        z = torch.cat([z[:, 0], z[:, 1]])                                 # [2N, d], i <-> i+N
        loss = nt_xent(z, self.cfg.temp)
        probe_feats = h[:, 1:].mean(1).detach()      # monitor = audited h (trunk-GAP, F1); image-major
        return ({"loss": loss, "nt_xent": loss}, probe_feats, V)

    @torch.inference_mode()
    def eval_features(self, modules, x, device):
        return modules["backbone"].forward_features(x)[:, 1:].mean(1)     # audited h (GAP)

    def probe_dim(self):
        return 384
