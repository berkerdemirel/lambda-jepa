"""SimCLR (Chen et al. 2020) — toy-rung instance per D-012: canonical loss (NT-Xent, symmetric,
in-batch negatives), canonical head shape (2-layer MLP projector), canonical augs (paper stack),
house AdamW schedule (toy adaptation — LARS enters with the verbatim donor recipes at M1.5).
Donor cross-check: third_party/solo-learn @9187ea3 (methods/simclr.py; temp 0.2 = their IN-100)."""
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
    """z [2N, d] view-major pairs (i <-> i+N). Symmetric InfoNCE with in-batch negatives."""
    z = F.normalize(z, dim=1)
    n2 = z.shape[0]
    sim = (z @ z.T) / temp
    sim.fill_diagonal_(float("-inf"))
    pos = torch.arange(n2, device=z.device).roll(n2 // 2)
    return F.cross_entropy(sim, pos)


def uniformity(x, t=2.0):
    """Wang-Isola (2020) uniformity = log E_{i!=j} exp(-t||xi-xj||^2) on the sphere. SimCLR's OWN
    non-collapse mechanism (negative repulsion) made explicit as a differentiable functional.
    Squared distances via the inner-product identity (no sqrt -> no inf grad on the zero diagonal);
    diagonal excluded by an out-of-place mask (in-place after exp breaks autograd)."""
    x = F.normalize(x, dim=1)
    n = x.size(0)
    sq = (x.pow(2).sum(1, keepdim=True) + x.pow(2).sum(1) - 2 * x @ x.T).clamp_min(0)
    w = torch.exp(-t * sq) * (1 - torch.eye(n, device=x.device))
    return (w.sum() / (n * (n - 1))).log()


def alignment(xa, xb):
    """Wang-Isola alignment = E||xi_a - xi_b||^2 on positive pairs (normalized). SimCLR's OWN
    invariance mechanism (positive attraction)."""
    return (F.normalize(xa, dim=1) - F.normalize(xb, dim=1)).pow(2).sum(1).mean()


class SimCLR(SSLMethod):
    name = "simclr"

    def build_modules(self):
        trunk = build_vit_trunk(self.frame.model_name, self.frame.img_size,
                                drop_path_rate=self.cfg.drop_path)
        proj = simclr_projector(384, self.cfg.proj_hidden, self.cfg.proj_dim)
        if self.cfg.get("h_reg") == "sacreg":     # E20 calibrated zoo floor (no RNG at construction)
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
        z = modules["projector"](h[:, 0]).reshape(N, V, -1)               # loss input: CLS (trained)
        z = torch.cat([z[:, 0], z[:, 1]])                                 # [2N, d], i <-> i+N
        loss = nt_xent(z, self.cfg.temp)
        terms = {"nt_xent": loss}
        # E17 h-pull (D-039): simclr's OWN mechanism at h (declared CLS) — uniformity = its
        # non-collapse (the softmax denominator / negatives), alignment = its invariance (the
        # numerator / positives), as Wang-Isola functionals. Additive on the unchanged NT-Xent;
        # weights are the 10%-pull doses. Both logged for the pull measurement.
        hv = h[:, 0].reshape(N, V, -1)
        if self.cfg.get("h_uniform", 0.0):
            u = uniformity(hv.reshape(N * V, -1), t=self.cfg.get("h_unif_t", 2.0))
            loss = loss + self.cfg.h_uniform * u
            terms["h_uniform"] = u
        if self.cfg.get("h_align", 0.0):
            a = alignment(hv[:, 0], hv[:, 1])
            loss = loss + self.cfg.h_align * a
            terms["h_align"] = a
        # E20 (calibrated zoo floor): additive moment floor at declared h (pooled-view CLS,
        # the floor's own batch-moment convention); NT-Xent untouched. Dose = per-method
        # calibrated share (E20 card).
        if self.cfg.get("h_reg") == "sacreg":
            h_loss = self.floor(h[:, 0])
            loss = loss + self.cfg.h_lamb * h_loss
            terms["h_moment_kl"] = h_loss
        probe_feats = h[:, 0].detach()               # monitor = declared h (projector-input CLS, D-036); image-major
        return ({"loss": loss, **terms}, probe_feats, V)

    @torch.inference_mode()
    def eval_features(self, modules, x, device):
        return modules["backbone"].forward_features(x)[:, 0]              # declared h (projector-input CLS, D-036)

    def probe_dim(self):
        return 384
