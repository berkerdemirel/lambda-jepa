"""LeJEPA — EXACT port of the official minimal recipe (../lejepa/scripts/minimal_imagenette.py,
Balestriero & LeCun; our M1 ground truth: wandb lejepa-reproduce/z2zqw1bs, final test/acc 0.90217
at ep800 with {lamb 0.02, V 4, proj_dim 16, lr 2e-3, bs 256}).

Port-exactness choices (D-011): module CREATION ORDER mirrors the official script (encoder → probe
[trainer] → SIGReg) so the seed-0 init RNG stream matches as closely as a refactor allows; the
encoder is built as timm-ViT(num_classes=512) exactly like the official ViTEncoder and SPLIT into
trunk+embed roles only at checkpoint time; NO grad clipping (official: none); cosine eta_min=1e-3
(official — note: violates the house eta_min≤lr/20 rule BY DESIGN, recorded in D-011); SIGReg slices
are unseeded per step (official); DataLoader persistent_workers=False (official — worker aug RNG
would differ otherwise).

Deviations from the paper (inherited from the official minimal recipe, not ours): single-dataset toy
scale; probe is the online monitor only.
"""
import timm
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from torchvision.ops import MLP

from sslgap.data import ViewsDataset
from sslgap.methods.base import SSLMethod


class SIGReg(nn.Module):
    """Verbatim from the official script (unseeded slices, device-agnostic)."""

    def __init__(self, knots=17, n_slices=256, t_max=3.0):
        super().__init__()
        self.n_slices = n_slices
        t = torch.linspace(0, t_max, knots, dtype=torch.float32)
        dt = t_max / (knots - 1)
        weights = torch.full((knots,), 2 * dt, dtype=torch.float32)
        weights[[0, -1]] = dt
        window = torch.exp(-t.square() / 2.0)
        self.register_buffer("t", t)
        self.register_buffer("phi", window)
        self.register_buffer("weights", weights * window)

    def forward(self, proj):
        A = torch.randn(proj.size(-1), self.n_slices, device=proj.device)
        A = A.div_(A.norm(p=2, dim=0))
        x_t = (proj @ A).unsqueeze(-1) * self.t
        err = (x_t.cos().mean(-3) - self.phi).square() + x_t.sin().mean(-3).square()
        statistic = (err @ self.weights) * proj.size(-2)
        return statistic.mean()


def lejepa_encoder(model_name, img_size, emb_dim=512, drop_path=0.1):
    """The official ViTEncoder backbone half: timm ViT WITH the emb Linear (num_classes=emb_dim)."""
    return timm.create_model(model_name, pretrained=False, num_classes=emb_dim,
                             drop_path_rate=drop_path, img_size=img_size)


def lejepa_projector(proj_dim, emb_dim=512, hidden=2048):
    """The official projector: torchvision MLP with BatchNorm1d."""
    return MLP(emb_dim, [hidden, hidden, proj_dim], norm_layer=nn.BatchNorm1d)


class LeJEPA(SSLMethod):
    name = "lejepa"

    def build_modules(self):
        # creation order mirrors the official script: backbone-with-emb, then projector.
        enc = lejepa_encoder(self.frame.model_name, self.frame.img_size,
                             emb_dim=self.cfg.emb_dim, drop_path=self.cfg.drop_path)
        proj = lejepa_projector(self.cfg.proj_dim, emb_dim=self.cfg.emb_dim)
        self.sigreg = SIGReg()
        return nn.ModuleDict({"encoder": enc, "projector": proj})

    def arch(self):
        return {"encoder": {"class": "sslgap.methods.lejepa.lejepa_encoder",
                            "kwargs": {"model_name": self.frame.model_name,
                                       "img_size": self.frame.img_size,
                                       "emb_dim": self.cfg.emb_dim,
                                       "drop_path": self.cfg.drop_path}},
                "projector": {"class": "sslgap.methods.lejepa.lejepa_projector",
                              "kwargs": {"proj_dim": self.cfg.proj_dim,
                                         "emb_dim": self.cfg.emb_dim}}}

    def build_train_dataset(self):
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

    def training_step(self, modules, views, device):
        N, V = views.shape[:2]
        emb = modules["encoder"](views.flatten(0, 1))                      # [N*V, emb_dim]
        proj = modules["projector"](emb).reshape(N, V, -1).transpose(0, 1)  # [V, N, proj_dim]
        inv_loss = (proj.mean(0) - proj).square().mean()
        sigreg_loss = self.sigreg.to(device)(proj)
        loss = sigreg_loss * self.cfg.lamb + inv_loss * (1 - self.cfg.lamb)
        return ({"loss": loss, "sigreg": sigreg_loss, "inv": inv_loss}, emb.detach())

    @torch.inference_mode()
    def eval_features(self, modules, x, device):
        return modules["encoder"](x)

    def probe_dim(self):
        return self.cfg.emb_dim
