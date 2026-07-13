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

Arm machinery riding on the port (all off by default; defaults reproduce the port byte-exactly):
E10 (sigreg_at / proj_depth / embed_calib, D-016/D-021) and E12 (spec_norm on the projector,
floor=moment → MomentFloor, D-026 — see docs/experiments/E12_moment_floor.md).
"""
import timm
import torch
import torch.nn as nn
from torch.nn.utils.parametrizations import spectral_norm
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from torchvision.ops import MLP

from sslgap.data import ViewsDataset
from sslgap.methods._common import MomentFloor
from sslgap.methods.base import SSLMethod


class SIGReg(nn.Module):
    """Verbatim from the official script (unseeded slices, device-agnostic). standardize=True
    (E12 F-wave arm f4, D-027) z-scores each slice over the batch before the CF distance —
    kills the moment channel, isolating the shape/anti-CLT (cluster) response (R4c(d)'s listed
    intervention; framework-E7(i)); dead-slice division guarded by the std clamp (gradient
    vanishes at exact death — the term sees degeneracy but is not its fixer). Default path is
    numerically identical to the port."""

    def __init__(self, knots=17, n_slices=256, t_max=3.0, standardize=False):
        super().__init__()
        self.n_slices = n_slices
        self.standardize = standardize
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
        s = proj @ A
        if self.standardize:
            s = (s - s.mean(-2, keepdim=True)) / s.std(-2, keepdim=True).clamp_min(1e-4)
        x_t = s.unsqueeze(-1) * self.t
        err = (x_t.cos().mean(-3) - self.phi).square() + x_t.sin().mean(-3).square()
        statistic = (err @ self.weights) * proj.size(-2)
        return statistic.mean()


def lejepa_encoder(model_name, img_size, emb_dim=512, drop_path=0.1):
    """The official ViTEncoder backbone half: timm ViT WITH the emb Linear (num_classes=emb_dim)."""
    return timm.create_model(model_name, pretrained=False, num_classes=emb_dim,
                             drop_path_rate=drop_path, img_size=img_size)


def lejepa_projector(proj_dim, emb_dim=512, hidden=2048, depth=3, spec_norm=False):
    """The official projector: torchvision MLP with BatchNorm1d. depth counts trainable layers
    (E10 arm D): 3 = official [hidden, hidden, proj_dim]; 0 = Identity (whole loss on the
    embedding — the no-buffer arm); 1..2 = shallower MLPs ending at proj_dim. spec_norm (E12 p1,
    D-026) parametrizes each Linear with spectral normalization — bounds the Linears' spectral
    norms only; BN affines stay unnormalized (declared caveat on the E12 card, σ_min/eff-rank of
    the fitted head watched via D-015)."""
    if depth == 0:
        return nn.Identity()
    mlp = MLP(emb_dim, [hidden] * (depth - 1) + [proj_dim], norm_layer=nn.BatchNorm1d)
    if spec_norm:
        for m in mlp.modules():
            if isinstance(m, nn.Linear):
                spectral_norm(m)
    return mlp


# MomentFloor moved to sslgap/methods/_common.py (E12 cross-method arms, 2026-07-12);
# imported at the top so all existing references keep working.

class DiagMomentFloor(nn.Module):
    """E12 F-wave arm f5 (D-027): per-dim Gaussian-moment calibration, full-D, no joint term.
    0.5·(mean(mu²) + mean(var − 1 − log var)). The AFFINE-ABSORBABLE calibration cell: the
    encoder can comply exactly via a diagonal rescale + bias of the emb Linear (information-
    free), so any probe tax here measures optimization interference, not representational
    damage; A3 − f5 isolates the whitening/decorrelation component of the full-KL floor.
    Structure hidden in correlations is allowed BY DESIGN; per-dim log barrier keeps per-dim
    anti-collapse. Subspace-free (also the estimator-noise control vs the sliced KL)."""

    def forward(self, x):
        with torch.autocast(x.device.type, enabled=False):
            x = x.reshape(-1, x.size(-1)).float()
            mu = x.mean(0)
            var = x.var(0).clamp_min(1e-8)
            return 0.5 * (mu.square().mean() + (var - 1 - var.log()).mean())


class SpectralFloor(nn.Module):
    """E12 F-wave arm f3 (D-027): one-sided anti-degeneracy floor — the refined-thesis
    candidate after the M2 read ('anisotropy IS semantics'). On a fresh random d'-subspace:
    barrier ONLY on relative eigenvalue deficiency (λ̃ = λ·d'/tr < tau); anisotropy above the
    floor is untouched. Scale-inflation gaming is closed by the scalar trace pin; 'junk-dim'
    compliance is allowed by design (the certificate is spectral non-degeneracy, not content).
    tau/d'/eps declared fixed on the card."""

    def __init__(self, d_slice=128, tau=0.01, eps=1e-6):
        super().__init__()
        self.d_slice, self.tau, self.eps = d_slice, tau, eps

    def forward(self, x):
        with torch.autocast(x.device.type, enabled=False):
            x = x.reshape(-1, x.size(-1)).float()
            Q, _ = torch.linalg.qr(torch.randn(x.size(1), self.d_slice, device=x.device))
            p = x @ Q
            mu = p.mean(0)
            pc = p - mu
            cov = pc.T @ pc / (p.size(0) - 1) + self.eps * torch.eye(self.d_slice, device=x.device)
            lam = torch.linalg.eigvalsh(cov)
            tr = lam.sum().clamp_min(1e-8)
            lam_n = (lam * self.d_slice / tr).clamp_min(1e-12)
            rank_t = torch.relu((self.tau / lam_n).log()).mean()
            return rank_t + (tr / self.d_slice - 1).square() + mu.square().mean()


# h-side regularizer routing (E12 §Amendment + F-wave): cfg.h_reg -> wandb term key
H_KEYS = {"sigreg": "h_sigreg", "moment": "h_moment_kl", "sigreg_std": "h_sigreg_std",
          "moment_diag": "h_moment_diag", "spec_floor": "h_spec_floor"}


class LeJEPA(SSLMethod):
    name = "lejepa"
    _epoch = 0                    # tracked via on_epoch_start; gates h_start_ep (F-wave f6)

    def on_epoch_start(self, modules, epoch):
        self._epoch = epoch

    def build_modules(self):
        # creation order mirrors the official script: backbone-with-emb, then projector.
        # (spec_norm and MomentFloor consume no RNG at construction — init streams unshifted.)
        enc = lejepa_encoder(self.frame.model_name, self.frame.img_size,
                             emb_dim=self.cfg.emb_dim, drop_path=self.cfg.drop_path)
        proj = lejepa_projector(self.cfg.proj_dim, emb_dim=self.cfg.emb_dim,
                                depth=self.cfg.get("proj_depth", 3),
                                spec_norm=self.cfg.get("spec_norm", False))
        self.sigreg = SIGReg()
        if self.cfg.get("floor", "sigreg") == "moment" or self.cfg.get("h_reg") == "moment":
            self.floor = MomentFloor()
        h_reg = self.cfg.get("h_reg")
        if h_reg == "sigreg_std":
            self.sigreg_std = SIGReg(standardize=True)
        elif h_reg == "moment_diag":
            self.diag_floor = DiagMomentFloor()
        elif h_reg == "spec_floor":
            self.spec_floor = SpectralFloor()
        return nn.ModuleDict({"encoder": enc, "projector": proj})

    def arch(self):
        # depth recorded only when non-default so pre-existing checkpoints' arch dicts (and any
        # in-flight requeue resume asserts) stay byte-identical; rebuild defaults to depth=3.
        proj_kwargs = {"proj_dim": self.cfg.proj_dim, "emb_dim": self.cfg.emb_dim}
        if self.cfg.get("proj_depth", 3) != 3:
            proj_kwargs["depth"] = self.cfg.proj_depth
        if self.cfg.get("spec_norm", False):
            proj_kwargs["spec_norm"] = True
        return {"encoder": {"class": "sslgap.methods.lejepa.lejepa_encoder",
                            "kwargs": {"model_name": self.frame.model_name,
                                       "img_size": self.frame.img_size,
                                       "emb_dim": self.cfg.emb_dim,
                                       "drop_path": self.cfg.drop_path}},
                "projector": {"class": "sslgap.methods.lejepa.lejepa_projector",
                              "kwargs": proj_kwargs}}

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

    def training_step(self, modules, views, device, y=None):
        # E10 arms (D-016, card §SIGReg-on-h): sigreg_at ∈ {proj (A, default), embed (B), cls (C)}
        # routes ONLY the SIGReg input; invariance stays at proj.out and modules are identical, so
        # arch/extraction/probing are unchanged across arms.
        # embed_calib (E10 arm B‴/e10Bi): one-shot data-dependent init of the emb Linear — fold
        # first-batch per-dim (mu, sigma) into (W, b) so the embed starts unit-scale/zero-mean.
        # The timm head init (trunc_normal std .02, NOT fan-in-scaled) leaves the embed ~10x
        # under-scaled; sigreg's unbuffered opening rescale of the trunk is the measured collapse
        # trigger (grad-share diag; E10 card amendment). Loss untouched; the calibrated weights
        # persist through checkpoints, the extras flag guards re-entry on resume.
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
            feats = modules["encoder"].forward_features(x)                 # [N*V, T, trunk_dim]
            cls = feats[:, 0]
            emb = modules["encoder"].forward_head(feats)                   # [N*V, emb_dim]
        else:
            emb = modules["encoder"](x)                                    # [N*V, emb_dim]
        proj = modules["projector"](emb).reshape(N, V, -1).transpose(0, 1)  # [V, N, proj_dim]
        inv_loss = (proj.mean(0) - proj).square().mean()
        sig_in = {"proj": proj,
                  "embed": emb.reshape(N, V, -1).transpose(0, 1),
                  "cls": cls.reshape(N, V, -1).transpose(0, 1) if sig_at == "cls" else None}[sig_at]
        # E12 (D-026): floor ∈ {sigreg (default), moment} picks the regularizer applied to sig_in
        # — same routing, different term. "moment" = MomentFloor (arm A3, the thesis cell).
        if self.cfg.get("floor", "sigreg") == "moment":
            reg_key, reg_loss = "moment_kl", self.floor(sig_in)
        else:
            reg_key, reg_loss = "sigreg", self.sigreg.to(device)(sig_in)
        loss = reg_loss * self.cfg.lamb + inv_loss * (1 - self.cfg.lamb)
        terms = {reg_key: reg_loss, "inv": inv_loss}
        # E12 amendment (D-026) + F-wave (D-027): h_reg picks an ADDITIVE regularizer at the
        # embedding, weight h_lamb; h_start_ep (f6) delays enablement (§4-p2's burn-in variant).
        # The shipped z-side term always stays: without its scale pin at proj.out, inv
        # (scale-dependent) admits a lazy-projector minimum — measured on the e12a{2,3} smokes.
        h_reg = self.cfg.get("h_reg")
        if h_reg and self._epoch >= self.cfg.get("h_start_ep", 0):
            emb_in = emb.reshape(N, V, -1).transpose(0, 1)
            h_loss = {"sigreg": lambda: self.sigreg.to(device)(emb_in),
                      "moment": lambda: self.floor(emb_in),
                      "sigreg_std": lambda: self.sigreg_std.to(device)(emb_in),
                      "moment_diag": lambda: self.diag_floor(emb_in),
                      "spec_floor": lambda: self.spec_floor(emb_in)}[h_reg]()
            loss = loss + self.cfg.h_lamb * h_loss
            terms[H_KEYS[h_reg]] = h_loss
        # H-wave (D-035): tiny ADDITIVE view-invariance pull at the embedding itself (H3: move
        # invariance work out of the projector MLP). Same functional form as the proj-space inv
        # term; weight declared (not equal-pull measured) — per-term logging watches its share.
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
        return self.cfg.emb_dim

    def extras(self):
        return {"embed_calibrated": getattr(self, "_calibrated", False)}

    def load_extras(self, extras):
        self._calibrated = extras.get("embed_calibrated", False)
