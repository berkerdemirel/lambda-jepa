"""FloorSSL — the house method, independent class (Berker 2026-07-19: "an independent class
for our method and we dont rely on vicreg's architecture. i want to see both versions, one
copying vicreg part (expander BN etc) and the other should try the bn free way").

L = w_inv·MSE(z_a, z_b) + w_floor·SpectralConditioner(z pooled) + h_lamb·SpectralConditioner(cls)

The three-term recipe from the E19/E20 lineage: MSE alignment at z; the TWO-SIDED spectral
conditioner as the SOLE anti-collapse at z (destination duty, E19-T2 doses); the same
conditioner at declared h (cls = projector input, D-036) at calibrated share (E19-T1 dose
law). NAMING (D-059): nothing here is floored — the KL term taxes Σ deviations from I in
both directions; "floor" survives only in FROZEN identifiers (method key/run-ids `floorssl`,
cfg keys `w_floor`/`z_floor*`/`h_floor_batch`, logged term keys `moment_kl`/`h_moment_kl`) —
those are provenance/continuity, not claims. head_norm switch:
  "bn"   — vicreg's expander verbatim (Linear-BN-ReLU ×2 + Linear). With seed-0 construction
           this class is BYTE-IDENTICAL in init to the vicreg-class floorssl_hz arms
           (same call order: trunk → head Linears/BNs; SpectralConditioner draws no construction
           RNG) — verified by the migration byte-check (E21 card).
  "none" — the BN-free head (Linear-ReLU ×2 + Linear): the D-043/E19-T1 conduit hypothesis
           at full depth — without the BN firewall the z-floor's conditioning can reach back
           into the trunk. The z-floor's unit-variance demand replaces BN's scale pin AT z
           (the lejepa-measured lazy-head minimum guard); hidden layers keep no pin —
           declared risk, kill-triggers standing (E21 card).
Aug pipeline stays the BYOL pair (aug family is a separate axis; deltas remain loss/arch-only
vs the vicreg-lane controls). Trainer surface mirrors vicreg's (house scheduler, single
param group)."""
import torch
import torch.nn as nn
import torch.nn.functional as F

from sslgap.data import ViewsDataset, byol_pair
from sslgap.methods._common import HingeFloor, SpectralConditioner, house_scheduler, trunk_arch
from sslgap.methods.base import SSLMethod
from sslgap.models.backbones import build_vit_trunk


def floorssl_head(in_dim=384, hidden=2048, out_dim=2048, norm="bn"):
    """norm="bn" reproduces vicreg_expander byte-for-byte (same layer order ⇒ same seed-0
    draws; BN affines init deterministically, so the Linears also byte-match the "none"
    variant's)."""
    if norm == "bn":
        return nn.Sequential(nn.Linear(in_dim, hidden), nn.BatchNorm1d(hidden),
                             nn.ReLU(inplace=True),
                             nn.Linear(hidden, hidden), nn.BatchNorm1d(hidden),
                             nn.ReLU(inplace=True),
                             nn.Linear(hidden, out_dim))
    return nn.Sequential(nn.Linear(in_dim, hidden), nn.ReLU(inplace=True),
                         nn.Linear(hidden, hidden), nn.ReLU(inplace=True),
                         nn.Linear(hidden, out_dim))


class FloorSSL(SSLMethod):
    name = "floorssl"

    def build_modules(self):
        trunk = build_vit_trunk(self.frame.model_name, self.frame.img_size,
                                drop_path_rate=self.cfg.drop_path)
        head = floorssl_head(384, self.cfg.expander_hidden, self.cfg.expander_dim,
                             norm=self.cfg.head_norm)
        # z_floor axis (E21 fix session, D-049): "kl" = the symmetric SpectralConditioner (Sigma=I,
        # the method's identity) | "hinge" = the one-sided HingeFloor (Sigma>=I) — Berker
        # 2026-07-19: hinge VETOED as method ("vicreg with slicing"); diagnostic arm only.
        # h-floor stays symmetric KL — the <=6%-share conditioner is the certified-GOOD
        # regime (E19-T1/E20). The z estimator's slice cannot exceed the space: at
        # expander_dim <= 128 (the D-050 small-z direction) the floor reads the EXACT full
        # covariance — slice sampling noise vanishes by construction; at 2048 this is the
        # unchanged d'=128 slice.
        # h_floor_batch (D-058, vm3; Berker: "you cannot justify such asymmetry"): pooled
        # (default — the E20-certified conditioner regime) | view_mean — symmetric payment
        # with the z-floor. Same estimator co-design as z: view-mean drops the h-floor's n
        # to bs -> h_d_slice=32, drawn as the first-32 sub-frame of the canonical 128-frame
        # (per-step RNG streams stay aligned across payment variants).
        self.cond_h = SpectralConditioner(d_slice=self.cfg.get("h_d_slice") or 128, d_draw=128)
        # z_floor_batch payment axis (D-051; Berker 2026-07-20): "pooled" floors the bs*V
        # batch — per direction it reads across-image + within-image (aug) variance, so aug
        # spread pays the floor and competes with inv over the same quantity (the 2048-d
        # collapse's 92%-aug-paid equilibrium lived in this channel). "view_mean" floors the
        # per-image view means — reading across + within/V: the floor's demand lands on image
        # spread, the aug-payment channel shrinks by V, and within-scatter control rests on
        # inv alone (declared risk, P-vm-B). ESTIMATOR CO-DESIGN (the card's wall): view_mean
        # drops the floor's n from bs*V to bs, so z_d_slice must keep n/d' >= 4 (bs=128 ->
        # 32); d_draw stays the canonical min(128, D) so per-step RNG streams stay aligned
        # across payment variants. h-floor stays pooled cls — single axis; the <=6%-share
        # conditioner is the certified-GOOD regime.
        d_canon = min(128, self.cfg.expander_dim)
        d_z = self.cfg.get("z_d_slice") or d_canon
        self.cond_z = (HingeFloor(d_slice=d_z) if self.cfg.get("z_floor", "kl") == "hinge"
                        else SpectralConditioner(d_slice=d_z, d_draw=d_canon))
        return nn.ModuleDict({"backbone": trunk, "projector": head})

    def arch(self):
        return {"backbone": trunk_arch(self.frame, self.cfg.drop_path),
                "projector": {"class": "sslgap.methods.floorssl.floorssl_head",
                              "kwargs": {"in_dim": 384, "hidden": self.cfg.expander_hidden,
                                         "out_dim": self.cfg.expander_dim,
                                         "norm": self.cfg.head_norm}}}

    def build_train_dataset(self):
        # aug axis (E21 arm 3; Berker 2026-07-19: "lejepa's augmentations are better for our
        # hz run"): "byol" = the asymmetric pair (E19 lineage default); "lejepa" = the
        # V-view symmetric strong-photometric family (ViewsDataset default stack — the E20
        # zoo's biggest-win family). V=4/bs=128 keeps the floor's pooled n = 512.
        if self.cfg.get("aug", "byol") == "lejepa":
            return ViewsDataset(self.frame.dataset, "train", V=self.cfg.get("V", 4),
                                img_size=self.frame.img_size, data_root=self.frame.data_root)
        return ViewsDataset(self.frame.dataset, "train", V=2, img_size=self.frame.img_size,
                            data_root=self.frame.data_root,
                            transforms=byol_pair(self.frame.img_size))

    def param_groups(self, modules):
        return [{"params": [p for m in modules.values() for p in m.parameters()],
                 "lr": self.cfg.lr, "weight_decay": self.cfg.wd}]

    def build_scheduler(self, optimizer, steps_per_epoch, total_steps):
        return house_scheduler(optimizer, steps_per_epoch, total_steps,
                               self.cfg.warmup_ep, self.cfg.eta_min)

    def training_step(self, modules, views, device, y=None):
        N, V = views.shape[:2]
        tok = modules["backbone"].forward_features(views.flatten(0, 1))
        cls = tok[:, 0]
        z = modules["projector"](cls).reshape(N, V, -1)
        # inv = all-pairs mean MSE: the V-generic form that REDUCES EXACTLY to the lineage's
        # pairwise MSE at V=2 (one pair) — no special case, v2 byte-compat preserved.
        inv = sum(F.mse_loss(z[:, u], z[:, w])
                  for u in range(V) for w in range(u + 1, V)) / (V * (V - 1) / 2)
        # term order mirrors the vicreg-class floorssl arms (z-floor drawn before h-floor)
        # so the per-step fresh-frame RNG sequence matches the migrated lineage.
        zin = (z.mean(1) if self.cfg.get("z_floor_batch", "pooled") == "view_mean"
               else z.reshape(N * V, -1))
        # queue_steps (D-064, vm4 "matched vm3"): a detached ring of the last k steps'
        # conditioner inputs widens the moment estimate (view_mean n = bs -> (k+1)*bs) so the
        # slice can stay at the family's canonical d'=128 — the estimator wall (D-051) removed
        # by temporal accumulation instead of bs. Gradient flows only through the current
        # rows; queue re-warms over k steps after every resume (declared).
        q = int(self.cfg.get("queue_steps", 0) or 0)
        if q:
            zq = getattr(self, "_zq", [])
            reg_z = self.cond_z(torch.cat([zin] + zq) if zq else zin)
            self._zq = [zin.detach()] + zq[:q - 1]
        else:
            reg_z = self.cond_z(zin)
        hin = (cls.reshape(N, V, -1).mean(1)
               if self.cfg.get("h_floor_batch", "pooled") == "view_mean" else cls)
        if q:
            hq = getattr(self, "_hq", [])
            reg_h = self.cond_h(torch.cat([hin] + hq) if hq else hin)
            self._hq = [hin.detach()] + hq[:q - 1]
        else:
            reg_h = self.cond_h(hin)
        loss = self.cfg.w_inv * inv + self.cfg.w_floor * reg_z + self.cfg.h_lamb * reg_h
        terms = {"inv": inv, "moment_kl": reg_z, "h_moment_kl": reg_h}
        probe_feats = cls.detach()                    # declared h (projector-input CLS, D-036)
        return ({"loss": loss, **terms}, probe_feats, V)

    @torch.inference_mode()
    def eval_features(self, modules, x, device):
        return modules["backbone"].forward_features(x)[:, 0]

    def probe_dim(self):
        return 384
