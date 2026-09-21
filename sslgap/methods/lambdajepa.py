"""LambdaJEPA — the house method, independent class (Berker 2026-07-19: "an independent class
for our method and we dont rely on vicreg's architecture. i want to see both versions, one
copying vicreg part (expander BN etc) and the other should try the bn free way").

L = w_inv·MSE(z_a, z_b) + w_floor·SACReg(z pooled) + h_lamb·SACReg(cls)

The three-term recipe from the E19/E20 lineage: MSE alignment at z; the TWO-SIDED spectral
conditioner as the SOLE anti-collapse at z (destination duty, E19-T2 doses); the same
conditioner at declared h (cls = projector input, D-036) at calibrated share (E19-T1 dose
law). NAMING (D-059): nothing here is floored — the KL term taxes Σ deviations from I in
both directions; "floor" survives only in FROZEN identifiers (pre-2026-09-21 run-ids `floorssl`,
cfg keys `w_floor`/`z_floor*`/`h_floor_batch`, logged term keys `moment_kl`/`h_moment_kl`) —
those are provenance/continuity, not claims. head_norm switch:
  "bn"   — vicreg's expander verbatim (Linear-BN-ReLU ×2 + Linear). With seed-0 construction
           this class is BYTE-IDENTICAL in init to the vicreg-class floorssl_hz arms
           (same call order: trunk → head Linears/BNs; SACReg draws no construction
           RNG) — verified by the migration byte-check (E21 card).
  "none" — the BN-free head (Linear-ReLU ×2 + Linear): the D-043/E19-T1 conduit hypothesis
           at full depth — without the BN firewall the z-floor's conditioning can reach back
           into the trunk. The z-floor's unit-variance demand replaces BN's scale pin AT z
           (the lejepa-measured lazy-head minimum guard); hidden layers keep no pin —
           declared risk, kill-triggers standing (E21 card).
Aug pipeline stays the BYOL pair (aug family is a separate axis; deltas remain loss/arch-only
vs the vicreg-lane controls). Trainer surface mirrors vicreg's (house scheduler, single
param group)."""
import copy
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from sslgap.data import (LejepaMultiCropDataset, LightlyLejepaMultiCropDataset,
                         ViewsDataset, byol_pair)
from sslgap.methods._common import HingeFloor, SACReg, house_scheduler, trunk_arch
from sslgap.methods.base import SSLMethod
from sslgap.models.backbones import build_vit_trunk
from sslgap.models.heads import BottleneckStage, ResBlock


def lambdajepa_head(in_dim=384, hidden=2048, out_dim=2048, norm="bn"):
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


def lambdajepa_res_head(in_dim=384, hidden=2048, out_dim=2048, depth=2, norm="none"):
    """REJECTED E23 rev1 ladder (card §Launch log 2026-07-29: the skip = an always-open
    linear h→z path, depth cannot modulate leakage; 6/6 collapsed). Kept for checkpoint
    assembly only — build_modules no longer offers it."""
    first, last = nn.Linear(in_dim, hidden), nn.Linear(hidden, out_dim)
    return nn.Sequential(first, *[ResBlock(hidden, norm) for _ in range(depth)], last)


def lambdajepa_stage_head(in_dim=384, hidden=2048, out_dim=256, depth=2, m=256):
    """REJECTED E23 rev2 ladder (card §Launch log 2026-07-30: the BARE adapter left the
    2048-d entry unpinned — collapsed on the healthy byol lane too, 9/9). Kept for
    checkpoint assembly only — build_modules no longer offers it."""
    return nn.Sequential(nn.Linear(in_dim, hidden),
                         *[BottleneckStage(hidden, m) for _ in range(depth)],
                         nn.Linear(hidden, out_dim))


def lambdajepa_ladder_head(in_dim=384, hidden=2048, out_dim=256, depth=2, width=None):
    """E23 rev3 ladder (Berker's careful-mode): the LEGACY expander anatomy itself with
    `depth` hidden BN-ReLU layers — depth=2, width=hidden is BYTE-IDENTICAL to
    lambdajepa_head(norm="bn"), so the ladder grows out of the certified-healthy cell in
    both directions. Every hidden Linear is BN-pinned (the invariant every healthy
    floorssl head shares; both rejected scaffolds broke it). depth=0 = the bare linear
    projector (folklore anchor, declared-risk). `width` = middle-layer width, the fine
    capacity dial at fixed depth (width=hidden ≡ legacy shapes)."""
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
    # E24 (D-070): term -> cfg-weight map for the realized w·g share logger (train.py);
    # trunk-only g_enc convention (the standing pull instrument's).
    PULL_W = {"inv": "w_inv", "moment_kl": "w_floor", "h_moment_kl": "h_lamb"}

    def build_modules(self):
        # Recipe v2 (D-079a): multicrop feeds 96-px locals through the same trunk ->
        # dynamic_img_size (pos-embed interpolation; @224 forward parity-asserted by
        # e27_selftest). _dim = trunk width, so the arch axis (384/768/1024) reaches the
        # head/probe dims without hardcoding — byte-identical at ViT-S.
        self._mc = self.cfg.get("aug") in ("lejepa_mc", "lightly_mc")
        trunk = build_vit_trunk(self.frame.model_name, self.frame.img_size,
                                dynamic_img_size=self._mc,
                                drop_path_rate=self.cfg.drop_path)
        self._dim = trunk.num_features
        # grad_ckpt (E24-vm diagnostics): trunk activation checkpointing — memory for
        # ~30% compute, same math. bs=512 V=4 @128 (the no-ring n/d'=4 cell) needs ~78GB
        # flat; checkpointed it fits any 80GB card with margin. Default off.
        if self.cfg.get("grad_ckpt"):
            trunk.set_grad_checkpointing()
        # head_layers (E23 rev3): the legacy-anatomy ladder — depth as the leakage dial,
        # head_width the fine dial; (2, hidden) ≡ the legacy bn expander byte-identically;
        # null = the legacy expander path.
        k = self.cfg.get("head_layers")
        head = (lambdajepa_ladder_head(self._dim, self.cfg.expander_hidden,
                                     self.cfg.expander_dim,
                                     depth=k, width=self.cfg.get("head_width"))
                if k is not None
                else lambdajepa_head(self._dim, self.cfg.expander_hidden,
                                   self.cfg.expander_dim, norm=self.cfg.head_norm))
        # z_floor axis (E21 fix session, D-049): "kl" = the symmetric SACReg (Sigma=I,
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
        # floor_shrink (D-073): OAS estimator variant on BOTH taps (symmetric payment,
        # D-058); null = legacy exact-scatter estimators, byte-identical.
        shr = self.cfg.get("floor_shrink")
        # h-slice (Recipe v2, D-079a): d' scales with trunk width at held ratio (S 128,
        # B 256, L 384 — Berker: "make the slice ratio 384 for L"); d_draw stays the
        # canonical 128-frame while d' <= 128 (D-058 sub-frame discipline) and becomes
        # the full d'-frame above it. OAS = the sanctioned B/L fallback if ring rows
        # stale ("we can fallback to oas in case features move too fast").
        d_h = self.cfg.get("h_d_slice") or 128
        self.cond_h = SACReg(d_slice=d_h, d_draw=max(128, d_h), shrink=shr)
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
                        else SACReg(d_slice=d_z, d_draw=d_canon, shrink=shr))
        mods = nn.ModuleDict({"backbone": trunk, "projector": head})
        # swa (D-095; Berker 2026-08-10 "implement swa, as described in lejepa"): the
        # paper's entire spec is "we apply SWA on the encoder producing mu in Eq. (6)"
        # (Izmailov equal-weight averaging; their je.py unpublished). mu lives in z space,
        # so the averaged twin is the full z path: trunk + projector. deepcopy AFTER
        # student init draws no RNG — the student stays byte-identical to its parent lane.
        # Twin is grad-free and eval (the house sslx-control convention: no drop_path
        # stochasticity in targets); its BN buffers track the student's (copied at each
        # update). The anchor SET stays the lane's own (all-views mean), so SWA is the
        # single delta vs the parent; their V_g-anchored mu is the separate inv-anchor axis.
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
        # aug axis (E21 arm 3; Berker 2026-07-19: "lejepa's augmentations are better for our
        # hz run"): "byol" = the asymmetric pair (E19 lineage default); "lejepa" = the
        # V-view symmetric strong-photometric family (ViewsDataset default stack — the E20
        # zoo's biggest-win family). V=4/bs=128 keeps the floor's pooled n = 512.
        # "lejepa_mc" (Recipe v2, D-079a): the LeJEPA-recommended multicrop — Vg globals +
        # Vl locals under the same symmetric family; the E27 in1k frame point.
        # "lightly_mc" (D-095, Berker: "the new runs should do the locals accordingly …
        # use lightly's lejepa augmentations"): the exact Lightly view stack behind their
        # 64.0 row — 2g@224 (0.3,1) + 6l@96 (0.05,0.3), 0.4-family jitter, true-p .2
        # solarize on global-2 only, bicubic — under OUR loss: the frame-matched (and
        # FLOP-matched) A/B against the e27 lejepa reproduction.
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
        # mlp_wd (E23, D-067): the projector in its own wd group — head expressivity dialed
        # independently of backbone wd (meaningful in the BN-free lane, where weight scale is
        # not gauge). null = the legacy single group.
        mwd = self.cfg.get("mlp_wd")
        if mwd is None:
            # student modules only (byte-identical legacy: the dict held exactly these
            # two roles before the swa twin existed; the twin is never optimized)
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
        """queue_steps (D-064, vm4 "matched vm3"): a detached ring of the last q steps'
        conditioner inputs widens the moment estimate (view_mean n = bs -> (q+1)*bs) so the
        slice keeps its d' — the estimator wall (D-051) removed by temporal accumulation.
        Gradient flows only through the current rows; the ring re-warms over q steps after
        every resume (declared). Per-tap since Recipe v2 (the h ring grows with d'_h while z
        stays at the certified 3); cat order and detach timing byte-match the pre-refactor
        inline code (e27_selftest regression)."""
        if not q:
            return x
        buf = getattr(self, attr, [])
        out = torch.cat([x] + buf) if buf else x
        setattr(self, attr, [x.detach()] + buf[:q - 1])
        return out

    def _swa_mu(self, modules, views):
        """SWA anchor (D-095): Eq. (7)'s mu produced by the averaged twin — the twin
        forwards the SAME views the student sees and the anchor is its per-image z-mean,
        no grad. Twin stays eval (train_mode): deterministic targets, BN on the buffers
        copied from the student at each update."""
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
        # SWA update (D-095): swa="uniform" = equal-weight running average over steps
        # (Izmailov form, the paper's literal citation; no cadence exists to import —
        # je.py unpublished). D-098 postmortem: uniform-from-step-0 anchors early
        # training to near-init weights (both swa lanes ran −14/−15 under their bases
        # with pathological h_moment_kl) → swa="ema" = the lineage teacher (DINO
        # convention: tau cosine swa_tau→1 over total steps), declared deviation from
        # the paper's literal SWA. BN buffers are COPIED from the student either way
        # (running stats are already temporal averages).
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
            # Recipe v2 multicrop (D-079a): Vg globals + Vl locals through the one trunk
            # (dynamic pos-embed); inv = the LeJEPA view-to-mean form over ALL views —
            # exactly all-pairs x (V-1)/2V at any fixed view set, the constant absorbed by
            # the dose procedure; conditioner stream = per-image mean over all views, the
            # inv fixed point (Berker 2026-08-06: "conditioner stream should use mean of
            # all"). view_mean payment at both taps is the only anatomy here BY DESIGN.
            #
            # cond_stream (D-084, the card's pre-registered reserve invoked on the ep10
            # evidence): "all" (default, byte-identical legacy) | "globals" = the
            # conditioner reads the mean over the Vg GLOBALS only, while inv keeps every
            # view. Rationale: inv's job is to make all views agree — locals belong there.
            # The conditioner's job is to shape the distribution of per-image CENTERS, and
            # a mean taken over aggressive locals estimates that center with crop noise:
            # measured at ep10, the mc cell's view-mean z retains 63% of the pooled
            # variance (trace/d .170 vs target 1.0) against 91% for the mild-local cell
            # and 84% for the V=4 d256vm4 lane — the conditioner was fighting an
            # averaging artifact, not the encoder. Globals-only also makes the stream
            # V-INDEPENDENT (always a mean over 2 views at fixed 224 geometry), which is
            # what makes doses comparable across aug families and across arches.
            # Fresh n stays bs, so the ring/d' co-design (PROTOCOL §6.10) is unchanged.
            # cond_stream="grouped" (D-087 PROPOSED, wave 4): the conditioner reads TWO
            # scale-homogeneous streams — the per-image mean over the Vg globals AND the
            # per-image mean over the Vl locals — one KL each, averaged into the term.
            # Rationale (card §(g) option 4): the all-views mean is a malformed
            # conditioning object under multicrop (locals are deliberately partial-content
            # views, not exchangeable identity estimates — mc's mean carried trace/d .17,
            # §(d)); globals-only blinds the conditioner to the aug axis and lets inv
            # collapse local content (wave 3, D-086). Grouped restores exchangeability
            # WITHIN each stream while keeping locals constrained — the structural
            # analogue of the per-view SIGReg under which the e27lej control is healthy
            # on this exact aug. Each stream keeps fresh n = bs and its own ring.
            g, l = views
            N, Vg = g.shape[:2]
            V = Vg + l.shape[1]
            cls = torch.cat(
                [modules["backbone"].forward_features(g.flatten(0, 1))[:, 0]
                 .reshape(N, Vg, -1),
                 modules["backbone"].forward_features(l.flatten(0, 1))[:, 0]
                 .reshape(N, V - Vg, -1)], 1)
            z = modules["projector"](cls.flatten(0, 1)).reshape(N, V, -1)
            # swa (D-095): the anchor becomes the averaged twin's mean — same view set,
            # same view-to-mean form; conditioner streams stay on the student.
            inv = ((z - self._swa_mu(modules, views)).square().mean()
                   if "teacher_backbone" in modules
                   else (z - z.mean(1, keepdim=True)).square().mean())
            cs = self.cfg.get("cond_stream")
            if cs == "grouped":
                zin = (z[:, :Vg].mean(1), z[:, Vg:].mean(1))
                hin = (cls[:, :Vg].mean(1), cls[:, Vg:].mean(1))
            elif cs == "perview":
                # D-087 sweep: the SIGReg placement transplanted — one KL per VIEW's
                # batch (no cross-view averaging anywhere in the estimator), averaged
                # over views; n = bs per call, so pair with floor_shrink=oas, no ring
                # (the D-075 n/d'=1 answer; the healthy e27lej control's anatomy).
                zin = tuple(z[:, v] for v in range(V))
                hin = tuple(cls[:, v] for v in range(V))
            else:
                ns = Vg if cs == "globals" else V
                zin, hin = z[:, :ns].mean(1), cls[:, :ns].mean(1)
            probe_feats = cls.flatten(0, 1).detach()  # declared h: all-view CLS (D-036)
        else:
            N, V = views.shape[:2]
            tok = modules["backbone"].forward_features(views.flatten(0, 1))
            cls = tok[:, 0]
            z = modules["projector"](cls).reshape(N, V, -1)
            # inv = all-pairs mean MSE: the V-generic form that REDUCES EXACTLY to the
            # lineage's pairwise MSE at V=2 (one pair) — no special case, v2 byte-compat
            # preserved. swa (D-095): anchor -> the twin's all-view mean, kept on the
            # all-pairs SCALE via x 2V/(V-1) (selftest §3's identity) so the lane's
            # calibrated w_inv keeps its realized pull at the twin==student init.
            inv = ((z - self._swa_mu(modules, views)).square().mean() * (2 * V / (V - 1))
                   if "teacher_backbone" in modules
                   else sum(F.mse_loss(z[:, u], z[:, w])
                            for u in range(V) for w in range(u + 1, V)) / (V * (V - 1) / 2))
            zin = (z.mean(1) if self.cfg.get("z_floor_batch", "pooled") == "view_mean"
                   else z.reshape(N * V, -1))
            hin = (cls.reshape(N, V, -1).mean(1)
                   if self.cfg.get("h_floor_batch", "pooled") == "view_mean" else cls)
            probe_feats = cls.detach()                # declared h (projector-input CLS, D-036)
        # term order mirrors the vicreg-class floorssl arms (z-floor drawn before h-floor)
        # so the per-step fresh-frame RNG sequence matches the migrated lineage. Grouped
        # streams (tuples) get one KL per group, averaged — group order globals-then-locals,
        # each group with its own ring (_zq0/_zq1; the share logger snapshots by prefix).
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
