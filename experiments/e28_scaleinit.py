"""E28 — backbone init-scale ×10 with the ViT's final LayerNorm relocated to the projector,
on the IN-100 z-only lane (Berker 2026-08-13).

The question: does raising the BACKBONE's initialization scale alone — with no regularizer of
any kind on h — buy stronger feature learning and a higher-rank / higher-stable-rank h?

The lane is `in100.floorssl.s0.d256vm4zonly` VERBATIM (h_lamb=0: MSE invariance AND the
two-sided spectral conditioner both live at z, behind the MLP), with exactly two deltas:

  (1) h := the UNNORMALIZED final ViT representation. timm applies the trunk's final
      LayerNorm inside forward_features; we MOVE that module — the same object, the same
      weights, no new RNG draw — to the front of the projector, so

          h = ViTBlocks(x),   z = MLP(LN_final(h))

      is the same function of x as before while h is read pre-norm. Verified numerically at
      ordinary init by `+e28.mode=selftest` §1 (z unchanged; LN(h_new) == h_ref).
  (2) every learned affine WEIGHT in the backbone is multiplied by 10 at init: attn.qkv
      (Q/K/V, fused by timm), attn.proj (attention output), mlp.fc1, mlp.fc2 — 4 tensors ×
      12 blocks — plus patch_embed.proj (Berker 2026-08-13: "i think patch embed proj should
      be scaled too"), 49 in total. LayerNorm/BatchNorm parameters, EVERY bias, pos_embed,
      cls_token and the whole projector keep their ordinary init (selftest §2 asserts the
      complement is bit-identical to the unscaled build).

Everything else is the in100 pipeline unchanged (aug lejepa V=4, bs=128, doses w_inv=32.8 /
w_floor=157.8 / h_lamb=0, expander_dim=256, view-mean payment at both taps with queue_steps=3,
AdamW lr 1e-3 wd 5e-2, 10-ep warmup, cosine to 1e-5, 100 epochs, seed 0, the online probe
monitor) — enforced mechanically at startup by a resolved-config diff against the reference
checkpoint (the twin-launch rule): the only keys allowed to differ are `tag` and `e28.*`.

Diagnostics per epoch, on the full val split under the deterministic eval transform:
eigenspectrum + effective rank + stable rank + RankMe of unnormalized h, of LN(h) (the
ordinary ViT h) and of z; plus the backbone's relative parameter displacement from init
‖θ−θ₀‖/‖θ₀‖ (total, per tensor family, per block) and the weight-norm ratio ‖θ‖/‖θ₀‖ — weight
decay acts on the ×10 weights, so the pair is what separates learning from decay.

SEPARATE BY REQUEST ("do not contaminate the existing codebase"): this file copies the frame
loop out of experiments/train.py (the realized-share logger is dropped — it is off in this
lane and consumes no RNG when off) and SUBCLASSES FloorSSL instead of editing it. Nothing
under sslgap/ is touched. The diagnostics here are experiment-local by the same request; if
they are kept they move to sslgap/metrics/ first (D-054).

Modes (`+e28.mode=`): `train` (default) · `selftest` (the gate) · `control` (the same
diagnostics over the landed ×1 checkpoints) · `initsweep` (h/z geometry at step 0 as a
function of the init scale, both patch_embed variants, + the per-block rank profile) ·
`viewsweep` (Ω = view-scatter / image-scatter and the inv term at step 0, per scale AND per
weight family — why inv is unstable, and which knob buys rank without the attention knife
edge).

  bash slurm/e28_launch.sh          # init sweep + the 3×8h training chain (each segment
                                    # re-runs the selftest as its own gate)
"""
import csv
import math
import os
import random

import hydra
import numpy as np
import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
import wandb
from omegaconf import DictConfig, OmegaConf
from torch.amp import GradScaler, autocast
from torch.utils.data import DataLoader

from sslgap.ckpt.schema import provenance_stamp, save_checkpoint
from sslgap.data import ViewsDataset, seed_everything, seed_worker
from sslgap.methods._common import SpectralConditioner
from sslgap.methods.base import Frame
from sslgap.methods.floorssl import FloorSSL, floorssl_head
from sslgap.metrics.spectra import (covariance_eigs, effective_rank, participation_ratio,
                                    power_law_alpha, rankme)
from sslgap.models.backbones import build_vit_trunk

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))   # hydra chdir-proof paths
SCALE = 0.1                       # the intervention: ×10 on the block weight matrices at init
SCALED = ("attn.qkv.weight", "attn.proj.weight", "mlp.fc1.weight", "mlp.fc2.weight")
CTRL_RUN = "in100.floorssl.s0.d256vm4zonly"                          # the ×1 lane this twins
REF_CKPT = os.path.join(ROOT, f"outputs/{CTRL_RUN}_ep100.pt")
OUT = os.path.join(ROOT, "results/e28")
FREE_KEYS = ("tag", "e28")         # the only resolved-config keys allowed to differ from REF


# ---------------------------------------------------------------------------------------------
# architecture: the relocation. The two rebuild blocks live here so a landed checkpoint
# reassembles without the trainer (adapters._resolve imports them by dotted name).

def e28_trunk(model_name, img_size, dynamic_img_size=False, drop_path_rate=0.0):
    """The frame trunk with its final LayerNorm REMOVED — forward_features now returns the
    unnormalized block output. The norm is not deleted from the model: e28_projector holds it."""
    trunk = build_vit_trunk(model_name, img_size, dynamic_img_size, drop_path_rate)
    trunk.norm = nn.Identity()
    return trunk


def e28_projector(in_dim=384, hidden=2048, out_dim=256, norm="bn"):
    """LN_final ∘ (the stock floorssl expander). timm gives vit_small_patch16_224 a
    timm.layers.LayerNorm(eps=1e-6) as its final norm; build_modules asserts the module it
    moves is exactly that, so this rebuild is faithful rather than merely similar."""
    return nn.Sequential(timm.layers.LayerNorm(in_dim, eps=1e-6),
                         *floorssl_head(in_dim, hidden, out_dim, norm=norm))


def _affine(n):
    """A learned affine layer's parameter (Linear/Conv), i.e. NOT a normalization gain/bias
    and not an embedding table."""
    return ".norm" in n or n.endswith(("pos_embed", "cls_token")) or n.startswith("norm.")


SELECTORS = {
    # THE ARM (Berker 2026-08-13, "the more uniform the better … hard to justify why we leave
    # some parts out"): every learned affine WEIGHT in the backbone — the four per-block
    # families and the patch embedding. 49 tensors at ViT-S.
    "all": lambda n: (n.startswith("blocks.") and n.endswith(SCALED))
                     or n == "patch_embed.proj.weight",
    # uniformity variants, for the sweep only — is the arm's exclusion list load-bearing?
    "w_and_b": lambda n: not _affine(n),          # + every affine BIAS: each layer's map ×α
    "everything": lambda n: True,                 # every backbone parameter, no exclusions
    # family splits, for the sweep only — WHICH weights carry the effect (they are diagnostic
    # cells, not candidate arms: a selective intervention has no principled justification)
    "blocks_only": lambda n: n.startswith("blocks.") and n.endswith(SCALED),
    "no_qkv": lambda n: (n.startswith("blocks.") and n.endswith(SCALED[1:]))
                        or n == "patch_embed.proj.weight",
    "qkv_only": lambda n: n.startswith("blocks.") and n.endswith(SCALED[:1]),
    "mlp_only": lambda n: n.startswith("blocks.") and n.endswith(SCALED[2:]),
}


def scale_backbone(trunk, scale, select="all"):
    """×scale on the backbone parameters chosen by SELECTORS[select]. Returns the names hit.

    What the default "all" does and does not do at the input. x0 = patch_embed(img) +
    pos_embed, and block 0 sees LN(x0), which is invariant to x0's overall scale — so scaling
    patch_embed is NOT a ×10 on the network's input; it changes the image-content :
    positional-signal ratio inside x0, since pos_embed/cls_token are not scaled."""
    hit = [n for n, _ in trunk.named_parameters() if SELECTORS[select](n)]
    assert hit, f"selector {select} matched nothing"
    with torch.no_grad():
        for n in hit:
            trunk.get_parameter(n).mul_(scale)
    return hit


class SpectatorConditioner(nn.Module):
    """The h-tap conditioner at h_lamb = 0 — DECLARED DIVERGENCE, forced by the intervention
    (gate job 63436960 §3; card §The spectator wall).

    This lane weights the h-conditioner at exactly 0; the reference lane still COMPUTES it, so
    that the per-step RNG stream stays aligned with the h-treated arms (the D-063 zonly
    convention). Reading h unnormalized at ×10 makes that spectator a crash: on step 1 the
    ring is empty, so the estimator sees n = d' = 128 and its centered slice covariance is
    structurally singular; SpectralConditioner's ridge is an ABSOLUTE eps=1e-4, which is ~3e-4
    relative to h's eigenvalues at ×1 (Cholesky survives) but ~1e-8 at ×10 — below fp32
    resolution, so `linalg.cholesky` raises. Measured: h@×1 n=128 → 0.720; h@×10 n=128 → FAIL,
    n=256 → 4405; z (rms .098, the scale LN pins) → finite at every ring size.

    So: draw the IDENTICAL random slice — same shapes, same call order, hence a bit-identical
    per-step RNG stream (QR and Cholesky consume no RNG) — and return a gradient-free 0 instead
    of factorizing a matrix nobody uses. Nothing that the term is there for is lost: the loss,
    every gradient and the RNG stream are the reference lane's, and h's geometry is measured
    far better by the per-epoch spectra than by a KL this would have logged.

    NOT a fix to SpectralConditioner: that estimator's absolute ridge is simply not scale-free,
    which matters the moment anyone conditions on an unnormalized tap. Reported, not patched."""

    def __init__(self, ref):
        super().__init__()
        self.d_slice, self.d_draw, self.rho_last = ref.d_slice, ref.d_draw, None

    def forward(self, x):
        with torch.autocast(x.device.type, enabled=False):
            x = x.reshape(-1, x.size(-1)).float()
            torch.randn(x.size(1), self.d_draw, device=x.device)   # the stream-parity draw
            return x.new_zeros(())


class E28FloorSSL(FloorSSL):
    """floorssl with h read pre-norm and the block weights ×scale at init. The loss, the aug,
    the doses, the optimizer and the step are inherited untouched: `cls` inside training_step
    is now the unnormalized h (it feeds the zero-weighted h-conditioner and the detached probe
    features), and `projector(cls)` re-applies the very LayerNorm the trunk gave up, so every
    gradient that reaches the trunk is the reference lane's."""

    def __init__(self, cfg, frame, scale=SCALE, select="all", hold=False):
        super().__init__(cfg, frame)
        self._scale, self._select, self._hold_on = float(scale), select, bool(hold)
        self.scaled_names, self._hold = [], {}

    @torch.no_grad()
    def apply_hold(self, modules):
        """SCALE HOLD — the intervention made permanent instead of initial.

        Measured (card §Weight decay): the network anneals the ×N away. The effective attention
        temperature γ·W_qkv falls 10.0 → 1.15 by ep45 while h's effective rank tracks it down
        182 → 18, and the ×1 control climbs to the same ~1.2 from below: there is an attractor,
        and an INITIALIZATION has no way to resist it. So hold the scale: after every optimizer
        step each backbone tensor is renormalized to its INIT Frobenius norm. The optimizer may
        ROTATE the weights freely — it may not resize them.

        Uniform and self-selecting: every backbone parameter with a non-zero init norm is held
        (all affine weights, all normalization gains, pos_embed, cls_token); the zero-init
        biases are excluded because there is no norm to hold. The projector is untouched — the
        intervention is "weight scaling on the ViT". This is a projection onto a constraint set,
        not a loss term, so it changes the fixed points rather than the starting point — the
        structural property an init cannot have. It also makes weight decay a no-op on the held
        tensors, which is why wd stays at the reference lane's 5e-2."""
        for n, p in modules["backbone"].named_parameters():
            t = self._hold.get(n)
            if t:
                p.mul_(t / (p.norm() + 1e-12))

    def build_modules(self):
        mods = super().build_modules()        # unchanged construction order ⇒ unchanged init RNG
        trunk, head = mods["backbone"], mods["projector"]
        ln = trunk.norm
        assert type(ln) is timm.layers.LayerNorm and ln.eps == 1e-6, \
            f"unexpected trunk final norm {type(ln)} eps={getattr(ln, 'eps', None)}"
        assert tuple(ln.normalized_shape) == (self._dim,)
        trunk.norm = nn.Identity()                       # h := pre-norm final representation
        mods["projector"] = nn.Sequential(ln, *head)     # z = MLP(LN(h)); same object, no RNG
        assert float(self.cfg.h_lamb) == 0.0, "E28 is the z-only lane: h_lamb must be 0"
        self.cond_h = SpectatorConditioner(self.cond_h)  # see the class docstring
        if self._scale != 1.0:
            self.scaled_names = scale_backbone(trunk, self._scale, self._select)
        if self._hold_on:      # recorded at INIT, so resume restores the same targets
            self._hold = {n: p.norm().item() for n, p in trunk.named_parameters()
                          if p.norm().item() > 0}
        return mods

    def arch(self):
        a = super().arch()
        a["backbone"] = {"class": "experiments.e28_scaleinit.e28_trunk",
                         "kwargs": a["backbone"]["kwargs"]}
        a["projector"] = {"class": "experiments.e28_scaleinit.e28_projector",
                          "kwargs": {"in_dim": self._dim, "hidden": self.cfg.expander_hidden,
                                     "out_dim": self.cfg.expander_dim,
                                     "norm": self.cfg.head_norm}}
        return a


# ---------------------------------------------------------------------------------------------
# diagnostics (experiment-local by request; promote to sslgap/metrics/ before any reuse — D-054)

def spectrum_row(X):
    """One space's geometry on the fixed eval set. Returns (summary dict, eigenvalues desc)."""
    X = np.asarray(X, dtype=np.float64)
    eigs = covariance_eigs(X)                       # centered covariance, float64, descending
    tr, lam1, d = float(eigs.sum()), float(eigs[0]), X.shape[1]
    er = effective_rank(eigs)
    return {"n": X.shape[0], "d": d,
            "rms": math.sqrt(tr / d),                       # per-coordinate scale of the space
            "mean_norm": float(np.linalg.norm(X.mean(0))),
            "trace": tr, "lam1": lam1, "lam_min": float(eigs[-1]),
            "stable_rank": tr / (lam1 + 1e-300),            # ‖X‖_F² / ‖X‖₂², centered
            "effrank": er,                                  # exp-entropy of the spectrum
            "effrank_frac": er / d,
            "pr": participation_ratio(eigs),
            "rankme": rankme(X),                            # uncentered (Garrido et al. 2023)
            "alpha": power_law_alpha(eigs),                 # −slope of log λ vs log rank
            "top10_frac": float(eigs[:10].sum() / (tr + 1e-300)),
            "n_eig_1e3": int((eigs > 1e-3 * lam1).sum())}, eigs


@torch.no_grad()
def measure_spaces(modules, loader, device):
    """h_raw (unnormalized final CLS), h_ln (= LN_final(h_raw), the ordinary ViT h) and z over
    the whole loader, eval mode, fp32. Trajectory-invisible: no grad, BN buffers untouched in
    eval mode, train/eval flags restored, and the caller forks the RNG (the loader draws a
    worker base_seed from the global generator when it is iterated)."""
    was = {k: m.training for k, m in modules.items()}
    for m in modules.values():
        m.eval()
    ln = modules["projector"][0]
    hs, hls, zs = [], [], []
    for views, _ in loader:
        x = views.to(device, non_blocking=True).flatten(0, 1)
        h = modules["backbone"].forward_features(x)[:, 0].float()
        hs.append(h.cpu())
        hls.append(ln(h).float().cpu())
        zs.append(modules["projector"](h).float().cpu())
    for k, m in modules.items():
        m.train(was[k])
    return {"h_raw": torch.cat(hs).numpy(), "h_ln": torch.cat(hls).numpy(),
            "z": torch.cat(zs).numpy()}


def flat_params(modules):
    return {f"{role}.{n}": p for role, m in modules.items() for n, p in m.named_parameters()}


def param_groups_map(names, n_blocks):
    """Tensor families for the displacement read; per-block rows last."""
    bb = [n for n in names if n.startswith("backbone.")]
    g = {"backbone": bb,
         "scaled48": [n for n in bb if n.startswith("backbone.blocks.") and n.endswith(SCALED)],
         "qkv": [n for n in bb if n.endswith("attn.qkv.weight")],
         "attn_proj": [n for n in bb if n.endswith("attn.proj.weight")],
         "fc1": [n for n in bb if n.endswith("mlp.fc1.weight")],
         "fc2": [n for n in bb if n.endswith("mlp.fc2.weight")],
         "block_ln": [n for n in bb if ".norm1." in n or ".norm2." in n],
         "biases": [n for n in bb if n.endswith(".bias")],
         "patch_embed": [n for n in bb if n.startswith("backbone.patch_embed.")],
         "pos_cls": [n for n in bb if n.endswith(("pos_embed", "cls_token"))],
         "final_ln": [n for n in names if n.startswith("projector.0.")],
         "projector_mlp": [n for n in names
                           if n.startswith("projector.") and not n.startswith("projector.0.")]}
    for i in range(n_blocks):
        g[f"block{i:02d}"] = [n for n in bb if n.startswith(f"backbone.blocks.{i}.")]
    return {k: v for k, v in g.items() if v}


def displacement(cur, ref, groups):
    """Relative parameter displacement from init per group: ‖θ−θ₀‖/‖θ₀‖ (how far the weights
    moved in units of their own init scale) and ‖θ‖/‖θ₀‖ (what weight decay took back)."""
    rows = {}
    for gname, names in groups.items():
        d2 = r2 = c2 = 0.0
        for n in names:
            a, b = cur[n].detach().double(), ref[n].double()
            d2 += float((a - b).pow(2).sum())
            r2 += float(b.pow(2).sum())
            c2 += float(a.pow(2).sum())
        rows[gname] = {"n_par": sum(cur[n].numel() for n in names),
                       "rel_disp": math.sqrt(d2 / r2), "norm_ratio": math.sqrt(c2 / r2)}
    return rows


def append_csv(path, rows):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    new = not os.path.exists(path)
    with open(path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        if new:
            w.writeheader()
        w.writerows(rows)


def save_spectra(path, epoch, eigs_by_space):
    """One npz per run: `ep{e}.{space}` -> the full eigenvalue vector (rewritten each epoch,
    ~1 MB at 100 epochs)."""
    store = dict(np.load(path)) if os.path.exists(path) else {}
    for space, eigs in eigs_by_space.items():
        store[f"ep{epoch}.{space}"] = eigs
    np.savez(path, **store)


def gauge_row(X):
    """The gauge decomposition of UNNORMALIZED h (Berker 2026-08-13: "with 5x it collapses
    harder … h_raw effrank as low as 2").

    z = MLP(LN_final(h)), and LN is invariant to h -> a·h + b·1 for any PER-SAMPLE scalars
    a>0, b. So the common mode (the all-ones direction 1/√D) and the per-sample scale are
    EXACT GAUGE FREEDOMS of this objective: nothing in the loss pins them, and once they drift
    they can carry almost all of h_raw's variance. Then "effective rank of h_raw" measures the
    drift, not the representation. This splits the two apart:

      gauge_share  — fraction of tr(Σ_h) carried by the all-ones direction
      scale_share  — fraction carried by the per-sample scale (‖h_i‖ variation, after the mean)
      effrank_perp — effective rank with the common mode projected OUT (the honest h rank)
    """
    X = np.asarray(X, dtype=np.float64)
    N, D = X.shape
    u = np.ones(D) / math.sqrt(D)
    c = X @ u                                   # coefficient along the all-ones direction
    Xp = X - np.outer(c, u)                     # common mode projected out
    tot = float(((X - X.mean(0)) ** 2).sum(1).mean())
    s = np.linalg.norm(Xp, axis=1)              # per-sample scale, after the mean is removed
    row, eigs = spectrum_row(Xp)
    return {"gauge_share": float(c.var() / (tot + 1e-300)),
            "scale_share": float(s.var() / (tot + 1e-300)),
            "effrank_perp": row["effrank"], "stable_rank_perp": row["stable_rank"],
            "rms_perp": row["rms"], "common_mode_mean": float(c.mean()),
            "common_mode_std": float(c.std())}


@torch.no_grad()
def view_alignment(modules, views, device):
    """Is the network learning to agree across views? (Berker 2026-08-13: "inv is not going
    down".) Raw inv is not the readout — the z-conditioner inflates z's scale over training
    (z rms .05 → .54 in epoch 1), which raises the view MSE mechanically. Two scale-free
    forms on ONE fixed V=4 batch: Ω = W/B (view scatter / image-centre scatter, the house
    orbit calculus) and inv / (2·var z), the fraction of "two independent draws" — 1.0 means
    the views carry no information about each other, 0 means perfect agreement."""
    from sslgap.metrics.orbit_energy import orbit_energies

    was = {k: m.training for k, m in modules.items()}
    for m in modules.values():
        m.eval()
    N, V = views.shape[:2]
    h = modules["backbone"].forward_features(views.to(device).flatten(0, 1))[:, 0].float()
    z = modules["projector"](h).float()
    h, z = h.reshape(N, V, -1), z.reshape(N, V, -1)
    inv = float(sum(F.mse_loss(z[:, u], z[:, w]) for u in range(V)
                    for w in range(u + 1, V)) / (V * (V - 1) / 2))
    var_z = float(z.reshape(-1, z.shape[-1]).var(0).mean())
    eh, ez = (orbit_energies([s[:, v].cpu().numpy() for v in range(V)]) for s in (h, z))
    for k, m in modules.items():
        m.train(was[k])
    return {"inv": inv, "inv_over_indep": inv / (2 * var_z + 1e-12), "var_z": var_z,
            "omega_h": eh["omega"], "omega_z": ez["omega"],
            "W_h": eh["W"], "B_h": eh["B"], "W_z": ez["W"], "B_z": ez["B"]}


def run_diagnostics(run_id, epoch, step, modules, loader, device, theta0, groups, log=True,
                    aug_batch=None):
    """The per-epoch read: three spaces × spectrum + the displacement table (+ view alignment
    when a fixed augmented batch is supplied — its own CSV, so no existing header shifts)."""
    py_s, np_s = random.getstate(), np.random.get_state()
    with torch.random.fork_rng():
        feats = measure_spaces(modules, loader, device)
    random.setstate(py_s), np.random.set_state(np_s)
    diag_rows, eigs_by_space, sc = [], {}, {}
    for space, X in feats.items():
        row, eigs = spectrum_row(X)
        eigs_by_space[space] = eigs
        diag_rows.append({"run_id": run_id, "epoch": epoch, "step": step, "space": space, **row})
        sc.update({f"diag/{space}/{k}": v for k, v in row.items() if k not in ("n", "d")})
    append_csv(f"{OUT}/{run_id}_diag.csv", diag_rows)
    save_spectra(f"{OUT}/{run_id}_spectra.npz", epoch, eigs_by_space)
    disp = displacement(flat_params(modules), theta0, groups)
    append_csv(f"{OUT}/{run_id}_disp.csv",
               [{"run_id": run_id, "epoch": epoch, "step": step, "group": g, **v}
                for g, v in disp.items()])
    sc.update({f"disp/{g}": v["rel_disp"] for g, v in disp.items()})
    sc.update({f"wnorm/{g}": v["norm_ratio"] for g, v in disp.items()})
    gr = gauge_row(feats["h_raw"])
    append_csv(f"{OUT}/{run_id}_gauge.csv",
               [{"run_id": run_id, "epoch": epoch, "step": step, **gr}])
    sc.update({f"gauge/{k}": v for k, v in gr.items()})
    al = None
    if aug_batch is not None:
        py2, np2 = random.getstate(), np.random.get_state()
        with torch.random.fork_rng():
            al = view_alignment(modules, aug_batch, device)
        random.setstate(py2), np.random.set_state(np2)
        append_csv(f"{OUT}/{run_id}_align.csv",
                   [{"run_id": run_id, "epoch": epoch, "step": step, **al}])
        sc.update({f"align/{k}": v for k, v in al.items()})
    if log:
        wandb.log(sc, step=step)
    print(f"[e28] ep{epoch} " + " ".join(
        f"{s}: rms={sc[f'diag/{s}/rms']:.3g} effrank={sc[f'diag/{s}/effrank']:.1f} "
        f"srank={sc[f'diag/{s}/stable_rank']:.1f}" for s in ("h_raw", "h_ln", "z"))
        + f" | disp bb={disp['backbone']['rel_disp']:.4f} "
          f"scaled={disp['scaled48']['rel_disp']:.4f} "
          f"| wnorm bb={disp['backbone']['norm_ratio']:.4f}"
        + f" | gauge={gr['gauge_share']:.3f} effrank_perp={gr['effrank_perp']:.1f}"
        + (f" | inv/indep={al['inv_over_indep']:.3f} omega_h={al['omega_h']:.2f} "
           f"omega_z={al['omega_z']:.2f}" if al else ""), flush=True)
    return sc


# ---------------------------------------------------------------------------------------------
# guards

def _flat(d, pre=""):
    out = {}
    for k, v in d.items():
        if isinstance(v, dict):
            out.update(_flat(v, f"{pre}{k}."))
        else:
            out[f"{pre}{k}"] = v
    return out


def guard_cfg(cfg, ref_path, allow=()):
    """The twin-launch rule in code (memory: twin-launch-config-diff — the zonly relaunch was
    born from a hand-copied override set that silently took bs=256/num_classes=10): the
    resolved config must equal the reference checkpoint's cfg on every key except `tag`, the
    `e28.*` namespace, and any key the run DECLARES it is changing via `+e28.allow=[...]`.
    The declaration is the point — an intended deviation gets named and printed, an
    unintended one still kills the job."""
    ref = torch.load(ref_path, map_location="cpu", weights_only=False, mmap=True)["cfg"]
    a, b = _flat(ref), _flat(OmegaConf.to_container(cfg, resolve=True))
    diff = {k: (a.get(k, "<absent>"), b.get(k, "<absent>"))
            for k in set(a) | set(b) if a.get(k, "<absent>") != b.get(k, "<absent>")}
    bad = {k: v for k, v in diff.items()
           if k.split(".")[0] not in FREE_KEYS and k not in allow}
    print(f"[e28] resolved-cfg diff vs {os.path.basename(ref_path)}: "
          + (", ".join(f"{k}: {v[0]} -> {v[1]}" for k, v in sorted(diff.items())) or "none"),
          flush=True)
    for k in allow:
        assert k in diff, f"declared deviation {k} does not actually differ from the reference"
        print(f"[e28] DECLARED DEVIATION  {k}: {diff[k][0]} -> {diff[k][1]}", flush=True)
    assert not bad, f"config drift vs the reference lane (only {FREE_KEYS} + declared " \
                    f"e28.allow may differ): {bad}"


def selftest(cfg, frame, select="all", hold=False):
    """Gates every launch. §1 the relocation is numerically a no-op for z at ordinary init;
    §2 the ×10 hits exactly the intended weight matrices and nothing else; §3 the scaled model
    is trainable (finite loss/grads/step) and h is genuinely unnormalized."""
    dev = frame.device

    def build(cls, **kw):
        seed_everything(cfg.seed)
        m = cls(cfg.method, frame, **kw)
        return m, m.build_modules().to(dev)

    _, ref = build(FloorSSL)
    _, new = build(E28FloorSSL, scale=1.0)
    ten_m, ten = build(E28FloorSSL, scale=SCALE, select=select, hold=hold)
    for mods in (ref, new, ten):
        for m in mods.values():
            m.eval()

    # §1 relocation: same weights, same z, h now pre-norm ---------------------------------------
    print("\n[selftest] §1 LN relocation @ ordinary init (scale=1)")
    rp = dict(ref["backbone"].named_parameters())
    npar = dict(new["backbone"].named_parameters())
    assert set(rp) - set(npar) == {"norm.weight", "norm.bias"}, set(rp) ^ set(npar)
    for n in npar:
        assert torch.equal(rp[n], npar[n]), f"trunk param changed: {n}"
    ln = new["projector"][0]
    assert torch.equal(ln.weight, rp["norm.weight"]) and torch.equal(ln.bias, rp["norm.bias"])
    for i, (a, b) in enumerate(zip(list(ref["projector"]), list(new["projector"])[1:])):
        assert type(a) is type(b), i
        assert all(torch.equal(x, y) for x, y in zip(a.state_dict().values(),
                                                     b.state_dict().values())), i
    print("       params: trunk bit-identical, final LN moved verbatim, expander bit-identical")

    torch.manual_seed(1234)
    x = torch.randn(32, 3, frame.img_size, frame.img_size, device=dev)
    with torch.no_grad():
        h_ref = ref["backbone"].forward_features(x)[:, 0]
        z_ref = ref["projector"](h_ref)
        h_new = new["backbone"].forward_features(x)[:, 0]
        z_new = new["projector"](h_new)
        dz = (z_ref - z_new).abs().max().item()
        dh = (h_ref - ln(h_new)).abs().max().item()
        with autocast(dev, dtype=torch.bfloat16):
            dz_bf = (ref["projector"](ref["backbone"].forward_features(x)[:, 0])
                     - new["projector"](new["backbone"].forward_features(x)[:, 0])
                     ).abs().max().item()
    print(f"       fp32 max|z_ref − z_new|      = {dz:.3e}   (max|z| {z_ref.abs().max():.3f})")
    print(f"       fp32 max|h_ref − LN(h_new)|  = {dh:.3e}   (max|h| {h_ref.abs().max():.3f})")
    print(f"       bf16 max|z_ref − z_new|      = {dz_bf:.3e}")
    print(f"       h_ref rms {h_ref.std():.4f} (normalized) vs h_new rms {h_new.std():.4f} (raw)")
    assert dz <= 1e-5 * max(1.0, z_ref.abs().max().item()), dz
    assert dh <= 1e-5 * max(1.0, h_ref.abs().max().item()), dh
    assert not torch.allclose(ln(h_new), h_new, atol=1e-3), "h is not actually unnormalized"

    # §2 scaling audit ---------------------------------------------------------------------------
    print("\n[selftest] §2 ×%g scaling audit (selector %r)" % (SCALE, select))
    p1, p10 = npar, dict(ten["backbone"].named_parameters())
    hit = set(ten_m.scaled_names)
    assert hit == {n for n, _ in p1.items() if SELECTORS[select](n)}, "selector mismatch"
    if select == "all":                       # the arm: 4 families × 12 blocks + patch_embed
        assert len(hit) == 4 * len(ten["backbone"].blocks) + 1 == 49, len(hit)
    for n in p1:
        if n in hit:
            assert torch.equal(p10[n], p1[n] * SCALE), f"{n} not exactly ×{SCALE}"
        else:
            assert torch.equal(p10[n], p1[n]), f"{n} MUST NOT be scaled"
    for (n, a), (_, b) in zip(new["projector"].named_parameters(),
                              ten["projector"].named_parameters()):
        assert torch.equal(a, b), f"projector.{n} MUST NOT be scaled"
    fams = sorted({n.split(".", 2)[2] if n.startswith("blocks.") else n for n in hit})
    print(f"       scaled {len(hit)} tensors, families: {fams}")
    print("       untouched: " + ", ".join(sorted(
        {"block LayerNorm" if ".norm" in n else "bias" if n.endswith("bias") else n.split(".")[0]
         for n in p1 if n not in hit})) + ", projector (incl. the moved final LN)")

    # §3 the scaled model is trainable ------------------------------------------------------------
    print("\n[selftest] §3 ×%g health" % SCALE)
    with torch.no_grad():
        h10 = ten["backbone"].forward_features(x)[:, 0]
        z10 = ten["projector"](h10)
    print(f"       h rms {h10.std():.3f} (×1: {h_new.std():.4f})   "
          f"z rms {z10.std():.4f} (×1: {z_new.std():.4f})  — LN pins z's scale")
    assert torch.isfinite(h10).all() and torch.isfinite(z10).all()

    # §3b the estimator at the OPERATING shape: step 1 has an empty ring, so both taps see
    # n = bs = 128 rows in a d'=128 slice — the structurally singular case that the absolute
    # 1e-4 ridge can no longer carry once h is read unnormalized at ×10 (card §The spectator
    # wall). z is the objective's term and must be finite; h's is the spectator.
    print("\n[selftest] §3b conditioners at the step-1 ring state (n = d' = 128)")
    with torch.no_grad():
        xb = torch.randn(128, 3, frame.img_size, frame.img_size, device=dev)
        hb = ten["backbone"].forward_features(xb)[:, 0]
        zb = ten["projector"](hb)
    stock = SpectralConditioner(d_slice=128, d_draw=128).to(dev)
    try:
        with torch.no_grad():
            stock_h = f"{float(stock(hb)):.1f}"
    except Exception as e:                     # the crash this lane would have died on
        stock_h = f"RAISES {type(e).__name__}"
    with torch.no_grad():
        reg_z, reg_h = float(ten_m.cond_z(zb)), float(ten_m.cond_h(hb))
    print(f"       stock SpectralConditioner on unnormalized h (rms {hb.std():.1f}): {stock_h}")
    print(f"       cond_z (the objective's term, z rms {zb.std():.4f}): {reg_z:.4f}   "
          f"cond_h (spectator, h_lamb=0): {reg_h:.1f}")
    assert math.isfinite(reg_z) and reg_h == 0.0

    # §3c one real optimizer step (bs=32 keeps the gate runnable on any card; the operating
    # bs=128 estimator shape is covered by §3b)
    ten_m.train_mode(ten)
    V = int(cfg.method.get("V", 4))
    views = torch.randn(32, V, 3, frame.img_size, frame.img_size, device=dev)
    opt = torch.optim.AdamW(ten_m.param_groups(ten))
    with autocast(dev, dtype=torch.bfloat16):
        terms, feats, k = ten_m.training_step(ten, views, dev)
    print("\n[selftest] §3c one AdamW step (synthetic bs=32: moment_kl is estimator-degenerate "
          "by construction — the check is finiteness)\n       terms: "
          + " ".join(f"{a}={b.item():.4f}" for a, b in terms.items()))
    assert all(torch.isfinite(v) for v in terms.values()), terms
    terms["loss"].backward()
    gn = torch.nn.utils.clip_grad_norm_([p for g in opt.param_groups for p in g["params"]],
                                        frame.grad_clip).item()
    opt.step()
    bad = [n for n, p in ten["backbone"].named_parameters() if not torch.isfinite(p).all()]
    print(f"       grad_norm {gn:.3f}, k={k}, probe_feats {tuple(feats.shape)}, "
          f"non-finite params after the step: {len(bad)}")
    assert math.isfinite(gn) and not bad
    if hold:
        print("\n[selftest] §4 scale hold")
        n0 = {n: p.norm().item() for n, p in ten["backbone"].named_parameters()}
        for _ in range(3):                      # the step above already perturbed the weights
            with autocast(dev, dtype=torch.bfloat16):
                t, _, _ = ten_m.training_step(ten, views, dev)
            opt.zero_grad(); t["loss"].backward(); opt.step(); ten_m.apply_hold(ten)
        n1 = {n: p.norm().item() for n, p in ten["backbone"].named_parameters()}
        held = [n for n in ten_m._hold]
        drift = max(abs(n1[n] / ten_m._hold[n] - 1) for n in held)
        free = [n for n in n0 if n not in ten_m._hold]
        print(f"       {len(held)} tensors pinned, {len(free)} free (zero-init biases); "
              f"max |‖θ‖/‖θ₀‖ − 1| over pinned = {drift:.2e}")
        print(f"       free tensors did move: max |Δ‖θ‖| = "
              f"{max(abs(n1[n] - n0[n]) for n in free):.3e}")
        assert drift < 1e-5, drift
    print("\n[selftest] ALL GREEN\n", flush=True)


SWEEP_SCALES = (1.0, 1.5, 2.0, 3.0, 5.0, 10.0, 20.0, 50.0, 100.0)
DEPTH_SCALES = (1.0, 3.0, 10.0, 100.0)


def init_sweep(cfg, frame, loader, scales=SWEEP_SCALES):
    """Berker 2026-08-13: "effective rank 180 is a good start … i was expecting to reach a
    higher number". This measures the question instead of arguing it — h/z geometry AT
    INITIALIZATION (no training, no optimizer) as a function of the init scale, for both
    patch_embed variants, on the same val set the trajectory diagnostics use. Two outputs:

      results/e28/e28_initsweep.csv  — one row per (scale, patch_embed, space)
      results/e28/e28_initdepth.csv  — effrank of the raw residual-stream CLS after EVERY
                                       block, at a few scales: where along depth the rank is
                                       built and whether it is then given back.

    Cost is a forward pass per cell, so the whole thing is minutes."""
    for scale in scales:
        for sel in ("all", "blocks_only"):
            if scale == 1.0 and sel != "all":
                continue                                  # ×1 is the same model either way
            seed_everything(cfg.seed)
            mods = E28FloorSSL(cfg.method, frame, scale=scale,
                               select=sel).build_modules().to(frame.device)
            feats = measure_spaces(mods, loader, frame.device)
            rows = []
            for space, X in feats.items():
                row, _ = spectrum_row(X)
                rows.append({"scale": scale, "select": sel, "space": space, **row})
            append_csv(f"{OUT}/e28_initsweep.csv", rows)
            print(f"[sweep] ×{scale:<5g} {sel:<12} " + "  ".join(
                f"{r['space']}: rms={r['rms']:.3g} effrank={r['effrank']:.1f}"
                f"({r['effrank_frac']:.2f}d) srank={r['stable_rank']:.1f} "
                f"alpha={r['alpha']:.2f}" for r in rows), flush=True)

    for scale in (s for s in DEPTH_SCALES if s in set(scales)):
        seed_everything(cfg.seed)
        mods = E28FloorSSL(cfg.method, frame, scale=scale).build_modules().to(frame.device)
        trunk = mods["backbone"].eval()
        nb = len(trunk.blocks)
        cls = [[] for _ in range(nb)]
        with torch.no_grad():
            for views, _ in loader:
                x = views.to(frame.device, non_blocking=True).flatten(0, 1)
                # norm=True passes each block's output through trunk.norm, which is Identity
                # here — so these are the RAW residual-stream states, the h of a trunk cut at
                # that depth.
                for i, (_, pre) in enumerate(trunk.get_intermediate_layers(
                        x, n=list(range(nb)), return_prefix_tokens=True, norm=True)):
                    cls[i].append(pre[:, 0].float().cpu())
        rows = []
        for i in range(nb):
            row, _ = spectrum_row(torch.cat(cls[i]).numpy())
            rows.append({"scale": scale, "block": i + 1, **row})
        append_csv(f"{OUT}/e28_initdepth.csv", rows)
        print(f"[depth] ×{scale:<5g} effrank by block: "
              + " ".join(f"{r['effrank']:.0f}" for r in rows)
              + " | rms: " + " ".join(f"{r['rms']:.3g}" for r in rows), flush=True)


FAMILY_SETS = ("all", "w_and_b", "everything", "no_qkv", "qkv_only", "mlp_only")


def view_sweep(cfg, frame, loader, scales=(1.0, 2.0, 3.0, 5.0, 10.0, 20.0)):
    """Why inv is unstable at ×10, measured (Berker 2026-08-13: "inv became very unstable …
    trying to figure out how to make this rich learning regime").

    The invariance term's difficulty is a property of the network AT INIT: if two augmented
    views of one image already land as far apart as two different images, inv has nothing to
    grip. That ratio is the repo's own instrument — Ω = W/B, orbit pair energy over centre
    pair energy (sslgap/metrics/orbit_energy.py). Ω ≪ 1 = views agree (ordered); Ω ≈ 1 = a
    view of an image is as far from its twin as from another image (chaotic).

    Measured per (scale × which weight families carry it), on ONE fixed 256-image batch drawn
    with the lane's own V=4 lejepa aug: Ω_h, Ω_z, the loss's own inv value, and h's effective
    rank on the val set. The family split is the design question: attention logits scale as
    scale² (both q and k are scaled) so `qkv` is the sharpening knob, while attn.proj/fc1/fc2
    only raise the branch-to-residual ratio — `no_qkv` asks whether the rank can be had
    without the near-hard-attention knife edge.

      results/e28/e28_viewsweep.csv
    """
    from sslgap.metrics.orbit_energy import orbit_energies

    seed_everything(cfg.seed)
    ds = E28FloorSSL(cfg.method, frame, scale=1.0).build_train_dataset()
    vx = next(iter(DataLoader(ds, batch_size=256, shuffle=True, num_workers=8,
                              generator=torch.Generator().manual_seed(0))))[0]
    for scale in scales:
        for fam in FAMILY_SETS:
            if scale == 1.0 and fam != "all":
                continue
            seed_everything(cfg.seed)
            mods = E28FloorSSL(cfg.method, frame, scale=scale,
                               select=fam).build_modules().to(frame.device)
            for m in mods.values():
                m.eval()
            N, V = vx.shape[:2]
            with torch.no_grad():
                hf = mods["backbone"].forward_features(
                    vx.to(frame.device).flatten(0, 1))[:, 0].float()
                z = mods["projector"](hf).float()
            h, z = hf.reshape(N, V, -1), z.reshape(N, V, -1)
            inv = sum(F.mse_loss(z[:, u], z[:, w]) for u in range(V)
                      for w in range(u + 1, V)) / (V * (V - 1) / 2)
            eh, ez = (orbit_energies([s[:, v].cpu().numpy() for v in range(V)])
                      for s in (h, z))
            rank = spectrum_row(measure_spaces(mods, loader, frame.device)["h_raw"])[0]
            row = {"scale": scale, "select": fam, "inv": float(inv),
                   "omega_h": eh["omega"], "omega_z": ez["omega"],
                   "W_h": eh["W"], "B_h": eh["B"], "W_z": ez["W"], "B_z": ez["B"],
                   "h_effrank": rank["effrank"], "h_stable_rank": rank["stable_rank"],
                   "h_rms": rank["rms"], "h_alpha": rank["alpha"]}
            append_csv(f"{OUT}/e28_viewsweep.csv", [row])
            print(f"[views] ×{scale:<5g} {fam:<9} omega_h={eh['omega']:7.3f} "
                  f"omega_z={ez['omega']:7.3f} inv={float(inv):7.4f} "
                  f"h_effrank={rank['effrank']:6.1f} srank={rank['stable_rank']:5.1f}",
                  flush=True)


def gauge_probe(cfg, frame, loader):
    """Run the gauge decomposition over landed checkpoints of every E28 lane + the ×1 control."""
    cells = [(CTRL_RUN, 1.0, ep) for ep in (25, 50, 100)] + \
            [("in100.floorssl.s0.e28x5", 5.0, ep) for ep in (25, 50)] + \
            [("in100.floorssl.s0.e28x10", 10.0, ep) for ep in (25, 50)]
    rows = []
    for run, scale, ep in cells:
        path = os.path.join(ROOT, f"outputs/{run}_ep{ep}.pt")
        if not os.path.exists(path):
            continue
        seed_everything(cfg.seed)
        mods = E28FloorSSL(cfg.method, frame, scale=scale).build_modules()
        sd = torch.load(path, map_location="cpu", weights_only=False)["modules"]
        if "norm.weight" in sd["backbone"]:                       # a stock ×1 checkpoint
            proj = {f"{int(k.split('.')[0]) + 1}.{k.split('.', 1)[1]}": v
                    for k, v in sd["projector"].items()}
            proj["0.weight"] = sd["backbone"].pop("norm.weight")
            proj["0.bias"] = sd["backbone"].pop("norm.bias")
            sd = {"backbone": sd["backbone"], "projector": proj}
        mods["backbone"].load_state_dict(sd["backbone"])
        mods["projector"].load_state_dict(sd["projector"])
        feats = measure_spaces(mods.to(frame.device), loader, frame.device)
        raw, _ = spectrum_row(feats["h_raw"])
        ln, _ = spectrum_row(feats["h_ln"])
        g = gauge_row(feats["h_raw"])
        rows.append({"run": run, "scale": scale, "epoch": ep,
                     "h_raw_effrank": raw["effrank"], "h_ln_effrank": ln["effrank"], **g})
        print(f"[gauge] {run.split('.')[-1]:<14} ep{ep:<4} h_raw effrank={raw['effrank']:6.1f} "
              f"| common-mode share={g['gauge_share']:.4f} scale share={g['scale_share']:.4f} "
              f"| effrank_perp={g['effrank_perp']:6.1f}  h_ln effrank={ln['effrank']:6.1f}",
              flush=True)
    if rows:
        append_csv(f"{OUT}/e28_gauge_ckpts.csv", rows)


def dose_probe(cfg, frame):
    """Why the ×10 network is not training, in the house instrument (Berker 2026-08-13: "the
    network does not look like it is training. lets work on that").

    E24-T1/D-070: a recipe is portable in REALIZED SHARE, never in nominal weight. The per-term
    trunk-only pull g_k = ‖∂ term_k / ∂ backbone‖ changes when the network changes, so the same
    (w_inv, w_floor) means a different balance at ×10 than at ×1 — and the balance is what
    decides whether inv collapses the representation or the conditioner holds it open. This
    measures g on ONE fixed V=4 batch for each cell (×1 and ×10, at init and at the operating
    state), with the conditioner RINGS WARMED first (the cold-ring trap, HISTORY 2026-08-07):
    queue_steps=3 means a cold ring feeds the estimator n=bs instead of 4·bs and reads a
    different g.

    Output: g_k, the realized share s_k = w_k·g_k / Σ w·g, and the total pull T = Σ w·g, plus
    the re-dose w_k = s*_k·T*/g_k that would put the ×10 lane on the ×1 lane's profile.
    """
    seed_everything(cfg.seed)
    ds = E28FloorSSL(cfg.method, frame, scale=1.0).build_train_dataset()
    dl = DataLoader(ds, batch_size=cfg.bs, shuffle=True, num_workers=8,
                    generator=torch.Generator().manual_seed(0))
    batches = [b[0] for b, _ in zip(dl, range(5))]          # 4 to warm the rings + 1 to measure
    w = {"inv": float(cfg.method.w_inv), "moment_kl": float(cfg.method.w_floor)}
    cells = [("x1", 1.0, None), ("x1", 1.0, f"outputs/{CTRL_RUN}_ep25.pt"),
             ("x10", SCALE, None),
             ("x10", SCALE, f"outputs/in100.floorssl.s0.e28x10_last.pt")]
    rows = []
    for tag, scale, ckpt in cells:
        seed_everything(cfg.seed)
        method = E28FloorSSL(cfg.method, frame, scale=scale)
        mods = method.build_modules()
        if ckpt and os.path.exists(os.path.join(ROOT, ckpt)):
            pay = torch.load(os.path.join(ROOT, ckpt), map_location="cpu", weights_only=False)
            sd = pay["modules"]
            if "norm.weight" in sd["backbone"]:                    # the stock ×1 checkpoints
                proj = {f"{int(k.split('.')[0]) + 1}.{k.split('.', 1)[1]}": v
                        for k, v in sd["projector"].items()}
                proj["0.weight"] = sd["backbone"].pop("norm.weight")
                proj["0.bias"] = sd["backbone"].pop("norm.bias")
                sd = {"backbone": sd["backbone"], "projector": proj}
            mods["backbone"].load_state_dict(sd["backbone"])
            mods["projector"].load_state_dict(sd["projector"])
            state = f"ep{pay['epoch'] + 1}"
        elif ckpt:
            print(f"[dose] {tag}: {ckpt} absent — skipped", flush=True)
            continue
        else:
            state = "init"
        mods = mods.to(frame.device)
        method.train_mode(mods)
        params = [p for p in mods["backbone"].parameters() if p.requires_grad]
        with torch.no_grad(), autocast(frame.device, dtype=torch.bfloat16):
            for b in batches[:4]:                                  # warm the conditioner rings
                method.training_step(mods, b.to(frame.device), frame.device)
        with autocast(frame.device, dtype=torch.bfloat16):
            terms, _, _ = method.training_step(mods, batches[4].to(frame.device), frame.device)
        g = {}
        for i, k in enumerate(w):
            gr = torch.autograd.grad(terms[k], params, retain_graph=i < len(w) - 1,
                                     allow_unused=True)
            g[k] = torch.cat([t.reshape(-1).float() for t in gr if t is not None]).norm().item()
        T = sum(w[k] * g[k] for k in w)
        row = {"cell": f"{tag}.{state}", **{f"g_{k}": g[k] for k in w},
               **{f"share_{k}": w[k] * g[k] / T for k in w}, "T": T,
               **{f"term_{k}": float(terms[k]) for k in terms}}
        rows.append(row)
        print(f"[dose] {row['cell']:<10} " + " ".join(
            f"{k}: g={g[k]:9.4f} w·g={w[k] * g[k]:9.3f} share={w[k] * g[k] / T:.3f}"
            for k in w) + f" | T={T:9.3f}", flush=True)
    append_csv(f"{OUT}/e28_doses.csv", rows)
    ref = next((r for r in rows if r["cell"].startswith("x1.") and "ep" in r["cell"]), rows[0])
    tgt = next((r for r in rows if r["cell"].startswith("x10.") and "ep" in r["cell"]), None)
    if tgt:
        print(f"\n[dose] re-dose ×10 onto the {ref['cell']} profile "
              f"(s_inv={ref['share_inv']:.3f}, s_z={ref['share_moment_kl']:.3f}, "
              f"T={ref['T']:.3f}):", flush=True)
        for k in w:
            print(f"        w_{k} = {ref[f'share_{k}'] * ref['T'] / tgt[f'g_{k}']:10.4f}  "
                  f"(now {w[k]})", flush=True)


def control_diagnostics(cfg, frame, loader):
    """The ×1 reference curve, for free: the SAME diagnostics on the landed d256vm4zonly
    checkpoints (ep0 = the seed-0 init rebuild). The relocation is applied after
    load_state_dict, so h_raw there is the same pre-norm object this run trains on — §1 proves
    the relocation changes nothing else."""
    seed_everything(cfg.seed)
    theta0 = {k: v.detach().clone()
              for k, v in flat_params(E28FloorSSL(cfg.method, frame,
                                                  scale=1.0).build_modules()).items()}
    groups = param_groups_map(list(theta0), 12)
    theta0 = {k: v.to(frame.device) for k, v in theta0.items()}
    for ep in (0, 25, 50, 75, 100):
        path = os.path.join(ROOT, f"outputs/{CTRL_RUN}_ep{ep}.pt")
        if ep and not os.path.exists(path):
            print(f"[e28] control ep{ep}: {path} absent — skipped", flush=True)
            continue
        seed_everything(cfg.seed)
        mods = E28FloorSSL(cfg.method, frame, scale=1.0).build_modules()
        if ep:
            sd = torch.load(path, map_location="cpu", weights_only=False)["modules"]
            proj = {f"{int(k.split('.')[0]) + 1}.{k.split('.', 1)[1]}": v
                    for k, v in sd["projector"].items()}          # the LN takes index 0
            proj["0.weight"] = sd["backbone"].pop("norm.weight")
            proj["0.bias"] = sd["backbone"].pop("norm.bias")
            mods["backbone"].load_state_dict(sd["backbone"])
            mods["projector"].load_state_dict(proj)
        run_diagnostics(CTRL_RUN, ep, 0, mods.to(frame.device), loader, frame.device,
                        theta0, groups, log=False)


# ---------------------------------------------------------------------------------------------

@hydra.main(version_base=None, config_path="configs", config_name="train")
def main(cfg: DictConfig):
    e28 = cfg.e28 if "e28" in cfg else {}
    mode = e28.get("mode", "train")
    workers = (cfg.num_workers if cfg.num_workers is not None
               else int(os.environ.get("SLURM_CPUS_PER_TASK", 8)))
    frame = Frame(name=cfg.frame.name, model_name=cfg.frame.model_name,
                  img_size=cfg.frame.img_size, dataset=cfg.frame.dataset,
                  data_root=cfg.frame.get("data_root"), epochs=cfg.frame.epochs,
                  seed=cfg.seed, grad_clip=cfg.method.get("grad_clip", cfg.frame.grad_clip),
                  num_workers=workers, device=cfg.device)
    guard_cfg(cfg, e28.get("ref_ckpt", REF_CKPT), tuple(e28.get("allow") or ()))
    os.makedirs(OUT, exist_ok=True)
    select = e28.get("select", "all")
    if mode == "selftest":
        return selftest(cfg, frame, select, bool(e28.get("hold", False)))

    run_id = f"{frame.name}.{cfg.method.name}.s{cfg.seed}" + (f".{cfg.tag}" if cfg.tag else "")
    out_dir = os.path.expanduser(cfg.out_dir)
    ckpt_base = os.path.join(out_dir, run_id)
    last_path = f"{ckpt_base}_last.pt"

    seed_everything(cfg.seed)
    method = E28FloorSSL(cfg.method, frame, scale=float(e28.get("scale", SCALE)),
                         select=select, hold=bool(e28.get("hold", False)))
    modules = method.build_modules().to(frame.device)
    probe = nn.Sequential(nn.LayerNorm(method.probe_dim()),
                          nn.Linear(method.probe_dim(), cfg.num_classes)).to(frame.device)
    hold_msg = (f"ON — {len(method._hold)} backbone tensors pinned to their init norm"
                if method._hold_on else "off")
    print(f"[e28] init ×{method._scale} on {len(method.scaled_names)} tensors "
          f"(selector {select!r}); scale-hold {hold_msg}", flush=True)

    val_ds = ViewsDataset(frame.dataset, "validation", V=1, img_size=frame.img_size,
                          data_root=frame.data_root)
    nw = frame.num_workers
    val = DataLoader(val_ds, batch_size=256, num_workers=nw, worker_init_fn=seed_worker,
                     pin_memory=cfg.pin_memory,
                     persistent_workers=cfg.persistent_workers and nw > 0)
    if mode == "control":
        return control_diagnostics(cfg, frame, val)
    if mode == "initsweep":
        return init_sweep(cfg, frame, val, tuple(e28.get("scales") or SWEEP_SCALES))
    if mode == "gauge":
        return gauge_probe(cfg, frame, val)
    if mode == "doses":
        return dose_probe(cfg, frame)
    if mode == "viewsweep":
        return view_sweep(cfg, frame, val, tuple(e28.get("scales") or (1.0, 2.0, 3.0, 5.0, 10.0, 20.0)))

    # θ₀: the displacement reference must survive requeue/resume, so it is written once, before
    # the first step, and reloaded by every later segment.
    theta_path = f"{OUT}/{run_id}_theta0.pt"
    if os.path.exists(theta_path):
        theta0 = torch.load(theta_path, map_location=frame.device, weights_only=True)
        assert set(theta0) == set(flat_params(modules)), "θ₀ snapshot / model mismatch"
    else:
        theta0 = {k: v.detach().clone() for k, v in flat_params(modules).items()}
        torch.save({k: v.cpu() for k, v in theta0.items()}, theta_path)
        print(f"[e28] wrote init snapshot {theta_path}", flush=True)
    theta0 = {k: v.to(frame.device) for k, v in theta0.items()}
    groups = param_groups_map(list(theta0), len(modules["backbone"].blocks))

    train_ds = method.build_train_dataset()
    aug_batch = next(iter(DataLoader(train_ds, batch_size=256, shuffle=True, num_workers=8,
                                     generator=torch.Generator().manual_seed(0))))[0]
    g = torch.Generator().manual_seed(cfg.seed)
    train = DataLoader(train_ds, batch_size=cfg.bs, shuffle=True, drop_last=True,
                       num_workers=nw, generator=g, worker_init_fn=seed_worker,
                       pin_memory=cfg.pin_memory,
                       persistent_workers=cfg.persistent_workers and nw > 0,
                       prefetch_factor=cfg.prefetch_factor if nw > 0 else None)

    # optimizer axis (+e28.opt): the lazy/rich statement is a GRADIENT-DESCENT statement —
    # under GD the update carries the gradient's magnitude, so a ×α init suppresses relative
    # motion like 1/α². Adam divides by its own running RMS, which throws that magnitude away
    # and leaves only ~1/α; it is a weaker test of the claim. "sgd" = SGD+momentum on the
    # method's params. The PROBE always stays on its own AdamW: it is measurement apparatus,
    # not part of the method, and an SGD-trained monitor would under-read every arm. Splitting
    # it out is numerically identical for the AdamW arms (Adam state is per-parameter).
    opt_groups = method.param_groups(modules)      # NOT `groups` — that is the diagnostics map
    opt = (torch.optim.SGD(opt_groups, momentum=float(e28.get("momentum", 0.9)))
           if e28.get("opt") == "sgd" else torch.optim.AdamW(opt_groups))
    probe_opt = torch.optim.AdamW(probe.parameters(), lr=cfg.probe_lr,
                                  weight_decay=cfg.probe_wd)
    steps_per_epoch = len(train)
    scheduler = method.build_scheduler(opt, steps_per_epoch, steps_per_epoch * frame.epochs)
    scaler = GradScaler()

    start_ep, best_acc, wandb_id = 0, 0.0, None
    if cfg.resume and os.path.exists(last_path):
        pay = torch.load(last_path, map_location=frame.device, weights_only=False)
        saved_arch = {k: v for k, v in pay["arch"].items() if k != "probe"}
        assert saved_arch == method.arch(), (
            f"resume refused: {last_path} was trained with a different architecture. Delete the "
            f"stale run_id checkpoints or change tag= to start a fresh run.")
        for role, sd in pay["modules"].items():
            (probe if role == "probe" else modules[role]).load_state_dict(sd)
        opt.load_state_dict(pay["optim"]["opt"])
        if "probe_opt" in pay["optim"]:
            probe_opt.load_state_dict(pay["optim"]["probe_opt"])
        scheduler.load_state_dict(pay["optim"]["scheduler"])
        scaler.load_state_dict(pay["optim"]["scaler"])
        method.load_extras(pay.get("extras", {}))
        start_ep, best_acc = pay["epoch"] + 1, pay.get("best_acc") or 0.0
        wandb_id = pay["provenance"].get("wandb_id")
        print(f"[e28] resumed {run_id} at epoch {start_ep}", flush=True)

    run = wandb.init(project=cfg.wandb_project, name=run_id, id=wandb_id, resume="allow",
                     mode=cfg.wandb_mode, config=OmegaConf.to_container(cfg, resolve=True))

    def save(path, epoch):
        save_checkpoint(path, method=cfg.method.name, epoch=epoch,
                        step=(epoch + 1) * steps_per_epoch,
                        frame=OmegaConf.to_container(cfg.frame, resolve=True),
                        cfg=OmegaConf.to_container(cfg, resolve=True),
                        arch={**method.arch(),
                              "probe": {"class": "experiments.train.probe_arch",
                                        "kwargs": {"dim": method.probe_dim(),
                                                   "num_classes": cfg.num_classes}}},
                        modules={**{k: v for k, v in modules.items()}, "probe": probe},
                        extras=method.extras(),
                        optim={"opt": opt.state_dict(), "probe_opt": probe_opt.state_dict(),
                               "scheduler": scheduler.state_dict(),
                               "scaler": scaler.state_dict()},
                        best_acc=best_acc,
                        provenance=provenance_stamp(wandb_id=run.id, run_id=run_id,
                                                    seed=cfg.seed))

    cadence = set(frame.cadence()) | {int(e) for e in (cfg.extra_cadence or [])}
    gnorm_med, step = None, start_ep * steps_per_epoch
    if start_ep == 0:                                   # epoch 0 = the init read (θ = θ₀)
        run_diagnostics(run_id, 0, 0, modules, val, frame.device, theta0, groups,
                        aug_batch=aug_batch)
    for epoch in range(start_ep, frame.epochs):
        method.train_mode(modules)
        probe.train()
        method.on_epoch_start(modules, epoch)
        for batch_x, y in train:                        # aug=lejepa ⇒ views is a [N,V,C,H,W] tensor
            batch_x = batch_x.to(frame.device, non_blocking=True)
            y = y.to(frame.device, non_blocking=True)
            with autocast(frame.device, dtype=torch.bfloat16):
                terms, probe_feats, k = method.training_step(modules, batch_x, frame.device, y=y)
                y_rep = y.repeat_interleave(k) if k > 1 else y
                probe_loss = F.cross_entropy(probe(probe_feats), y_rep)
                loss = terms["loss"] + probe_loss
            opt.zero_grad()
            probe_opt.zero_grad()
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            scaler.unscale_(probe_opt)
            gn = torch.nn.utils.clip_grad_norm_(
                [p for o in (opt, probe_opt) for grp in o.param_groups
                 for p in grp["params"]],
                frame.grad_clip if frame.grad_clip else float("inf")).item()
            gnorm_med = gn if gnorm_med is None else 0.99 * gnorm_med + 0.01 * gn
            scaler.step(opt)
            scaler.step(probe_opt)
            if method._hold_on:
                method.apply_hold(modules)
            scaler.update()
            scheduler.step()
            step += 1
            wandb.log({**{f"train/{k_}": v.item() for k_, v in terms.items()},
                       "train/probe": probe_loss.item(), "train/grad_norm": gn,
                       "lr": scheduler.get_last_lr()[0]}, step=step)
            if gnorm_med and gn > 100 * gnorm_med:
                print(f"[e28] INCIDENT: grad_norm {gn:.1f} > 100x running median "
                      f"{gnorm_med:.3f} at step {step} (kill-trigger)", flush=True)

        if (epoch + 1) % cfg.eval_every == 0 or epoch + 1 == frame.epochs:
            for m in modules.values():
                m.eval()
            probe.eval()
            correct, n = 0, 0
            with torch.inference_mode():
                for views, y in val:
                    x = views.to(frame.device, non_blocking=True).flatten(0, 1)
                    y = y.to(frame.device, non_blocking=True)
                    with autocast(frame.device, dtype=torch.bfloat16):
                        logits = probe(method.eval_features(modules, x, frame.device))
                    correct += (logits.argmax(1) == y).sum().item()
                    n += y.numel()
            acc = correct / n
            wandb.log({"test/acc": acc, "test/epoch": epoch}, step=step)
            print(f"[e28] {run_id} ep{epoch + 1}/{frame.epochs} probe_acc={acc:.4f}", flush=True)
            if acc > best_acc:
                best_acc = acc
                save(f"{ckpt_base}_best.pt", epoch)
        run_diagnostics(run_id, epoch + 1, step, modules, val, frame.device, theta0,
                        groups, aug_batch=aug_batch)
        save(last_path, epoch)
        if (epoch + 1) in cadence:
            save(f"{ckpt_base}_ep{epoch + 1}.pt", epoch)
    wandb.finish()
    print(f"[e28] done {run_id}: best={best_acc:.4f}")


if __name__ == "__main__":
    main()
