"""Adapters lifting legacy checkpoint formats into LoadedCkpt (schema.py).

M0 sources:
- lejepa minimal (../lejepa/ckpt_*.pt, keys net/probe/cfg/epoch): net = timm ViT with
  num_classes=512 ("backbone.*", whose head.* Linear IS the recipe's 512-d "emb" — our z.embed,
  D-003) + torchvision-MLP projector ("proj.*").
- ssl_explore DINO control (outputs/inv_dino-in100_ep*.pt): student ViTEncoder ("net": backbone.*
  incl. an UNTRAINED 384->512 head and an UNTRAINED proj.* — both receive no loss gradient in
  train_dinov2.py and are excluded here), DINOHead ("head"), EMA teacher ("teacher_bb" raw timm sd,
  "teacher_head"). Probed branch per DINO protocol: teacher.

Every adapter takes `random_init=True` to build the SAME architecture freshly seeded — the
random-init null of PROTOCOL §6.6.
"""
import random

import torch
import torch.nn as nn
from torchvision.ops import MLP

from sslgap.ckpt.schema import Branch, LoadedCkpt
from sslgap.methods.ijepa import MaskSampler
from sslgap.models.backbones import build_vit_trunk
from sslgap.models.heads import (ByolHeads, DINOHead, DinoHeadTaps, LejepaHeads, LinearTap,
                                 TVMLPTaps)
from sslgap.models.vitops import vit_tokens


def _split_prefix(sd, prefix):
    return {k[len(prefix):]: v for k, v in sd.items() if k.startswith(prefix)}


def _trunk_from_vit_sd(vit_sd, model_name, img_size, dynamic_img_size, drop_path=0.1):
    """vit_sd = a timm ViT state dict possibly containing classifier head.* keys.
    Returns (trunk with num_classes=0 loaded, the head Linear or None)."""
    trunk = build_vit_trunk(model_name, img_size, dynamic_img_size, drop_path_rate=drop_path)
    head_w, head_b = vit_sd.get("head.weight"), vit_sd.get("head.bias")
    trunk_sd = {k: v for k, v in vit_sd.items() if not k.startswith("head.")}
    missing, unexpected = trunk.load_state_dict(trunk_sd, strict=False)
    if unexpected:
        raise ValueError(f"unexpected trunk keys: {unexpected[:5]}")
    if [m for m in missing if "head" not in m]:
        raise ValueError(f"missing trunk keys: {missing[:5]}")
    embed = None
    if head_w is not None:
        embed = nn.Linear(head_w.shape[1], head_w.shape[0], bias=head_b is not None)
        embed.weight.data.copy_(head_w)
        if head_b is not None:
            embed.bias.data.copy_(head_b)
    return trunk, embed


def from_lejepa_minimal(path, run_id, random_init=False, seed=0):
    ck = torch.load(path, map_location="cpu", weights_only=False)
    cfg = dict(ck["cfg"])
    model_name = cfg.get("model_name", "vit_small_patch8_224")
    img_size = cfg.get("img_size", 128)
    dyn = cfg.get("dynamic_img_size", False)
    proj_dim = cfg["proj_dim"]
    if random_init:
        torch.manual_seed(seed)
        trunk = build_vit_trunk(model_name, img_size, dyn, drop_path_rate=0.1)
        embed = nn.Linear(384, 512)
        proj = MLP(512, [2048, 2048, proj_dim], norm_layer=nn.BatchNorm1d)
    else:
        net_sd = ck["net"]
        trunk, embed = _trunk_from_vit_sd(_split_prefix(net_sd, "backbone."),
                                          model_name, img_size, dyn)
        proj = MLP(512, [2048, 2048, proj_dim], norm_layer=nn.BatchNorm1d)
        proj.load_state_dict(_split_prefix(net_sd, "proj."))
    heads = LejepaHeads(embed, proj)
    prov = {"source": str(path), "epoch": ck.get("epoch"), "adapter": "lejepa_minimal",
            "random_init": random_init, "seed": seed if random_init else None}
    return LoadedCkpt(run_id=run_id, method="lejepa",
                      frame={"model_name": model_name, "img_size": img_size,
                             "dynamic_img_size": dyn, "dataset": cfg.get("dataset", "imagenette")},
                      cfg=cfg, branches={"student": Branch(trunk, heads, "cls")},
                      probed_branch="student", provenance=prov)


def from_sslx_dino(path, run_id, random_init=False, seed=0):
    ck = torch.load(path, map_location="cpu", weights_only=False)
    cfg = dict(ck["cfg"])
    model_name, img_size = cfg["model_name"], cfg["img_size"]
    dyn = cfg.get("dynamic_img_size", False)
    hd = dict(in_dim=384, hidden=cfg["head_hidden"], bottleneck=cfg["bottleneck"],
              K=cfg["K"], norm_last_layer=cfg.get("norm_last_layer", True))

    def build_branch(vit_sd, head_sd):
        trunk, _ = _trunk_from_vit_sd(vit_sd, model_name, img_size, dyn)  # untrained embed dropped
        head = DINOHead(**hd)
        head.load_state_dict(head_sd)
        return Branch(trunk, DinoHeadTaps(head), "cls")

    if random_init:
        torch.manual_seed(seed)
        branches = {"student": Branch(build_vit_trunk(model_name, img_size, dyn, 0.1),
                                      DinoHeadTaps(DINOHead(**hd)), "cls")}
    else:
        branches = {"student": build_branch(_split_prefix(ck["net"], "backbone."), ck["head"]),
                    "teacher": build_branch(ck["teacher_bb"], ck["teacher_head"])}
    prov = {"source": str(path), "epoch": ck.get("epoch"), "adapter": "sslx_dino",
            "random_init": random_init, "seed": seed if random_init else None,
            "note": "student net.proj + net.backbone.head were untrained in this trainer; excluded"}
    return LoadedCkpt(run_id=run_id, method="dino",
                      frame={"model_name": model_name, "img_size": img_size,
                             "dynamic_img_size": dyn, "dataset": cfg.get("dataset", "imagenet"),
                             "data_root": cfg.get("data_root")},
                      cfg=cfg,
                      branches=branches,
                      probed_branch="student" if random_init else "teacher",
                      provenance=prov)


class MaeDecTaps(nn.Module):
    """MAE z-taps at mask-ratio 0 (PROTOCOL §3): the full normed encoder sequence is the decoder
    input — identical to a training pass with every patch visible (the scatter is the identity,
    mask token unused; vit_tokens(keep=all) == forward_features per vitops_self_test). Taps =
    patch-token means after decoder blocks {2,5,8}, cls excluded (h.gap convention), raw block
    outputs; the norm->pred pixel head is the loss space, handled metric-by-metric."""

    def __init__(self, decoder, tap_blocks=(2, 5, 8)):
        super().__init__()
        self.dec = decoder
        self.tap_blocks = set(tap_blocks)

    def forward(self, seq):
        z = self.dec.embed(seq)
        z = z + self.dec.pos.to(z.dtype)
        out = {}
        for k, blk in enumerate(self.dec.blocks, 1):
            z = blk(z)
            if k in self.tap_blocks:
                out[f"dec.tap{k}"] = z[:, 1:].mean(1)
        return out


class IjepaPredTaps(nn.Module):
    """I-JEPA z (PROTOCOL §3): predictor outputs mean-pooled over all target tokens, computed
    from a context-only trunk pass exactly as trained. ONE mask layout (seed 0), sampled at
    construction under forked RNG and shared by every image — stored features must not depend
    on batch composition or order."""

    def __init__(self, trunk, predictor, grid, n_targets, seed=0):
        super().__init__()
        self.trunk = trunk
        self.pred = predictor
        state = random.getstate()                 # MaskSampler draws from global random + torch
        random.seed(seed)
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed)
            ctx, tgt = MaskSampler(grid=grid, n_targets=n_targets)(1)
        random.setstate(state)
        self.register_buffer("ctx_idx", ctx)      # [1, Kc]
        self.register_buffer("tgt_idx", tgt)      # [1, T, Kt]

    def forward(self, x):
        B = x.shape[0]
        ctx_idx = self.ctx_idx.expand(B, -1)
        ctx = vit_tokens(self.trunk, x, keep=ctx_idx)[:, 1:]
        pred = self.pred(ctx, ctx_idx, self.tgt_idx.expand(B, -1, -1))
        return {"pred.out": pred.mean((1, 2))}


def _asm_projector(mods, ck):
    """simclr / vicreg: single branch, projector taps off trunk CLS (what the trainer fed it);
    h = student trunk-CLS = the projector input (D-036; was trunk-GAP under F1)."""
    return ({"student": Branch(mods["backbone"], TVMLPTaps(mods["projector"], "proj"), "cls")},
            "student", "student.h.cls")


def _asm_byol(mods, ck):
    student = Branch(mods["backbone"], ByolHeads(mods["projector"], mods["predictor"]), "cls")
    teacher = Branch(mods["teacher_backbone"],
                     TVMLPTaps(mods["teacher_projector"], "proj"), "cls")  # target space, stored
    return ({"student": student, "teacher": teacher}, "student", "student.h.cls")  # D-036: projector input (was GAP)


def _asm_dino(mods, ck):
    return ({"student": Branch(mods["backbone"], DinoHeadTaps(mods["projector"]), "cls"),
             "teacher": Branch(mods["teacher_backbone"],
                               DinoHeadTaps(mods["teacher_projector"]), "cls")},
            "teacher", "teacher.h.cls")          # F2: teacher last-layer CLS, no concat


def _asm_mae(mods, ck):
    return ({"student": Branch(mods["backbone"], MaeDecTaps(mods["decoder"]), "seq")},
            "student", "student.h.gap")


def _asm_ijepa(mods, ck):
    grid = int(ck["arch"]["predictor"]["kwargs"]["n_patches"] ** 0.5)
    heads = IjepaPredTaps(mods["backbone"], mods["predictor"], grid,
                          n_targets=ck["cfg"]["method"]["n_targets"])
    return ({"student": Branch(mods["backbone"], heads, "image"),
             "teacher": Branch(mods["teacher_backbone"])},   # h only: no z on the EMA target
            "teacher", "teacher.h.gap")         # F2: teacher last-layer avgpooled patches


def _asm_pivot(mods, ck):
    """E15 head-less MVI: single branch, no heads — the trained block IS h (D-032)."""
    return ({"student": Branch(mods["backbone"])}, "student", "student.h.gap")


def _asm_deitlite(mods, ck):
    """Supervised anchor: single branch; h = last-layer CLS (classifier input, D-003v2);
    z.logits = the CE loss space."""
    return ({"student": Branch(mods["backbone"], LinearTap(mods["classifier"]), "cls")},
            "student", "student.h.cls")


def _asm_lejepa(mods, ck):
    # encoder = timm ViT WITH the emb Linear (exact port); split into trunk + embed for the
    # two-space layout — z.embed is LeJEPA's h (D-003v2 F4).
    fr = ck["frame"]
    trunk, embed = _trunk_from_vit_sd(mods["encoder"].state_dict(), fr["model_name"],
                                      fr["img_size"], fr.get("dynamic_img_size", False))
    return ({"student": Branch(trunk, LejepaHeads(embed, mods["projector"]), "cls")},
            "student", "student.z.embed")


_NATIVE_ASM = {"simclr": _asm_projector, "vicreg": _asm_projector, "byol": _asm_byol,
               "dino": _asm_dino, "mae": _asm_mae, "ijepa": _asm_ijepa, "lejepa": _asm_lejepa, "deitlite": _asm_deitlite, "pivot": _asm_pivot}


def _resolve(dotted):
    mod, _, attr = dotted.rpartition(".")
    import importlib
    return getattr(importlib.import_module(mod), attr)


def from_native(path, run_id, random_init=False, seed=0):
    """sslgap/ckpt/v1 payloads (our M1+ trainers): rebuild every module from its arch block, then
    assemble branches per method (PROTOCOL §3 h/z + D-003v2 F-rulings). random_init rebuilds the
    same arch freshly seeded; teachers then copy their student counterparts — every trainer
    initializes EMA branches by deepcopy, so the epoch-0 null has teacher == student."""
    ck = torch.load(path, map_location="cpu", weights_only=False)
    if ck.get("format") != "sslgap/ckpt/v1":
        raise ValueError(f"not a native ckpt: {ck.get('format')!r} ({path})")
    if random_init:
        torch.manual_seed(seed)
    mods = {}
    for role, spec in ck["arch"].items():
        m = _resolve(spec["class"])(**spec["kwargs"])
        if not random_init:
            m.load_state_dict(ck["modules"][role])
        mods[role] = m
    if random_init:
        for role in mods:
            if role.startswith("teacher_"):
                mods[role].load_state_dict(mods[role.removeprefix("teacher_")].state_dict())

    branches, probed, h_space = _NATIVE_ASM[ck["method"]](mods, ck)

    prov = {"source": str(path), "epoch": ck.get("epoch"), "adapter": "native",
            "random_init": random_init, "seed": seed if random_init else None,
            "h_space": h_space, "train_provenance": ck.get("provenance")}
    return LoadedCkpt(run_id=run_id, method=ck["method"],
                      frame={**ck["frame"], "dynamic_img_size": ck["frame"].get("dynamic_img_size",
                                                                                False)},
                      cfg=ck["cfg"], branches=branches, probed_branch=probed, provenance=prov)


ADAPTERS = {"lejepa_minimal": from_lejepa_minimal, "sslx_dino": from_sslx_dino,
            "native": from_native}


def load(adapter, path, run_id, **kw):
    return ADAPTERS[adapter](path, run_id, **kw)
