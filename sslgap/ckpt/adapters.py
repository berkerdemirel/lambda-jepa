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
import torch
import torch.nn as nn
from torchvision.ops import MLP

from sslgap.ckpt.schema import Branch, LoadedCkpt
from sslgap.models.backbones import build_vit_trunk
from sslgap.models.heads import DINOHead, DinoHeadTaps, LejepaHeads


def _split_prefix(sd, prefix):
    return {k[len(prefix):]: v for k, v in sd.items() if k.startswith(prefix)}


def _trunk_from_vit_sd(vit_sd, model_name, img_size, dynamic_img_size, drop_path=0.1):
    """vit_sd = a timm ViT state dict possibly containing classifier head.* keys.
    Returns (trunk with num_classes=0 loaded, the head Linear or None)."""
    trunk = build_vit_trunk(model_name, img_size, dynamic_img_size, drop_path_rate=drop_path)
    head_w, head_b = vit_sd.get("head.weight"), vit_sd.get("head.bias")
    trunk_sd = {k: v for k, v in vit_sd.items() if not k.startswith("head.")}
    missing, unexpected = trunk.load_state_dict(trunk_sd, strict=False)
    assert not unexpected, f"unexpected trunk keys: {unexpected[:5]}"
    assert not [m for m in missing if "head" not in m], f"missing trunk keys: {missing[:5]}"
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


ADAPTERS = {"lejepa_minimal": from_lejepa_minimal, "sslx_dino": from_sslx_dino}


def load(adapter, path, run_id, **kw):
    return ADAPTERS[adapter](path, run_id, **kw)
