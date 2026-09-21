"""Checkpoint schema v1 + the uniform in-memory form every checkpoint loads into.

Two layers:
- `LoadedCkpt` — what the extraction/audit pipeline consumes. Adapters (adapters.py) lift legacy
  formats (lejepa minimal, ssl_explore DINO control, later solo-learn/official releases) into it;
  our own trainers (M1+) will save the native v1 payload below.
- native payload `sslgap/ckpt/v1` — the head-preserving format for our trainers: every module role
  (backbone, projector, predictor, decoder, teacher_*) with an `arch` block (class + kwargs) so any
  head can be rebuilt WITHOUT the trainer, plus full cfg and provenance. This is what makes the
  audit possible on our checkpoints and impossible on most public ones.
"""
import json
import os
import subprocess
from dataclasses import dataclass, field

import torch
import torch.nn as nn

FORMAT = "sslgap/ckpt/v1"


@dataclass
class Branch:
    """One weight branch (student, or an EMA teacher). `trunk` must expose forward_features()
    (timm ViT convention, num_classes=0 semantics — the classifier head is never part of h).
    `heads` maps the branch's trunk features to named z-taps: a module whose forward(feature)
    returns {tap_name: tensor}; `head_input` says which trunk feature it consumes: "cls" | "gap" |
    "seq" (full normed token sequence, MAE decoder) | "image" (raw batch — heads running their own
    trunk pass, I-JEPA context-only)."""
    trunk: nn.Module
    heads: nn.Module | None = None
    head_input: str = "cls"          # "cls" | "gap" | "seq" | "image"


@dataclass
class LoadedCkpt:
    run_id: str
    method: str
    frame: dict                      # model_name, img_size, dynamic_img_size, dataset, num_classes
    cfg: dict                        # full source cfg (verbatim)
    branches: dict[str, Branch]      # "student" [, "teacher"]
    probed_branch: str               # the branch the method's paper evaluates (PROTOCOL §3)
    provenance: dict = field(default_factory=dict)

    def eval_(self, device):
        for br in self.branches.values():
            br.trunk.to(device).eval().requires_grad_(False)
            if br.heads is not None:
                br.heads.to(device).eval().requires_grad_(False)
        return self


def provenance_stamp(**extra):
    """Best-effort provenance for payloads and feature-store meta."""
    try:
        sha = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True,
                             cwd=os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
                             ).stdout.strip()
    except OSError:
        sha = ""
    return {"git_sha": sha, "slurm_job_id": os.environ.get("SLURM_JOB_ID", ""),
            "host": os.uname().nodename, "torch": torch.__version__, **extra}


def save_checkpoint(path, *, method, epoch, step, frame, cfg, arch, modules, extras=None,
                    optim=None, meters=None, best_acc=None, provenance=None):
    """Native v1 payload (used by our M1+ trainers). `arch[role] = {"class": "sslgap.models...",
    "kwargs": {...}}` must rebuild each module in `modules[role]` without the trainer."""
    payload = {"format": FORMAT, "method": method, "epoch": epoch, "step": step,
               "frame": frame, "cfg": cfg, "arch": arch,
               "modules": {k: v.state_dict() if isinstance(v, nn.Module) else v
                           for k, v in modules.items()},
               "extras": extras or {}, "optim": optim or {}, "meters": meters or {},
               "best_acc": best_acc, "provenance": provenance or provenance_stamp()}
    errs = validate(payload)
    if errs:
        raise ValueError(f"refusing to save invalid ckpt: {errs}")
    torch.save(payload, path)


REQUIRED_ROLES = {  # minimal module set per method for the native format (M1 trainers)
    "simclr": {"backbone", "projector"},
    "lambdajepa": {"backbone", "projector"},
    "byol": {"backbone", "projector", "predictor", "teacher_backbone", "teacher_projector"},
    "vicreg": {"backbone", "projector"},
    "dino": {"backbone", "projector", "teacher_backbone", "teacher_projector"},
    "mae": {"backbone", "decoder"},
    "ijepa": {"backbone", "predictor", "teacher_backbone"},
    "lejepa": {"encoder", "projector"},   # encoder = timm ViT WITH the emb Linear (exact port);
                                          # the extraction adapter splits trunk/embed (D-003v2 F4)
}


# 2026-09-21 rename (D-127): the method `floorssl` is `lambdajepa`, the regularizer class
# `SpectralConditioner` is `SACReg`, the zoo's backbone hook `h_reg=moment` is `h_reg=sacreg`.
# Checkpoints written before that day carry the old names in `method`, the arch class paths and
# the stored cfg; `modernize` maps them on load. Run ids, file names and stored provenance stay.
LEGACY_METHOD = {"floorssl": "lambdajepa"}
LEGACY_CLASS = {f"sslgap.methods.floorssl.floorssl_{k}": f"sslgap.methods.lambdajepa.lambdajepa_{k}"
                for k in ("head", "ladder_head", "res_head", "stage_head")}
LEGACY_REG = {"moment": "sacreg"}


def modernize(payload):
    payload["method"] = LEGACY_METHOD.get(payload.get("method"), payload.get("method"))
    for spec in payload.get("arch", {}).values():
        spec["class"] = LEGACY_CLASS.get(spec["class"], spec["class"])
    m = (payload.get("cfg") or {}).get("method") or {}
    if m.get("name") in LEGACY_METHOD:
        m["name"] = LEGACY_METHOD[m["name"]]
    for k in ("h_reg", "floor"):
        if m.get(k) in LEGACY_REG:
            m[k] = LEGACY_REG[m[k]]
    return payload


def load_payload(path, **kw):
    """torch.load of a native payload with the legacy names mapped (every reader goes through here)."""
    return modernize(torch.load(path, weights_only=False, **kw))


def validate(payload):
    errs = []
    if payload.get("format") != FORMAT:
        errs.append(f"format != {FORMAT}")
    for k in ("method", "frame", "cfg", "arch", "modules", "provenance"):
        if k not in payload:
            errs.append(f"missing key {k}")
    need = REQUIRED_ROLES.get(payload.get("method"), set())
    have = set(payload.get("modules", {}))
    if not need <= have:
        errs.append(f"method {payload.get('method')}: missing roles {need - have}")
    missing_arch = set(payload.get("modules", {})) - set(payload.get("arch", {}))
    if missing_arch:
        errs.append(f"modules without arch block: {missing_arch}")
    return errs


def write_meta(path, meta):
    with open(path, "w") as f:
        json.dump(meta, f, indent=1, default=str)
