"""E24 wave 0 — the generalized w·g pull instrument: per-term trunk-gradient norms and
realized shares for ANY floorssl checkpoint (formation state) or its fresh init, both
frames. Usage: python e24_pull.py <label> <ckpt> <state: ckpt|init> [...]  (triples repeat).
The nominal→realized transform these rows define is the wave-1 dosing basis (w_i ∝
s_i^target/g_i, total Σw·g held at the certified level). bs=128 fixed (house convention;
shares are ratios). Appends → results/diag/e24_pull.csv. RAW."""
import csv
import os
import sys

import torch
from omegaconf import OmegaConf
from torch.amp import autocast
from torch.utils.data import DataLoader

from sslgap.data import seed_everything
from sslgap.methods import METHODS
from sslgap.methods.base import Frame

ROOT = "/nfs/scistore19/locatgrp/bdemirel/ssl_project"
OUT = f"{ROOT}/results/diag/e24_pull.csv"
W = {"inv": "w_inv", "moment_kl": "w_floor", "h_moment_kl": "h_lamb"}


def measure(label, ck_path, state, dev="cuda"):
    base = torch.load(ck_path, map_location="cpu", weights_only=False)
    mcfg = dict(base["cfg"]["method"])
    fr = base["cfg"]["frame"]
    cfg = OmegaConf.create(mcfg)
    frame = Frame(name=fr["name"], model_name=fr["model_name"], img_size=fr["img_size"],
                  dataset=fr["dataset"], data_root=fr["data_root"], epochs=fr["epochs"],
                  seed=0, grad_clip=1.0, num_workers=0, device=dev)
    seed_everything(0)
    method = METHODS["floorssl"](cfg, frame)
    modules = method.build_modules().to(dev)
    if state == "ckpt":
        for role, sd in base["modules"].items():
            if role != "probe":
                modules[role].load_state_dict(sd)
        method.load_extras(base.get("extras", {}))
    loader = DataLoader(method.build_train_dataset(), batch_size=128, shuffle=True,
                        drop_last=True, num_workers=0,
                        generator=torch.Generator().manual_seed(0))
    views, y = next(iter(loader))
    params = [p for p in modules["backbone"].parameters() if p.requires_grad]
    with autocast(dev, dtype=torch.bfloat16):
        terms, _, _ = method.training_step(modules, views.to(dev), dev, y=y.to(dev))
    names = [k for k in terms if k != "loss"]
    gs, rows = {}, []
    for i, k in enumerate(names):
        g = torch.autograd.grad(terms[k], params, retain_graph=i < len(names) - 1,
                                allow_unused=True)
        gs[k] = torch.cat([x.reshape(-1).float() for x in g if x is not None]).norm().item()
    tot = sum(float(mcfg[W[k]]) * gs[k] for k in names)
    ep = base.get("epoch", -1) if state == "ckpt" else -1
    for k in names:
        w = float(mcfg[W[k]])
        rows.append({"label": label, "state": state, "ckpt": os.path.basename(ck_path),
                     "aug": mcfg.get("aug", "byol"), "V": mcfg.get("V", 2),
                     "ep": ep, "term": k, "g_enc": round(gs[k], 6), "w": w,
                     "wg": round(w * gs[k], 4), "share": round(w * gs[k] / tot, 4),
                     "wg_total": round(tot, 4), "loss_value": round(float(terms[k]), 6),
                     "slurm_job": os.environ.get("SLURM_JOB_ID", "")})
        print(f"{label:16s} {state:5s} {k:12s} g={gs[k]:9.4f} wg={w*gs[k]:10.2f} "
              f"share={w*gs[k]/tot:.3f} loss={float(terms[k]):.4f}", flush=True)
    return rows


def main():
    args = sys.argv[1:]
    assert args and len(args) % 3 == 0, "triples: <label> <ckpt> <ckpt|init>"
    rows = []
    for i in range(0, len(args), 3):
        rows += measure(args[i], args[i + 1], args[i + 2])
    write_header = not os.path.exists(OUT)
    with open(OUT, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        if write_header:
            w.writeheader()
        w.writerows(rows)
    print("appended", OUT, flush=True)


if __name__ == "__main__":
    main()
