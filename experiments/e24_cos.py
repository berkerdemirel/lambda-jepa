"""E24-vm coordinate question — force GEOMETRY at matched formation states: per-term
trunk-gradient vectors -> pairwise cosines + naive shares, measured on a SEQUENCE of
batches so the vm4 queue ring warms exactly as in training (batch 0 = cold queue =
the e24_pull.py convention that produced the .87 vm4 rows; batches >= queue_steps =
warm ring = the operating estimator, n_eff/d' restored). Discriminates the two E24-vm
hypotheses in one instrument: (competition) pooled floors should oppose inv
(cos(g_inv, g_zfloor) < 0) while view-mean floors are ~orthogonal — at z-level this is
exact (view-mean floor's z-grad is constant across views = pure center force; inv's is
mean-zero across views = pure scatter force); the cosine tests whether it survives the
shared trunk. (artifact) the cold ring reads n/d' = 1 -> inflated conditioner loss/grad;
warm-vs-cold rows quantify it. Frozen-model ring rows are same-distribution (no stale-
model lag — declared deviation from training's moving ring). bs=128 fixed (house).
Usage: python e24_cos.py <label> <ckpt> [...]  (pairs repeat; formation ckpts).
Appends -> results/diag/e24_cos.csv. RAW."""
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
from sslgap.ckpt.schema import load_payload

ROOT = "/nfs/scistore19/locatgrp/bdemirel/ssl_project"
OUT = f"{ROOT}/results/diag/e24_cos.csv"
W = {"inv": "w_inv", "moment_kl": "w_floor", "h_moment_kl": "h_lamb"}
NB = 6


def measure(label, ck_path, dev="cuda"):
    base = load_payload(ck_path, map_location="cpu")
    mcfg = dict(base["cfg"]["method"])
    fr = base["cfg"]["frame"]
    cfg = OmegaConf.create(mcfg)
    frame = Frame(name=fr["name"], model_name=fr["model_name"], img_size=fr["img_size"],
                  dataset=fr["dataset"], data_root=fr["data_root"], epochs=fr["epochs"],
                  seed=0, grad_clip=1.0, num_workers=0, device=dev)
    seed_everything(0)
    method = METHODS["lambdajepa"](cfg, frame)
    modules = method.build_modules().to(dev)
    for role, sd in base["modules"].items():
        if role != "probe":
            modules[role].load_state_dict(sd)
    method.load_extras(base.get("extras", {}))
    loader = DataLoader(method.build_train_dataset(), batch_size=128, shuffle=True,
                        drop_last=True, num_workers=0,
                        generator=torch.Generator().manual_seed(0))
    params = [p for p in modules["backbone"].parameters() if p.requires_grad]
    anat = {"zb": mcfg.get("z_floor_batch", "pooled"),
            "hb": mcfg.get("h_floor_batch", "pooled"),
            "q": int(mcfg.get("queue_steps", 0) or 0)}
    ep = base.get("epoch", -1)
    it = iter(loader)
    rows = []
    for b in range(NB):
        qfill = len(getattr(method, "_zq", []))
        views, y = next(it)
        with autocast(dev, dtype=torch.bfloat16):
            terms, _, _ = method.training_step(modules, views.to(dev), dev, y=y.to(dev))
        names = [k for k in terms if k != "loss"]
        vec = {}
        for i, k in enumerate(names):
            g = torch.autograd.grad(terms[k], params, retain_graph=i < len(names) - 1,
                                    allow_unused=True)
            vec[k] = torch.cat([(x if x is not None else torch.zeros_like(p)).reshape(-1)
                                for x, p in zip(g, params)]).float()
        gn = {k: vec[k].norm().item() for k in names}
        w = {k: float(mcfg[W[k]]) for k in names}
        tot = sum(w[k] * gn[k] for k in names)
        # f64 dot: fp32 over 21M dims drifts ~.5% (the e21_vmcos control lesson)
        cos = {f"cos_{a}_{c}": float((vec[a].double() @ vec[c].double())
                                     / (vec[a].double().norm() * vec[c].double().norm()))
               for a, c in [("inv", "moment_kl"), ("inv", "h_moment_kl"),
                            ("moment_kl", "h_moment_kl")]}
        del vec
        rows.append({"label": label, "ckpt": os.path.basename(ck_path), "ep": ep,
                     "aug": mcfg.get("aug", "byol"), "V": mcfg.get("V", 2), **anat,
                     "batch": b, "qfill": qfill,
                     **{f"g_{k}": round(gn[k], 6) for k in names},
                     **{f"s_{k}": round(w[k] * gn[k] / tot, 4) for k in names},
                     **{k: round(v, 4) for k, v in cos.items()},
                     "wg_total": round(tot, 4),
                     **{f"loss_{k}": round(float(terms[k]), 6) for k in names},
                     "slurm_job": os.environ.get("SLURM_JOB_ID", "")})
        print(f"{label:12s} b{b} qfill={qfill} "
              + " ".join(f"s_{k}={w[k] * gn[k] / tot:.3f}(g{gn[k]:.3f})" for k in names)
              + " | " + " ".join(f"{k}={v:+.3f}" for k, v in cos.items()), flush=True)
    return rows


def main():
    args = sys.argv[1:]
    assert args and len(args) % 2 == 0, "pairs: <label> <ckpt>"
    rows = []
    for i in range(0, len(args), 2):
        rows += measure(args[i], args[i + 1])
    write_header = not os.path.exists(OUT)
    with open(OUT, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        if write_header:
            w.writeheader()
        w.writerows(rows)
    print("appended", OUT, flush=True)


if __name__ == "__main__":
    main()
