"""E23 B-L1 rev3 — the toy lane bridge on the CERTIFIED formation state (e23y256leg ep38,
the healthy .745 cell; the first pull ran on the sick rev2 anchor and is discarded as
state-confounded). Swap ONLY the aug pipeline — per-term trunk-gradient norms (g_enc,
trunk-module-only) under the byol pair vs the lejepa V=4 family, both at bs=128. Reported
per Berker's dominance lens (2026-07-30: "regularization dominating inv"): raw g_enc AND
the realized WEIGHTED pull w_t·g_t with its share of the total — the healthy byol share
profile is the dose-transfer target (w'_t on lejepa set to reproduce it). Rows ->
results/diag/e23_pull.csv (append; state column disambiguates)."""
import csv
import os

import torch
from omegaconf import OmegaConf
from torch.amp import autocast
from torch.utils.data import DataLoader

from sslgap.data import seed_everything
from sslgap.methods import METHODS
from sslgap.methods.base import Frame

ROOT = "/nfs/scistore19/locatgrp/bdemirel/ssl_project"
OUT = f"{ROOT}/results/diag/e23_pull.csv"
CK = "outputs/toy.floorssl.s0.e23y256leg_ep38.pt"
W = {"inv": "w_inv", "moment_kl": "w_floor", "h_moment_kl": "h_lamb"}


def main():
    dev = "cuda"
    base = torch.load(f"{ROOT}/{CK}", map_location="cpu", weights_only=False)
    mcfg_ref = dict(base["cfg"]["method"])
    fr = base["cfg"]["frame"]
    rows = []
    for aug in ("byol", "lejepa"):
        mcfg = dict(mcfg_ref)
        if aug == "lejepa":
            mcfg.update(aug="lejepa", V=4)
        else:
            mcfg.pop("aug", None), mcfg.pop("V", None)
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
        views, y = next(iter(loader))
        params = [p for p in modules["backbone"].parameters() if p.requires_grad]
        with autocast(dev, dtype=torch.bfloat16):
            terms, _, _ = method.training_step(modules, views.to(dev), dev, y=y.to(dev))
        names = [k for k in terms if k != "loss"]
        gs = {}
        for i, k in enumerate(names):
            g = torch.autograd.grad(terms[k], params, retain_graph=i < len(names) - 1,
                                    allow_unused=True)
            gs[k] = torch.cat([x.reshape(-1).float() for x in g
                               if x is not None]).norm().item()
        tot = sum(float(mcfg_ref[W[k]]) * gs[k] for k in names)
        for k in names:
            w = float(mcfg_ref[W[k]])
            rows.append({"state": "y256leg_ep38", "ep": base["epoch"] + 1, "aug": aug,
                         "term": k, "g_enc": round(gs[k], 6), "w": w,
                         "wg": round(w * gs[k], 4), "share": round(w * gs[k] / tot, 4),
                         "loss_value": round(float(terms[k]), 6),
                         "slurm_job": os.environ.get("SLURM_JOB_ID", "")})
            print(f"y256leg_ep38 {aug:7s} {k:12s} g={gs[k]:8.4f} wg={w*gs[k]:9.2f} "
                  f"share={w*gs[k]/tot:.3f} loss={float(terms[k]):.4f}", flush=True)
    with open(OUT, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print("wrote", OUT, flush=True)


if __name__ == "__main__":
    main()
