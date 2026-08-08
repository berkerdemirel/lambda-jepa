"""D-051 dose check (Berker 2026-07-20: "since we changed the way we compute the loss,
hyperparam sensitivity can occur — briefly check if chosen params are good"). The carried
w_floor=45 was the hinge-arm convention (no bridge exists for a changed term-form); the
realized pull IS measurable: inv and h are byte-identical terms across forms (pinned-RNG
proof), so the whole question is g_z(view_mean)/g_z(pooled) at held states. Grid: the pooled
d256 cell's init + ep25 (the certified formation-state choice, E20 law) + ep50 (ratio
stability), each under BOTH z-floor forms with everything else fixed — same batch (seeded
loader) and same slice frame (d_draw parity: the vm 32-slice is the first-32-column subframe
of the very Q the pooled 128-slice uses at matched RNG) ⇒ common-random-numbers ratio.
g_enc = trunk-module-only (e12h_pull convention). Rows -> results/diag/e21_vmpull.csv.
Equal-pull prescription if the ratio is off: w'_z = 45 * g_z(pooled)/g_z(vm) at ep25. RAW."""
import csv
import os

import torch
from omegaconf import OmegaConf
from torch.amp import autocast
from torch.utils.data import DataLoader

import sys
sys.path.insert(0, "/nfs/scistore19/locatgrp/bdemirel/ssl_project")
from sslgap.data import seed_everything
from sslgap.methods import METHODS
from sslgap.methods.base import Frame

ROOT = "/nfs/scistore19/locatgrp/bdemirel/ssl_project"
OUT = f"{ROOT}/results/diag/e21_vmpull.csv"
STATES = [("init", None), ("ep25", "outputs/in100.floorssl.s0.d256_ep25.pt"),
          ("ep50", "outputs/in100.floorssl.s0.d256_ep50.pt")]
FORMS = [("pooled", {}), ("view_mean", {"z_floor_batch": "view_mean", "z_d_slice": 32})]


def main():
    dev = "cuda"
    base = torch.load(f"{ROOT}/outputs/in100.floorssl.s0.d256_ep25.pt",
                      map_location="cpu", weights_only=False)
    mcfg_ref = dict(base["cfg"]["method"])
    fr = base["cfg"]["frame"]
    rows = []
    for state, ck in STATES:
        for form, ovr in FORMS:
            cfg = OmegaConf.create({**mcfg_ref, **ovr})
            frame = Frame(name=fr["name"], model_name=fr["model_name"],
                          img_size=fr["img_size"], dataset=fr["dataset"],
                          data_root=fr["data_root"], epochs=fr["epochs"], seed=0,
                          grad_clip=1.0, num_workers=0, device=dev)
            seed_everything(0)
            method = METHODS["floorssl"](cfg, frame)
            modules = method.build_modules().to(dev)
            ep = 0
            if ck:
                pay = torch.load(f"{ROOT}/{ck}", map_location="cpu", weights_only=False)
                for role, sd in pay["modules"].items():
                    if role != "probe":
                        modules[role].load_state_dict(sd)
                method.load_extras(pay.get("extras", {}))
                ep = pay["epoch"] + 1
            loader = DataLoader(method.build_train_dataset(), batch_size=128, shuffle=True,
                                drop_last=True, num_workers=0,
                                generator=torch.Generator().manual_seed(0))
            views, y = next(iter(loader))
            params = [p for p in modules["backbone"].parameters() if p.requires_grad]
            torch.manual_seed(4242)          # matched slice frame + drop_path across forms
            with autocast(dev, dtype=torch.bfloat16):
                terms, _, _ = method.training_step(modules, views.to(dev), dev, y=y.to(dev))
            names = [k for k in terms if k != "loss"]
            for i, k in enumerate(names):
                g = torch.autograd.grad(terms[k], params, retain_graph=i < len(names) - 1,
                                        allow_unused=True)
                g = torch.cat([x.reshape(-1).float() for x in g if x is not None]).norm().item()
                rows.append({"state": state, "ep": ep, "form": form, "term": k,
                             "g_enc": round(g, 6), "loss_value": round(float(terms[k]), 6),
                             "slurm_job": os.environ.get("SLURM_JOB_ID", "")})
                print(f"{state:6s} ep{ep:<3d} {form:10s} {k:12s} g_enc={g:10.4f} "
                      f"loss={float(terms[k]):.4f}", flush=True)
    with open(OUT, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print("wrote", OUT, flush=True)


if __name__ == "__main__":
    main()
