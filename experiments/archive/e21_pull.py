"""E21 collapse fix — the frame-bridge pull (Berker 2026-07-19: "floorssl runs collapsed.
fix them."). Both lejepa-aug cells hit the pre-registered kill (probe ≤2x chance, ep>=10)
via a SMOOTH equilibrium that sacrifices inv (no incident; v2 = same doses under the byol
pair was healthy and climbing) — the verbatim-dose bet across aug frames failed, E19-T1's
error class. Init pulls cannot carry the fix: a random trunk maps both aug families to
near-identical term values (measured: inv .234 vs .224) — the adversary difference lives at
FORMATION states, the same reason the E20 law measured at ep25 controls.

Bridge: hold the HEALTHY formation state fixed (v2_best, ep6; byte-proven loadable into the
floorssl class) and swap ONLY the aug pipeline — per-term trunk-gradient norms under the
byol pair vs the lejepa V=4 family. Equal-pull re-dose per term: w'_t = w_t * g_t(byol) /
g_t(lejepa) keeps every term's realized pull at its healthy-lineage value under the new
frame. Also measured: fresh init under both frames (the null that motivated this) and the
laug _best (ep3, pre-collapse peak) for the trajectory record. g_enc = trunk-module-only
(e12h_pull convention). Rows -> results/diag/e21_pull.csv. RAW; dose derivation on the E21
card; relaunch under the derived doses."""
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
OUT = f"{ROOT}/results/diag/e21_pull.csv"
GRID = [("init", None, "byol"), ("init", None, "lejepa"),
        ("v2_best", "outputs/in100.vicreg.s0.floorssl_hz_v2_best.pt", "byol"),
        ("v2_best", "outputs/in100.vicreg.s0.floorssl_hz_v2_best.pt", "lejepa"),
        ("laug_best", "outputs/in100.floorssl.s0.lejepa_augs_best.pt", "lejepa")]
def main():
    dev = "cuda"
    base = torch.load(f"{ROOT}/outputs/in100.floorssl.s0.lejepa_augs_best.pt",
                      map_location="cpu", weights_only=False)
    mcfg_ref = dict(base["cfg"]["method"])          # the launched arm's own method cfg
    fr = base["cfg"]["frame"]
    rows = []
    for state, ck, aug in GRID:
        mcfg = {**mcfg_ref, "head_norm": "bn"}
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
        ep = 0
        if ck:
            pay = torch.load(f"{ROOT}/{ck}", map_location="cpu", weights_only=False)
            for role, sd in pay["modules"].items():
                if role != "probe":
                    modules[role].load_state_dict(sd)   # v2 = vicreg-class: same role names,
            method.load_extras(pay.get("extras", {}))   # byte-proven migration (E21 card)
            ep = pay["epoch"] + 1
        loader = DataLoader(method.build_train_dataset(), batch_size=128, shuffle=True,
                            drop_last=True, num_workers=0,
                            generator=torch.Generator().manual_seed(0))
        views, y = next(iter(loader))
        params = [p for p in modules["backbone"].parameters() if p.requires_grad]
        with autocast(dev, dtype=torch.bfloat16):
            terms, _, _ = method.training_step(modules, views.to(dev), dev, y=y.to(dev))
        names = [k for k in terms if k != "loss"]
        for i, k in enumerate(names):
            g = torch.autograd.grad(terms[k], params, retain_graph=i < len(names) - 1,
                                    allow_unused=True)
            g = torch.cat([x.reshape(-1).float() for x in g if x is not None]).norm().item()
            rows.append({"state": state, "ep": ep, "aug": aug, "term": k,
                         "g_enc": round(g, 6), "loss_value": round(float(terms[k]), 6),
                         "slurm_job": os.environ.get("SLURM_JOB_ID", "")})
            print(f"{state:10s} ep{ep:<3d} {aug:7s} {k:12s} g_enc={g:10.4f} "
                  f"loss={float(terms[k]):.4f}", flush=True)
    new = not os.path.exists(OUT)
    with open(OUT, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        if new:
            w.writeheader()
        w.writerows(rows)
    print("appended", OUT, flush=True)


if __name__ == "__main__":
    main()
