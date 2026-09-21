"""D-051 gradient-interaction read (Berker 2026-07-20: "since they are not competing maybe
it is still fine? the effective diff is the learning rate like change right? how's the
interaction between floor pull vs inv pull?"). Trunk-space cosines between per-term
gradients at the pooled d256 cell's held states (init / ep25 / ep50), both z-floor forms on
the SAME weights, same batch, same slice frame (d_draw CRN). Reads: (a) cos(g_inv, g_z) per
form — the competition's gradient signature and whether view-mean relaxes it; (b)
cos(g_z_pooled, g_z_vm) — if ≈1 the form change is a per-term scalar ("lr-like" on that
term) and the ×1.162 magnitude is the whole story; (c) cos(g_h, g_z) rider. Controls: inv
and h are byte-identical terms across forms ⇒ cos(inv_p, inv_v) and cos(h_p, h_v) must be
1.000 exactly (instrument sanity). Also prints per-term norms + the w-weighted total norm
vs grad_clip=1.0 (the clip-active fact: realized steps are direction-only ⇒ term weights
set the MIX; e20-opposition caveat stands — single-batch cosines are a read, not a doser).
Rows -> results/diag/e21_vmcos.csv. RAW."""
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
OUT = f"{ROOT}/results/diag/e21_vmcos.csv"
STATES = [("init", None), ("ep25", "outputs/in100.floorssl.s0.d256_ep25.pt"),
          ("ep50", "outputs/in100.floorssl.s0.d256_ep50.pt")]
W = {"inv": 32.8, "moment_kl": 45.0, "h_moment_kl": 0.617}


def cos(a, b):
    a, b = a.double(), b.double()   # fp32 dot over 21M dims drifts ~.5% (controls read >1);
    return float((a @ b) / (a.norm() * b.norm() + 1e-12))   # f64 makes the control exact


def main():
    dev = "cuda"
    base = torch.load(f"{ROOT}/outputs/in100.floorssl.s0.d256_ep25.pt",
                      map_location="cpu", weights_only=False)
    mcfg_ref = dict(base["cfg"]["method"])
    fr = base["cfg"]["frame"]
    frame = Frame(name=fr["name"], model_name=fr["model_name"], img_size=fr["img_size"],
                  dataset=fr["dataset"], data_root=fr["data_root"], epochs=fr["epochs"],
                  seed=0, grad_clip=1.0, num_workers=0, device=dev)
    rows = []
    for state, ck in STATES:
        seed_everything(0)
        method_p = METHODS["lambdajepa"](OmegaConf.create(mcfg_ref), frame)
        modules = method_p.build_modules().to(dev)
        method_v = METHODS["lambdajepa"](OmegaConf.create(
            {**mcfg_ref, "z_floor_batch": "view_mean", "z_d_slice": 32}), frame)
        method_v.build_modules()   # constructs its floors; modules discarded — the step runs
        # on method_p's modules (same weights), and RNG is re-pinned before every step.
        if ck:
            pay = torch.load(f"{ROOT}/{ck}", map_location="cpu", weights_only=False)
            for role, sd in pay["modules"].items():
                if role != "probe":
                    modules[role].load_state_dict(sd)
            method_p.load_extras(pay.get("extras", {}))
        loader = DataLoader(method_p.build_train_dataset(), batch_size=128, shuffle=True,
                            drop_last=True, num_workers=0,
                            generator=torch.Generator().manual_seed(0))
        views, y = next(iter(loader))
        params = [p for p in modules["backbone"].parameters() if p.requires_grad]
        G = {}
        for form, method in (("pooled", method_p), ("vm", method_v)):
            torch.manual_seed(4242)
            with autocast(dev, dtype=torch.bfloat16):
                terms, _, _ = method.training_step(modules, views.to(dev), dev, y=y.to(dev))
            names = [k for k in terms if k != "loss"]
            for i, k in enumerate(names):
                g = torch.autograd.grad(terms[k], params, retain_graph=i < len(names) - 1,
                                        allow_unused=True)
                G[(form, k)] = torch.cat([x.reshape(-1).float() for x in g
                                          if x is not None]).cpu()
        for k in ("inv", "h_moment_kl"):   # byte-level control: identical terms across forms
            print(f"{state:6s} ctrl bitwise {k}: "
                  f"{torch.equal(G[('pooled', k)], G[('vm', k)])}", flush=True)
        pairs = [("ctrl cos(inv_p,inv_v)", ("pooled", "inv"), ("vm", "inv")),
                 ("ctrl cos(h_p,h_v)", ("pooled", "h_moment_kl"), ("vm", "h_moment_kl")),
                 ("cos(z_p,z_v)", ("pooled", "moment_kl"), ("vm", "moment_kl")),
                 ("cos(inv,z) pooled", ("pooled", "inv"), ("pooled", "moment_kl")),
                 ("cos(inv,z) vm", ("vm", "inv"), ("vm", "moment_kl")),
                 ("cos(inv,h)", ("pooled", "inv"), ("pooled", "h_moment_kl")),
                 ("cos(h,z) pooled", ("pooled", "h_moment_kl"), ("pooled", "moment_kl")),
                 ("cos(h,z) vm", ("pooled", "h_moment_kl"), ("vm", "moment_kl"))]
        for lab, a, b in pairs:
            c = cos(G[a], G[b])
            rows.append({"state": state, "read": lab, "cos": round(c, 4)})
            print(f"{state:6s} {lab:22s} {c:+.4f}", flush=True)
        for form in ("pooled", "vm"):
            tot = sum(W[k] * G[(form, k)] for k in ("inv", "moment_kl", "h_moment_kl"))
            rows.append({"state": state, "read": f"wtot_norm {form}",
                         "cos": round(float(tot.norm()), 3)})
            print(f"{state:6s} wtot_norm {form:7s}       {float(tot.norm()):8.2f}  "
                  f"(clip=1.0 {'ACTIVE' if tot.norm() > 1 else 'inactive'})", flush=True)
    with open(OUT, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print("wrote", OUT, flush=True)


if __name__ == "__main__":
    main()
