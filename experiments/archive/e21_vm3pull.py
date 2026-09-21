"""D-058 vm3 gate: (i) default-path byte regression for the new h_floor_batch/h_d_slice keys
(pinned-RNG step with keys ABSENT vs explicitly pooled/128 must produce identical terms —
the additive-knob discipline), then (ii) the h-payment dose bridge at the vm2 ep25 held
state (the certified formation-state convention): per-term trunk pulls with the h-floor on
pooled CLS (d'=128) vs per-image view-mean CLS (d'=32, first-32 sub-frame — common frame
RNG via the d_draw=128 parity), same batch, same seeds. Equal-pull prescription (vmpull
precedent, applied outright): h_lamb' = 0.617 * g_h(pooled)/g_h(view_mean); w_inv/w_floor
unchanged (their terms are byte-identical across the h axis).
Rows -> results/diag/e21_vm3pull.csv. RAW."""
import csv
import os
import sys

import torch
from omegaconf import OmegaConf
from torch.amp import autocast
from torch.utils.data import DataLoader

sys.path.insert(0, "/nfs/scistore19/locatgrp/bdemirel/ssl_project")
from sslgap.data import seed_everything
from sslgap.methods import METHODS
from sslgap.methods.base import Frame

ROOT = "/nfs/scistore19/locatgrp/bdemirel/ssl_project"
OUT = f"{ROOT}/results/diag/e21_vm3pull.csv"
CKPT = f"{ROOT}/outputs/in100.floorssl.s0.d256vm2_ep25.pt"
FORMS = [("absent", {}),
         ("pooled128", {"h_floor_batch": "pooled", "h_d_slice": 128}),
         ("view_mean32", {"h_floor_batch": "view_mean", "h_d_slice": 32})]


def step(method, modules, views, dev):
    params = [p for p in modules["backbone"].parameters() if p.requires_grad]
    torch.manual_seed(4242)
    with autocast(dev, dtype=torch.bfloat16):
        terms, _, _ = method.training_step(modules, views.to(dev), dev, y=None)
    out = {}
    names = [k for k in terms if k != "loss"]
    for i, k in enumerate(names):
        g = torch.autograd.grad(terms[k], params, retain_graph=i < len(names) - 1,
                                allow_unused=True)
        g = torch.cat([x.reshape(-1).float() for x in g if x is not None]).norm().item()
        out[k] = (float(terms[k]), g)
    return out


def main():
    dev = "cuda"
    base = torch.load(CKPT, map_location="cpu", weights_only=False)
    fr = base["cfg"]["frame"]
    rows, ref = [], None
    for tag, ovr in FORMS:
        cfg = OmegaConf.create({**dict(base["cfg"]["method"]), **ovr})
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
        views, _ = next(iter(loader))
        res = step(method, modules, views, dev)
        for k, (v, g) in res.items():
            rows.append({"form": tag, "term": k, "loss_value": round(v, 6),
                         "g_enc": round(g, 6),
                         "slurm_job": os.environ.get("SLURM_JOB_ID", "")})
            print(f"{tag:12s} {k:12s} loss={v:.6f} g_enc={g:.4f}", flush=True)
        if tag == "absent":
            ref = res
        if tag == "pooled128":
            same = all(abs(res[k][0] - ref[k][0]) < 1e-9 for k in res)
            print(f"DEFAULT-PATH REGRESSION: {'PASS' if same else 'FAIL'}", flush=True)
    gp = [r["g_enc"] for r in rows if r["form"] == "pooled128" and r["term"] == "h_moment_kl"][0]
    gv = [r["g_enc"] for r in rows if r["form"] == "view_mean32" and r["term"] == "h_moment_kl"][0]
    print(f"h_lamb' = 0.617 * {gp:.4f}/{gv:.4f} = {0.617 * gp / gv:.4f}", flush=True)
    with open(OUT, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print("wrote", OUT, flush=True)


if __name__ == "__main__":
    main()
