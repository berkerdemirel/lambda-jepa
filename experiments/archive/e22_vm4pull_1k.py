"""E22 vm4 dataset-axis dose bridge (D-065): the FULL vm4 config (view-mean both taps,
d_slice 128 both, queue_steps=3) at the vm2 ep25 held state, queue warmed on 3 seeded
batches per dataset, measured on the 4th; rule as E22 — per-term w(in1k) = w_vm4-in100 *
g_in100/g_in1k, band [0.90,1.10]; vm4-in100 doses = 32.8/157.8/1.894. Original header:

E22 vm3 dataset-axis dose bridge (D-061): the symmetric-payment config
(h_floor_batch=view_mean, h_d_slice=32) held at vm2 ep25, IN-100 vs IN-1k batches — the
vm-h term's dataset ratio is unmeasured (E22 measured the POOLED h term). Rule as E22:
per-term w(in1k) = w_vm3_in100 * g_in100/g_in1k applied only if a mean ratio leaves
[0.90, 1.10]; vm3-in100 doses = 32.8/38.7/0.339. Original E22 header follows.

E22 dataset-axis dose bridge (D-055). The d256vm2 doses (32.8/38.7/.617) were certified on
IN-100 under lejepa V=4; E19-T1 says verbatim cross-frame transplants are the certified error
class, and D-047 showed the aug axis alone silently shifted g_inv by -24%. Before the IN-1k
launch, measure whether the DATASET axis shifts the per-term realized pulls: hold the vm2 ep25
formation state (the certified state choice) and swap ONLY the data source — same method cfg,
same lejepa V=4 pipeline, same seeded loader construction, torch.manual_seed(4242) before the
step so the floor's slice frame Q and drop_path draws are common random numbers across datasets.
Two batches per dataset for a noise floor. g_enc = trunk-module-only (e12h_pull convention).
PRE-DECLARED rule (card, before numbers): per-term w' = w * g_in100/g_in1k applied only if any
term's mean ratio leaves [0.90, 1.10]; otherwise launch doses verbatim (comparability).
Rows -> results/diag/e22_pull.csv. RAW."""
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
OUT = f"{ROOT}/results/diag/e22_vm4pull.csv"
CKPT = f"{ROOT}/outputs/in100.floorssl.s0.d256vm2_ep25.pt"
DATASETS = [("in100", "imagenet100", "~/data/imagenet100"),
            ("in1k", "imagenet1k", "~/data/imagenet")]


def main():
    dev = "cuda"
    base = torch.load(CKPT, map_location="cpu", weights_only=False)
    mcfg = OmegaConf.create({**dict(base["cfg"]["method"]), "h_floor_batch": "view_mean",
                             "h_d_slice": 128, "z_d_slice": 128, "queue_steps": 3})
    fr = base["cfg"]["frame"]
    rows = []
    for tag, ds, root in DATASETS:
        frame = Frame(name=tag, model_name=fr["model_name"], img_size=fr["img_size"],
                      dataset=ds, data_root=root, epochs=fr["epochs"], seed=0,
                      grad_clip=1.0, num_workers=0, device=dev)
        seed_everything(0)
        method = METHODS["floorssl"](mcfg, frame)
        modules = method.build_modules().to(dev)
        for role, sd in base["modules"].items():
            if role != "probe":
                modules[role].load_state_dict(sd)
        method.load_extras(base.get("extras", {}))
        loader = DataLoader(method.build_train_dataset(), batch_size=128, shuffle=True,
                            drop_last=True, num_workers=0,
                            generator=torch.Generator().manual_seed(0))
        it = iter(loader)
        for _ in range(3):                    # warm the queue on 3 seeded batches
            w, _ = next(it)
            torch.manual_seed(4242)
            with autocast(dev, dtype=torch.bfloat16):
                method.training_step(modules, w.to(dev), dev, y=None)
        for b in range(2):
            views, y = next(it)
            params = [p for p in modules["backbone"].parameters() if p.requires_grad]
            torch.manual_seed(4242)      # common slice frame + drop_path across datasets
            with autocast(dev, dtype=torch.bfloat16):
                terms, _, _ = method.training_step(modules, views.to(dev), dev, y=None)
            names = [k for k in terms if k != "loss"]
            for i, k in enumerate(names):
                g = torch.autograd.grad(terms[k], params, retain_graph=i < len(names) - 1,
                                        allow_unused=True)
                g = torch.cat([x.reshape(-1).float() for x in g if x is not None]).norm().item()
                rows.append({"dataset": tag, "state": "vm2_ep25", "batch": b, "term": k,
                             "g_enc": round(g, 6), "loss_value": round(float(terms[k]), 6),
                             "slurm_job": os.environ.get("SLURM_JOB_ID", "")})
                print(f"{tag:6s} b{b} {k:12s} g_enc={g:10.4f} loss={float(terms[k]):.4f}",
                      flush=True)
    with open(OUT, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print("wrote", OUT, flush=True)


if __name__ == "__main__":
    main()
