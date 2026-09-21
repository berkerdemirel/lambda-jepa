"""E20 dino-fix companion: the OPPOSITION read — per control (ep25, formation window), the
cosine between the floor's trunk-gradient and the lane's own weighted trunk-gradient at the
declared tap. The share rule doses by lane-pull MAGNITUDE; dino's overshoot (E20-T1) suggests
direction matters: a lane whose objective already does floor-work (dino's teacher centering)
is an ALLY, not an adversary, and the same share over-doses it. Prediction on card: dino
cos ≥ ~0 while the view adversaries (vicreg/simclr/byol) are negative. Batches/machinery =
the e12h_pull convention (seed-0 first batch, training autocast, ckpt cfg + h_reg=sacreg).
Appends results/diag/e20_opposition.csv. RAW; no takeaway."""
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
OUT = f"{ROOT}/results/diag/e20_opposition.csv"
RUNS = ["simclr", "byol", "vicreg", "dino", "lejepa", "mae", "ijepa"]


def flat(grads):
    return torch.cat([g.reshape(-1).float() for g in grads if g is not None])


def to_dev(x, d):
    # dino's multi-crop loader yields a list of tensors (e12h_pull convention)
    return tuple(to_dev(t, d) for t in x) if isinstance(x, (tuple, list)) \
        else x.to(d, non_blocking=True)


def main():
    dev = "cuda"
    rows = []
    for m in RUNS:
        pay = torch.load(f"{ROOT}/outputs/in100.{m}.s0_ep25.pt", map_location="cpu",
                         weights_only=False)
        cfg = OmegaConf.create({**pay["cfg"]["method"], "h_reg": "sacreg", "h_lamb": 0.02})
        fr = pay["cfg"]["frame"]
        frame = Frame(name=fr["name"], model_name=fr["model_name"], img_size=fr["img_size"],
                      dataset=fr["dataset"], data_root=fr["data_root"], epochs=fr["epochs"],
                      seed=0, grad_clip=1.0, num_workers=0, device=dev)
        seed_everything(0)
        method = METHODS[m](cfg, frame)
        modules = method.build_modules().to(dev)
        for role, sd in pay["modules"].items():
            if role != "probe":
                modules[role].load_state_dict(sd)
        method.load_extras(pay.get("extras", {}))
        loader = DataLoader(method.build_train_dataset(), batch_size=pay["cfg"]["bs"],
                            shuffle=True, drop_last=True, num_workers=0,
                            generator=torch.Generator().manual_seed(0))
        views, y = next(iter(loader))
        enc_key = "encoder" if "encoder" in modules else "backbone"
        params = [p for p in modules[enc_key].parameters() if p.requires_grad]
        with autocast(dev, dtype=torch.bfloat16):
            terms, _, _ = method.training_step(modules, to_dev(views, dev), dev,
                                               y=y.to(dev))
        lane = terms["loss"] - 0.02 * terms["h_moment_kl"]      # shipped weighted lane loss
        g_lane = flat(torch.autograd.grad(lane, params, retain_graph=True, allow_unused=True))
        g_h = flat(torch.autograd.grad(terms["h_moment_kl"], params, allow_unused=True))
        cos = float((g_lane @ g_h) / (g_lane.norm() * g_h.norm()))
        rows.append({"method": m, "cos_lane_floor": round(cos, 5),
                     "g_lane_norm": round(float(g_lane.norm()), 5),
                     "g_floor_norm": round(float(g_h.norm()), 5),
                     "slurm_job": os.environ.get("SLURM_JOB_ID", "")})
        print(f"{m:8s} cos(lane, floor) = {cos:+.4f}   |g_lane| {g_lane.norm():.4f}  "
              f"|g_floor| {g_h.norm():.4f}", flush=True)
    new = not os.path.exists(OUT)
    with open(OUT, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        if new:
            w.writeheader()
        w.writerows(rows)
    print("appended", OUT, flush=True)


if __name__ == "__main__":
    main()
