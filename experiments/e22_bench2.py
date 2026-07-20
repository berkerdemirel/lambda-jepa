"""E22 bench round 2 (Berker: lightly reports 35 min/ep on 2x4090 — "are we sure that there's
no more improvement to be done?"). Two decision-relevant measurements, loader-side only (no
GPU): (1) worker scaling 24/28/32 at the H100-node CPU share (does the chain's 24w/28c leave
throughput on the table); (2) the pre-resized-dataset lever — re-encode 300 samples at
shorter-side 256 / q87 in memory, then measure decode+aug on those vs the full-res originals:
projects the per-image CPU cost and file size if we built a resized copy (a DECLARED aug-
statistics deviation — RRC would crop from 256-res, losing high-frequency detail for small
crops; measurement first, decision Berker's). Rows -> results/diag/e22_bench2.csv. RAW."""
import csv
import io
import os
import random
import sys
import time

import torch
from omegaconf import OmegaConf
from PIL import Image
from torch.utils.data import DataLoader

sys.path.insert(0, "/nfs/scistore19/locatgrp/bdemirel/ssl_project")
from sslgap.data import seed_everything
from sslgap.methods import METHODS
from sslgap.methods.base import Frame

ROOT = "/nfs/scistore19/locatgrp/bdemirel/ssl_project"
OUT = f"{ROOT}/results/diag/e22_bench2.csv"
CKPT = f"{ROOT}/outputs/in100.floorssl.s0.d256vm2_ep25.pt"
rows = []


def emit(part, **kw):
    kw = {"part": part, **kw}
    rows.append(kw)
    print("  ".join(f"{k}={v}" for k, v in kw.items()), flush=True)


def main():
    base = torch.load(CKPT, map_location="cpu", weights_only=False)
    mcfg = OmegaConf.create(dict(base["cfg"]["method"]))
    fr = base["cfg"]["frame"]
    frame = Frame(name="in1k", model_name=fr["model_name"], img_size=fr["img_size"],
                  dataset="imagenet1k", data_root="~/data/imagenet", epochs=100, seed=0,
                  grad_clip=1.0, num_workers=0, device="cpu")
    seed_everything(0)
    method = METHODS["floorssl"](mcfg, frame)
    ds = method.build_train_dataset()
    emit("env", cpus=os.environ.get("SLURM_CPUS_PER_TASK", "?"),
         node=os.environ.get("SLURMD_NODENAME", "?"))

    # (2) pre-resized lever, in memory
    folder = ds.split_src._folder
    idx = random.Random(1).sample(range(len(folder)), 300)
    tfm = ds.tfms[0]
    origs, resized = [], []
    for i in idx:
        with open(folder.samples[i][0], "rb") as f:
            origs.append(f.read())
    t0 = time.time()
    full = [Image.open(io.BytesIO(b)).convert("RGB") for b in origs]
    t_dec_full = time.time() - t0
    nb = 0
    for im in full:
        s = 256 / min(im.size)
        r = im.resize((max(1, round(im.size[0] * s)), max(1, round(im.size[1] * s))),
                      Image.BILINEAR) if s < 1 else im
        buf = io.BytesIO()
        r.save(buf, "JPEG", quality=87)
        resized.append(buf.getvalue())
        nb += len(resized[-1])
    t0 = time.time()
    small = [Image.open(io.BytesIO(b)).convert("RGB") for b in resized]
    t_dec_small = time.time() - t0
    t0 = time.time()
    for im in full:
        for _ in range(4):
            tfm(im)
    t_aug_full = time.time() - t0
    t0 = time.time()
    for im in small:
        for _ in range(4):
            tfm(im)
    t_aug_small = time.time() - t0
    emit("preresize", decode_full_ms=round(1e3 * t_dec_full / 300, 2),
         decode_256_ms=round(1e3 * t_dec_small / 300, 2),
         aug4_full_ms=round(1e3 * t_aug_full / 300, 2),
         aug4_256_ms=round(1e3 * t_aug_small / 300, 2),
         mean_kb_full=round(sum(len(b) for b in origs) / 300 / 1024, 1),
         mean_kb_256=round(nb / 300 / 1024, 1))

    # (1) worker scaling at the H100-node CPU share
    for nw in (24, 28, 32):
        loader = DataLoader(ds, batch_size=128, shuffle=True, drop_last=True,
                            num_workers=nw, pin_memory=False, prefetch_factor=2,
                            persistent_workers=True,
                            generator=torch.Generator().manual_seed(0))
        it = iter(loader)
        for _ in range(15):
            next(it)
        t0 = time.time()
        for _ in range(100):
            next(it)
        dt = time.time() - t0
        emit("loader", workers=nw, bs=128, batches_per_s=round(100 / dt, 3),
             imgs_per_s=round(12800 / dt, 1))
        del it, loader

    with open(OUT, "w", newline="") as f:
        keys = []
        for r in rows:
            keys += [k for k in r if k not in keys]
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)
    print("wrote", OUT, flush=True)


if __name__ == "__main__":
    main()
