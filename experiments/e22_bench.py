"""E22 input-pipeline benchmark (Berker 2026-07-20: "speed benchmarking playing with IO speed
(if we are compute bound its ok but if IO bound there's hope …) playing with nof workers,
persistent workers, pin memory, async loading or sharded dataset"). Live dmon on both running
jobs showed GPU duty cycle ~20-30% with half the workers in D-state -> input-bound; this job
quantifies the levers. Three parts, all on the IN-1k store with the vm2 method config:
(1) loader-only throughput grid (batches/s): workers x pin_memory x prefetch_factor,
    persistent_workers=True, shuffle=True (training access pattern), bs 128 V=4 (+ one bs 256
    row) — the number to beat is the GPU's consumption rate;
(2) per-stage decomposition on 300 random samples: raw NFS read | JPEG decode | 4x lejepa aug
    (separates file-IO from CPU decode: says whether sharding would even help);
(3) GPU step time fwd+bwd at bs 128 and bs 256 (records device + peak memory; the H100 rate
    is this scaled by the device factor; also answers whether bs 256 fits one card).
Rows -> results/diag/e22_bench.csv. RAW."""
import csv
import io
import os
import random
import sys
import time

import torch
from omegaconf import OmegaConf
from PIL import Image
from torch.amp import autocast
from torch.utils.data import DataLoader

sys.path.insert(0, "/nfs/scistore19/locatgrp/bdemirel/ssl_project")
from sslgap.data import seed_everything
from sslgap.methods import METHODS
from sslgap.methods.base import Frame

ROOT = "/nfs/scistore19/locatgrp/bdemirel/ssl_project"
OUT = f"{ROOT}/results/diag/e22_bench.csv"
CKPT = f"{ROOT}/outputs/in100.floorssl.s0.d256vm2_ep25.pt"
WARM, MEAS = 15, 100
rows = []


def emit(part, **kw):
    kw = {"part": part, **kw}
    rows.append(kw)
    print("  ".join(f"{k}={v}" for k, v in kw.items()), flush=True)


def main():
    dev = "cuda"
    base = torch.load(CKPT, map_location="cpu", weights_only=False)
    mcfg = OmegaConf.create(dict(base["cfg"]["method"]))
    fr = base["cfg"]["frame"]
    frame = Frame(name="in1k", model_name=fr["model_name"], img_size=fr["img_size"],
                  dataset="imagenet1k", data_root="~/data/imagenet", epochs=100, seed=0,
                  grad_clip=1.0, num_workers=0, device=dev)
    seed_everything(0)
    method = METHODS["floorssl"](mcfg, frame)
    t0 = time.time()
    ds = method.build_train_dataset()
    emit("scan", imagefolder_scan_s=round(time.time() - t0, 1), n=len(ds),
         cpus=os.environ.get("SLURM_CPUS_PER_TASK", "?"))

    # (2) stage decomposition first (cheap, independent of loader state)
    folder = ds.split_src._folder
    idx = random.Random(0).sample(range(len(folder)), 300)
    t0 = time.time(); nbytes = 0
    for i in idx:
        with open(folder.samples[i][0], "rb") as f:
            nbytes += len(f.read())
    t_read = time.time() - t0
    imgs = []
    t0 = time.time()
    for i in idx:
        with open(folder.samples[i][0], "rb") as f:
            b = f.read()
        imgs.append(Image.open(io.BytesIO(b)).convert("RGB"))
    t_dec = time.time() - t0 - t_read * 0  # read included; decode-only = this minus a re-read pass
    # re-read pass to subtract (files now in page cache -> lower bound on read; report both)
    t0 = time.time()
    for i in idx:
        with open(folder.samples[i][0], "rb") as f:
            f.read()
    t_read_cached = time.time() - t0
    tfm = ds.tfms[0]
    t0 = time.time()
    for im in imgs:
        for _ in range(4):
            tfm(im)
    t_aug = time.time() - t0
    emit("stages", read_ms_per_img_cold=round(1e3 * t_read / 300, 2),
         read_ms_per_img_cached=round(1e3 * t_read_cached / 300, 2),
         decode_plus_read_ms=round(1e3 * t_dec / 300, 2),
         aug4_ms_per_img=round(1e3 * t_aug / 300, 2),
         mean_file_kb=round(nbytes / 300 / 1024, 1))

    # (1) loader grid
    grid = [(8, True, 2), (8, False, 2), (12, True, 2), (16, True, 2), (16, False, 2),
            (16, True, 6), (24, True, 2), (24, True, 6)]
    for nw, pin, pf in grid:
        loader = DataLoader(ds, batch_size=128, shuffle=True, drop_last=True,
                            num_workers=nw, pin_memory=pin, prefetch_factor=pf,
                            persistent_workers=True,
                            generator=torch.Generator().manual_seed(0))
        it = iter(loader)
        for _ in range(WARM):
            next(it)
        t0 = time.time()
        for _ in range(MEAS):
            next(it)
        dt = time.time() - t0
        emit("loader", workers=nw, pin=pin, prefetch=pf, bs=128,
             batches_per_s=round(MEAS / dt, 3), imgs_per_s=round(128 * MEAS / dt, 1))
        del it, loader
    loader = DataLoader(ds, batch_size=256, shuffle=True, drop_last=True, num_workers=24,
                        pin_memory=True, prefetch_factor=4, persistent_workers=True,
                        generator=torch.Generator().manual_seed(0))
    it = iter(loader)
    for _ in range(8):
        next(it)
    t0 = time.time()
    for _ in range(50):
        next(it)
    dt = time.time() - t0
    emit("loader", workers=24, pin=True, prefetch=4, bs=256,
         batches_per_s=round(50 / dt, 3), imgs_per_s=round(256 * 50 / dt, 1))
    del it, loader

    # (3) GPU step time at bs 128 / 256 (fresh modules, vm2 cfg; batch cached on device)
    modules = method.build_modules().to(dev)
    params = [p for m in modules.values() for p in m.parameters() if p.requires_grad]
    for bs in (128, 256):
        views = torch.randn(bs, 4, 3, 224, 224, device=dev)
        torch.cuda.reset_peak_memory_stats()
        for i in range(35):
            if i == 5:
                torch.cuda.synchronize(); t0 = time.time()
            with autocast(dev, dtype=torch.bfloat16):
                terms, _, _ = method.training_step(modules, views, dev, y=None)
            g = torch.autograd.grad(terms["loss"], params, allow_unused=True)
            del g, terms
        torch.cuda.synchronize()
        step = (time.time() - t0) / 30
        emit("gpu_step", device=torch.cuda.get_device_name(0), bs=bs,
             step_s=round(step, 4), batches_per_s=round(1 / step, 3),
             imgs_per_s=round(bs / step, 1),
             peak_mem_gb=round(torch.cuda.max_memory_allocated() / 2**30, 1))

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
