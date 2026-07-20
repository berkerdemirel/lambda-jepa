"""E21 centered-kNN companion (the E20 reporting convention carried to D-050: the knn bars of
record are CENTERED — train-mean removed from train and val — so every knn column cited against
them must be the same instrument). knn200 raw vs centered for the six landed dim cells + the
collapsed 2048-d cell (lejepa_augs2 @ep4, contrast) at trunk h.cls, and the three comparators
(lejepa e20f · lejepa ctrl · f2) at BOTH h.cls and their declared z.embed. Skips absent
stores. Overwrites results/diag/e21_centered.csv (idempotent re-runs as landings arrive). RAW."""
import csv
import os

import numpy as np
import torch

import sys
sys.path.insert(0, "/nfs/scistore19/locatgrp/bdemirel/ssl_project")
torch.set_num_threads(8)
from sslgap.probes.knn import knn_topk_acc

ROOT = "/nfs/scistore19/locatgrp/bdemirel/ssl_project"
FEAT = f"{ROOT}/features"
OUT = f"{ROOT}/results/diag/e21_centered.csv"
H = "student.h.cls"
RUNS = [(f"in100.floorssl.s0.d{d}.ext", H) for d in (16, 32, 64, 128, 256, 512)] + \
       [("in100.floorssl.s0.lejepa_augs2.ext", H)] + \
       [(r, sp) for r in ("in100.lejepa.s0.e20f.ext", "in100.lejepa.s0.ext",
                          "in100.lejepa.s0.e12f2.ext")
        for sp in (H, "student.z.embed")] + \
       [("in100.dino.s0.e20f_lam008.ext", "teacher.h.cls"),
        ("in100.dino.s0.ext", "teacher.h.cls")]


def load(run, sp):
    for man in ("v1L", "v1"):
        d, dv = f"{FEAT}/{run}/in100.train500.{man}", f"{FEAT}/{run}/in100.val.{man}"
        if os.path.exists(f"{d}/{sp}.npy"):
            return (np.load(f"{d}/{sp}.npy").astype(np.float32), np.load(f"{d}/labels.npy"),
                    np.load(f"{dv}/{sp}.npy").astype(np.float32), np.load(f"{dv}/labels.npy"))
    return None


def knn200(tr, ytr, va, yva):
    return float(knn_topk_acc(torch.from_numpy(tr), torch.from_numpy(ytr).long(),
                              torch.from_numpy(va), torch.from_numpy(yva).long(),
                              num_classes=100, knn_k=200, knn_t=0.1))


rows = []
print(f"{'run':40s} {'space':16s} {'knn200':>7s} {'knn200_c':>9s} {'Δcenter':>8s}")
for run, sp in RUNS:
    got = load(run, sp)
    if got is None:
        print(f"{run:40s} {sp:16s}   (store absent — skipped)")
        continue
    tr, ytr, va, yva = got
    raw = knn200(tr, ytr, va, yva)
    mu = tr.mean(0, keepdims=True)
    cen = knn200(tr - mu, ytr, va - mu, yva)
    rows.append({"run": run, "space": sp, "knn200_raw": round(raw, 4),
                 "knn200_centered": round(cen, 4)})
    print(f"{run:40s} {sp:16s} {raw:7.4f} {cen:9.4f} {100 * (cen - raw):+8.1f}", flush=True)
with open(OUT, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0]))
    w.writeheader()
    w.writerows(rows)
print("wrote", OUT)
