"""E20 centered-kNN companion (Berker 2026-07-18: "also report centered knn"). For every
landed e20f cell and its control: knn200 at declared h, RAW vs CENTERED (train-mean removed
from train and val — the E17 mean-sensitive/centered split; centered kNN reads ordering with
the cone contribution removed). Auto-detects manifest flavor (v1L from E17-era extractions,
v1 from default extractions); skips runs whose stores are absent (rerun as cells land).
Appends results/diag/e20_centered.csv. RAW; no takeaway."""
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
OUT = f"{ROOT}/results/diag/e20_centered.csv"
CELLS = [  # (method, declared-h space, control run, e20f run)
    ("simclr", "student.h.cls", "in100.simclr.s0.ext", "in100.simclr.s0.e20f.ext"),
    ("byol", "student.h.cls", "in100.byol.s0.ext", "in100.byol.s0.e20f.ext"),
    ("vicreg", "student.h.cls", "in100.vicreg.s0.e17c.ext", "in100.vicreg.s0.e20f.ext"),
    ("dino", "teacher.h.cls", "in100.dino.s0.e17c.ext", "in100.dino.s0.e20f.ext"),
    ("lejepa", "student.z.embed", "in100.lejepa.s0.e17c.ext", "in100.lejepa.s0.e20f.ext"),
    ("mae", "student.h.gap", "in100.mae.s0.ext", "in100.mae.s0.e20f.ext"),
    ("ijepa", "teacher.h.gap", "in100.ijepa.s0.ext", "in100.ijepa.s0.e20f.ext"),
]


def load(run, sp):
    for man in ("v1L", "v1"):
        d = f"{FEAT}/{run}/in100.train500.{man}"
        dv = f"{FEAT}/{run}/in100.val.{man}"
        if os.path.exists(f"{d}/{sp}.npy"):
            return (np.load(f"{d}/{sp}.npy").astype(np.float32), np.load(f"{d}/labels.npy"),
                    np.load(f"{dv}/{sp}.npy").astype(np.float32), np.load(f"{dv}/labels.npy"))
    return None


def knn200(tr, ytr, va, yva):
    return float(knn_topk_acc(torch.from_numpy(tr), torch.from_numpy(ytr).long(),
                              torch.from_numpy(va), torch.from_numpy(yva).long(),
                              num_classes=100, knn_k=200, knn_t=0.1))


rows = []
print(f"{'run':34s} {'space':16s} {'knn200':>7s} {'knn200_c':>9s} {'Δcenter':>8s}")
for m, sp, ctrl, arm in CELLS:
    for run in (ctrl, arm):
        got = load(run, sp)
        if got is None:
            print(f"{run:34s} {sp:16s}   (store absent — skipped)")
            continue
        tr, ytr, va, yva = got
        raw = knn200(tr, ytr, va, yva)
        mu = tr.mean(0, keepdims=True)
        cen = knn200(tr - mu, ytr, va - mu, yva)
        rows.append({"method": m, "run": run, "space": sp,
                     "knn200_raw": round(raw, 4), "knn200_centered": round(cen, 4)})
        print(f"{run:34s} {sp:16s} {raw:7.4f} {cen:9.4f} {100 * (cen - raw):+8.1f}",
              flush=True)
new = not os.path.exists(OUT)
with open(OUT, "a", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0]))
    if new:
        w.writeheader()
    w.writerows(rows)
print("appended", OUT, flush=True)
