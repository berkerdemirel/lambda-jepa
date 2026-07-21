"""Positive- vs random-pair cosine pass over stored audit_v1 pairs (Berker 2026-07-20: the
figure panels must show pos-cos and rand-cos SEPARATELY — alignment ≡ 2−2·cos_invariance on
normalized features, so the two battery pair metrics are one number rendered twice). Random
pairs = cross-view different-image pairs under the SAME stack (metrics/pairs.pair_margin —
the M1 instrument), so only image identity differs. Covers the E20 zoo (arm+ctrl), the dino
dose cells, the D-050 dim bracket + collapsed laug2, and f2. Overwrites
results/diag/e2x_posneg.csv (idempotent re-runs as stores land). RAW."""
import csv
import glob
import os

import numpy as np

import sys
sys.path.insert(0, "/nfs/scistore19/locatgrp/bdemirel/ssl_project")
from sslgap.metrics.pairs import pair_margin

ROOT = "/nfs/scistore19/locatgrp/bdemirel/ssl_project"
FEAT = f"{ROOT}/features"
OUT = f"{ROOT}/results/diag/e2x_posneg.csv"
ZOO = ["simclr", "byol", "vicreg", "dino", "lejepa", "mae", "ijepa"]
RUNS = [f"in100.{m}.s0.ext" for m in ZOO] + [f"in100.{m}.s0.e20f.ext" for m in ZOO] + \
       ["in100.dino.s0.e12gd.ext", "in100.dino.s0.e12gd2.ext",
        "in100.dino.s0.e20f_lam06.ext", "in100.dino.s0.e20f_lam008.ext"] + \
       [f"in100.floorssl.s0.d{d}.ext" for d in (16, 32, 64, 128, 256, 512)] + \
       ["in100.floorssl.s0.lejepa_augs2.ext", "in100.lejepa.s0.e12f2.ext",
        "in100.floorssl.s0.d256vm2.ext", "in100.floorssl.s0.d256e200.ext"]

rows = []
print(f"{'run':40s} {'space':22s} {'pos_cos':>8s} {'rand_cos':>9s} {'margin':>7s}")
for run in RUNS:
    d = f"{FEAT}/{run}/in100.pairs100.v1@audit_v1"
    stores = sorted(glob.glob(f"{d}/*.viewA.npy"))
    if not stores:
        print(f"{run:40s}   (pairs store absent — skipped)")
        continue
    for pa in stores:
        space = os.path.basename(pa)[: -len(".viewA.npy")]
        A = np.load(pa).astype(np.float64)
        B = np.load(pa.replace(".viewA.npy", ".viewB.npy")).astype(np.float64)
        m = pair_margin(A, B)
        rows.append({"run": run, "space": space, "n": len(A), "d": A.shape[1],
                     **{k: round(v, 4) for k, v in m.items()}})
        print(f"{run:40s} {space:22s} {m['pos_cos']:8.4f} {m['rand_cos']:9.4f} "
              f"{m['cos_margin']:7.4f}", flush=True)
with open(OUT, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0]))
    w.writeheader()
    w.writerows(rows)
print("wrote", OUT, f"({len(rows)} rows)")
