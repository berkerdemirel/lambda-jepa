"""Offline probes trained on augmented-view features (Berker 2026-07-13, G-wave session).

The online monitor fits its LayerNorm+Linear head on features of the training-augmentation
views; the battery's offline probes train on clean-frame features. Across the three calibration
pairs the monitor reads the floor arm ~+3 pts more favorably than the converged offline probes
(vicreg's sign flip is that bias crossing zero; results/diag/e12g_probe_table.md). This job puts
converged offline probes on the monitor's training distribution — same optimizer discipline,
same clean-val evaluation, only the training features change.

Train sets per (run, declared/monitor space):
  aug2x10k  — viewA+viewB of the method's own-stack pair store (10k pairs-manifest images x 2)
  clean10k  — 100/class seeded subsample of the clean train500 store (image-count-matched)
Probes: linear_raw_v2 + linear_house_v2 (house == the monitor's architecture). Eval: in100.val.v1.
Numbers land raw in results/diag/e12_aug_probes.csv (row appended after every fit).

  sbatch slurm/e12_aug_probes.sbatch
"""
import os

import numpy as np
import pandas as pd

from sslgap.probes import linear_house_v2, linear_raw_v2

FEAT = "/nfs/scistore19/locatgrp/bdemirel/ssl_project/features"
OUT = "/nfs/scistore19/locatgrp/bdemirel/ssl_project/results/diag/e12_aug_probes.csv"
PAIRS = "in100.pairs100.v1"
JOBS = [  # (run_id, space, own training stack); lejepa trains on the orbit == audit_v1
    ("in100.lejepa.s0.e12f2.ext", "student.z.embed", "audit_v1"),
    ("in100.lejepa.s0.e12c1.ext", "student.z.embed", "audit_v1"),
    ("in100.vicreg.s0.e12gv.ext", "student.h.gap", "own_vicreg"),
    ("in100.vicreg.s0.e12gvc.ext", "student.h.gap", "own_vicreg"),
    ("in100.dino.s0.e12gd.ext", "teacher.h.cls", "own_dino"),   # monitor taps teacher CLS
    ("in100.dino.s0.e12gd.ext", "student.h.cls", "own_dino"),
    ("in100.dino.s0.e12gdc.ext", "teacher.h.cls", "own_dino"),
    ("in100.dino.s0.e12gdc.ext", "student.h.cls", "own_dino"),
]
PROBES = {"linear_raw_v2": linear_raw_v2, "linear_house_v2": linear_house_v2}


def _load(run, key, space):
    return np.load(os.path.join(FEAT, run, key, space + ".npy")).astype(np.float32)


def _labels(run, key):
    return np.load(os.path.join(FEAT, run, key, "labels.npy"))


def _sub100(y, seed=0):
    rng = np.random.default_rng(seed)
    return np.concatenate([rng.choice(np.where(y == c)[0], 100, replace=False)
                           for c in np.unique(y)])


def main():
    rows = []
    for run, space, stack in JOBS:
        pdir = f"{PAIRS}@{stack}"
        Xa = np.concatenate([_load(run, pdir, f"{space}.view{v}") for v in "AB"])
        ya = np.tile(_labels(run, pdir), 2)
        y500 = _labels(run, "in100.train500.v1")
        idx = _sub100(y500)
        Xc, yc = _load(run, "in100.train500.v1", space)[idx], y500[idx]
        Xv, yv = _load(run, "in100.val.v1", space), _labels(run, "in100.val.v1")
        for tname, Xt, yt in (("aug2x10k", Xa, ya), ("clean10k", Xc, yc)):
            for pname, fn in PROBES.items():
                r = fn(Xt, yt, Xv, yv, num_classes=100)
                rows.append({"run_id": run, "space": space, "stack": stack,
                             "train_set": tname, "probe": pname, "n_train": len(yt),
                             "d": Xt.shape[1], "val_acc": r["val_acc"],
                             "best_ep": r["best_ep"], "epochs_run": r["epochs_run"]})
                print(f"[aug_probes] {run} {space} {tname} {pname}: {r['val_acc']:.4f} "
                      f"({r['best_ep']}/{r['epochs_run']})", flush=True)
                pd.DataFrame(rows).to_csv(OUT, index=False)
    print(f"[aug_probes] wrote {OUT} ({len(rows)} rows)")


if __name__ == "__main__":
    main()
