"""Linear accessibility of the projector output from the backbone (D-015's head-linearity
index, built at last; Berker's request 2026-08-07).

How much of z is reachable from h by a LINEAR map, and where the head's nonlinearity lives.
Two variants over the same V=8 augmentation-cloud stores, which is the point of the pair:

  centers  per-image cloud CENTRES: mean_v h[i,v] -> mean_v z[i,v]. Asks whether the
           semantic content the two spaces carry is linearly related, with augmentation
           variation averaged out.
  views    corresponding views DIRECTLY: h[i,v] -> z[i,v] over all (image, view) rows.
           Asks the same of the full map, augmentation directions included.

Reading the contrast: centers >> views means the head is linear on content and nonlinear on
augmentation directions (it is doing its work on the aug axes); centers ~ views means the
nonlinearity is not aug-specific.

  eval     third variant for continuity with the retired e2x_zpred numbers: the deterministic
           eval-transform features (one view per image), train-manifest fit -> val-manifest score.

SPLIT DISCIPLINE: `centers`/`views` split 80/20 **by IMAGE**, never by row. A row-wise split
would put other views of the same image on both sides and make `views` trivially easy.

CAVEAT THAT RIDES WITH EVERY R² HERE (D-060, Berker 2026-07-21): a contractive many-to-little
head reads HIGH R² while doing heavy nonlinear work — R² measures linear REACHABILITY of the
output, not head magnitude. That is why every row also carries the fitted map's spectrum
(sigma_min, cond, effrank_map) and the residual spectrum, which read contraction directly.
D-060 cancelled R² as a STANDALONE metric; it is reported here only alongside those.

Scope: runs with h = student.h.cls (384, ViT-S) and z = student.z.proj.out (256).
NOTE: for `floorssl` that z IS the declared loss-terminal space; for `byol` the declared z is
pred.out and proj.out is an intermediate tap (PROTOCOL §3) — the `declared_z` column says which.

  sbatch slurm/head_linearity.sbatch                    # all discovered runs
  sbatch slurm/head_linearity.sbatch --runs toy.floorssl.s0.e23cK3.extL
Writes results/diag/head_linearity.csv. Numbers land RAW.
"""
import csv
import glob
import os
import sys

import numpy as np

from sslgap.metrics.cross import linear_map_fit
from sslgap.paths import DIAG, FEATURES

H, Z = "student.h.cls", "student.z.proj.out"
ORBIT = {"toy": "imagenette.train.v1@audit_v1.o8", "in100": "in100.pairs100.v1@audit_v1.o8"}
EVAL = {"toy": ("imagenette.train", "imagenette.val"),
        "in100": ("in100.train500", "in100.val")}
DECLARED_Z = {"lambdajepa": True, "byol": False, "lejepa": True, "vicreg": True, "simclr": True}
SPLIT = 0.8


def _load(run, man, space):
    for flavor in ("v1L", "v1", ""):
        p = FEATURES / run / (f"{man}.{flavor}" if flavor else man) / f"{space}.npy"
        if p.exists():
            return np.load(p).astype(np.float64)
    return None


def _orbit(run, man, space):
    vs = sorted(glob.glob(str(FEATURES / run / man / f"{space}.view*.npy")),
                key=lambda p: int(p.split("view")[-1].split(".")[0]))
    return np.stack([np.load(v).astype(np.float64) for v in vs]) if vs else None


def rows_for(run, frame):
    out = []
    hv, zv = _orbit(run, ORBIT[frame], H), _orbit(run, ORBIT[frame], Z)
    if hv is not None and zv is not None and hv.shape[-1] == 384 and zv.shape[-1] == 256:
        V, N = hv.shape[0], hv.shape[1]
        k = int(N * SPLIT)
        idx = np.random.default_rng(0).permutation(N)      # image-level split, seeded
        tr, va = idx[:k], idx[k:]
        out.append(("centers", linear_map_fit(hv[:, tr].mean(0), zv[:, tr].mean(0),
                                              hv[:, va].mean(0), zv[:, va].mean(0)), N, V))
        out.append(("views", linear_map_fit(
            hv[:, tr].reshape(-1, 384), zv[:, tr].reshape(-1, 256),
            hv[:, va].reshape(-1, 384), zv[:, va].reshape(-1, 256)), N, V))
    mtr, mva = EVAL[frame]
    Ht, Zt, Hv, Zv = (_load(run, mtr, H), _load(run, mtr, Z),
                      _load(run, mva, H), _load(run, mva, Z))
    if all(x is not None for x in (Ht, Zt, Hv, Zv)) and Ht.shape[-1] == 384 and Zt.shape[-1] == 256:
        out.append(("eval", linear_map_fit(Ht, Zt, Hv, Zv), len(Ht), 1))
    return out


def main():
    args = sys.argv[1:]
    only = None
    if "--runs" in args:
        only = set(args[args.index("--runs") + 1].split(","))
    runs = sorted(r for r in os.listdir(FEATURES)
                  if (r.startswith("toy.") or r.startswith("in100.")) and (not only or r in only))
    rows = []
    for run in runs:
        frame = "toy" if run.startswith("toy.") else "in100"
        method = run.split(".")[1]
        for variant, res, n, V in rows_for(run, frame):
            rows.append({"run": run, "frame": frame, "method": method, "variant": variant,
                         "declared_z": int(DECLARED_Z.get(method, False)),
                         "n_images": n, "V": V, **res})
            print(f"{run:44s} {variant:8s} r2={res['r2_total']:.4f} "
                  f"meandim={res['r2_meandim']:.4f} sig_min={res['sigma_min']:.4g} "
                  f"cond={res['cond']:.3g} effrank_map={res['effrank_map']:.1f} "
                  f"resid_effrank={res['resid_effrank']:.1f}", flush=True)
    assert rows, "no runs with h=384 (student.h.cls) and z=256 (student.z.proj.out) found"
    out = DIAG / "head_linearity.csv"
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"\n[write] {out} ({len(rows)} rows over {len(set(r['run'] for r in rows))} runs)")


if __name__ == "__main__":
    main()
