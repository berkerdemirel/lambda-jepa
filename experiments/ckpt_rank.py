"""The estimator-wall readout at h: effective rank per checkpoint, against the conditioner's
own rows-per-slice-dimension ratio (Berker 2026-08-27).

Why: the achieved effective rank of the backbone representation orders almost perfectly by
n_eff/d', the number of rows the h conditioner sees per slice dimension — not by the value of
the h term. That is the D-051 estimator wall the ring (D-064) was built to remove:
`view_mean` feeds n = bs rows into a d'-dim slice, so n < d' makes the sample covariance
singular by construction and the OAS shrinkage, not the data, sets the logdet barrier over the
null directions. The running 400-epoch grid is split across both regimes (B1/B3 at ratio 0.5,
B2'/L2' on the ring at 4.0/16), so the question is measurable on checkpoints already on disk.

Reads the FROZEN epoch checkpoints only (_ep*.pt) — `_last.pt` is rewritten by the running
jobs and a concurrent read is a torn file, not a measurement.

Writes NO feature store (disk is the binding constraint): one forward pass over the canonical
in1k val manifest under the house eval transform, spectrum computed in memory, one CSV row.

  python experiments/ckpt_rank.py <ckpt> [<ckpt> ...]
CSV (appended per checkpoint, wall-safe): results/diag/ckpt_rank.csv
"""
import os
import re
import sys

import numpy as np
import pandas as pd
import torch
from torch.amp import autocast
from torch.utils.data import DataLoader

from sslgap.ckpt import adapters
from sslgap.data import EvalDataset, _Source
from sslgap.metrics.spectra import (covariance_eigs, effective_rank, participation_ratio,
                                    power_law_alpha, rankme)
from sslgap.models.backbones import trunk_features

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MANIFEST = f"{ROOT}/features/manifests/in1k.val.v1.csv"
VAL = os.path.expanduser("~/data/imagenet/val")
OUT = f"{ROOT}/results/diag/ckpt_rank.csv"
NULL_REL = 1e-6            # eigenvalues below this x lam_max are the LayerNorm null + noise


def h_cls(ckpt, device, bs=256, workers=12):
    tag = re.sub(r"_ep\d+\.pt$|_last\.pt$|_best\.pt$", "", os.path.basename(ckpt))
    loaded = adapters.load("native", ckpt, tag).eval_(device)
    trunk = loaded.branches[loaded.probed_branch].trunk
    img = loaded.frame["img_size"]
    dl = DataLoader(EvalDataset(MANIFEST, _Source("imagefolder", root=VAL), img),
                    batch_size=bs, shuffle=False, num_workers=workers, pin_memory=True,
                    persistent_workers=True)
    feats = []
    with torch.no_grad(), autocast("cuda", dtype=torch.bfloat16):
        for x, _ in dl:
            feats.append(trunk_features(trunk, x.to(device, non_blocking=True))["cls"].float().cpu())
    return torch.cat(feats).numpy().astype(np.float64), loaded


def conditioner_parts(eigs, mu2, D):
    """The conditioner's own KL/d split (sslgap/methods/_common.py):
    KL/d = 0.5*(a - log g - 1) + 0.5*||mu||^2/d = 0.5*(scale + shape) + mean, where
    scale = a - log a - 1 depends only on total variance and shape = log(a/g) is the
    scale-free log AM/GM gap — the anti-degeneracy barrier."""
    lam = eigs[eigs >= NULL_REL * eigs[0]]
    a, logg = lam.mean(), np.log(lam).mean()
    return dict(n_null=int(len(eigs) - len(lam)),
                scale=0.5 * (a - np.log(a) - 1), shape=0.5 * (np.log(a) - logg),
                mean_term=0.5 * mu2 / D, floor=float(lam[-1] / a))


def main():
    device = "cuda"
    torch.backends.cudnn.benchmark = True
    workers = int(os.environ.get("SLURM_CPUS_PER_TASK", 12))
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    for ckpt in sys.argv[1:]:
        X, loaded = h_cls(ckpt, device, workers=workers)
        n, D = X.shape
        mu2 = float((X.mean(0) ** 2).sum())
        eigs = covariance_eigs(X)
        m = loaded.cfg.get("method", {})
        bs = loaded.cfg.get("bs")
        d_h = m.get("h_d_slice") or 128
        qh = m.get("queue_steps") if m.get("h_queue_steps") is None else m.get("h_queue_steps")
        n_eff = bs * ((qh or 0) + 1)
        row = dict(ckpt=os.path.basename(ckpt),
                   run=re.sub(r"_(ep\d+|last|best)\.pt$", "", os.path.basename(ckpt)),
                   epoch=(loaded.provenance.get("epoch") or 0) + 1,
                   model=loaded.frame["model_name"], D=D, n_images=n, bs=bs,
                   h_d_slice=d_h, shrink=m.get("floor_shrink"), h_queue=qh,
                   n_eff=n_eff, ratio=n_eff / d_h,
                   effrank=effective_rank(eigs), effrank_over_D=effective_rank(eigs) / D,
                   part_ratio=participation_ratio(eigs), rankme=rankme(X),
                   alpha=power_law_alpha(eigs), **conditioner_parts(eigs, mu2, D))
        pd.DataFrame([row]).to_csv(OUT, mode="a", header=not os.path.exists(OUT), index=False)
        print(f"[rank] {row['run']:26s} ep{row['epoch']:<4d} ratio={row['ratio']:5.2f} "
              f"effrank={row['effrank']:7.1f} (/D {row['effrank_over_D']:.3f}) "
              f"shape={row['shape']:.3f} nnull={row['n_null']}", flush=True)


if __name__ == "__main__":
    main()
