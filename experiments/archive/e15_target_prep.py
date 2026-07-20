"""E15 (D-032) target freeze: standardization stats, sigma_med, two-bandwidth RFF W/b, the
variance-floor level gamma, and the constant-h audit floor Var(z) — all measured ONCE on the
stored E14 ctx features of the frozen tokenizer (in100.mae.s0 h.gap on sharp 96-crops: exactly
the training ctx distribution, 20k samples) and written to outputs/e15_target_v1.npz. Seed 15.
CPU, seconds; run before any E15 training code."""
import hashlib
import os

import numpy as np
from scipy.spatial.distance import pdist

ROOT = "/nfs/scistore19/locatgrp/bdemirel/ssl_project"
SRC = f"{ROOT}/features/in100.mae.s0.ext/in100.pairs100.v1@foveal_v1_ctx"
OUT = f"{ROOT}/outputs/e15_target_v1.npz"
BLOCK, SEED = 192, 15

Y = np.concatenate([np.asarray(np.load(f"{SRC}/student.h.gap.view{v}.npy"), dtype=np.float64)
                    for v in "AB"])
mu, sd = Y.mean(0), Y.std(0).clip(1e-6)
Ys = (Y - mu) / sd
rng = np.random.default_rng(SEED)
sub = rng.choice(len(Ys), size=2048, replace=False)
sigma_med = float(np.median(pdist(Ys[sub])))
W = np.concatenate([rng.standard_normal((Y.shape[1], BLOCK)) / (m * sigma_med)
                    for m in (1.0, 2.0)], axis=1)
b = rng.uniform(0, 2 * np.pi, 2 * BLOCK)
Z = np.sqrt(2.0 / BLOCK) * np.cos(Ys @ W + b)
dim_std = Z.std(0)
gamma = float(0.5 * np.median(dim_std))
var_z = float((dim_std ** 2).mean())
np.savez(OUT, mu=mu.astype(np.float32), sd=sd.astype(np.float32), W=W.astype(np.float32),
         b=b.astype(np.float32), block=BLOCK, gamma=gamma, var_z=var_z, sigma_med=sigma_med,
         seed=SEED, src=SRC, n=len(Y))
print(f"sigma_med={sigma_med:.3f} gamma={gamma:.5f} var_z={var_z:.6f} "
      f"dim_std[min/med/max]={dim_std.min():.4f}/{np.median(dim_std):.4f}/{dim_std.max():.4f}")
print(f"wrote {OUT} sha256[:16]={hashlib.sha256(open(OUT,'rb').read()).hexdigest()[:16]}")
