"""SIGReg honesty-cell decomposition (Berker 2026-07-08: 'i was expecting perfect sampling noise
vs no-training baselines'). For each LeJEPA checkpoint x space, recompute the EP meter fresh on
(i) the stored features, (ii) a moment-matched Gaussian N(mean, FULL covariance) — same N, same
estimator (nulls.gaussian_match), and (iii) an ISOTROPIC N(0, I) sample of the same (N, d) — the
pure sampling-noise floor. Nothing is done to the data; the references decompose the total
deviation from SIGReg's target (isotropic Gaussian) into a covariance part (fixable by a linear
map) and a shape part (no linear map can fix). kurt_topeig.worst alongside: affine-invariant, so
its two Gaussian references coincide up to noise.

  python experiments/sigreg_decomp.py
"""
import os

import numpy as np
import pandas as pd

from sslgap.metrics.isotropy import epps_pulley
from sslgap.metrics.nulls import gaussian_match
from sslgap.metrics.spectra import top_eigvec_excess_kurtosis

RES = "/nfs/scistore19/locatgrp/bdemirel/ssl_project/results"
FEAT = "/nfs/scistore19/locatgrp/bdemirel/ssl_project/features"
CELLS = [("toy.lejepa.s0.ext", "150ep grid (house recipe)"),
         ("toy.lejepa-lamb002.ext", "800ep official ckpt (M0)")]
SPACES = ["student.z.embed", "student.z.proj.out"]


def kurt_worst(X):
    return float(np.abs(top_eigvec_excess_kurtosis(X)).max())


def main():
    rng = np.random.default_rng(0)
    rows = []
    for rid, label in CELLS:
        for space in SPACES:
            X = np.load(os.path.join(FEAT, rid, "imagenette.train.v1", f"{space}.npy")
                        ).astype(np.float64)
            G = gaussian_match(X, seed=0)
            I = rng.standard_normal(X.shape)
            rows.append({"ckpt": label, "space": space, "n": X.shape[0], "d": X.shape[1],
                         "EP_data": epps_pulley(X), "EP_covmatched_gauss": epps_pulley(G),
                         "EP_isotropic_floor": epps_pulley(I),
                         "kurt_data": kurt_worst(X), "kurt_covmatched_gauss": kurt_worst(G),
                         "kurt_isotropic_floor": kurt_worst(I)})
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(RES, "M1", "sigreg_decomp.csv"), index=False)
    md = ["# SIGReg honesty cells — three-reference decomposition (numbers only)\n",
          "> Emitted by experiments/sigreg_decomp.py. EP meter: 256 seeded slices, per-dim",
          "> standardized (NOT whitened) — the SIGReg statistic as a meter. References:",
          "> `isotropic_floor` = N(0,I) sample, same (N,d) → pure sampling noise (the oracle if",
          "> SIGReg's target were exactly met); `covmatched_gauss` = N(mean, full covariance of",
          "> the data) → what a perfect GAUSSIAN with this run's covariance would read; the data",
          "> are never transformed. data − covmatched = shape excess (no linear map fixes it);",
          "> covmatched − floor = covariance/anisotropy part (a linear map could fix it).",
          "> Untrained-net reference (matrix null col): EP 569 (embed) / 405 (proj) for the grid",
          "> run. kurt_topeig.worst is affine-invariant → its two references coincide (~noise).\n",
          df.to_markdown(index=False, floatfmt=".3f"), "\n"]
    with open(os.path.join(RES, "M1", "SIGREG_DECOMP.md"), "w") as f:
        f.write("\n".join(md))
    print(f"[sigreg_decomp] wrote results/M1/sigreg_decomp.csv + SIGREG_DECOMP.md ({len(df)} rows)")


if __name__ == "__main__":
    main()
