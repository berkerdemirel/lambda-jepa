"""Metric-vs-performance ordering (bookkeeping, Berker 2026-07-10): per battery metric, Spearman
between the method ordering by that metric and by the headline probes (linear_raw_v2, knn), at
the same space, per rung. Signed rho as-is (metric direction carried by the sign). Small-n
(6-7 methods), exploratory per D-010 — E3 owns the evidential version.

  python experiments/metric_probe_ordering.py   -> results/diag/metric_vs_probe_ordering.csv
"""
import os

import pandas as pd
from scipy.stats import spearmanr

from experiments.report_m1 import H_SPACE, Z_FINAL

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(ROOT, "results")
ORDER = ["simclr", "byol", "vicreg", "dino", "mae", "ijepa", "lejepa"]
RUNS = {"toy": {"simclr": "toy.simclr.s0.ext", "byol": "toy.byol.s0.ext",
                "vicreg": "toy.vicreg.s0.ext", "dino": "toy.dino.s0.probefix.ext",
                "mae": "toy.mae.s0.ext", "ijepa": "toy.ijepa.s0.ext", "lejepa": "toy.lejepa.s0.ext"},
        "in100": {m: f"in100.{m}.s0.ext" for m in ORDER}}
METRICS = ["uniformity", "variance_floor.min_over_mean_std", "offdiag_redundancy.mean_abs_corr",
           "rankme", "kurt_topeig.worst", "epps_pulley", "kurt_slices_mean_abs"]


def main():
    rows = []
    for rung, runs in RUNS.items():
        bat = {m: pd.read_csv(f"{RES}/battery/{rid}.csv") for m, rid in runs.items()}
        prb = {m: pd.read_csv(f"{RES}/probes/{rid}.csv") for m, rid in runs.items()}
        for at in ("h", "z"):
            space = {m: (H_SPACE[m] if at == "h" else Z_FINAL[m]) for m in ORDER}
            probes = {}
            for pr in ("linear_raw_v2", "knn_v1_k200"):
                probes[pr] = {m: float(r.val_acc.iloc[0]) for m in ORDER if space[m]
                              if len(r := prb[m][(prb[m].space == space[m]) & (prb[m].probe == pr)])}
            for met in METRICS:
                vals = {}
                for m in ORDER:
                    if space[m] is None:
                        continue
                    r = bat[m][(bat[m].space == space[m]) & (bat[m].metric == met)
                               & (bat[m].variant == "raw|full")]
                    if len(r):
                        v = float(r.value.iloc[0])
                        if met == "rankme":
                            v /= float(r.d.iloc[0])
                        vals[m] = v
                row = {"rung": rung, "at": at,
                       "metric": met + ("(/dim)" if met == "rankme" else "")}
                for pr, pv in probes.items():
                    common = sorted(set(vals) & set(pv))
                    if len(common) >= 4:
                        rho, _ = spearmanr([vals[m] for m in common], [pv[m] for m in common])
                        row[f"rho_{pr.split('_')[0] if pr.startswith('lin') else 'knn'}"] = round(float(rho), 3)
                        row["n"] = len(common)
                if len(row) > 3:
                    rows.append(row)
    df = pd.DataFrame(rows)
    out = os.path.join(RES, "diag", "metric_vs_probe_ordering.csv")
    df.to_csv(out, index=False)
    print(df.to_string(index=False))
    print(f"[metric-probe] wrote {out}")


if __name__ == "__main__":
    main()
