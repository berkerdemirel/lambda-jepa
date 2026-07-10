"""Toy vs IN-100 method-ordering agreement (bookkeeping, Berker 2026-07-10): for every battery
metric / pair-margin / headline probe, at h and at z.final, rank the methods at each rung and
report Spearman/Kendall agreement. Small-n (6-7 methods) — recorded as bookkeeping, not evidence.

  python experiments/toy_in100_ordering.py   -> results/diag/toy_in100_ordering.csv
"""
import os

import pandas as pd
from scipy.stats import kendalltau, spearmanr

from experiments.report_m1 import H_SPACE, Z_FINAL

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(ROOT, "results")
ORDER = ["simclr", "byol", "vicreg", "dino", "mae", "ijepa", "lejepa"]
TOY_RUN = {"simclr": "toy.simclr.s0.ext", "byol": "toy.byol.s0.ext", "vicreg": "toy.vicreg.s0.ext",
           "dino": "toy.dino.s0.probefix.ext", "mae": "toy.mae.s0.ext",
           "ijepa": "toy.ijepa.s0.ext", "lejepa": "toy.lejepa.s0.ext"}
IN_RUN = {m: f"in100.{m}.s0.ext" for m in ORDER}
BATTERY_METRICS = ["uniformity", "variance_floor.min_over_mean_std",
                   "offdiag_redundancy.mean_abs_corr", "rankme", "kurt_topeig.worst",
                   "epps_pulley"]
PROBES = ["linear_raw_v2", "knn_v1_k200"]
PAIR_COLS = ["cos_margin", "align_rel"]


def battery_vals(runs, metric, at):
    out = {}
    for m, rid in runs.items():
        sp = H_SPACE[m] if at == "h" else Z_FINAL[m]
        if sp is None:
            continue
        df = pd.read_csv(f"{RES}/battery/{rid}.csv")
        r = df[(df.space == sp) & (df.metric == metric) & (df.variant == "raw|full")]
        if len(r):
            v = float(r.value.iloc[0])
            if metric == "rankme":
                v /= float(r.d.iloc[0])
            out[m] = v
    return out


def probe_vals(runs, probe, at):
    out = {}
    for m, rid in runs.items():
        sp = H_SPACE[m] if at == "h" else Z_FINAL[m]
        if sp is None:
            continue
        df = pd.read_csv(f"{RES}/probes/{rid}.csv")
        r = df[(df.space == sp) & (df.probe == probe)]
        if len(r):
            out[m] = float(r.val_acc.iloc[0])
    return out


def pair_vals(runs, csv, col, at):
    df = pd.read_csv(csv)
    out = {}
    for m, rid in runs.items():
        sp = H_SPACE[m] if at == "h" else Z_FINAL[m]
        if sp is None:
            continue
        r = df[(df.run_id == rid) & (df.space == sp) & (df["stack"] == "audit_v1")]
        if len(r):
            out[m] = float(r[col].iloc[0])
    return out


def main():
    rows = []
    quantities = ([("battery", met) for met in BATTERY_METRICS]
                  + [("probe", p) for p in PROBES] + [("pairs", c) for c in PAIR_COLS])
    for kind, q in quantities:
        for at in ("h", "z"):
            if kind == "battery":
                a, b = battery_vals(TOY_RUN, q, at), battery_vals(IN_RUN, q, at)
            elif kind == "probe":
                a, b = probe_vals(TOY_RUN, q, at), probe_vals(IN_RUN, q, at)
            else:
                a = pair_vals(TOY_RUN, f"{RES}/M1/pair_margin.csv", q, at)
                b = pair_vals(IN_RUN, f"{RES}/M2/pair_margin.csv", q, at)
            common = sorted(set(a) & set(b))
            if len(common) < 4:
                continue
            x, y = [a[m] for m in common], [b[m] for m in common]
            rho, _ = spearmanr(x, y)
            tau, _ = kendalltau(x, y)
            rows.append({"quantity": (q if kind != "probe" else q), "kind": kind, "at": at,
                         "n_methods": len(common), "spearman": round(float(rho), 3),
                         "kendall": round(float(tau), 3)})
    df = pd.DataFrame(rows)
    out = os.path.join(RES, "diag", "toy_in100_ordering.csv")
    df.to_csv(out, index=False)
    print(df.to_string(index=False))
    print(f"[ordering] wrote {out}")


if __name__ == "__main__":
    main()
