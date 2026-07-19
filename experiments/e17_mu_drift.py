"""T5 loose end (D-039/HANDOVER open item 1, LOW): attribute vicreg c015's UNATTRIBUTED ‖μ‖²
halving (48.7 → 20.8 by ep100 while E‖x‖² grew 68 → 383 — not a norm-budget squeeze; the var+cov
term is mean-blind, so its own gradient is not the direct cause). Cadence trajectory at declared
h (student.h.cls, train500.v1L): per checkpoint ep25/50/75/100 vs ctrl — ‖μ‖², tr Σ, E‖x‖²,
mu_share, and the μ DIRECTION geometry (cos to ep100's μ and to ctrl's μ): a mean that shrinks
in place (cos ≈ 1, norm ↓) reads as decay against an indirect pressure; a rotating mean reads as
re-equilibration of what the cone points at. Timing (early jump vs slow monotone drift) is the
second attribution axis — read jointly with the run's per-term wandb curves. RAW numbers →
results/diag/e17_mu_drift.csv; no takeaway here."""
import csv
import os

import numpy as np

ROOT = "/nfs/scistore19/locatgrp/bdemirel/ssl_project"
# cadence extracts ride the default lane (train500.v1); e17c/c015-ep100 were the E17
# declaration-agnostic extraction (v1L). Base spaces are identical; only L-tap ride-alongs differ.
RUNS = [("ctrl", "in100.vicreg.s0.e17c.ext", "v1L"),
        ("c015.ep25", "in100.vicreg.s0.hpull_varcov.c015.ep25.ext", "v1"),
        ("c015.ep50", "in100.vicreg.s0.hpull_varcov.c015.ep50.ext", "v1"),
        ("c015.ep75", "in100.vicreg.s0.hpull_varcov.c015.ep75.ext", "v1"),
        ("c015.ep100", "in100.vicreg.s0.hpull_varcov.c015.ext", "v1L")]


def main():
    mus, rows = {}, []
    for tag, rid, lane in RUNS:
        X = np.load(f"{ROOT}/features/{rid}/in100.train500.{lane}/student.h.cls.npy").astype(np.float64)
        mu = X.mean(0)
        Xc = X - mu
        tr = float((Xc ** 2).sum(1).mean())
        mus[tag] = mu
        rows.append({"tag": tag, "mu_sq": round(float(mu @ mu), 2), "tr_sigma": round(tr, 2),
                     "e_xsq": round(float((X ** 2).sum(1).mean()), 2),
                     "mu_share": round(float(mu @ mu) / float((X ** 2).sum(1).mean()), 4)})
    for r in rows:
        u = mus[r["tag"]]
        r["cos_to_ep100"] = round(float(u @ mus["c015.ep100"] /
                                        (np.linalg.norm(u) * np.linalg.norm(mus["c015.ep100"]) + 1e-12)), 4)
        r["cos_to_ctrl"] = round(float(u @ mus["ctrl"] /
                                       (np.linalg.norm(u) * np.linalg.norm(mus["ctrl"]) + 1e-12)), 4)
        print(r, flush=True)
    out = f"{ROOT}/results/diag/e17_mu_drift.csv"
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print("wrote", out)


if __name__ == "__main__":
    main()
