"""Depth-ladder grid for the calibration pairs (Berker 2026-07-13): 3 rows x 3 methods —
(1) view-invariance cos margin (V=8 orbit stores, audit_v1; from e12g_orbit_invariance.csv),
(2) converged linear probes across the same ladder (L-tap probes trained fresh on the v1L
per-layer stores; final-tap values verbatim from the canon results/probes CSVs),
(3) Gaussianity across layers as diag moment-KL to N(0,I) (the floor's own matched loss;
e12_floor_values.diag_read convention; Epps-Pulley rejected as headline — foolable, never alone).

Replaces results/figures/e12g/e12g_depth_invariance.png in place. Raw numbers land in
results/diag/e12g_ladder_probes.csv + e12g_ladder_diagkl.csv.

  sbatch slurm/e12g_ladder.sbatch
"""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd

from sslgap.probes import linear_raw_v2

ROOT = "/nfs/scistore19/locatgrp/bdemirel/ssl_project"
FEAT = f"{ROOT}/features"
FIGD = f"{ROOT}/results/figures/e12g"

PAIRS = [
    ("lejepa", "in100.lejepa.s0.e12f2.ext", "in100.lejepa.s0.e12c1.ext", "student.z.embed"),
    ("vicreg", "in100.vicreg.s0.e12gv.ext", "in100.vicreg.s0.e12gvc.ext", "student.h.cls"),  # D-036: projector input
    ("dino", "in100.dino.s0.e12gd.ext", "in100.dino.s0.e12gdc.ext", "teacher.h.cls"),
]
COL = {"lejepa": "#0072B2", "vicreg": "#D55E00", "dino": "#009E73"}
LADDER = {
    "lejepa": ["student.h.cls.L03", "student.h.cls.L06", "student.h.cls.L09", "student.h.cls",
               "student.z.embed", "student.z.proj.tap1", "student.z.proj.tap2",
               "student.z.proj.out"],
    "vicreg": ["student.h.cls.L03", "student.h.cls.L06", "student.h.cls.L09", "student.h.cls",
               "student.z.proj.tap1", "student.z.proj.tap2", "student.z.proj.out"],
    "dino": ["teacher.h.cls.L03", "teacher.h.cls.L06", "teacher.h.cls.L09", "teacher.h.cls",
             "teacher.z.dino.tap1", "teacher.z.dino.tap2", "teacher.z.dino.bottleneck"],
}
LTAPS = {m: [t for t in taps if ".L0" in t] for m, taps in LADDER.items()}


def short(tap):
    return (tap.split(".", 1)[1].replace("h.cls", "h").replace("h.gap", "h")
            .replace("z.proj.", "").replace("z.dino.", "").replace("z.embed", "embed"))


def diag_read(X):
    mu = X.mean(0)
    var = X.var(0).clip(1e-8)
    return float(0.5 * (np.mean(mu ** 2) + np.mean(var - 1 - np.log(var))))


def canon_probe(run):
    df = pd.read_csv(f"{ROOT}/results/probes/{run}.csv")
    df = df[df.probe == "linear_raw_v2"]
    return {r.space: r.val_acc for r in df.itertuples()}  # last row wins on re-probes


def main():
    runs = [(m, run, h) for m, arm, ctl, h in PAIRS for run in (arm, ctl)]

    # ---- L-tap probes (fresh, v1L stores) + diag-KL over the full ladder (v1L) ---------------
    prows, krows = [], []
    for m, run, _h in runs:
        ytr = np.load(f"{FEAT}/{run}/in100.train500.v1L/labels.npy")
        yva = np.load(f"{FEAT}/{run}/in100.val.v1L/labels.npy")
        for tap in LADDER[m]:
            X = np.load(f"{FEAT}/{run}/in100.train500.v1L/{tap}.npy").astype(np.float32)
            krows.append({"run_id": run, "space": tap, "diag_kl": diag_read(X.astype(np.float64))})
            if tap in LTAPS[m]:
                Xv = np.load(f"{FEAT}/{run}/in100.val.v1L/{tap}.npy").astype(np.float32)
                r = linear_raw_v2(X, ytr, Xv, yva, num_classes=100)
                prows.append({"run_id": run, "space": tap, "probe": "linear_raw_v2",
                              "val_acc": r["val_acc"], "best_ep": r["best_ep"],
                              "epochs_run": r["epochs_run"], "n_train": len(ytr)})
                print(f"[ladder] {run} {tap}: {r['val_acc']:.4f} "
                      f"({r['best_ep']}/{r['epochs_run']})", flush=True)
                pd.DataFrame(prows).to_csv(f"{ROOT}/results/diag/e12g_ladder_probes.csv",
                                           index=False)
    pd.DataFrame(krows).to_csv(f"{ROOT}/results/diag/e12g_ladder_diagkl.csv", index=False)
    lp = {(r["run_id"], r["space"]): r["val_acc"] for r in prows}
    kl = {(r["run_id"], r["space"]): r["diag_kl"] for r in krows}
    canon = {run: canon_probe(run) for _m, run, _h in runs}
    inv = pd.read_csv(f"{ROOT}/results/diag/e12g_orbit_invariance.csv")
    inv = inv[inv["stack"] == "audit_v1"]
    iv = {(r.run_id, r.space): r.cos_margin for r in inv.itertuples()}

    # ---- 3x3 grid --------------------------------------------------------------------------
    fig, axes = plt.subplots(3, 3, figsize=(15, 10.5))
    rows = [("view-invariance margin\n(pos − rand cos, V=8)",
             lambda run, tap: iv.get((run, tap))),
            ("linear probe val acc\n(converged raw_v2)",
             lambda run, tap: lp.get((run, tap), canon[run].get(tap))),
            ("diag moment-KL to N(0,I)\n(the matched loss; log)",
             lambda run, tap: kl.get((run, tap)))]
    for ci, (m, arm, ctl, h) in enumerate(PAIRS):
        taps = LADDER[m]
        for ri, (ylab, get) in enumerate(rows):
            ax = axes[ri][ci]
            for run, ls, filled in ((arm, "-", True), (ctl, "--", False)):
                ys = [get(run, t) for t in taps]
                ax.plot(range(len(taps)), ys, ls=ls, marker="o", ms=5.5, lw=1.6, color=COL[m],
                        mfc=COL[m] if filled else "white", mew=1.3)
            ax.axvline(taps.index(h), color="#999999", lw=0.8, ls=":")
            if ri == 2:
                ax.set_yscale("log")
                ax.set_xticks(range(len(taps)), [short(t) for t in taps], rotation=45,
                              ha="right", fontsize=8)
            else:
                ax.set_xticks(range(len(taps)), [""] * len(taps))
            if ri == 0:
                ax.set_title(m, fontsize=11, color=COL[m])
            if ci == 0:
                ax.set_ylabel(ylab, fontsize=9)
            ax.grid(alpha=0.25, lw=0.5)
            ax.spines[["top", "right"]].set_visible(False)
    hs = [Line2D([], [], color="#444444", marker="o", ls="-", ms=6, label="floor arm"),
          Line2D([], [], color="#444444", marker="o", ls="--", ms=6, mfc="white", mew=1.3,
                 label="matched control")]
    fig.legend(handles=hs, loc="upper right", ncol=2, fontsize=9, frameon=False)
    fig.suptitle("Depth ladder, calibration pairs (audit_v1) — dotted line = calibrated tap. "
                 "Row 2: L-tap probes fresh on v1L stores, final taps = canon probe CSVs. "
                 "Row 3: diag moment-KL on v1L train500 features.", fontsize=10, x=0.02,
                 ha="left")
    fig.tight_layout(rect=(0, 0.005, 1, 0.94))
    fig.savefig(f"{FIGD}/e12g_depth_invariance.png", dpi=180)
    print(f"[ladder] wrote {FIGD}/e12g_depth_invariance.png + 2 CSVs")


if __name__ == "__main__":
    main()
