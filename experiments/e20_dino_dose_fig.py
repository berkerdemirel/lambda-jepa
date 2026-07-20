"""E20 dino dose curve (E20-T1 completion; lam008/lam06 landed 2026-07-20). Offline probes at
teacher.h.cls vs the h-floor dose λ: the five single-tap points {ctrl 0, .008, .02 gd, .06,
.258 e20f} + the two-tap gd2 as an open confounded marker. ctrl drawn as horizontal reference
lines (λ=0 has no log position); knn panel carries the centered-ctrl line (the knn_c
convention) with arm-side raw points (floor arms are centering-invariant, measured E20/E21).
Reads the probe CSVs + e20_centered.csv -> results/figures/e20/e20_dino_dose.png. RAW."""
import csv
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = "/nfs/scistore19/locatgrp/bdemirel/ssl_project"
SP = "teacher.h.cls"
CELLS = [(.008, "in100.dino.s0.e20f_lam008.ext"), (.02, "in100.dino.s0.e12gd.ext"),
         (.06, "in100.dino.s0.e20f_lam06.ext"), (.258, "in100.dino.s0.e20f.ext")]
GD2 = (.0601, "in100.dino.s0.e12gd2.ext")  # nominal .02+.0401 two-tap — confounded, open marker
CTRL = "in100.dino.s0.ext"


def probes(run):
    path = f"{ROOT}/results/probes/{run}.csv"
    return {(r["space"], r["probe"]): float(r["val_acc"]) for r in csv.DictReader(open(path))} \
        if os.path.exists(path) else {}


cen = {(r["run"], r["space"]): float(r["knn200_centered"])
       for r in csv.DictReader(open(f"{ROOT}/results/diag/e20_centered.csv"))}
P = {run: probes(run) for _, run in CELLS + [GD2]}
pc = probes(CTRL)

fig, axes = plt.subplots(1, 2, figsize=(10.5, 3.9), facecolor="white")
for ax, probe, title, ctrl_v, ctrl_lab in [
        (axes[0], "linear_raw_v2", "offline lin (raw v2)", pc[(SP, "linear_raw_v2")], "ctrl (λ=0)"),
        (axes[1], "knn_v1_k200", "offline kNN200 (arm raw ≈ centered; ctrl CENTERED)",
         cen.get(("in100.dino.s0.e17c.ext", SP)), "ctrl centered (λ=0)")]:
    xs = [lam for lam, run in CELLS if P[run]]
    ys = [P[run][(SP, probe)] for lam, run in CELLS if P[run]]
    ax.plot(xs, ys, "-o", color="#3d65d0", ms=5, lw=1.4)
    for x, y in zip(xs, ys):
        ax.annotate(f"{y:.4f}", (x, y), textcoords="offset points", xytext=(0, 7),
                    ha="center", fontsize=6.5)
    if P[GD2[1]]:
        v = P[GD2[1]][(SP, probe)]
        ax.plot([GD2[0]], [v], "o", mfc="none", mec="#3d65d0", ms=6)
        ax.annotate(f"gd2 (two-tap) {v:.4f}", (GD2[0], v), textcoords="offset points",
                    xytext=(0, -12), ha="center", fontsize=6, color="#666666")
    ax.axhline(ctrl_v, ls="--", color="#8c8c8c", lw=1.2)
    ax.text(.0085, ctrl_v, f"{ctrl_lab} {ctrl_v:.4f}", fontsize=6.5, color="#555555",
            va="bottom")
    ax.set_xscale("log")
    ax.set_xticks([lam for lam, _ in CELLS])
    ax.set_xticklabels([".008", ".02\n(gd)", ".06", ".258\n(e20f)"], fontsize=7)
    ax.set_xlabel("h-floor dose λ (log)", fontsize=8)
    ax.set_title(title, fontsize=9)
    ax.tick_params(length=0, labelsize=7)
    ax.margins(y=0.2)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
fig.suptitle("E20 dino h-floor dose curve @teacher.h.cls, ep100 offline — E20-T1 completion (RAW)",
             fontsize=10)
fig.tight_layout(rect=(0, 0.02, 1, 0.93))
fig.text(0.01, 0.005, "arm knn points are raw; floor arms measured centering-invariant (E20/E21) · ctrl knn line is the CENTERED value "
         "(e17c-flavor store; raw .5994) · online monitors ran higher (lam008 .7092 / lam06 .6980) — monitor optimism, offline is the record",
         fontsize=6, color="#555555")
out = f"{ROOT}/results/figures/e20/e20_dino_dose.png"
fig.savefig(out, dpi=160)
print("wrote", out)
