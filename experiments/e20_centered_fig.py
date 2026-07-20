"""E20 probe figure: linear panel + raw-vs-centered kNN200 at declared h, control and e20f arm
per zoo lane (Berker 2026-07-20: "for e20 we dont have linear probe fig — add a linear probe
panel"). Top: offline linear_raw_v2 — affine-invariant (the bias absorbs the mean), so the
comparison stays raw-vs-raw by construction. Bottom: the E20 reporting convention behind
E20-T3's knn_c deltas — the hatched gap raw->centered on the CONTROL bar is the free centering
worth a raw comparison would have overstated; arm bars are centering-invariant (cones trained
away). Reads results/diag/e20_centered.csv (run/space per lane, e17c ctrl flavor where the
kNN pass used it) + results/probes/<run>.csv -> results/figures/e20/e20_centered.png.
RAW rendering, no takeaway."""
import csv
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = "/nfs/scistore19/locatgrp/bdemirel/ssl_project"
ORDER = ["lejepa", "byol", "simclr", "vicreg", "dino", "ijepa", "mae"]
df = pd.read_csv(f"{ROOT}/results/diag/e20_centered.csv").drop_duplicates()
df["side"] = np.where(df.run.str.contains(r"\.e20f"), "arm", "ctrl")


def lin(run, space):
    path = f"{ROOT}/results/probes/{run}.csv"
    if not os.path.exists(path):
        return None
    for r in csv.DictReader(open(path)):
        if r["space"] == space and r["probe"] == "linear_raw_v2":
            return float(r["val_acc"])
    return None


fig, (axl, ax) = plt.subplots(2, 1, figsize=(11, 6.4), sharex=True, facecolor="white")
for li, lane in enumerate(ORDER):
    g = df[df.method == lane]
    if len(g) < 2:
        continue
    for si, (side, color) in enumerate([("ctrl", "#8c8c8c"), ("arm", "#3d65d0")]):
        r = g[g.side == side].iloc[0]
        x = li * 2.6 + si
        v = lin(r.run, r.space)
        if v is not None:
            axl.bar(x, v, width=0.62, color=color)
            axl.text(x, v, f"{v*100:.1f}", ha="center", va="bottom", fontsize=6.5)
        ax.bar(x - 0.21, r.knn200_raw, width=0.4, color=color)
        ax.bar(x + 0.21, r.knn200_centered, width=0.4, color=color, alpha=0.45,
               hatch="//", edgecolor="white")
        ax.text(x - 0.21, r.knn200_raw, f"{r.knn200_raw*100:.1f}", ha="center",
                va="bottom", fontsize=6.5)
        ax.text(x + 0.21, r.knn200_centered, f"{r.knn200_centered*100:.1f}", ha="center",
                va="bottom", fontsize=6.5)
axl.set_title("offline linear probe (raw v2) @ declared h — affine-invariant, raw both sides",
              fontsize=9.5)
axl.set_ylabel("lin acc", fontsize=9)
ax.set_title("kNN200 @ declared h — raw (solid) vs centered (hatched); the ctrl hatch gap = free centering worth",
             fontsize=9.5)
ax.set_ylabel("kNN200 acc", fontsize=9)
ax.set_xticks([i * 2.6 + 0.5 for i in range(len(ORDER))])
ax.set_xticklabels(ORDER, fontsize=9)
for a in (axl, ax):
    a.margins(y=0.18)
    a.tick_params(length=0, labelsize=8)
    for s in ("top", "right"):
        a.spines[s].set_visible(False)
fig.suptitle("E20: offline probes at declared h — control (grey) and e20f floor arm (blue) per lane (RAW)",
             fontsize=10.5)
fig.tight_layout(rect=(0, 0, 1, 0.96))
out = f"{ROOT}/results/figures/e20/e20_centered.png"
fig.savefig(out, dpi=160)
print("wrote", out)
