"""E17 centered-vs-raw figure (Berker 2026-07-16: "generate centered versions of the figures,
where it matters"). Only the mean-sensitive readouts change under centering: the cosine-kNN probe
(train-mean removed at eval, bank+queries) and the orbit view-alignment pos (pooled-mean removed).
Linear probes and the shape battery (EP/kurt/effrank/RankMe/corr) are location-free already.
Reads results/diag/e17_centered.csv (written by e17_centered_probe.py). RAW rendering, no takeaway."""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = "/nfs/scistore19/locatgrp/bdemirel/ssl_project"
df = pd.read_csv(f"{ROOT}/results/diag/e17_centered.csv")
GROUPS = [("lejepa", "lejepa (shipped lane)"), ("lejepa_e12lane", "lejepa (E12 lane: C1/f2)"),
          ("simclr", "simclr"), ("byol", "byol"), ("dino", "dino"), ("vicreg", "vicreg")]
PANELS = [("kNN k=200 %", "knn200", "knn200_ctr"), ("kNN k=20 %", "knn20", "knn20_ctr"),
          ("view-align pos_cos", "pos_raw", "pos_c")]
BASE = {"lejepa": "#4292c6", "lejepa_e12lane": "#08519c", "simclr": "#807dba",
        "byol": "#238b45", "dino": "#525252", "vicreg": "#e6550d"}

fig, axes = plt.subplots(len(GROUPS), len(PANELS), figsize=(11, 2.1 * len(GROUPS)),
                         facecolor="white")
for gi, (key, label) in enumerate(GROUPS):
    g = df[df.method == key]
    for pi, (title, craw, cctr) in enumerate(PANELS):
        ax = axes[gi, pi]
        x = np.arange(len(g))
        ax.bar(x - 0.19, g[craw], width=0.36, color=BASE[key], label="raw")
        ax.bar(x + 0.19, g[cctr], width=0.36, color=BASE[key], alpha=0.45,
               hatch="//", edgecolor="white", label="centered")
        for xi, (vr, vc) in enumerate(zip(g[craw], g[cctr])):
            ax.text(xi - 0.19, vr, f"{vr:.3g}", ha="center", va="bottom", fontsize=6.5)
            ax.text(xi + 0.19, vc, f"{vc:.3g}", ha="center", va="bottom", fontsize=6.5)
        ax.set_xticks(x)
        ax.set_xticklabels(g.tag, fontsize=7.5)
        ax.margins(y=0.25)
        ax.set_yticks([])
        ax.tick_params(length=0)
        for s in ("top", "right", "left"):
            ax.spines[s].set_visible(False)
        if gi == 0:
            ax.set_title(title, fontsize=9)
        if pi == 0:
            ax.set_ylabel(label, fontsize=8)
        if gi == 0 and pi == 0:
            ax.legend(fontsize=7, frameon=False, loc="lower right")
fig.suptitle("E17 — mean-sensitive readouts, raw vs centered (kNN: train-mean removed at eval; "
             "pos: pooled-mean removed). RAW.", fontsize=10, y=0.995)
fig.tight_layout(rect=(0, 0, 1, 0.98))
out = f"{ROOT}/results/figures/e17/e17_centered.png"
os.makedirs(os.path.dirname(out), exist_ok=True)
fig.savefig(out, dpi=150, bbox_inches="tight", facecolor="white")
print(f"[e17cfig] wrote {out}")
