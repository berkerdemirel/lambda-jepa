"""E20 centered-probe figure: raw vs centered kNN200 at declared h, control and e20f arm per
zoo lane (the reporting convention behind E20-T3's knn_c deltas made visible — the hatched gap
raw->centered on the CONTROL bar is the free centering worth the raw comparison would have
overstated). Reads results/diag/e20_centered.csv -> results/figures/e20/e20_centered.png.
RAW rendering, no takeaway."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = "/nfs/scistore19/locatgrp/bdemirel/ssl_project"
ORDER = ["lejepa", "byol", "simclr", "vicreg", "dino", "ijepa", "mae"]
df = pd.read_csv(f"{ROOT}/results/diag/e20_centered.csv")
df["side"] = np.where(df.run.str.contains(r"\.e20f"), "arm", "ctrl")

fig, ax = plt.subplots(figsize=(11, 3.4), facecolor="white")
for li, lane in enumerate(ORDER):
    g = df[df.method == lane]
    if len(g) < 2:
        continue
    for si, (side, color) in enumerate([("ctrl", "#8c8c8c"), ("arm", "#3d65d0")]):
        r = g[g.side == side].iloc[0]
        x = li * 2.6 + si
        ax.bar(x - 0.21, r.knn200_raw, width=0.4, color=color)
        ax.bar(x + 0.21, r.knn200_centered, width=0.4, color=color, alpha=0.45,
               hatch="//", edgecolor="white")
        ax.text(x - 0.21, r.knn200_raw, f"{r.knn200_raw*100:.1f}", ha="center",
                va="bottom", fontsize=6.5)
        ax.text(x + 0.21, r.knn200_centered, f"{r.knn200_centered*100:.1f}", ha="center",
                va="bottom", fontsize=6.5)
ax.set_xticks([i * 2.6 + 0.5 for i in range(len(ORDER))])
ax.set_xticklabels(ORDER, fontsize=9)
ax.margins(y=0.18)
ax.set_ylabel("kNN200 @ declared h", fontsize=9)
ax.set_title("E20: kNN raw (solid) vs centered (hatched) — control (grey) and e20f arm (blue) per lane",
             fontsize=10)
ax.tick_params(length=0, labelsize=8)
for s in ("top", "right"):
    ax.spines[s].set_visible(False)
fig.tight_layout()
out = f"{ROOT}/results/figures/e20/e20_centered.png"
fig.savefig(out, dpi=160)
print("wrote", out)
