"""E17 figure: augmentation-cloud touch% vs kNN@h (Berker 2026-07-16: record the connectivity
finding visually). One panel; per-family polylines in intervention order (control -> +spread ->
+spread+inv). Reads results/diag/e17_overlap.csv (e17_overlap.py) + results/probes/<run>.csv.
RAW rendering; the path shapes ARE the finding: +inv legs move down-left (connectivity cost)
in every family; spread legs move right, with kNN up except where the term carries shape
damage (lejepa's own SIGReg)."""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

ROOT = "/nfs/scistore19/locatgrp/bdemirel/ssl_project"
ov = pd.read_csv(f"{ROOT}/results/diag/e17_overlap.csv")


def knn(run, space):
    df = pd.read_csv(f"{ROOT}/results/probes/{run}.csv")
    v = df[(df.space == space) & (df.probe == "knn_v1_k200")].val_acc
    return 100 * float(v.iloc[0])


ov["knn200"] = [knn(r, s) for r, s in zip(ov.run, ov.space)]

FAMS = [  # (method key, label, color, tag order = intervention order)
    ("lejepa_e12", "lejepa E12 lane: moment-KL floor (+inv)", "#08519c", ["C1", "f2", "f8_weak_inv", "f7_strong_inv"]),
    ("lejepa", "lejepa E17: own SIGReg (+inv)", "#6baed6", ["ctrl", "sigreg", "sigreg_inv"]),
    ("simclr", "simclr: uniformity (+align)", "#807dba", ["ctrl", "uniform", "uniform_align"]),
    ("byol", "byol: predictive align", "#238b45", ["ctrl", "align"]),
    ("dino", "dino: proto-CE", "#525252", ["ctrl", "protoce"]),
    ("vicreg", "vicreg control", "#e6550d", ["ctrl"]),
]
NICE = {"C1": "C1", "f2": "f2", "f8_weak_inv": "f8", "f7_strong_inv": "f7", "ctrl": "ctrl",
        "sigreg": "sigreg", "sigreg_inv": "+inv", "uniform": "uniform", "uniform_align": "+align",
        "align": "align", "protoce": "protoce"}

fig, ax = plt.subplots(figsize=(8.6, 6.4), facecolor="white")
for key, label, c, order in FAMS:
    g = ov[ov.method == key].set_index("tag").loc[order]
    x, y = 100 * g.frac_touch.to_numpy(), g.knn200.to_numpy()
    ax.plot(x, y, "-", color=c, lw=1.6, alpha=0.85, zorder=2, label=label)
    ax.scatter(x, y, s=[70] + [46] * (len(x) - 1), color=c, zorder=3,
               edgecolor="white", linewidth=1.2)
    for xi, yi, t in zip(x, y, order):
        ax.annotate(NICE[t], (xi, yi), textcoords="offset points", xytext=(6, 6),
                    fontsize=8.5, color=c)
    for i in range(len(x) - 1):                      # arrowheads mid-segment
        ax.annotate("", xy=((x[i] + x[i + 1]) / 2, (y[i] + y[i + 1]) / 2),
                    xytext=(x[i], y[i]),
                    arrowprops=dict(arrowstyle="-|>", color=c, lw=0, mutation_scale=13,
                                    shrinkA=0, shrinkB=0), zorder=2)
ax.set_xlabel("augmentation-cloud touch %   (frac. images whose view-cloud reaches its same-class NN;  2r ≥ d_intra)", fontsize=9.5)
ax.set_ylabel("kNN k=200 @ declared h  (%)", fontsize=9.5)
ax.set_title("E17/H-wave — connectivity vs neighborhoods: every +inv leg moves down-left (RAW)\n"
             "large dot = control of its family; arrows = control → +spread → +spread+inv",
             fontsize=10)
ax.grid(alpha=0.25, lw=0.6)
for s in ("top", "right"):
    ax.spines[s].set_visible(False)
ax.legend(fontsize=8, frameon=False, loc="lower left")
out = f"{ROOT}/results/figures/e17/e17_touch_vs_knn.png"
os.makedirs(os.path.dirname(out), exist_ok=True)
fig.savefig(out, dpi=170, bbox_inches="tight", facecolor="white")
print(f"[e17ovfig] wrote {out}")
