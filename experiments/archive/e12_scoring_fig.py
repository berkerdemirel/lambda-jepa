"""E12 scoring overview figure (presentation artifact; numbers verbatim from
results/probes/*.csv + results/diag/e12_class_align.csv). Three panels:
(A) moment-KL dose-response at h (probes), (B) dose-response (geometry),
(C) all-arm probe-pair map. Okabe-Ito CVD-safe colors; distinct markers as
secondary encoding; direct labels; single axis per panel."""
import csv
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = "/nfs/scistore19/locatgrp/bdemirel/ssl_project"
BLUE, VERM, GREEN, GRAY = "#0072B2", "#D55E00", "#009E73", "#666666"


def probes(rid):
    out = {}
    with open(f"{ROOT}/results/probes/{rid}.csv") as f:
        for r in csv.DictReader(f):
            if r["space"] == "student.z.embed":
                out[r["probe"]] = float(r["val_acc"])
    return out


align = {}
with open(f"{ROOT}/results/diag/e12_class_align.csv") as f:
    for r in csv.DictReader(f):
        align[r["run"].replace("in100.lejepa.s0.", "").replace(".ext", "") or "lane"] = r

P = {a: probes(f"in100.lejepa.s0.{a}.ext") for a in
     ["e12a1", "e12a2", "e12a3", "e12c1", "e12f1", "e12f2", "e12f3", "e12f4", "e12f5", "e12f6"]}
P["lane"] = probes("in100.lejepa.s0.ext")
align["lane"] = align.pop("ext")
A = lambda a, k: float(align[a][k])

fig, axes = plt.subplots(1, 3, figsize=(15, 4.6))
fig.subplots_adjust(wspace=0.28, left=0.05, right=0.99, bottom=0.13, top=0.86)

# --- Panel A: moment-KL dose-response, probes -------------------------------------------------
ax = axes[0]
lam = [0.0, 0.02, 0.10, 0.478]
arms = ["e12c1", "e12f2", "e12f1", "e12a3"]
for key, col, lab, mk in [("linear_raw_v2", BLUE, "linear (converged)", "o"),
                          ("knn_v1_k200", VERM, "kNN k=200", "s")]:
    ax.plot(lam, [P[a][key] for a in arms], marker=mk, color=col, lw=2, ms=7, label=lab)
    ax.plot([0.478], [P["e12f6"][key]], marker=mk, mfc="white", color=col, ms=9, ls="none")
ax.annotate("open = f6 (enabled ep10)", (0.478, P["e12f6"]["knn_v1_k200"]),
            textcoords="offset points", xytext=(-8, -16), fontsize=8, color=GRAY, ha="right")
ax.annotate("f2", (0.02, P["e12f2"]["linear_raw_v2"]), textcoords="offset points",
            xytext=(0, 8), fontsize=9, ha="center")
ax.set_xlabel("moment-KL weight λ_h at h (0 = control C1)")
ax.set_ylabel("val accuracy")
ax.set_title("A · Moment floor at h: dose–response (probes)")
ax.legend(frameon=False, fontsize=9, loc="lower left")

# --- Panel B: dose-response, geometry ----------------------------------------------------------
ax = axes[1]
for key, col, lab, mk in [("kmeans100_NMI", GREEN, "class NMI (kmeans-100)", "o"),
                          ("between/total_var", GRAY, "between/total variance", "s")]:
    ax.plot(lam, [A(a, key) for a in arms], marker=mk, color=col, lw=2, ms=7, label=lab)
ax.axhline(A("null", "kmeans100_NMI"), color=GREEN, lw=1, ls=":")
ax.annotate("randinit-null NMI", (0.25, A("null", "kmeans100_NMI")), fontsize=8,
            color=GREEN, va="bottom")
ax.set_xlabel("moment-KL weight λ_h at h")
ax.set_ylabel("index (0–1)")
ax.set_title("B · Same arms: class structure vs anisotropy")
ax.legend(frameon=False, fontsize=9)

# --- Panel C: all-arm probe-pair map ------------------------------------------------------------
ax = axes[2]
fam = {"e12c1": (GRAY, "C1 (control)", "o"), "lane": (GRAY, "M2 lane", "D"),
       "e12f2": (BLUE, "f2 KL@.02", "o"), "e12f1": (BLUE, "f1 KL@.10", "s"),
       "e12a3": (BLUE, "A3 KL@.478", "^"), "e12f6": (BLUE, "f6 KL@.478 ep10", "v"),
       "e12f5": (GREEN, "f5 diag@.55", "o"), "e12f3": (GREEN, "f3 rankfloor@.48", "s"),
       "e12a2": (VERM, "A2 CF@.026", "o"), "e12f4": (VERM, "f4 stdCF@.071", "s"),
       "e12a1": (VERM, "A1 Dlr", "^")}
for a, (col, lab, mk) in fam.items():
    x, y = P[a]["linear_raw_v2"], P[a]["knn_v1_k200"]
    filled = col if a != "lane" else "white"
    ax.plot([x], [y], marker=mk, color=col, mfc=filled, ms=9, ls="none")
    off = {"e12c1": (10, -3, "left"), "e12f2": (0, 9, "center"), "lane": (-10, -4, "right"),
           "e12f1": (0, 9, "center"), "e12f5": (10, -3, "left"), "e12f3": (0, -14, "center"),
           "e12a3": (10, 2, "left"), "e12f6": (10, -10, "left"), "e12a2": (0, 9, "center"),
           "e12f4": (0, 9, "center"), "e12a1": (0, 9, "center")}[a]
    ax.annotate(lab, (x, y), textcoords="offset points", xytext=off[:2], fontsize=8,
                ha=off[2], color="#222222")
lims = [0.25, 0.70]
ax.plot(lims, lims, color="#cccccc", lw=1, zorder=0)
ax.annotate("kNN = linear", (0.655, 0.66), fontsize=8, color=GRAY, rotation=38)
ax.set_xlim(0.40, 0.70)
ax.set_ylim(0.22, 0.66)
ax.set_xlabel("converged linear at h")
ax.set_ylabel("kNN (k=200) at h")
ax.set_title("C · All arms: probe pair at h  (color = term family)")

for ax in axes:
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(alpha=0.25, lw=0.5)

out = f"{ROOT}/results/figures/e12/e12_scoring_overview.png"
os.makedirs(os.path.dirname(out), exist_ok=True)
fig.savefig(out, dpi=160)
print("wrote", out)
