"""E12 dose figure v2 (Berker 2026-07-12: 'i wanna see the same curve where we have the
corresponding loss on y axis'). Left = probes vs dose (as before); right = the floor's own
satisfaction at h (audit-side moment-KL, results/diag/e12_floor_values.csv) vs dose, log-y.
Same x, same arms; f6 (delayed) as open markers at lambda=.478."""
import csv

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = "/nfs/scistore19/locatgrp/bdemirel/ssl_project"
BLUE, VERM, GREEN = "#0072B2", "#D55E00", "#009E73"


def probes(rid):
    out = {}
    with open(f"{ROOT}/results/probes/{rid}.csv") as f:
        for r in csv.DictReader(f):
            if r["space"] == "student.z.embed":
                out[r["probe"]] = float(r["val_acc"])
    return out


fv = {}
with open(f"{ROOT}/results/diag/e12_floor_values.csv") as f:
    for r in csv.DictReader(f):
        fv[r["arm"]] = float(r["moment_kl_sliced"])

lam = [0.0, 0.02, 0.10, 0.478]
arms = ["e12c1", "e12f2", "e12f1", "e12a3"]
P = {a: probes(f"in100.lejepa.s0.{a}.ext") for a in arms + ["e12f6"]}

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10.5, 4.4))
fig.subplots_adjust(wspace=0.28, left=0.07, right=0.98, bottom=0.14, top=0.86)

for key, col, lab, mk in [("linear_raw_v2", BLUE, "linear (converged)", "o"),
                          ("knn_v1_k200", VERM, "kNN k=200", "s")]:
    ax1.plot(lam, [P[a][key] for a in arms], marker=mk, color=col, lw=2, ms=7, label=lab)
    ax1.plot([0.478], [P["e12f6"][key]], marker=mk, mfc="white", color=col, ms=9, ls="none")
ax1.set_xlabel("moment-KL weight λ_h at h (0 = control C1)")
ax1.set_ylabel("val accuracy")
ax1.set_title("Probes vs dose")
ax1.legend(frameon=False, fontsize=9, loc="lower left")

ax2.plot(lam, [fv[a] for a in arms], marker="o", color=GREEN, lw=2, ms=7,
         label="moment-KL at h (audit estimator)")
ax2.plot([0.478], [fv["e12f6"]], marker="o", mfc="white", color=GREEN, ms=9, ls="none")
ax2.set_yscale("log")
for a, x in zip(arms, lam):
    ax2.annotate(f"{fv[a]:.2f}", (x, fv[a]), textcoords="offset points", xytext=(6, 6), fontsize=8)
ax2.annotate("f6 (delayed)", (0.478, fv["e12f6"]), textcoords="offset points", xytext=(-8, -14),
             fontsize=8, ha="right", color="#666666")
ax2.set_xlabel("moment-KL weight λ_h at h")
ax2.set_ylabel("floor value at h (log scale, lower = satisfied)")
ax2.set_title("Constraint satisfaction vs dose — λ=.02 already buys ~93% of it")

for ax in (ax1, ax2):
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(alpha=0.25, lw=0.5)

out = f"{ROOT}/results/figures/e12/e12_dose_vs_loss.png"
fig.savefig(out, dpi=160)
print("wrote", out)
