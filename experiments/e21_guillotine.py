"""Guillotine curves for the quad roster (Berker 2026-07-21: "where're the guillotine
curves"). Probe accuracy vs depth station: trunk L03/L06/L09 (from the .extL stores when
landed — skip-if-absent, the house pattern) then gap · cls · [embed] · tap1 · tap2 · z-out.
Depth is categorical (archs differ; per-run polylines over their own stations). Two panels:
linear_raw_v2 and knn_v1_k200. E02 remains the pre-registered full experiment; this is the
raw depth read for the current comparison roster. RAW, no takeaway."""
import csv
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = "/nfs/scistore19/locatgrp/bdemirel/ssl_project"
RUNS = [("d64", "in100.floorssl.s0.d64.ext", "#3d65d0", "student"),
        ("d256", "in100.floorssl.s0.d256.ext", "#1b2c60", "student"),
        ("d256vm2", "in100.floorssl.s0.d256vm2.ext", "#6a3fb5", "student"),
        ("d256e200", "in100.floorssl.s0.d256e200.ext", "#0f7b8a", "student"),
        ("e20f", "in100.lejepa.s0.e20f.ext", "#2e8b57", "student"),
        ("ctrl", "in100.lejepa.s0.ext", "#8c8c8c", "student"),
        ("dino λ.008", "in100.dino.s0.e20f_lam008.ext", "#d4820a", "teacher"),
        ("dino ctrl", "in100.dino.s0.ext", "#b39b77", "teacher")]
STATIONS = ["L03", "L06", "L09", "gap", "cls", "embed", "tap1", "tap2", "z.out"]


def probes(run):
    out = {}
    for path in (f"{ROOT}/results/probes/{run}.csv",
                 f"{ROOT}/results/probes/{run.replace('.ext', '.extL')}.csv"):
        if os.path.exists(path):
            for r in csv.DictReader(open(path)):
                out[(r["space"], r["probe"])] = float(r["val_acc"])
    return out


def station_of(space, branch):
    """Map a stored space name to its depth station (None = not on this run's path)."""
    s = space
    if not s.startswith(branch):
        return None
    for k in ("L03", "L06", "L09"):
        if s.endswith(f".{k}"):
            return k
    if s.endswith(".h.gap"):
        return "gap"
    if s.endswith(".h.cls"):
        return "cls"
    if s.endswith(".z.embed"):
        return "embed"
    if s.endswith("tap1"):
        return "tap1"
    if s.endswith("tap2"):
        return "tap2"
    if s.endswith(".z.proj.out") or s.endswith(".bottleneck"):
        return "z.out"
    return None


fig, (axl, axk) = plt.subplots(1, 2, figsize=(13, 4.6), facecolor="white")
for lab, run, c, branch in RUNS:
    P = probes(run)
    for ax, probe in ((axl, "linear_raw_v2"), (axk, "knn_v1_k200")):
        pts = {}
        for (space, pr), v in P.items():
            if pr != probe:
                continue
            st = station_of(space, branch)
            if st:
                pts[st] = v
        xs = [i for i, st in enumerate(STATIONS) if st in pts]
        ys = [pts[STATIONS[i]] for i in xs]
        if xs:
            ax.plot(xs, ys, "-o", color=c, ms=4, lw=1.3, label=lab if ax is axl else None)
            ax.annotate(f"{ys[-1]:.3f}", (xs[-1], ys[-1]), xytext=(4, 0),
                        textcoords="offset points", fontsize=5.4, color=c)
for ax, t in ((axl, "linear_raw_v2 vs depth"), (axk, "knn_v1_k200 vs depth")):
    ax.set_xticks(range(len(STATIONS)))
    ax.set_xticklabels(STATIONS, fontsize=7)
    ax.axvline(4.5, color="#bbbbbb", lw=0.8, ls=":")   # trunk | head boundary (post-cls)
    ax.set_title(t, fontsize=9)
    ax.tick_params(length=0, labelsize=7)
    ax.margins(y=0.12)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
axl.legend(fontsize=6, frameon=False, loc="lower left", ncol=2)
fig.suptitle("E21 roster guillotine: offline v2 probes at every stored depth station — trunk L-taps join as .extL stores land "
             "(dotted line = trunk|head boundary; declared h = cls [teacher for dino]; dino z.out = bottleneck) — RAW",
             fontsize=9.5)
fig.tight_layout(rect=(0, 0.02, 1, 0.93))
fig.text(0.01, 0.005, "stations are categorical (archs differ); lejepa runs carry the extra embed station; probes = converged v2 (D-020)",
         fontsize=6, color="#555555")
out = f"{ROOT}/results/figures/e21/e21_guillotine.png"
fig.savefig(out, dpi=160)
print("wrote", out)
