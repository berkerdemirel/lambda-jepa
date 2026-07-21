"""e20f cadence guillotine, full-spec (Berker 2026-07-21: "present rankme curves effective
rank curves, linear probes, knn accs, gaussianity metric curves, pos invariance curves,
random invariance curves, class cos curves"). 2x4 panels, each = one quantity vs depth
station (L03 L06 L09 gap cls embed tap1 tap2 z.out), four curves = the cadence ckpts
ep25 (light) -> ep100 (dark). Probes from the merged per-space CSVs (raw+knn diagnostic
subset, .extL stores); the six metric families from results/diag/e20f_depth_metrics.csv
(sslgap.metrics functions reused; pos/rand = views 0/1 of the audit_v1 o8 orbits — the
pairs path cannot carry L-taps, the orbit path can). RAW, no takeaway."""
import csv
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = "/nfs/scistore19/locatgrp/bdemirel/ssl_project"
CKPTS = [("ep25", 25, "#b7e4c7"), ("ep50", 50, "#74c69d"),
         ("ep75", 75, "#2d6a4f"), ("ep100", 100, "#081c15")]
STATIONS = ["L03", "L06", "L09", "gap", "cls", "embed", "tap1", "tap2", "z.out"]


def station_of(s):
    for k in ("L03", "L06", "L09"):
        if s.endswith(f".{k}"):
            return k if ".h.cls." in s else None   # cls-readout L-taps (gap.Lkk plotted separately? keep cls path)
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
    if s.endswith(".z.proj.out"):
        return "z.out"
    return None


P = {}   # (ep, station, probe) -> acc
for lab, ep, _ in CKPTS:
    path = f"{ROOT}/results/probes/in100.lejepa.s0.e20f.ep{ep}.extL.csv"
    if os.path.exists(path):
        for r in csv.DictReader(open(path)):
            st = station_of(r["space"])
            if st:
                P[(ep, st, r["probe"])] = float(r["val_acc"])

M = {}   # (ep, station) -> metric row
mpath = f"{ROOT}/results/diag/e20f_depth_metrics.csv"
if os.path.exists(mpath):
    for r in csv.DictReader(open(mpath)):
        st = station_of(r["space"])
        if st:
            M[(int(r["ep"]), st)] = r


def curve(ax, ep, c, get, ls="-"):
    xs, ys = [], []
    for i, st in enumerate(STATIONS):
        v = get(ep, st)
        if v is not None:
            xs.append(i), ys.append(v)
    if xs:
        ax.plot(xs, ys, ls, marker="o", color=c, ms=3.5, lw=1.2)


def mget(key, cast=float):
    def g(ep, st):
        r = M.get((ep, st))
        if r is None or r.get(key) in (None, "", "None"):
            return None
        return cast(r[key])
    return g


PANELS = [
    ("linear probe (raw v2)", lambda ep, st: P.get((ep, st, "linear_raw_v2"))),
    ("kNN200", lambda ep, st: P.get((ep, st, "knn_v1_k200"))),
    ("rankme", mget("rankme")),
    ("effective_rank", mget("effective_rank")),
    ("gauss_kl_full.total (per dim)", mget("gauss_kl_total")),
    ("pos invariance (pos_cos, o8 audit_v1)", mget("pos_cos")),
    ("random invariance (rand_cos)", mget("rand_cos")),
    ("class cos — same (solid) / diff (dashed)", None),
]

fig, axes = plt.subplots(2, 4, figsize=(17.5, 7), facecolor="white")
for ax, (title, get) in zip(axes.ravel(), PANELS):
    for lab, ep, c in CKPTS:
        if get is not None:
            curve(ax, ep, c, get)
        else:
            curve(ax, ep, c, mget("class_cos_same"))
            curve(ax, ep, c, mget("class_cos_diff"), ls="--")
    ax.set_xticks(range(len(STATIONS)))
    ax.set_xticklabels(STATIONS, fontsize=6.4, rotation=45)
    ax.axvline(4.5, color="#bbbbbb", lw=0.8, ls=":")
    ax.set_title(title, fontsize=8.5)
    ax.tick_params(length=0, labelsize=6.5)
    ax.margins(y=0.12)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
axes[0, 0].legend(handles=[plt.Line2D([], [], color=c, lw=2, label=lab) for lab, _, c in CKPTS],
                  fontsize=6.5, frameon=False, loc="lower right")
fig.suptitle("e20f cadence guillotine — eight quantities vs depth station, ep25 (light) → ep100 (dark); "
             "dotted = trunk|head boundary; trunk L-taps = cls readout — RAW", fontsize=10, y=0.995)
fig.tight_layout(rect=(0, 0.03, 1, 0.955))
fig.text(0.01, 0.005, ".extL diagnostic stores (raw+knn probe subset; not the landing convention) · metrics from train500.v1L "
         "clean features via sslgap.metrics (rankme uncentered SVs; effrank centered eigs; gauss_kl_full = moment-KL to N(0,I)) · "
         "pos/rand from o8 audit_v1 views 0/1 (pair_margin) · class-cos exact label-conditioned means · declared h = embed",
         fontsize=5.8, color="#555555")
out = f"{ROOT}/results/figures/e20/e20f_guillotine.png"
os.makedirs(os.path.dirname(out), exist_ok=True)
fig.savefig(out, dpi=160)
print("wrote", out)
