"""floorssl-family guillotine (Berker 2026-07-21: "when in100 runs land, do the same figure
between the models: vm3zonly, vm3x2, vm3, vm2, d256, d128 and d64"): 7 cell rows x 8
quantity columns vs depth, one curve per row, under the zoo2 rules (no gap station, 0-1
axes for bounded quantities, ranks / per-station d with z-out d annotations, gauss_kl per
row). The h-dose triplet (zonly 0x / vm3 1x / vm3x2 2x) + payment pair (vm2/vm3) + the dim
bracket tail (d256/d128/d64). RAW, no takeaway."""
import csv
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = "/nfs/scistore19/locatgrp/bdemirel/ssl_project"
STATIONS = ["L03", "L06", "L09", "gap", "cls", "embed", "tap1", "tap2", "z.out"]
FAMILIES = [
    ("vm4 (matched vm3)", [("vm4", "vm4", "in100.floorssl.s0.d256vm4.extL", "#c23b3b")]),
    ("vm3zonly (h 0×)", [("zonly", "vm3zonly", "in100.floorssl.s0.d256vm3zonly.extL", "#0f7b8a")]),
    ("vm3x2 (h/z 2×)", [("x2", "vm3x2", "in100.floorssl.s0.d256vm3x2.extL", "#d4820a")]),
    ("vm3 (symmetric)", [("vm3", "vm3", "in100.floorssl.s0.d256vm3.extL", "#8a5cb8")]),
    ("vm2 (z-only vm)", [("vm2", "vm2", "in100.floorssl.s0.d256vm2.extL", "#6a3fb5")]),
    ("d256 (pooled)", [("d256", "d256", "in100.floorssl.s0.d256.extL", "#1a2f6e")]),
    ("d128", [("d128", "d128", "in100.floorssl.s0.d128.extL", "#3d65d0")]),
    ("d64", [("d64", "d64", "in100.floorssl.s0.d64.extL", "#7bb3d9")]),
]


def station_of(s):
    for k in ("L03", "L06", "L09"):
        if s.endswith(f".{k}"):
            return k if ".h.cls." in s else None
    if s.endswith(".z.dec.tap2"):
        return "tap1"        # mae decoder blocks onto head stations in depth order (declared)
    if s.endswith(".z.dec.tap5"):
        return "tap2"
    if s.endswith(".z.dec.tap8"):
        return "z.out"
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
    if s.endswith(".z.proj.out") or s.endswith(".bottleneck") or s.endswith(".z.pred.out"):
        return "z.out"
    return None


P = {}
for fam, members in FAMILIES:
    for lab, mlab, run, c in members:
        path = f"{ROOT}/results/probes/{run}.csv"
        if os.path.exists(path):
            for r in csv.DictReader(open(path)):
                if r["space"].startswith("student"):
                    st = station_of(r["space"])
                    if st:
                        P[(mlab, st, r["probe"])] = float(r["val_acc"])

M = {}
for mp in (f"{ROOT}/results/diag/e21_vm_depth_metrics.csv",):
    if os.path.exists(mp):
        for r in csv.DictReader(open(mp)):
            st = station_of(r["space"])
            if st:
                M[(r["ep"], st)] = r


def mget(key, per_d=False):
    def g(mlab, st):
        r = M.get((mlab, st))
        v = None if r is None else r.get(key)
        if v in (None, "", "None"):
            return None
        return float(v) / float(r["d"]) if per_d else float(v)
    return g


QUANTS = [  # (name, getter, ylim: (0,1) | None=column-shared autoscale)
    ("lin_v2", lambda m, s: P.get((m, s, "linear_raw_v2")), (0, 1)),
    ("knn200", lambda m, s: P.get((m, s, "knn_v1_k200")), (0, 1)),
    ("rankme / d", mget("rankme", per_d=True), (0, 1)),
    ("effrank / d", mget("effective_rank", per_d=True), (0, 1)),
    ("gauss_kl_full", mget("gauss_kl_total"), None),
    ("pos_cos", mget("pos_cos"), (0, 1)),
    ("rand_cos", mget("rand_cos"), (0, 1)),
    ("class margin (same−diff)",
     lambda m, s: (lambda a, b: None if a is None or b is None else a - b)(
         mget("class_cos_same")(m, s), mget("class_cos_diff")(m, s)), (0, 1)),
]

ZOO2 = "--zoo2" in sys.argv     # per-method gauss ylims + z-out d annotations + NO gap
                                 # station (Berker: parallel pooling readout, not a depth step)
if ZOO2:
    STATIONS = [s for s in STATIONS if s != "gap"]
gk = mget("gauss_kl_total")
gk_vals = [gk(mlab, st) for _, mem in FAMILIES for _, mlab, _, _ in mem
           for st in STATIONS if gk(mlab, st) is not None]
GK_LIM = (0, max(gk_vals) * 1.08) if gk_vals else None


def row_gk_lim(members):
    vals = [gk(mlab, st) for _, mlab, _, _ in members for st in STATIONS
            if gk(mlab, st) is not None]
    return (0, max(vals) * 1.12) if vals else GK_LIM

fig, axes = plt.subplots(len(FAMILIES), len(QUANTS), figsize=(2.45 * len(QUANTS), 2.0 * len(FAMILIES)),
                         facecolor="white")
for row, (fam, members) in enumerate(FAMILIES):
    for col, (qname, get, ylim) in enumerate(QUANTS):
        ax = axes[row, col]
        for lab, mlab, run, c in members:
            def draw(g, ls="-"):
                xs = [i for i, st in enumerate(STATIONS) if g(mlab, st) is not None]
                ys = [g(mlab, STATIONS[i]) for i in xs]
                if xs:
                    ax.plot(xs, ys, ls, marker="o", color=c, ms=2.6, lw=1.0,
                            label=lab if (col == 0 and ls == "-") else None)
            draw(get)
        ax.set_ylim(*(ylim if ylim else (row_gk_lim(members) if ZOO2 else GK_LIM)))
        if ZOO2 and qname in ("rankme / d", "effrank / d"):
            for lab, mlab, run, c in members:          # annotate the z-out dimension so the
                r = M.get((mlab, "z.out"))             # fraction is interpretable per lane
                v = get(mlab, "z.out")
                if r is not None and v is not None:
                    ax.annotate(f"d={r['d']}", (STATIONS.index("z.out"), v), fontsize=4.6,
                                color=c, xytext=(2, 3), textcoords="offset points")
                    break
        ax.set_xticks(range(len(STATIONS)))
        ax.set_xticklabels(STATIONS if row == len(FAMILIES) - 1 else [], fontsize=5.2,
                           rotation=45)
        ax.axvline(STATIONS.index("cls") + 0.5, color="#bbbbbb", lw=0.7, ls=":")
        if row == 0:
            ax.set_title(qname, fontsize=8)
        if col == 0:
            ax.set_ylabel(fam, fontsize=8.5)
            ax.legend(fontsize=4.6, frameon=False, loc="upper left")
        ax.tick_params(length=0, labelsize=5.6)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
fig.suptitle("floorssl family guillotine — h-dose triplet (zonly/vm3/vm3x2) · payment pair (vm2/vm3) · dim bracket (d256/128/64) — 8 quantities vs depth; "
             "y fixed 0–1 for bounded quantities, ranks ÷ station dimension, gauss_kl per-method rows (zoo2) or column-shared "
             "(L-taps = trunk cls readouts; dotted = trunk|head; class-cos as the MARGIN same−diff) — RAW",
             fontsize=9.5, y=0.998)
fig.tight_layout(rect=(0, 0.008, 1, 0.98))
out = f"{ROOT}/results/figures/e21/e21_guillotine_vm.png"
fig.savefig(out, dpi=150)
print("wrote", out)
