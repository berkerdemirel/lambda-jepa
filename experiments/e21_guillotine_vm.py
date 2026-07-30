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
FAMILIES = [   # Berker 2026-07-22: zonly = the grey CONTROL in every row
    ("d256 (pooled)", [("zonly ctrl", "vm3zonly", "in100.floorssl.s0.d256vm3zonly.extL", "#8c8c8c"),
                       ("d256", "d256", "in100.floorssl.s0.d256.extL", "#1a2f6e")]),
    ("vm2", [("zonly ctrl", "vm3zonly", "in100.floorssl.s0.d256vm3zonly.extL", "#8c8c8c"),
             ("vm2", "vm2", "in100.floorssl.s0.d256vm2.extL", "#6a3fb5")]),
    ("vm3", [("zonly ctrl", "vm3zonly", "in100.floorssl.s0.d256vm3zonly.extL", "#8c8c8c"),
             ("vm3", "vm3", "in100.floorssl.s0.d256vm3.extL", "#8a5cb8")]),
    ("vm3x2", [("zonly ctrl", "vm3zonly", "in100.floorssl.s0.d256vm3zonly.extL", "#8c8c8c"),
               ("vm3x2", "vm3x2", "in100.floorssl.s0.d256vm3x2.extL", "#d4820a")]),
    ("vm4 (matched vm3)", [("zonly ctrl", "vm3zonly", "in100.floorssl.s0.d256vm3zonly.extL", "#8c8c8c"),
                           ("vm4", "vm4", "in100.floorssl.s0.d256vm4.extL", "#c23b3b")]),
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

# Ω/a/b/Λ columns from the E23 orbit calculus (Berker 2026-07-30): per-station W/B from
# the o8 retro (results/diag/e23_retro_spaces.csv, raw framing); a/b/Λ oriented
# station→z.out (remaining-path transmission: cls = the declared headline, ≡1 at z.out).
WB, WBo = {}, {}
_PREF = {"dino_ctrl": "teacher", "dino_e20f": "teacher"}   # declared teacher lane (zoo)
for _fam, _members in FAMILIES:
    for _lab, _mlab, _run, _c in _members:
        pref = _PREF.get(_mlab, "student")
        for r in csv.DictReader(open(f"{ROOT}/results/diag/e23_retro_spaces.csv")):
            if r["run"] != _run or r["framing"] != "raw":
                continue
            st = station_of(r["space"])
            if st and (r["space"].startswith(pref) or (_mlab, st) not in WB):
                WB[(_mlab, st)] = (float(r["W"]), float(r["B"]))
        if (_mlab, "z.out") in WB:
            WBo[_mlab] = WB[(_mlab, "z.out")]


def wbget(kind):
    def g(mlab, st):
        wb, o = WB.get((mlab, st)), WBo.get(mlab)
        if wb is None or o is None:
            return None
        W, B = wb
        Wo, Bo = o
        if kind == "omega":
            return W / B
        a, b = (Wo / W) ** 0.5, (Bo / B) ** 0.5
        return a if kind == "a" else b if kind == "b" else b / a
    return g


def _lims(g):
    vs = [g(mlab, st) for _, mem in FAMILIES for _, mlab, _, _ in mem
          for st in STATIONS if g(mlab, st) is not None]
    if not vs:
        return (0, 1)
    lo, hi = min(vs), max(vs)
    pad = 0.06 * ((hi - lo) or 1)
    return (lo - pad, hi + pad)


QUANTS += [("Ω (W/B)", wbget("omega"), _lims(wbget("omega"))),
           ("a(s→z.out)", wbget("a"), _lims(wbget("a"))),
           ("b(s→z.out)", wbget("b"), _lims(wbget("b"))),
           ("Λ(s→z.out)", wbget("lam"), _lims(wbget("lam")))]

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
fig.suptitle("floorssl family guillotine — d256 · vm2 · vm3 · vm3x2 · vm4 rows, zonly (no h-conditioner) as the grey control in every row — 12 quantities vs depth (+Ω · a·b·Λ station→z.out, o8 orbit calculus); "
             "y fixed 0–1 for bounded quantities, ranks ÷ station dimension, gauss_kl per-method rows (zoo2) or column-shared "
             "(L-taps = trunk cls readouts; dotted = trunk|head; class-cos as the MARGIN same−diff) — RAW",
             fontsize=9.5, y=0.998)
fig.tight_layout(rect=(0, 0.008, 1, 0.98))
out = f"{ROOT}/results/figures/e21/e21_guillotine_vm.png"
fig.savefig(out, dpi=150)
print("wrote", out)
