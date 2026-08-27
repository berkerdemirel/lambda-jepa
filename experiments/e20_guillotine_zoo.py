"""E20 zoo guillotine, method by method (Berker 2026-07-21: "i asked you to report these 8
metrics row by row changing the method. i talk about 20f experiment"): 7 method rows (each
lane's e20f floor arm, colored, vs its control, grey) x 8 quantity columns vs depth. Same
harmonized rules: bounded quantities on 0-1, ranks / per-station d, class-cos as margin,
gauss_kl column-shared. Controls: the e17c re-extractions (same ep100 ckpts) for the five
E17 methods; fresh .extL for mae/ijepa. mae's decoder taps map onto the head stations in
depth order (dec.tap2->tap1, tap5->tap2, tap8->z.out; declared); ijepa z = pooled pred.out.
dino's e20f = the calibrated overshoot dose as run (E20-T1). RAW, no takeaway."""
import csv
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = "/nfs/scistore19/locatgrp/bdemirel/ssl_project"
STATIONS = ["L03", "L06", "L09", "gap", "cls", "embed", "tap1", "tap2", "z.out"]
FAMILIES = [
    ("simclr", [("ctrl", "simclr_ctrl", "in100.simclr.s0.e17c.ext", "#8c8c8c"),
                ("e20f", "simclr_e20f", "in100.simclr.s0.e20f.extL", "#3d65d0")]),
    ("byol", [("ctrl", "byol_ctrl", "in100.byol.s0.e17c.ext", "#8c8c8c"),
              ("e20f", "byol_e20f", "in100.byol.s0.e20f.extL", "#6a3fb5")]),
    ("vicreg", [("ctrl", "vicreg_ctrl", "in100.vicreg.s0.e17c.ext", "#8c8c8c"),
                ("e20f", "vicreg_e20f", "in100.vicreg.s0.e20f.extL", "#c23b3b")]),
    ("dino", [("ctrl", "dino_ctrl", "in100.dino.s0.e17c.ext", "#8c8c8c"),
              # overshoot λ=.258 demoted to the light shade; the winner-dose arm takes the
              # family color (Berker 2026-08-26: "e20fwlo is the winner ... add it to our zoo")
              ("e20f (overshoot dose)", "dino_e20f", "in100.dino.s0.e20f.extL", "#e8bd7a"),
              ("e20fwlo (winner λ=.01)", "dino_e20fwlo", "in100.dino.s0.e20fwlo.extL", "#d4820a")]),
    ("lejepa", [("ctrl", "lejepa_ctrl", "in100.lejepa.s0.e17c.ext", "#8c8c8c"),
                ("e20f", "lejepa_e20f", "in100.lejepa.s0.e20f.ep100.extL", "#2e8b57")]),
    # house VISReg pair (E29; Berker 2026-08-26 "do the same for visreg treated + untreated")
    ("visreg", [("ctrl", "visreg_ctrl", "in100.visreg.s0.extL", "#8c8c8c"),
                ("+floor λ=.0123", "visreg_f", "in100.visreg.s0.visregf.extL", "#b3477d")]),
    ("mae", [("ctrl", "mae_ctrl", "in100.mae.s0.extL", "#8c8c8c"),
             ("e20f", "mae_e20f", "in100.mae.s0.e20f.extL", "#8a5cb8")]),
    ("ijepa", [("ctrl", "ijepa_ctrl", "in100.ijepa.s0.extL", "#8c8c8c"),
               ("e20f", "ijepa_e20f", "in100.ijepa.s0.e20f.extL", "#0f7b8a")]),
    # ours (Berker 2026-08-13): zonly = the D-100 h_lamb=0 twin (whole objective behind
    # the MLP) as the untreated row, vm4 = the full h+z objective — the same ±h-moment
    # contrast as the e20f rows, on our own loss. Depth metrics ride
    # results/diag/e20_ours_depth_metrics.csv; W/B rows appended to the e23 retro CSV.
    ("ours", [("z-only (no h term)", "ours_zonly",
               "in100.floorssl.s0.d256vm4zonly.extL", "#8c8c8c"),
              ("h+z (vm4)", "ours_vm4", "in100.floorssl.s0.d256vm4.extL", "#008300")]),
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
for mp in (f"{ROOT}/results/diag/e17_depth_metrics.csv",
           f"{ROOT}/results/diag/e20_zoo_depth_metrics.csv",
           f"{ROOT}/results/diag/e20_ours_depth_metrics.csv",
           f"{ROOT}/results/diag/e20_new_arms_depth_metrics.csv"):
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


QUANTS = [  # (name, getter, ylim: (0,1) | "row"=per-method autoscale | None=column-shared)
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
_PREF = {"dino_ctrl": "teacher", "dino_e20f": "teacher",
         "dino_e20fwlo": "teacher"}   # declared teacher lane (zoo)
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


# Ω/a/b/Λ y-lims per method row (Berker 2026-08-03): unbounded raw-energy ratios read
# within a feature space, not across — one method's magnitude must not set the column scale.
QUANTS += [("Ω (W/B)", wbget("omega"), "row"),
           ("a(s→z.out)", wbget("a"), "row"),
           ("b(s→z.out)", wbget("b"), "row"),
           ("Λ(s→z.out)", wbget("lam"), "row")]


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


def row_lim(g, members):
    vals = [g(mlab, st) for _, mlab, _, _ in members for st in STATIONS
            if g(mlab, st) is not None]
    if not vals:
        return (0, 1)
    lo, hi = min(vals), max(vals)
    pad = 0.06 * ((hi - lo) or 1)
    return (max(0.0, lo - pad), hi + pad)

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
        ax.set_ylim(*(row_lim(get, members) if ylim == "row" else
                      ylim if ylim else (row_gk_lim(members) if ZOO2 else GK_LIM)))
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
fig.suptitle("E20 zoo guillotine — 8 method rows × 12 quantity columns (+Ω · a·b·Λ station→z.out, o8 orbit calculus), floor arm (colored) vs control (grey) vs depth; "
             "y fixed 0–1 for bounded quantities, ranks ÷ station dimension, gauss_kl + Ω·a·b·Λ per-method-row y (unbounded, within-space reading) "
             "(L-taps = trunk cls readouts; dotted = trunk|head; class-cos as the MARGIN same−diff) — RAW",
             fontsize=9.5, y=0.998)
fig.tight_layout(rect=(0, 0.008, 1, 0.98))
out = f"{ROOT}/results/figures/e20/e20_guillotine_zoo{'2' if ZOO2 else ''}.png"
fig.savefig(out, dpi=150)
print("wrote", out)
