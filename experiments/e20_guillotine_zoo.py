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
              ("e20f (overshoot dose)", "dino_e20f", "in100.dino.s0.e20f.extL", "#d4820a")]),
    ("lejepa", [("ctrl", "lejepa_ctrl", "in100.lejepa.s0.e17c.ext", "#8c8c8c"),
                ("e20f", "lejepa_e20f", "in100.lejepa.s0.e20f.ep100.extL", "#2e8b57")]),
    ("mae", [("ctrl", "mae_ctrl", "in100.mae.s0.extL", "#8c8c8c"),
             ("e20f", "mae_e20f", "in100.mae.s0.e20f.extL", "#8a5cb8")]),
    ("ijepa", [("ctrl", "ijepa_ctrl", "in100.ijepa.s0.extL", "#8c8c8c"),
               ("e20f", "ijepa_e20f", "in100.ijepa.s0.e20f.extL", "#0f7b8a")]),
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
           f"{ROOT}/results/diag/e20_zoo_depth_metrics.csv"):
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

# column-shared range for unbounded quantities
gk = mget("gauss_kl_total")
gk_vals = [gk(mlab, st) for _, mem in FAMILIES for _, mlab, _, _ in mem
           for st in STATIONS if gk(mlab, st) is not None]
GK_LIM = (0, max(gk_vals) * 1.08) if gk_vals else None

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
        ax.set_ylim(*(ylim if ylim else GK_LIM))
        ax.set_xticks(range(len(STATIONS)))
        ax.set_xticklabels(STATIONS if row == len(FAMILIES) - 1 else [], fontsize=5.2,
                           rotation=45)
        ax.axvline(4.5, color="#bbbbbb", lw=0.7, ls=":")
        if row == 0:
            ax.set_title(qname, fontsize=8)
        if col == 0:
            ax.set_ylabel(fam, fontsize=8.5)
            ax.legend(fontsize=4.6, frameon=False, loc="upper left")
        ax.tick_params(length=0, labelsize=5.6)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
fig.suptitle("E20 zoo guillotine — 7 method rows × 8 quantity columns, e20f floor arm (colored) vs control (grey) vs depth; "
             "y fixed 0–1 for bounded quantities, ranks ÷ station dimension, gauss_kl column-shared "
             "(L-taps = trunk cls readouts; dotted = trunk|head; class-cos as the MARGIN same−diff) — RAW",
             fontsize=9.5, y=0.998)
fig.tight_layout(rect=(0, 0.008, 1, 0.98))
out = f"{ROOT}/results/figures/e20/e20_guillotine_zoo.png"
fig.savefig(out, dpi=150)
print("wrote", out)
