"""E21 D-050 dim-bracket figures (Berker 2026-07-20: metric figures "separately comparing the
controls"). (a) e21_metric_panel.png — the battery read as BAR PANELS over the cell roster (Berker
2026-07-21: "too many special symbols just make bar plots") — d16..512 · laug2* · vm2 · e200*
against the two references the dial interpolates:
lejepa ctrl (sigreg, z=16) and vicreg ctrl (var+cov, z=2048), plus the collapsed 2048-d cell
(lejepa_augs2 @ep4 kill state — NOT ep100; contrast only) once its battery lands. Top row = each
run's loss-terminal z (dims differ — dim-comparable stats only: effrank/d, per-dim moment-KL,
top-10 kurt, min/mean std, audit_v1 pos/rand pair cosines — pos and rand SEPARATELY per Berker
2026-07-20: alignment ≡ 2−2·cos_invariance on normalized features, one number rendered twice).
Bottom row = trunk h.cls (384-d, arch-matched; the retained floor-at-h read).
(b) e21_probe_vs_dim.png — the OFFLINE v2 probes of record vs dim (lin = linear_raw_v2 @h.cls;
knn = centered kNN200 from results/diag/e21_centered.csv when present, raw knn_v1_k200
otherwise) with the aug-matched bars read from their own record files (lejepa e20f @z.embed
and @h.cls; f2 @z.embed centered once its surviving store is re-read). Skips any cell whose
files are absent (re-run as landings arrive). RAW rendering, no takeaway."""
import csv
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = "/nfs/scistore19/locatgrp/bdemirel/ssl_project"
DIMS = [16, 32, 64, 128, 256, 512]
FLOOR = [(d, f"in100.floorssl.s0.d{d}.ext") for d in DIMS]
LAUG2 = (2048, "in100.floorssl.s0.lejepa_augs2.ext")  # collapsed @ep4 (contrast, not ep100)
CTRLS = [("lejepa ctrl (sigreg)", 16, "in100.lejepa.s0.ext", "#2e8b57", "s"),
         ("vicreg ctrl (var+cov)", 2048, "in100.vicreg.s0.ext", "#d4820a", "D"),
         ("d256vm2 (view-mean floor)", 256, "in100.floorssl.s0.d256vm2.ext", "#6a3fb5", "^"),
         ("d256e200 (200-ep budget)", 256, "in100.floorssl.s0.d256e200.ext", "#0f7b8a", "v")]
Z, H = "student.z.proj.out", "student.h.cls"
BLUE, RED = "#3d65d0", "#c23b3b"


def load(path, manifest=None):
    """manifest: substring filter — pairs files carry one row-block per aug stack; without the
    filter a keyed dict keeps whichever block is last (ctrl files end on own/foveal stacks).
    Cross-run pair reads are defined on the fixed audit_v1 stack (PROTOCOL §5)."""
    if not os.path.exists(path):
        return None
    d = {}
    for r in csv.DictReader(open(path)):
        if manifest and manifest not in r["manifest"]:
            continue
        d[(r["space"], r["metric"] + "|" + r["variant"])] = (float(r["value"]), int(r["d"]))
    return d


def batt(run):
    b = load(f"{ROOT}/results/battery/{run}.csv")
    p = load(f"{ROOT}/results/battery/{run}.pairs.csv", "@audit_v1")
    return None if b is None else {**b, **(p or {})}


def probes(run):
    path = f"{ROOT}/results/probes/{run}.csv"
    if not os.path.exists(path):
        return None
    return {(r["space"], r["probe"]): float(r["val_acc"]) for r in csv.DictReader(open(path))}


def centered(path):
    if not os.path.exists(path):
        return {}
    return {(r["run"], r["space"]): float(r["knn200_centered"]) for r in csv.DictReader(open(path))}


runs = {run: batt(run) for _, run in FLOOR}
laug2 = batt(LAUG2[1])
ctrls = [(lab, x, batt(run), c, m) for lab, x, run, c, m in CTRLS]
# pos/rand pair cosines (Berker 2026-07-20: alignment ≡ 2−2·cos_invariance — show the two pair
# populations separately instead). run e2x_posneg.py first.
PN_PATH = f"{ROOT}/results/diag/e2x_posneg.csv"
pn = {(r["run"], r["space"]): r for r in csv.DictReader(open(PN_PATH))} if os.path.exists(PN_PATH) else {}


def pncos(run, space, key):
    r = pn.get((run, space))
    return None if r is None else float(r[key])

PANELS = [  # (space, key, title, sub, transform: (value, d) -> plotted)
    (Z, "effective_rank|raw|full", "z: effrank / d", "1 = isotropic, →0 collapsed/aniso", lambda v, d: v / d),
    (Z, "rankme|raw|full", "z: rankme / d", "uncentered SV entropy (cone-sensitive)", lambda v, d: v / d),
    (Z, "gauss_kl_full.total|raw|full", "z: gauss_kl_full.total", "moment-KL to N(0,I) per dim (Σ=I conditioner read)", None),
    (Z, "kurt_topeig.worst|raw|full", "z: kurt_topeig.worst", "max |excess kurt| top-10 eigdirs", None),
    (Z, "variance_floor.min_over_mean_std|raw|full", "z: min/mean per-dim std", "scale-floor health (0 = dead dims)", None),
    (Z, "POSNEG", "z: pair cosines", "pos solid / rand light (audit_v1)", None),
    (H, "effective_rank|raw|full", "h.cls: effective_rank", "of 384 (arch-matched)", None),
    (H, "rankme|raw|full", "h.cls: rankme", "uncentered SV entropy (cone-sensitive)", None),
    (H, "kurt_topeig.worst|raw|full", "h.cls: kurt_topeig.worst", "max |excess kurt| top-10 eigdirs", None),
    (H, "uniformity|raw|full", "h.cls: uniformity", "Wang–Isola (lower = more spread)", None),
    (H, "gauss_kl_full.total|raw|full", "h.cls: gauss_kl_full.total", "moment-KL to N(0,I) per dim", None),
    (H, "POSNEG", "h.cls: pair cosines", "pos solid / rand light (audit_v1)", None),
]


DIMCOLS = ["#7bb3d9", "#3d65d0", "#2e4a9e", "#1a2f6e", "#8a5cb8", "#b8608a"]
ORDER = [("vm2", "d256vm2 (view-mean floor)"), ("e200*", "d256e200 (200-ep budget)"),
         ("lejepa\nctrl", "lejepa ctrl (sigreg)"), ("vicreg\nctrl", "vicreg ctrl (var+cov)")]
cmap = {lab: (run, c) for lab, x, run, c, m in CTRLS}
ALL = [(f"d{d}", run, runs[run], col) for (d, run), col in zip(FLOOR, DIMCOLS)]
ALL += [("laug2*", LAUG2[1], laug2, RED)]
ALL += [(short, cmap[full][0], batt(cmap[full][0]), cmap[full][1]) for short, full in ORDER]


def bar_panel(ax, space, key, title, sub, tf):
    """Bars over the categorical cell roster (Berker 2026-07-21: "too many special symbols
    just make bar plots"); POSNEG = paired bars, pos solid / rand light."""
    if key == "POSNEG":
        for i, (lab, run, b, c) in enumerate(ALL):
            pos, rand = pncos(run, space, "pos_cos"), pncos(run, space, "rand_cos")
            if pos is not None:
                ax.bar(i - 0.19, pos, width=0.36, color=c)
            if rand is not None:
                ax.bar(i + 0.19, rand, width=0.36, color=c, alpha=0.38)
    else:
        for i, (lab, run, b, c) in enumerate(ALL):
            g = None if b is None else b.get((space, key))
            v = None if g is None else (tf(*g) if tf else g[0])
            if v is not None:
                ax.bar(i, v, width=0.62, color=c)
                ax.text(i, v, f"{v:.3g}", ha="center",
                        va="bottom" if v >= 0 else "top", fontsize=5.2)
    ax.set_xticks(range(len(ALL)))
    ax.set_xticklabels([a[0] for a in ALL], fontsize=5.6, rotation=45)
    ax.set_title(f"{title}\n{sub}", fontsize=8)
    ax.axhline(0, color="#666666", lw=0.6)
    ax.margins(y=0.2)
    ax.tick_params(length=0, labelsize=6.5)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)


fig, axes = plt.subplots(2, 6, figsize=(20.5, 6.4), facecolor="white")
for ax, (space, key, title, sub, tf) in zip(axes.ravel(), PANELS):
    bar_panel(ax, space, key, title, sub, tf)
fig.suptitle("E21 battery metrics per cell — floorssl d16–512 · laug2* (collapsed @ep4, contrast) · vm2 (view-mean) · "
             "e200* (200-ep budget) · lejepa/vicreg ctrls — RAW",
             fontsize=10, y=0.995)
fig.tight_layout(rect=(0, 0.03, 1, 0.965))
fig.text(0.01, 0.005, "top row: each run's loss-terminal z (dims differ — dim-comparable stats only) · "
         "bottom row: trunk h.cls, 384-d all runs · pairs under the fixed audit_v1 stack (pos solid / rand light) · "
         "floorssl cells under lejepa V=4; ctrls under their own recipes · e200* carries its declared 200-ep schedule",
         fontsize=6.5, color="#555555")
out = f"{ROOT}/results/figures/e21/e21_metric_panel.png"
fig.savefig(out, dpi=160)
print("wrote", out)

# ---- (b) offline probe-vs-dim ----------------------------------------------------------------
cen21 = centered(f"{ROOT}/results/diag/e21_centered.csv")
cen20 = centered(f"{ROOT}/results/diag/e20_centered.csv")
pf = {run: probes(run) for _, run in FLOOR}
pl2 = probes(LAUG2[1])
pe20f = probes("in100.lejepa.s0.e20f.ext")
pf2 = probes("in100.lejepa.s0.e12f2.ext")

fig, (axl, axk) = plt.subplots(1, 2, figsize=(11.5, 4.1), facecolor="white")

lin = [(d, pf[run][(H, "linear_raw_v2")]) for d, run in FLOOR if pf[run]]
axl.plot(*zip(*lin), "-o", color=BLUE, ms=5, lw=1.4, label="floorssl @h.cls (lin raw v2)")
for d, v in lin:
    axl.annotate(f"{v:.4f}", (d, v), textcoords="offset points", xytext=(0, 6),
                 ha="center", fontsize=6)
if pl2:
    v = pl2[(H, "linear_raw_v2")]
    axl.plot([2048], [v], "x", color=RED, ms=8, mew=2)
    axl.annotate(f"{v:.4f}\n(collapsed @ep4)", (2048, v), textcoords="offset points",
                 xytext=(-7, 8), ha="right", fontsize=6, color=RED)
for src, space, ls, c, lab in [(pe20f, "student.z.embed", "--", "#2e8b57", "lejepa e20f @z.embed (declared)"),
                               (pe20f, H, "-.", "#2e8b57", "lejepa e20f @h.cls"),
                               (pf2, "student.z.embed", ":", "#777777", "f2 @z.embed")]:
    if src:
        v = src[(space, "linear_raw_v2")]
        axl.axhline(v, ls=ls, color=c, lw=1, label=f"{lab} {v:.4f}")
axl.set_title("OFFLINE lin probe (linear_raw_v2) — the number of record", fontsize=9)

knn_raw = [(d, pf[run][(H, "knn_v1_k200")]) for d, run in FLOOR if pf[run]]
axk.plot(*zip(*knn_raw), "--o", color="#9db4e8", ms=4, lw=1, mfc="none", label="floorssl knn200 raw")
knn_c = [(d, cen21[(run, H)]) for d, run in FLOOR if (run, H) in cen21]
if knn_c:
    axk.plot(*zip(*knn_c), "-o", color=BLUE, ms=5, lw=1.4, label="floorssl knn200 CENTERED")
    for d, v in knn_c:
        axk.annotate(f"{v:.4f}", (d, v), textcoords="offset points", xytext=(0, 6),
                     ha="center", fontsize=6)
if pl2:
    axk.plot([2048], [pl2[(H, "knn_v1_k200")]], "x", color=RED, ms=8, mew=2)
f2c = cen21.get(("in100.lejepa.s0.e12f2.ext", "student.z.embed"))
bars_k = [(cen20.get(("in100.lejepa.s0.e20f.ext", "student.z.embed")), "--", "#2e8b57",
           "lejepa e20f @z.embed knn_c"),
          (cen21.get(("in100.lejepa.s0.e20f.ext", H)), "-.", "#2e8b57", "lejepa e20f @h.cls knn_c"),
          (f2c or (pf2 and pf2[("student.z.embed", "knn_v1_k200")]), ":", "#777777",
           "f2 @z.embed knn_c" if f2c else "f2 @z.embed knn RAW")]
for v, ls, c, lab in bars_k:
    if v:
        axk.axhline(v, ls=ls, color=c, lw=1, label=f"{lab} {v:.4f}")
axk.set_title("OFFLINE kNN200 — centered where the store allows (bars' convention)", fontsize=9)

for ax in (axl, axk):
    ax.set_xscale("log", base=2)
    ax.set_xticks(DIMS + [2048])
    ax.set_xticklabels([str(d) for d in DIMS] + ["2048"], fontsize=7)
    ax.set_xlabel("out-dim of the run's z", fontsize=8)
    ax.legend(fontsize=6.5, loc="lower right", frameon=False)
    ax.tick_params(length=0, labelsize=7)
    ax.margins(y=0.15)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
fig.suptitle("E21 (D-050): offline v2 probe vs out-dim, ep100 — vs the aug-matched lejepa e20f / f2 bars (RAW)",
             fontsize=10)
fig.tight_layout(rect=(0, 0, 1, 0.94))
out = f"{ROOT}/results/figures/e21/e21_probe_vs_dim.png"
fig.savefig(out, dpi=160)
print("wrote", out)
