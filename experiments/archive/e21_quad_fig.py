"""E21 comparison figure (Berker 2026-07-20, three iterations same day): the D-050 bracket's
two offline peaks (d64 lin / d256 knn) vs the lejepa-lane floor arm (e20f) + lejepa control
+ THE DINO ARMS (his regenerate ask: dose-winner lam008 + dino control; the full dose family
lives on e20_dino_dose.png). Probe bars at each run's DECLARED h (trunk CLS; teacher for
dino) — the ◆ embed overlays were removed per his parse feedback; lejepa declared-embed
numbers live on e21_probe_vs_dim.png. Pair cosines as M1-grammar dumbbells (rand ○ → pos ●,
audit_v1). Two metrics added on his ask: (a) class-cos dumbbells (diff-class ○ → same-class
●, e2x_classcos.csv — label-conditioned structure at declared h); (b) paired rankme/d
(h solid · z light — the uncentered twin of the effrank panels; replaced the z-pred R²
panel cancelled by D-060). RAW, no takeaway; convergence caveat on-figure (E20-T2)."""
import csv
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = "/nfs/scistore19/locatgrp/bdemirel/ssl_project"
#        label       run                                color      declared-h        loss-terminal z
RUNS = [("d64", "in100.floorssl.s0.d64.ext", "#3d65d0", "student.h.cls", "student.z.proj.out"),
        ("d256", "in100.floorssl.s0.d256.ext", "#1b2c60", "student.h.cls", "student.z.proj.out"),
        ("d256vm2", "in100.floorssl.s0.d256vm2.ext", "#6a3fb5", "student.h.cls", "student.z.proj.out"),
        ("d256e200", "in100.floorssl.s0.d256e200.ext", "#0f7b8a", "student.h.cls", "student.z.proj.out"),
        ("e20f", "in100.lejepa.s0.e20f.ext", "#2e8b57", "student.h.cls", "student.z.proj.out"),
        ("ctrl", "in100.lejepa.s0.ext", "#8c8c8c", "student.h.cls", "student.z.proj.out"),
        ("dino λ.008", "in100.dino.s0.e20f_lam008.ext", "#d4820a", "teacher.h.cls", "student.z.dino.bottleneck"),
        ("dino ctrl", "in100.dino.s0.ext", "#b39b77", "teacher.h.cls", "student.z.dino.bottleneck")]


def probes(run):
    path = f"{ROOT}/results/probes/{run}.csv"
    return {(r["space"], r["probe"]): float(r["val_acc"]) for r in csv.DictReader(open(path))} \
        if os.path.exists(path) else {}


def batt(run):
    path = f"{ROOT}/results/battery/{run}.csv"
    return {(r["space"], r["metric"] + "|" + r["variant"]): (float(r["value"]), int(r["d"]))
            for r in csv.DictReader(open(path))} if os.path.exists(path) else {}


def csvmap(path, keys):
    return {tuple(r[k] for k in keys): r for r in csv.DictReader(open(path))} \
        if os.path.exists(path) else {}


P = {lab: probes(run) for lab, run, *_ in RUNS}
B = {lab: batt(run) for lab, run, *_ in RUNS}
CEN = csvmap(f"{ROOT}/results/diag/e21_centered.csv", ("run", "space"))
PN = csvmap(f"{ROOT}/results/diag/e2x_posneg.csv", ("run", "space"))
CC = csvmap(f"{ROOT}/results/diag/e2x_classcos.csv", ("run",))
ZP = csvmap(f"{ROOT}/results/diag/e2x_zpred.csv", ("run",))


def bar_panel(ax, title, sub, vals, fmt="{:.3g}"):
    for i, (lab, v, c) in enumerate(vals):
        if v is None:
            continue
        ax.bar(i, v, width=0.62, color=c)
        ax.text(i, v, fmt.format(v), ha="center", va="bottom" if v >= 0 else "top",
                fontsize=6)
    ax.set_xticks(range(len(vals)))
    ax.set_xticklabels([l for l, _, _ in vals], fontsize=5.8, rotation=30)
    ax.set_title(f"{title}\n{sub}", fontsize=7.6)
    ax.margins(y=0.26)
    ax.tick_params(length=0, labelsize=6.5)
    ax.axhline(0, color="#666666", lw=0.6)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)


def dumbbell_panel(ax, title, sub, rows):
    """rows: [(label, filled_value, open_value, color)] — open ○ -> filled ● (M1 grammar)."""
    ys = range(len(rows), 0, -1)
    for y, (lab, hi, lo, c) in zip(ys, rows):
        if hi is None or lo is None:
            continue
        ax.plot([lo, hi], [y, y], color=c, lw=2, solid_capstyle="round")
        ax.plot(lo, y, "o", ms=5.5, mfc="white", mec=c, mew=1.3)
        ax.plot(hi, y, "o", ms=6.5, color=c)
        ax.annotate(f"{hi:.3f}", (hi, y), xytext=(5, 3), textcoords="offset points", fontsize=5.6)
        ax.annotate(f"{lo:.3f}", (lo, y), xytext=(-5, 3), textcoords="offset points",
                    fontsize=5.6, ha="right")
    ax.set_yticks(list(ys))
    ax.set_yticklabels([r[0] for r in rows], fontsize=6)
    ax.set_title(f"{title}\n{sub}", fontsize=7.6)
    ax.margins(x=0.24, y=0.2)
    ax.tick_params(length=0, labelsize=6.5)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)


def pn_pair(run, space):
    r = PN.get((run, space))
    return (None, None) if r is None else (float(r["pos_cos"]), float(r["rand_cos"]))


def bval(lab, space, key, tf=None):
    g = B[lab].get((space, key))
    return None if g is None else (tf(*g) if tf else g[0])


fig, axes = plt.subplots(2, 6, figsize=(19, 6.6), facecolor="white")
ax = axes.ravel()

bar_panel(ax[0], "offline lin (raw v2) @trunk h", "CLS; teacher branch for dino",
          [(lab, P[lab].get((h, "linear_raw_v2")), c) for lab, run, c, h, z in RUNS],
          fmt="{:.4f}")
bar_panel(ax[1], "offline kNN200 CENTERED @trunk h", "same instrument all runs",
          [(lab, (lambda r: float(r["knn200_centered"]) if r else None)(CEN.get((run, h))), c)
           for lab, run, c, h, z in RUNS], fmt="{:.4f}")
dumbbell_panel(ax[2], "pair cos @trunk h", "rand ○ → pos ● (audit_v1)",
               [(lab, *pn_pair(run, h), c) for lab, run, c, h, z in RUNS])
dumbbell_panel(ax[3], "class cos @trunk h", "diff-class ○ → same-class ● (train, exact)",
               [(lab, (lambda r: float(r["cos_same"]) if r else None)(CC.get((run,))),
                 (lambda r: float(r["cos_diff"]) if r else None)(CC.get((run,))), c)
                for lab, run, c, h, z in RUNS])
bar_panel(ax[4], "h: effective_rank", "of 384 (declared h)",
          [(lab, bval(lab, h, "effective_rank|raw|full"), c) for lab, run, c, h, z in RUNS])
bar_panel(ax[5], "h: kurt_topeig.worst", "max |excess kurt| top-10 eigdirs",
          [(lab, bval(lab, h, "kurt_topeig.worst|raw|full"), c) for lab, run, c, h, z in RUNS])
bar_panel(ax[6], "h: uniformity", "Wang–Isola (lower = more spread)",
          [(lab, bval(lab, h, "uniformity|raw|full"), c) for lab, run, c, h, z in RUNS])
bar_panel(ax[7], "h: gauss_kl_full.total", "moment-KL to N(0,I) per dim",
          [(lab, bval(lab, h, "gauss_kl_full.total|raw|full"), c) for lab, run, c, h, z in RUNS])
bar_panel(ax[8], "z: effrank / d", "loss-terminal z (d = 64/256/16/16/256/256)",
          [(lab, bval(lab, z, "effective_rank|raw|full", lambda v, d: v / d), c)
           for lab, run, c, h, z in RUNS])
bar_panel(ax[9], "z: gauss_kl_full.total", "moment-KL to N(0,I) per dim",
          [(lab, bval(lab, z, "gauss_kl_full.total|raw|full"), c) for lab, run, c, h, z in RUNS])
dumbbell_panel(ax[10], "pair cos @loss-terminal z", "rand ○ → pos ● (audit_v1)",
               [(lab, *pn_pair(run, z), c) for lab, run, c, h, z in RUNS])
# freed by the cancelled z-pred panel (D-060) -> rankme (Berker 2026-07-21: "did you add
# rankme as well? it would be good to have"): normalized rankme/d, h solid vs z light —
# the uncentered/cone-sensitive twin of the effrank panels (they dissociate; METRICS.md).
for i, (lab, run, c, h, z) in enumerate(RUNS):
    vh = bval(lab, h, "rankme|raw|full", lambda v, d: v / d)
    vz = bval(lab, z, "rankme|raw|full", lambda v, d: v / d)
    if vh is not None:
        ax[11].bar(i - 0.19, vh, width=0.36, color=c)
    if vz is not None:
        ax[11].bar(i + 0.19, vz, width=0.36, color=c, alpha=0.38)
ax[11].set_xticks(range(len(RUNS)))
ax[11].set_xticklabels([lab for lab, *_ in RUNS], fontsize=5.8, rotation=30)
ax[11].set_title("rankme / d — h solid · z light\nuncentered SV entropy (cone-sensitive)",
                 fontsize=7.6)
ax[11].margins(y=0.26)
ax[11].tick_params(length=0, labelsize=6.5)
for s in ("top", "right"):
    ax[11].spines[s].set_visible(False)

fig.suptitle("E21 comparison: floorssl d64 · d256 · d256vm2 · d256e200(ep200) vs lejepa e20f · ctrl vs dino λ.008 · ctrl — offline probes + battery + head/class reads (RAW)",
             fontsize=10, y=0.995)
fig.tight_layout(rect=(0, 0.05, 1, 0.965))
fig.text(0.01, 0.005,
         "lejepa-lane four are aug-matched (lejepa V=4 family); dino pair trains under its own multi-crop recipe — cross-lane rows are cross-recipe · 100-ep endpoint reads, "
         "monitors still climbing at cut (E20-T2 caveat) · declared h: student.h.cls / teacher.h.cls (dino); lejepa declared-embed probe numbers on e21_probe_vs_dim.png · "
         "z row: floorssl z.proj.out (owned conditioner space) / lejepa 16-d sigreg out / dino 256-d student bottleneck · pairs audit_v1; rand = cross-view "
         "different-image · class-cos exact over all train pairs, label-conditioned",
         fontsize=5.8, color="#555555")
out = f"{ROOT}/results/figures/e21/e21_quad.png"
fig.savefig(out, dpi=160)
print("wrote", out)
