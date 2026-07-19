#!/usr/bin/env python
"""Regenerate the data-driven figures for docs/slides/sslgap_journey.html.

House dataviz: Okabe-Ito (CVD-validated, worst-adjacent deuteranopia dE 17.9),
direct labels, thin marks, recessive axes, white surface (deck is light).
Agreed numbers are transcribed from the AGREED-TAKEAWAY / card tables (the
values Berker signed off on); training curves come from the wandb cache under
CACHE. Every figure names its source on the slide caption, not here.
"""
import os
import numpy as np
import pandas as pd
import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

CACHE = "/tmp/claude-1104964/-nfs-scistore19-locatgrp-bdemirel-ssl-project/2c3b9172-384d-429f-ae40-f31c2d46ad0a/scratchpad/wandb_cache"
OUT = os.path.join(os.path.dirname(__file__), "assets")
os.makedirs(OUT, exist_ok=True)

# ---- Okabe-Ito ------------------------------------------------------------
OI = dict(blue="#0072B2", orange="#E69F00", green="#009E73", purple="#CC79A7",
          sky="#56B4E9", vermillion="#D55E00", darkred="#B22222", black="#1a1a1a")
INK, MUTE, FAINT, GRID = "#1a1a1a", "#5c5651", "#8a8279", "#e7e3dd"
GOOD, BAD, NEUTRAL = OI["green"], OI["vermillion"], "#9a938a"
METHOD = dict(dino=OI["blue"], simclr=OI["sky"], byol=OI["green"], vicreg=OI["orange"],
              ijepa=OI["purple"], lejepa=OI["vermillion"], mae="#3a3a3a",
              deitlite=OI["darkred"], randinit="#b9b2a8")

mpl.rcParams.update({
    "figure.dpi": 150, "savefig.dpi": 150, "savefig.bbox": "tight",
    "font.family": "DejaVu Sans", "font.size": 12.5,
    "axes.edgecolor": FAINT, "axes.linewidth": 1.0, "axes.labelcolor": INK,
    "axes.titlecolor": INK, "text.color": INK, "xtick.color": MUTE, "ytick.color": MUTE,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.9,
    "figure.facecolor": "white", "axes.facecolor": "white", "savefig.facecolor": "white",
})


def _save(fig, name):
    p = os.path.join(OUT, name)
    fig.savefig(p)
    plt.close(fig)
    print("wrote", name)


def _clean(ax, ygrid=True):
    ax.set_axisbelow(True)
    ax.grid(axis="y" if ygrid else "x")
    ax.grid(axis="x" if ygrid else "y", visible=False)
    ax.tick_params(length=0)


# ============================================================ ACT 3 — E12 ===
ARMS = [  # tag, dose-label, lin, knn200, knn20, B/T, NMI, effrank
    ("f2", "calibration λ.02", .6578, .6056, .6156, .157, .533, 217.2),
    ("C1", "control (none)", .6456, .5306, .5652, .482, .527, 22.0),
    ("f1", "calibration λ.10", .6338, .5718, .5736, .096, .440, 252.3),
    ("f5", "diagonal only λ.55", .6322, .5074, .5396, .357, .504, 6.8),
    ("f3", "rank floor λ.48", .6312, .4928, .5254, .441, .480, 9.7),
    ("f4", "shape only λ.07", .6136, .2724, .3722, .340, .161, 2.2),
    ("lane", "baseline", .6022, .5240, .5530, .512, .521, 17.9),
    ("A3", "calibration λ.48", .5070, .3250, .2500, .043, .053, 274.8),
    ("A1", "both terms at h", .4732, .3754, .4006, .421, .427, 22.1),
    ("A2", "shape λ.03", .4310, .3364, .3254, .119, .305, 76.3),
]
NULL_NMI, NULL_LIN = .091, .30  # randinit reference (lin ~chance-ish placeholder line only for NMI axis)


def fig_e12_probe_bars():
    arms = sorted(ARMS, key=lambda r: r[2])
    tags = [a[0] for a in arms]
    lin = [a[2] for a in arms]
    knn = [a[3] for a in arms]
    y = np.arange(len(arms))
    fig, ax = plt.subplots(figsize=(9.2, 5.3))
    h = 0.38
    for yi, a in zip(y, arms):
        c = GOOD if a[0] == "f2" else (BAD if a[0] in ("A3", "A2", "A1") else NEUTRAL)
        ax.barh(yi + h/2, a[2], height=h, color=c, zorder=3)
        ax.barh(yi - h/2, a[3], height=h, color=c, alpha=0.5, zorder=3)
        ax.text(a[2] + .006, yi + h/2, f"{a[2]:.3f}", va="center", ha="left", fontsize=9.5, color=INK)
        ax.text(a[3] + .006, yi - h/2, f"{a[3]:.3f}", va="center", ha="left", fontsize=9.5, color=MUTE)
    ax.axvline(.6022, color=MUTE, lw=1.1, ls=(0, (4, 3)), zorder=2)
    ax.text(.6022, len(arms)-.35, " baseline", color=MUTE, fontsize=9, ha="left", va="top")
    ax.set_yticks(y)
    ax.set_yticklabels([f"{a[1]}" for a in arms], fontsize=10.5)
    ax.set_xlim(0, .74)
    ax.set_xlabel("probe accuracy at the backbone  (solid = linear · faded = kNN)")
    ax.set_title("A calibration term at the backbone: dose from tonic to toxic", fontsize=13, loc="left", pad=10)
    ax.text(.01, 1.015, "", transform=ax.transAxes)
    # legend chips
    ax.scatter([], [], marker="s", s=90, color=GOOD, label="best (low dose)")
    ax.scatter([], [], marker="s", s=90, color=NEUTRAL, label="control / mild")
    ax.scatter([], [], marker="s", s=90, color=BAD, label="high dose / erased")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.14), ncol=3, frameon=False,
              fontsize=9.5, handletextpad=.4, columnspacing=1.4)
    _clean(ax, ygrid=False)
    ax.grid(axis="x")
    _save(fig, "e12_probe_bars.png")


def fig_e12_nmi_scatter():
    fig, ax = plt.subplots(figsize=(7.6, 5.5))
    for a in ARMS:
        tag, _, lin, knn, knn20, bt, nmi, eff = a
        c = GOOD if tag == "f2" else (BAD if tag in ("A3", "A2", "A1", "f4") else NEUTRAL)
        big = tag in ("f2", "A3", "C1", "f4")
        ax.scatter(nmi, knn, s=200 if big else 110, color=c, zorder=4,
                   edgecolor="white", linewidth=1.4)
        # per-label offsets to declutter the C1/lane/f5/f3 cluster
        off = {"f2": (.012, .010, "left"), "C1": (-.014, .012, "right"), "lane": (.013, -.004, "left"),
               "f5": (.013, .012, "left"), "f3": (-.013, -.016, "right"), "f1": (-.010, .014, "right"),
               "A1": (.013, .002, "left"), "A2": (.013, -.004, "left"), "A3": (.013, .006, "left"),
               "f4": (.006, -.028, "left")}
        dx, dy, ha = off.get(tag, (.012, .012, "left"))
        ax.annotate(tag, (nmi, knn), (nmi + dx, knn + dy),
                    fontsize=10.5, ha=ha, color=INK, fontweight="bold" if big else "normal")
    ax.axvline(NULL_NMI, color=NEUTRAL, ls=(0, (2, 3)), lw=1.2)
    ax.text(NULL_NMI + .004, .60, "untrained\nbaseline", color=MUTE, fontsize=8.6, va="top")
    ax.set_xlabel("class alignment  (geometry follows labels →)")
    ax.set_ylabel("kNN probe accuracy  ↑")
    ax.set_title("When calibration scrubs class structure, neighborhoods die too",
                 fontsize=12.5, loc="left", pad=10)
    ax.annotate("full strength:\ncalibrated but scrubbed",
                (.053, .325), (.19, .34), fontsize=9, color=BAD,
                arrowprops=dict(arrowstyle="->", color=BAD, lw=1.3))
    ax.annotate("low strength:\nhelps, structure kept",
                (.533, .606), (.28, .55), fontsize=9, color=GOOD,
                arrowprops=dict(arrowstyle="->", color=GOOD, lw=1.3))
    _clean(ax)
    _save(fig, "e12_nmi_scatter.png")


def fig_e12_dose_curve():
    lam = [0, .02, .10, .478]
    lin = [.6456, .6578, .6338, .5070]
    floor = [1.65, .123, .087, .070]
    xlab = ["0\ncontrol", ".02", ".10", ".478"]
    x = np.arange(len(lam))
    fig, axes = plt.subplots(1, 2, figsize=(10.4, 4.5))
    ax = axes[0]
    ax.plot(x, lin, "-o", color=OI["blue"], lw=2.4, ms=9, zorder=4)
    for xi, v in zip(x, lin):
        ax.text(xi, v + .006, f"{v:.3f}", ha="center", fontsize=9.6, color=INK)
    ax.axhline(.6022, color=MUTE, ls=(0, (4, 3)), lw=1.1)
    ax.text(3.02, .6022, "baseline", color=MUTE, fontsize=9, va="center")
    ax.scatter([1], [.6578], s=260, facecolor="none", edgecolor=GOOD, linewidth=2.4, zorder=5)
    ax.set_xticks(x); ax.set_xticklabels(xlab, fontsize=10)
    ax.set_ylim(.48, .68)
    ax.set_ylabel("linear probe accuracy")
    ax.set_title("probe accuracy: interior optimum", fontsize=11.5, loc="left")
    _clean(ax)
    ax = axes[1]
    ax.plot(x, floor, "-o", color=OI["vermillion"], lw=2.4, ms=9, zorder=4)
    for xi, v in zip(x, floor):
        ax.text(xi, v + .06, f"{v:.2f}", ha="center", fontsize=9.6, color=INK)
    ax.set_xticks(x); ax.set_xticklabels(xlab, fontsize=10)
    ax.set_ylabel("distance from a unit Gaussian  (↓ = calibrated)")
    ax.set_title("constraint satisfaction: saturates by λ.02", fontsize=11.5, loc="left")
    ax.annotate("93% of achievable\nreduction already at λ.02",
                (1, .123), (1.35, .9), fontsize=9, color=MUTE,
                arrowprops=dict(arrowstyle="->", color=MUTE, lw=1.2))
    _clean(ax)
    fig.suptitle("A conditioner with an interior optimum, not a destination",
                 fontsize=13, x=.01, ha="left", y=1.02)
    _save(fig, "e12_dose_curve.png")


def fig_e12_floor_training():
    fig, ax = plt.subplots(figsize=(8.4, 4.6))
    series = [("floor_e12a3.csv", "A3 · λ.478", OI["vermillion"], "-"),
              ("floor_e12f1.csv", "f1 · λ.10", OI["orange"], "-"),
              ("floor_e12f2.csv", "f2 · λ.02", OI["green"], "-")]
    for fn, lab, c, ls in series:
        d = pd.read_csv(os.path.join(CACHE, fn))
        d = d[d.epoch > 0.5]
        ax.plot(d.epoch, d.h_moment_kl, ls, color=c, lw=2.2, label=lab)
        yv = d.h_moment_kl.iloc[-1]
        ax.text(101, yv, lab.split(" · ")[0], color=c, fontsize=10, va="center", fontweight="bold")
    ax.set_yscale("log")
    ax.set_xlim(0, 112)
    ax.set_xlabel("epoch")
    ax.set_ylabel("training moment-KL at h  (log, ↓)")
    ax.set_title("Heavier dose drives calibration deeper — but accuracy peaks at the lightest",
                 fontsize=11.8, loc="left", pad=10)
    _clean(ax)
    _save(fig, "e12_floor_training.png")


def _acc(fn):
    d = pd.read_csv(os.path.join(CACHE, fn)).sort_values("epoch")
    return d.epoch.values, d.acc.values


def fig_acc_lejepa():
    fig, ax = plt.subplots(figsize=(8.2, 4.6))
    for fn, lab, c, ls in [("acc_lejepa.csv", "LeJEPA (baseline)", OI["vermillion"], (0, (5, 2))),
                           ("acc_e12c1.csv", "control", NEUTRAL, "-"),
                           ("acc_e12f2.csv", "+ calibration λ.02", OI["green"], "-")]:
        e, a = _acc(fn)
        ax.plot(e, a, color=c, lw=2.4, linestyle=ls, label=lab)
        ax.text(e[-1] + .8, a[-1], lab.split(" ·")[0].replace(" lane", "").replace(" control", ""),
                color=c, fontsize=9.5, va="center", fontweight="bold")
    ax.set_xlabel("epoch"); ax.set_ylabel("online probe accuracy")
    ax.set_xlim(0, 112)
    ax.set_title("A whisper of the term is a net-positive conditioner (online monitor)",
                 fontsize=12, loc="left", pad=10)
    ax.legend(loc="lower right", frameon=False, fontsize=9.8)
    _clean(ax)
    _save(fig, "acc_lejepa.png")


def fig_acc_gwave():
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), sharey=False)
    groups = [("VICReg", [("acc_vicreg.csv", "baseline", NEUTRAL, (0, (5, 2))),
                                        ("acc_e12gvc.csv", "control", "#7d766c", "-"),
                                        ("acc_e12gv.csv", "+ calibration λ.02", OI["orange"], "-")]),
              ("DINO", [("acc_dino.csv", "baseline", NEUTRAL, (0, (5, 2))),
                                        ("acc_e12gdc.csv", "control", "#7d766c", "-"),
                                        ("acc_e12gd.csv", "+ calibration λ.02", OI["blue"], "-")])]
    for ax, (title, series) in zip(axes, groups):
        for fn, lab, c, ls in series:
            e, a = _acc(fn)
            ax.plot(e, a, color=c, lw=2.3, linestyle=ls, label=lab)
        ax.set_title(title, fontsize=11.5, loc="left")
        ax.set_xlabel("epoch"); ax.set_xlim(0, 105)
        ax.legend(loc="lower right", frameon=False, fontsize=9)
        _clean(ax)
    axes[0].set_ylabel("online probe accuracy")
    fig.suptitle("Is the conditioner method-general?  (monitor-only — no conclusion yet)",
                 fontsize=12, x=.01, ha="left", y=1.02)
    _save(fig, "acc_gwave.png")


# ============================================================ ACT 4 — PIVOT =
def _rho_bars(ax, labels, lins, knns, colors, title, note=None, ymax=.85):
    x = np.arange(len(labels))
    w = 0.38
    for xi, (lin, knn, c) in enumerate(zip(lins, knns, colors)):
        ax.bar(xi - w/2, abs(lin), w, color=c, zorder=3)
        ax.bar(xi + w/2, abs(knn), w, color=c, alpha=0.5, zorder=3)
        ax.text(xi - w/2, abs(lin) + .012, f"{lin:+.2f}".replace("+", "−" if lin < 0 else "+").replace("−−", "−"),
                ha="center", fontsize=9, color=INK)
        ax.text(xi + w/2, abs(knn) + .012, f"{knn:+.2f}".replace("+", "−" if knn < 0 else "+").replace("−−", "−"),
                ha="center", fontsize=9, color=MUTE)
    ax.axhline(.44, color=OI["vermillion"], ls=(0, (4, 3)), lw=1.2, zorder=2)
    ax.text((len(labels)-1)/2.0, .452, "significance bar  |ρ|=.44", color=OI["vermillion"],
            fontsize=8.6, ha="center", va="bottom")
    ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=10)
    ax.set_ylim(0, ymax)
    ax.set_ylabel("|Spearman ρ|  vs probe rank\n(solid = linear · faded = kNN)")
    ax.set_title(title, fontsize=12, loc="left", pad=10)
    if note:
        ax.text(.0, -.22, note, transform=ax.transAxes, fontsize=9, color=MUTE)
    _clean(ax)


def fig_e13_ranker_bars():
    fig, ax = plt.subplots(figsize=(7.6, 5.3))
    labels = ["predictive score\n(D_read)", "best marginal statistic\n(worst-direction kurtosis)"]
    lins = [-.716, -.546]
    knns = [-.653, -.426]
    colors = [OI["blue"], NEUTRAL]
    _rho_bars(ax, labels, lins, knns, colors,
              "The predictive score ranks the models better than the best statistic we measure",
              note="ranking agreement with true probe accuracy · higher bar = stronger ranker", ymax=.85)
    ax.annotate("predictive\ninformation wins", (0, .716), (0.42, .80),
                fontsize=10.5, color=OI["blue"], ha="center",
                arrowprops=dict(arrowstyle="->", color=OI["blue"], lw=1.4))
    _save(fig, "e13_ranker_bars.png")


def fig_e13_family_decomp():
    fig, ax = plt.subplots(figsize=(8.2, 5.0))
    groups = ["within one family", "across families"]
    x = np.arange(2); w = .36
    dread = [.700, .483]
    kurt = [.282, .717]
    b1 = ax.bar(x - w/2, dread, w, color=OI["blue"], zorder=3, label="D_read (predictive)")
    b2 = ax.bar(x + w/2, kurt, w, color=NEUTRAL, zorder=3, label="best statistic")
    for xi, v in zip(x - w/2, dread):
        ax.text(xi, v + .015, f"−.{int(round(v*1000)):03d}"[:4].replace("−.70", "−.70"), ha="center", fontsize=10, color=INK)
    for xi, v in zip(x + w/2, kurt):
        ax.text(xi, v + .015, f"−.{int(round(v*1000)):03d}"[:4], ha="center", fontsize=10, color=MUTE)
    # overwrite with clean labels
    for xi, v, col in [(x[0]-w/2, .700, INK), (x[1]-w/2, .483, INK)]:
        pass
    ax.set_xticks(x); ax.set_xticklabels(groups, fontsize=10.5)
    ax.set_ylim(0, .85)
    ax.set_ylabel("|Spearman ρ| vs linear probe rank")
    ax.set_title("The two best rankers cover opposite halves of the zoo",
                 fontsize=11.8, loc="left", pad=10)
    ax.legend(loc="upper left", frameon=False, fontsize=9.6)
    ax.annotate("the statistic is blind\nwithin one method family",
                (x[0]+w/2, .282), (x[0]+.05, .52), fontsize=8.8, color=MUTE,
                arrowprops=dict(arrowstyle="->", color=MUTE, lw=1.1))
    _clean(ax)
    _save(fig, "e13_family_decomp.png")


def fig_e14_strata():
    fig, ax = plt.subplots(figsize=(9.0, 5.1))
    labels = ["same spot", "nearby", "disjoint", "disjoint\nrandom target", "disjoint\nsemantic target"]
    lins = [-.526, -.404, -.253, .154, -.794]
    knns = [-.546, -.335, -.193, .109, -.664]
    colors = [OI["blue"], OI["blue"], OI["blue"], "#c9c2b8", OI["green"]]
    x = np.arange(len(labels)); w = .38
    for xi, (lin, knn, c) in enumerate(zip(lins, knns, colors)):
        ax.bar(xi - w/2, lin, w, color=c, zorder=3)
        ax.bar(xi + w/2, knn, w, color=c, alpha=.5, zorder=3)
        ax.text(xi - w/2, lin + (.02 if lin > 0 else -.055), f"{lin:+.2f}".replace("+", "−" if lin < 0 else "+").replace("−−", "−"),
                ha="center", fontsize=9, color=INK)
    ax.axhline(0, color=FAINT, lw=1)
    ax.axhline(-.44, color=OI["vermillion"], ls=(0, (4, 3)), lw=1.1)
    ax.text(4.4, -.46, "claim bar", color=OI["vermillion"], fontsize=8.4, ha="right", va="top")
    ax.axhline(-.546, color=NEUTRAL, ls=(0, (1, 2)), lw=1)
    ax.text(0.0, -.566, "best statistic", color=MUTE, fontsize=8.2, ha="left", va="top")
    ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=9.8)
    ax.set_ylim(-.92, .30)
    ax.set_ylabel("Spearman ρ   (↓ = stronger ranker)")
    ax.set_title("Overlap-controlled: the reconstruction target ranking dies at zero overlap",
                 fontsize=11.2, loc="left", pad=10)
    ax.text(.02, .97, "solid = linear · faded = kNN-200", transform=ax.transAxes,
            fontsize=8.4, color=MUTE, va="top")
    ax.annotate("ranking dies\nat zero overlap", (2, -.253), (1.1, .10),
                fontsize=9, color=OI["blue"], ha="center",
                arrowprops=dict(arrowstyle="->", color=OI["blue"], lw=1.2))
    ax.annotate("a semantic target still\nranks (−.79) — but cannot\nbeat the teacher it copies",
                (4, -.794), (1.9, -.70), fontsize=9, color=OI["green"], ha="center",
                arrowprops=dict(arrowstyle="->", color=OI["green"], lw=1.3))
    _clean(ax)
    _save(fig, "e14_strata.png")


def fig_e14_blur():
    fig, ax = plt.subplots(figsize=(7.8, 4.6))
    labels = ["semantic target\n(across families)", "recon. target\n(across families)", "recon. target\n(one family)"]
    foveal = [-.794, -.253, -.654]
    blur = [-.707, -.270, -.654]
    x = np.arange(len(labels)); w = .36
    ax.bar(x - w/2, np.abs(foveal), w, color=OI["purple"], zorder=3, label="with sharp window")
    ax.bar(x + w/2, np.abs(blur), w, color="#d9c7e0", zorder=3, label="blurred (no window)")
    for xi, v in zip(x - w/2, foveal):
        ax.text(xi, abs(v) + .012, f"{v:+.2f}".replace("+", "−"), ha="center", fontsize=9, color=INK)
    for xi, v in zip(x + w/2, blur):
        ax.text(xi, abs(v) + .012, f"{v:+.2f}".replace("+", "−"), ha="center", fontsize=9, color=MUTE)
    ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=9.5)
    ax.set_ylim(0, .92)
    ax.set_ylabel("|Spearman ρ| vs linear probe")
    ax.set_title("The honest null: removing the sharp window changes almost nothing",
                 fontsize=11.4, loc="left", pad=10)
    ax.legend(loc="upper right", frameon=False, fontsize=9)
    ax.text(.0, -.20, "the ranking rides the blurry scene, not the sharp window  (frozen models)",
            transform=ax.transAxes, fontsize=8.8, color=MUTE)
    _clean(ax)
    _save(fig, "e14_blur.png")


# ============================================================ ACT 5 — E15/16 =
def fig_pivot_tiers():
    fig, ax = plt.subplots(figsize=(8.6, 4.4))
    labels = ["v1 ceiling", "MAE raw h", "target ceiling", "VICReg h", "LeJEPA h", "DINO h"]
    vals = [.378, .434, .442, .590, .602, .686]
    cols = ["#b9b2a8", METHOD["mae"], "#8a8279", METHOD["vicreg"], METHOD["lejepa"], METHOD["dino"]]
    x = np.arange(len(labels))
    ax.bar(x, vals, .62, color=cols, zorder=3)
    for xi, v in zip(x, vals):
        ax.text(xi, v + .008, f"{v:.3f}", ha="center", fontsize=9.6, color=INK)
    ax.axhspan(.378, .442, color="#f1ede6", zorder=0)
    ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=9.3)
    ax.set_ylim(0, .74)
    ax.set_ylabel("linear probe accuracy at h")
    ax.set_title("The bar to clear: the frozen target's own ceiling  →  the method zoo",
                 fontsize=11.2, loc="left", pad=10)
    ax.annotate("the target's own ceiling\n(what the frozen target carries)", (0, .378), (1.4, .18),
                fontsize=9, color=MUTE, arrowprops=dict(arrowstyle="->", color=MUTE, lw=1.2))
    _clean(ax)
    _save(fig, "pivot_tiers.png")


def fig_e15_failure():
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    ax = axes[0]
    for fn, lab, c in [("tension_e15a.csv", "arm A (3-term)", OI["blue"]),
                       ("tension_e15b.csv", "arm B (+transport)", OI["sky"])]:
        d = pd.read_csv(os.path.join(CACHE, fn))
        d = d[(d.epoch > 0.3)]
        ax.plot(d.epoch, d.pred_over_varz, color=c, lw=2.2, label=lab)
    ax.axhline(1.0, color=OI["vermillion"], ls=(0, (4, 3)), lw=1.4)
    ax.text(72, 1.06, "constant-h floor = 1  (predict the scene-mean)", color=OI["vermillion"],
            fontsize=8.4, ha="right", va="bottom")
    ax.axhline(0.30, color=MUTE, ls=(0, (1, 2)), lw=1.2)
    ax.text(72, .265, "perfect-image-predictor bound ≈ .30 (irreducible draw-noise)",
            color=MUTE, fontsize=8.0, ha="right", va="top")
    ax.set_yscale("log")
    ax.set_xlabel("epoch"); ax.set_ylabel("pred_over_varz  (tension dial, log)")
    ax.set_title("tension collapses: target ~80% mined by ep70", fontsize=11, loc="left")
    ax.legend(loc="upper right", frameon=False, fontsize=9)
    ax.annotate("beats the floor by ~ep10\n(image-specific, NOT gamed)\nbut bottoms at ~.44 — the\ntarget's information is the ceiling",
                (30, .46), (24, 3.2), fontsize=8.4, color=MUTE, ha="left",
                arrowprops=dict(arrowstyle="->", color=MUTE, lw=1.1))
    _clean(ax)
    ax = axes[1]
    for fn, lab, c in [("acc_e15a.csv", "arm A", OI["blue"]), ("acc_e15b.csv", "arm B", OI["sky"])]:
        e, a = _acc(fn)
        ax.plot(e, a, color=c, lw=2.3, label=lab)
    ax.axhline(.378, color="#b9b2a8", ls="-", lw=1.6)
    ax.text(2, .385, "target ceiling = .378", color=MUTE, fontsize=8.8, va="bottom")
    ax.axhline(.434, color=METHOD["mae"], ls=(0, (4, 3)), lw=1.2)
    ax.text(2, .441, "MAE raw h = .434", color=METHOD["mae"], fontsize=8.6, va="bottom")
    ax.set_xlabel("epoch"); ax.set_ylabel("online probe accuracy")
    ax.set_ylim(0, .5); ax.set_xlim(0, 76)
    ax.set_title("probe plateaus at ~.35 — below its own ceiling", fontsize=11, loc="left")
    ax.legend(loc="lower right", frameon=False, fontsize=9)
    _clean(ax)
    fig.suptitle("The honest failure: information-starved, killed early",
                 fontsize=13, x=.01, ha="left", y=1.02)
    _save(fig, "e15_failure.png")


def fig_e16_tension_contrast():
    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    labels = ["v1 · A", "v1 · B", "v2 · a", "v2 · b"]
    vals = [4.4, 1.2, 26.1, 27.7]
    cols = ["#c9c2b8", "#c9c2b8", OI["green"], OI["green"]]
    x = np.arange(4)
    ax.bar(x, vals, .6, color=cols, zorder=3)
    for xi, v in zip(x, vals):
        ax.text(xi, v + .5, f"{v:.1f}×", ha="center", fontsize=11, color=INK, fontweight="bold")
    ax.axhline(1.0, color=OI["vermillion"], ls=(0, (4, 3)), lw=1.3)
    ax.text(3.4, 1.7, "floor = 1 (trivially solved)", color=OI["vermillion"], fontsize=8.6, ha="right")
    ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=10)
    ax.set_ylim(0, 32)
    ax.set_ylabel("pred_over_varz at end of 2-ep smoke")
    ax.set_title("The fix has 20× the tension: the earlier failure signature does not recur",
                 fontsize=11, loc="left", pad=10)
    ax.annotate("position-marginal:\nnearly solved at ep2", (.5, 2.8), (.5, 13),
                fontsize=9, color=MUTE, ha="center", arrowprops=dict(arrowstyle="->", color=MUTE, lw=1.1))
    ax.annotate("position-conditioned dense:\nfar from solved", (2, 26.1), (2.5, 30.5),
                fontsize=9, color=OI["green"], ha="center", arrowprops=dict(arrowstyle="->", color=OI["green"], lw=1.2))
    _clean(ax)
    _save(fig, "e16_tension_contrast.png")


def fig_e16_budget_gate():
    fig, ax = plt.subplots(figsize=(7.4, 4.5))
    labels = ["scene gist", "position-aware", "whole image"]
    lin = [.442, .5096, .3476]
    cols = ["#8a8279", OI["blue"], "#b9b2a8"]
    x = np.arange(3)
    ax.bar(x, lin, .55, color=cols, zorder=3)
    for xi, v in zip(x, lin):
        ax.text(xi, v + .008, f"{v:.3f}", ha="center", fontsize=10, color=INK)
    ax.annotate("", (1, .5096), (0, .442), arrowprops=dict(arrowstyle="<->", color=GOOD, lw=1.6))
    ax.text(.5, .53, "+.068\npasses", ha="center", color=GOOD, fontsize=9.5, fontweight="bold")
    ax.axhline(.590, color=METHOD["vicreg"], ls=(0, (4, 3)), lw=1.2)
    ax.text(2.4, .597, "VICReg .590", color=METHOD["vicreg"], fontsize=8.6, ha="right", va="bottom")
    ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=10)
    ax.set_ylim(0, .66)
    ax.set_ylabel("linear probe accuracy")
    ax.set_title("Budget check: position-specific content is real (measured before training)",
                 fontsize=11, loc="left", pad=10)
    _clean(ax)
    _save(fig, "e16_budget_gate.png")


def fig_e16_current():
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4))
    ax = axes[0]
    for fn, lab, c in [("tension_e16a.csv", "arm a (dense-local)", OI["green"]),
                       ("tension_e16b.csv", "arm b (+global anchor)", OI["blue"])]:
        d = pd.read_csv(os.path.join(CACHE, fn)); d = d[d.epoch > 0.3]
        ax.plot(d.epoch, d.pred_over_varz, color=c, lw=2.2, label=lab)
    ax.axhline(1.0, color=OI["vermillion"], ls=(0, (4, 3)), lw=1.2)
    ax.text(1, 1.18, "constant-mean floor = 1", color=OI["vermillion"], fontsize=8.2)
    ax.set_yscale("log"); ax.set_xlabel("epoch"); ax.set_ylabel("pred_over_varz (log)")
    ax.set_title("dial descends below floor: position-specific prediction (not gamed)", fontsize=10.2, loc="left")
    ax.legend(loc="upper right", frameon=False, fontsize=9)
    _clean(ax)
    ax = axes[1]
    for fn, lab, c in [("acc_e16a.csv", "arm a", OI["green"]), ("acc_e16b.csv", "arm b", OI["blue"])]:
        e, a = _acc(fn)
        ax.plot(e, a, "-o", color=c, lw=2.2, ms=5, label=lab)
    ax.axhline(.442, color="#8a8279", ls="-", lw=1.4); ax.text(1, .45, "target ceiling .442", color=MUTE, fontsize=8.6)
    ax.set_xlabel("epoch"); ax.set_ylabel("online probe accuracy")
    ax.set_ylim(0, .5)
    ax.set_title("still rising (in-flight, ~ep25–30)", fontsize=11, loc="left")
    ax.legend(loc="lower right", frameon=False, fontsize=9)
    _clean(ax)
    fig.suptitle("Current status — running, monitor-only, no conclusion (not yet converged)",
                 fontsize=12, x=.01, ha="left", y=1.02)
    _save(fig, "e16_current.png")


def fig_invariance_hz():
    """View-invariance (cos-margin) at the backbone h vs the loss space z, from
    results/M2/pair_margin.csv (faithful; h/z per the project's space map)."""
    import csv
    d = {}
    with open("results/M2/pair_margin.csv") as f:
        for r in csv.DictReader(f):
            if r["stack"] != "audit_v1":
                continue
            d[(r["run_id"], r["space"])] = float(r["cos_margin"])
    M = [("simclr", "in100.simclr.s0.ext", "student.h.gap", "student.z.proj.out"),
         ("byol", "in100.byol.s0.ext", "student.h.gap", "student.z.proj.out"),
         ("vicreg", "in100.vicreg.s0.ext", "student.h.gap", "student.z.proj.out"),
         ("dino", "in100.dino.s0.ext", "teacher.h.gap", "teacher.z.dino.bottleneck"),
         ("lejepa", "in100.lejepa.s0.ext", "student.z.embed", "student.z.proj.out"),
         ("ijepa", "in100.ijepa.s0.ext", "teacher.h.gap", "student.z.pred.out"),
         ("mae", "in100.mae.s0.ext", "student.h.gap", None)]
    fig, ax = plt.subplots(figsize=(9.0, 4.7))
    x = np.arange(len(M)); w = .38
    for xi, (m, run, hs, zs) in zip(x, M):
        c = METHOD[m]
        hv = d[(run, hs)]
        ax.bar(xi - w/2, hv, w, color=c, alpha=.30, zorder=3)
        ax.text(xi - w/2, hv + .015, f"{hv:.2f}", ha="center", fontsize=8.6, color=MUTE)
        if zs and (run, zs) in d:
            zv = d[(run, zs)]
            ax.bar(xi + w/2, zv, w, color=c, zorder=3)
            ax.text(xi + w/2, zv + .015, f"{zv:.2f}", ha="center", fontsize=8.6, color=INK)
    ax.set_xticks(x); ax.set_xticklabels([m for m, *_ in M], fontsize=10.5)
    ax.set_ylim(0, 1.04); ax.set_ylabel("view-invariance  (cos-margin)  ↑")
    ax.set_title("Invariance is sharp at the loss space and fades at the backbone",
                 fontsize=12, loc="left", pad=10)
    ax.text(.005, .965, "solid = loss space  ·  faded = backbone", transform=ax.transAxes,
            ha="left", va="top", fontsize=9.2, color=MUTE)
    ax.annotate("no augmentations →\nnothing to be invariant to", (6 - w/2, .132), (4.4, .42),
                fontsize=8.6, color=MUTE, ha="center",
                arrowprops=dict(arrowstyle="->", color=MUTE, lw=1.1))
    _clean(ax)
    _save(fig, "invariance_hz.png")


if __name__ == "__main__":
    fig_invariance_hz()
    fig_e12_probe_bars()
    fig_e12_nmi_scatter()
    fig_e12_dose_curve()
    fig_e12_floor_training()
    fig_acc_lejepa()
    fig_acc_gwave()
    fig_e13_ranker_bars()
    fig_e13_family_decomp()
    fig_e14_strata()
    fig_e14_blur()
    fig_pivot_tiers()
    fig_e15_failure()
    fig_e16_tension_contrast()
    fig_e16_budget_gate()
    fig_e16_current()
    print("done")
