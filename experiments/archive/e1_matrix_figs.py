"""E1-IN100 matrix figures (M2, seed 0) — the joint-reading visual layer over
results/M2/E1_IN100_MATRIX.md. Four figures, numbers only (no glyph is scored here):
battery h-vs-z dumbbells, E01-T2 kurt sign-check (toy -> IN-100 movement), headline probes
vs anchors, pair margins. Emits results/figures/m2/e1_*.png.

  python experiments/e1_matrix_figs.py
"""
import os

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from experiments.report_m1 import H_SPACE, Z_FINAL

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(ROOT, "results")
OUT = os.path.join(RES, "figures", "m2")
ORDER = ["simclr", "byol", "vicreg", "dino", "mae", "ijepa", "lejepa"]
# validated categorical palette (dataviz reference order), fixed per method across all M2 figures
COLOR = {"simclr": "#2a78d6", "byol": "#1baf7a", "vicreg": "#eda100", "dino": "#008300",
         "mae": "#4a3aa7", "ijepa": "#e34948", "lejepa": "#e87ba4"}
INK, MUT, GRID = "#0b0b0b", "#52514e", "#d8d7d2"
TOY_RUN = {"simclr": "toy.simclr.s0.ext", "byol": "toy.byol.s0.ext", "vicreg": "toy.vicreg.s0.ext",
           "dino": "toy.dino.s0.probefix.ext", "mae": "toy.mae.s0.ext",
           "ijepa": "toy.ijepa.s0.ext", "lejepa": "toy.lejepa.s0.ext"}


def bat(rid):
    return pd.read_csv(f"{RES}/battery/{rid}.csv")


def cell(df, space, metric, variant="raw|full"):
    r = df[(df.space == space) & (df.metric == metric) & (df.variant == variant)]
    return float(r.value.iloc[0]) if len(r) else float("nan")


def style(ax, logy=False):
    for s in ax.spines.values():
        s.set_color(GRID)
    ax.tick_params(labelsize=8, colors=MUT)
    if logy:
        ax.set_yscale("log")


def fig_battery():
    metrics = [("uniformity", "uniformity", False, False),
               ("variance_floor.min_over_mean_std", "var floor (min/mean std)", False, False),
               ("offdiag_redundancy.mean_abs_corr", "decorrelation (mean |corr|)", False, False),
               ("rankme", "eff. rank fraction (RankMe / dim)", False, True),
               ("kurt_topeig.worst", "isotropy (kurt worst, top-10 eig)", True, False),
               ("epps_pulley", "Epps-Pulley (paired)", True, False)]
    data = {m: bat(f"in100.{m}.s0.ext") for m in ORDER}
    shared_null = bat("in100.randinit-s0.ext")
    own_null = {"lejepa": bat("in100.lejepa.s0.null.ext")}   # own-arch null (D-007 gap closed)

    def celld(df, space, met):
        r = df[(df.space == space) & (df.metric == met) & (df.variant == "raw|full")]
        return (float(r.value.iloc[0]), float(r.d.iloc[0])) if len(r) else (float("nan"), float("nan"))

    fig, axes = plt.subplots(2, 3, figsize=(13.5, 7.2), facecolor="white")
    for ax, (met, title, logy, frac) in zip(axes.flat, metrics):
        for i, m in enumerate(ORDER):
            h, z = H_SPACE[m], Z_FINAL[m]
            vh, dh = celld(data[m], h, met)
            vz, dz = celld(data[m], z, met) if z else ((float("nan"),) * 2)
            nh, dn = (celld(own_null[m], h, met) if m in own_null
                      else celld(shared_null, h.replace("teacher.", "student."), met))
            if frac:
                vh, vz, nh = vh / dh, vz / dz, nh / dn
            c = COLOR[m]
            if vz == vz:
                ax.plot([i, i], [vh, vz], color=c, lw=1.6, alpha=0.7, zorder=2)
                ax.plot(i, vz, "o", ms=11, mfc="none", mec=c, mew=1.7, zorder=4)
            ax.plot(i, vh, "o", ms=7, mfc=c, mec=c, zorder=3)
            if nh == nh:
                ax.plot(i, nh, "_", ms=11, color="#8a8a85", mew=1.8, zorder=2)
        ax.set_xticks(range(len(ORDER)))
        ax.set_xticklabels(ORDER, rotation=45, ha="right", fontsize=8)
        ax.set_title(title, fontsize=10, color=INK)
        style(ax, logy)
    handles = [plt.Line2D([], [], marker="o", ls="", mfc=INK, mec=INK, label="h (filled)"),
               plt.Line2D([], [], marker="o", ls="", ms=9, mfc="none", mec=INK, label="z.final (open ring)"),
               plt.Line2D([], [], marker="_", ls="", color="#8a8a85", ms=11, mew=1.8,
                          label="randinit null at h (own-arch for lejepa)")]
    fig.legend(handles=handles, loc="upper center", ncol=3, fontsize=9, frameon=False,
               bbox_to_anchor=(0.5, 0.955))
    fig.suptitle("E1 IN-100 seed-0 battery: every desideratum at h (filled) and z.final (open) - unscored",
                 fontsize=12, color=INK)
    fig.tight_layout(rect=(0, 0, 1, 0.91))
    fig.savefig(f"{OUT}/e1_battery_hz.png", dpi=150)


def fig_kurt_signs():
    fig, ax = plt.subplots(figsize=(7.6, 6.6), facecolor="white")
    ax.plot([1e-2, 400], [1e-2, 400], color=GRID, lw=1.2, zorder=1)
    ax.text(9, 12.5, "z = h (flat)", fontsize=8, color=MUT, rotation=38)
    for m in ORDER:
        h, z = H_SPACE[m], Z_FINAL[m]
        if not z:
            continue
        d_toy, d_in = bat(TOY_RUN[m]), bat(f"in100.{m}.s0.ext")
        th, tz = cell(d_toy, h, "kurt_topeig.worst"), cell(d_toy, z, "kurt_topeig.worst")
        ih, iz = cell(d_in, h, "kurt_topeig.worst"), cell(d_in, z, "kurt_topeig.worst")
        c = COLOR[m]
        ax.plot(th, tz, "o", ms=6, mfc="white", mec=c, mew=1.3, zorder=3)
        ax.annotate("", xy=(ih, iz), xytext=(th, tz),
                    arrowprops=dict(arrowstyle="->", color=c, lw=1.1, alpha=0.6))
        ax.plot(ih, iz, "o", ms=8, mfc=c, mec=c, zorder=4)
        dx, dy = (6, -13) if m == "ijepa" else (6, 5)     # dodge the lejepa/ijepa collision
        ax.annotate(m, (ih, iz), xytext=(dx, dy), textcoords="offset points", fontsize=9, color=INK)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(0.25, 30)
    ax.set_ylim(0.25, 500)
    ax.set_xlabel("kurt_topeig.worst at h  (|excess kurtosis|, worst of top-10 eigendirections)",
                  fontsize=9, color=INK)
    ax.set_ylabel("kurt_topeig.worst at z.final", fontsize=9, color=INK)
    ax.set_title("E01-T2 sign check: fourth-moment families, toy (hollow) -> IN-100 (solid)\n"
                 "above diagonal = cluster-sharpening at z; below = Gaussian-smoothing at z - unscored",
                 fontsize=10.5, color=INK)
    ax.text(0.03, 0.03, "matched-Gaussian nulls: 0.04-0.12 (both axes)", transform=ax.transAxes,
            fontsize=8, color=MUT)
    style(ax)
    fig.tight_layout()
    fig.savefig(f"{OUT}/e1_kurt_signs.png", dpi=150)


def fig_probes():
    probes = {m: pd.read_csv(f"{RES}/probes/in100.{m}.s0.ext.csv") for m in ORDER}
    deit = pd.read_csv(f"{RES}/probes/in100.deitlite.s0.ext.csv")
    rnd = pd.read_csv(f"{RES}/probes/in100.randinit-s0.ext.csv")

    def acc(df, space, probe):
        r = df[(df.space == space) & (df.probe == probe)]
        return float(r.val_acc.iloc[0]) if len(r) else float("nan")

    # the non-headline trunk readout per method (CLS if h=GAP, GAP if h=CLS; lejepa = the raw
    # CLS one Linear upstream of its official h=z.embed - the F4 tension cell)
    ALT = {"simclr": "student.h.gap", "byol": "student.h.gap", "vicreg": "student.h.gap",  # D-036: h=CLS now, so GAP is the alt readout
           "mae": "student.h.cls", "dino": "teacher.h.gap", "ijepa": "teacher.h.cls",
           "lejepa": "student.h.cls"}
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 5.2), facecolor="white", sharey=True)
    for ax, probe, title in zip(axes, ["linear_raw_v2", "knn_v1_k200"],
                                ["linear_raw_v2 (converged linear)", "kNN k=200 (weighted cosine)"]):
        for y, m in enumerate(ORDER):
            h, z = H_SPACE[m], Z_FINAL[m]
            vh, c = acc(probes[m], h, probe), COLOR[m]
            if z:
                vz = acc(probes[m], z, probe)
                ax.plot([vz, vh], [y, y], color=c, lw=1.6, alpha=0.7)
                ax.plot(vz, y, "o", ms=11, mfc="none", mec=c, mew=1.7, zorder=4)
            ax.plot(vh, y, "o", ms=8, mfc=c, mec=c, zorder=3)
            va = acc(probes[m], ALT[m], probe)
            if va == va:
                ax.plot(va, y, "s", ms=5, mfc="none", mec=c, mew=1.3, zorder=3)
        for v, lab, ls in [(acc(deit, "student.h.cls", probe), "deitlite h.cls", "--"),
                           (acc(rnd, "student.h.cls", probe), "randinit h.cls", ":")]:  # D-036: null matches the audited tap (CLS)
            ax.axvline(v, color="#8a8a85", ls=ls, lw=1.3)
            ax.text(v + 0.006, -0.42, f"{lab} {v:.3f}", fontsize=7.5, color=MUT, rotation=90,
                    va="bottom")
        ax.set_yticks(range(len(ORDER)))
        ax.set_yticklabels(ORDER, fontsize=9)
        ax.set_xlabel("IN-100 val top-1", fontsize=9, color=INK)
        ax.set_title(title, fontsize=10, color=INK)
        ax.set_xlim(0.05, 0.75)
        ax.set_ylim(-0.5, 6.6)
        style(ax)
    axes[0].plot([], [], "o", mfc=INK, mec=INK, label="h (filled)")
    axes[0].plot([], [], "o", ms=9, mfc="none", mec=INK, label="z.final (open ring)")
    axes[0].plot([], [], "s", ms=5, mfc="none", mec=INK, label="alt trunk readout (CLS<->GAP; lejepa: raw CLS)")
    axes[0].legend(fontsize=8, frameon=False, loc="upper left", bbox_to_anchor=(0.01, 0.98))
    fig.suptitle("E1 IN-100 seed-0 headline probes (D-020 v2) at h vs z.final, with anchors - unscored",
                 fontsize=12, color=INK)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(f"{OUT}/e1_probes.png", dpi=150)


def fig_pair_margins():
    pm = pd.read_csv(f"{RES}/M2/pair_margin.csv")

    def row(rid, space):
        # pm.stack would resolve to DataFrame.stack (the method) — index the column explicitly
        r = pm[(pm.run_id == rid) & (pm.space == space) & (pm["stack"] == "audit_v1")]
        return r.iloc[0] if len(r) else None

    fig, axes = plt.subplots(1, 2, figsize=(12.5, 5.2), facecolor="white")
    for ax, col, title, better in zip(
            axes, ["cos_margin", "align_rel"],
            ["pos-pair minus random-pair cosine (higher = view-specific geometry)",
             "align_pos / align_rand (lower = more view-invariant, D-013)"], [1, -1]):
        for y, m in enumerate(ORDER):
            rid, c = f"in100.{m}.s0.ext", COLOR[m]
            rh, rz = row(rid, H_SPACE[m]), (row(rid, Z_FINAL[m]) if Z_FINAL[m] else None)
            vh = float(rh[col]) if rh is not None else float("nan")
            if rz is not None:
                vz = float(rz[col])
                ax.plot([vz, vh], [y, y], color=c, lw=1.6, alpha=0.7)
                ax.plot(vz, y, "o", ms=11, mfc="none", mec=c, mew=1.7, zorder=4)
            ax.plot(vh, y, "o", ms=8, mfc=c, mec=c, zorder=3)
            rn = row("in100.randinit-s0.ext", H_SPACE[m].replace("teacher.", "student."))
            if rn is not None:
                ax.plot(float(rn[col]), y, "_", ms=11, color="#8a8a85", mew=1.8)
        ax.set_yticks(range(len(ORDER)))
        ax.set_yticklabels(ORDER, fontsize=9)
        ax.set_title(title, fontsize=9.5, color=INK)
        style(ax)
    handles = [plt.Line2D([], [], marker="o", ls="", mfc=INK, mec=INK, label="h (filled)"),
               plt.Line2D([], [], marker="o", ls="", ms=9, mfc="none", mec=INK, label="z.final (open ring)"),
               plt.Line2D([], [], marker="_", ls="", color="#8a8a85", ms=11, mew=1.8, label="randinit null")]
    fig.legend(handles=handles, loc="upper center", ncol=3, fontsize=8.5, frameon=False,
               bbox_to_anchor=(0.5, 0.92))
    fig.suptitle("E1 IN-100 seed-0 pair margins (D-013): alignment scored on margins, never raw - unscored",
                 fontsize=12, color=INK)
    fig.tight_layout(rect=(0, 0, 1, 0.87))
    fig.savefig(f"{OUT}/e1_pair_margins.png", dpi=150)


def fig_kurt_variants():
    """Berker 2026-07-10: is `worst` the right kurt cell at IN-100 — show the three estimator
    variants side by side, each against its own matched-Gaussian null."""
    variants = [("kurt_topeig.worst", "worst |excess| of top-10 eigendirections", "log"),
                ("kurt_topeig.mean", "signed mean of top-10 eigendirections", "symlog"),
                ("kurt_slices_mean_abs", "mean |excess| over 256 random slices", "log")]
    data = {m: bat(f"in100.{m}.s0.ext") for m in ORDER}
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.8), facecolor="white")
    for ax, (met, title, sc) in zip(axes, variants):
        for i, m in enumerate(ORDER):
            for space, dx, filled in ((H_SPACE[m], -0.18, True), (Z_FINAL[m], 0.18, False)):
                if space is None:
                    continue
                r = data[m][(data[m].space == space) & (data[m].metric == met)
                            & (data[m].variant == "raw|full")]
                if not len(r):
                    continue
                v, ng = float(r.value.iloc[0]), float(r.null_gauss.iloc[0])
                c = COLOR[m]
                ax.plot(i + dx, v, "o", ms=7, mfc=(c if filled else "none"), mec=c, mew=1.6, zorder=3)
                ax.plot(i + dx, ng, "_", ms=9, color="#8a8a85", mew=1.6, zorder=2)
        if sc == "log":
            ax.set_yscale("log")
        elif sc == "symlog":
            ax.set_yscale("symlog", linthresh=1)
            ax.axhline(0, color=GRID, lw=1)
        ax.set_xticks(range(len(ORDER)))
        ax.set_xticklabels(ORDER, rotation=45, ha="right", fontsize=8)
        ax.set_title(title, fontsize=9.5, color=INK)
        style(ax)
    handles = [plt.Line2D([], [], marker="o", ls="", mfc=INK, mec=INK, label="h (filled)"),
               plt.Line2D([], [], marker="o", ls="", mfc="none", mec=INK, label="z.final (open)"),
               plt.Line2D([], [], marker="_", ls="", color="#8a8a85", ms=9, mew=1.6,
                          label="matched-Gaussian null (per space)")]
    fig.legend(handles=handles, loc="upper center", ncol=3, fontsize=9, frameon=False,
               bbox_to_anchor=(0.5, 0.93))
    fig.suptitle("Excess-kurtosis estimator variants at IN-100, each vs its own matched-Gaussian null - unscored",
                 fontsize=12, color=INK)
    fig.tight_layout(rect=(0, 0, 1, 0.86))
    fig.savefig(f"{OUT}/e1_kurt_variants.png", dpi=150)


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    fig_battery()
    fig_kurt_signs()
    fig_probes()
    fig_pair_margins()
    fig_kurt_variants()
    print(f"[e1figs] wrote 5 figures to {OUT}")
