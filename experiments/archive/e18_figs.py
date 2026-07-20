"""Session 2026-07-17 discussion figures (Berker: visuals over number tables).
(1) e17/e17_reach_explainer.png — the reachability instrument, drawn: edge-cost definition,
    per-class readouts, and the proposed v2 negative-relay comparison (Berker's addition).
(2) e17/e17_mu_drift.png — c015 μ-drift cadence (E17 card §POST-CLOSURE ADDENDUM), visual form.
(3) e18/e18_cf_targets.png — Gaussian vs t_ν CF targets at the 17 knots + the slice-kurtosis
    sign-flip that pre-registration branch P-C encodes.
FS-only figures, indexed in results/figures/INDEX.md."""
import csv

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle, FancyArrowPatch

ROOT = "/nfs/scistore19/locatgrp/bdemirel/ssl_project"
BLUE, ORANGE, GREEN, RED, GRAY = "#3d65d0", "#e07b39", "#3a8c5c", "#c04f4f", "#8a8a8a"
rng = np.random.default_rng(3)


def cloud(ax, c, r, color, n=8, label=None):
    pts = c + rng.normal(0, r * 0.45, (n, 2))
    ax.scatter(*pts.T, s=14, color=color, alpha=0.65, zorder=3)
    ax.add_patch(Circle(c, r, fill=False, ls="--", lw=1.2, ec=color, alpha=0.8))
    ax.plot(*c, marker="x", ms=8, color=color, mew=2, zorder=4)
    if label:
        ax.annotate(label, c, textcoords="offset points", xytext=(0, -14 - 72 * r),
                    ha="center", fontsize=9, color=color)
    return np.array(c), r


def fig_reach():
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6))
    for ax in axes:
        ax.set_aspect("equal")
        ax.axis("off")

    ax = axes[0]
    ax.set_title("edge cost between instance clouds\nw = max(0, d(x̄ᵢ,x̄ⱼ) − (rᵢ+rⱼ))", fontsize=10)
    a, ra = cloud(ax, (0, 0), 0.8, BLUE, label="instance i\n(8 views, radius rᵢ)")
    b, rb = cloud(ax, (1.35, 0.25), 0.7, BLUE, label="instance j")
    c, rc = cloud(ax, (3.3, -0.15), 0.52, BLUE, label="instance k")
    ax.plot(*zip(a, b), color=GREEN, lw=2.5, zorder=2)
    ax.annotate("clouds touch:\nw = 0 (free travel)", ((a + b) / 2 + [0, 0.55]),
                ha="center", fontsize=9, color=GREEN)
    u = (c - b) / np.linalg.norm(c - b)
    s1, s2 = b + u * rb, c - u * rc
    ax.plot(*zip(b, c), color=GRAY, lw=0.8, ls=":")
    ax.plot(*zip(s1, s2), color=ORANGE, lw=3, zorder=2)
    ax.annotate("gap = bridge cost", ((s1 + s2) / 2 + [0.1, 0.42]), ha="center",
                fontsize=9, color=ORANGE)
    ax.annotate("r knobs: r_mean · r90 (~robust furthest-aug) · r_max\nsupport variant: min view-pair distance (gap_min)",
                (1.6, -1.55), ha="center", fontsize=8.5, color="#444444")
    ax.set_xlim(-1.3, 4.3)
    ax.set_ylim(-1.9, 1.7)

    ax = axes[1]
    ax.set_title("per-class readouts (one graph per class)", fontsize=10)
    centers = [(-0.1, 0.75), (1.0, 1.05), (1.9, 0.55), (1.05, 0.0), (3.3, 0.9), (3.95, 0.25), (3.6, -1.0)]
    radii = [0.42, 0.4, 0.44, 0.38, 0.4, 0.36, 0.3]
    cs = [cloud(ax, cc, rr, BLUE if i < 4 else (GREEN if i < 6 else GRAY))
          for i, (cc, rr) in enumerate(zip(centers, radii))]
    for i, j in [(0, 1), (1, 2), (1, 3)]:
        ax.plot(*zip(cs[i][0], cs[j][0]), color=BLUE, lw=2)
    ax.plot(*zip(cs[4][0], cs[5][0]), color=GREEN, lw=2)
    for i, j in [(2, 4), (5, 6)]:
        u = (cs[j][0] - cs[i][0]) / np.linalg.norm(cs[j][0] - cs[i][0])
        ax.plot(*zip(cs[i][0] + u * cs[i][1], cs[j][0] - u * cs[j][1]),
                color=ORANGE, lw=2.5, ls="-")
    ax.annotate("zero-cost components (percolation):\n|C| = 4, 2, 1 → perc_lcc 4/7, perc_seed (16+4+1)/49",
                (1.8, -1.9), ha="center", fontsize=8.5, color="#444444")
    ax.annotate("MST = free edges + the orange bridges\nmst_cost = Σ gaps · mst_max = worst bridge",
                (1.8, 2.15), ha="center", fontsize=8.5, color=ORANGE)
    ax.set_xlim(-1.0, 4.9)
    ax.set_ylim(-2.4, 2.7)

    ax = axes[2]
    ax.set_title("v2 addition (Berker): negative-relay comparison\nspan positives via anything vs via positives only", fontsize=10)
    p1, r1 = cloud(ax, (0, 0), 0.62, BLUE, label="positive")
    p2, r2 = cloud(ax, (3.4, 0.1), 0.62, BLUE, label="positive")
    q, rq = cloud(ax, (1.7, -0.05), 0.55, RED, label="negative class")
    u = (p2 - p1) / np.linalg.norm(p2 - p1)
    ax.plot(*zip(p1 + [0, 0.75], p2 + [0, 0.75]), color=ORANGE, lw=2.5)
    ax.annotate("within-class bridge: costly", ((p1 + p2) / 2 + [0, 1.0]), ha="center",
                fontsize=9, color=ORANGE)
    for s, t in [(p1, q), (q, p2)]:
        v = (t - s) / np.linalg.norm(t - s)
        ax.plot(*zip(s + v * 0.6, t - v * 0.55), color=RED, lw=2.5, ls="-")
    ax.annotate("relay through negative territory: cheap\n⇒ class is interleaved, not self-connected",
                (1.7, -1.35), ha="center", fontsize=9, color=RED)
    ax.annotate("readout: cost ratio (unrestricted / within-class) per class",
                (1.7, -2.1), ha="center", fontsize=8.5, color="#444444")
    ax.set_xlim(-1.2, 4.6)
    ax.set_ylim(-2.5, 2.2)

    fig.tight_layout()
    fig.savefig(f"{ROOT}/results/figures/e17/e17_reach_explainer.png", dpi=160)
    plt.close(fig)


def fig_mu():
    rows = list(csv.DictReader(open(f"{ROOT}/results/diag/e17_mu_drift.csv")))
    d = {r["tag"]: {k: float(v) for k, v in r.items() if k != "tag"} for r in rows}
    eps = [25, 50, 75, 100]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))

    ax = axes[0]
    for key, col, lab in [("mu_sq", BLUE, "‖μ‖²"), ("tr_sigma", ORANGE, "tr Σ"),
                          ("e_xsq", GRAY, "E‖x‖²")]:
        y = [d[f"c015.ep{e}"][key] for e in eps]
        ax.plot(eps, y, marker="o", color=col, lw=2)
        ax.annotate(lab, (eps[-1], y[-1]), textcoords="offset points", xytext=(8, -3),
                    color=col, fontsize=10)
        ax.axhline(d["ctrl"][key], color=col, ls=":", lw=1.2, alpha=0.7)
    ax.annotate("dotted = control (e17c)", (25, d["ctrl"]["e_xsq"]), textcoords="offset points",
                xytext=(0, 6), fontsize=8.5, color="#444444")
    ax.set_yscale("log")
    ax.set_xlabel("epoch (c015 cadence)")
    ax.set_title("dilution is done by ep25 (trΣ, E‖x‖² frozen);\nthe mean first INFLATES, then decays ×3.9 alone", fontsize=10)
    ax.set_xticks(eps)
    ax.grid(alpha=0.15)

    ax = axes[1]
    for key, col, lab in [("cos_to_ctrl", BLUE, "cos(μ_ep, μ_ctrl)"),
                          ("cos_to_ep100", ORANGE, "cos(μ_ep, μ_ep100)")]:
        y = [d[f"c015.ep{e}"][key] for e in eps]
        ax.plot(eps, y, marker="o", color=col, lw=2)
        ax.annotate(lab, (eps[0], y[0]), textcoords="offset points", xytext=(2, 8),
                    color=col, fontsize=9.5)
    ax.axhline(0, color="#444444", lw=0.8)
    ax.annotate("ends ⊥ the control cone (−.07):\nresidual mean is a NEW rotating direction,\nnot the old cone shrunk",
                (52, 0.30), fontsize=9, color="#444444")
    ax.set_xlabel("epoch")
    ax.set_ylim(-0.15, 1.05)
    ax.set_title("the mean direction rotates while it shrinks", fontsize=10)
    ax.set_xticks(eps)
    ax.grid(alpha=0.15)

    fig.tight_layout()
    fig.savefig(f"{ROOT}/results/figures/e17/e17_mu_drift.png", dpi=160)
    plt.close(fig)


def fig_cf():
    import torch
    import sys
    sys.path.insert(0, ROOT)
    from sslgap.methods.lejepa import t_nu_cf
    tt = torch.linspace(0, 3, 300)
    tk = torch.linspace(0, 3, 17)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))

    ax = axes[0]
    g = np.exp(-tt.numpy() ** 2 / 2)
    c82, c15 = t_nu_cf(tt, 8.2).numpy(), t_nu_cf(tt, 15.24).numpy()
    ax.plot(tt, g, color=GRAY, lw=2, ls="--")
    ax.plot(tt, c82, color=ORANGE, lw=2)
    ax.plot(tt, c15, color=BLUE, lw=2)
    ax.fill_between(tt, g, c82, color=ORANGE, alpha=0.15)
    for nu, col in [(None, GRAY), (8.2, ORANGE), (15.24, BLUE)]:
        yk = np.exp(-tk.numpy() ** 2 / 2) if nu is None else t_nu_cf(tk, nu).numpy()
        ax.plot(tk, yk, "o", ms=4, color=col)
    ax.annotate("Gaussian target (E17 sigreg)", (1.15, 0.44), color=GRAY, fontsize=9.5)
    ax.annotate("t_8.2 (launched)", (2.15, 0.20), color=ORANGE, fontsize=9.5)
    ax.annotate("t_15.2 (queued discriminator)", (1.28, 0.02), color=BLUE, fontsize=9.5)
    ax.annotate("shaded = the ONLY thing E18 changes\n(moments 1–2 matched; targets differ in tails)",
                (0.62, 0.86), fontsize=9, color="#444444")
    ax.set_xlabel("t (the 17 CF knots)")
    ax.set_ylabel("target CF φ(t)")
    ax.set_title("one-constant swap at the knots", fontsize=10)
    ax.grid(alpha=0.15)

    ax = axes[1]
    items = [("Gaussian = t_∞:  demands κ = 0", 0.0, GRAY),
             ("t_15.2:  demands κ = 6/(15.2−4) = 0.53", 0.53, BLUE),
             ("data slices (e17c):  measured κ = 0.53", 0.534, "#222222"),
             ("t_8.2:  demands κ = 6/(8.2−4) = 1.43", 1.43, ORANGE)]
    for y, (lab, x, col) in enumerate(items):
        ax.plot(x, y, "o", ms=10, color=col)
        ax.annotate(lab, (x, y), textcoords="offset points", xytext=(12, -4),
                    fontsize=10, color=col)
    ax.axvline(0.534, color="#222222", lw=0.8, ls=":")
    ax.add_patch(FancyArrowPatch((0.5, 0.35), (0.06, 0.05), arrowstyle="->", mutation_scale=14, color=GRAY))
    ax.annotate("Gaussian: squash tails", (0.09, -0.32), fontsize=9, color=GRAY)
    ax.add_patch(FancyArrowPatch((0.57, 2.65), (1.38, 2.95), arrowstyle="->", mutation_scale=14, color=ORANGE))
    ax.annotate("t_8.2: FATTEN tails (sign flip → branch P-C)", (0.62, 2.42), fontsize=9, color=ORANGE)
    ax.annotate("t_15.2: ≈ zero tail pull\n(pure shape-anchoring cell)", (0.62, 1.12), fontsize=9, color=BLUE)
    ax.set_xlim(-0.25, 2.35)
    ax.set_ylim(-0.7, 3.5)
    ax.set_yticks([])
    ax.set_xlabel("excess kurtosis κ a slice is asked to have   (smaller ν = heavier tails: κ = 6/(ν−4))")
    ax.set_title("why ν matters: what each target demands of slices", fontsize=10)
    ax.grid(alpha=0.15, axis="x")

    fig.tight_layout()
    fig.savefig(f"{ROOT}/results/figures/e18/e18_cf_targets.png", dpi=160)
    plt.close(fig)


if __name__ == "__main__":
    import os
    os.makedirs(f"{ROOT}/results/figures/e18", exist_ok=True)
    fig_reach()
    fig_mu()
    fig_cf()
    print("wrote e17_reach_explainer.png, e17_mu_drift.png, e18_cf_targets.png")
