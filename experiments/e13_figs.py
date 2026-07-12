"""E13 figures (presentation artifacts; numbers verbatim from results/diag/e13_*.csv +
results/probes/*.csv). Four PNGs: (1) Spearman bars vs the locked baselines at the primary cell,
(2) distortion-vs-probe scatters (the two meters, matched to their predicted probe), (3) rank-corr
heatmap over the (D_m, sigma) sweep, (4) target-audit panel (PIVOT-E0 reject gates). Okabe-Ito
CVD-safe colors, distinct markers as secondary encoding, direct labels, one axis per panel."""
import csv
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from e13_pivot_rung0 import GATE, PRIMARY, RANDINIT, RANKED, LEJEPA_SUB, probe_rows

ROOT = "/nfs/scistore19/locatgrp/bdemirel/ssl_project"
BLUE, VERM, GREEN, ORNG, GRAY = "#0072B2", "#D55E00", "#009E73", "#E69F00", "#666666"
FIGD = f"{ROOT}/results/figures/e13"
os.makedirs(FIGD, exist_ok=True)


def short(rid):
    s = rid.replace("in100.", "").replace(".ext", "").replace(".s0", "")
    return s.replace("lejepa.e12", "").replace("dino-ctrl.", "ctrl-")


def rows(path):
    with open(f"{ROOT}/results/diag/{path}") as f:
        return list(csv.DictReader(f))


dist, corr, audit = rows("e13_distortion.csv"), rows("e13_rank_corr.csv"), rows("e13_target_audit.csv")
PD, PM = str(PRIMARY[0]), str(PRIMARY[1])
probe = {rid: probe_rows(rid, h) for rid, h in RANKED.items()}
probe[RANDINIT] = probe_rows(RANDINIT, GATE[RANDINIT])


def sym(tk, dm, mult, met):
    """direction-averaged metric per run at one cell (lower-is-better as stored)"""
    acc = {}
    for r in dist:
        if r["tokenizer"] == tk and r["D_m"] == str(dm) and r["sig_mult"] == str(mult):
            acc.setdefault(r["run"], []).append(float(r[met]))
    return {k: float(np.mean(v)) for k, v in acc.items() if len(v) == 2}


def crow(ranker, zoo, pb, cell=None):
    for r in corr:
        if r["ranker"] == ranker and r["zoo"] == zoo and r["probe"] == pb and (cell is None or r["cell"] == cell):
            return float(r["rho"]), float(r["perm_p"])
    return None, None


PRIM_TAG = f"T=mae|D_m={PRIMARY[0]}|s={PRIMARY[1]}"

# --- Fig 1: Spearman bars at the primary cell vs the locked label-free baselines ---------------
fig, axes = plt.subplots(1, 2, figsize=(13, 6.5), sharey=False)
fig.subplots_adjust(left=0.30, right=0.98, wspace=0.55, top=0.90, bottom=0.09)
pivot_meters = [("d_read", "D_read (ridge 1−R²)"), ("d_kern", "D_kern (Gram align)"),
                ("d_read_dof", "D_read dof-matched"), ("cka", "−CKA")]
for ax, pb, ttl in [(axes[0], "linear_raw_v2", "vs converged linear at h"),
                    (axes[1], "knn_v1_k200", "vs kNN k=200 at h")]:
    bars = []
    for met, lab in pivot_meters:
        rho, p = crow(met, "full20", pb, PRIM_TAG)
        if rho is not None:
            bars.append((lab, rho, p, BLUE))
    bat = [(r["ranker"][8:], float(r["rho"]), float(r["perm_p"]))
           for r in corr if r["ranker"].startswith("battery:") and r["zoo"] == "full20" and r["probe"] == pb]
    bat.sort(key=lambda t: -abs(t[1]))
    for name, rho, p in bat[:10]:
        col = GREEN if name.endswith("|e13") else GRAY
        bars.append((name, rho, p, col))
    bars.sort(key=lambda t: t[1])
    ypos = np.arange(len(bars))
    ax.barh(ypos, [b[1] for b in bars], color=[b[3] for b in bars], height=0.62)
    ax.set_yticks(ypos, [b[0] + (" *" if b[2] < 0.05 else "") for b in bars], fontsize=8)
    for v in (-0.44, 0.44):
        ax.axvline(v, color=VERM, lw=1, ls="--")
    ax.axvline(0, color="#999999", lw=0.8)
    ax.set_xlim(-1, 1)
    ax.set_xlabel("Spearman ρ (ranker value vs probe acc)")
    ax.set_title(ttl, fontsize=10)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="x", alpha=0.25, lw=0.5)
fig.suptitle("E13 primary cell (T=mae, D_m=1024, σ=med): PIVOT meters (blue) vs top-10 battery"
             " (gray) + loss-functional (green) baselines; * = perm p<.05; dashed = |ρ|=.44 bar",
             fontsize=9.5)
fig.savefig(f"{FIGD}/e13_spearman_bars.png", dpi=160)
plt.close(fig)

# --- Fig 2: the two meters vs their predicted probe columns ------------------------------------
fam = lambda rid: (BLUE, "o") if rid in LEJEPA_SUB else \
    ((ORNG, "s") if "ctrl" in rid else ((GREEN, "D") if "deitlite" in rid else (VERM, "^")))
fig, axes = plt.subplots(1, 2, figsize=(12.5, 5.4))
fig.subplots_adjust(left=0.07, right=0.99, wspace=0.24, top=0.88, bottom=0.12)
for ax, met, pb, xl in [(axes[0], "d_read", "linear_raw_v2", "D_read at primary cell (lower = better)"),
                        (axes[1], "d_kern", "knn_v1_k200", "D_kern at primary cell (lower = better)")]:
    vals = sym("mae", *PRIMARY, met)
    for rid in list(RANKED) + [RANDINIT]:
        if rid not in vals:
            continue
        col, mk = fam(rid) if rid in RANKED else (GRAY, "x")
        ax.plot([vals[rid]], [probe[rid][pb]], marker=mk, color=col, ms=8,
                mfc="white" if rid == RANDINIT else col, ls="none")
        ax.annotate(short(rid), (vals[rid], probe[rid][pb]), textcoords="offset points",
                    xytext=(4, 4), fontsize=7, color="#222222")
    rho, p = crow(met, "full20", pb, PRIM_TAG)
    ax.set_title(f"{met} vs {pb}   ρ={rho:+.2f} (perm p={p:.3f}, n=20)", fontsize=10)
    ax.set_xlabel(xl)
    ax.set_ylabel("probe val acc")
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(alpha=0.25, lw=0.5)
fig.suptitle("E13: each meter vs its predicted probe column (blue = lejepa family, vermillion = core,"
             " orange = dino-ctrl, green = deitlite, gray x = randinit gate)", fontsize=9.5)
fig.savefig(f"{FIGD}/e13_scatter.png", dpi=160)
plt.close(fig)

# --- Fig 3: rank-corr over the sweep ------------------------------------------------------------
from e13_pivot_rung0 import D_M, SIG_MULT
fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.8))
fig.subplots_adjust(left=0.09, right=0.97, wspace=0.30, top=0.84, bottom=0.14)
for ax, met, pb in [(axes[0], "d_read", "linear_raw_v2"), (axes[1], "d_kern", "knn_v1_k200")]:
    M = np.full((len(D_M) + 1, len(SIG_MULT)), np.nan)
    for i, dm in enumerate(D_M):
        for j, mu in enumerate(SIG_MULT):
            rho, _ = crow(met, "full20", pb, f"T=mae|D_m={dm}|s={mu}")
            M[i, j] = np.nan if rho is None else rho
    rho_raw, _ = crow(met, "full20", pb, "T=mae|D_m=0|s=-1.0")
    M[-1, :] = np.nan
    M[-1, 0] = np.nan if rho_raw is None else rho_raw
    im = ax.imshow(M, cmap="RdBu", vmin=-0.9, vmax=0.9, aspect="auto")
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            if np.isfinite(M[i, j]):
                ax.text(j, i, f"{M[i, j]:+.2f}", ha="center", va="center", fontsize=8,
                        color="#111111" if abs(M[i, j]) < 0.55 else "white")
    ax.set_xticks(range(len(SIG_MULT)), [f"{m}×" for m in SIG_MULT])
    ax.set_yticks(range(len(D_M) + 1), [str(d) for d in D_M] + ["raw"])
    ax.set_xlabel("σ / σ_med")
    ax.set_ylabel("D_m")
    pi, pj = D_M.index(PRIMARY[0]), SIG_MULT.index(PRIMARY[1])
    ax.add_patch(plt.Rectangle((pj - 0.5, pi - 0.5), 1, 1, fill=False, edgecolor="#111111", lw=2))
    ax.set_title(f"ρ({met}, {pb}) — T=mae, full20 (box = primary)", fontsize=10)
fig.colorbar(im, ax=axes, fraction=0.025, pad=0.02, label="Spearman ρ")
fig.savefig(f"{FIGD}/e13_cell_heatmap.png", dpi=160)
plt.close(fig)

# --- Fig 4: target audit (PIVOT-E0 reject gates) ------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.6))
fig.subplots_adjust(left=0.07, right=0.99, wspace=0.26, top=0.86, bottom=0.14)
cols = dict(zip(D_M, [BLUE, GREEN, ORNG, VERM]))
A = [r for r in audit if r["tokenizer"] == "mae" and r["dir"] == "A2B" and r["D_m"] != "0"]
for ax, key, yl, ttl in [(axes[0], "kern_offdiag_mean", "target kernel off-diag mean",
                          "A · kernel concentration (gates: ~0 orthogonal, ~1 constant)"),
                         (axes[1], "sketch_effrank_sub2048", "sketch effective rank (2048-sub)",
                          "B · target sketch effective rank")]:
    for dm in D_M:
        pts = sorted([(float(r["sig_mult"]), float(r[key])) for r in A if r["D_m"] == str(dm)])
        ax.plot([p[0] for p in pts], [p[1] for p in pts], marker="o", ms=5, lw=1.8,
                color=cols[dm], label=f"D_m={dm}")
    ax.set_xscale("log", base=2)
    ax.set_xlabel("σ / σ_med")
    ax.set_ylabel(yl)
    ax.set_title(ttl, fontsize=10)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(alpha=0.25, lw=0.5)
axes[0].axhspan(0.95, 1.0, color="#f0f0f0")
axes[0].axhspan(0.0, 0.02, color="#f0f0f0")
axes[1].set_yscale("log")
axes[0].legend(frameon=False, fontsize=8)
prim = [r for r in A if r["D_m"] == PD and r["sig_mult"] == PM][0]
fig.suptitle(f"E13 target audit, T=mae A2B — primary cell: B/T={prim['target_bt']},"
             f" kmeans100-NMI={prim['target_kmeans100_nmi']}, effrank={prim['sketch_effrank_sub2048']}",
             fontsize=9.5)
fig.savefig(f"{FIGD}/e13_target_audit.png", dpi=160)
plt.close(fig)

print("wrote 4 figures to", FIGD)
