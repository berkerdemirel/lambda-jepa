"""E13 re-read WITHOUT the supervised anchor (Berker 2026-07-12, post-numbers directive;
REGISTERED-LATE sensitivity, does not replace the pre-registered n=20 outcome). Rationale on the
card: deitlite's probe column is mechanism-linked — supervised CE optimizes exactly the linear
class separability the probe measures while discarding context information, an advantage/handicap
pair no SSL member has; on a dense/multi-task probe suite it would invert. Recomputes every rank
correlation on full19 (= full20 − deitlite) and nonlejepa8, battery + moment baselines included,
and regenerates the affected figures with the _n19 suffix. n=20 artifacts untouched."""
import csv
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from e13_pivot_rung0 import (CELLS, D_M, GATE, LEJEPA_SUB, PRIMARY, PROBES, RANDINIT, RANKED,
                             SIG_MULT, TOKENIZERS, probe_rows, read_baselines, spearman_perm)

ROOT = "/nfs/scistore19/locatgrp/bdemirel/ssl_project"
DEIT = "in100.deitlite.s0.ext"
FIGD = f"{ROOT}/results/figures/e13"
BLUE, VERM, GREEN, ORNG, GRAY = "#0072B2", "#D55E00", "#009E73", "#E69F00", "#666666"

ranked19 = {r: h for r, h in RANKED.items() if r != DEIT}
zoos = {"full19": list(ranked19), "lejepa11": [r for r in ranked19 if r in LEJEPA_SUB],
        "nonlejepa8": [r for r in ranked19 if r not in LEJEPA_SUB]}
probe = {rid: probe_rows(rid, h) for rid, h in RANKED.items()}
probe[RANDINIT] = probe_rows(RANDINIT, GATE[RANDINIT])

with open(f"{ROOT}/results/diag/e13_distortion.csv") as f:
    dist = list(csv.DictReader(f))
sym = {}
for r in dist:
    sym.setdefault((r["tokenizer"], r["D_m"], r["sig_mult"], r["run"]), []).append(r)


def symval(tk, dm, mult, rid, met):
    rows = sym.get((tk, str(dm), str(mult), rid), [])
    if len(rows) != 2:
        return None
    v = float(np.mean([float(r[met]) for r in rows]))
    return -v if met == "cka" else v


corr_rows = []


def add_corr(ranker, cell_tag, vals):
    for zname, zrids in zoos.items():
        rids = [r for r in zrids if r in vals and vals[r] is not None]
        if len(rids) < 5:
            continue
        for pb in PROBES:
            rho, p = spearman_perm([vals[r] for r in rids], [probe[r][pb] for r in rids])
            corr_rows.append({"ranker": ranker, "cell": cell_tag, "zoo": zname, "n": len(rids),
                              "probe": pb, "rho": round(rho, 4), "perm_p": round(p, 5)})


for tk in TOKENIZERS:
    for dm, mult in CELLS:
        tag = f"T={tk}|D_m={dm}|s={mult}"
        for met in ("d_read", "d_read_dof", "d_kern", "cka"):
            add_corr(met, tag, {r: symval(tk, dm, mult, r, met) for r in ranked19})

base, bkeys = read_baselines(ranked19)
for bk in bkeys:
    add_corr(f"battery:{bk}", "train500@h", {r: base[r][bk] for r in ranked19 if bk in base[r]})

out = f"{ROOT}/results/diag/e13_rank_corr_n19.csv"
with open(out, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(corr_rows[0]))
    w.writeheader(); w.writerows(corr_rows)
print(f"wrote {out} ({len(corr_rows)} rows)", flush=True)


def crow(ranker, zoo, pb, cell=None):
    for r in corr_rows:
        if r["ranker"] == ranker and r["zoo"] == zoo and r["probe"] == pb and (cell is None or r["cell"] == cell):
            return float(r["rho"]), float(r["perm_p"])
    return None, None


PRIM_TAG = f"T=mae|D_m={PRIMARY[0]}|s={PRIMARY[1]}"

# --- bars_n19 ------------------------------------------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(13, 6.5))
fig.subplots_adjust(left=0.30, right=0.98, wspace=0.55, top=0.90, bottom=0.09)
for ax, pb, ttl in [(axes[0], "linear_raw_v2", "vs converged linear at h"),
                    (axes[1], "knn_v1_k200", "vs kNN k=200 at h")]:
    bars = []
    for met, lab in [("d_read", "D_read (ridge 1−R²)"), ("d_kern", "D_kern (Gram align)"),
                     ("d_read_dof", "D_read dof-matched"), ("cka", "−CKA")]:
        rho, p = crow(met, "full19", pb, PRIM_TAG)
        if rho is not None:
            bars.append((lab, rho, p, BLUE))
    bat = [(r["ranker"][8:], float(r["rho"]), float(r["perm_p"]))
           for r in corr_rows if r["ranker"].startswith("battery:") and r["zoo"] == "full19" and r["probe"] == pb]
    bat.sort(key=lambda t: -abs(t[1]))
    for name, rho, p in bat[:10]:
        bars.append((name, rho, p, GREEN if name.endswith("|e13") else GRAY))
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
fig.suptitle("E13 primary cell, SUPERVISED ANCHOR EXCLUDED (n=19, Berker-directed re-read):"
             " PIVOT meters (blue) vs top-10 battery (gray) + loss-functional (green); * = perm p<.05",
             fontsize=9.5)
fig.savefig(f"{FIGD}/e13_spearman_bars_n19.png", dpi=160)
plt.close(fig)

# --- scatter_n19 ----------------------------------------------------------------------------------
fam = lambda rid: (BLUE, "o") if rid in LEJEPA_SUB else \
    ((ORNG, "s") if "ctrl" in rid else (VERM, "^"))
fig, axes = plt.subplots(1, 2, figsize=(12.5, 5.4))
fig.subplots_adjust(left=0.07, right=0.99, wspace=0.24, top=0.88, bottom=0.12)
for ax, met, pb, xl in [(axes[0], "d_read", "linear_raw_v2", "D_read at primary cell (lower = better)"),
                        (axes[1], "d_kern", "knn_v1_k200", "D_kern at primary cell (lower = better)")]:
    for rid in list(RANKED) + [RANDINIT]:
        v = symval("mae", *PRIMARY, rid, met)
        if v is None:
            continue
        if rid in ranked19:
            col, mk = fam(rid)
            mfc = col
        else:  # deitlite + randinit: unranked anchors, gray
            col, mk, mfc = GRAY, ("D" if rid == DEIT else "x"), "white"
        ax.plot([v], [probe[rid][pb]], marker=mk, color=col, ms=8, mfc=mfc, ls="none")
        ax.annotate(("deitlite (excl.)" if rid == DEIT else
                     rid.replace("in100.", "").replace(".ext", "").replace(".s0", "")
                        .replace("lejepa.e12", "").replace("dino-ctrl.", "ctrl-")),
                    (v, probe[rid][pb]), textcoords="offset points", xytext=(4, 4), fontsize=7,
                    color="#222222")
    rho, p = crow(met, "full19", pb, PRIM_TAG)
    ax.set_title(f"{met} vs {pb}   ρ={rho:+.2f} (perm p={p:.3f}, n=19)", fontsize=10)
    ax.set_xlabel(xl)
    ax.set_ylabel("probe val acc")
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(alpha=0.25, lw=0.5)
fig.suptitle("E13 re-read: supervised anchor shown but EXCLUDED from ρ (gray diamond);"
             " blue = lejepa family, vermillion = core SSL, orange = dino-ctrl, gray x = randinit",
             fontsize=9.5)
fig.savefig(f"{FIGD}/e13_scatter_n19.png", dpi=160)
plt.close(fig)

# --- heatmap_n19 ----------------------------------------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.8))
fig.subplots_adjust(left=0.09, right=0.97, wspace=0.30, top=0.84, bottom=0.14)
for ax, met, pb in [(axes[0], "d_read", "linear_raw_v2"), (axes[1], "d_kern", "knn_v1_k200")]:
    M = np.full((len(D_M) + 1, len(SIG_MULT)), np.nan)
    for i, dm in enumerate(D_M):
        for j, mu in enumerate(SIG_MULT):
            rho, _ = crow(met, "full19", pb, f"T=mae|D_m={dm}|s={mu}")
            M[i, j] = np.nan if rho is None else rho
    rho_raw, _ = crow(met, "full19", pb, "T=mae|D_m=0|s=-1.0")
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
    ax.set_title(f"ρ({met}, {pb}) — T=mae, n=19 (box = primary)", fontsize=10)
fig.colorbar(im, ax=axes, fraction=0.025, pad=0.02, label="Spearman ρ")
fig.savefig(f"{FIGD}/e13_cell_heatmap_n19.png", dpi=160)
plt.close(fig)

print("wrote 3 _n19 figures", flush=True)

# --- console summary ------------------------------------------------------------------------------
print("\n== n19 primary cell (full19) ==")
for met in ("d_read", "d_read_dof", "d_kern", "cka"):
    for pb in PROBES:
        print(f"  {met:11s} x {pb:14s}", crow(met, "full19", pb, PRIM_TAG))
print("== n19 battery best ==")
for pb in PROBES:
    bat = sorted([(abs(float(r["rho"])), r["ranker"], float(r["rho"]), float(r["perm_p"]))
                  for r in corr_rows if r["ranker"].startswith("battery:") and r["zoo"] == "full19"
                  and r["probe"] == pb], reverse=True)
    for a, name, rho, p in bat[:3]:
        print(f"  {pb:14s} {name:45s} rho={rho:+.3f} p={p:.4f}")
print("== n19 sub-zoos at primary ==")
for z in ("lejepa11", "nonlejepa8"):
    for met in ("d_read", "d_read_dof"):
        for pb in PROBES:
            print(f"  {z:11s} {met:11s} x {pb:14s}", crow(met, z, pb, PRIM_TAG))
print("== n19 clearing-cell count (d_read/d_read_dof, T=mae, |rho|>=.44 & p<.05) ==")
n = sum(1 for r in corr_rows if r["zoo"] == "full19" and r["cell"].startswith("T=mae")
        and r["ranker"] in ("d_read", "d_read_dof") and abs(float(r["rho"])) >= 0.44
        and float(r["perm_p"]) < 0.05)
print(f"  {n} rows")
print("== n19 T=randinit primary (P4 re-check) ==")
for pb in PROBES:
    print(f"  d_read x {pb:14s}", crow("d_read", "full19", pb, "T=randinit|D_m=1024|s=1.0"))
