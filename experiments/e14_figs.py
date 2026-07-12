"""E14 figures (presentation artifacts; numbers verbatim from results/diag/e14_*.csv). Three PNGs
requested in the 2026-07-13 discussion: (1) distortion-vs-probe scatters for the four decisive
cells (mae copy/near/far + dino far), (2) held-out R² levels per member across strata — "is the
ridge predicting anything, and is the spread meaningful?", (3) strata geometry on real images
(fovea box + target box overlaid) — what copy/near/far MEAN. E13 fig conventions (Okabe-Ito,
direct labels)."""
import csv
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, "/nfs/scistore19/locatgrp/bdemirel/ssl_project")
from e13_pivot_rung0 import GATE, RANDINIT, RANKED, LEJEPA_SUB, probe_rows
from sslgap.data import _Source, foveal_boxes, foveal_iou, read_manifest
from torchvision.transforms import v2

ROOT = "/nfs/scistore19/locatgrp/bdemirel/ssl_project"
BLUE, VERM, GREEN, ORNG, GRAY = "#0072B2", "#D55E00", "#009E73", "#E69F00", "#666666"
FIGD = f"{ROOT}/results/figures/e14"
os.makedirs(FIGD, exist_ok=True)
DINO = "in100.dino.s0.ext"

short = lambda rid: rid.replace("in100.", "").replace(".ext", "").replace(".s0", "") \
    .replace("lejepa.e12", "").replace("dino-ctrl.", "ctrl-")
fam = lambda rid: (BLUE, "o") if rid in LEJEPA_SUB else \
    ((ORNG, "s") if "ctrl" in rid else ((GREEN, "D") if "deitlite" in rid else (VERM, "^")))

with open(f"{ROOT}/results/diag/e14_distortion.csv") as f:
    dist = list(csv.DictReader(f))
with open(f"{ROOT}/results/diag/e14_rank_corr.csv") as f:
    corr = list(csv.DictReader(f))
probe = {rid: probe_rows(rid, h) for rid, h in RANKED.items()}
probe[RANDINIT] = probe_rows(RANDINIT, GATE[RANDINIT])


def sym(tk, stratum, met="d_read_dof", dm="1024", mult="1.0"):
    acc = {}
    for r in dist:
        if (r["tokenizer"], r["stratum"], r["D_m"], r["sig_mult"]) == (tk, stratum, dm, mult) and r[met] != "":
            acc.setdefault(r["run"], []).append(float(r[met]))
    return {k: float(np.mean(v)) for k, v in acc.items() if len(v) == 2}


def crho(ranker, cell, zoo, pb):
    for r in corr:
        if (r["ranker"], r["cell"], r["zoo"], r["probe"]) == (ranker, cell, zoo, pb):
            return float(r["rho"]), float(r["perm_p"]), int(r["n"])
    return None, None, None


# --- Fig 1: scatters for the four decisive cells ------------------------------------------------
PANELS = [("mae", "copy", "full19"), ("mae", "near", "full19"),
          ("mae", "far", "full19"), ("dino", "far", "full19")]
fig, axes = plt.subplots(2, 2, figsize=(12.8, 10.6))
fig.subplots_adjust(left=0.07, right=0.99, wspace=0.22, hspace=0.30, top=0.92, bottom=0.06)
for ax, (tk, stratum, zoo) in zip(axes.flat, PANELS):
    vals = sym(tk, stratum)
    cell = f"T={tk}|{stratum}|D_m=1024|s=1.0"
    for rid in list(RANKED) + [RANDINIT]:
        if rid not in vals:
            continue
        selfref = (tk == "dino" and rid == DINO)
        col, mk = fam(rid) if rid in RANKED else (GRAY, "x")
        ax.plot([vals[rid]], [probe[rid]["linear_raw_v2"]], marker=mk, color=col, ms=8,
                mfc="white" if (rid == RANDINIT or selfref) else col, ls="none")
        ax.annotate(short(rid) + (" (self)" if selfref else ""), (vals[rid], probe[rid]["linear_raw_v2"]),
                    textcoords="offset points", xytext=(4, 4), fontsize=7, color="#222222")
    rl, pl, n = crho("d_read_dof", cell, zoo, "linear_raw_v2")
    rk, pk, _ = crho("d_read_dof", cell, zoo, "knn_v1_k200")
    ax.set_title(f"T={tk} · {stratum}   ρ_lin={rl:+.2f} (p={pl:.3f}, n={n})   ρ_knn={rk:+.2f} (p={pk:.3f})",
                 fontsize=10)
    ax.set_xlabel("D_read dof-matched (lower = better predicted)")
    ax.set_ylabel("linear probe val acc")
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(alpha=0.25, lw=0.5)
fig.suptitle("E14 scatters, dof-matched D_read vs linear probe — mae copy/near/far + dino far control\n"
             "(blue o = lejepa family, vermillion ^ = core, orange s = dino-ctrl, green D = deitlite"
             " [unranked anchor], gray x = randinit gate, hollow = self-ref, excluded from ρ)", fontsize=9.5)
fig.savefig(f"{FIGD}/e14_scatter.png", dpi=160)
plt.close(fig)

# --- Fig 2: held-out R² levels per member (the predictability question) -------------------------
cols_x = [("mae", "copy"), ("mae", "near"), ("mae", "far"), ("dino", "far")]
V = {c: sym(*c) for c in cols_x}
Vv = {c: sym(*c, met="d_read") for c in cols_x}          # val-selected (capacity-free) secondary
fig, ax = plt.subplots(figsize=(10.5, 6.4))
fig.subplots_adjust(left=0.08, right=0.82, top=0.90, bottom=0.10)
X = np.arange(len(cols_x))
for rid in list(RANKED) + [RANDINIT]:
    ys = [1 - V[c][rid] if rid in V[c] else np.nan for c in cols_x]
    col, _ = fam(rid) if rid in RANKED else (GRAY, "x")
    ax.plot(X[:3], ys[:3], color=col, lw=1.1, alpha=0.75, marker="o", ms=4,
            ls="--" if rid == RANDINIT else "-")
    if np.isfinite(ys[3]):
        ax.plot([X[2:4]], [ys[2:4]], color=col, lw=0.8, alpha=0.35, ls=":")
        ax.plot([X[3]], [ys[3]], color=col, marker="o", ms=5)
    ax.annotate(short(rid), (X[0] - 0.06, ys[0]), fontsize=6.5, va="center", ha="right", color=col)
ax.set_xticks(X, ["mae·copy", "mae·near", "mae·far", "dino·far"])
ax.set_xlim(-0.75, 3.35)
ax.set_ylabel("held-out R² of the ridge (1 − D_read, dof*=256)")
ax.set_title("E14: how well IS the target predicted? per-member held-out R² by stratum/tokenizer\n"
             "(lines = same member across mae strata; dashed gray = randinit; right column = dino target)",
             fontsize=10)
ax.spines[["top", "right"]].set_visible(False)
ax.grid(axis="y", alpha=0.25, lw=0.5)
fig.savefig(f"{FIGD}/e14_r2_levels.png", dpi=160)
plt.close(fig)

# summary table for the reply (printed; mirrors what the figure shows)
print("tokenizer/stratum | member held-out R2 (dof=256) min/med/max | spread(sd) | val-best R2 med")
for c in cols_x:
    r2 = np.array([1 - v for k, v in V[c].items() if k in RANKED])
    r2v = np.array([1 - v for k, v in Vv[c].items() if k in RANKED])
    print(f"  {c[0]}·{c[1]:5s}  {r2.min():.3f}/{np.median(r2):.3f}/{r2.max():.3f}   sd={r2.std():.4f}"
          f"   valbest_med={np.median(r2v):.3f}")

# --- Fig 3: what the strata MEAN, on real images -------------------------------------------------
items = read_manifest(f"{ROOT}/features/manifests/in100.pairs100.v1.csv")
order = np.random.default_rng(14).permutation(len(items))          # same picks as the preview
picked, want = [], {"far": 3, "near": 3, "copy": 3}
for ref, _ in (items[i] for i in order):
    s, _, _ = foveal_boxes(ref, "foveal_v1")
    if want.get(s, 0) > 0:
        want[s] -= 1
        picked.append((s, ref))
    if not any(want.values()):
        break
picked.sort(key=lambda t: ["copy", "near", "far"].index(t[0]))
source = _Source("imagefolder", root=os.path.expanduser("~/data/imagenet100/train"))
base_tfm = v2.Compose([v2.Resize(224), v2.CenterCrop(224)])
from sslgap.data import foveal_ctx, foveal_event

fig, axes = plt.subplots(3, 6, figsize=(15.5, 8.2))
fig.subplots_adjust(left=0.045, right=0.995, wspace=0.06, hspace=0.16, top=0.88, bottom=0.02)
for k, (stratum, ref) in enumerate(picked):
    row, blk = divmod(k, 3)
    base = base_tfm(source(ref))
    s, a, b = foveal_boxes(ref, "foveal_v1")
    axL, axR = axes[["copy", "near", "far"].index(stratum), 2 * blk], \
        axes[["copy", "near", "far"].index(stratum), 2 * blk + 1]
    axL.imshow(foveal_event(base, a, 96))
    axL.add_patch(plt.Rectangle(a, 96, 96, fill=False, edgecolor="red", lw=2))
    axL.add_patch(plt.Rectangle(b, 96, 96, fill=False, edgecolor="#00B4FF", lw=2, ls="--"))
    axR.imshow(foveal_ctx(base, b, 96))
    for spine in axR.spines.values():
        spine.set_edgecolor("#00B4FF"); spine.set_linewidth(2); spine.set_linestyle("--")
    axL.set_title(f"event_A ({stratum}, IoU={foveal_iou(a, b, 96):.2f})", fontsize=8.5)
    axR.set_title("target region, SHARP → T", fontsize=8.5)
    for ax in (axL, axR):
        ax.set_xticks([]), ax.set_yticks([])
for i, s in enumerate(["copy", "near", "far"]):
    axes[i, 0].set_ylabel(s, fontsize=11)
fig.suptitle("E14 strata on real images — LEFT of each pair: what the checkpoint sees (event_A:"
             " sharp inside RED fovea, ×4 surround elsewhere; BLUE DASHED = target region).\n"
             "RIGHT: what the tokenizer sees (that region SHARP). The ridge predicts the tokenizer"
             " descriptor of BLUE from the checkpoint's response to LEFT.\n"
             "copy: blue = red (re-encode own sharp content) · near: partial overlap · far: blue"
             " visible only at ×4 in LEFT → true peripheral prediction", fontsize=9.5)
fig.savefig(f"{FIGD}/e14_strata_geometry.png", dpi=160)
plt.close(fig)
print("wrote 3 figures to", FIGD)
