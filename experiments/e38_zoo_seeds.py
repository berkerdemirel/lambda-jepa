"""E38 — the appendix treatment figure with seeds (docs/experiments/E38_in100_seeds.md).

Why: `paper_exhibits.make_treatment_fig` (fig_treatment_appendix: 7 families × 11 quantities vs
station, control grey vs + regularizer in the family colour) rests on one seed per cell. E38 lands
seeds 1 and 2 of every cell; this generator redraws the same panels with the SAME harmonization
rules (stations, getters, per-cell y range, backbone|projector rule) as a seed mean line with a
shaded band (95 % t-interval of the mean, or ±1 sample std with --band std). Seeds are dots where n
< 3 so a half-landed cell is visible as such. Berker 2026-09-19: "we will be recreating this figure
with the seeded evals (except for the methods ijepa and mae)".

Inputs per (family, arm, seed): probes `results/probes/<store>.csv`; depth metrics — seed 0 from
the paper's CSVs (keyed by the paper labels), seeds 1, 2 from `results/diag/e38/<store>_depth_metrics.csv`
(keyed by the store name); W/B from `results/diag/e23_retro_spaces.csv` (raw framing, keyed by
store; DINO reads the teacher lane as the paper does). Pure reader.
Usage: python experiments/e38_zoo_seeds.py [--band ci95|std] [--out PATH]
"""
import csv
import math
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from paper_exhibits import (CTRL, FAMILIES as PAPER_FAMILIES, GRID, MUT, STATIONS, XLABELS,  # noqa: E402
                            _spines, station_of)
from e38_seeds import FAMILIES, SEEDS, T975, store_name  # noqa: E402

ROOT = "/nfs/scistore19/locatgrp/bdemirel/ssl_project"
QUANTS = ["Linear", "kNN-200", "RankMe / d", "EffRank / d", "Gaussian KL",
          "cos margin (pos$-$rand)", "class margin", "$\\Theta$ (W/B)",
          "a (s$\\to$z)", "b (s$\\to$z)", "$\\Lambda$ (s$\\to$z)"]
ACCENT = {f[0]: f[2] for f in PAPER_FAMILIES}
PAPER_LABEL = {f[0]: (f[1][0][1], f[1][1][1]) for f in PAPER_FAMILIES}   # seed-0 depth-metric keys
TEACHER = {"DINO"}   # W/B read on the teacher lane (paper_exhibits pref map)


def load_probes(store):
    P = {}
    p = f"{ROOT}/results/probes/{store}.csv"
    if os.path.exists(p):
        for r in csv.DictReader(open(p)):
            if r["space"].startswith("student"):
                st = station_of(r["space"])
                if st:
                    P[(st, r["probe"])] = float(r["val_acc"])
    return P


def load_depth(store, seed, paper_label):
    """Rows keyed by station. Seed 0 = the paper's CSVs under the paper label; later seeds = the
    per-run E38 file, whose `ep` column carries the store name."""
    M = {}
    if seed == 0:
        files = [f"{ROOT}/results/diag/{n}.csv" for n in
                 ("e17_depth_metrics", "e20_zoo_depth_metrics", "e20_ours_depth_metrics",
                  "e20_new_arms_depth_metrics")]
        key = paper_label
    else:
        files, key = [f"{ROOT}/results/diag/e38/{store}_depth_metrics.csv"], store
    for fp in files:
        if not os.path.exists(fp):
            continue
        for r in csv.DictReader(open(fp)):
            if r["ep"] != key:
                continue
            st = station_of(r["space"])
            if st:
                M[st] = r
    return M


_RETRO = None


def load_wb(store, teacher):
    global _RETRO
    if _RETRO is None:
        _RETRO = list(csv.DictReader(open(f"{ROOT}/results/diag/e23_retro_spaces.csv")))
    pref = "teacher" if teacher else "student"
    WB = {}
    for r in _RETRO:
        if r["run"] != store or r["framing"] != "raw":
            continue
        st = station_of(r["space"])
        if st and (r["space"].startswith(pref) or st not in WB):
            WB[st] = (float(r["W"]), float(r["B"]))
    return WB


def curves(store, seed, paper_label, teacher):
    """station -> value for every quantity, one (arm, seed)."""
    P, M, WB = load_probes(store), load_depth(store, seed, paper_label), load_wb(store, teacher)
    o = WB.get("z.out")

    def m(st, key, per_d=False):
        r = M.get(st)
        v = None if r is None else r.get(key)
        if v in (None, "", "None"):
            return None
        return float(v) / float(r["d"]) if per_d else float(v)

    def diff(st, a, b):
        x, y = m(st, a), m(st, b)
        return None if x is None or y is None else x - y

    def trans(st, kind):
        if st not in WB or o is None:
            return None
        W, B = WB[st]
        a, b = (o[0] / W) ** 0.5, (o[1] / B) ** 0.5
        return {"a": a, "b": b, "lam": b / a}[kind]

    G = {"Linear": lambda st: P.get((st, "linear_raw_v2")),
         "kNN-200": lambda st: P.get((st, "knn_v1_k200")),
         "RankMe / d": lambda st: m(st, "rankme", True),
         "EffRank / d": lambda st: m(st, "effective_rank", True),
         "Gaussian KL": lambda st: m(st, "gauss_kl_total"),
         "cos margin (pos$-$rand)": lambda st: diff(st, "pos_cos", "rand_cos"),
         "class margin": lambda st: diff(st, "class_cos_same", "class_cos_diff"),
         "$\\Theta$ (W/B)": lambda st: None if st not in WB else WB[st][0] / WB[st][1],
         "a (s$\\to$z)": lambda st: trans(st, "a"),
         "b (s$\\to$z)": lambda st: trans(st, "b"),
         "$\\Lambda$ (s$\\to$z)": lambda st: trans(st, "lam")}
    return {q: {st: G[q](st) for st in STATIONS} for q in QUANTS}


def agg(vals, band):
    v = [x for x in vals if x is not None]
    if not v:
        return None, None, 0
    mean = sum(v) / len(v)
    if len(v) < 2:
        return mean, None, 1
    sd = math.sqrt(sum((x - mean) ** 2 for x in v) / (len(v) - 1))
    half = sd if band == "std" else T975[len(v)] * sd / math.sqrt(len(v))
    return mean, half, len(v)


def main():
    band = sys.argv[sys.argv.index("--band") + 1] if "--band" in sys.argv else "ci95"
    out = (sys.argv[sys.argv.index("--out") + 1] if "--out" in sys.argv
           else f"{ROOT}/results/figures/e38_treatment_appendix_seeds.png")
    nr, nc = len(FAMILIES), len(QUANTS)
    fig, axes = plt.subplots(nr, nc, figsize=(2.15 * nc, 1.55 * nr), squeeze=False)
    xs = np.arange(len(STATIONS))
    n_seen = {}
    for i, (fam, meth, ctag, ttag, _space, s0) in enumerate(FAMILIES):
        arms = [("control" if fam != "Ours" else "z-only", ctag, s0[0], PAPER_LABEL[fam][0], CTRL),
                ("treated" if fam != "Ours" else "h+z", ttag, s0[1], PAPER_LABEL[fam][1], ACCENT[fam])]
        per_arm = []
        for lab, tag, s0name, plabel, col in arms:
            per_seed = {s: curves(store_name(meth, s, tag, s0name), s, plabel, fam in TEACHER)
                        for s in SEEDS}
            per_arm.append((lab, col, per_seed))
        for j, q in enumerate(QUANTS):
            ax = axes[i][j]
            allv = []
            for k, (lab, col, per_seed) in enumerate(per_arm):
                means, halves, pts = [], [], []
                for st in STATIONS:
                    vals = [per_seed[s][q][st] for s in SEEDS]
                    mean, half, n = agg(vals, band)
                    means.append(mean), halves.append(half)
                    pts += [(STATIONS.index(st), v) for v in vals if v is not None]
                    n_seen[(fam, lab, q)] = max(n_seen.get((fam, lab, q), 0), n)
                sel = [t for t in range(len(STATIONS)) if means[t] is not None]
                if sel:
                    px = np.array(sel)
                    py = np.array([means[t] for t in sel])
                    ax.plot(px, py, "-o", ms=2.6, lw=1.5, color=col, zorder=3 + k,
                            label=lab if j == 0 else None)
                    hb = np.array([halves[t] if halves[t] is not None else 0.0 for t in sel])
                    if hb.any():
                        ax.fill_between(px, py - hb, py + hb, color=col, alpha=0.18, lw=0,
                                        zorder=2 + k)
                        allv += list(py - hb) + list(py + hb)
                    if pts and len(SEEDS) > 1:
                        ax.scatter(*zip(*pts), s=5, color=col, alpha=0.55, zorder=4 + k)
                    allv += [v for _, v in pts]
            if allv:
                lo, hi = min(allv), max(allv)
                pad = 0.10 * (hi - lo or abs(hi) or 1.0)
                ax.set_ylim(lo - pad, hi + pad)
            ax.set_xlim(-0.4, len(STATIONS) - 0.6)
            ax.set_xticks(xs)
            ax.set_xticklabels(XLABELS if i == nr - 1 else [], fontsize=6.5, rotation=45)
            ax.tick_params(axis="y", labelsize=7)
            ax.axvline(3.5, color=GRID, lw=0.8, zorder=1)
            ax.grid(color=GRID, lw=0.5, zorder=0)
            _spines(ax)
            if i == 0:
                ax.set_title(q, fontsize=9.5)
            if j == 0:
                ax.set_ylabel(fam, fontsize=10)
                ax.legend(fontsize=6.5, frameon=False, loc="best", handlelength=1.2, borderpad=0.2)
    ns = sorted({n for n in n_seen.values()})
    fig.text(0.005, -0.012,
             f"Line = mean over seeds, band = {'95% t-interval of the mean' if band == 'ci95' else '±1 sample std'}, "
             f"dots = seeds (landed n per cell: {ns}). DINO treated = winner dose (λ=.01, D-104). "
             "Vertical rule = backbone | projector boundary.", fontsize=7.5, color=MUT)
    fig.tight_layout(h_pad=0.7, w_pad=0.9)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    fig.savefig(out, dpi=150)
    print(f"[e38] wrote {out}; seeds landed per (family, arm): "
          + ", ".join(f"{f}/{a}: {max(n for (ff, aa, _), n in n_seen.items() if (ff, aa) == (f, a))}"
                      for f, a in sorted({(f, a) for f, a, _ in n_seen})))


if __name__ == "__main__":
    main()
