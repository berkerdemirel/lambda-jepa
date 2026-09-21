"""E38 — seed statistics for the paper's IN-100 controlled pairs (docs/experiments/E38_in100_seeds.md).

Why: the paper's Section "Controlled experiments on ImageNet-100" rested on one seed per cell.
E38 repeats every cell (control and + regularizer) at seeds 1 and 2; this script turns the landed
probe CSVs into the seed table: per-seed values, mean, sample std, the 95 % t-interval of the mean
(n = number of landed seeds), and the PAIRED treated − control delta per seed (seeds are matched
across the pair, so the paired interval is the honest one for "the effect"). Readers = the paper's
(linear_raw_v2, knn_v1_k200 at the retained representation: the 512-d embedding for LeJEPA and
VISReg, the trunk CLS otherwise — paper_exhibits.H_STATION). Seed-0 rows are the existing paper
cells (their historical store names are listed below); seeds 1, 2 land as `in100.<m>.s<k>[.<tag>].extL`.

Outputs: results/e38/e38_seeds.csv (long), results/e38/e38_summary.csv, docs/paper/blocks/
e38-seed-table.tex, results/figures/e38_seed_bars.png. Pure reader: nothing here trains or extracts.
"""
import csv
import math
import os
import sys

ROOT = "/nfs/scistore19/locatgrp/bdemirel/ssl_project"
PROBES = ("linear_raw_v2", "knn_v1_k200")
SEEDS = (0, 1, 2)
# family, method, control tag, treated tag, station (space suffix), seed-0 store names (ctrl, treat)
FAMILIES = [
    ("LeJEPA", "lejepa", "", "e20f", "student.z.embed",
     ("in100.lejepa.s0.e17c.ext", "in100.lejepa.s0.e20f.ep100.extL")),
    ("VICReg", "vicreg", "", "e20f", "student.h.cls",
     ("in100.vicreg.s0.e17c.ext", "in100.vicreg.s0.e20f.extL")),
    ("SimCLR", "simclr", "", "e20f", "student.h.cls",
     ("in100.simclr.s0.e17c.ext", "in100.simclr.s0.e20f.extL")),
    ("DINO", "dino", "", "e20fwlo", "student.h.cls",
     ("in100.dino.s0.e17c.ext", "in100.dino.s0.e20fwlo.extL")),
    ("BYOL", "byol", "", "e20f", "student.h.cls",
     ("in100.byol.s0.e17c.ext", "in100.byol.s0.e20f.extL")),
    ("VISReg", "visreg", "", "visregf", "student.z.embed",
     ("in100.visreg.s0.extL", "in100.visreg.s0.visregf.extL")),
    ("Ours", "floorssl", "d256vm4zonly", "d256vm4", "student.h.cls",   # run-id string of the existing stores (the method key is lambdajepa since D-127)
     ("in100.floorssl.s0.d256vm4zonly.extL", "in100.floorssl.s0.d256vm4.extL")),
]
T975 = {1: float("nan"), 2: 12.706, 3: 4.303, 4: 3.182, 5: 2.776}   # two-sided 95 % t quantiles


def store_name(method, seed, tag, s0_name):
    if seed == 0:
        return s0_name
    return f"in100.{method}.s{seed}" + (f".{tag}" if tag else "") + ".extL"


def read_probe(store, space, probe):
    p = f"{ROOT}/results/probes/{store}.csv"
    if not os.path.exists(p):
        return None
    for r in csv.DictReader(open(p)):
        if r["space"] == space and r["probe"] == probe:
            return 100 * float(r["val_acc"])
    return None


def stats(vals):
    v = [x for x in vals if x is not None]
    n = len(v)
    if n == 0:
        return dict(n=0, mean=None, std=None, ci=None)
    m = sum(v) / n
    sd = math.sqrt(sum((x - m) ** 2 for x in v) / (n - 1)) if n > 1 else None
    ci = T975[n] * sd / math.sqrt(n) if n > 1 else None
    return dict(n=n, mean=m, std=sd, ci=ci)


def collect():
    long, summary = [], []
    for fam, meth, ctag, ttag, space, s0 in FAMILIES:
        for probe in PROBES:
            per = {"control": [], "treated": [], "delta": []}
            for s in SEEDS:
                c = read_probe(store_name(meth, s, ctag, s0[0]), space, probe)
                t = read_probe(store_name(meth, s, ttag, s0[1]), space, probe)
                d = None if c is None or t is None else t - c
                per["control"].append(c), per["treated"].append(t), per["delta"].append(d)
                long.append(dict(family=fam, probe=probe, seed=s, station=space,
                                 control=c, treated=t, delta=d))
            row = dict(family=fam, probe=probe, station=space)
            for arm in ("control", "treated", "delta"):
                st = stats(per[arm])
                row.update({f"{arm}_{k}": v for k, v in st.items()})
            summary.append(row)
    return long, summary


def fmt(m, ci):
    if m is None:
        return "---"
    return f"{m:.1f}" if ci is None else f"{m:.1f} $\\pm$ {ci:.1f}"


def tex(summary):
    """One table: per family, control / + regularizer / paired delta for linear and kNN, each
    mean ± 95 % t-interval half-width over the landed seeds (n in the last column)."""
    by = {(r["family"], r["probe"]): r for r in summary}
    lines = ["\\begin{tabular}{lrrrrrrr}", "\\toprule",
             "Method & \\multicolumn{3}{c}{Linear} & \\multicolumn{3}{c}{kNN-200} & $n$ \\\\",
             "\\cmidrule(lr){2-4}\\cmidrule(lr){5-7}",
             " & control & + reg. & $\\Delta$ & control & + reg. & $\\Delta$ & \\\\", "\\midrule"]
    for fam, *_ in FAMILIES:
        cells, n = [], 0
        for probe in PROBES:
            r = by[(fam, probe)]
            cells += [fmt(r["control_mean"], r["control_ci"]), fmt(r["treated_mean"], r["treated_ci"]),
                      fmt(r["delta_mean"], r["delta_ci"])]
            n = min(r["control_n"], r["treated_n"])
        lines.append(f"{fam} & " + " & ".join(cells) + f" & {n} \\\\")
    lines += ["\\bottomrule", "\\end{tabular}"]
    os.makedirs(f"{ROOT}/docs/paper/blocks", exist_ok=True)
    with open(f"{ROOT}/docs/paper/blocks/e38-seed-table.tex", "w") as f:
        f.write("% generated by experiments/e38_seeds.py — mean ± 95% t-CI over landed seeds\n")
        f.write("\n".join(lines) + "\n")
    return lines


def figure(summary):
    """Two panels (linear, kNN), one bar pair per family: control (grey) vs + regularizer
    (family colour), error bar = 95 % t-interval of the mean over the landed seeds; seed points
    overlaid so n is visible. Known units (top-1 %), no derived axes."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    long, _ = collect()
    color = {"LeJEPA": "#2e8b57", "VICReg": "#c23b3b", "SimCLR": "#3d65d0", "DINO": "#d4820a",
             "BYOL": "#6a3fb5", "VISReg": "#b3477d", "Ours": "#008300"}
    fams = [f[0] for f in FAMILIES]
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.6), sharey=False)
    for ax, probe, title in zip(axes, PROBES, ("IN-100 linear top-1 (%)", "IN-100 kNN-200 top-1 (%)")):
        for i, fam in enumerate(fams):
            r = next(s for s in summary if s["family"] == fam and s["probe"] == probe)
            pts = [x for x in long if x["family"] == fam and x["probe"] == probe]
            for j, (arm, col) in enumerate((("control", "#9a9a9a"), ("treated", color[fam]))):
                if r[f"{arm}_mean"] is None:
                    continue
                x = i + (-0.2 if j == 0 else 0.2)
                ax.bar(x, r[f"{arm}_mean"], width=0.38, color=col, alpha=0.9,
                       yerr=r[f"{arm}_ci"] or 0, capsize=3, error_kw=dict(lw=1))
                ys = [p[arm] for p in pts if p[arm] is not None]
                ax.scatter([x] * len(ys), ys, s=9, color="k", zorder=3)
        ax.set_xticks(range(len(fams)), fams, fontsize=9)
        ax.set_title(title, fontsize=10)
        lo = min(v for s in summary if s["probe"] == probe for v in (s["control_mean"], s["treated_mean"]) if v)
        ax.set_ylim(max(0, lo - 8), None)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
    axes[0].text(0.01, 0.98, "grey = control, colour = + backbone regularizer; bar = mean, whisker = 95% CI, dots = seeds",
                 transform=axes[0].transAxes, fontsize=7, va="top")
    fig.tight_layout()
    os.makedirs(f"{ROOT}/results/figures", exist_ok=True)
    fig.savefig(f"{ROOT}/results/figures/e38_seed_bars.png", dpi=160)
    print("[e38] wrote results/figures/e38_seed_bars.png")


def main():
    long, summary = collect()
    os.makedirs(f"{ROOT}/results/e38", exist_ok=True)
    with open(f"{ROOT}/results/e38/e38_seeds.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(long[0].keys()))
        w.writeheader(), w.writerows(long)
    with open(f"{ROOT}/results/e38/e38_summary.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(summary[0].keys()))
        w.writeheader(), w.writerows(summary)
    for ln in tex(summary):
        print(ln)
    if "--fig" in sys.argv:
        figure(summary)


if __name__ == "__main__":
    main()
