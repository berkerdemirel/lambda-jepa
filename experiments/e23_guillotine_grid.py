"""E23 stage-C grid guillotines (Berker 2026-08-03: "we also repeated all the toy
experiments to see these metrics — i don't see the guillotine figure"): two figures in the
e23_guillotine_1k format — WIDTH ladder (K=2, hidden 32→2048) and DEPTH ladder (hidden
2048, K 0→6) — rows = cells, h0 twins overlaid GREY on their h.65 row (W64/W64h0,
Llr/K2h0, K6/K6h0; the zoo2 ctrl-vs-arm grammar). Columns are LABEL-FREE per Berker's
2026-08-03 ruling (no class margin; labels enter only through the probe columns):
lin · knn · rankme/d · effrank/d · gauss_kl · pos/rand cos · Ω · a·b·Λ station→z.out
(remaining-path). Ω/a/b/Λ + gauss_kl y-lims per ROW (unbounded ratios, within-space
reading — E20-T4 session). Sources: landing-chain probes/battery + e23_grid_spaces.csv
(D-068 instruments) + pair_margin on the o8 stores. Stations CSV for the Block-2 table →
results/diag/e23_grid_stations.csv. RAW; no takeaway here."""
import csv
import sys

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = "/nfs/scistore19/locatgrp/bdemirel/ssl_project"
sys.path.insert(0, ROOT)
from sslgap.extract.store import FeatureStore          # noqa: E402
from sslgap.metrics.pairs import pair_margin           # noqa: E402

TAGS = ["e23jW32", "e23jW64", "e23jW128", "e23jW256", "e23jW512", "e23Llr",
        "e23jK0", "e23jK1", "e23jK3", "e23jK4", "e23jK6",
        "e23jW64h0", "e23jK2h0", "e23jK6h0"]
RID = {t: f"toy.floorssl.s0.{t}.extL" for t in TAGS}
O8 = "imagenette.train.v1@audit_v1.o8"
ST = ["L03", "L06", "L09", "cls"] + [f"tap{k}" for k in range(1, 7)] + ["z.out"]
SPACE = {**{k: f"student.h.cls.{k}" for k in ("L03", "L06", "L09")}, "cls": "student.h.cls",
         **{f"tap{k}": f"student.z.proj.tap{k}" for k in range(1, 7)},
         "z.out": "student.z.proj.out"}
COLS = ["lin_v2", "knn200", "rankme/d", "effrank/d", "gauss_kl_full", "pos_cos",
        "rand_cos", "Ω", "a(s→z.out)", "b(s→z.out)", "Λ(s→z.out)"]
BOUNDED = set(COLS[:4]) | {"pos_cos", "rand_cos"}
ROWLIM = {"gauss_kl_full", "Ω", "a(s→z.out)", "b(s→z.out)", "Λ(s→z.out)"}
# figure rows: (label, main tag, twin tag or None, color)
WIDTH = [("W32", "e23jW32", None, "#3d65d0"), ("W64", "e23jW64", "e23jW64h0", "#6a3fb5"),
         ("W128", "e23jW128", None, "#0f7b8a"), ("W256", "e23jW256", None, "#c22f2f"),
         ("W512", "e23jW512", None, "#d4820a"), ("W2048=Llr", "e23Llr", "e23jK2h0", "#2e8b57")]
DEPTH = [("K0", "e23jK0", None, "#3d65d0"), ("K1", "e23jK1", None, "#6a3fb5"),
         ("K2=Llr", "e23Llr", "e23jK2h0", "#2e8b57"), ("K3", "e23jK3", None, "#0f7b8a"),
         ("K4", "e23jK4", None, "#c22f2f"), ("K6", "e23jK6", "e23jK6h0", "#d4820a")]


def cell_stats(store, run):
    P = {(r["space"], r["probe"]): float(r["val_acc"])
         for r in csv.DictReader(open(f"{ROOT}/results/probes/{run}.csv"))}
    B = {(r["space"], r["metric"]): (float(r["value"]), float(r["d"]))
         for r in csv.DictReader(open(f"{ROOT}/results/battery/{run}.csv"))
         if r["variant"] == "raw|full"}
    WB = {r["space"]: (float(r["W"]), float(r["B"]))
          for r in csv.DictReader(open(f"{ROOT}/results/diag/e23_grid_spaces.csv"))
          if r["run"] == run and r["framing"] == "raw"}
    Wo, Bo = WB[SPACE["z.out"]]
    out = {}
    for st in ST:
        sp = SPACE[st]
        if (sp, "linear_raw_v2") not in P:
            continue
        pm = pair_margin(np.asarray(store.get(run, O8, f"{sp}.view0"), np.float32),
                         np.asarray(store.get(run, O8, f"{sp}.view1"), np.float32))
        out[st] = {
            "lin_v2": P[(sp, "linear_raw_v2")], "knn200": P[(sp, "knn_v1_k200")],
            "rankme/d": B[(sp, "rankme")][0] / B[(sp, "rankme")][1],
            "effrank/d": B[(sp, "effective_rank")][0] / B[(sp, "effective_rank")][1],
            "gauss_kl_full": B[(sp, "gauss_kl_full.total")][0],
            "pos_cos": pm["pos_cos"], "rand_cos": pm["rand_cos"],
            "Ω": WB[sp][0] / WB[sp][1],
            "a(s→z.out)": (Wo / WB[sp][0]) ** 0.5,
            "b(s→z.out)": (Bo / WB[sp][1]) ** 0.5,
            "Λ(s→z.out)": ((Bo / WB[sp][1]) / (Wo / WB[sp][0])) ** 0.5}
    print(f"[stat] {run} ({len(out)} stations)", flush=True)
    return out


def draw(fig_rows, stats, title, path):
    fig, axes = plt.subplots(len(fig_rows), len(COLS),
                             figsize=(2.45 * len(COLS), 2.0 * len(fig_rows)),
                             facecolor="white")
    for i, (lab, main, twin, c) in enumerate(fig_rows):
        members = [(twin, "#9a9a9a")] * bool(twin) + [(main, c)]
        for j, q in enumerate(COLS):
            ax = axes[i, j]
            for tag, col in members:
                xs = [k for k, s in enumerate(ST) if s in stats[tag]]
                ax.plot(xs, [stats[tag][ST[k]][q] for k in xs], "-o", color=col,
                        ms=2.8, lw=1.2, label=tag if j == 0 else None)
            if q in BOUNDED:
                ax.set_ylim(0, 1.0)
            else:
                vs = [stats[t][s][q] for t, _ in members for s in stats[t]]
                lo, hi = min(vs), max(vs)
                pad = 0.06 * ((hi - lo) or 1)
                ax.set_ylim(max(0.0, lo - pad), hi + pad)
            ax.axvline(3.5, color="#bbbbbb", lw=0.8, ls=":")
            ax.set_xticks(range(len(ST)))
            ax.set_xticklabels(ST if i == len(fig_rows) - 1 else [], fontsize=5.4,
                               rotation=45)
            ax.tick_params(length=0, labelsize=6)
            for s in ("top", "right"):
                ax.spines[s].set_visible(False)
            if i == 0:
                ax.set_title(q, fontsize=8.5)
            if j == 0:
                ax.set_ylabel(lab, fontsize=8.5)
                ax.legend(fontsize=5.4, frameon=False, loc="best")
    fig.suptitle(title, fontsize=9.5, y=0.998)
    fig.tight_layout(rect=(0, 0.008, 1, 0.97))
    fig.savefig(path, dpi=150)
    print("wrote", path, flush=True)


def main():
    store = FeatureStore(f"{ROOT}/features")
    stats = {t: cell_stats(store, RID[t]) for t in TAGS}
    with open(f"{ROOT}/results/diag/e23_grid_stations.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["run", "station"] + COLS)
        w.writeheader()
        for t in TAGS:
            for st in ST:
                if st in stats[t]:
                    w.writerow({"run": t, "station": st, **stats[t][st]})
    base = ("rows = cells, h0 twin GREY on its h.65 row; LABEL-FREE columns (no class "
            "margin — Berker 2026-08-03); y 0–1 bounded, ranks ÷ station d, gauss_kl + "
            "Ω·a·b·Λ per-row y (station→z.out remaining-path, o8 calculus) — RAW")
    draw(WIDTH, stats, f"E23 stage-C WIDTH ladder guillotine (K=2, hidden 32→2048) — {base}",
         f"{ROOT}/results/figures/e23/e23_guillotine_grid_width.png")
    draw(DEPTH, stats, f"E23 stage-C DEPTH ladder guillotine (hidden 2048, K 0→6) — {base}",
         f"{ROOT}/results/figures/e23/e23_guillotine_grid_depth.png")


if __name__ == "__main__":
    main()
