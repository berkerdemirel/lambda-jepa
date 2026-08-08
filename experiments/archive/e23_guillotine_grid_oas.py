"""E23 stage-C′ grid guillotines — the vm-OAS share-pinned capacity grid (D-069 as
amended; Berker 2026-08-06: 'we do not have grid width and grid depth guillotine curves
postfix (oas with the toy)'): WIDTH ladder (K=2, hidden 32→512) and DEPTH ladder
(hidden 2048, K 0→6) in the stage-C figure grammar; the 2048/K2 row = the landed E24
v-cell references (vc solid — the C′ dose-basis cell; v4oas + vb thin dashed — the
flat-basin spread; vh0 GREY = the h0 reference; K6/W64 rows carry their own h0 twins
grey). Columns LABEL-FREE (D-068 addendum): lin · knn · rankme/d · effrank/d ·
gauss_kl · pos/rand cos · Ω · a·b·Λ station→z.out. Sources: re-landed chain probes/
battery + e23c_grid_spaces.csv + e24_toy_spaces.csv (v-cells) + pair_margin on o8.
Stations CSV → results/diag/e23c_grid_stations.csv. RAW; no takeaway here."""
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

C2 = ["e23cW32", "e23cW64", "e23cW128", "e23cW256", "e23cW512", "e23cK0", "e23cK1",
      "e23cK3", "e23cK4", "e23cK6", "e23cW64h0", "e23cK6h0"]
VREF = ["e24vc", "e24v4oas", "e24vb", "e24vh0"]
RID = {t: f"toy.floorssl.s0.{t}.extL" for t in C2 + VREF}
O8 = "imagenette.train.v1@audit_v1.o8"
ST = ["L03", "L06", "L09", "cls"] + [f"tap{k}" for k in range(1, 7)] + ["z.out"]
SPACE = {**{k: f"student.h.cls.{k}" for k in ("L03", "L06", "L09")}, "cls": "student.h.cls",
         **{f"tap{k}": f"student.z.proj.tap{k}" for k in range(1, 7)},
         "z.out": "student.z.proj.out"}
COLS = ["lin_v2", "knn200", "rankme/d", "effrank/d", "gauss_kl_full", "pos_cos",
        "rand_cos", "Ω", "a(s→z.out)", "b(s→z.out)", "Λ(s→z.out)"]
BOUNDED = set(COLS[:4]) | {"pos_cos", "rand_cos"}
# figure rows: label, main tag, color — ONE line per row, no grey overlays
# (Berker 2026-08-06: "i want no gray curves on these figures"; h0 twins + the
# v4oas/vb basin refs stay in e23c_grid_stations.csv, tables only)
WIDTH = [("W32", "e23cW32", "#3d65d0"),
         ("W64", "e23cW64", "#6a3fb5"),
         ("W128", "e23cW128", "#0f7b8a"),
         ("W256", "e23cW256", "#c22f2f"),
         ("W512", "e23cW512", "#d4820a"),
         ("W2048=vc", "e24vc", "#2e8b57")]
DEPTH = [("K0", "e23cK0", "#3d65d0"),
         ("K1", "e23cK1", "#6a3fb5"),
         ("K2=vc", "e24vc", "#2e8b57"),
         ("K3", "e23cK3", "#0f7b8a"),
         ("K4", "e23cK4", "#c22f2f"),
         ("K6", "e23cK6", "#d4820a")]
WBSRC = ["results/diag/e23c_grid_spaces.csv", "results/diag/e24_toy_spaces.csv"]


def load_wb():
    wb = {}
    for src in WBSRC:
        for r in csv.DictReader(open(f"{ROOT}/{src}")):
            if r["framing"] == "raw":
                wb[(r["run"], r["space"])] = (float(r["W"]), float(r["B"]))
    return wb


def cell_stats(store, run, wb):
    P = {(r["space"], r["probe"]): float(r["val_acc"])
         for r in csv.DictReader(open(f"{ROOT}/results/probes/{run}.csv"))}
    B = {(r["space"], r["metric"]): (float(r["value"]), float(r["d"]))
         for r in csv.DictReader(open(f"{ROOT}/results/battery/{run}.csv"))
         if r["variant"] == "raw|full"}
    Wo, Bo = wb[(run, SPACE["z.out"])]
    out = {}
    for st in ST:
        sp = SPACE[st]
        if (sp, "linear_raw_v2") not in P or (run, sp) not in wb:
            continue
        pm = pair_margin(np.asarray(store.get(run, O8, f"{sp}.view0"), np.float32),
                         np.asarray(store.get(run, O8, f"{sp}.view1"), np.float32))
        out[st] = {
            "lin_v2": P[(sp, "linear_raw_v2")], "knn200": P[(sp, "knn_v1_k200")],
            "rankme/d": B[(sp, "rankme")][0] / B[(sp, "rankme")][1],
            "effrank/d": B[(sp, "effective_rank")][0] / B[(sp, "effective_rank")][1],
            "gauss_kl_full": B[(sp, "gauss_kl_full.total")][0],
            "pos_cos": pm["pos_cos"], "rand_cos": pm["rand_cos"],
            "Ω": wb[(run, sp)][0] / wb[(run, sp)][1],
            "a(s→z.out)": (Wo / wb[(run, sp)][0]) ** 0.5,
            "b(s→z.out)": (Bo / wb[(run, sp)][1]) ** 0.5,
            "Λ(s→z.out)": ((Bo / wb[(run, sp)][1]) / (Wo / wb[(run, sp)][0])) ** 0.5}
    print(f"[stat] {run} ({len(out)} stations)", flush=True)
    return out


def draw(fig_rows, stats, title, path):
    fig, axes = plt.subplots(len(fig_rows), len(COLS),
                             figsize=(2.45 * len(COLS), 2.0 * len(fig_rows)),
                             facecolor="white")
    for i, (lab, main, c) in enumerate(fig_rows):
        for j, q in enumerate(COLS):
            ax = axes[i, j]
            xs = [k for k, s in enumerate(ST) if s in stats[main]]
            ax.plot(xs, [stats[main][ST[k]][q] for k in xs], "-", marker="o",
                    color=c, ms=2.4, lw=1.2, label=main if j == 0 else None)
            if q in BOUNDED:
                ax.set_ylim(0, 1.0)
            else:
                vs = [stats[main][s][q] for s in stats[main]]
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
                ax.legend(fontsize=5.0, frameon=False, loc="best")
    fig.suptitle(title, fontsize=9.5, y=0.998)
    fig.tight_layout(rect=(0, 0.008, 1, 0.97))
    fig.savefig(path, dpi=150)
    print("wrote", path, flush=True)


def main():
    store = FeatureStore(f"{ROOT}/features")
    wb = load_wb()
    stats = {t: cell_stats(store, RID[t], wb) for t in C2 + VREF}
    with open(f"{ROOT}/results/diag/e23c_grid_stations.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["run", "station"] + COLS)
        w.writeheader()
        for t in C2 + VREF:
            for st in ST:
                if st in stats[t]:
                    w.writerow({"run": t, "station": st, **stats[t][st]})
    base = ("vm-OAS share-pinned grid (D-069 as amended); 2048/K2 row = the landed E24 "
            "vc cell (the C′ dose basis); one line per row, no overlays (h0 twins + "
            "v4oas/vb refs in e23c_grid_stations.csv); LABEL-FREE columns; y 0–1 "
            "bounded, ranks ÷ station d, gauss_kl + Ω·a·b·Λ per-row y (station→z.out "
            "remaining-path, o8 calculus) — RAW")
    draw(WIDTH, stats, f"E23 stage-C′ WIDTH ladder guillotine (K=2, hidden 32→2048) — {base}",
         f"{ROOT}/results/figures/e23/e23_guillotine_grid_width_oas.png")
    draw(DEPTH, stats, f"E23 stage-C′ DEPTH ladder guillotine (hidden 2048, K 0→6) — {base}",
         f"{ROOT}/results/figures/e23/e23_guillotine_grid_depth_oas.png")


if __name__ == "__main__":
    main()
