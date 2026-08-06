"""E24 v-cell guillotines (Berker 2026-08-04: "i wanna see the guillotine for e24. also
for toy ones … for guillotine use the columns at our zoo2 png") — the zoo2 12-column
format on the E24 view-mean cells: rows = cells, columns = lin_v2 · knn200 · rankme/d ·
effrank/d · gauss_kl_full · pos_cos · rand_cos · class margin (same−diff) · Ω ·
a(s→z.out) · b(s→z.out) · Λ(s→z.out); x = L03 L06 L09 cls tap1 tap2 z.out (dotted =
trunk|head). Grey context: toy = the other cells (in1k_guillotine_3cell of-record
grammar); in100 = the E21 ring anchor d256vm4 as the ONLY shaded curve on every row
(Berker 2026-08-04: "so that the diff is obvious") — its battery refired through the
standard audit so grey and colored ride one pipeline. y 0–1 for bounded, ranks ÷
station d; gauss_kl + Ω·a·b·Λ per-ROW y (zoo2 rule — unbounded, within-space reading;
lims cover the grey reference too). Sources: landing probes/battery; Ω/a/b/Λ +
pos/rand computed HERE from the o8 orbit stores (orbit_energies over all 8 views —
canonical channel, no diag-CSV dependency); class margin on val features
(metrics.pairs.class_margin). Stations CSV → results/diag/e24_stations.csv (full
rewrite of the frames run). RAW; interpretation joint."""
import csv
import sys

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = "/nfs/scistore19/locatgrp/bdemirel/ssl_project"
sys.path.insert(0, ROOT)
from sslgap.extract.store import FeatureStore                    # noqa: E402
from sslgap.metrics.orbit_energy import orbit_energies           # noqa: E402
from sslgap.metrics.pairs import class_margin, pair_margin       # noqa: E402

ST = ["L03", "L06", "L09", "cls", "tap1", "tap2", "z.out"]
SPACE = {"L03": "student.h.cls.L03", "L06": "student.h.cls.L06", "L09": "student.h.cls.L09",
         "cls": "student.h.cls", "tap1": "student.z.proj.tap1", "tap2": "student.z.proj.tap2",
         "z.out": "student.z.proj.out"}
COLS = ["lin_v2", "knn200", "rankme/d", "effrank/d", "gauss_kl_full", "pos_cos",
        "rand_cos", "class margin (same−diff)", "Ω", "a(s→z.out)", "b(s→z.out)",
        "Λ(s→z.out)"]
BOUNDED = set(COLS[:4]) | {"pos_cos", "rand_cos", "class margin (same−diff)"}
PAL = ["#2e8b57", "#3d65d0", "#6a3fb5", "#0f7b8a", "#c22f2f", "#d4820a",
       "#8a5cb8", "#b0308a", "#5c7a29"]
FRAMES = {
    "toy": {"tags": ["v4oas", "va", "vb", "vc", "vh0", "vac", "v1", "v2lr", "v3nq"],
            "run": "toy.floorssl.s0.e24{t}.extL",
            "o8": "imagenette.train.v1@audit_v1.o8", "val": "imagenette.val.v1L",
            "title": "E24 toy view-mean guillotine — OAS map (v4oas/va/vb/vc/vh0/vac) + "
                     "ring/bs512 variants (v1/v2lr/v3nq)"},
    "in100": {"tags": ["va", "vb", "vc", "vh0", "vac", "vcc"],
              "run": "in100.floorssl.s0.e24{t}.extL",
              "o8": "in100.pairs100.v1@audit_v1.o8", "val": "in100.val.v1L",
              "ref": ("vm4 ring anchor", "in100.floorssl.s0.d256vm4.extL"),
              "title": "E24 in100 view-mean guillotine — mirrors (va/vb/vc/vh0) + "
                       "corrected twins (vac wall / vcc high-z) vs the E21 ring "
                       "anchor d256vm4 (grey, the only shaded curve)"},
}


def cell_stats(store, run, o8, val):
    P = {(r["space"], r["probe"]): float(r["val_acc"])
         for r in csv.DictReader(open(f"{ROOT}/results/probes/{run}.csv"))}
    B = {(r["space"], r["metric"]): (float(r["value"]), float(r["d"]))
         for r in csv.DictReader(open(f"{ROOT}/results/battery/{run}.csv"))
         if r["variant"] == "raw|full"}
    V = store.meta(run, o8)["v"]
    yv = store.labels(run, val)
    E = {}
    for st in ST:
        sp = SPACE[st]
        views = [np.asarray(store.get(run, o8, f"{sp}.view{k}"), np.float64)
                 for k in range(V)]
        E[st] = orbit_energies(views)
    Wo, Bo = E["z.out"]["W"], E["z.out"]["B"]
    out = {}
    for st in ST:
        sp = SPACE[st]
        pm = pair_margin(np.asarray(store.get(run, o8, f"{sp}.view0"), np.float32),
                         np.asarray(store.get(run, o8, f"{sp}.view1"), np.float32))
        W, Bc = E[st]["W"], E[st]["B"]
        out[st] = {
            "lin_v2": P[(sp, "linear_raw_v2")], "knn200": P[(sp, "knn_v1_k200")],
            "rankme/d": B[(sp, "rankme")][0] / B[(sp, "rankme")][1],
            "effrank/d": B[(sp, "effective_rank")][0] / B[(sp, "effective_rank")][1],
            "gauss_kl_full": B[(sp, "gauss_kl_full.total")][0],
            "pos_cos": pm["pos_cos"], "rand_cos": pm["rand_cos"],
            "class margin (same−diff)": class_margin(
                np.asarray(store.get(run, val, sp)), yv),
            "Ω": W / Bc,
            "a(s→z.out)": (Wo / W) ** 0.5, "b(s→z.out)": (Bo / Bc) ** 0.5,
            "Λ(s→z.out)": ((Bo / Bc) / (Wo / W)) ** 0.5}
    print(f"[stat] {run} ({len(out)} stations)", flush=True)
    return out


def draw(frame, cfg, stats, path, ref=None):
    tags = cfg["tags"]
    fig, axes = plt.subplots(len(tags), len(COLS),
                             figsize=(2.55 * len(COLS), 2.0 * len(tags)),
                             facecolor="white")
    for i, tag in enumerate(tags):
        for j, q in enumerate(COLS):
            ax = axes[i, j]
            if ref is not None:                            # single grey reference
                rlab, rstats = ref
                ax.plot(range(len(ST)), [rstats[s][q] for s in ST], "-o",
                        color="#8c8c8c", ms=2.6, lw=1.1,
                        label=rlab if j == 0 else None)
            else:
                for t2 in tags:                            # grey context = the others
                    if t2 != tag:
                        ax.plot(range(len(ST)), [stats[t2][s][q] for s in ST],
                                "-o", color="#c9c9c9", ms=2.0, lw=0.8)
            ax.plot(range(len(ST)), [stats[tag][s][q] for s in ST], "-o",
                    color=PAL[i % len(PAL)], ms=3.2, lw=1.5,
                    label=tag if j == 0 else None)
            ax.axvline(3.5, color="#bbbbbb", lw=0.8, ls=":")
            if q in BOUNDED:
                ax.set_ylim(0, 1.0)
            else:                                          # zoo2 rule: per-row y
                vs = [stats[tag][s][q] for s in ST]
                if ref is not None:
                    vs = vs + [ref[1][s][q] for s in ST]
                lo, hi = min(vs), max(vs)
                pad = 0.08 * ((hi - lo) or 1)
                ax.set_ylim(max(0.0, lo - pad), hi + pad)
            ax.set_xticks(range(len(ST)))
            ax.set_xticklabels(ST if i == len(tags) - 1 else [], fontsize=5.6,
                               rotation=45)
            ax.tick_params(length=0, labelsize=6)
            for s in ("top", "right"):
                ax.spines[s].set_visible(False)
            if i == 0:
                ax.set_title(q, fontsize=8.5)
            if j == 0:
                ax.set_ylabel(tag, fontsize=8.5)
                ax.legend(fontsize=6, frameon=False, loc="best")
    grey_note = (f"grey = {ref[0]}" if ref is not None else "grey = the others")
    fig.suptitle(f"{cfg['title']} — rows = cells ({grey_note}); y 0–1 bounded, "
                 f"ranks ÷ station d, gauss_kl + Ω·a·b·Λ per-row y (o8 calculus, "
                 f"station→z.out remaining-path); dotted = trunk|head — RAW",
                 fontsize=9.5, y=0.998)
    fig.tight_layout(rect=(0, 0.008, 1, 0.975))
    fig.savefig(path, dpi=150)
    print("wrote", path, flush=True)


def main():
    which = sys.argv[1] if len(sys.argv) > 1 else "both"
    frames = ["toy", "in100"] if which == "both" else [which]
    store = FeatureStore(f"{ROOT}/features")
    rows = []
    for frame in frames:
        cfg = FRAMES[frame]
        stats = {t: cell_stats(store, cfg["run"].format(t=t), cfg["o8"], cfg["val"])
                 for t in cfg["tags"]}
        ref = None
        if "ref" in cfg:
            rlab, rrun = cfg["ref"]
            ref = (rlab, cell_stats(store, rrun, cfg["o8"], cfg["val"]))
            for st in ST:
                rows.append({"frame": frame, "run": rrun.split(".")[-2] + "(ref)",
                             "station": st, **ref[1][st]})
        for t in cfg["tags"]:
            for st in ST:
                rows.append({"frame": frame, "run": t, "station": st, **stats[t][st]})
        draw(frame, cfg, stats,
             f"{ROOT}/results/figures/e24/e24_guillotine_{frame}.png", ref=ref)
    with open(f"{ROOT}/results/diag/e24_stations.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["frame", "run", "station"] + COLS)
        w.writeheader()
        w.writerows(rows)
    print(f"[write] results/diag/e24_stations.csv ({len(rows)} rows)", flush=True)


if __name__ == "__main__":
    main()
