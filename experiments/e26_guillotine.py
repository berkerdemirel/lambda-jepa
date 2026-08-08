"""E26 public-zoo guillotine (Berker 2026-08-06: "i am waiting for e26 public zoo
guillotine fig") — zoo2 format restricted per D-076: rows = 6 public models, columns =
lin_v2 · knn200 · rankme/d · effrank/d · gauss_kl_full · pos_cos · rand_cos · class
margin (same−diff) · Ω; x = L03 L06 L09 cls ONLY (no head stations, no a/b/Λ; no
cross-model overlay — grey removed per Berker 2026-08-06). siglip rides gap L-taps +
pool at cls (no prefix token; card-noted).
Sources: probe CSVs where written + a stdout HARVEST of the [probe] headline lines for
the 4h-TIMEOUT probe jobs (their completed spaces are logged; the 12h top-ups
63097596/7 fill mae-L09 + dinov2-L03/06/09 — cells plot as gaps until then, script
idempotent); battery raw|full (val manifest preferred, train fallback); Ω + pos/rand
from the pub o8 stores (canonical channel); class margin on val features. Rewrites
results/diag/e26_stations.csv with the full column set. RAW; interpretation joint."""
import csv
import glob
import os
import re
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

MODELS = ["dinov2b14", "dinov3b16", "dinob16", "siglipb16", "clipb16", "maeb16"]
ST = ["L03", "L06", "L09", "cls"]
O8, VAL, TRAIN = "in1k.pairs10.v1@audit_v1.o8", "in1k.val.v1L", "in1k.train.v1L"
COLS = ["lin_v2", "knn200", "rankme/d", "effrank/d", "gauss_kl_full", "pos_cos",
        "rand_cos", "class margin (same−diff)", "Ω"]
BOUNDED = set(COLS[:4]) | {"pos_cos", "rand_cos", "class margin (same−diff)"}
PAL = ["#2e8b57", "#3d65d0", "#6a3fb5", "#0f7b8a", "#c22f2f", "#d4820a"]


def space_of(model, st):
    if st == "cls":
        return "pub.h.pool" if model == "siglipb16" else "pub.h.cls"
    tap = f"L{st[1:]}"
    kind = "gap" if model == "siglipb16" else "cls"
    return f"pub.h.{kind}.{tap}"


def probe_table():
    """(run, space, probe) -> acc from probe CSVs, overlaid by stdout harvest."""
    P = {}
    for f in glob.glob(f"{ROOT}/results/probes/in1k.pub.*.csv"):
        run = os.path.basename(f)[:-4]
        for r in csv.DictReader(open(f)):
            P[(run, r["space"], r["probe"])] = float(r["val_acc"])
    pat = re.compile(r"\[probe\] (in1k\.pub\.\S+) (\S+): (.*)")
    for f in sorted(glob.glob(f"{ROOT}/outputs/gap-probe_*.out")):
        for line in open(f, errors="ignore"):
            m = pat.match(line.strip())
            if not m:
                continue
            run, space, rest = m.groups()
            for kv in rest.split():
                k, v = kv.split("=")
                P[(run, space, k)] = float(v)
    return P


def model_stats(store, P, model):
    run = f"in1k.pub.{model}"
    B = {}
    for r in csv.DictReader(open(f"{ROOT}/results/battery/{run}.csv")):
        if r["variant"] != "raw|full":
            continue
        key = (r["space"], r["metric"])
        if r["manifest"] == VAL or key not in B:       # val preferred, train fallback
            B[key] = (float(r["value"]), float(r["d"]))
    V = store.meta(run, O8)["v"]
    yv = store.labels(run, VAL)
    out = {}
    for st in ST:
        sp = space_of(model, st)
        views = [np.asarray(store.get(run, O8, f"{sp}.view{k}"), np.float64)
                 for k in range(V)]
        e = orbit_energies(views)
        pm = pair_margin(views[0].astype(np.float32), views[1].astype(np.float32))
        out[st] = {
            "lin_v2": P.get((run, sp, "linear_raw_v2"), np.nan),
            "knn200": P.get((run, sp, "knn_v1_k200"), np.nan),
            "rankme/d": B[(sp, "rankme")][0] / B[(sp, "rankme")][1],
            "effrank/d": B[(sp, "effective_rank")][0] / B[(sp, "effective_rank")][1],
            "gauss_kl_full": B[(sp, "gauss_kl_full.total")][0],
            "pos_cos": pm["pos_cos"], "rand_cos": pm["rand_cos"],
            "class margin (same−diff)": class_margin(
                np.asarray(store.get(run, VAL, sp)), yv),
            "Ω": e["omega"]}
    n_nan = sum(1 for st in ST for q in ("lin_v2", "knn200")
                if np.isnan(out[st][q]))
    print(f"[stat] {run} ({n_nan} probe cells pending)", flush=True)
    return out


def draw(stats, path):
    fig, axes = plt.subplots(len(MODELS), len(COLS),
                             figsize=(2.55 * len(COLS), 2.0 * len(MODELS)),
                             facecolor="white")
    for i, m in enumerate(MODELS):
        for j, q in enumerate(COLS):
            ax = axes[i, j]
            ax.plot(range(len(ST)), [stats[m][s][q] for s in ST], "-o",
                    color=PAL[i], ms=3.2, lw=1.5, label=m if j == 0 else None)
            if q in BOUNDED:
                ax.set_ylim(0, 1.0)
            else:
                vs = [stats[mm][s][q] for mm in MODELS for s in ST
                      if np.isfinite(stats[mm][s][q])]
                lo, hi = min(vs), max(vs)
                pad = 0.08 * ((hi - lo) or 1)
                ax.set_ylim(max(0.0, lo - pad), hi + pad)
            ax.set_xticks(range(len(ST)))
            ax.set_xticklabels(ST if i == len(MODELS) - 1 else [], fontsize=6)
            ax.tick_params(length=0, labelsize=6)
            for s in ("top", "right"):
                ax.spines[s].set_visible(False)
            if i == 0:
                ax.set_title(q, fontsize=8.5)
            if j == 0:
                ax.set_ylabel(m, fontsize=8.5)
                ax.legend(fontsize=6, frameon=False, loc="best")
    fig.suptitle("E26 public-zoo guillotine — IN-1k val reference band (D-076: trunk "
                 "stations only, no head taps; siglip = gap L-taps + pool at cls; "
                 "Ω + gauss_kl shared y across rows, bounded 0–1 otherwise) — RAW",
                 fontsize=9.5, y=0.998)
    fig.tight_layout(rect=(0, 0.008, 1, 0.975))
    fig.savefig(path, dpi=150)
    print("wrote", path, flush=True)


def main():
    store = FeatureStore(f"{ROOT}/features")
    P = probe_table()
    stats = {m: model_stats(store, P, m) for m in MODELS}
    os.makedirs(f"{ROOT}/results/figures/e26", exist_ok=True)
    draw(stats, f"{ROOT}/results/figures/e26/e26_guillotine_zoo.png")
    with open(f"{ROOT}/results/diag/e26_stations.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["model", "station", "space"] + COLS)
        w.writeheader()
        for m in MODELS:
            for st in ST:
                w.writerow({"model": m, "station": st, "space": space_of(m, st),
                            **stats[m][st]})
    print("[write] results/diag/e26_stations.csv (full columns)", flush=True)


if __name__ == "__main__":
    main()
