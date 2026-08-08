"""IN-1k 3-cell guillotine in the e21_guillotine_vm OF-RECORD format (Berker 2026-07-30):
rows = vm2/vm3/vm4 (other two cells as light-grey reference per row), columns = lin_v2 ·
knn200 · rankme/d · effrank/d · gauss_kl_full · pos_cos · rand_cos · class margin
(same−diff); x = L03 L06 L09 cls tap1 tap2 z.out; dotted = trunk|head boundary; y fixed
0-1 for bounded quantities, ranks ÷ station dimension. Sources (all D-066 standard frame):
probes = full-1.28M-train fits; rankme/effrank/gauss_kl = 50k-val battery; pos/rand =
pair_margin on the o8 orbit stores (view0/view1, in1k.pairs10.v1@audit_v1.o8, L-taps
included); class margin = label-conditioned mean cosines on the 50k val features
(class-sum identity). Station stats also land in results/diag/e23_1k_stations.csv. RAW."""
import csv
import os
import sys

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = "/nfs/scistore19/locatgrp/bdemirel/ssl_project"
sys.path.insert(0, ROOT)
from sslgap.extract.store import FeatureStore          # noqa: E402
from sslgap.metrics.pairs import pair_margin           # noqa: E402

RUNS = [("vm2", "in1k.floorssl.s0.d256vm.extL", "#6a3fb5"),
        ("vm3", "in1k.floorssl.s0.d256vm3.extL", "#0f7b8a"),
        ("vm4", "in1k.floorssl.s0.d256vm4.extL", "#c22f2f")]
ST = ["L03", "L06", "L09", "cls", "tap1", "tap2", "z.out"]
SPACE = {"L03": "student.h.cls.L03", "L06": "student.h.cls.L06", "L09": "student.h.cls.L09",
         "cls": "student.h.cls", "tap1": "student.z.proj.tap1", "tap2": "student.z.proj.tap2",
         "z.out": "student.z.proj.out"}
O8, VAL = "in1k.pairs10.v1@audit_v1.o8", "in1k.val.v1L"
# Ω per station; a/b/Λ oriented station→z.out (remaining-path transmission: at cls = the
# declared headline transmission, ≡1 at z.out) — Berker 2026-07-30; from the o8 retro W/B.
COLS = ["lin_v2", "knn200", "rankme/d", "effrank/d", "gauss_kl_full", "pos_cos",
        "rand_cos", "class margin (same−diff)", "Ω", "a(s→z.out)", "b(s→z.out)",
        "Λ(s→z.out)"]
BOUNDED = set(COLS[:4]) | {"pos_cos", "rand_cos", "class margin (same−diff)"}


def class_margin(X, y):
    X = X.astype(np.float32)
    X /= np.linalg.norm(X, axis=1, keepdims=True) + 1e-12
    same_n, same_s, S = 0.0, 0.0, np.zeros(X.shape[1], np.float64)
    tot_c2 = 0.0
    for c in np.unique(y):
        xc = X[y == c]
        sc = xc.sum(0, dtype=np.float64)
        n = len(xc)
        same_s += sc @ sc - n
        same_n += n * (n - 1)
        S += sc
        tot_c2 += sc @ sc
    N = len(X)
    same = same_s / same_n
    diff = (S @ S - tot_c2) / (N * N - sum((y == c).sum() ** 2 for c in np.unique(y)))
    return same - diff


def main():
    store = FeatureStore(f"{ROOT}/features")
    stats = {}                                          # (run, station) -> dict
    for lab, run, _ in RUNS:
        P = {(r["space"], r["probe"]): float(r["val_acc"])
             for r in csv.DictReader(open(f"{ROOT}/results/probes/{run}.csv"))}
        B = {(r["space"], r["metric"]): (float(r["value"]), float(r["d"]))
             for r in csv.DictReader(open(f"{ROOT}/results/battery/{run}.csv"))
             if r["variant"] == "raw|full"}
        WB = {r["space"]: (float(r["W"]), float(r["B"]))
              for r in csv.DictReader(open(f"{ROOT}/results/diag/e23_retro_spaces.csv"))
              if r["run"] == run and r["framing"] == "raw"}
        Wo, Bo = WB[SPACE["z.out"]]
        y = store.labels(run, O8)
        yv = store.labels(run, VAL)
        for st in ST:
            sp = SPACE[st]
            pm = pair_margin(np.asarray(store.get(run, O8, f"{sp}.view0"), np.float32),
                             np.asarray(store.get(run, O8, f"{sp}.view1"), np.float32))
            stats[(run, st)] = {
                "lin_v2": P[(sp, "linear_raw_v2")], "knn200": P[(sp, "knn_v1_k200")],
                "rankme/d": B[(sp, "rankme")][0] / B[(sp, "rankme")][1],
                "effrank/d": B[(sp, "effective_rank")][0] / B[(sp, "effective_rank")][1],
                "gauss_kl_full": B[(sp, "gauss_kl_full.total")][0],
                "pos_cos": pm["pos_cos"], "rand_cos": pm["rand_cos"],
                "class margin (same−diff)": class_margin(
                    np.asarray(store.get(run, VAL, sp)), yv),
                "Ω": WB[sp][0] / WB[sp][1],
                "a(s→z.out)": (Wo / WB[sp][0]) ** 0.5,
                "b(s→z.out)": (Bo / WB[sp][1]) ** 0.5,
                "Λ(s→z.out)": ((Bo / WB[sp][1]) / (Wo / WB[sp][0])) ** 0.5}
            print(f"[stat] {lab} {st}", flush=True)
    with open(f"{ROOT}/results/diag/e23_1k_stations.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["run", "station"] + COLS)
        w.writeheader()
        for (run, st), d in stats.items():
            w.writerow({"run": run, "station": st, **d})

    fig, axes = plt.subplots(len(RUNS), len(COLS), figsize=(33, 8.2), facecolor="white")
    lims = {q: (min(stats[k][q] for k in stats), max(stats[k][q] for k in stats))
            for q in COLS if q not in BOUNDED}
    for i, (lab, run, c) in enumerate(RUNS):
        for j, q in enumerate(COLS):
            ax = axes[i, j]
            for lab2, run2, _ in RUNS:                  # grey context first
                if run2 != run:
                    ax.plot(range(len(ST)), [stats[(run2, s)][q] for s in ST],
                            "-o", color="#bbbbbb", ms=2.4, lw=0.9)
            ys = [stats[(run, s)][q] for s in ST]
            ax.plot(range(len(ST)), ys, "-o", color=c, ms=3.4, lw=1.5,
                    label=lab if j == 0 else None)
            ax.axvline(3.5, color="#bbbbbb", lw=0.8, ls=":")
            if q in BOUNDED:
                ax.set_ylim(0, 1.0)
            else:
                lo, hi = lims[q]
                ax.set_ylim(lo - 0.06 * (hi - lo), hi + 0.06 * (hi - lo))
            ax.set_xticks(range(len(ST)))
            ax.set_xticklabels(ST if i == len(RUNS) - 1 else [], fontsize=6, rotation=45)
            ax.tick_params(length=0, labelsize=6.5)
            for s in ("top", "right"):
                ax.spines[s].set_visible(False)
            if i == 0:
                ax.set_title(q, fontsize=9)
            if j == 0:
                ax.set_ylabel(lab, fontsize=9)
                ax.legend(fontsize=6.5, frameon=False, loc="upper left")
    fig.suptitle("IN-1k 3-cell guillotine — D-066 standard frame; rows = cells (grey = the other two); "
                 "y 0–1 for bounded, ranks ÷ station d; probes full-train, battery 50k val, "
                 "pos/rand o8, class margin val — RAW", fontsize=10)
    fig.tight_layout(rect=(0, 0.01, 1, 0.94))
    out = f"{ROOT}/results/figures/e23/in1k_guillotine_3cell.png"
    fig.savefig(out, dpi=160)
    print("wrote", out, flush=True)


if __name__ == "__main__":
    main()
