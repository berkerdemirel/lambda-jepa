"""E20 comparison pass (d): class-pair d' at declared h, e20f arm vs control per zoo lane.
For every class pair (i,j): project both classes on the mean-difference axis u = (mu_i-mu_j)/|.|;
d' = |mu_i-mu_j| / sqrt((var_i(u)+var_j(u))/2) — the two-class linear discriminability the
probe actually consumes, per pair, so WHERE the floor's lin/knn gains live (many-pairs-slightly
vs few-pairs-strongly) is visible. Lane->runs/space mapping read from results/diag/
e20_centered.csv (the landing convention incl. control flavor). Per-pair arrays ->
results/diag/e20_pair_dprime.npz; summary csv -> results/diag/e20_pair_dprime.csv; figure
(ctrl-vs-arm scatter per lane) -> results/figures/e20/e20_pair_dprime.png. RAW; no takeaway."""
import csv
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = "/nfs/scistore19/locatgrp/bdemirel/ssl_project"
FEAT = f"{ROOT}/features"
ORDER = ["lejepa", "byol", "simclr", "vicreg", "dino", "ijepa", "mae"]


def train_store(run):
    for fl in ("in100.train500.v1", "in100.train500.v1L"):
        if os.path.isdir(f"{FEAT}/{run}/{fl}"):
            return f"{FEAT}/{run}/{fl}"
    raise FileNotFoundError(run)


def pair_dprime(run, space):
    d = train_store(run)
    X = np.load(f"{d}/{space}.npy").astype(np.float64)
    y = np.load(f"{d}/labels.npy")
    classes = np.unique(y)
    K = len(classes)
    Xc = [X[y == c] for c in classes]
    Mu = np.stack([x.mean(0) for x in Xc])
    out = np.zeros((K, K))
    for i in range(K):
        Dif = Mu - Mu[i]                              # [K, D]
        nrm = np.linalg.norm(Dif, axis=1)
        nrm[i] = 1.0
        U = Dif / nrm[:, None]                        # unit dirs i->j
        Pi = Xc[i] @ U.T                              # [n_i, K] class i on every dir
        vi = Pi.var(0)
        vj = np.array([Xc[j] @ U[j] for j in range(K)]).var(1)
        with np.errstate(divide="ignore", invalid="ignore"):
            out[i] = nrm / np.sqrt(0.5 * (vi + vj))
    iu = np.triu_indices(K, 1)
    return out[iu]                                     # [K*(K-1)/2]


def main():
    cells = {}
    for r in csv.DictReader(open(f"{ROOT}/results/diag/e20_centered.csv")):
        side = "arm" if ".e20f" in r["run"] else "ctrl"
        cells.setdefault(r["method"], {})[side] = (r["run"], r["space"])
    arrays, rows = {}, []
    for m in ORDER:
        if m not in cells or len(cells[m]) < 2:
            print(f"[dprime] {m}: incomplete cells, skipped")
            continue
        for side in ("ctrl", "arm"):
            run, space = cells[m][side]
            dp = pair_dprime(run, space)
            arrays[f"{m}.{side}"] = dp
            rows.append({"lane": m, "side": side, "run": run, "space": space,
                         "mean": round(float(dp.mean()), 4), "median": round(float(np.median(dp)), 4),
                         "p10": round(float(np.percentile(dp, 10)), 4),
                         "p90": round(float(np.percentile(dp, 90)), 4),
                         "min": round(float(dp.min()), 4)})
        a, c = arrays[f"{m}.arm"], arrays[f"{m}.ctrl"]
        frac = float((a > c).mean())
        rows.append({"lane": m, "side": "delta", "run": "", "space": "",
                     "mean": round(float((a - c).mean()), 4),
                     "median": round(float(np.median(a - c)), 4),
                     "p10": round(float(np.percentile(a - c, 10)), 4),
                     "p90": round(float(np.percentile(a - c, 90)), 4),
                     "min": round(frac, 4)})    # delta row: min column carries frac(arm>ctrl)
        print(f"[dprime] {m}: ctrl mean {c.mean():.3f} arm mean {a.mean():.3f} "
              f"frac(arm>ctrl) {frac:.3f}", flush=True)
    np.savez(f"{ROOT}/results/diag/e20_pair_dprime.npz", **arrays)
    with open(f"{ROOT}/results/diag/e20_pair_dprime.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    lanes = [m for m in ORDER if f"{m}.arm" in arrays]
    fig, axes = plt.subplots(1, len(lanes), figsize=(2.6 * len(lanes), 2.9), facecolor="white")
    for ax, m in zip(np.atleast_1d(axes), lanes):
        c, a = arrays[f"{m}.ctrl"], arrays[f"{m}.arm"]
        lim = max(np.percentile(c, 99.5), np.percentile(a, 99.5)) * 1.05
        ax.plot([0, lim], [0, lim], color="#999", lw=0.8, zorder=1)
        ax.scatter(c, a, s=2, alpha=0.15, color="#3d65d0", edgecolors="none", zorder=2)
        ax.set_xlim(0, lim); ax.set_ylim(0, lim)
        ax.set_title(f"{m}  Δmed {np.median(a-c):+.2f}", fontsize=9)
        ax.set_xlabel("ctrl d'", fontsize=8)
        ax.tick_params(labelsize=7)
    np.atleast_1d(axes)[0].set_ylabel("e20f arm d'", fontsize=8)
    fig.suptitle("E20: class-pair d' at declared h — arm vs control (each dot = one of 4950 class pairs)",
                 fontsize=10)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    out = f"{ROOT}/results/figures/e20/e20_pair_dprime.png"
    fig.savefig(out, dpi=160)
    print("wrote", out, flush=True)


if __name__ == "__main__":
    main()
