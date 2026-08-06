"""E24 canonical-o8 sorting figure (Berker 2026-08-04: "+ o8 figs") — y vs Ω_h(end) on
the LANDING channel (audit_v1 o8, raw framing), the Spearman table drawn. Two y modes
(argv[1]): `acc` = final best test acc (wandb; per-family ρ from e24_omega_sort.csv) →
e24_o8_acc_omega.png; `knn` (Berker's repeat ask) = knn_v1_k200 at the declared h from
the landing probes — both axes on the frozen landed checkpoint — with per-family ρ
computed here (scipy spearmanr) → e24_o8_knn_omega.png. Left: toy, all 29 cells colored
by family (pooled h-free / pooled h.03 / vm-OAS / view-mean variants). Right: the six
in100 v-cells — Ω_h(o8) computed here from the landed stores (orbit_energies,
student.h.cls). Points labeled with short tags. RAW; no takeaway."""
import csv
import sys

import matplotlib
import numpy as np
from scipy.stats import spearmanr

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = "/nfs/scistore19/locatgrp/bdemirel/ssl_project"
sys.path.insert(0, ROOT)
from sslgap.extract.store import FeatureStore                    # noqa: E402
from sslgap.metrics.orbit_energy import orbit_energies           # noqa: E402

POOLED_HFREE = ["s0.e24s0a", "s0.e24s0b", "s0.e24s0c", "s0.e24s0d", "s0.e24s0ac",
                "s0.e24s0bc", "s0.e24s0cc", "s0.e24wz05", "s0.e24wz12", "s0.e24wz18",
                "s0.e24wz25", "s0.e24wt05", "s0.e24wt2", "s1.e24wrep"]
POOLED_H03 = ["s0.e24s1a", "s0.e24s1b", "s0.e24s1c", "s0.e24s1ac", "s0.e24s1bc",
              "s0.e24s1cc"]
VM_OAS = ["s0.e24va", "s0.e24vb", "s0.e24vc", "s0.e24vac", "s0.e24v4oas", "s0.e24vh0"]
VARIANTS = ["s0.e24v1", "s0.e24v2lr", "s0.e24v3nq"]
FAMS = [("pooled h-free", POOLED_HFREE, "#3d65d0", "pooled h-free n14"),
        ("pooled h.03", POOLED_H03, "#d4820a", "pooled h.03 n6"),
        ("vm-OAS", VM_OAS, "#2e8b57", "vm-OAS n6"),
        ("vm variants", VARIANTS, "#8a5cb8", None)]
IN100 = ["va", "vb", "vc", "vh0", "vac", "vcc"]


def short(tag):
    return tag.split(".")[-1].removeprefix("e24")


def knn_at_h(run):
    return {(r["space"], r["probe"]): float(r["val_acc"])
            for r in csv.DictReader(open(f"{ROOT}/results/probes/{run}.csv"))
            }[("student.h.cls", "knn_v1_k200")]


def main():
    mode = sys.argv[1] if len(sys.argv) > 1 else "acc"
    cells = {r["tag"]: (float(r["acc_best"]), float(r["omega_h_o8_raw"]))
             for r in csv.DictReader(open(f"{ROOT}/results/diag/e24_omega_sort.csv"
                                          .replace(".csv", "_cells.csv")))}
    if mode == "knn":
        yof = {t: knn_at_h(f"toy.floorssl.{t}.extL") for t in cells}
        ylab = "kNN200 @ h (val, landing probes)"
    else:
        yof = {t: cells[t][0] for t in cells}
        ylab = "final best acc"
    rhos_acc = {r["group"]: float(r["rho"])
                for r in csv.DictReader(open(f"{ROOT}/results/diag/e24_omega_sort.csv"))
                if r["framing"] == "raw"}
    fig, (axt, axi) = plt.subplots(1, 2, figsize=(13.5, 5.6), facecolor="white")
    for name, tags, col, rkey in FAMS:
        xs = [cells[t][1] for t in tags]
        ys = [yof[t] for t in tags]
        if mode == "knn":
            lab = f"{name} (ρ {spearmanr(xs, ys).statistic:+.3f})" if rkey else name
        else:
            lab = f"{name} (ρ {rhos_acc[rkey]:+.3f})" if rkey else name
        axt.scatter(xs, ys, s=26, color=col, label=lab, zorder=3)
        for t, x, y in zip(tags, xs, ys):
            axt.annotate(short(t), (x, y), fontsize=5.6, xytext=(2.5, 2.5),
                         textcoords="offset points", color=col)
    all_tags = [t for _, tags, _, _ in FAMS for t in tags]
    rho_all = (spearmanr([cells[t][1] for t in all_tags],
                         [yof[t] for t in all_tags]).statistic if mode == "knn"
               else rhos_acc["combined+variants n29"])
    axt.set_title(f"toy — 29 cells (combined ρ {rho_all:+.3f})", fontsize=10)
    store = FeatureStore(f"{ROOT}/features")
    if mode != "knn":
        import wandb
        api = wandb.Api(timeout=120)
    for t in IN100:
        run = f"in100.floorssl.s0.e24{t}.extL"
        o8 = "in100.pairs100.v1@audit_v1.o8"
        V = store.meta(run, o8)["v"]
        e = orbit_energies([np.asarray(store.get(run, o8, f"student.h.cls.view{k}"),
                                       np.float64) for k in range(V)])
        if mode == "knn":
            y = knn_at_h(run)
        else:
            y = float("nan")
            for r in api.runs("causal-learning-ai-ista/sslgap",
                              filters={"display_name": f"in100.floorssl.s0.e24{t}"}):
                vals = [row["test/acc"] for row in
                        r.history(keys=["test/acc"], samples=2000, pandas=False)
                        if row.get("test/acc") is not None]
                if vals:
                    y = np.nanmax([y] + vals)
        axi.scatter([e["omega"]], [y], s=34, color="#2e8b57", zorder=3)
        axi.annotate(t, (e["omega"], y), fontsize=8, xytext=(4, 3),
                     textcoords="offset points")
        print(f"[in100] {t}: {mode} {y:.4f} omega_h(o8) {e['omega']:.3f}", flush=True)
    axi.set_title("in100 — 6 v-cells (mirrors + twins)", fontsize=10)
    for ax in (axt, axi):
        ax.set_xlabel("Ω_h(end) — audit_v1 o8, raw", fontsize=9)
        ax.set_ylabel(ylab, fontsize=9)
        ax.tick_params(labelsize=8)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
    axt.legend(fontsize=7.5, frameon=False, loc="lower left")
    fig.suptitle(f"E24 canonical-o8 sorting — {ylab} vs Ω_h(end) on the landing "
                 f"channel; per-family Spearman in legend — RAW", fontsize=10.5)
    fig.tight_layout(rect=(0, 0.01, 1, 0.95))
    out = f"{ROOT}/results/figures/e24/e24_o8_{'knn' if mode == 'knn' else 'acc'}_omega.png"
    fig.savefig(out, dpi=160)
    print("wrote", out, flush=True)


if __name__ == "__main__":
    main()
