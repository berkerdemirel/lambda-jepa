"""E13 extension (Berker 2026-07-13): every zoo method as the D_read target. For each of
{byol, simclr, lejepa, ijepa, dino, vicreg, mae} build the primary-cell RFF sketch of ITS declared
h (D_m=1024, sigma = per-target median heuristic, seed = primary-cell convention) and ridge-read
it from every OTHER checkpoint's h (deitlite discarded; direction-symmetrized A2B/B2A, exact E13
machinery imported). Deliverable: 7 scatters D_read vs linear_raw_v2 with Spearman + perm p —
does readout distance rank quality, or only affinity to the target family (E14-T2's question,
now per-target)? No d_kern by request. Numbers land raw.

  sbatch slurm/e13_targets.sbatch
"""
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from e13_pivot_rung0 import (LEJEPA_SUB, PRIMARY, RANKED, Side, load_pairs, probe_rows,
                             spearman_perm, splits, standardize)
from scipy.spatial.distance import pdist

ROOT = "/nfs/scistore19/locatgrp/bdemirel/ssl_project"
FIGD = f"{ROOT}/results/figures/e13_targets"

TARGETS = {"byol": ("in100.byol.s0.ext", "student.h.gap"),
           "simclr": ("in100.simclr.s0.ext", "student.h.gap"),
           "lejepa": ("in100.lejepa.s0.ext", "student.z.embed"),
           "ijepa": ("in100.ijepa.s0.ext", "teacher.h.gap"),
           "dino": ("in100.dino.s0.ext", "teacher.h.cls"),
           "vicreg": ("in100.vicreg.s0.ext", "student.h.gap"),
           "mae": ("in100.mae.s0.ext", "student.h.gap")}
ZOO = {r: s for r, s in RANKED.items() if "deitlite" not in r}   # 19 checkpoints
D_M, SEED = PRIMARY[0], 1312                                     # primary-cell conventions
BLUE, VERM, ORNG, GRAY = "#0072B2", "#D55E00", "#E69F00", "#666666"


def fam(rid):
    if rid in LEJEPA_SUB:
        return BLUE, "o", "lejepa family"
    if "ctrl" in rid:
        return ORNG, "s", "dino-ctrl"
    return VERM, "^", "core zoo"


def short(rid):
    s = rid.replace("in100.", "").replace(".ext", "").replace(".s0", "")
    return s.replace("lejepa.e12", "").replace("dino-ctrl.", "ctrl-")


def main():
    os.makedirs(FIGD, exist_ok=True)
    y = np.load(f"{ROOT}/features/{next(iter(ZOO))}/in100.pairs100.v1@audit_v1/labels.npy")
    folds = splits(y)
    itr = folds[0]
    n_tr = len(itr)
    probe = {rid: probe_rows(rid, h)["linear_raw_v2"] for rid, h in ZOO.items()}

    side_cache, drows, rrows = {}, [], []
    for tname, (t_rid, t_space) in TARGETS.items():
        members = {r: s for r, s in ZOO.items() if r != t_rid}
        acc = {r: [] for r in members}
        for dirn, (v_in, v_out) in {"A2B": ("A", "B"), "B2A": ("B", "A")}.items():
            Y0 = standardize(load_pairs(t_rid, t_space, v_out), itr)
            sub = np.random.default_rng(1).choice(itr, size=2048, replace=False)
            sigma0 = float(np.median(pdist(Y0[sub].astype(np.float64))))
            rng = np.random.default_rng(SEED)
            W = (rng.standard_normal((Y0.shape[1], D_M)) / sigma0).astype(np.float32)
            b = rng.uniform(0, 2 * np.pi, D_M).astype(np.float32)
            Phi = np.sqrt(2.0 / D_M) * np.cos(Y0 @ W + b)
            print(f"[e13t] T={tname} dir={dirn} sigma0={sigma0:.3f}", flush=True)
            for rid, h in members.items():
                key = (dirn, rid)
                if key not in side_cache:
                    side_cache[key] = Side(rid, h, v_in, *folds)
                m, _, _ = side_cache[key].fit_eval(Phi, folds, n_tr)
                acc[rid].append(m["d_read"])
        vals = {r: float(np.mean(v)) for r, v in acc.items()}
        for r, v in vals.items():
            drows.append({"target": tname, "run": r, "d_read_sym": round(v, 5),
                          "linear_raw_v2": probe[r]})

        subsets = {"all_others": list(members)}
        if tname == "lejepa":
            subsets["excl_lejepa_family"] = [r for r in members if r not in LEJEPA_SUB]
        if tname == "dino":
            subsets["excl_dino_ctrl"] = [r for r in members if "ctrl" not in r]
        rhos = {}
        for sname, rids in subsets.items():
            rho, p = spearman_perm([vals[r] for r in rids], [probe[r] for r in rids])
            rhos[sname] = (rho, p, len(rids))
            rrows.append({"target": tname, "subset": sname, "n": len(rids),
                          "rho": round(rho, 4), "perm_p": round(p, 5)})
            print(f"[e13t] T={tname} {sname}: rho={rho:+.3f} p={p:.4f} n={len(rids)}", flush=True)

        fig, ax = plt.subplots(figsize=(6.4, 5.4))
        for rid in members:
            col, mk, _ = fam(rid)
            ax.plot([vals[rid]], [probe[rid]], marker=mk, color=col, ms=8, ls="none")
            ax.annotate(short(rid), (vals[rid], probe[rid]), textcoords="offset points",
                        xytext=(4, 4), fontsize=7, color="#222222")
        rho, p, n = rhos["all_others"]
        sub = "".join(f"\n{s}: ρ={r_:+.2f} (p={p_:.3f}, n={n_})"
                      for s, (r_, p_, n_) in rhos.items() if s != "all_others")
        ax.set_title(f"target = {tname}   ρ={rho:+.2f} (perm p={p:.3f}, n={n}){sub}", fontsize=10)
        ax.set_xlabel(f"D_read onto {tname} sketch (primary cell; lower = better predicted)")
        ax.set_ylabel("linear_raw_v2 val acc at declared h")
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(alpha=0.25, lw=0.5)
        fig.text(0.01, 0.005, "blue o = lejepa family · orange s = dino-ctrl · vermillion ^ = "
                 "core zoo (deitlite discarded)", fontsize=7.5, color="#555555")
        fig.tight_layout(rect=(0, 0.02, 1, 1))
        fig.savefig(f"{FIGD}/e13_dread_scatter_{tname}.png", dpi=170)
        plt.close(fig)

    pd.DataFrame(drows).to_csv(f"{ROOT}/results/diag/e13_targets_dread.csv", index=False)
    pd.DataFrame(rrows).to_csv(f"{ROOT}/results/diag/e13_targets_rho.csv", index=False)
    print(f"[e13t] wrote 7 scatters under {FIGD}/ + 2 CSVs", flush=True)


if __name__ == "__main__":
    main()
