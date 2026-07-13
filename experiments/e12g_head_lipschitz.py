"""Head-MLP Lipschitzness across the six calibration-pair variants (Berker 2026-07-13; H1 first
measurement, docs/theory/HEAD_OVERLAP_LIPSCHITZ.md; D-035 companion instrument — no training).

Two instruments:
(1) EMPIRICAL per-segment local Lipschitz from the V=8 orbit stores (audit_v1): for every image
    and unordered view pair, r = ||seg_out(A) - seg_out(B)|| / ||seg_in(A) - seg_in(B)|| along
    aug-displacement directions — the segment between consecutive head taps as the network
    actually stretches/contracts view differences. Distribution stats per (run, segment).
(2) WEIGHT-SIDE per-layer spectral norms sigma_max from the final ckpts' head modules (2-D
    weights; spectral-norm-parametrized layers reported on their `original` tensor and flagged —
    their effective sigma_max is pinned ~1 by construction). Activations/LayerNorm make the
    product a loose bound; the empirical instrument is primary.

Outputs: results/diag/e12g_head_lipschitz.csv (empirical) + e12g_head_specnorms.csv (weights) +
results/figures/e12g/e12g_head_lipschitz.png. Numbers land raw.

  sbatch slurm/e12g_lipschitz.sbatch
"""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd
import torch

ROOT = "/nfs/scistore19/locatgrp/bdemirel/ssl_project"
FEAT = f"{ROOT}/features"
FIGD = f"{ROOT}/results/figures/e12g"

PAIRS = [
    ("lejepa", "in100.lejepa.s0.e12f2.ext", "in100.lejepa.s0.e12c1.ext"),
    ("vicreg", "in100.vicreg.s0.e12gv.ext", "in100.vicreg.s0.e12gvc.ext"),
    ("dino", "in100.dino.s0.e12gd.ext", "in100.dino.s0.e12gdc.ext"),
]
COL = {"lejepa": "#0072B2", "vicreg": "#D55E00", "dino": "#009E73"}
# consecutive head taps, declared branch (the MLP the method trains between h and its loss space)
CHAIN = {
    "lejepa": ["student.z.embed", "student.z.proj.tap1", "student.z.proj.tap2",
               "student.z.proj.out"],
    "vicreg": ["student.h.gap", "student.z.proj.tap1", "student.z.proj.tap2",
               "student.z.proj.out"],
    "dino": ["teacher.h.cls", "teacher.z.dino.tap1", "teacher.z.dino.tap2",
             "teacher.z.dino.bottleneck"],
}
HEAD_MODULES = {"lejepa": ["projector"], "vicreg": ["projector"],
                "dino": ["projector", "teacher_projector"]}
V = 8


def views(run, space):
    d = f"{FEAT}/{run}/in100.pairs100.v1@audit_v1.o8"
    return [np.load(f"{d}/{space}.view{k}.npy").astype(np.float32) for k in range(V)]


def seg_ratios(run, sin, sout):
    """All C(V,2) view-pair ratios ||d_out||/||d_in|| -> stats dict."""
    A, B = views(run, sin), views(run, sout)
    rs = []
    for u in range(V):
        for w in range(u + 1, V):
            dx = np.linalg.norm(A[u] - A[w], axis=1)
            dy = np.linalg.norm(B[u] - B[w], axis=1)
            rs.append(dy / (dx + 1e-12))
    r = np.concatenate(rs)
    q = np.quantile(r, [0.5, 0.9, 0.99])
    return {"median": float(q[0]), "p90": float(q[1]), "p99": float(q[2]),
            "max": float(r.max()), "mean": float(r.mean()), "n": int(r.size)}


def spec_rows(method, run):
    ck = f"{ROOT}/outputs/{run.replace('.ext', '')}_ep100.pt"
    pay = torch.load(ck, map_location="cpu", weights_only=False)
    rows = []
    for mod in HEAD_MODULES[method]:
        sd = pay["modules"][mod]
        for k, wt in sd.items():
            plain = k.endswith("weight") and "parametrizations" not in k
            param = k.endswith("parametrizations.weight.original")
            if (plain or param) and wt.ndim == 2:
                s = torch.linalg.svdvals(wt.float())
                rows.append({"run_id": run, "module": mod, "layer": k,
                             "shape": "x".join(map(str, wt.shape)),
                             "sigma_max": float(s[0]), "sigma_min": float(s[-1]),
                             "spec_normed": int(param)})
    return rows


def main():
    os.makedirs(FIGD, exist_ok=True)
    erows, wrows = [], []
    for m, arm, ctl in PAIRS:
        for run in (arm, ctl):
            chain = CHAIN[m]
            for sin, sout in zip(chain[:-1], chain[1:]):
                st = seg_ratios(run, sin, sout)
                erows.append({"method": m, "run_id": run, "segment": f"{sin}->{sout}", **st})
                print(f"[lip] {run} {sin}->{sout}: med {st['median']:.3f} p90 {st['p90']:.3f}",
                      flush=True)
            st = seg_ratios(run, chain[0], chain[-1])
            erows.append({"method": m, "run_id": run,
                          "segment": f"{chain[0]}->{chain[-1]} (end2end)", **st})
            wrows += spec_rows(m, run)
    pd.DataFrame(erows).to_csv(f"{ROOT}/results/diag/e12g_head_lipschitz.csv", index=False)
    pd.DataFrame(wrows).to_csv(f"{ROOT}/results/diag/e12g_head_specnorms.csv", index=False)

    fig, axes = plt.subplots(2, 3, figsize=(14.5, 7.6))
    E = pd.DataFrame(erows)
    W = pd.DataFrame(wrows)
    for ci, (m, arm, ctl) in enumerate(PAIRS):
        ax = axes[0][ci]
        segs = [s for s in E[E.method == m].segment.unique() if "end2end" not in s]
        labs = [s.split("->")[1].split(".")[-1] for s in segs]
        for run, ls, filled in ((arm, "-", True), (ctl, "--", False)):
            d = E[(E.run_id == run) & (~E.segment.str.contains("end2end"))]
            med = [d[d.segment == s]["median"].iloc[0] for s in segs]
            p90 = [d[d.segment == s]["p90"].iloc[0] for s in segs]
            x = np.arange(len(segs))
            ax.plot(x, med, ls=ls, marker="o", ms=6, lw=1.7, color=COL[m],
                    mfc=COL[m] if filled else "white", mew=1.3)
            ax.plot(x, p90, ls=ls, marker="_", ms=9, lw=0.8, color=COL[m], alpha=0.55)
        ax.set_xticks(range(len(segs)), [f"->{l}" for l in labs], fontsize=8)
        ax.set_title(m, fontsize=11, color=COL[m])
        ax.set_yscale("log")
        if ci == 0:
            ax.set_ylabel("empirical segment Lipschitz\n(median dot, p90 dash; log)", fontsize=9)
        ax.grid(alpha=0.25, lw=0.5)
        ax.spines[["top", "right"]].set_visible(False)

        ax = axes[1][ci]
        for run, ls, filled in ((arm, "-", True), (ctl, "--", False)):
            d = W[(W.run_id == run) & (W.module.isin(("projector",)))]
            x = np.arange(len(d))
            ax.plot(x, d.sigma_max.to_numpy(), ls=ls, marker="o", ms=6, lw=1.7, color=COL[m],
                    mfc=COL[m] if filled else "white", mew=1.3)
            for xi, (sn, sm) in enumerate(zip(d.spec_normed, d.sigma_max)):
                if sn:
                    ax.annotate("sn", (xi, sm), textcoords="offset points", xytext=(0, 6),
                                fontsize=7, color="#777777", ha="center")
        ax.set_xlabel("head Linear layer index (student projector)", fontsize=8)
        ax.set_yscale("log")
        if ci == 0:
            ax.set_ylabel("weight sigma_max per layer\n('sn' = spectral-norm param.)", fontsize=9)
        ax.grid(alpha=0.25, lw=0.5)
        ax.spines[["top", "right"]].set_visible(False)
    hs = [Line2D([], [], color="#444444", marker="o", ls="-", ms=6, label="floor arm"),
          Line2D([], [], color="#444444", marker="o", ls="--", ms=6, mfc="white", mew=1.3,
                 label="matched control")]
    fig.legend(handles=hs, loc="upper right", ncol=2, fontsize=9, frameon=False)
    fig.suptitle("Head-MLP Lipschitzness, arm vs control — top: empirical view-pair stretch per "
                 "head segment (V=8 orbit stores, audit_v1); bottom: per-layer weight spectral "
                 "norms (final ckpts)", fontsize=10, x=0.02, ha="left")
    fig.tight_layout(rect=(0, 0.01, 1, 0.93))
    fig.savefig(f"{FIGD}/e12g_head_lipschitz.png", dpi=180)
    print(f"[lip] wrote {FIGD}/e12g_head_lipschitz.png + 2 CSVs")


if __name__ == "__main__":
    main()
