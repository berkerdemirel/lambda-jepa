"""H-wave visualization (Berker request): per method, arms = control / +calib / +calib-more,
across {invariance, gaussianity(=moment-match + excess-kurtosis + Epps-Pulley), linear, kNN} at h
and at z; plus a de-confounded head Lipschitz (operator weight spectral norm, data-independent).
RAW numbers rendered; no takeaway. lejepa: c1/f2/f7. dino: gdc/gd/gd2."""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

ROOT = "/nfs/scistore19/locatgrp/bdemirel/ssl_project"
FEAT = f"{ROOT}/features"
FIGD = f"{ROOT}/results/figures/e12h"
os.makedirs(FIGD, exist_ok=True)
STACK = "audit_v1"

# (label, run_id, color) — light->dark method-hue ramp = control -> +calibration
LEJ = dict(name="lejepa", h="student.z.embed", z="student.z.proj.out",
           arms=[("control (c1)", "in100.lejepa.s0.e12c1.ext", "#9ecae1"),
                 ("+floor/kl (f2)", "in100.lejepa.s0.e12f2.ext", "#4292c6"),
                 ("+kl+inv (f7)", "in100.lejepa.s0.e12f7.ext", "#08306b")],
           chain=["student.z.embed", "student.z.proj.tap1", "student.z.proj.tap2",
                  "student.z.proj.out"], head="projector")
DINO = dict(name="dino", h="teacher.h.cls", z="teacher.z.dino.bottleneck",
            arms=[("control (gdc)", "in100.dino.s0.e12gdc.ext", "#a1d99b"),
                  ("+floor cls (gd)", "in100.dino.s0.e12gd.ext", "#41ab5d"),
                  ("+floor cls+gap (gd2)", "in100.dino.s0.e12gd2.ext", "#00441b")],
            chain=["teacher.h.cls", "teacher.z.dino.tap1", "teacher.z.dino.tap2",
                   "teacher.z.dino.bottleneck"], head="teacher_projector")
VICREG = dict(name="vicreg", h="student.h.cls", z="student.z.proj.out",
              arms=[("control (gvc)", "in100.vicreg.s0.e12gvc.ext", "#fdae6b"),
                    ("+floor GAP (gv)", "in100.vicreg.s0.e12gv.ext", "#e6550d"),
                    ("+floor CLS (gv2)", "in100.vicreg.s0.e12gv2.ext", "#a63603")],
              chain=["student.h.cls", "student.z.proj.tap1", "student.z.proj.tap2",
                     "student.z.proj.out"], head="projector")

METRICS = [("view-align ↑\npos_cos", "↑ same-image views aligned", ("orb", "pos")),
           ("cone / neg ↓\nrand_cos", "↓ random pairs decorrelated", ("orb", "rand")),
           ("cos_margin\n(pos − rand)", "read WITH pos & rand", ("orb", "margin")),
           ("gaussianity 1\nmoment-KL (2 mom.)", "↓ moments match", "diag"),
           ("gaussianity 2\nexcess kurt (worst)", "↓ less heavy-tail", ("bat", "kurt_topeig.worst")),
           ("gaussianity 3\nEpps-Pulley", "↓ more Gaussian", ("bat", "epps_pulley")),
           ("linear probe\nraw_v2 (%)", "↑ better", ("probe", "linear_raw_v2")),
           ("kNN k=200 (%)", "↑ better", ("probe", "knn_v1_k200"))]


def orbit(run, want):
    d = f"{FEAT}/{run}/in100.pairs100.v1@{STACK}.o8"; m = json.load(open(f"{d}/meta.json"))
    V = m["v"]
    p = np.random.default_rng(0).permutation(m["n"]); q = np.roll(p, 1); out = {}
    for sp in want:
        vs = [np.load(f"{d}/{sp}.view{k}.npy").astype(np.float64) for k in range(V)]
        vs = [x / (np.linalg.norm(x, axis=1, keepdims=True) + 1e-12) for x in vs]
        pc, rc = [], []
        for u in range(V):
            for w in range(u + 1, V):
                a, b = vs[u], vs[w]; pc.append((a * b).sum(1).mean()); rc.append((a[p] * b[q]).sum(1).mean())
        pos, rand = float(np.mean(pc)), float(np.mean(rc))
        out[sp] = {"pos": pos, "rand": rand, "margin": pos - rand}
    return out


def diag(run, sp):
    X = np.load(f"{FEAT}/{run}/in100.train500.v1L/{sp}.npy").astype(np.float64)
    mu, var = X.mean(0), X.var(0).clip(1e-8)
    return float(0.5 * (np.mean(mu ** 2) + np.mean(var - 1 - np.log(var))))


def bat(run):
    df = pd.read_csv(f"{ROOT}/results/battery/{run}.csv"); df = df[df.variant == "raw|full"]
    return {(r.space, r.metric): r.value for r in df.itertuples()}


def prb(run):
    df = pd.read_csv(f"{ROOT}/results/probes/{run}.csv")
    return {(r.space, r.probe): r.val_acc for r in df.itertuples()}


def specnorms(run, head):
    ck = f"{ROOT}/outputs/{run.replace('.ext', '')}_ep100.pt"
    sd = torch.load(ck, map_location="cpu", weights_only=False)["modules"][head]
    out = []
    for k, wt in sd.items():
        if wt.ndim == 2 and (k.endswith("weight") and "parametriz" not in k
                             or k.endswith("parametrizations.weight.original")):
            out.append(float(torch.linalg.svdvals(wt.float())[0]))
    return out


def value(cfg, run, space, src, ob, ba, pr):
    if src == "diag":
        return diag(run, space)
    kind, key = src
    if kind == "orb":
        return (ob[run].get(space) or {}).get(key)
    if kind == "bat":
        return ba[run].get((space, key))
    v = pr[run].get((space, key))
    return v * 100 if v is not None else None            # probes -> %


def build(cfg):
    runs = [r for _, r, _ in cfg["arms"]]
    ob = {r: orbit(r, [cfg["h"], cfg["z"]]) for r in runs}
    ba = {r: bat(r) for r in runs}; pr = {r: prb(r) for r in runs}
    fig = plt.figure(figsize=(8.2, 19), facecolor="white")
    gs = fig.add_gridspec(len(METRICS) + 1, 2, hspace=0.62, wspace=0.28,
                      height_ratios=[1] * len(METRICS) + [1.25])
    for mi, (mlabel, dircue, src) in enumerate(METRICS):
        for ci, (space, sname) in enumerate([(cfg["h"], "h"), (cfg["z"], "z")]):
            ax = fig.add_subplot(gs[mi, ci])
            vals = [value(cfg, r, space, src, ob, ba, pr) for _, r, _ in cfg["arms"]]
            cols = [c for _, _, c in cfg["arms"]]
            xs = np.arange(len(vals))
            bars = ax.bar(xs, [v if v is not None else 0 for v in vals], color=cols,
                          width=0.68, edgecolor="white", linewidth=1.2)
            for x, v in zip(xs, vals):
                ax.text(x, (v if v else 0), f"{v:.3g}" if v is not None else "n/a",
                        ha="center", va="bottom", fontsize=7.5, color="#333")
            ax.set_xticks(xs); ax.set_xticklabels([sname] * len(vals), fontsize=7, color="#888")
            ax.set_title(f"{mlabel} @ {sname}", fontsize=8.5, color="#222")
            ax.margins(y=0.22); ax.tick_params(length=0)
            for s in ("top", "right", "left"):
                ax.spines[s].set_visible(False)
            ax.set_yticks([])
            if ci == 0:
                ax.text(-0.02, 1.14, dircue, transform=ax.transAxes, fontsize=7, color="#0072B2",
                        style="italic", ha="left")
    # ---- de-confounded head Lipschitz: operator weight spectral norm (data-independent) ----
    axl = fig.add_subplot(gs[len(METRICS), :])
    sn = {lab: specnorms(r, cfg["head"]) for lab, r, _ in cfg["arms"]}
    L = max(len(v) for v in sn.values()); x = np.arange(L); w = 0.26
    for j, (lab, r, c) in enumerate(cfg["arms"]):
        y = sn[lab] + [0] * (L - len(sn[lab]))
        axl.bar(x + (j - 1) * w, y, width=w, color=c, edgecolor="white", linewidth=1,
                label=lab.split(" (")[0])
        for xx, yy in zip(x + (j - 1) * w, y):
            axl.text(xx, yy, f"{yy:.2f}", ha="center", va="bottom", fontsize=6.5, color="#444")
    axl.set_title("de-confounded head Lipschitz = weight spectral norm σ_max per head layer "
                  "(operator bound, data-independent)", fontsize=8.5, color="#222")
    axl.set_xticks(x); axl.set_xticklabels([f"head layer {i+1}" for i in range(L)], fontsize=7.5)
    axl.legend(fontsize=7.5, frameon=False, ncol=3, loc="upper center", bbox_to_anchor=(0.5, -0.12))
    for s in ("top", "right"):
        axl.spines[s].set_visible(False)
    axl.tick_params(length=0)
    fig.suptitle(f"{cfg['name'].upper()} H-wave — arms × metrics @ h and @ z  (RAW; arms in legend)",
                 fontsize=11, y=0.995)
    out = f"{FIGD}/e12h_{cfg['name']}_hz.png"
    fig.savefig(out, dpi=155, bbox_inches="tight", facecolor="white"); plt.close(fig)
    print(f"[e12h] wrote {out}", flush=True)


for cfg in (LEJ, DINO, VICREG):
    build(cfg)
print("[e12h] done", flush=True)
