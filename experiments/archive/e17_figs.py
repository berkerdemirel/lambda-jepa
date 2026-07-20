"""E17 h-pull figures — per method, bars = control / +own-non-collapse@h / +own-non-collapse+inv@h,
across {invariance triple, gaussianity (moment-KL, worst excess-kurt, Epps-Pulley), linear, kNN}
@ h and @ z.final; plus the de-confounded head Lipschitz (weight sigma_max/layer, data-independent).
RAW numbers rendered; NO takeaway (CLAUDE.md). Robust to arms not yet landed (skips missing bars/
methods). Mirrors e12h_figs conventions; arm sets + ctrl->lane checkpoint mapping are E17's."""
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
FIGD = f"{ROOT}/results/figures/e17"
os.makedirs(FIGD, exist_ok=True)
STACK = "audit_v1"

# (label, run_id, color) light->dark = control -> +own-non-collapse -> +own-non-collapse+inv
CFG = {
    "lejepa": dict(h="student.z.embed", z="student.z.proj.out", heads=["projector"],
                   chain=["student.z.embed", "student.z.proj.tap1", "student.z.proj.tap2", "student.z.proj.out"],
                   arms=[("control", "in100.lejepa.s0.e17c.ext", "#9ecae1"),
                         ("+SIGReg@h", "in100.lejepa.s0.hpull_sigreg.ext", "#4292c6"),
                         ("+SIGReg+inv", "in100.lejepa.s0.hpull_sigreg_inv.ext", "#08306b")]),
    "vicreg": dict(h="student.h.cls", z="student.z.proj.out", heads=["projector"],
                   chain=["student.h.cls", "student.z.proj.tap1", "student.z.proj.tap2", "student.z.proj.out"],
                   arms=[("control", "in100.vicreg.s0.e17c.ext", "#fdae6b"),
                         ("+var/cov@h", "in100.vicreg.s0.hpull_varcov.ext", "#e6550d"),
                         ("+var/cov+inv", "in100.vicreg.s0.hpull_varcov_inv.ext", "#a63603")]),
    "simclr": dict(h="student.h.cls", z="student.z.proj.out", heads=["projector"],
                   chain=["student.h.cls", "student.z.proj.tap1", "student.z.proj.out"],
                   arms=[("control", "in100.simclr.s0.e17c.ext", "#bcbddc"),
                         ("+uniform@h", "in100.simclr.s0.hpull_uniform.ext", "#807dba"),
                         ("+uniform+align", "in100.simclr.s0.hpull_uniform_align.ext", "#3f007d")]),
    "byol": dict(h="student.h.cls", z="student.z.pred.out", heads=["projector", "predictor"],
                 chain=["student.h.cls", "student.z.proj.tap1", "student.z.proj.out",
                        "student.z.pred.tap1", "student.z.pred.out"],
                 arms=[("control", "in100.byol.s0.e17c.ext", "#a1d99b"),
                       ("+align@h", "in100.byol.s0.hpull_align.ext", "#238b45")]),
    "dino": dict(h="teacher.h.cls", z="teacher.z.dino.bottleneck", heads=["teacher_projector"],
                 chain=["teacher.h.cls", "teacher.z.dino.tap1", "teacher.z.dino.tap2", "teacher.z.dino.bottleneck"],
                 arms=[("control", "in100.dino.s0.e17c.ext", "#d9d9d9"),
                       ("+proto-CE@h", "in100.dino.s0.hpull_protoce.ext", "#252525")]),
}
METRICS = [("view-align ↑\npos_cos", ("orb", "pos")), ("cone / neg ↓\nrand_cos", ("orb", "rand")),
           ("cos_margin\n(pos − rand)", ("orb", "margin")), ("moment-KL ↓\n(diag, 2-mom)", "diag"),
           ("excess kurt ↓\n(worst)", ("bat", "kurt_topeig.worst")), ("Epps-Pulley ↓", ("bat", "epps_pulley")),
           ("effective rank\n(centered cov)", ("bat", "effective_rank")), ("RankMe", ("bat", "rankme")),
           ("off-diag mean|corr| ↓", ("bat", "offdiag_redundancy.mean_abs_corr")),
           ("linear probe %\n↑", ("probe", "linear_raw_v2")), ("kNN k=200 % ↑", ("probe", "knn_v1_k200"))]


def have(run):
    return os.path.isdir(f"{FEAT}/{run}/in100.pairs100.v1@{STACK}.o8")


def orbit(run, want):
    d = f"{FEAT}/{run}/in100.pairs100.v1@{STACK}.o8"; m = json.load(open(f"{d}/meta.json")); V = m["v"]
    p = np.random.default_rng(0).permutation(m["n"]); q = np.roll(p, 1); out = {}
    for sp in want:
        vs = [np.load(f"{d}/{sp}.view{k}.npy").astype(np.float64) for k in range(V)]
        vs = [x / (np.linalg.norm(x, axis=1, keepdims=True) + 1e-12) for x in vs]
        pc, rc = [], []
        for u in range(V):
            for w in range(u + 1, V):
                pc.append((vs[u] * vs[w]).sum(1).mean()); rc.append((vs[u][p] * vs[w][q]).sum(1).mean())
        pos, rand = float(np.mean(pc)), float(np.mean(rc)); out[sp] = {"pos": pos, "rand": rand, "margin": pos - rand}
    return out


def diag(run, sp):
    X = np.load(f"{FEAT}/{run}/in100.train500.v1L/{sp}.npy").astype(np.float64)
    mu, var = X.mean(0), X.var(0).clip(1e-8); return float(0.5 * (np.mean(mu ** 2) + np.mean(var - 1 - np.log(var))))


def bat(run):
    df = pd.read_csv(f"{ROOT}/results/battery/{run}.csv"); df = df[df.variant == "raw|full"]
    return {(r.space, r.metric): r.value for r in df.itertuples()}


def prb(run):
    df = pd.read_csv(f"{ROOT}/results/probes/{run}.csv"); return {(r.space, r.probe): r.val_acc for r in df.itertuples()}


def specnorms(run, heads):
    ck = f"{ROOT}/outputs/{run.replace('.e17c.ext', '').replace('.ext', '')}_ep100.pt"
    sd = torch.load(ck, map_location="cpu", weights_only=False)["modules"]; out = []
    for hk in heads:
        for k, wt in sd.get(hk, {}).items():
            if wt.ndim == 2 and (k.endswith("weight") and "parametriz" not in k
                                 or k.endswith("parametrizations.weight.original1")):
                out.append(float(torch.linalg.svdvals(wt.float())[0]))
    return out


def value(run, space, src, ob, ba, pr):
    if src == "diag":
        return diag(run, space)
    kind, key = src
    if kind == "orb":
        return (ob[run].get(space) or {}).get(key)
    if kind == "bat":
        return ba[run].get((space, key))
    v = pr[run].get((space, key)); return v * 100 if v is not None else None


def build(name, cfg):
    arms = [(l, r, c) for l, r, c in cfg["arms"] if have(r)]
    if len(arms) < 2:
        print(f"[e17fig] {name}: only {len(arms)}/{len(cfg['arms'])} extracted — skip"); return
    runs = [r for _, r, _ in arms]
    ob = {r: orbit(r, [cfg["h"], cfg["z"]]) for r in runs}; ba = {r: bat(r) for r in runs}; pr = {r: prb(r) for r in runs}
    fig = plt.figure(figsize=(8.2, 25), facecolor="white")
    gs = fig.add_gridspec(len(METRICS) + 1, 2, hspace=0.6, wspace=0.28, height_ratios=[1] * len(METRICS) + [1.25])
    for mi, (mlabel, src) in enumerate(METRICS):
        for ci, (space, sname) in enumerate([(cfg["h"], "h"), (cfg["z"], "z")]):
            ax = fig.add_subplot(gs[mi, ci])
            vals = [value(r, space, src, ob, ba, pr) for _, r, _ in arms]
            xs = np.arange(len(vals))
            ax.bar(xs, [v if v is not None else 0 for v in vals], color=[c for _, _, c in arms],
                   width=0.68, edgecolor="white", linewidth=1.2)
            for x, v in zip(xs, vals):
                ax.text(x, v if v else 0, f"{v:.3g}" if v is not None else "n/a", ha="center", va="bottom", fontsize=7)
            ax.set_xticks(xs); ax.set_xticklabels([sname] * len(vals), fontsize=7, color="#888")
            ax.set_title(f"{mlabel} @ {sname}", fontsize=8.5); ax.margins(y=0.22); ax.tick_params(length=0)
            ax.set_yticks([])
            for s in ("top", "right", "left"):
                ax.spines[s].set_visible(False)
    axl = fig.add_subplot(gs[len(METRICS), :])
    sn = {l: specnorms(r, cfg["heads"]) for l, r, _ in arms}
    L = max(len(v) for v in sn.values()); x = np.arange(L); w = 0.8 / len(arms)
    for j, (l, r, c) in enumerate(arms):
        y = sn[l] + [0] * (L - len(sn[l]))
        axl.bar(x + (j - (len(arms) - 1) / 2) * w, y, width=w, color=c, edgecolor="white", linewidth=1, label=l)
    axl.set_title("de-confounded head Lipschitz = weight σmax / head layer (operator, data-independent)", fontsize=8.5)
    axl.set_xticks(x); axl.set_xticklabels([f"L{i+1}" for i in range(L)], fontsize=7.5)
    axl.legend(fontsize=7.5, frameon=False, ncol=len(arms), loc="upper center", bbox_to_anchor=(0.5, -0.12))
    for s in ("top", "right"):
        axl.spines[s].set_visible(False)
    axl.tick_params(length=0)
    fig.suptitle(f"{name.upper()} E17 h-pull — own term @ h vs control (RAW; bars in legend)", fontsize=11, y=0.997)
    out = f"{FIGD}/e17_{name}_hz.png"; fig.savefig(out, dpi=150, bbox_inches="tight", facecolor="white"); plt.close(fig)
    print(f"[e17fig] wrote {out}")


for name, cfg in CFG.items():
    build(name, cfg)
print("[e17fig] done")
