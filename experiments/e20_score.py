"""E20 comparison pass (c): the e17_score deep pass re-pointed at the view-lane e20f arms —
each lane's calibrated-floor arm vs its M2 control (NOT the e17c controls; E20's frame).
Head burden = projector inv-jump triple + jump split (Dpos vs -Drand) + operator sigma_max +
seg stretch; h side = cone decomposition, diag-KL, battery tails, probes. Needs the @o8 orbit
stores (extraction jobs 2026-07-19) + train500.v1 (cone/diagKL read v1, not v1L — the L-tap
store was an e17-lane accident, the metric only needs the declared-h space). RAW; no takeaway."""
import json
import os

import numpy as np
import pandas as pd
import torch

ROOT = "/nfs/scistore19/locatgrp/bdemirel/ssl_project"
FEAT = f"{ROOT}/features"
STACK = "audit_v1"
KNN, LIN = "knn_v1_k200", "linear_raw_v2"

# Controls follow the landing convention (e20_centered CELLS): vicreg/dino/lejepa read the
# e17c-flavor extraction (same M2 checkpoint, v1L manifests — the card's headline ctrl
# numbers), simclr/byol the .s0.ext one. The .s0.ext orbit extractions for the other three
# land anyway (flavor cross-check, ~free).
M = {
    "lejepa": dict(ctrl="in100.lejepa.s0.e17c.ext",
                   arms=[("e20f", "in100.lejepa.s0.e20f.ext")],
                   h="student.z.embed", z="student.z.proj.out",
                   chain=["student.z.embed", "student.z.proj.tap1", "student.z.proj.tap2",
                          "student.z.proj.out"], heads=["projector"]),
    "vicreg": dict(ctrl="in100.vicreg.s0.e17c.ext",
                   arms=[("e20f", "in100.vicreg.s0.e20f.ext")],
                   h="student.h.cls", z="student.z.proj.out",
                   chain=["student.h.cls", "student.z.proj.tap1", "student.z.proj.tap2",
                          "student.z.proj.out"], heads=["projector"]),
    "simclr": dict(ctrl="in100.simclr.s0.ext",
                   arms=[("e20f", "in100.simclr.s0.e20f.ext")],
                   h="student.h.cls", z="student.z.proj.out",
                   chain=["student.h.cls", "student.z.proj.tap1", "student.z.proj.out"],
                   heads=["projector"]),
    "byol": dict(ctrl="in100.byol.s0.ext",
                 arms=[("e20f", "in100.byol.s0.e20f.ext")],
                 h="student.h.cls", z="student.z.pred.out",
                 chain=["student.h.cls", "student.z.proj.tap1", "student.z.proj.out",
                        "student.z.pred.tap1", "student.z.pred.out"],
                 heads=["projector", "predictor"]),
    "dino": dict(ctrl="in100.dino.s0.e17c.ext",
                 arms=[("e20f", "in100.dino.s0.e20f.ext")],
                 h="teacher.h.cls", z="teacher.z.dino.bottleneck",
                 chain=["teacher.h.cls", "teacher.z.dino.tap1", "teacher.z.dino.tap2",
                        "teacher.z.dino.bottleneck"], heads=["teacher_projector"]),
}


def have(run):
    return os.path.isdir(f"{FEAT}/{run}/in100.pairs100.v1@{STACK}.o8")


def orbit(run, spaces):
    d = f"{FEAT}/{run}/in100.pairs100.v1@{STACK}.o8"
    m = json.load(open(f"{d}/meta.json")); V = m["v"]
    p = np.random.default_rng(0).permutation(m["n"]); q = np.roll(p, 1); out = {}
    for sp in spaces:
        vs = [np.load(f"{d}/{sp}.view{k}.npy").astype(np.float64) for k in range(V)]
        vs = [x / (np.linalg.norm(x, axis=1, keepdims=True) + 1e-12) for x in vs]
        pc, rc = [], []
        for u in range(V):
            for w in range(u + 1, V):
                pc.append((vs[u] * vs[w]).sum(1).mean()); rc.append((vs[u][p] * vs[w][q]).sum(1).mean())
        pos, rand = float(np.mean(pc)), float(np.mean(rc))
        out[sp] = dict(pos=pos, rand=rand, margin=pos - rand)
    return out


def seg_lip(run, chain):
    d = f"{FEAT}/{run}/in100.pairs100.v1@{STACK}.o8"; V = 8
    vw = lambda sp: [np.load(f"{d}/{sp}.view{k}.npy").astype(np.float32) for k in range(V)]
    out = {}
    for sin, sout in zip(chain[:-1], chain[1:]):
        if not (os.path.exists(f"{d}/{sin}.view0.npy") and os.path.exists(f"{d}/{sout}.view0.npy")):
            continue
        A, B = vw(sin), vw(sout); rs = []
        for u in range(V):
            for w in range(u + 1, V):
                dx = np.linalg.norm(A[u] - A[w], axis=1); dy = np.linalg.norm(B[u] - B[w], axis=1)
                rs.append(dy / (dx + 1e-12))
        key = f"{'.'.join(sin.split('.')[-2:])}->{'.'.join(sout.split('.')[-2:])}"
        out[key] = float(np.median(np.concatenate(rs)))    # 2-component names: byol's proj/pred taps collide on the last component alone
    return out


def specnorms(run, heads):
    ck = f"{ROOT}/outputs/{run.replace('.e17c.ext', '').replace('.ext', '')}_ep100.pt"
    sd = torch.load(ck, map_location="cpu", weights_only=False)["modules"]
    out = []
    for hkey in heads:
        for k, wt in sd.get(hkey, {}).items():
            if wt.ndim == 2 and (k.endswith("weight") and "parametriz" not in k
                                 or k.endswith("parametrizations.weight.original1")):
                out.append(round(float(torch.linalg.svdvals(wt.float())[0]), 3))
    return out


def _train_store(run):
    for fl in ("in100.train500.v1", "in100.train500.v1L"):
        if os.path.isdir(f"{FEAT}/{run}/{fl}"):
            return fl
    raise FileNotFoundError(f"no train500 store for {run}")


def diag_kl(run, sp):
    X = np.load(f"{FEAT}/{run}/{_train_store(run)}/{sp}.npy").astype(np.float64)
    mu, var = X.mean(0), X.var(0).clip(1e-8)
    return round(float(0.5 * (np.mean(mu ** 2) + np.mean(var - 1 - np.log(var)))), 3)


def cone(run, sp):
    X = np.load(f"{FEAT}/{run}/{_train_store(run)}/{sp}.npy").astype(np.float64)
    ec = lambda A: float((np.linalg.norm((A / np.linalg.norm(A, axis=1, keepdims=True)).sum(0)) ** 2
                          - len(A)) / (len(A) * (len(A) - 1)))
    mu = X.mean(0)
    return {"rand": round(ec(X), 3), "centered": round(ec(X - mu), 3),
            "mu_share": round(float(mu @ mu / (np.linalg.norm(X, axis=1) ** 2).mean()), 3)}


def probes(run):
    df = pd.read_csv(f"{ROOT}/results/probes/{run}.csv")
    return {(r.space, r.probe): r.val_acc for r in df.itertuples()}


def bat(run, sp, metric):
    df = pd.read_csv(f"{ROOT}/results/battery/{run}.csv")
    df = df[(df.space == sp) & (df.metric == metric) & (df.variant == "raw|full")]
    return float(df.value.iloc[0]) if len(df) else None


def dl(a, b):
    return None if a is None or b is None else round((a - b) * 100, 1)


print("=" * 80, "\nE20 view-lane SCORING — RAW (e20f arm vs M2 control). Direction check MECHANICAL.\n" + "=" * 80)
for name, cfg in M.items():
    landed = [(t, r) for t, r in cfg["arms"] if have(r)]
    if not have(cfg["ctrl"]) or not landed:
        print(f"\n### {name}: waiting ({sum(have(r) for _, r in cfg['arms'])}/{len(cfg['arms'])} arms, "
              f"ctrl={'ok' if have(cfg['ctrl']) else 'MISSING'})"); continue
    h, z = cfg["h"], cfg["z"]
    oc = orbit(cfg["ctrl"], [h, z]); pc = probes(cfg["ctrl"])
    print(f"\n### {name}  (h={h}  z.final={z}) ###")
    print(f"  control  inv-jump(margin@z-margin@h)={oc[z]['margin']-oc[h]['margin']:+.3f}  "
          f"[h: pos {oc[h]['pos']:.3f} rand {oc[h]['rand']:.3f} margin {oc[h]['margin']:.3f}]  "
          f"headσmax={specnorms(cfg['ctrl'], cfg['heads'])}  diagKL@h={diag_kl(cfg['ctrl'], h)}")
    print(f"           [z: pos {oc[z]['pos']:.3f} rand {oc[z]['rand']:.3f}]  jump-split "
          f"(Δpos {oc[z]['pos']-oc[h]['pos']:+.3f}, -Δrand {-(oc[z]['rand']-oc[h]['rand']):+.3f})"
          f"  cone@h {cone(cfg['ctrl'], h)}")
    for t, r in landed:
        oa, pa = orbit(r, [h, z]), probes(r)
        jump_a, jump_c = oa[z]["margin"] - oa[h]["margin"], oc[z]["margin"] - oc[h]["margin"]
        print(f"  {t}:")
        print(f"    HEAD BURDEN  inv-jump {jump_a:+.3f} (ctrl {jump_c:+.3f}, Δ={jump_a-jump_c:+.3f})"
              f"  | h-triple pos {oa[h]['pos']:.3f} rand {oa[h]['rand']:.3f} margin {oa[h]['margin']:.3f}"
              f" (Δmargin {dl(oa[h]['margin'],oc[h]['margin'])})")
        print(f"    [z: pos {oa[z]['pos']:.3f} rand {oa[z]['rand']:.3f}]  jump-split "
              f"(Δpos {oa[z]['pos']-oa[h]['pos']:+.3f}, -Δrand {-(oa[z]['rand']-oa[h]['rand']):+.3f})"
              f"  cone@h {cone(r, h)}")
        print(f"    head σmax/layer {specnorms(r, cfg['heads'])}  (ctrl {specnorms(cfg['ctrl'], cfg['heads'])})"
              f"  | seg-stretch {seg_lip(r, cfg['chain'])}")
        print(f"    diagKL@h {diag_kl(r, h)} (ctrl {diag_kl(cfg['ctrl'], h)})  | "
              f"kurt_worst {bat(r,h,'kurt_topeig.worst')} EP {bat(r,h,'epps_pulley')} effrank {bat(r,h,'effective_rank')}")
        print(f"    PROBES@h  lin {pa.get((h,LIN))} (Δ{dl(pa.get((h,LIN)),pc.get((h,LIN)))})  "
              f"knn200 {pa.get((h,KNN))} (Δ{dl(pa.get((h,KNN)),pc.get((h,KNN)))})")
print("=" * 80)
