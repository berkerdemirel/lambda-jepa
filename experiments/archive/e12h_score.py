"""H-wave scoring (item 4) — dino gd2 + lejepa f7/f8 + vicreg gv2/gvcls arms vs pre-registered
predictions (E12 §H-WAVE). RAW NUMBERS ONLY + mechanical direction check; NO takeaway (CLAUDE.md
contract). Reuses e12g_hz_figs.orbit_invariance (cos_margin) + e12g_head_lipschitz.seg convention.
CPU / numpy over the @o8 orbit stores + the landed probe/battery CSVs (+ torch for the f8-P4
operator sigma_max on head weights)."""
import json
import os

import numpy as np
import pandas as pd

ROOT = "/nfs/scistore19/locatgrp/bdemirel/ssl_project"
FEAT = f"{ROOT}/features"
STACK = "audit_v1"

# arm, control(s), declared-h tap, GAP sibling tap, head chain (real path), z.final
DINO = dict(arm="in100.dino.s0.e12gd2.ext", gd="in100.dino.s0.e12gd.ext",
            gdc="in100.dino.s0.e12gdc.ext", h="teacher.h.cls", gap="teacher.h.gap",
            chain=["teacher.h.cls", "teacher.z.dino.tap1", "teacher.z.dino.tap2",
                   "teacher.z.dino.bottleneck"])
LEJ = dict(arm="in100.lejepa.s0.e12f7.ext", f2="in100.lejepa.s0.e12f2.ext",
           c1="in100.lejepa.s0.e12c1.ext", h="student.z.embed", zout="student.z.proj.out",
           chain=["student.z.embed", "student.z.proj.tap1", "student.z.proj.tap2",
                  "student.z.proj.out"])


def orbit_cos_margin(run):
    d = f"{FEAT}/{run}/in100.pairs100.v1@{STACK}.o8"
    meta = json.load(open(f"{d}/meta.json"))
    V, spaces = meta["v"], sorted({s.rsplit(".view", 1)[0] for s in meta["spaces"]})
    p = np.random.default_rng(0).permutation(meta["n"]); q = np.roll(p, 1)
    out = {}
    for sp in spaces:
        vs = [np.load(f"{d}/{sp}.view{k}.npy").astype(np.float64) for k in range(V)]
        vs = [x / (np.linalg.norm(x, axis=1, keepdims=True) + 1e-12) for x in vs]
        pc, rc = [], []
        for u in range(V):
            for w in range(u + 1, V):
                a, b = vs[u], vs[w]
                pc.append((a * b).sum(1).mean()); rc.append((a[p] * b[q]).sum(1).mean())
        out[sp] = float(np.mean(pc) - np.mean(rc))                      # cos_margin
    return out


def seg_lip(run, chain):
    d = f"{FEAT}/{run}/in100.pairs100.v1@{STACK}.o8"; V = 8
    def vw(sp): return [np.load(f"{d}/{sp}.view{k}.npy").astype(np.float32) for k in range(V)]
    out = {}
    for sin, sout in zip(chain[:-1], chain[1:]):
        A, B = vw(sin), vw(sout); rs = []
        for u in range(V):
            for w in range(u + 1, V):
                dx = np.linalg.norm(A[u] - A[w], axis=1); dy = np.linalg.norm(B[u] - B[w], axis=1)
                rs.append(dy / (dx + 1e-12))
        out[f"{sin.split('.')[-1]}->{sout.split('.')[-1]}"] = float(np.median(np.concatenate(rs)))
    return out


def probes(run):
    df = pd.read_csv(f"{ROOT}/results/probes/{run}.csv")
    return {(r.space, r.probe): r.val_acc for r in df.itertuples()}


def battery(run, metric):
    df = pd.read_csv(f"{ROOT}/results/battery/{run}.csv")
    df = df[(df.metric == metric) & (df.variant == "raw|full")]
    return {r.space: r.value for r in df.itertuples()}


def diag_read(run, space):  # e12_floor_values convention: diagonal moment-KL to N(0,1)
    X = np.load(f"{FEAT}/{run}/in100.train500.v1L/{space}.npy").astype(np.float64)
    mu, var = X.mean(0), X.var(0).clip(1e-8)
    return round(float(0.5 * (np.mean(mu ** 2) + np.mean(var - 1 - np.log(var)))), 3)


def d(a, b):  # delta in points
    return None if (a is None or b is None) else round((a - b) * 100, 1)


print("=" * 78, "\nH-WAVE SCORING — RAW (dino gd2, lejepa f7). Direction check is MECHANICAL.\n" + "=" * 78)

# ---------------- DINO gd2 ----------------
pa, pgd, pgdc = probes(DINO["arm"]), probes(DINO["gd"]), probes(DINO["gdc"])
knn, lin = "knn_v1_k200", "linear_raw_v2"
print("\n### gd2 (dino, floor CLS+GAP) vs gdc(no floor) / gd(CLS-only) — teacher branch ###")
print("gd2-P1  kNN gain over gdc at BOTH trunk readouts (predict: both > 0):")
for tap in (DINO["h"], DINO["gap"]):
    print(f"   {tap:16s} knn200  gd2 {pa.get((tap,knn)):.4f}  gdc {pgdc.get((tap,knn)):.4f}"
          f"  Δ={d(pa.get((tap,knn)),pgdc.get((tap,knn)))}")
print("gd2-P2  placement asymmetry (gd gave GAP +lin 'floating gain', taxed floored CLS):")
for tap in (DINO["h"], DINO["gap"]):
    print(f"   {tap:16s} lin  gd2 {d(pa.get((tap,lin)),pgdc.get((tap,lin)))}  gd {d(pgd.get((tap,lin)),pgdc.get((tap,lin)))} (vs gdc)"
          f"   | knn gd2 {d(pa.get((tap,knn)),pgdc.get((tap,knn)))}  gd {d(pgd.get((tap,knn)),pgdc.get((tap,knn)))}")
print("gd2-P3  deep-head pocket persists (tap2/bottleneck lin deficit + bottleneck diag-KL):")
er_a, er_gdc = battery(DINO["arm"], "effective_rank"), battery(DINO["gdc"], "effective_rank")
for tap in ("teacher.z.dino.tap2", "teacher.z.dino.bottleneck"):
    print(f"   {tap:26s} lin Δ(gd2-gdc)={d(pa.get((tap,lin)),pgdc.get((tap,lin)))}"
          f"   effrank gd2 {er_a.get(tap)} gdc {er_gdc.get(tap)}")
print(f"   bottleneck diag-KL (teacher): gd2 {diag_read(DINO['arm'],'teacher.z.dino.bottleneck')}"
      f"  gdc {diag_read(DINO['gdc'],'teacher.z.dino.bottleneck')}"
      f"   (student CLS floored tap): gd2 {diag_read(DINO['arm'],'student.z.dino.bottleneck')}"
      f" gdc {diag_read(DINO['gdc'],'student.z.dino.bottleneck')}")
print("gd2-P4  mid-trunk linear (gd was -2.9/-5.2/-3.5 at L03/06/09); gd2 vs gdc:")
for L in ("L03", "L06", "L09"):
    tap = f"teacher.h.cls.{L}"
    print(f"   {tap:20s} lin Δ(gd2-gdc)={d(pa.get((tap,lin)),pgdc.get((tap,lin)))}"
          f"   Δ(gd-gdc)={d(pgd.get((tap,lin)),pgdc.get((tap,lin)))}")

# ---------------- LEJEPA f7 ----------------
pf7, pf2 = probes(LEJ["arm"]), probes(LEJ["f2"])
inv_f7, inv_f2 = orbit_cos_margin(LEJ["arm"]), orbit_cos_margin(LEJ["f2"])
lip_f7, lip_f2 = seg_lip(LEJ["arm"], LEJ["chain"]), seg_lip(LEJ["f2"], LEJ["chain"])
print("\n### f7 (lejepa, floor+inv at embed) vs f2 (floor only) ###")
print(f"f7-P1  inv margin at embed rises vs f2:  embed cos_margin  f7 {inv_f7.get(LEJ['h']):.4f}"
      f"  f2 {inv_f2.get(LEJ['h']):.4f}  Δ={d(inv_f7.get(LEJ['h']),inv_f2.get(LEJ['h']))}")
jf7 = inv_f7.get(LEJ["zout"]) - inv_f7.get(LEJ["h"]); jf2 = inv_f2.get(LEJ["zout"]) - inv_f2.get(LEJ["h"])
print(f"f7-P2  projector inv JUMP (margin@out - margin@embed) shrinks:  f7 {jf7:.4f}  f2 {jf2:.4f}"
      f"  (predict f7 < f2)")
print(f"f7-P3  head empirical Lipschitz (median stretch) <= f2 on embed->out segments:")
for seg in lip_f7:
    print(f"   {seg:16s}  f7 {lip_f7[seg]:.3f}  f2 {lip_f2[seg]:.3f}")
print(f"f7-P4  no probe tax at embed:  knn200 f7 {pf7.get((LEJ['h'],knn)):.4f} f2 {pf2.get((LEJ['h'],knn)):.4f}"
      f" Δ={d(pf7.get((LEJ['h'],knn)),pf2.get((LEJ['h'],knn)))}"
      f"  | lin Δ={d(pf7.get((LEJ['h'],lin)),pf2.get((LEJ['h'],lin)))}")

# ---------------- LEJEPA f8 (shape-dominant: floor:inv = 4.5:1, the f7 mirror) ----------------
F8 = "in100.lejepa.s0.e12f8.ext"


def specnorms(run):  # f8-P4 operator head-Lipschitz: weight sigma_max per projector layer.
    # E12-lane heads are spectral_norm-parametrized (p1/D-026): the effective weight is stored as
    # parametrizations.weight.original0/1 (weight_norm) or .original (spectral_norm) — for the
    # sigma_max readout we reconstruct nothing and read the largest singular value of the raw
    # original matrix for spectral_norm (sigma_max of W/sigma(W) is 1 by construction; the raw
    # original documents the pre-normalization scale, labeled as such).
    import torch
    ck = f"{ROOT}/outputs/{run.replace('.ext', '')}_ep100.pt"
    sd = torch.load(ck, map_location="cpu", weights_only=False)["modules"]
    out = []
    for k, w in sd.get("projector", {}).items():
        if w.ndim != 2:
            continue
        if (k.endswith("weight") and "parametriz" not in k
                or k.endswith("parametrizations.weight.original1")
                or k.endswith("parametrizations.weight.original")):
            tag = "raw-orig" if k.endswith(".original") else ""
            out.append((round(float(torch.linalg.svdvals(w.float())[0]), 2), tag))
    return [f"{v}{'(' + t + ')' if t else ''}" for v, t in out]


if os.path.isdir(f"{FEAT}/{F8}"):
    pf8, pc1 = probes(F8), probes(LEJ["c1"])
    inv_f8, lip_f8 = orbit_cos_margin(F8), seg_lip(F8, LEJ["chain"])
    ep = {r: battery(r, "epps_pulley").get(LEJ["h"]) for r in (F8, LEJ["arm"], LEJ["f2"], LEJ["c1"])}
    kw = {r: battery(r, "kurt_topeig.worst").get(LEJ["h"]) for r in (F8, LEJ["arm"], LEJ["f2"], LEJ["c1"])}
    print("\n### f8 (lejepa, SHAPE-DOMINANT floor:inv=4.5:1) vs f2 (floor) / f7 (inv-dominant) / c1 ###")
    print(f"f8-P1  calibration holds (predict ~f2, not f7):  embed diag-KL  f8 {diag_read(F8, LEJ['h'])}"
          f"  f2 {diag_read(LEJ['f2'], LEJ['h'])}  f7 {diag_read(LEJ['arm'], LEJ['h'])}  c1 {diag_read(LEJ['c1'], LEJ['h'])}")
    print(f"       EP  f8 {ep[F8]:.1f}  f2 {ep[LEJ['f2']]:.1f}  f7 {ep[LEJ['arm']]:.1f}  c1 {ep[LEJ['c1']]:.1f}"
          f"   | kurt_worst  f8 {kw[F8]:.2f}  f2 {kw[LEJ['f2']]:.2f}  f7 {kw[LEJ['arm']]:.2f}  c1 {kw[LEJ['c1']]:.2f}")
    print(f"f8-P2  partial invariance (predict f2 < f8 < f7):  embed cos_margin"
          f"  f8 {inv_f8.get(LEJ['h']):.4f}  f2 {inv_f2.get(LEJ['h']):.4f}  f7 {inv_f7.get(LEJ['h']):.4f}")
    print(f"f8-P3  best-of-both probes (predict >= f2):  embed  lin f8 {pf8.get((LEJ['h'],lin)):.4f}"
          f" (Δf2 {d(pf8.get((LEJ['h'],lin)),pf2.get((LEJ['h'],lin)))}, Δc1 {d(pf8.get((LEJ['h'],lin)),pc1.get((LEJ['h'],lin)))})"
          f"  knn200 f8 {pf8.get((LEJ['h'],knn)):.4f}"
          f" (Δf2 {d(pf8.get((LEJ['h'],knn)),pf2.get((LEJ['h'],knn)))}, Δc1 {d(pf8.get((LEJ['h'],knn)),pc1.get((LEJ['h'],knn)))})")
    jf8 = inv_f8.get(LEJ["zout"]) - inv_f8.get(LEJ["h"])
    print(f"f8-P4  head burden between f2 and f7:  inv-jump  f8 {jf8:.4f}  (f2 {jf2:.4f}, f7 {jf7:.4f})"
          f"   | head σmax/layer  f8 {specnorms(F8)}  f2 {specnorms(LEJ['f2'])}  f7 {specnorms(LEJ['arm'])}")
    print(f"       seg-stretch f8 {lip_f8}")

# ---------------- VICREG gv2 / gvcls (CLS placement matrix; D-036 h = projector-input CLS) ----------------
VIC = dict(gv2="in100.vicreg.s0.e12gv2.ext", gvcls="in100.vicreg.s0.e12gvcls.ext",
           gv="in100.vicreg.s0.e12gv.ext", gvc="in100.vicreg.s0.e12gvc.ext",
           cls="student.h.cls", gap="student.h.gap", zout="student.z.proj.out",
           chain=["student.h.cls", "student.z.proj.tap1", "student.z.proj.tap2",
                  "student.z.proj.out"])
if all(os.path.isdir(f"{FEAT}/{VIC[k]}") for k in ("gv2", "gvcls", "gv", "gvc")):
    pv = {k: probes(VIC[k]) for k in ("gv2", "gvcls", "gv", "gvc")}
    iv = {k: orbit_cos_margin(VIC[k]) for k in ("gv2", "gvcls", "gvc")}
    lv = {k: seg_lip(VIC[k], VIC["chain"]) for k in ("gv2", "gvcls", "gvc")}
    cls, gap, zout = VIC["cls"], VIC["gap"], VIC["zout"]
    print("\n### gv2 (vicreg, floor@CLS) / gvcls (floor+inv@CLS) vs gvc (matched no-floor) — the CLS placement matrix ###")
    print("gv2-P1  T7 tap-local at CLS (predict kNN gain @cls, any lin tax local):")
    print(f"   cls  lin gv2 {pv['gv2'].get((cls,lin)):.4f} (Δgvc {d(pv['gv2'].get((cls,lin)),pv['gvc'].get((cls,lin)))})"
          f"  knn200 gv2 {pv['gv2'].get((cls,knn)):.4f} (Δgvc {d(pv['gv2'].get((cls,knn)),pv['gvc'].get((cls,knn)))})")
    print("gv2-P2  the gv-interpretation stake — audited-GAP readouts vs gvc (gv gave gap +0.5 lin/+1.0 knn):")
    print(f"   gap  lin Δ(gv2-gvc)={d(pv['gv2'].get((gap,lin)),pv['gvc'].get((gap,lin)))}"
          f"  knn Δ={d(pv['gv2'].get((gap,knn)),pv['gvc'].get((gap,knn)))}"
          f"   | gv reference: lin Δ={d(pv['gv'].get((gap,lin)),pv['gvc'].get((gap,lin)))}"
          f" knn Δ={d(pv['gv'].get((gap,knn)),pv['gvc'].get((gap,knn)))}")
    print("gv2-P3  z-side invisibility (T8):")
    print(f"   proj.out  lin Δ(gv2-gvc)={d(pv['gv2'].get((zout,lin)),pv['gvc'].get((zout,lin)))}"
          f"  knn Δ={d(pv['gv2'].get((zout,knn)),pv['gvc'].get((zout,knn)))}")
    print(f"gv2-P4  head stretch real path:  cls->tap1  gv2 {lv['gv2'].get('cls->tap1'):.3f}"
          f"  gvc {lv['gvc'].get('cls->tap1'):.3f}   (gv corrected ref .547)")
    print("gvcls-P1  head burden (predict margin@cls up, inv-jump shrink > gv2's):")
    for k in ("gvc", "gv2", "gvcls"):
        print(f"   {k:5s} margin@cls {iv[k].get(cls):.4f}  inv-jump(zout-cls) {iv[k].get(zout)-iv[k].get(cls):+.4f}")
    print("gvcls-P2  T7 relocation + gap sibling:")
    print(f"   cls  lin Δ(gvcls-gvc)={d(pv['gvcls'].get((cls,lin)),pv['gvc'].get((cls,lin)))}"
          f"  knn Δ={d(pv['gvcls'].get((cls,knn)),pv['gvc'].get((cls,knn)))}"
          f"   | gap  lin Δ={d(pv['gvcls'].get((gap,lin)),pv['gvc'].get((gap,lin)))}"
          f"  knn Δ={d(pv['gvcls'].get((gap,knn)),pv['gvc'].get((gap,knn)))}")
    print(f"gvcls-P3  head stretch:  cls->tap1  gvcls {lv['gvcls'].get('cls->tap1'):.3f}  (<= gv .547 predicted)")
    print(f"gvcls-P4  z-side:  proj.out  lin Δ(gvcls-gvc)={d(pv['gvcls'].get((zout,lin)),pv['gvc'].get((zout,lin)))}"
          f"  knn Δ={d(pv['gvcls'].get((zout,knn)),pv['gvc'].get((zout,knn)))}")
    print("placement contrast gv2-gv (CLS-floor vs GAP-floor, same λ):")
    for tap, nm in ((cls, "cls"), (gap, "gap")):
        print(f"   {nm}  lin Δ(gv2-gv)={d(pv['gv2'].get((tap,lin)),pv['gv'].get((tap,lin)))}"
              f"  knn Δ={d(pv['gv2'].get((tap,knn)),pv['gv'].get((tap,knn)))}")
print("=" * 78)
