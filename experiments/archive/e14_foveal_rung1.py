"""E14 — PIVOT rung-1 (D-031): does D_read still rank on the DECLARED foveal channel?

Reuses E13's estimator machinery (imported; PAIRS key repointed to foveal_v1) over the E14
extractions: per checkpoint and per overlap stratum (far/near/copy, recomputed deterministically
from sslgap.data.foveal_boxes), ridge from event-h onto the RFF sketch of the frozen tokenizer's
sharp-context descriptor. PRIMARY metric = dof-matched d_read (E13-T3b); primary cell =
far / D_m=1024 / sigma=sigma_med / T=mae. T=randinit leakage control; T=dino teacher-CLS
specificity control (primary cell, descriptive). Pre-registration: docs/experiments/
E14_foveal_rung1.md. Pure function over stored arrays; numbers land raw.
"""
import argparse
import csv
import os
import sys

import numpy as np
from scipy.spatial.distance import pdist

sys.path.insert(0, "/nfs/scistore19/locatgrp/bdemirel/ssl_project")
import e13_pivot_rung0 as e13
from sslgap.data import foveal_boxes, foveal_iou, read_manifest

ROOT = e13.ROOT
e13.PAIRS = "in100.pairs100.v1@foveal_v1"          # member/event side loads (Side, labels)
CTX = "in100.pairs100.v1@foveal_v1_ctx"            # tokenizer/target side
MANIFEST = f"{ROOT}/features/manifests/in100.pairs100.v1.csv"

RANKED, GATE, RANDINIT, LEJEPA_SUB = e13.RANKED, e13.GATE, e13.RANDINIT, e13.LEJEPA_SUB
DEITLITE = "in100.deitlite.s0.ext"
TOKENIZERS = {"mae": ("in100.mae.s0.ext", "student.h.gap"),
              "randinit": (RANDINIT, "student.h.gap"),
              "dino": ("in100.dino.s0.ext", "teacher.h.cls")}
PRIMARY = (1024, 1.0)
CELLS_FAR = [(256, 0.5), (256, 1.0), (256, 2.0), (1024, 0.5), (1024, 1.0), (1024, 2.0), (0, -1.0)]
STRATA = ("far", "near", "copy")
DOF_STAR = e13.DOF_STAR


def load_ctx(rid, space, view):
    x = np.load(f"{ROOT}/features/{rid}/{CTX}/{space}.view{view}.npy", mmap_mode="r")
    return np.asarray(x, dtype=np.float32)


def fit_eval_dof(S, Phi, folds, n_tr):
    """E13's Side.fit_eval with the dof-matched alpha PRIMARY: returns its test residuals for
    bootstrap (E13 bootstrapped the val-selected fit; E14's primary is dof*=256)."""
    itr, iva, ite = folds
    ytr = Phi[itr]
    ybar = ytr.mean(0, dtype=np.float64).astype(np.float32)
    G = S.U.T @ (ytr - ybar)
    Yf = {"va": Phi[iva] - ybar, "te": Phi[ite] - ybar}
    den = {k: float((Yf[k] ** 2).sum(dtype=np.float64)) for k in Yf}

    def dist(fold, alpha):
        shrink = (S.s / (S.s ** 2 + alpha)).astype(np.float32)
        E = S.P[fold] @ (shrink[:, None] * G) - Yf[fold]
        return float(np.einsum("ij,ij->", E, E, dtype=np.float64) / den[fold]), E

    va = [dist("va", c * n_tr)[0] for c in e13.ALPHA_C]
    best = int(np.argmin(va))
    d_read, _ = dist("te", e13.ALPHA_C[best] * n_tr)
    d_read_dof, E_te = dist("te", S.alpha_for_dof(DOF_STAR))
    res_img = np.einsum("ij,ij->i", E_te, E_te, dtype=np.float64)
    den_img = (Yf["te"] ** 2).sum(1, dtype=np.float64)
    return {"d_read": d_read, "d_read_dof": d_read_dof, "r2_val_best": 1 - va[best],
            "alpha_c": e13.ALPHA_C[best], "alpha_edge": int(best in (0, len(e13.ALPHA_C) - 1))}, res_img, den_img


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry", action="store_true")
    ap.add_argument("--blur", action="store_true", help="E14 addendum: no-fovea null events")
    args = ap.parse_args()
    sfx = "_blur" if args.blur else ""
    if args.blur:
        e13.PAIRS = "in100.pairs100.v1@blur_v1"    # events only; ctx targets reused from foveal

    ranked = dict(RANKED) if not args.dry else \
        {r: RANKED[r] for r in ("in100.dino.s0.ext", "in100.lejepa.s0.e12c1.ext")}
    tokenizers = TOKENIZERS if not args.dry else {"mae": TOKENIZERS["mae"]}
    if args.blur:
        tokenizers = {k: TOKENIZERS[k] for k in ("mae", "dino") if k in tokenizers or not args.dry}

    # ---- strata from the deterministic channel (single source of truth: foveal_boxes) ----
    refs = [r for r, _ in read_manifest(MANIFEST)]
    y = np.load(f"{ROOT}/features/{next(iter(ranked))}/{e13.PAIRS}/labels.npy")
    assert len(refs) == len(y)
    for rid in list(ranked) + list(GATE):
        assert np.array_equal(y, np.load(f"{ROOT}/features/{rid}/{e13.PAIRS}/labels.npy")), rid

    sidx, iou = {s: [] for s in STRATA}, {s: [] for s in STRATA}
    for i, ref in enumerate(refs):
        s, a, b = foveal_boxes(ref, "foveal_v1")
        sidx[s].append(i)
        iou[s].append(foveal_iou(a, b, 96))
    folds_s, srows = {}, []
    for s in STRATA:
        idx = np.asarray(sidx[s])
        tr, va, te = e13.splits(y[idx])                      # class-stratified within stratum
        folds_s[s] = (idx[tr], idx[va], idx[te])             # global row indices
        q = np.quantile(iou[s], [0, .25, .5, .75, 1])
        srows.append({"stratum": s, "n": len(idx), "n_tr": len(tr), "n_va": len(va), "n_te": len(te),
                      "iou_mean": round(float(np.mean(iou[s])), 4),
                      **{f"iou_q{int(p * 100)}": round(float(v), 4) for p, v in zip([0, .25, .5, .75, 1], q)}})
    print("[e14] strata:", srows, flush=True)

    cells_for = {"mae": {"far": CELLS_FAR, "near": [PRIMARY], "copy": [PRIMARY]},
                 "randinit": {"far": [PRIMARY]}, "dino": {"far": [PRIMARY]}}
    if args.blur:                                   # addendum: far primary cell only
        cells_for = {"mae": {"far": [PRIMARY]}, "dino": {"far": [PRIMARY]}}
    if args.dry and not args.blur:
        cells_for["mae"] = {"far": [PRIMARY, (0, -1.0)], "near": [PRIMARY], "copy": [PRIMARY]}

    dist_rows, audit_rows, gate_boot = [], [], {}
    side_cache = {}

    for tk, (t_rid, t_space) in tokenizers.items():
        for dirn, (v_in, v_out) in {"A2B": ("A", "B"), "B2A": ("B", "A")}.items():
            members = {**ranked, **{r: sp for r, sp in GATE.items() if r != t_rid or tk == "randinit"}}
            Yraw = load_ctx(t_rid, t_space, v_out)
            Y0 = {s: e13.standardize(Yraw, folds_s[s][0]) for s in STRATA}
            pool = np.concatenate([Y0[s][folds_s[s][0]] for s in STRATA])   # sigma_med: pooled, stratum-agnostic
            sub = np.random.default_rng(1).choice(len(pool), size=2048, replace=False)
            sigma0 = float(np.median(pdist(pool[sub].astype(np.float64))))
            print(f"[e14] T={tk} dir={dirn} sigma0={sigma0:.3f} members={len(members)}", flush=True)

            for stratum in cells_for[tk]:
                folds = folds_s[stratum]
                itr, iva, ite = folds
                n_tr = len(itr)
                for ci, (dm, mult) in enumerate(cells_for[tk][stratum]):
                    seed = 1400 + ci
                    if dm == 0:
                        Phi, sig = Y0[stratum], -1.0
                    else:
                        rng = np.random.default_rng(seed)   # per-cell seed shared across strata/tokenizers
                        sig = mult * sigma0
                        W = (rng.standard_normal((Yraw.shape[1], dm)) / sig).astype(np.float32)
                        b = rng.uniform(0, 2 * np.pi, dm).astype(np.float32)
                        Phi = np.sqrt(2.0 / dm) * np.cos(Y0[stratum] @ W + b)
                    od = e13.offdiag(Phi[ite] @ Phi[ite].T).astype(np.float64)
                    is_primary = (dm, mult) == PRIMARY and stratum == "far"
                    arow = {"tokenizer": tk, "dir": dirn, "stratum": stratum, "D_m": dm, "sig_mult": mult,
                            "sigma": round(sig, 4), "sigma0": round(sigma0, 4),
                            "kern_offdiag_mean": round(float(od.mean()), 4), "kern_offdiag_sd": round(float(od.std()), 4),
                            "kern_offdiag_p05": round(float(np.quantile(od, .05)), 4),
                            "kern_offdiag_p95": round(float(np.quantile(od, .95)), 4),
                            "sketch_effrank_sub2048": round(e13.effrank_sub(Phi, itr), 2),
                            "target_bt": round(e13.bt_ratio(Phi, y, itr), 4),
                            "target_kmeans100_nmi": round(e13.kmeans_nmi(Phi, y, itr), 4) if (is_primary or dm == 0) else ""}
                    audit_rows.append(arow)
                    if args.dry:
                        print("  audit:", arow, flush=True)

                    t_rank = e13.zrank(od) if is_primary else None      # D_kern: primary-cell descriptive only
                    Pc = Phi[ite] - Phi[ite].mean(0)
                    pnorm = float(np.linalg.norm((Pc @ Pc.T).astype(np.float64)))

                    for rid, h in members.items():
                        key = (dirn, rid, stratum)
                        if key not in side_cache:
                            side_cache[key] = e13.Side(rid, h, v_in, *folds)
                        S = side_cache[key]
                        m, res_img, den_img = fit_eval_dof(S, Phi, folds, n_tr)
                        row = {"tokenizer": tk, "dir": dirn, "stratum": stratum, "D_m": dm, "sig_mult": mult,
                               "run": rid, "ranked": int(rid in ranked), "self_ref": int(rid == t_rid),
                               "d_read": round(m["d_read"], 5), "d_read_dof": round(m["d_read_dof"], 5),
                               "r2_val_best": round(m["r2_val_best"], 5), "alpha_c": m["alpha_c"],
                               "alpha_edge": m["alpha_edge"], "d_kern": "", "cka": ""}
                        if is_primary:
                            d_kern, cka = S.kern(t_rank, Pc, pnorm)
                            row["d_kern"], row["cka"] = round(d_kern, 5), round(cka, 5)
                            if tk == "mae":
                                br = np.random.default_rng(3).integers(0, len(ite), (e13.N_BOOT, len(ite)))
                                gate_boot.setdefault(rid, []).append(
                                    float((res_img[br].sum(1) / den_img[br].sum(1)).std()))
                        dist_rows.append(row)
                        if args.dry:
                            print("   ", row, flush=True)

    if args.dry:
        print("[e14] DRY RUN complete — nothing written.", flush=True)
        return

    os.makedirs(f"{ROOT}/results/diag", exist_ok=True)
    for name, rows in ((f"e14_distortion{sfx}", dist_rows), (f"e14_target_audit{sfx}", audit_rows),
                       ("e14_strata", srows)):
        with open(f"{ROOT}/results/diag/{name}.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader(); w.writerows(rows)
        print(f"wrote results/diag/{name}.csv ({len(rows)} rows)", flush=True)

    # ---- rank-correlation stage (n19 primary per E13-T3a; deitlite descriptive) ----
    probe = {rid: e13.probe_rows(rid, h) for rid, h in RANKED.items()}
    probe[RANDINIT] = e13.probe_rows(RANDINIT, GATE[RANDINIT])
    base, bkeys = ({}, []) if args.blur else \
        e13.read_baselines({r: h for r, h in ranked.items() if r != DEITLITE})
    sym = {}
    for r in dist_rows:
        sym.setdefault((r["tokenizer"], r["stratum"], r["D_m"], r["sig_mult"], r["run"]), []).append(r)

    def symval(tk, st, dm, mult, rid, met):
        rows = sym.get((tk, st, dm, mult, rid), [])
        if len(rows) != 2 or any(r[met] == "" for r in rows):
            return None
        v = float(np.mean([float(r[met]) for r in rows]))
        return (-v if met == "cka" else v) if np.isfinite(v) else None

    n19 = [r for r in ranked if r != DEITLITE]
    zoos = {"full19": n19, "full20_anchor_desc": list(ranked),
            "lejepa11": [r for r in n19 if r in LEJEPA_SUB],
            "nonlejepa8": [r for r in n19 if r not in LEJEPA_SUB]}
    corr_rows = []

    def add_corr(ranker, cell_tag, vals, extra_zoos=None):
        for zname, zrids in {**zoos, **(extra_zoos or {})}.items():
            rids = [r for r in zrids if r in vals and r in probe]
            if len(rids) < 5:
                continue
            for pb in e13.PROBES:
                rho, p = e13.spearman_perm([vals[r] for r in rids], [probe[r][pb] for r in rids])
                corr_rows.append({"ranker": ranker, "cell": cell_tag, "zoo": zname, "n": len(rids),
                                  "probe": pb, "rho": round(rho, 4), "perm_p": round(p, 5)})

    for tk in tokenizers:
        for stratum, cells in cells_for[tk].items():
            for dm, mult in cells:
                tag = f"T={tk}|{stratum}|D_m={dm}|s={mult}"
                mets = ("d_read_dof", "d_read") + (("d_kern", "cka") if (dm, mult) == PRIMARY and stratum == "far" else ())
                for met in mets:
                    vals = {r: symval(tk, stratum, dm, mult, r, met) for r in ranked}
                    vals = {r: v for r, v in vals.items() if v is not None}
                    if tk == "dino":                          # self-ref out; paired T=mae on same subset (P6)
                        vals.pop(TOKENIZERS["dino"][0], None)
                        if met in ("d_read_dof", "d_read"):
                            mae_sub = {r: symval("mae", stratum, dm, mult, r, met) for r in vals}
                            add_corr(f"{met}@mae|subset:dino18", tag,
                                     {r: v for r, v in mae_sub.items() if v is not None})
                    if tk == "mae" and (dm, mult) == PRIMARY and stratum == "far" and met == "d_read_dof":
                        vr = symval(tk, stratum, dm, mult, RANDINIT, met)
                        if vr is not None:      # extended vals only add rids to the extra zoo
                            add_corr(met, tag, dict(vals, **{RANDINIT: vr}),
                                     {"full19+randinit": n19 + [RANDINIT]})
                            continue
                    add_corr(met, tag, vals)

    for bk in bkeys:
        vals = {r: base[r][bk] for r in n19 if bk in base.get(r, {})}
        add_corr(f"battery:{bk}", "train500@h", vals)
        if len(vals) < len(n19):
            prim = {r: symval("mae", "far", *PRIMARY, r, "d_read_dof") for r in vals}
            add_corr(f"d_read_dof@primary|subset:{bk}", "pairing",
                     {r: v for r, v in prim.items() if v is not None})

    with open(f"{ROOT}/results/diag/e14_rank_corr{sfx}.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(corr_rows[0]))
        w.writeheader(); w.writerows(corr_rows)
    print(f"wrote results/diag/e14_rank_corr{sfx}.csv ({len(corr_rows)} rows)", flush=True)
    if args.blur:            # levels/gate comparison vs the foveal run happens in analysis
        return

    # ---- gates (P1/K1) + strata ordering (P3) + levels vs E13 (shift cost) ----
    prim_read = {r: symval("mae", "far", *PRIMARY, r, "d_read_dof") for r in list(ranked) + [RANDINIT]}
    trained = np.array([prim_read[r] for r in ranked])
    se_med = float(np.median([np.mean(v) for v in gate_boot.values()]))
    order_ok = 0
    lev_rows = []
    e13_prim = {}
    if os.path.exists(f"{ROOT}/results/diag/e13_distortion.csv"):
        with open(f"{ROOT}/results/diag/e13_distortion.csv") as f:
            acc = {}
            for r in csv.DictReader(f):
                if r["tokenizer"] == "mae" and r["D_m"] == "1024" and r["sig_mult"] == "1.0":
                    acc.setdefault(r["run"], []).append(float(r["d_read_dof"]))
            e13_prim = {r: float(np.mean(v)) for r, v in acc.items() if len(v) == 2}
    for rid in list(ranked) + [RANDINIT]:
        vals = {s: symval("mae", s, *PRIMARY, rid, "d_read_dof") for s in STRATA}
        mono = int(vals["copy"] < vals["near"] < vals["far"]) if None not in vals.values() else ""
        order_ok += (mono == 1 and rid in ranked)
        lev_rows.append({"run": rid, "ranked": int(rid in ranked),
                         **{f"d_read_dof_{s}": round(vals[s], 5) if vals[s] is not None else "" for s in STRATA},
                         "p3_mono": mono, "e13_augview_primary": round(e13_prim[rid], 5) if rid in e13_prim else "",
                         "delta_vs_e13": round(vals["far"] - e13_prim[rid], 5)
                                         if rid in e13_prim and vals["far"] is not None else ""})
    with open(f"{ROOT}/results/diag/e14_levels.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(lev_rows[0]))
        w.writeheader(); w.writerows(lev_rows)
    grow = {"randinit_d_read_dof": round(prim_read[RANDINIT], 5),
            "trained_min": round(float(trained.min()), 5), "trained_max": round(float(trained.max()), 5),
            "trained_mean": round(float(trained.mean()), 5), "trained_sd": round(float(trained.std()), 5),
            "randinit_margin_sd": round(float((prim_read[RANDINIT] - trained.mean()) / (trained.std() + 1e-12)), 2),
            "n_trained_worse_than_randinit": int((trained >= prim_read[RANDINIT]).sum()),
            "boot_se_median": round(se_med, 6),
            "dynamic_range_ratio": round(float(trained.std() / (se_med + 1e-12)), 1),
            "p3_mono_count": order_ok, "p3_mono_of": len(ranked)}
    with open(f"{ROOT}/results/diag/e14_gate.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(grow))
        w.writeheader(); w.writerow(grow)
    print("gate:", grow, flush=True)


if __name__ == "__main__":
    main()
