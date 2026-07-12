"""E13 — PIVOT rung-0 (D-029): does held-out predictive-state distortion rank our zoo?

Zero-training test of the PIVOT proposal's load-bearing premise (its predictions 7/19) over stored
`in100.pairs100.v1@audit_v1` features: per checkpoint, ridge-regress viewA h onto an RFF sketch of
a frozen tokenizer's viewB descriptor (D_read, information meter) and align the raw h Gram to the
target sketch kernel (D_kern, deployed-geometry meter — the pre-registered refinement: PIVOT's
"distortion" conflates the two). Spearman vs the converged D-020 h-probe table, against the locked
battery/moment baselines (E01-T15 = the bar). Full pre-registration: docs/experiments/
E13_pivot_rung0.md. Pure function over stored arrays; numbers land raw.
"""
import argparse
import csv
import os

import numpy as np
from scipy.spatial.distance import pdist
from scipy.stats import rankdata
from sklearn.cluster import KMeans
from sklearn.metrics import normalized_mutual_info_score

ROOT = "/nfs/scistore19/locatgrp/bdemirel/ssl_project"
PAIRS = "in100.pairs100.v1@audit_v1"
TRAIN500 = "in100.train500.v1"

E12 = [f"in100.lejepa.s0.e12{a}.ext" for a in ("a1", "a2", "a3", "c1", "f1", "f2", "f3", "f4", "f5", "f6")]
RANKED = {
    "in100.dino.s0.ext": "teacher.h.cls",
    "in100.simclr.s0.ext": "student.h.gap",
    "in100.byol.s0.ext": "student.h.gap",
    "in100.vicreg.s0.ext": "student.h.gap",
    "in100.ijepa.s0.ext": "teacher.h.gap",
    "in100.lejepa.s0.ext": "student.z.embed",
    "in100.deitlite.s0.ext": "student.h.cls",
    "in100.dino-ctrl.ep25.ext": "teacher.h.cls",
    "in100.dino-ctrl.ep50.ext": "teacher.h.cls",
    "in100.dino-ctrl.ep100.ext": "teacher.h.cls",
    **{r: "student.z.embed" for r in E12},
}
GATE = {"in100.randinit-s0.ext": "student.h.cls", "in100.mae.s0.ext": "student.h.gap"}
RANDINIT = "in100.randinit-s0.ext"
LEJEPA_SUB = set(E12) | {"in100.lejepa.s0.ext"}
TOKENIZERS = {"mae": ("in100.mae.s0.ext", "student.h.gap"),
              "randinit": (RANDINIT, "student.h.gap")}

D_M = [64, 256, 1024, 4096]
SIG_MULT = [0.25, 0.5, 1.0, 2.0, 4.0]
CELLS = [(d, m) for d in D_M for m in SIG_MULT] + [(0, -1.0)]  # (0, -1.0) = raw-descriptor arm
PRIMARY = (1024, 1.0)
ALPHA_C = [1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0]
DOF_STAR = 256
PROBES = ("linear_raw_v2", "knn_v1_k200")
N_PERM, N_BOOT = 20000, 200
_PERM_CACHE = {}


def load_pairs(rid, space, view):
    x = np.load(f"{ROOT}/features/{rid}/{PAIRS}/{space}.view{view}.npy", mmap_mode="r")
    return np.asarray(x, dtype=np.float32)


def splits(labels, seed=0):
    rng = np.random.default_rng(seed)
    tr, va, te = [], [], []
    for c in np.unique(labels):
        idx = rng.permutation(np.where(labels == c)[0])
        n = len(idx)
        tr += list(idx[: int(0.6 * n)]); va += list(idx[int(0.6 * n): int(0.8 * n)]); te += list(idx[int(0.8 * n):])
    return np.sort(tr), np.sort(va), np.sort(te)


def standardize(X, itr):
    mu = X[itr].mean(0, dtype=np.float64)
    sd = X[itr].std(0, dtype=np.float64).clip(1e-6)
    return ((X - mu) / sd).astype(np.float32)


def zrank(v):
    r = rankdata(v).astype(np.float32)
    r -= r.mean()
    nrm = np.linalg.norm(r)
    return None if nrm < 1e-9 else r / nrm


def spearman_perm(x, y):
    rx, ry = zrank(np.asarray(x, float)), zrank(np.asarray(y, float))
    if rx is None or ry is None:
        return 0.0, 1.0
    rho = float(rx @ ry)
    n = len(rx)
    if n not in _PERM_CACHE:
        rng = np.random.default_rng(2)
        _PERM_CACHE[n] = np.stack([rng.permutation(n) for _ in range(N_PERM)])
    perms = rx[_PERM_CACHE[n]]
    p = float((1 + (np.abs(perms @ ry) >= abs(rho) - 1e-12).sum()) / (1 + N_PERM))
    return rho, p


def offdiag(G):
    return G[np.triu_indices(len(G), k=1)]


def effrank_sub(Phi, itr, seed=4):
    sub = np.random.default_rng(seed).choice(itr, size=min(2048, len(itr)), replace=False)
    P = Phi[sub] - Phi[sub].mean(0)
    s = np.linalg.svd(P, compute_uv=False).astype(np.float64) ** 2
    return float(s.sum() ** 2 / (s ** 2).sum())


def bt_ratio(Phi, y, itr):
    P, yy = Phi[itr], y[itr]
    mu = P.mean(0, dtype=np.float64)
    tot = ((P - mu) ** 2).sum(1, dtype=np.float64).mean()
    btw = sum((yy == c).mean() * ((P[yy == c].mean(0, dtype=np.float64) - mu) ** 2).sum() for c in np.unique(yy))
    return float(btw / tot)


def kmeans_nmi(Phi, y, itr, k=100):
    P = Phi[itr] - Phi[itr].mean(0)
    _, _, Vt = np.linalg.svd(P, full_matrices=False)
    z = KMeans(n_clusters=k, n_init=4, random_state=0).fit_predict((P @ Vt[:128].T).astype(np.float32))
    return float(normalized_mutual_info_score(y[itr], z))


def sliced_kl(X, draws=32, d=128, eps=1e-4, n=4096, seed=0):
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(draws):
        idx = rng.choice(len(X), size=n, replace=False)
        Q, _ = np.linalg.qr(rng.standard_normal((X.shape[1], d)))
        p = X[idx] @ Q
        mu = p.mean(0)
        pc = p - mu
        cov = pc.T @ pc / (n - 1) + eps * np.eye(d)
        _, logdet = np.linalg.slogdet(cov)
        vals.append(0.5 * (np.trace(cov) + mu @ mu - d - logdet) / d)
    return float(np.mean(vals))


def diag_kl(X):
    mu = X.mean(0)
    var = X.var(0).clip(1e-8)
    return float(0.5 * (np.mean(mu ** 2) + np.mean(var - 1 - np.log(var))))


class Side:
    """Per-(direction, run) cache: standardized design SVD + Gram ranks of raw test-fold h."""

    def __init__(self, rid, space, view, itr, iva, ite):
        X = load_pairs(rid, space, view)
        Xs = standardize(X, itr)
        U, s, Vt = np.linalg.svd(Xs[itr], full_matrices=False)
        self.U, self.s = U, s.astype(np.float64)
        self.P = {"va": Xs[iva] @ Vt.T, "te": Xs[ite] @ Vt.T}
        Xc = X[ite].astype(np.float64) - X[itr].mean(0, dtype=np.float64)
        Xn = (Xc / (np.linalg.norm(Xc, axis=1, keepdims=True) + 1e-12)).astype(np.float32)
        self.gram_rank = zrank(offdiag(Xn @ Xn.T))
        self.Xte_c = (X[ite] - X[ite].mean(0, dtype=np.float64)).astype(np.float32)
        self.xnorm = float(np.linalg.norm((self.Xte_c.T @ self.Xte_c).astype(np.float64)))

    def dof(self, a):
        return float((self.s ** 2 / (self.s ** 2 + a)).sum())

    def alpha_for_dof(self, target):
        lo, hi = 1e-10, 1e12
        for _ in range(80):
            mid = np.sqrt(lo * hi)
            lo, hi = (mid, hi) if self.dof(mid) > target else (lo, mid)
        return np.sqrt(lo * hi)

    def fit_eval(self, Phi, folds, n_tr):
        itr, iva, ite = folds
        ytr = Phi[itr]
        ybar = ytr.mean(0, dtype=np.float64).astype(np.float32)
        G = self.U.T @ (ytr - ybar)  # centered targets = ridge with intercept (dry-run fix)
        Yf = {"va": Phi[iva] - ybar, "te": Phi[ite] - ybar}
        den = {k: float((Yf[k] ** 2).sum(dtype=np.float64)) for k in Yf}

        def dist(fold, alpha):
            shrink = (self.s / (self.s ** 2 + alpha)).astype(np.float32)
            E = self.P[fold] @ (shrink[:, None] * G) - Yf[fold]
            return float(np.einsum("ij,ij->", E, E, dtype=np.float64) / den[fold]), E

        va = [dist("va", c * n_tr)[0] for c in ALPHA_C]
        best = int(np.argmin(va))
        d_read, E_te = dist("te", ALPHA_C[best] * n_tr)
        d_read_dof, _ = dist("te", self.alpha_for_dof(DOF_STAR))
        res_img = np.einsum("ij,ij->i", E_te, E_te, dtype=np.float64)
        den_img = (Yf["te"] ** 2).sum(1, dtype=np.float64)
        return {"d_read": d_read, "d_read_dof": d_read_dof, "r2_val_best": 1 - va[best],
                "alpha_c": ALPHA_C[best], "alpha_edge": int(best in (0, len(ALPHA_C) - 1)),
                "dof": self.dof(ALPHA_C[best] * n_tr)}, res_img, den_img

    def kern(self, target_rank, Pc, pnorm):
        d_kern = float("nan") if (self.gram_rank is None or target_rank is None) \
            else 1.0 - float(self.gram_rank @ target_rank)
        c = (self.Xte_c.T @ Pc).astype(np.float64)
        cka = (c ** 2).sum() / (self.xnorm * pnorm + 1e-12)
        return d_kern, float(cka)


def probe_rows(rid, space):
    with open(f"{ROOT}/results/probes/{rid}.csv") as f:
        rows = [r for r in csv.DictReader(f) if r["space"] == space and r["probe"] in PROBES]
    return {r["probe"]: float(r["val_acc"]) for r in rows}  # last row wins on re-probes


def read_baselines(ranked):
    base = {}
    for rid, h in ranked.items():
        vals = {}
        bpath = f"{ROOT}/results/battery/{rid}.csv"
        if os.path.exists(bpath):
            with open(bpath) as f:
                for r in csv.DictReader(f):
                    if r["manifest"] == TRAIN500 and r["space"] == h and r["value"] not in ("", "nan"):
                        vals[f'{r["metric"]}|{r["variant"]}'] = float(r["value"])
        else:
            print(f"[e13] note: no battery CSV for {rid}", flush=True)
        X = np.asarray(np.load(f"{ROOT}/features/{rid}/{TRAIN500}/{h}.npy", mmap_mode="r"), dtype=np.float64)
        vals["moment_kl_sliced|e13"] = sliced_kl(X)
        vals["diag_kl|e13"] = diag_kl(X)
        base[rid] = vals
    keys = [k for k in sorted({k for v in base.values() for k in v})
            if sum(k in v for v in base.values()) >= 15]
    return base, keys


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry", action="store_true")
    args = ap.parse_args()

    ranked = dict(RANKED) if not args.dry else \
        {r: RANKED[r] for r in ("in100.dino.s0.ext", "in100.lejepa.s0.e12c1.ext")}
    cells = CELLS if not args.dry else [PRIMARY, (0, -1.0)]

    y = np.load(f"{ROOT}/features/{next(iter(ranked))}/{PAIRS}/labels.npy")
    for rid in list(ranked) + list(GATE):
        assert np.array_equal(y, np.load(f"{ROOT}/features/{rid}/{PAIRS}/labels.npy")), rid
    folds = splits(y)
    itr, iva, ite = folds
    n_tr = len(itr)
    print(f"[e13] zoo={len(ranked)} ranked + {len(GATE)} gate | splits {n_tr}/{len(iva)}/{len(ite)}", flush=True)

    dist_rows, audit_rows, gate_boot = [], [], {}
    side_cache = {}

    for tk, (t_rid, t_space) in TOKENIZERS.items():
        for dirn, (v_in, v_out) in {"A2B": ("A", "B"), "B2A": ("B", "A")}.items():
            members = {**ranked, **{r: s for r, s in GATE.items() if r != t_rid or tk == "randinit"}}
            Y0 = standardize(load_pairs(t_rid, t_space, v_out), itr)
            sub = np.random.default_rng(1).choice(itr, size=2048, replace=False)
            sigma0 = float(np.median(pdist(Y0[sub].astype(np.float64))))
            print(f"[e13] T={tk} dir={dirn} sigma0={sigma0:.3f} members={len(members)}", flush=True)

            for ci, (dm, mult) in enumerate(cells):
                seed = 1300 + ci
                if dm == 0:
                    Phi, sig = Y0, -1.0
                else:
                    rng = np.random.default_rng(seed)
                    sig = mult * sigma0
                    W = (rng.standard_normal((Y0.shape[1], dm)) / sig).astype(np.float32)
                    b = rng.uniform(0, 2 * np.pi, dm).astype(np.float32)
                    Phi = np.sqrt(2.0 / dm) * np.cos(Y0 @ W + b)
                od = offdiag(Phi[ite] @ Phi[ite].T).astype(np.float64)
                t_rank = zrank(od)
                Pc = Phi[ite] - Phi[ite].mean(0)
                pnorm = float(np.linalg.norm((Pc @ Pc.T).astype(np.float64)))  # ==||Pc.T@Pc||_F, cheaper side
                is_primary = (dm, mult) == PRIMARY
                arow = {"tokenizer": tk, "dir": dirn, "D_m": dm, "sig_mult": mult, "sigma": round(sig, 4),
                        "seed": seed if dm else -1, "sigma0": round(sigma0, 4),
                        "kern_offdiag_mean": round(float(od.mean()), 4), "kern_offdiag_sd": round(float(od.std()), 4),
                        "kern_offdiag_p05": round(float(np.quantile(od, .05)), 4),
                        "kern_offdiag_p95": round(float(np.quantile(od, .95)), 4),
                        "sketch_effrank_sub2048": round(effrank_sub(Phi, itr), 2),
                        "target_bt": round(bt_ratio(Phi, y, itr), 4),
                        "target_kmeans100_nmi": round(kmeans_nmi(Phi, y, itr), 4) if (is_primary or dm == 0) else ""}
                audit_rows.append(arow)
                if args.dry:
                    print("  audit:", arow, flush=True)

                for rid, h in members.items():
                    key = (dirn, rid)
                    if key not in side_cache:
                        side_cache[key] = Side(rid, h, v_in, itr, iva, ite)
                    S = side_cache[key]
                    m, res_img, den_img = S.fit_eval(Phi, folds, n_tr)
                    d_kern, cka = S.kern(t_rank, Pc, pnorm)
                    if is_primary and tk == "mae":
                        br = np.random.default_rng(3).integers(0, len(ite), (N_BOOT, len(ite)))
                        gate_boot.setdefault(rid, []).append(float((res_img[br].sum(1) / den_img[br].sum(1)).std()))
                    dist_rows.append({"tokenizer": tk, "dir": dirn, "D_m": dm, "sig_mult": mult, "run": rid,
                                      "ranked": int(rid in ranked), "self_ref": int(rid == t_rid),
                                      "d_read": round(m["d_read"], 5), "d_read_dof": round(m["d_read_dof"], 5),
                                      "d_kern": round(d_kern, 5), "cka": round(cka, 5),
                                      "r2_val_best": round(m["r2_val_best"], 5), "alpha_c": m["alpha_c"],
                                      "alpha_edge": m["alpha_edge"], "dof": round(m["dof"], 1)})
                    if args.dry:
                        print("   ", dist_rows[-1], flush=True)

    if args.dry:
        print("[e13] DRY RUN complete — nothing written.", flush=True)
        return

    os.makedirs(f"{ROOT}/results/diag", exist_ok=True)
    for name, rows in (("e13_distortion", dist_rows), ("e13_target_audit", audit_rows)):
        with open(f"{ROOT}/results/diag/{name}.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader(); w.writerows(rows)
        print(f"wrote results/diag/{name}.csv ({len(rows)} rows)", flush=True)

    # ---- rank-correlation stage (mechanical; E01-T15-style, this zoo) ----
    probe = {rid: probe_rows(rid, h) for rid, h in RANKED.items()}
    probe[RANDINIT] = probe_rows(RANDINIT, GATE[RANDINIT])
    for rid, v in probe.items():
        assert set(v) == set(PROBES), f"missing probe rows for {rid}"
    base, bkeys = read_baselines(ranked)
    sym = {}
    for r in dist_rows:
        sym.setdefault((r["tokenizer"], r["D_m"], r["sig_mult"], r["run"]), []).append(r)

    def symval(tk, dm, mult, rid, met):
        rows = sym.get((tk, dm, mult, rid), [])
        if len(rows) != 2:
            return None
        v = float(np.mean([r[met] for r in rows]))
        if not np.isfinite(v):
            return None
        return -v if met == "cka" else v  # uniform lower-is-better convention

    zoos = {"full20": list(ranked), "lejepa11": [r for r in ranked if r in LEJEPA_SUB],
            "nonlejepa9": [r for r in ranked if r not in LEJEPA_SUB]}
    corr_rows = []

    def add_corr(ranker, cell_tag, vals, extra_zoos=None):
        for zname, zrids in {**zoos, **(extra_zoos or {})}.items():
            rids = [r for r in zrids if r in vals and r in probe]
            if len(rids) < 5:
                continue
            for pb in PROBES:
                rho, p = spearman_perm([vals[r] for r in rids], [probe[r][pb] for r in rids])
                corr_rows.append({"ranker": ranker, "cell": cell_tag, "zoo": zname, "n": len(rids),
                                  "probe": pb, "rho": round(rho, 4), "perm_p": round(p, 5)})

    for tk in TOKENIZERS:
        for dm, mult in cells:
            tag = f"T={tk}|D_m={dm}|s={mult}"
            for met in ("d_read", "d_read_dof", "d_kern", "cka"):
                vals = {r: symval(tk, dm, mult, r, met) for r in ranked}
                vals = {r: v for r, v in vals.items() if v is not None}
                extra = None
                if tk == "mae" and (dm, mult) == PRIMARY and met in ("d_read", "d_kern"):
                    vr = symval(tk, dm, mult, RANDINIT, met)
                    if vr is not None:
                        vals_x = dict(vals, **{RANDINIT: vr})
                        extra = {"full20+randinit": list(ranked) + [RANDINIT]}
                        add_corr(met, tag, vals_x, extra)
                        continue
                add_corr(met, tag, vals)

    for bk in bkeys:
        vals = {r: base[r][bk] for r in ranked if bk in base[r]}
        add_corr(f"battery:{bk}", "train500@h", vals)
        if len(vals) < len(ranked):  # paired-subset PIVOT rho for fair comparison
            prim = {r: symval("mae", *PRIMARY, r, "d_read") for r in vals}
            add_corr(f"d_read@primary|subset:{bk}", "pairing", {r: v for r, v in prim.items() if v is not None})

    with open(f"{ROOT}/results/diag/e13_rank_corr.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(corr_rows[0]))
        w.writeheader(); w.writerows(corr_rows)
    print(f"wrote results/diag/e13_rank_corr.csv ({len(corr_rows)} rows)", flush=True)

    # ---- gate file (P1/K1) ----
    prim_read = {r: symval("mae", *PRIMARY, r, "d_read") for r in list(ranked) + [RANDINIT]}
    trained = np.array([prim_read[r] for r in ranked])
    se_med = float(np.median([np.mean(v) for v in gate_boot.values()]))
    grow = {"randinit_d_read": round(prim_read[RANDINIT], 5),
            "trained_min": round(float(trained.min()), 5), "trained_max": round(float(trained.max()), 5),
            "trained_mean": round(float(trained.mean()), 5), "trained_sd": round(float(trained.std()), 5),
            "randinit_margin_sd": round(float((prim_read[RANDINIT] - trained.mean()) / (trained.std() + 1e-12)), 2),
            "n_trained_worse_than_randinit": int((trained >= prim_read[RANDINIT]).sum()),
            "boot_se_median": round(se_med, 6),
            "dynamic_range_ratio": round(float(trained.std() / (se_med + 1e-12)), 1)}
    with open(f"{ROOT}/results/diag/e13_gate.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(grow))
        w.writeheader(); w.writerow(grow)
    print("gate:", grow, flush=True)


if __name__ == "__main__":
    main()
