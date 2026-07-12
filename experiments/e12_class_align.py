"""E12 class-alignment diagnostic (P2 instrument; D-026/D-027) — committed port of the E10-T3
scratch analysis (results/diag/e10_dlr_cluster_check.csv), same columns for continuity, plus
IN-100-scale additions (top-100 eigendirections, kmeans-100). Space = student.z.embed (LeJEPA's
h, D-003v2) on the train500 manifest. Pure function over stored arrays; numbers land raw.
"""
import argparse
import csv
import os

import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

ROOT = "/nfs/scistore19/locatgrp/bdemirel/ssl_project"
RUNS = ["in100.lejepa.s0.e12a1.ext", "in100.lejepa.s0.e12a2.ext", "in100.lejepa.s0.e12a3.ext",
        "in100.lejepa.s0.e12c1.ext", "in100.lejepa.s0.ext", "in100.lejepa.s0.null.ext",
        "in100.lejepa.s0.e12f1.ext", "in100.lejepa.s0.e12f2.ext", "in100.lejepa.s0.e12f3.ext",
        "in100.lejepa.s0.e12f4.ext", "in100.lejepa.s0.e12f5.ext", "in100.lejepa.s0.e12f6.ext"]
MANIFEST, SPACE = "in100.train500.v1", "student.z.embed"

# G-wave (D-028/D-030): per-run space map — vicreg floor trained at student trunk-GAP; dino floor
# trains student global-crop CLS while the AUDITED h is teacher.h.cls (score both, labeled).
# Controls e12gvc/e12gdc are the PRIMARY comparators; original lanes demote to reference rows.
G_RUNS = [("in100.vicreg.s0.e12gv.ext", "student.h.gap"),
          ("in100.vicreg.s0.e12gvc.ext", "student.h.gap"),
          ("in100.vicreg.s0.ext", "student.h.gap"),
          ("in100.dino.s0.e12gd.ext", "teacher.h.cls"),
          ("in100.dino.s0.e12gd.ext", "student.h.cls"),
          ("in100.dino.s0.e12gdc.ext", "teacher.h.cls"),
          ("in100.dino.s0.e12gdc.ext", "student.h.cls"),
          ("in100.dino.s0.ext", "teacher.h.cls"),
          ("in100.dino.s0.ext", "student.h.cls")]


def eta2(u, y, classes):
    tot = u.var()
    if tot < 1e-12:
        return 0.0
    mu = u.mean()
    b = sum((u[y == c].size / u.size) * (u[y == c].mean() - mu) ** 2 for c in classes)
    return float(b / tot)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gwave", action="store_true")
    args = ap.parse_args()
    items = G_RUNS if args.gwave else [(r, SPACE) for r in RUNS]
    out_name = "e12_class_align_g.csv" if args.gwave else "e12_class_align.csv"

    rows = []
    for rid, space in items:
        d = os.path.join(ROOT, "features", rid, MANIFEST)
        X = np.load(os.path.join(d, f"{space}.npy")).astype(np.float64)
        y = np.load(os.path.join(d, "labels.npy"))
        classes = np.unique(y)
        mu = X.mean(0)
        Xc = X - mu
        tot_var = (Xc ** 2).sum(1).mean()
        btw = sum((y == c).mean() * ((X[y == c].mean(0) - mu) ** 2).sum() for c in classes)
        cov = Xc.T @ Xc / (len(X) - 1)
        w, V = np.linalg.eigh(cov)
        order = np.argsort(w)[::-1]
        w, V = w[order], V[:, order]
        P = Xc @ V
        def dir_stats(k):
            e2, kurt = [], []
            for i in range(k):
                u = P[:, i]
                e2.append(eta2(u, y, classes))
                uc = (u - u.mean()) / (u.std() + 1e-12)
                kurt.append(float((uc ** 4).mean() - 3.0))
            return np.array(e2), np.array(kurt)
        e16, k16 = dir_stats(16)
        e100, _ = dir_stats(100)
        km = KMeans(n_clusters=len(classes), n_init=4, random_state=0).fit_predict(
            P[:, :128].astype(np.float32))
        rows.append({
            "run": rid, "space": space,
            "between/total_var": round(btw / tot_var, 4),
            "eig_top16_var_frac": round(float(w[:16].sum() / w.sum()), 4),
            "n_negkurt_top16": int((k16 < 0).sum()),
            "mean_eta2_top16": round(float(e16.mean()), 4),
            "corr(eta2,|kurt|)": round(float(np.corrcoef(e16, np.abs(k16))[0, 1]), 4),
            "mean_eta2_top100": round(float(e100.mean()), 4),
            "kmeans100_NMI": round(float(normalized_mutual_info_score(y, km)), 4),
            "kmeans100_ARI": round(float(adjusted_rand_score(y, km)), 4),
            "effrank_w": round(float(w.sum() ** 2 / (w ** 2).sum()), 2),
        })
        print(rows[-1], flush=True)
    out = os.path.join(ROOT, "results", "diag", out_name)
    with open(out, "w", newline="") as f:
        wcsv = csv.DictWriter(f, fieldnames=list(rows[0]))
        wcsv.writeheader()
        wcsv.writerows(rows)
    print("wrote", out)


if __name__ == "__main__":
    main()
