"""Depth-metric curves for the e20f cadence guillotine (Berker 2026-07-21: rankme, effrank,
gaussianity, pos/rand invariance, class-cos — all vs depth station, per checkpoint). Pure
functions REUSED from sslgap.metrics (D-054): rankme, effective_rank, gauss_kl_full,
pair_margin. Clean stats from the train500.v1L stores; pos/rand from views 0/1 of the
audit_v1 o8 orbit stores (the 2-view posneg convention; the pairs path cannot carry L-taps —
the orbit path can, its E02 purpose). Class-cos = exact label-conditioned mean cosines via
the class-sum identity. Rows -> results/diag/e20f_depth_metrics.csv. RAW."""
import csv
import sys

import numpy as np

sys.path.insert(0, "/nfs/scistore19/locatgrp/bdemirel/ssl_project")
from sslgap.extract.store import FeatureStore
from sslgap.metrics.isotropy import gauss_kl_full
from sslgap.metrics.pairs import pair_margin
from sslgap.metrics.spectra import effective_rank, rankme

ROOT = "/nfs/scistore19/locatgrp/bdemirel/ssl_project"
S = FeatureStore(f"{ROOT}/features")
MAN_L, MAN_O = "in100.train500.v1L", "in100.pairs100.v1@audit_v1.o8"
CKPTS = [(ep, f"in100.lejepa.s0.e20f.ep{ep}.extL") for ep in (25, 50, 75, 100)]


def class_cos(X, y):
    Xn = X / np.linalg.norm(X, axis=1, keepdims=True)
    same_num = same_den = 0.0
    s_all = Xn.sum(0)
    s_sq = 0.0
    for c in np.unique(y):
        s = Xn[y == c].sum(0)
        n = (y == c).sum()
        same_num += s @ s - n
        same_den += n * (n - 1)
        s_sq += s @ s
    N = len(Xn)
    diff = (s_all @ s_all - s_sq) / (N * N - sum(((y == c).sum()) ** 2 for c in np.unique(y)))
    return same_num / same_den, diff


rows = []
for ep, run in CKPTS:
    y = np.asarray(S.labels(run, MAN_L))
    for sp in sorted(S.spaces(run, MAN_L)):
        X = np.asarray(S.get(run, MAN_L, sp), dtype=np.float32)
        lam = np.clip(np.linalg.eigvalsh(np.cov(X.astype(np.float64), rowvar=False)), 0, None)
        cs, cd = class_cos(X, y)
        g = gauss_kl_full(X)
        r = {"ep": ep, "space": sp, "rankme": round(rankme(X), 2),
             "effective_rank": round(effective_rank(lam), 2),
             "gauss_kl_total": round(g["total"], 4),
             "class_cos_same": round(cs, 4), "class_cos_diff": round(cd, 4)}
        try:
            A = np.asarray(S.get(run, MAN_O, f"{sp}.view0"), dtype=np.float32)
            B = np.asarray(S.get(run, MAN_O, f"{sp}.view1"), dtype=np.float32)
            pm = pair_margin(A, B)
            r["pos_cos"], r["rand_cos"] = round(pm["pos_cos"], 4), round(pm["rand_cos"], 4)
        except Exception:
            r["pos_cos"] = r["rand_cos"] = None
        rows.append(r)
        print(r, flush=True)
with open(f"{ROOT}/results/diag/e20f_depth_metrics.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0]))
    w.writeheader()
    w.writerows(rows)
print("wrote e20f_depth_metrics.csv", flush=True)
