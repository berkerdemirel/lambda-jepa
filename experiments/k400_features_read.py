"""Optimizer-free readers on cached K400 clip features: weighted kNN and converged L-BFGS logistic regression."""
import argparse, glob, os, sys
import numpy as np, pandas as pd, torch, torch.nn.functional as F

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from sslgap.metrics.spectra import covariance_eigs, effective_rank, rankme
from sslgap.probes.knn import knn_topk_acc
from sslgap.probes.linear import linear_lbfgs_v1

TAPS = ("pool", "gap", "cls")

def load_split(d, split):
    parts = [torch.load(f, map_location="cpu", weights_only=False) for f in sorted(glob.glob(os.path.join(d, f"{split}_rank*.pt")))]
    assert parts, f"no {split} rank files under {d}"
    y = torch.cat([p["labels"] for p in parts]).numpy()
    states = {n: {k: torch.cat([p["states"][n][k] for p in parts]).float().numpy() for k in parts[0]["states"][n]} for n in parts[0]["states"]}
    return y, states, parts[0]["num_classes"]

def fisher_ratio(X, y, nc):
    mu = X.mean(0); within = 0.0; between = 0.0
    for c in range(nc):
        xc = X[y == c]
        if len(xc) == 0:
            continue
        m = xc.mean(0); within += ((xc - m) ** 2).sum(); between += len(xc) * ((m - mu) ** 2).sum()
    return float(between / (within + 1e-12))

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--cache", required=True); ap.add_argument("--out", default=f"{ROOT}/results/e34/diag/k400_features_read.csv")
    ap.add_argument("--device", default="cuda"); ap.add_argument("--lams", default="1e-5,1e-4,1e-3"); ap.add_argument("--max-iter", type=int, default=500)
    a = ap.parse_args()
    ytr, tr, nc = load_split(a.cache, "train"); yva, va, _ = load_split(a.cache, "val")
    lams = [float(x) for x in a.lams.split(",")]
    rows = []
    add = lambda st, tap, split, metric, v: rows.append({"state": st, "tap": tap, "split": split, "metric": metric, "value": float(v)})
    for st in tr:
        for tap in TAPS:
            Xtr, Xva = tr[st][tap], va[st][tap]
            add(st, tap, "val", "knn_top1", 100 * knn_topk_acc(Xtr, ytr, Xva, yva, num_classes=nc, knn_k=20, knn_t=0.07, device=a.device))
            mu, sd = Xtr.mean(0, keepdims=True), Xtr.std(0, keepdims=True) + 1e-6
            for lam in lams:
                r = linear_lbfgs_v1((Xtr - mu) / sd, ytr, (Xva - mu) / sd, yva, nc, lam=lam, max_iter=a.max_iter, device=a.device)
                add(st, tap, "train", f"lbfgs_{lam:g}", 100 * r["train_acc"]); add(st, tap, "val", f"lbfgs_{lam:g}", 100 * r["val_acc"])
                add(st, tap, "train", f"lbfgs_{lam:g}_gradnorm", r["grad_norm"])
            eig = covariance_eigs(Xtr); d = Xtr.shape[1]
            add(st, tap, "train", "norm_mean", np.linalg.norm(Xtr, axis=1).mean()); add(st, tap, "train", "rankme_over_d", rankme(Xtr) / d)
            add(st, tap, "train", "effrank_over_d", effective_rank(eig) / d); add(st, tap, "train", "top1_eig_share", eig[0] / eig.sum())
            cos = (Xtr * tr[st]["cls"]).sum(1) / (np.linalg.norm(Xtr, axis=1) * np.linalg.norm(tr[st]["cls"], axis=1) + 1e-8)
            add(st, tap, "train", "cos_to_cls", cos.mean()); add(st, tap, "val", "fisher_ratio", fisher_ratio(Xva, yva, nc))
            print(f"[k400-read] {st} {tap}: " + "  ".join(f"{r['metric']}={r['value']:.3f}" for r in rows if r["state"] == st and r["tap"] == tap), flush=True)
        s = tr[st]; within, seg = s["within_var"].mean(), s["seg_var"].mean(); between = covariance_eigs(s["gap"]).sum()
        tot = within + seg + between
        for k, v in (("within_frac", within / tot), ("seg_frac", seg / tot), ("between_frac", between / tot),
                     ("tok_norm_mean", s["tok_norm"].mean()), ("cls_norm_over_tok_norm", s["cls_norm"].mean() / s["tok_norm"].mean())):
            add(st, "tokens", "train", k, v)
    df = pd.DataFrame(rows); os.makedirs(os.path.dirname(a.out), exist_ok=True); df.to_csv(a.out, index=False)
    pd.set_option("display.width", 250)
    for split in ("val", "train"):
        print(f"\n== {split}\n" + df[df.split == split].pivot_table(index=["tap", "metric"], columns="state", values="value").round(3).to_string())
    print(f"[k400-read] wrote {a.out}")

if __name__ == "__main__":
    main()
