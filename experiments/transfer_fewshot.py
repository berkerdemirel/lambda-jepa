"""E25 few-shot track: LeJEPA-style k-shot linear probes (Berker 2026-08-05: "few shots
make more sense because lejepa has the matched epoch option at least").

Reference = LeJEPA Table 2 (arXiv:2511.08544): 1-shot / 10-shot / full linear on the
SAME 8 datasets, models pretrained 100 ep on IN-1k — epoch-matched to d256vm4. Their
k-shot mechanics are unpublished (paper appendix + repo checked 2026-08-05), so this is
a DECLARED RECONSTRUCTION (E25 card): support = k stratified images/class drawn from
the full labeled pool (train + official val), 20 independent draws (seeds 0-19),
logistic regression (lbfgs, C=1.0, max_iter=1000), scored on the official test split.
Both plain top-1 (primary, the likely Table-2 metric) and the E25 dataset metric
(per-class mean where applicable) are recorded. Features: declared h CLS + gap rider,
same preprocessing as the E25 linear track. The landed E25 C-swept linear = the "full"
column analog (protocol difference recorded).

  python experiments/transfer_fewshot.py [dataset ...]     # default: all 8
"""
import os
import sys

import numpy as np
from sklearn.linear_model import LogisticRegression

from transfer_probe import BENCH, CKPT, ROOT, build_split, extract
from sslgap.ckpt import adapters

OUT = os.path.join(ROOT, "results/transfer/in1k.floorssl.s0.d256vm4.fewshot.csv")
KSHOTS = (1, 10)
DRAWS = 20


def scores(clf, X, y):
    p = clf.predict(X)
    top1 = float((p == y).mean())
    perclass = float(np.mean([(p[y == c] == c).mean() for c in np.unique(y)]))
    return top1, perclass


def main():
    names = sys.argv[1:] or list(BENCH)
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    loaded = adapters.load("native", CKPT, run_id=os.path.basename(CKPT))
    loaded.eval_("cuda")
    with open(OUT, "a") as fh:
        for name in names:
            splits, metric = BENCH[name]
            X, Y = {}, {}
            for split, spec in splits.items():
                if isinstance(spec, str):
                    continue                      # carve markers: pool keeps official splits only
                X[split], Y[split] = extract(loaded, build_split(name, spec))
                print(f"[extract] {name}/{split}: n={len(Y[split])}", flush=True)
            pool_splits = [s for s in ("train", "val") if s in X]
            for space in ("cls", "gap"):
                Xp = np.concatenate([X[s][space] for s in pool_splits])
                yp = np.concatenate([Y[s] for s in pool_splits])
                Xte, yte = X["test"][space], Y["test"]
                cls_idx = {c: np.flatnonzero(yp == c) for c in np.unique(yp)}
                for k in KSHOTS:
                    t1s, pcs = [], []
                    for seed in range(DRAWS):
                        rng = np.random.default_rng(seed)
                        idx = np.concatenate([rng.choice(ix, size=k, replace=False)
                                              for ix in cls_idx.values()])
                        clf = LogisticRegression(solver="lbfgs", C=1.0, max_iter=1000)
                        clf.fit(Xp[idx], yp[idx])
                        t1, pc = scores(clf, Xte, yte)
                        t1s.append(t1); pcs.append(pc)
                    row = (f"{name},{space},{k},{DRAWS},"
                           f"{np.mean(t1s):.4f},{np.std(t1s):.4f},"
                           f"{metric},{np.mean(pcs):.4f},{np.std(pcs):.4f}")
                    fh.write(row + "\n"); fh.flush()
                    print(f"[fewshot] {row}", flush=True)


if __name__ == "__main__":
    main()
