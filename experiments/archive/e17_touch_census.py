"""Berker's touch census (2026-07-17b, replacing the nearest-20/pool-draw design of
e17_intersect.py — his spec verbatim): for EVERY instance as anchor, over ALL candidates:

  deg_pos(i) = #{same-class j : omega(i,j) > 0}   out of 99      (omega > 0 = the r_mean
  deg_neg(i) = #{foreign  k : omega(i,k) > 0}   out of ~9900      balls overlap = free edge)

normalized to probabilities  p_pos = deg_pos/99,  p_neg = deg_neg/9900  (pool-size-free — no
candidate cap, no matched draws needed), and compared ratio-like:

  enrichment E = p_pos / p_neg    ("a positive is E x more likely to touch me than a negative")
  purity      = deg_pos / (deg_pos + deg_neg)   vs base rate 99/9999 ~ .0099
                (of the clouds an anchor is actually glued to, how many are its own class —
                 the composition kNN lives in)

Radius convention r_mean (reach); alpha=.75 columns ride along as the operating-point
robustness check (same distance matrix, shrunken balls). Per-class rows (means over the
class's anchors; enrichment as ratio-of-means — per-anchor ratios blow up at deg_neg=0) ->
results/diag/e17_touch_census.csv; run summary = median over classes. RAW; no takeaway."""
import csv
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = "/nfs/scistore19/locatgrp/bdemirel/ssl_project"
from e17_intersect import RUNS, load  # noqa: E402  (same stores, same loader)


def main():
    rows, summary = [], {}
    for tag, (rid, space) in RUNS.items():
        views, y = load(rid, space)
        xb = views.mean(1).astype(np.float32)
        rr = np.linalg.norm(views - views.mean(1, keepdims=True), axis=2).mean(1).astype(np.float32)
        sq = (xb ** 2).sum(1)
        d = np.sqrt(np.maximum(0, sq[:, None] + sq[None] - 2 * xb @ xb.T))
        same = y[:, None] == y[None]
        np.fill_diagonal(same, False)
        stats = {}
        for a in (1.0, 0.75):
            T = (a * (rr[:, None] + rr[None]) - d) > 0
            np.fill_diagonal(T, False)
            deg_pos = (T & same).sum(1)
            deg_neg = (T & ~same).sum(1)
            # ~same has True on the diagonal (self); T's diagonal is False so deg_neg is clean,
            # but the denominator must exclude self.
            stats[a] = (deg_pos / same.sum(1), deg_neg / ((~same).sum(1) - 1), deg_pos, deg_neg)
        per_cls = []
        for c in np.unique(y):
            m = y == c
            row = {"tag": tag, "class": int(c)}
            for a in (1.0, 0.75):
                p_pos, p_neg, dp, dn = stats[a]
                pp, pn = float(p_pos[m].mean()), float(p_neg[m].mean())
                suf = "" if a == 1.0 else "_a75"
                row[f"p_pos{suf}"] = round(pp, 5)
                row[f"p_neg{suf}"] = round(pn, 5)
                row[f"enrich{suf}"] = round(pp / pn, 2) if pn > 0 else float("inf")
                tot = dp[m].sum() + dn[m].sum()
                row[f"purity{suf}"] = round(float(dp[m].sum() / tot), 4) if tot else float("nan")
            per_cls.append(row)
        rows += per_cls
        summary[tag] = {k: float(np.median([r[k] for r in per_cls if np.isfinite(r[k])]))
                        for k in per_cls[0] if k not in ("tag", "class")}
        s = summary[tag]
        print(f"{tag:22s} p_pos {s['p_pos']:.3f} p_neg {s['p_neg']:.4f} enrich {s['enrich']:5.1f} "
              f"purity {s['purity']:.3f} | a=.75: p_pos {s['p_pos_a75']:.3f} p_neg "
              f"{s['p_neg_a75']:.5f} enrich {s['enrich_a75']:6.1f} purity {s['purity_a75']:.3f} "
              f"(V={views.shape[1]})", flush=True)
    with open(f"{ROOT}/results/diag/e17_touch_census.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader(), w.writerows(rows)

    cent = {(r["method"], r["tag"]): float(r["knn200"])
            for r in csv.DictReader(open(f"{ROOT}/results/diag/e17_centered.csv"))}
    fig, axes = plt.subplots(1, 2, figsize=(13.4, 5.0))
    ax = axes[0]
    for tag, s in summary.items():
        m, t = ("lejepa_e12lane", tag.split("/")[1]) if tag.startswith("e12") else tuple(tag.split("/", 1))
        k = cent.get((m, t))
        ax.plot(s["p_neg"], s["p_pos"], "o", ms=8 if k else 6,
                color="#3d65d0" if k else "#9aa7c4")
        ax.annotate(tag, (s["p_neg"], s["p_pos"]), textcoords="offset points", xytext=(5, 3),
                    fontsize=7)
    g = np.array([1e-4, 1.0])
    for e, lab in [(1, "E=1 (chance)"), (10, "E=10"), (100, "E=100")]:
        ax.plot(g, np.minimum(e * g, 1), ls=":", color="#8a8a8a", lw=1)
        ax.annotate(lab, (2.2e-4, min(e * 3e-4, 0.9)), fontsize=7.5, color="#8a8a8a")
    ax.set_xscale("log"), ax.set_yscale("log")
    ax.set_xlabel("p_neg = P(a random NEGATIVE cloud touches you)   [of ~9900]")
    ax.set_ylabel("p_pos = P(a random SAME-CLASS cloud touches you)   [of 99]")
    ax.set_title("the census: every anchor × every candidate, ω>0 counts as probabilities\n"
                 "(dotted = enrichment isolines p_pos = E·p_neg)", fontsize=10)
    ax.grid(alpha=0.15)
    ax = axes[1]
    order = sorted(summary, key=lambda t: summary[t]["purity"])
    ypos = np.arange(len(order))
    ax.barh(ypos, [summary[t]["purity"] for t in order], height=0.6, color="#3d65d0", alpha=0.85)
    ax.plot([summary[t]["purity_a75"] for t in order], ypos, "o", ms=5, color="#e07b39", zorder=4)
    ax.axvline(99 / 9999, color="#c04f4f", lw=1.2, ls="--")
    ax.annotate("base rate .0099\n(random touching)", (99 / 9999 + 0.005, len(order) - 1.6),
                fontsize=8, color="#c04f4f")
    ax.set_yticks(ypos, order, fontsize=7.5)
    ax.set_xlabel("purity of the TOUCHING set: deg_pos / (deg_pos + deg_neg)\n"
                  "bar = α=1 (r_mean) · orange dot = α=.75")
    ax.set_title("of the clouds an anchor is glued to, how many are its own class", fontsize=10)
    ax.grid(alpha=0.15, axis="x")
    fig.tight_layout()
    fig.savefig(f"{ROOT}/results/figures/e17/e17_touch_census.png", dpi=160)
    print("wrote e17_touch_census.png + e17_touch_census.csv", flush=True)


if __name__ == "__main__":
    main()
