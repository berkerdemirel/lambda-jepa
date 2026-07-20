"""Berker's augmentation-cloud-intersection hypothesis, direct form (2026-07-17b): "correct
organization depends on cloud intersection — same-class has to be better than the negs."
The reach margin measures a DIFFERENT contrast (graph-level percolation vs a null class at a
radius threshold; its run-level aggregate reads orbit tightness — 3-axis figure). This
instrument measures the claim as stated, per anchor:

  omega(i,j) — cloud-pair intersection = signed BALL-OVERLAP DEPTH on the centroid axis:
  (r_i + r_j − d(x̄_i,x̄_j)) / (r_i + r_j), r = r_mean. Positive = the balls overlap with that
  fractional depth (omega>0 ⇔ a free percolation edge at α=1); negative = gap in radius units.
  The reach edge-cost made continuous, signed, and per-pair.
  [Instrument lineage, 2026-07-17b — two informative nulls before this form:
   v1 Schilling 1-NN mixing: degenerate everywhere (omega .003-.02) — a view's NN is almost
   always an own-cloud sibling; no interpenetration at within-cloud spacing scale.
   v2 view-in-ball support depth: still ~0 everywhere EXCEPT sigreg_inv (omega_frn .055 >
   omega_same .028, AUC .35 — the collapsed-cone space is the only one with literal point
   interpenetration, and it mixes with NEGATIVES) — high-d concentration: ball-touching almost
   never implies point-containment; views live on their own shell, the lens is measure-zero.
   ⇒ in these spaces "intersection" exists only as centroid-axis ball overlap; v3 measures it.]

  Two contrasts per anchor i (both needed — they answer different questions):
  - TAIL: omega over its S=20 nearest same-class clouds vs its M=10 globally-nearest foreign
    clouds. The foreign pool is ~100x larger, so this mixes interleaving with an order-
    statistics advantage — it measures the extreme foreign tail kNN actually meets.
  - POOL-MATCHED (the fair "better than the negs", reach-null convention): foreign candidates
    = top-20 among B=3 draws of 99 uniformly-sampled foreign instances (pool size matched to
    the 99 same-class instances). auc_pool ~ 1 = same-class clouds intersect deeper than
    equally-many negs; .5 = organization carries no intersection signal.

Per run: median/IQR of per-class median AUC + the omega levels themselves. o32 stores are
auto-preferred where they exist (denser mixing estimates). RAW -> results/diag/
e17_intersect.csv + results/figures/e17/e17_intersect.png; no takeaway here."""
import csv
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = "/nfs/scistore19/locatgrp/bdemirel/ssl_project"
RUNS = {
    "lejepa/ctrl": ("in100.lejepa.s0.e17c.ext", "student.z.embed"),
    "lejepa/sigreg": ("in100.lejepa.s0.hpull_sigreg.ext", "student.z.embed"),
    "lejepa/sigreg_inv": ("in100.lejepa.s0.hpull_sigreg_inv.ext", "student.z.embed"),
    "e12/C1": ("in100.lejepa.s0.e12c1.ext", "student.z.embed"),
    "e12/f2": ("in100.lejepa.s0.e12f2.ext", "student.z.embed"),
    "e12/f8": ("in100.lejepa.s0.e12f8.ext", "student.z.embed"),
    "e12/f7": ("in100.lejepa.s0.e12f7.ext", "student.z.embed"),
    "simclr/ctrl": ("in100.simclr.s0.e17c.ext", "student.h.cls"),
    "simclr/uniform": ("in100.simclr.s0.hpull_uniform.ext", "student.h.cls"),
    "simclr/uniform_align": ("in100.simclr.s0.hpull_uniform_align.ext", "student.h.cls"),
    "byol/ctrl": ("in100.byol.s0.e17c.ext", "student.h.cls"),
    "byol/align": ("in100.byol.s0.hpull_align.ext", "student.h.cls"),
    "dino/ctrl": ("in100.dino.s0.e17c.ext", "teacher.h.cls"),
    "dino/protoce": ("in100.dino.s0.hpull_protoce.ext", "teacher.h.cls"),
    "vicreg/ctrl": ("in100.vicreg.s0.e17c.ext", "student.h.cls"),
    "vicreg/varcov_c015": ("in100.vicreg.s0.hpull_varcov.c015.ext", "student.h.cls"),
}
O32 = {"in100.lejepa.s0.e17c.ext": "in100.lejepa.s0.o32",
       "in100.lejepa.s0.e12f2.ext": "in100.lejepa.s0.e12f2.o32",
       "in100.lejepa.s0.hpull_sigreg_inv.ext": "in100.lejepa.s0.hpull_sigreg_inv.o32"}
S_SAME, M_FRN = 20, 10


def load(run_id, space):
    import os
    store, suf = (O32[run_id], "o32") if run_id in O32 and os.path.isdir(
        f"{ROOT}/features/{O32.get(run_id, '')}") else (run_id, "o8")
    d = f"{ROOT}/features/{store}/in100.pairs100.v1@audit_v1.{suf}"
    V = json.load(open(f"{d}/meta.json"))["v"]
    views = np.stack([np.load(f"{d}/{space}.view{k}.npy") for k in range(V)], 1).astype(np.float32)
    views /= np.linalg.norm(views, axis=2, keepdims=True) + 1e-12
    print(f"[store] {run_id} <- {store} ({suf}, V={V})", flush=True)
    return views, np.load(f"{d}/labels.npy")


def overlap_depth(ca, ra, cb, rb):
    """Signed ball-overlap depth of anchor (centroid ca, radius ra) vs P candidates -> [P]."""
    d = np.linalg.norm(cb - ca, axis=1)
    return (ra + rb - d) / (ra + rb)


def main():
    rows, summary = [], {}
    rng = np.random.default_rng(0)
    for tag, (rid, space) in RUNS.items():
        views, y = load(rid, space)
        xb = views.mean(1)
        rr = np.linalg.norm(views - xb[:, None], axis=2).mean(1)      # r_mean, reach convention
        sq = (xb ** 2).sum(1)
        acc = {k: [] for k in ["auc_pool", "auc_tail", "w_same", "w_pool", "w_tail",
                               "touch_same", "touch_pool", "touch_tail"]}
        for c in np.unique(y):
            idx = np.flatnonzero(y == c)
            Dg = np.sqrt(np.maximum(0, sq[idx][:, None] + sq[None] - 2 * xb[idx] @ xb.T))
            frn_order = np.argsort(np.where(y[None] == c, np.inf, Dg), 1)[:, :M_FRN]
            pools = [rng.choice(np.flatnonzero(y != c), size=len(idx) - 1, replace=False)
                     for _ in range(3)]
            Dc = Dg[:, idx]
            np.fill_diagonal(Dc, np.inf)
            same_order = np.argsort(Dc, 1)[:, :S_SAME]

            def auc(a, b):
                return float((a[:, None] > b[None]).mean() + 0.5 * (a[:, None] == b[None]).mean())

            per = {k: [] for k in acc}
            for a in range(len(idx)):
                ai = idx[a]
                om_s = overlap_depth(xb[ai], rr[ai], xb[idx[same_order[a]]], rr[idx[same_order[a]]])
                om_t = overlap_depth(xb[ai], rr[ai], xb[frn_order[a]], rr[frn_order[a]])
                om_p = []
                for P in pools:
                    jp = P[np.argsort(Dg[a, P])[:S_SAME]]
                    om_p.append(overlap_depth(xb[ai], rr[ai], xb[jp], rr[jp]))
                per["auc_pool"].append(np.mean([auc(om_s, o) for o in om_p]))
                per["auc_tail"].append(auc(om_s, om_t))
                per["w_same"].append(om_s.mean())
                per["w_pool"].append(np.concatenate(om_p).mean())
                per["w_tail"].append(om_t.mean())
                # touch fraction = of these candidates, how many are reachable GAP-FREE
                # (omega>0 ⇔ balls overlap ⇔ a free percolation edge) — the "path" readout
                per["touch_same"].append(float((om_s > 0).mean()))
                per["touch_pool"].append(float((np.concatenate(om_p) > 0).mean()))
                per["touch_tail"].append(float((om_t > 0).mean()))
            row = {"tag": tag, "class": int(c),
                   **{k: round(float(np.median(v) if k.startswith("auc") else np.mean(v)), 4)
                      for k, v in per.items()}}
            rows.append(row)
            for k in acc:
                acc[k].append(row[k])
        summary[tag] = {k: float(np.median(v)) for k, v in acc.items()} | \
            {"q1": float(np.quantile(acc["auc_pool"], .25)),
             "q3": float(np.quantile(acc["auc_pool"], .75)), "V": views.shape[1]}
        s = summary[tag]
        print(f"{tag:22s} AUC pool {s['auc_pool']:.3f} [{s['q1']:.3f},{s['q3']:.3f}] tail "
              f"{s['auc_tail']:.3f} | omega same {s['w_same']:+.3f} pool {s['w_pool']:+.3f} "
              f"tail {s['w_tail']:+.3f} | touch same {s['touch_same']:.2f} pool "
              f"{s['touch_pool']:.2f} tail {s['touch_tail']:.2f} (V={s['V']})", flush=True)
    with open(f"{ROOT}/results/diag/e17_intersect.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader(), w.writerows(rows)

    cent = {(r["method"], r["tag"]): float(r["knn200"])
            for r in csv.DictReader(open(f"{ROOT}/results/diag/e17_centered.csv"))}
    fig, axes = plt.subplots(1, 2, figsize=(13.2, 4.8))
    ax = axes[0]
    for tag, s in summary.items():
        m, t = ("lejepa_e12lane", tag.split("/")[1]) if tag.startswith("e12") else tuple(tag.split("/", 1))
        k = cent.get((m, t))
        if k is None:
            continue
        ax.plot([k, k], [s["q1"], s["q3"]], color="#bbbbbb", lw=1.1, zorder=1)
        ax.plot(k, s["auc_pool"], "o", ms=7, color="#3d65d0", zorder=3)
        ax.annotate(tag, (k, s["auc_pool"]), textcoords="offset points", xytext=(5, 4), fontsize=7)
    ax.axhline(0.5, color="#8a8a8a", lw=0.9, ls=":")
    ax.annotate("AUC .5 = same-class intersection no better than equally-many negs", (0.02, 0.03),
                xycoords="axes fraction", fontsize=8, color="#666666")
    ax.set_xlabel("knn200 @ declared h (centered)")
    ax.set_ylabel("median per-class POOL-MATCHED AUC(ω same > ω negs)")
    ax.set_title("the hypothesis test at matched pool (99 vs 99): does same-class\n"
                 "cloud intersection beat the negs where kNN is won? (whiskers = class IQR)",
                 fontsize=10)
    ax.grid(alpha=0.15)
    ax = axes[1]
    ks = list(summary)
    x = np.arange(len(ks))
    ax.bar(x - 0.28, [summary[t]["w_same"] for t in ks], width=0.27, color="#3d65d0",
           label="ω same (top-20 of 99)")
    ax.bar(x, [summary[t]["w_pool"] for t in ks], width=0.27, color="#e07b39",
           label="ω negs, pool-matched (top-20 of 99)")
    ax.bar(x + 0.28, [summary[t]["w_tail"] for t in ks], width=0.27, color="#c04f4f",
           label="ω negs, global tail (top-10 of ~9900)")
    ax.set_xticks(x, [t.replace("/", "\n") for t in ks], fontsize=6.5)
    ax.axhline(0, color="#444444", lw=0.9)
    ax.set_ylabel("mean signed overlap depth ω")
    ax.set_title("intersection LEVELS (ω>0 = balls overlap ⇔ free percolation edge at α=1;\n"
                 "ω<0 = gap in radius units)", fontsize=10)
    ax.legend(fontsize=7.5)
    ax.grid(alpha=0.15, axis="y")
    fig.tight_layout()
    fig.savefig(f"{ROOT}/results/figures/e17/e17_intersect.png", dpi=160)
    print("wrote e17_intersect.png + e17_intersect.csv", flush=True)


if __name__ == "__main__":
    main()
