"""ViT-S/16 IN-1k metric zoo — the guillotine-zoo layout over OUR S cell (Berker
2026-08-24): one row per model, no gray context lines; stations = trunk depth
[L03, L06, L09, cls] + the post-projector z as the final station (the two-space axis);
lean rows (anchor: cls + z only) plot whatever stations their store carries.

Columns: lightly-protocol linear (bench @90, cls only) · kNN200 t=.07 (cls only) ·
rankme/d · effrank/d · gauss_kl_full · epps_pulley · kurt_topeig.worst · pos_cos ·
rand_cos · class margin (same−diff) · Ω=W/B (o8 orbit energy, E23 calculus).
EP is never shown without kurtosis (house rule); battery moments from in1k.val.v1L
raw|full; pairs metrics from the store's @audit_v1 pairs manifest (vm4 = pairs50/o10 —
smaller pair sets, declared); anchor lin/knn = their-code certified numbers (dagger).

Store-computed cells cache to results/compare/svit_zoo_computed.csv (compute-if-missing);
rerun after a new model's audit lands to add its row.
"""
import glob
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sslgap.metrics.orbit_energy import orbit_energies
from sslgap.metrics.pairs import class_margin, pair_margin

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FEAT = os.path.join(ROOT, "features")
CACHE = os.path.join(ROOT, "results/compare/svit_zoo_computed_v2.csv")
OUT = os.path.join(ROOT, "results/figures/e27/svit_metric_zoo.png")

# (label, run_id, certified (lin, knn) for rows whose bench is not ours)
MODELS = [
    ("v10u",            "in1k.floorssl.s0.e27v10u.extL",  None),
    ("vm4",             "in1k.floorssl.s0.d256vm4.extL",  None),
    ("lmcse",           "in1k.floorssl.s0.e27lmcse.extL", None),
    ("oas",             "in1k.floorssl.s0.e27oas.extL",   None),
    ("lmc",             "in1k.floorssl.s0.e27lmc.extL",   None),
    ("lightly anchor†", "in1k.lejepa.s0.lightly.extL",    (0.6411, 0.4706)),
    ("voas",            "in1k.floorssl.s0.e24voas.extL",  None),
    ("lmcs5",           "in1k.floorssl.s0.e27lmcs5.extL", None),
    ("lmcb5",           "in1k.floorssl.s0.e27lmcb5.extL", None),
    ("lejepa (ours)",   "in1k.lejepa.s0.e27lej.extL",     None),
]
STATIONS = ["L03", "L06", "L09", "cls", "p1", "p2", "z"]
SPACE = {"L03": "student.h.cls.L03", "L06": "student.h.cls.L06",
         "L09": "student.h.cls.L09", "cls": "student.h.cls",
         "p1": "student.z.proj.tap1", "p2": "student.z.proj.tap2",
         "z": "student.z.proj.out"}
SHARED_Y = {"lin (lightly)", "knn200 t.07", "rankme/d", "effrank/d",
            "pos_cos", "rand_cos", "class_margin"}     # bounded metrics share y across rows
                                                       # (Berker 08-24); the rest free per row
BATT = [("rankme/d", "rankme", True), ("effrank/d", "effective_rank", True),
        ("gauss_kl_full", "gauss_kl_full.total", False), ("epps_pulley", "epps_pulley", False),
        ("kurt_topeig.worst", "kurt_topeig.worst", False)]
COMPUTED = ["pos_cos", "rand_cos", "class_margin", "omega"]
COLS = ["lin (lightly)", "knn200 t.07"] + [b[0] for b in BATT] + COMPUTED


def _find(run_dir, suffix):
    hits = sorted(g for g in glob.glob(os.path.join(run_dir, "*@audit_v1" + suffix))
                  if (suffix == ".o8") == g.endswith(".o8"))
    return hits[0] if hits else None


def compute_store_metrics(run_id):
    """All store-computed cells for one model -> list of dict rows."""
    run_dir = os.path.join(FEAT, run_id)
    rows = []
    o8_dir = _find(run_dir, ".o8")
    # v1L = tap-ful store dir; lean extracts (D-102, h_layers=[]) write plain v1 —
    # same manifest + eval transform, different dir name.
    val_dir = next((os.path.join(run_dir, v) for v in ("in1k.val.v1L", "in1k.val.v1")
                    if os.path.isdir(os.path.join(run_dir, v))), None)
    y = np.load(os.path.join(val_dir, "labels.npy")) if val_dir else None
    for st in STATIONS:
        sp = SPACE[st]
        # pos/rand from the o8 orbit store (view0 vs view1, audit_v1) — the only pair
        # source that covers the tap stations (the viewA/B pairs manifests carry
        # cls/gap/z.out only; Berker 08-24 fix (ii)).
        if o8_dir and os.path.exists(os.path.join(o8_dir, sp + ".view0.npy")):
            A = np.load(os.path.join(o8_dir, sp + ".view0.npy")).astype(np.float32)
            B = np.load(os.path.join(o8_dir, sp + ".view1.npy")).astype(np.float32)
            pm = pair_margin(A, B)
            rows += [dict(run_id=run_id, station=st, metric="pos_cos", value=pm["pos_cos"],
                          manifest=os.path.basename(o8_dir)),
                     dict(run_id=run_id, station=st, metric="rand_cos", value=pm["rand_cos"],
                          manifest=os.path.basename(o8_dir))]
        if y is not None and os.path.exists(os.path.join(val_dir, sp + ".npy")):
            X = np.load(os.path.join(val_dir, sp + ".npy"))
            rows.append(dict(run_id=run_id, station=st, metric="class_margin",
                             value=class_margin(X, y), manifest="in1k.val.v1L"))
        if o8_dir and os.path.exists(os.path.join(o8_dir, sp + ".view0.npy")):
            views = [np.load(os.path.join(o8_dir, f"{sp}.view{v}.npy")).astype(np.float32)
                     for v in range(8)]
            rows.append(dict(run_id=run_id, station=st, metric="omega",
                             value=orbit_energies(views)["omega"],
                             manifest=os.path.basename(o8_dir)))
        print(f"[zoo] {run_id} {st}: computed", flush=True)
    return rows


def load_computed():
    cache = pd.read_csv(CACHE) if os.path.exists(CACHE) else pd.DataFrame(
        columns=["run_id", "station", "metric", "value", "manifest"])
    fresh = []
    for _, rid, _ in MODELS:
        if not os.path.isdir(os.path.join(FEAT, rid)):
            print(f"[zoo] {rid}: no store yet — row stays sparse", flush=True)
            continue
        if not (cache["run_id"] == rid).any():
            fresh += compute_store_metrics(rid)
    if fresh:
        cache = pd.concat([cache, pd.DataFrame(fresh)], ignore_index=True)
        os.makedirs(os.path.dirname(CACHE), exist_ok=True)
        cache.to_csv(CACHE, index=False)
    return cache


def batt_value(bdf, st, metric, per_d):
    r = bdf[(bdf.space == SPACE[st]) & (bdf.metric == metric) & (bdf.variant == "raw|full")]
    if r.empty:
        return np.nan
    v = float(r.value.iloc[0])
    return v / float(r.d.iloc[0]) if per_d else v


def main():
    computed = load_computed()
    cell = {}                                            # (label, col) -> {station: value}
    for label, rid, cert in MODELS:
        bp = os.path.join(ROOT, "results/battery", rid + ".csv")
        bdf = pd.read_csv(bp) if os.path.exists(bp) else None
        if bdf is not None:
            # e27-era audits ran the moment battery on full train (D-092 standard);
            # vm4's older audit on val; lean stores (D-102) use un-suffixed v1 names —
            # prefer train over val, tap-ful over lean, declare in title.
            man = next((m for m in ("in1k.train.v1L", "in1k.train.v1",
                                    "in1k.val.v1L", "in1k.val.v1")
                        if (bdf.manifest == m).any()), None)
            bdf = bdf[bdf.manifest == man] if man else bdf.iloc[0:0]
        for cname, bmetric, per_d in BATT:
            cell[(label, cname)] = {st: batt_value(bdf, st, bmetric, per_d)
                                    for st in STATIONS} if bdf is not None else {}
        cdf = computed[computed.run_id == rid]
        for m in COMPUTED:
            cell[(label, m)] = {r.station: r.value for r in
                                cdf[cdf.metric == m].itertuples()}
        if cert:
            lin, knn = cert
        else:
            lin = knn = np.nan
            fb = os.path.join(ROOT, "results/probes", rid + ".bench.csv")
            if os.path.exists(fb):
                b = pd.read_csv(fb)
                lin = float(b[b.epoch == 90].val_top1.iloc[0])
            fk = os.path.join(ROOT, "results/probes", rid + ".bench_knn.csv")
            if os.path.exists(fk):
                k = pd.read_csv(fk)
                knn = float(k[k.probe == "knn_k200_t0.07"].val_acc.iloc[0])
        cell[(label, "lin (lightly)")] = {"cls": lin}
        cell[(label, "knn200 t.07")] = {"cls": knn}

    ylim = {}
    for c in COLS:
        vals = [v for (lb, cc), d in cell.items() if cc == c
                for v in d.values() if np.isfinite(v)]
        if vals:
            lo, hi = min(vals), max(vals)
            pad = 0.06 * (hi - lo or abs(hi) or 1.0)
            ylim[c] = (lo - pad, hi + pad)

    nr, nc = len(MODELS), len(COLS)
    fig, axes = plt.subplots(nr, nc, figsize=(3.0 * nc, 1.75 * nr), squeeze=False)
    colors = plt.cm.tab10(np.linspace(0, 1, 10))
    xs = np.arange(len(STATIONS))
    for i, (label, rid, cert) in enumerate(MODELS):
        for j, c in enumerate(COLS):
            ax = axes[i][j]
            d = cell.get((label, c), {})
            pts = [(xs[k], d[st]) for k, st in enumerate(STATIONS)
                   if st in d and np.isfinite(d[st])]
            if pts:
                px, py = zip(*pts)
                ax.plot(px, py, "-o", color=colors[i], ms=4, lw=1.8)
            ax.set_xlim(-0.5, len(STATIONS) - 0.5)
            if c in SHARED_Y and c in ylim:
                ax.set_ylim(*ylim[c])
            ax.set_xticks(xs)
            ax.set_xticklabels(STATIONS if i == nr - 1 else [], fontsize=7)
            ax.tick_params(axis="y", labelsize=7)
            ax.grid(alpha=0.25, lw=0.5)
            if i == 0:
                ax.set_title(c, fontsize=11)
            if j == 0:
                ax.set_ylabel(label, fontsize=10)
    fig.suptitle(
        "ViT-S/16 IN-1k zoo — one row per model; stations L03/L06/L09/cls (trunk, final-norm on taps) "
        "+ p1/p2/z (projector layers, d = 2048/2048/z-dim); battery raw|full on the audited manifest "
        "(e27: full train; vm4: val)\npos/rand-cos + Ω from @audit_v1 o8 orbits, margin on val (vm4: "
        "o8-on-pairs10, smaller set); lin/knn = lightly bench at cls († = their-code certified); "
        "bounded metrics share y across rows, gauss/EP/kurt/Ω free per row; RAW", fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.965])
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    fig.savefig(OUT, dpi=140)
    print(f"[zoo] wrote {OUT}", flush=True)


if __name__ == "__main__":
    main()
