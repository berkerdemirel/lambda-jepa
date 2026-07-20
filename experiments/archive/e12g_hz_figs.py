"""h vs z for the three calibration pairs (Berker 2026-07-13): the calibrated tap h (where the
moment-KL floor was applied during training; the control reads the same tap without the floor)
against the method's own loss space z, across many metrics, as figures.

Sources: results/battery/<run>.csv (in-house singles on clean train500 features) · the V=8 orbit
stores in100.pairs100.v1@<stack>.o8 (invariance; audit_v1 = headline, own stacks also landed in
the CSV) · results/probes/<run>.csv (converged probes) · diag moment-KL recomputed on train500
features with e12_floor_values.diag_read's exact convention (self-checked against the recorded
G-wave values in results/diag/e12_floor_values_g.csv).

Invariance estimator (V=8): rows L2-normalized within each space (re-metrization lesson: raw
distances are not comparable across arms); pos = mean over the 28 unordered view pairs of the
same image; rand = same view pipeline on rolled image pairings (one seeded permutation);
cos_margin = pos_cos - rand_cos; align_rel = align_pos/align_rand (lower = more view-invariant
relative to the space's own compactness — M1/M2 pair_margin vocabulary).

Outputs: results/diag/e12g_orbit_invariance.csv + results/figures/e12g/{e12g_hz_battery,
e12g_hz_invariance,e12g_hz_probes,e12g_depth_invariance}.png. Numbers land raw; takeaways in
discussion. Okabe-Ito colors, arm=filled/solid vs control=open/dashed, one axis per panel.

  sbatch slurm/e12g_figs.sbatch
"""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd

ROOT = "/nfs/scistore19/locatgrp/bdemirel/ssl_project"
FEAT = f"{ROOT}/features"
FIGD = f"{ROOT}/results/figures/e12g"
INV_CSV = f"{ROOT}/results/diag/e12g_orbit_invariance.csv"

# (method, arm run, control run, calibrated tap h, loss space z, own stack)
PAIRS = [
    ("lejepa", "in100.lejepa.s0.e12f2.ext", "in100.lejepa.s0.e12c1.ext",
     "student.z.embed", "student.z.proj.out", "audit_v1"),
    ("vicreg", "in100.vicreg.s0.e12gv.ext", "in100.vicreg.s0.e12gvc.ext",
     "student.h.cls", "student.z.proj.out", "own_vicreg"),  # D-036: projector input (was GAP)
    ("dino", "in100.dino.s0.e12gd.ext", "in100.dino.s0.e12gdc.ext",
     "teacher.h.cls", "teacher.z.dino.bottleneck", "own_dino"),
]
COL = {"lejepa": "#0072B2", "vicreg": "#D55E00", "dino": "#009E73"}
METHODS = [p[0] for p in PAIRS]

# depth ladders for the preview figure (feature kind follows the method's h type, E02)
LADDER = {
    "lejepa": ["student.h.cls.L03", "student.h.cls.L06", "student.h.cls.L09", "student.h.cls",
               "student.z.embed", "student.z.proj.tap1", "student.z.proj.tap2",
               "student.z.proj.out"],
    "vicreg": ["student.h.cls.L03", "student.h.cls.L06", "student.h.cls.L09", "student.h.cls",
               "student.z.proj.tap1", "student.z.proj.tap2", "student.z.proj.out"],
    "dino": ["teacher.h.cls.L03", "teacher.h.cls.L06", "teacher.h.cls.L09", "teacher.h.cls",
             "teacher.z.dino.tap1", "teacher.z.dino.tap2", "teacher.z.dino.bottleneck"],
}


def diag_read(X):
    mu = X.mean(0)
    var = X.var(0).clip(1e-8)
    return float(0.5 * (np.mean(mu ** 2) + np.mean(var - 1 - np.log(var))))


def orbit_invariance(run, stack):
    d = os.path.join(FEAT, run, f"in100.pairs100.v1@{stack}.o8")
    meta = json.load(open(os.path.join(d, "meta.json")))
    V, spaces = meta["v"], sorted({s.rsplit(".view", 1)[0] for s in meta["spaces"]})
    p = np.random.default_rng(0).permutation(meta["n"])
    q = np.roll(p, 1)
    rows = []
    for sp in spaces:
        vs = [np.load(os.path.join(d, f"{sp}.view{k}.npy")).astype(np.float64) for k in range(V)]
        vs = [x / (np.linalg.norm(x, axis=1, keepdims=True) + 1e-12) for x in vs]
        pc, rc, pa, ra = [], [], [], []
        for u in range(V):
            for w in range(u + 1, V):
                a, b = vs[u], vs[w]
                pc.append((a * b).sum(1).mean())
                pa.append(((a - b) ** 2).sum(1).mean())
                rc.append((a[p] * b[q]).sum(1).mean())
                ra.append(((a[p] - b[q]) ** 2).sum(1).mean())
        rows.append({"run_id": run, "stack": stack, "space": sp, "v": V, "n": meta["n"],
                     "pos_cos": np.mean(pc), "rand_cos": np.mean(rc),
                     "cos_margin": np.mean(pc) - np.mean(rc), "align_pos": np.mean(pa),
                     "align_rand": np.mean(ra), "align_rel": np.mean(pa) / np.mean(ra)})
        print(f"[hz] inv {run} {stack} {sp}: margin {rows[-1]['cos_margin']:.4f} "
              f"align_rel {rows[-1]['align_rel']:.4f}", flush=True)
    return rows


def battery(run):
    df = pd.read_csv(f"{ROOT}/results/battery/{run}.csv")
    df = df[df.variant == "raw|full"]
    return {(r.space, r.metric): r.value for r in df.itertuples()}


def probes(run):
    df = pd.read_csv(f"{ROOT}/results/probes/{run}.csv")
    return {(r.space, r.probe): r.val_acc for r in df.itertuples()}


def hz_panel(ax, get, title, log=False):
    """get(method, run, h_space, z_space) -> (vh, vz) or None per value."""
    for mi, (m, arm, ctl, h, z, _own) in enumerate(PAIRS):
        off = (mi - 1) * 0.07
        for run, ls, filled in ((arm, "-", True), (ctl, "--", False)):
            vh, vz = get(m, run, h, z)
            xs = [x + off for x, v in zip((0, 1), (vh, vz)) if v is not None]
            ys = [v for v in (vh, vz) if v is not None]
            ax.plot(xs, ys, ls=ls, marker="o", ms=6.5, lw=1.7, color=COL[m],
                    mfc=COL[m] if filled else "white", mew=1.4, alpha=0.95)
    ax.set_xticks([0, 1], ["h", "z"])
    ax.set_xlim(-0.45, 1.45)
    ax.set_title(title, fontsize=10)
    if log:
        ax.set_yscale("log")
    ax.grid(alpha=0.25, lw=0.5)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)


def legend_handles():
    hs = [Line2D([], [], color=COL[m], marker="o", ls="-", ms=6.5, label=m) for m in METHODS]
    hs += [Line2D([], [], color="#444444", marker="o", ls="-", ms=6.5, label="floor arm"),
           Line2D([], [], color="#444444", marker="o", ls="--", ms=6.5, mfc="white", mew=1.4,
                  label="matched control")]
    return hs


def caption(fig, extra=""):
    fig.text(0.005, 0.005, "h = calibrated tap (lejepa z.embed · vicreg trunk-CLS · dino teacher "
             "CLS; floor applied in the arm only) · z = loss space (proj.out / proj.out / dino "
             "bottleneck)" + extra, fontsize=7.5, color="#555555", va="bottom")


def main():
    os.makedirs(FIGD, exist_ok=True)

    # ---- orbit invariance (all taps, audit + own stacks) -> CSV ------------------------------
    inv_rows = []
    for m, arm, ctl, h, z, own in PAIRS:
        for run in (arm, ctl):
            inv_rows += orbit_invariance(run, "audit_v1")
            if own != "audit_v1":
                inv_rows += orbit_invariance(run, own)
    inv = pd.DataFrame(inv_rows)
    inv.to_csv(INV_CSV, index=False)
    print(f"[hz] wrote {INV_CSV} ({len(inv)} rows)")
    I = {(r.run_id, r.stack, r.space): r for r in inv.itertuples()}

    # ---- diag moment-KL on clean train500 (floor-values convention + self-check) -------------
    dkl, checks = {}, {("in100.vicreg.s0.e12gv.ext", "student.h.cls"): 0.8131,  # D-036: CLS (GAP was 0.8157)
                       ("in100.vicreg.s0.e12gvc.ext", "student.h.cls"): 1.1962,  # D-036: CLS (GAP was 1.9486)
                       ("in100.dino.s0.e12gd.ext", "teacher.h.cls"): 0.3436,
                       ("in100.dino.s0.e12gdc.ext", "teacher.h.cls"): 0.6470}
    for m, arm, ctl, h, z, _own in PAIRS:
        for run in (arm, ctl):
            for sp in (h, z):
                X = np.load(f"{FEAT}/{run}/in100.train500.v1/{sp}.npy").astype(np.float64)
                dkl[(run, sp)] = diag_read(X)
                if (run, sp) in checks:
                    print(f"[hz] diag_kl self-check {run} {sp}: {dkl[(run, sp)]:.4f} "
                          f"vs recorded {checks[(run, sp)]:.4f}", flush=True)
    B = {run: battery(run) for _m, a, c, *_ in PAIRS for run in (a, c)}
    P = {run: probes(run) for _m, a, c, *_ in PAIRS for run in (a, c)}

    # ---- fig 1: in-house battery grid ---------------------------------------------------------
    grid = [("effective_rank", "effective rank", True), ("rankme", "RankMe", True),
            ("participation_ratio", "participation ratio", True),
            ("alpha", "α-ReQ spectral decay", False),
            ("epps_pulley", "Epps–Pulley (sliced Gaussianity)", True),
            ("kurt_topeig.worst", "worst top-eig |kurtosis|", True),
            ("offdiag_redundancy.mean_abs_corr", "mean |off-diag corr|", False),
            ("uniformity", "uniformity (Wang–Isola)", False),
            ("variance_floor.hinge", "variance-floor hinge", False),
            ("diag_kl", "diag moment-KL to N(0,I)", True)]
    fig, axes = plt.subplots(2, 5, figsize=(16.5, 7.2))
    for ax, (key, title, log) in zip(axes.flat, grid):
        if key == "diag_kl":
            get = lambda m, run, h, z: (dkl.get((run, h)), dkl.get((run, z)))
        else:
            get = lambda m, run, h, z, k=key: (B[run].get((h, k)), B[run].get((z, k)))
        hz_panel(ax, get, title, log)
    fig.legend(handles=legend_handles(), loc="upper right", ncol=5, fontsize=9, frameon=False)
    fig.suptitle("E12 G-wave calibration pairs — in-house battery, h vs z (train500 clean, "
                 "raw|full)", fontsize=12, x=0.02, ha="left")
    caption(fig)
    fig.tight_layout(rect=(0, 0.025, 1, 0.94))
    fig.savefig(f"{FIGD}/e12g_hz_battery.png", dpi=180)
    plt.close(fig)

    # ---- fig 2: invariance h vs z (audit stack, V=8) ------------------------------------------
    keys = [("pos_cos", "same-image view cosine (pos_cos)", False),
            ("rand_cos", "different-image cosine (rand_cos)", False),
            ("cos_margin", "cos margin (pos − rand)", False),
            ("align_rel", "relative alignment (lower = more invariant)", False)]
    fig, axes = plt.subplots(1, 4, figsize=(14.5, 3.9))
    for ax, (key, title, log) in zip(axes.flat, keys):
        get = lambda m, run, h, z, k=key: (getattr(I[(run, "audit_v1", h)], k),
                                           getattr(I[(run, "audit_v1", z)], k))
        hz_panel(ax, get, title, log)
    fig.legend(handles=legend_handles(), loc="upper right", ncol=5, fontsize=9, frameon=False)
    fig.suptitle("View-invariance, h vs z — V=8 orbit stores, audit_v1 stack (own stacks in "
                 "results/diag/e12g_orbit_invariance.csv)", fontsize=11, x=0.02, ha="left")
    caption(fig)
    fig.tight_layout(rect=(0, 0.04, 1, 0.87))
    fig.savefig(f"{FIGD}/e12g_hz_invariance.png", dpi=180)
    plt.close(fig)

    # ---- fig 3: converged probes h vs z --------------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(8.2, 3.9))
    for ax, (key, title) in zip(axes.flat, [("linear_raw_v2", "linear probe (converged raw_v2)"),
                                            ("knn_v1_k200", "kNN k=200")]):
        get = lambda m, run, h, z, k=key: (P[run].get((h, k)), P[run].get((z, k)))
        hz_panel(ax, get, title)
        ax.set_ylabel("val acc")
    fig.legend(handles=legend_handles(), loc="upper right", ncol=5, fontsize=8.5, frameon=False)
    fig.suptitle("Converged probes, h vs z", fontsize=11, x=0.02, ha="left")
    caption(fig)
    fig.tight_layout(rect=(0, 0.04, 1, 0.86))
    fig.savefig(f"{FIGD}/e12g_hz_probes.png", dpi=180)
    plt.close(fig)

    # ---- fig 4: invariance across the full depth ladder (preview for the H2 discussion) -------
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.1), sharey=True)
    for ax, (m, arm, ctl, h, z, _own) in zip(axes.flat, PAIRS):
        taps = LADDER[m]
        short = [t.split(".", 1)[1].replace("h.cls", "h").replace("h.gap", "h")
                  .replace("z.proj.", "").replace("z.dino.", "").replace("z.embed", "embed")
                 for t in taps]
        for run, ls, filled in ((arm, "-", True), (ctl, "--", False)):
            y = [getattr(I[(run, "audit_v1", t)], "cos_margin") for t in taps]
            ax.plot(range(len(taps)), y, ls=ls, marker="o", ms=6, lw=1.7, color=COL[m],
                    mfc=COL[m] if filled else "white", mew=1.3)
        ax.axvline(taps.index(h), color="#999999", lw=0.8, ls=":")
        ax.set_xticks(range(len(taps)), short, rotation=45, ha="right", fontsize=8)
        ax.set_title(m, fontsize=10, color=COL[m])
        ax.grid(alpha=0.25, lw=0.5)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
    axes[0].set_ylabel("cos margin (pos − rand)")
    fig.legend(handles=legend_handles()[3:], loc="upper right", ncol=2, fontsize=9, frameon=False)
    fig.suptitle("View-invariance margin across the depth ladder (audit_v1, V=8) — dotted line = "
                 "calibrated tap; preview for the overlap/E02 discussion", fontsize=11,
                 x=0.02, ha="left")
    fig.tight_layout(rect=(0, 0.01, 1, 0.9))
    fig.savefig(f"{FIGD}/e12g_depth_invariance.png", dpi=180)
    plt.close(fig)
    print(f"[hz] wrote 4 figures under {FIGD}/")


if __name__ == "__main__":
    main()
