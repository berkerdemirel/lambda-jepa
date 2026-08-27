"""Checklist figure deck (D-100/Berker 2026-08-12: every checklist item gets ONE figure
answering that question). Reads results/twospace + results/compare, writes
results/figures/paper/C<k>_<slug>.png; items whose inputs haven't landed are skipped
with a status line. Regenerate anytime: python experiments/paper_checklist_figs.py

Fixed palette semantics across the whole deck (validated categorical slots):
z space=blue, h space=orange, treatment(floorssl)=green, control(lejepa)=violet."""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

Z, H, TREAT, CTRL = "#2a78d6", "#eb6834", "#008300", "#4a3aa7"
MUT, INK = "#9a9891", "#0b0b0b"
TS, FIGDIR = "results/twospace", "results/figures/paper"
plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.alpha": .25, "grid.linewidth": .5,
                     "axes.titlesize": 9.5, "figure.dpi": 150})


def _load(run, part=""):
    p = f"{TS}/{run}{part}.csv"
    return pd.read_csv(p) if os.path.exists(p) else None


def _save(fig, name):
    os.makedirs(FIGDIR, exist_ok=True)
    fig.savefig(f"{FIGDIR}/{name}.png", bbox_inches="tight")
    plt.close(fig)
    print(f"  [rendered] {FIGDIR}/{name}.png")


def c1_exhibit():
    """C1: does the stated desideratum live in z and not transfer to h?"""
    df = pd.read_csv("results/compare/thm21_exhibit.csv")
    panels = [  # (method, run-substr, metric, label, log) — IN-100 CONTROL lanes only
        # (Berker 2026-08-12; treated e20f lanes are NOT the method — earlier render rode them)
        ("lejepa", "in100", "gauss_kl_full.total", "lejepa\nmoment KL to N(0,I)", True),
        ("vicreg", "in100", "variance_floor.min_over_mean_std",
         "vicreg\nvariance floor (1=met)", False),
        ("simclr", "in100", "cos_margin[own_simclr]",
         "simclr\nown-aug cos margin (D-013)", False),
    ]
    fig, axes = plt.subplots(1, len(panels), figsize=(7, 2.4))
    for ax, (meth, rung, metric, label, log) in zip(axes, panels):
        r = df[(df.method == meth) & (df.run.str.contains(rung)) & (df.metric == metric)]
        if r.empty:
            ax.axis("off"); continue
        hv, zv = float(r["h.cls"].iloc[0]), float(r["z.proj.out"].iloc[0])
        ax.plot([hv, zv], [0, 0], color=MUT, lw=1.5, zorder=1)
        ax.scatter([hv], [0], s=90, color=H, zorder=2)
        ax.scatter([zv], [0], s=90, color=Z, zorder=2)
        for v, c in ((hv, H), (zv, Z)):
            ax.annotate(f"{v:.3g}", (v, 0), textcoords="offset points",
                        xytext=(0, 10), ha="center", fontsize=8, color=INK)
        if log:
            ax.set_xscale("log")
        ax.set_yticks([]); ax.set_ylim(-.6, .9)
        ax.set_title(label, fontsize=8)
        ax.grid(axis="y", visible=False)
    fig.legend(handles=[plt.Line2D([], [], marker="o", ls="", color=H, label="h.cls (retained)"),
                        plt.Line2D([], [], marker="o", ls="", color=Z, label="z.proj.out (loss space)")],
               loc="upper center", ncol=2, frameon=False, bbox_to_anchor=(.5, 1.22))
    fig.suptitle("C1 — the trained desideratum, where the loss lives vs. where the model is used (IN-100)",
                 y=1.32, fontsize=10)
    _save(fig, "C1_desideratum_z_vs_h")


def _runs_landed(runs):
    return [(r, tag, col) for r, tag, col in runs if _load(r) is not None]


def _href(s):
    """The run's declared h = the non-layer-tap left side of its accessibility pairs
    (D-036: h is the projector input — z.embed for lejepa e20f stores, h.cls elsewhere)."""
    lefts = {sp.split("~")[0] for sp in s[s.kind == "accessibility"].space}
    lefts = {l for l in lefts if ".L0" not in l and ".L1" not in l and "z.proj" not in l}
    return sorted(lefts)[0] if lefts else None


def _zout(h_ref):
    return h_ref.split(".")[0] + ".z.proj.out"


def _zpair(s, h_ref):
    """The declared loss tap among the run's actual accessibility pairs (D-003v2):
    byol → z.pred.out; dino → z.dino.bottleneck (prototypes proxy, D-005);
    else z.proj.out. Branch-agnostic (dino pairs teacher-h with student-z)."""
    pairs = [sp for sp in s[s.kind == "accessibility"].space if sp.startswith(f"{h_ref}~")]
    for suffix in (".z.pred.out", ".z.proj.out", ".z.dino.bottleneck"):
        cand = sorted(p for p in pairs if p.endswith(suffix))
        if cand:
            return cand[0].split("~")[1]
    return None


RUNS_MAIN = [  # (run_id, short tag, lane color)
    ("in1k.floorssl.s0.e24voas.extL", "ours IN-1k (voas)", "#1baf7a"),
    ("in1k.floorssl.s0.d256vm4.extL", "ours IN-1k (vm4)", TREAT),
    ("in1k.lejepa.s0.e27lej.extL", "lejepa IN-1k", CTRL),
    ("in100.floorssl.s0.d256vm4.extL", "ours IN-100 (vm4)", "#eda100"),
    ("in100.lejepa.s0.ext", "lejepa IN-100 (control)", CTRL),
    ("in100.vicreg.s0.ext", "vicreg IN-100 (control)", MUT),
    ("in100.simclr.s0.ext", "simclr IN-100 (control)", MUT),
]


PAIRS_E12 = [  # (method, control run, treated run) — the CANONICAL ± treatment family is
    # the E20F wave (Berker 2026-08-12, results/figures/e20/e20_guillotine_zoo.png:
    # e20f floor arm colored vs control grey); e12 pairs retired from the deck.
    ("vicreg", "in100.vicreg.s0.ext", "in100.vicreg.s0.e20f.ext"),
    ("simclr", "in100.simclr.s0.ext", "in100.simclr.s0.e20f.ext"),
    ("byol", "in100.byol.s0.ext", "in100.byol.s0.e20f.ext"),
    ("dino*", "in100.dino.s0.ext", "in100.dino.s0.e20f.ext"),   # * overshoot dose
    ("lejepa", "in100.lejepa.s0.ext", "in100.lejepa.s0.e20f.ext"),
    # ours: the D-100 zonly twin (h_lamb=0, whole objective behind the MLP) is the
    # single-factor control for vm4's h term — same contrast as the e20f pairs
    ("ours", "in100.floorssl.s0.d256vm4zonly.extL", "in100.floorssl.s0.d256vm4.extL"),
]
OURS_IN100 = ("in100.floorssl.s0.d256vm4.extL", "ours")


def _h_row(run, kind, col=None):
    s = _load(run)
    if s is None: return None
    h_ref = _href(s)
    if kind == "capacity":
        r = s[(s.kind == "capacity") & (s.space == h_ref) &
              (s.manifest.str.contains("audit_v1"))]
    else:
        r = s[(s.kind == "accessibility") & (s.space == f"{h_ref}~{_zpair(s, h_ref)}") &
              (s.manifest.str.contains("audit_v1"))]
    return r.iloc[0] if len(r) else None


def c2_capacity():
    """C2: does the h moment term protect stable center capacity? (±treatment per method)"""
    groups = [(m, _h_row(c, "capacity"), _h_row(t, "capacity")) for m, c, t in PAIRS_E12]
    if all(c is None and t is None for _, c, t in groups):
        print("  [skip] C2"); return
    fig, ax = plt.subplots(figsize=(6.5, 3))
    x = 0.0; ticks, labs = [], []
    for m, c, t in groups:
        if c is not None:
            ax.bar(x, c.b_rank, width=.6, color=CTRL, label="control" if not ticks else None)
        if t is not None:
            ax.bar(x + .7, t.b_rank, width=.6, color=TREAT,
                   label="+h moment floor" if not ticks else None)
        ticks.append(x + .35); labs.append(m); x += 2.0
    ax.set_xticks(ticks, labs)
    ax.set_ylabel("stable-center rank of declared h (kept eigs of B̂)")
    ax.legend(frameon=False, loc="lower right")
    ax.set_title("C2 — stable center capacity at h, with vs without the moment term (IN-100, single-factor pairs)")
    _save(fig, "C2_capacity")


def _traj_cols():
    """Epoch-resolved (epoch, run) lanes per ±treatment pair, shared by C2b/C6.
    o8 stores carry ep25/50/75 for every arm EXCEPT lejepa-treated, whose early
    epochs were extracted as extL (D-100); ep100 = the flat final store."""
    def lane(prefix, final):
        return [(ep, f"{prefix}.ep{ep}.o8") for ep in (25, 50, 75)] + [(100, final)]
    lej_trt = ([(ep, f"in100.lejepa.s0.e20f.ep{ep}.extL") for ep in (25, 50, 75)]
               + [(100, "in100.lejepa.s0.e20f.ext")])
    return [(m, lane(c.rsplit(".ext", 1)[0], c),
             lej_trt if m == "lejepa" else lane(t.rsplit(".ext", 1)[0], t))
            for m, c, t in PAIRS_E12]


def c2b_capacity_traj():
    """C2b: the C2 center-capacity gap across training — the temporal 'protect' of
    Prop 3.2. Same routing as C6; plots b_rank (kept eigs of B̂) at declared h vs epoch."""
    cols = _traj_cols()
    fig, axes = plt.subplots(1, len(cols), figsize=(13, 2.8), sharey=True)
    drew = False
    for k, (meth, ctrl, trt) in enumerate(cols):
        ax = axes[k]
        for eps, col, ls, lab in ((ctrl, CTRL, "--", "control"),
                                  (trt, TREAT, "-", "+moment floor")):
            E, V = [], []
            for ep, run in eps:
                r = _h_row(run, "capacity")
                if r is not None:
                    E.append(ep); V.append(float(r.b_rank))
            if E:
                drew = True
                ax.plot(E, V, marker="o", ms=4, lw=1.5, color=col, ls=ls, label=lab)
        ax.set_title(meth, fontsize=9); ax.set_xlabel("epoch")
        if k == 0:
            ax.set_ylabel("stable-center rank\n(kept eigs of B̂)")
    if not drew:
        print("  [skip] C2b"); return
    axes[0].legend(frameon=False, fontsize=7, loc="lower right")
    fig.suptitle("C2b — stable center capacity across training, ± the h moment term (IN-100)")
    _save(fig, "C2b_capacity_traj")


def c3_accessibility():
    """C3: does the treatment improve accessibility? (± h moment floor, within-method —
    r_z identical within a pair, so R²_acc is directly comparable there and only there)"""
    groups = [(m, _h_row(c, "acc"), _h_row(t, "acc")) for m, c, t in PAIRS_E12]
    if all(c is None and t is None for _, c, t in groups):
        print("  [skip] C3"); return
    fig, ax = plt.subplots(figsize=(6.5, 3))
    x = 0.0; ticks, labs = [], []
    for m, c, t in groups:
        if c is None and t is None: continue
        rz = int((c if c is not None else t).r_z)
        if c is not None:
            ax.bar(x, c.r2_acc_holdout, width=.6, color=CTRL,
                   label="control" if not ticks else None)
            ax.scatter([x], [c.r2_acc_insample], marker="_", s=150, color=INK,
                       label="in-sample" if not ticks else None)
        if t is not None:
            ax.bar(x + .7, t.r2_acc_holdout, width=.6, color=TREAT,
                   label="+h moment floor" if not ticks else None)
            ax.scatter([x + .7], [t.r2_acc_insample], marker="_", s=150, color=INK)
        ticks.append(x + .35); labs.append(f"{m}\n(r_z={rz})"); x += 2.0
    ax.set_xticks(ticks, labs)
    ax.axhline(0, color=MUT, lw=.8)
    ax.set_ylabel("held-out whitened $R^2_{acc}$ (declared h → z.proj.out)")
    ax.legend(frameon=False, fontsize=8)
    ax.set_title("C3 — accessibility with vs without the moment term (IN-100)")
    _save(fig, "C3_accessibility")
    # the within-pair Δ table (Berker 2026-08-13: "a good table result") — regenerated
    # with the figure so it never drifts; r_z carried as the comparability key.
    rows = [{"method": m, "r_z": int((c if c is not None else t).r_z),
             "ctrl_r2_holdout": round(float(c.r2_acc_holdout), 4) if c is not None else None,
             "treat_r2_holdout": round(float(t.r2_acc_holdout), 4) if t is not None else None,
             "delta_holdout": round(float(t.r2_acc_holdout - c.r2_acc_holdout), 4)
                              if (c is not None and t is not None) else None,
             "ctrl_r2_insample": round(float(c.r2_acc_insample), 4) if c is not None else None,
             "treat_r2_insample": round(float(t.r2_acc_insample), 4) if t is not None else None}
            for m, c, t in groups if not (c is None and t is None)]
    os.makedirs("results/compare", exist_ok=True)
    pd.DataFrame(rows).to_csv("results/compare/c3_accessibility.csv", index=False)


NULLS = {  # per-pair dimension-null store: the RANDOM-INIT twin (same architecture,
    # seed 0, extracted 2026-08-25 with the pair lanes' audit_v1.o8 manifest). Control
    # and treated share one null (the e20f term changes the loss, not the modules).
    "vicreg": "in100.vicreg.s0.null.extL", "simclr": "in100.simclr.s0.null.extL",
    "byol": "in100.byol.s0.null.extL", "dino*": "in100.dino.s0.null.extL",
    "lejepa": "in100.lejepa.s0.null.extL", "ours": "in100.floorssl.s0.d256vm4null.extL"}


def c3b_accessibility_nulled():
    """C3b: accessibility normalized by the dimension-null (Berker 2026-08-25: bare
    R²_acc is overoptimistic when z is much lower-dimensional than h — lejepa's
    16-direction target is a low bar for a 384-d source). The null = the SAME
    measurement on the random-init twin: whatever R²_acc the architecture's widths
    and taps give before any training. Parity plot; height above the diagonal is
    the accessibility training built beyond the width prior."""
    trips = [(m, _h_row(c, "acc"), _h_row(t, "acc"), _h_row(NULLS[m], "acc"))
             for m, c, t in PAIRS_E12]
    trips = [(m, c, t, n) for m, c, t, n in trips if n is not None
             and not (c is None and t is None)]
    if not trips:
        print("  [skip] C3b (null twospace rows not landed)"); return
    fig, ax = plt.subplots(figsize=(4.6, 4.6))
    lo = min(0.0, min(float(n.r2_acc_holdout) for _, _, _, n in trips) - 0.05)
    ax.plot([lo, 1], [lo, 1], ls="--", lw=1, color=MUT, zorder=1)
    ax.annotate("y = x: no gain over the random-init twin", (0.98, 0.98),
                xycoords=("data", "data"), ha="right", va="top", fontsize=7,
                color=MUT, rotation=45, rotation_mode="anchor")
    rows = []
    for m, c, t, n in trips:
        xn = float(n.r2_acc_holdout)
        ys = [(c, CTRL), (t, TREAT)]
        ax.plot([xn, xn], [float(r.r2_acc_holdout) for r, _ in ys if r is not None] or [xn],
                color=MUT, lw=.8, zorder=2)
        for r, col in ys:
            if r is not None:
                ax.scatter([xn], [float(r.r2_acc_holdout)], s=55, color=col, zorder=3)
        ytop = max(float(r.r2_acc_holdout) for r, _ in ys if r is not None)
        ax.annotate(f"{m}\n$r_z$ {int((c if c is not None else t).r_z)} (null {int(n.r_z)})",
                    (xn, ytop), textcoords="offset points", xytext=(0, 7),
                    ha="center", fontsize=7, color=INK)
        rows.append({"method": m, "r_z_null": int(n.r_z),
                     "r_z_ctrl": int(c.r_z) if c is not None else None,
                     "r_z_treat": int(t.r_z) if t is not None else None,
                     "r2_null": round(xn, 4),
                     "ctrl_r2": round(float(c.r2_acc_holdout), 4) if c is not None else None,
                     "treat_r2": round(float(t.r2_acc_holdout), 4) if t is not None else None,
                     "ctrl_delta_over_null": round(float(c.r2_acc_holdout) - xn, 4)
                                             if c is not None else None,
                     "treat_delta_over_null": round(float(t.r2_acc_holdout) - xn, 4)
                                              if t is not None else None,
                     "ctrl_gap_closed": round((float(c.r2_acc_holdout) - xn) / (1 - xn), 4)
                                        if c is not None and xn < 1 else None,
                     "treat_gap_closed": round((float(t.r2_acc_holdout) - xn) / (1 - xn), 4)
                                         if t is not None and xn < 1 else None})
    ax.set_xlim(lo, 1.02); ax.set_ylim(lo, 1.02); ax.set_aspect("equal")
    ax.set_xlabel("held-out $R^2_{acc}$ of the random-init twin (the width prior)")
    ax.set_ylabel("held-out $R^2_{acc}$, trained")
    ax.legend(handles=[plt.Line2D([], [], marker="o", ls="", color=CTRL, label="control"),
                       plt.Line2D([], [], marker="o", ls="", color=TREAT,
                                  label="+h moment floor")],
              frameon=False, fontsize=8, loc="lower right")
    ax.set_title("C3b — accessibility over the dimension-null\n"
                 "(random-init twin, same architecture; IN-100)")
    _save(fig, "C3b_accessibility_nulled")
    pd.DataFrame(rows).to_csv("results/compare/c3b_accessibility_nulled.csv", index=False)


def _acc_stats(run, k=16):
    """Summary + spectra for the declared h→z pair: rate (R²_acc), whitened target
    rank r_z, explained-direction count R²·r_z, capacity fraction count/d_h, and the
    rank-matched best-k rate (1 − mean of the k smallest held-out residual eigs —
    every method graded on its k best-explained whitened directions)."""
    s = _load(run)
    if s is None: return None
    h_ref = _href(s)
    sp = f"{h_ref}~{_zpair(s, h_ref)}"
    r = s[(s.kind == "accessibility") & (s.space == sp) &
          (s.manifest.str.contains("audit_v1"))]
    if not len(r): return None
    r = r.iloc[0]
    P = _load(run, ".spectra")
    ev = P[(P.kind == "sigma_c") & (P.space == sp) & (P.split == "holdout") &
           (P.manifest.str.contains("audit_v1"))].sort_values("j").value.values
    r2, rz, dh = float(r.r2_acc_holdout), int(r.r_z), int(r.d_h)
    return {"r2": r2, "rz": rz, "count": r2 * rz, "cap": r2 * rz / dh,
            "bestk": float(1 - np.sort(ev)[:k].mean())}


def c3b_capacity():
    """C3b presentation (Berker 2026-08-26): the comparable pair of accessibility
    scalars. Panel A = VOLUME: explained-direction count over representation capacity
    (R²·r_z/d_h) — the treatment claim's axis (rate Δ undersells arms that also widen
    the target; the count credits built-AND-readable directions). Panel B = the
    rank-matched RATE over each method's 16 best-explained whitened directions —
    kills the narrow-target optimism (r_z=16 bare R² ≈ anyone's best-16)."""
    groups = [(m, _acc_stats(c), _acc_stats(t)) for m, c, t in PAIRS_E12]
    groups = [(m, c, t) for m, c, t in groups if c is not None and t is not None]
    if not groups:
        print("  [skip] C3b capacity"); return
    fig, (axA, axB) = plt.subplots(1, 2, figsize=(9.6, 3.4), width_ratios=[1.25, 1])
    x = 0.0; ticks, labs = [], []
    for m, c, t in groups:
        axA.bar(x, c["cap"], width=.6, color=CTRL, label="control" if not ticks else None)
        axA.bar(x + .7, t["cap"], width=.6, color=TREAT,
                label="+h moment floor" if not ticks else None)
        for xi, d in ((x, c), (x + .7, t)):
            axA.annotate(f"{d['count']:.0f}", (xi, d["cap"]), textcoords="offset points",
                         xytext=(0, 2), ha="center", fontsize=7, color=INK)
        axB.plot([x + .35, x + .35], [c["bestk"], t["bestk"]], color=MUT, lw=.8)
        axB.scatter([x + .35], [c["bestk"]], s=45, color=CTRL, zorder=3)
        axB.scatter([x + .35], [t["bestk"]], s=45, color=TREAT, zorder=3)
        ticks.append(x + .35)
        labs.append(f"{m}\n$r_z$ {c['rz']}→{t['rz']}")
        x += 2.0
    axA.annotate("width cap:\n16/16 explained", (ticks[4], groups[4][2]["cap"]),
                 textcoords="offset points", xytext=(0, 14), ha="center",
                 fontsize=6.5, color=MUT)
    axA.set_xticks(ticks, labs, fontsize=7.5)
    axB.set_xticks(ticks, [l.split("\n")[0] for l in labs], fontsize=7.5)
    axA.set_ylabel("accessible loss structure / capacity\n($R^2_{acc}\\cdot r_z\\,/\\,d_h$)")
    axA.set_title("A — how MUCH is linearly accessible\n(bar tops: explained-direction count)",
                  fontsize=9)
    axB.set_ylabel("$R^2$ over the 16 best-explained directions")
    axB.set_ylim(0.70, 1.005)
    axB.set_title("B — how WELL the best directions read\n(rank-matched: same support for every method)",
                  fontsize=9)
    axA.legend(frameon=False, fontsize=8, loc="upper left")
    fig.suptitle("C3b — accessibility made comparable: volume (A) + rank-matched rate (B), "
                 "± the h moment term (IN-100)", y=1.04, fontsize=10)
    _save(fig, "C3b_accessibility_capacity")
    rows = [{"method": m, **{f"ctrl_{k_}": round(v, 4) for k_, v in c.items()},
             **{f"treat_{k_}": round(v, 4) for k_, v in t.items()},
             "delta_count": round(t["count"] - c["count"], 1),
             "delta_cap": round(t["cap"] - c["cap"], 4)}
            for m, c, t in groups]
    pd.DataFrame(rows).to_csv("results/compare/c3b_capacity_best16.csv", index=False)


def _acc_profile(run):
    """Cumulative best-k curve: mean explained fraction over the k best-explained
    whitened directions, k = 1..r_z. Endpoint = the bare R²_acc; the k=16 point =
    the best-16 scalar; endpoint x·y = the explained-direction count."""
    s = _load(run)
    if s is None: return None
    h_ref = _href(s)
    sp = f"{h_ref}~{_zpair(s, h_ref)}"
    P = _load(run, ".spectra")
    ev = P[(P.kind == "sigma_c") & (P.space == sp) & (P.split == "holdout") &
           (P.manifest.str.contains("audit_v1"))].sort_values("j").value.values
    expl = np.sort(1 - ev)[::-1]
    return np.arange(1, len(expl) + 1), np.cumsum(expl) / np.arange(1, len(expl) + 1)


def c3b_profile():
    """C3b profile (Berker 2026-08-26 round 2: fixed slices stay asymmetric — panel A
    punishes narrow targets, best-16 is 3% of vicreg's target but 100% of lejepa's).
    The full curve subsumes every slice; methods are comparable at ANY support and
    honest at their OWN full support simultaneously."""
    fig, axes = plt.subplots(1, len(PAIRS_E12), figsize=(13, 2.6), sharey=True)
    drew = False
    for k_, (m, c, t) in enumerate(PAIRS_E12):
        ax = axes[k_]
        for run, col, ls, lab in ((c, CTRL, "--", "control"), (t, TREAT, "-", "+moment floor")):
            pr = _acc_profile(run)
            if pr is None: continue
            drew = True
            kk, y = pr
            ax.plot(kk, y, color=col, ls=ls, lw=1.5, label=lab)
            ax.scatter([kk[-1]], [y[-1]], s=22, color=col, zorder=3)
            right = kk[-1] > 150
            ax.annotate(f"{y[-1]:.2f}@{kk[-1]}", (kk[-1], y[-1]),
                        textcoords="offset points",
                        xytext=(-4 if right else 2, -9 if ls == "--" else 5),
                        ha="right" if right else "left", fontsize=6, color=col)
        ax.axvline(16, color=MUT, lw=.6, ls=":")
        ax.set_xscale("log"); ax.set_xticks([1, 4, 16, 64, 256], [1, 4, 16, 64, 256])
        ax.set_xlabel("k best directions", fontsize=7)
        ax.set_title(m, fontsize=9)
        ax.set_ylim(-0.05, 1.04)
    if not drew:
        print("  [skip] C3b profile"); return
    axes[0].set_ylabel("mean explained over\nthe k best directions")
    axes[0].legend(frameon=False, fontsize=7, loc="lower left")
    fig.suptitle("C3b — the accessibility profile (endpoint dot = bare $R^2_{acc}$ at the method's own $r_z$; "
                 "dotted line = k(16); IN-100)", y=1.08, fontsize=10)
    _save(fig, "C3b_accessibility_profile")


def c4_fidelity():
    """C4: does the view→center fidelity law Θ(I+Θ)⁻¹ hold? One facet per model
    (IN-100), points = that model's Θ eigendirections at its declared h + z.proj.out."""
    in100 = [(r, t, c) for r, t, c in RUNS_MAIN if r.startswith("in100")]
    landed = [(r, t) for r, t, _ in _runs_landed(in100) if _load(r, ".spectra") is not None]
    if not landed:
        print("  [skip] C4"); return
    ncol = min(3, len(landed))
    nrow = int(np.ceil(len(landed) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(3.4 * ncol, 3.2 * nrow),
                             squeeze=False, sharex=True, sharey=True)
    lims = (1e-3, 3)
    table = []
    for ax, (run, tag) in zip(axes.flat, landed):
        p = _load(run, ".spectra")
        s = _load(run)
        h_ref = _href(s)
        f = p[(p.kind == "fidelity") & (p.manifest.str.contains("audit_v1"))]
        n = 0
        for sp, c in ((h_ref, H), (_zpair(s, h_ref), Z)):
            e = f[f.space == sp]
            ax.scatter(e.pred_resid, e.obs_resid, s=7, alpha=.5, color=c, lw=0)
            n += len(e)
            summ = s[(s.kind == "fidelity") & (s.space == sp) &
                     (s.manifest.str.contains("audit_v1"))]
            if len(summ):
                sr = summ.iloc[0]
                table.append({"facet": tag, "space": sp, "n_dirs": int(len(e)),
                              "V_c": int(sr.V_c), "resid_logcorr": round(float(sr.resid_logcorr), 4),
                              "med_obs_over_pred": round(float(sr.resid_med_ratio), 3),
                              "w_law_relerr": round(float(sr.w_law_relerr), 3),
                              "lam_max": round(float(sr.theta_op), 3)})
        ax.plot(lims, lims, color=INK, lw=.9, ls="--")
        ax.set_xscale("log"); ax.set_yscale("log"); ax.set_xlim(lims); ax.set_ylim(lims)
        ax.set_title(f"{tag}  (n={n})", fontsize=8.5)
    for ax in axes.flat[len(landed):]:
        ax.axis("off")
    for ax in axes[-1]:
        ax.set_xlabel("predicted  λ/(1+λ) + λ/V_c", fontsize=8)
    for r in range(nrow):
        axes[r][0].set_ylabel("observed residual", fontsize=8)
    fig.legend(handles=[plt.Line2D([], [], marker="o", ls="", color=H, label="declared h"),
                        plt.Line2D([], [], marker="o", ls="", color=Z, label="z.proj.out"),
                        plt.Line2D([], [], color=INK, ls="--", label="obs = pred")],
               frameon=False, loc="upper center", ncol=3, bbox_to_anchor=(.5, 1.06))
    fig.suptitle("C4 — the view→center fidelity law, per model (IN-100)", y=1.12)
    _save(fig, "C4_fidelity_law")
    # held-out fit-quality table (Berker 2026-08-13: the evidence that the law works
    # OUT-OF-SAMPLE, not tautologically) — resid_logcorr, obs/pred tightness, and the
    # independent map check w_law_relerr = ‖W_LS − (I+Θ)⁻¹‖/‖(I+Θ)⁻¹‖.
    os.makedirs("results/compare", exist_ok=True)
    pd.DataFrame(table).to_csv("results/compare/c4_fidelity_fit.csv", index=False)


def c5_gs():
    """C5: how much retained thickness is excess (S) vs structured (G)? Bars = the
    MLP-tightened lower bound on the excess share tr(Ŝ)/tr(Θ) at declared h; markers =
    the linear closed-form bound (which carries no information beyond Θ — the MLP gap
    above it is the actual measurement). Lower bound ⇒ conservative for claiming excess."""
    def row(run):
        s = _load(run)
        if s is None: return None
        h_ref = _href(s)
        # gs_linear fallback: the MLP pass fails to tighten anyway (conditional-mean
        # linearity finding), so the linear rows carry the same effective bound
        for kind in ("gs_mlp", "gs_linear"):
            r = s[(s.kind == kind) & (s.space == h_ref) &
                  (s.manifest.str.contains("audit_v1"))]
            if len(r): return r.iloc[0]
        return None
    fig, ax = plt.subplots(figsize=(6.5, 3))
    x = 0.0; ticks, labs = [], []
    for m, c, t in PAIRS_E12:
        rc, rt = row(c), row(t)
        if rc is None and rt is None: continue
        for off, r, col, lab in ((0.0, rc, CTRL, "control"), (0.7, rt, TREAT, "+moment floor")):
            if r is None: continue
            # both are valid lower bounds on the excess — plot the better one
            ax.bar(x + off, max(r.S_lb_tr, r.S_lb_tr_linclosed) / r.theta_tr,
                   width=.6, color=col, label=lab if not ticks else None)
            ax.scatter([x + off], [r.S_lb_tr_linclosed / r.theta_tr], marker="D", s=30,
                       color=INK, zorder=3,
                       label="linear closed form" if (not ticks and off == 0) else None)
        ticks.append(x + .35); labs.append(m); x += 2.0
    if not ticks:
        print("  [skip] C5 (no gs_mlp rows)"); return
    ax.set_xticks(ticks, labs)
    ax.set_ylim(0, 1)
    ax.set_ylabel("excess share of thickness\n(lower bound  tr Ŝ / tr Θ  at declared h)")
    ax.legend(frameon=False, fontsize=8)
    ax.set_title("C5 — structured vs excess thickness (IN-100; MLP-tightened bound)")
    _save(fig, "C5_gs_split")


def c6_trajectory():
    """C6: does the moment treatment move thickness + accessibility over training?
    IN-100 only (no dataset mixing); one panel column per method, ± treatment."""
    cols_spec = _traj_cols()

    def series(eps, kind):
        E, V = [], []
        for ep, run in eps or []:
            r = _h_row(run, kind)
            if r is not None:
                E.append(ep)
                V.append(float(r.theta_tr_over_r if kind == "capacity" else r.r2_acc_holdout))
        return E, V

    fig, axes = plt.subplots(2, len(cols_spec), figsize=(13, 5), sharex=True)
    drew = False
    for k, (meth, ctrl, trt) in enumerate(cols_spec):
        for row, kind in ((0, "capacity"), (1, "acc")):
            ax = axes[row][k]
            for eps, col, ls, lab in ((ctrl, CTRL, "--", "control"),
                                      (trt, TREAT, "-", "+moment floor")):
                E, V = series(eps, kind)
                if E:
                    drew = True
                    ax.plot(E, V, marker="o", ms=4, lw=1.5, color=col, ls=ls, label=lab)
            if row == 0:
                ax.set_yscale("log"); ax.set_title(meth, fontsize=9)
            else:
                ax.set_ylim(-0.1, 1.05); ax.axhline(0, color=MUT, lw=.6)
                ax.set_xlabel("epoch")
            if k == 0:
                ax.set_ylabel("thickness trΘ/r" if row == 0 else "held-out $R^2_{acc}$")
    if not drew:
        print("  [skip] C6"); return
    axes[0][0].legend(frameon=False, fontsize=7)
    fig.suptitle("C6 — thickness (top) and accessibility (bottom) across training, ± the h moment term (IN-100)")
    _save(fig, "C6_trajectory")


def c7_depth():
    """C7: where along depth does the bridge form? IN-100 only, per-method panels,
    control vs +floor + ours (same structure as C6, x = tap depth at ep100)."""
    order = ["L03", "L06", "L09", "final"]

    def layered(run):   # first layered multi-view store that has rows: the e20-era
        # extL (treated arms; lejepa treated = its ep100.extL), the uniform .ep100.o8
        # (controls — no multi-view layered store predates it), else the flat .ext
        cands = [run.replace(".ext", ".extL"), run.replace(".ext", ".ep100.extL"),
                 run.replace(".ext", ".ep100.o8"), run]
        return next((c for c in cands if _load(c) is not None), run)

    cols_spec = [(m, layered(c), layered(t)) for m, c, t in PAIRS_E12]

    def profile(run, kind):
        s = _load(run)
        if s is None: return None
        h_ref = _href(s)
        branch = h_ref.split(".")[0]
        zt = _zpair(s, h_ref)
        vals = []
        for L in order:
            hsp = h_ref if L == "final" else f"{branch}.h.cls.{L}"
            if kind == "capacity":
                r = s[(s.kind == "capacity") & (s.space == hsp) &
                      (s.manifest.str.contains("audit_v1"))]
                vals.append(float(r.theta_tr_over_r.iloc[0]) if len(r) else np.nan)
            else:
                r = s[(s.kind == "accessibility") & (s.space == f"{hsp}~{zt}") &
                      (s.manifest.str.contains("audit_v1"))]
                vals.append(float(r.r2_acc_holdout.iloc[0]) if len(r) else np.nan)
        return vals if np.isfinite(vals).any() else None

    fig, axes = plt.subplots(2, len(cols_spec), figsize=(13, 5), sharex=True)
    x = np.arange(len(order))
    drew = False
    for k, (meth, ctrl, trt) in enumerate(cols_spec):
        for row, kind in ((0, "capacity"), (1, "acc")):
            ax = axes[row][k]
            for run, col, ls, lab in ((ctrl, CTRL, "--", "control"),
                                      (trt, TREAT, "-", "+moment floor")):
                v = profile(run, kind) if run else None
                if v is not None:
                    drew = True
                    ax.plot(x, v, marker="o", ms=4, lw=1.5, color=col, ls=ls, label=lab)
            if row == 0:
                ax.set_yscale("log"); ax.set_title(meth, fontsize=9)
            else:
                ax.set_ylim(-0.6, 1.05); ax.axhline(0, color=MUT, lw=.6)
                ax.set_xticks(x, order); ax.set_xlabel("h tap depth")
            if k == 0:
                ax.set_ylabel("thickness trΘ/r" if row == 0 else "held-out $R^2_{acc}$")
    if not drew:
        print("  [skip] C7"); return
    axes[0][0].legend(frameon=False, fontsize=7)
    fig.suptitle("C7 — thickness and the h→z bridge along the trunk (IN-100, ± the h moment term)")
    _save(fig, "C7_depth_profile")


def c8_term_shift():
    """C8 v2 (Berker 2026-08-12: rethought — the identity scatter invites a 'small Θ is
    good' misreading). New question: WHERE does the treatment's downstream change come
    from — center organization (d_h²) or the view-fidelity term (aᵀΘ(I+Θ)⁻¹a)?
    Per E12 pair, per class: Δterm = treated − control; bars = mean Δ of each term,
    marker = mean Δ probe error (≈ their sum by the identity). Negative = improvement.
    Can show the thickness term RISING while probes improve — too-thin is not the goal."""
    rows = []
    for m, cr, tr in PAIRS_E12:
        cc, tc = _load(cr, ".classes"), _load(tr, ".classes")
        if cc is None or tc is None: continue
        j = cc.merge(tc, on="cls", suffixes=("_c", "_t"))
        if not len(j): continue
        th_c = j.decomp_pred_c - j.d_h_c ** 2
        th_t = j.decomp_pred_t - j.d_h_t ** 2
        rows.append({"method": m,
                     "d_org": float((j.d_h_t ** 2 - j.d_h_c ** 2).mean()),
                     "d_thick": float((th_t - th_c).mean()),
                     "d_probe": float((j.probe_err_t - j.probe_err_c).mean())})
    if not rows:
        print("  [skip] C8 v2 (e12 class rows pending)"); return
    df = pd.DataFrame(rows)
    x = np.arange(len(df))
    fig, ax = plt.subplots(figsize=(6, 3.2))
    ax.bar(x - .17, df.d_org, width=.3, color=Z, label="Δ organization term  $d_h(y)^2$")
    ax.bar(x + .17, df.d_thick, width=.3, color=H,
           label="Δ view-fidelity term  $a^\\top\\Theta(I+\\Theta)^{-1}a$")
    ax.scatter(x, df.d_probe, marker="D", s=45, color=INK, zorder=3,
               label="Δ probe error (= sum, by the identity)")
    ax.axhline(0, color=MUT, lw=.8)
    ax.set_xticks(x, df.method)
    ax.set_ylabel("treated − control  (mean over classes;\nnegative = treatment improves)")
    ax.legend(frameon=False, fontsize=8)
    ax.set_title("C8 — where the treatment's downstream change comes from (IN-100, single-factor pairs)")
    _save(fig, "C8_term_shift")


def c8_decomposition():
    """C8 supporting panel: the decomposition identity itself (Thm 5.2iii) — validation
    that the two terms in C8 v2 are the exact split of probe error."""
    fig, ax = plt.subplots(figsize=(5, 4.6))
    drew = 0
    in100 = [(r, t, c) for r, t, c in RUNS_MAIN if r.startswith("in100")]
    for run, tag, col in _runs_landed(in100):
        c = _load(run, ".classes")
        if c is None or not len(c) or c.d_h.isna().all(): continue
        ax.scatter(c.decomp_pred, c.probe_err, s=8, alpha=.5, color=col, lw=0, label=tag)
        drew += 1
    if not drew:
        print("  [skip] C8"); return
    lims = np.array(ax.get_xlim())
    lo, hi = max(lims[0], 1e-2), lims[1]
    ax.plot([lo, hi], [lo, hi], color=INK, lw=1, ls="--", label="identity")
    ax.set_xlabel("predicted:  $d_h(y)^2 + a^\\top\\Theta(I+\\Theta)^{-1}a$")
    ax.set_ylabel("observed held-out single-view probe error")
    ax.legend(frameon=False, fontsize=7)
    ax.set_title("C8 supporting panel — the decomposition identity, one point per class (IN-100)")
    _save(fig, "C8b_decomposition_identity")


def c9_external_anchors():
    """C9: is objective (6.1) competitive at matched frame+FLOPs? S-rung scoreboard under
    the ONE shared offline protocol (bench_linear_v1 = Lightly's MAE linear recipe, 90 ep;
    D-092). External anchor = Lightly's own LeJEPA ViT-S/16 run reproduced in-house on
    THEIR code (certified 2026-08-12, HISTORY:840): linear 64.11 / kNN(k200,t.07) 47.06.
    tok/s = pretrain view tokens per sample — the FLOP position vs the anchor's 616."""
    LIGHTLY_LIN, LIGHTLY_KNN = .6411, .4706
    cells = [  # (bench csv stem, label, tok/s, color, ls)
        ("in1k.floorssl.s0.d256vm4.extL", "ours d256vm4 (V=4@224 house aug)", 788, TREAT, "-"),
        ("in1k.floorssl.s0.e27lmcb5.extL", "ours lmcb5 (anchor frame 2g+6l, bs512)", 616, "#4dab5b", "--"),
        ("in1k.floorssl.s0.e27lmc.extL", "ours lmc (anchor frame, bs128)", 616, "#8fce8f", ":"),
        ("in1k.floorssl.s0.e27oas.extL", "ours oas (house photom. 2g+8l)", 690, "#c98a00", "-"),
        ("in1k.floorssl.s0.e27lmcse.extL", "ours lmcse (anchor frame + swa)", 616, "#2e7d4f", "-."),
        ("in1k.lejepa.s0.e27lej.extL", "lejepa e27lej (our harness, 2g+8l)", 690, CTRL, "-"),
    ]
    fig, (ax, axk) = plt.subplots(1, 2, figsize=(8.6, 3.4), width_ratios=[2.4, 1])
    for stem, label, tok, col, ls in cells:
        p = f"results/probes/{stem}.bench.csv"
        if not os.path.exists(p): continue
        b = pd.read_csv(p)
        part = " [partial]" if b.epoch.max() < 90 else ""
        ax.plot(b.epoch, b.val_top1, color=col, ls=ls, lw=1.6,
                label=f"{label} — {b.val_top1.iloc[-1]*100:.2f}{part} · {tok} tok/s")
    ax.axhline(LIGHTLY_LIN, color=CTRL, lw=1.2, ls=(0, (6, 2)), alpha=.8)
    ax.annotate(f"external anchor: Lightly LeJEPA, THEIR code — {LIGHTLY_LIN*100:.2f} · 616 tok/s",
                (1, LIGHTLY_LIN), xytext=(2, 4), textcoords="offset points",
                fontsize=7, color=CTRL)
    ax.set_xlabel("linear-probe epoch (bench_linear_v1, frozen ep100 backbone)")
    ax.set_ylabel("IN-1k val top-1"); ax.set_ylim(.42, .70)
    ax.legend(frameon=False, fontsize=6.6, loc="lower right")
    knn = [("ours\nd256vm4", "in1k.floorssl.s0.d256vm4.extL", TREAT),
           ("lightly anchor\n(their code)", None, CTRL),
           ("lejepa\ne27lej", "in1k.lejepa.s0.e27lej.extL", CTRL)]
    xs, vs, cs = [], [], []
    for i, (lab, stem, col) in enumerate(knn):
        v = LIGHTLY_KNN
        if stem is not None:
            p = f"results/probes/{stem}.bench_knn.csv"
            if not os.path.exists(p): continue
            k = pd.read_csv(p)
            v = float(k[k["probe"] == "knn_k200_t0.07"].val_acc.iloc[0])
        xs.append(lab); vs.append(v); cs.append(col)
    bars = axk.bar(xs, vs, color=cs, width=.62, alpha=.85)
    axk.bar_label(bars, fmt="%.3f", fontsize=7)
    axk.set_ylabel("kNN top-1 (k200, t=.07, full bank)"); axk.set_ylim(0, .6)
    axk.tick_params(axis="x", labelsize=6.6)
    fig.suptitle("C9 — S rung: ours vs the certified external anchor, one shared offline protocol"
                 " (100-ep ViT-S/16 pretrains; B/L rungs land later)", y=1.02, fontsize=10)
    _save(fig, "C9_external_anchors")


def c11_too_thin():
    """C11: too-thin exhibit (Thm 5.2i) — do view-difficult distinctions require thickness?
    AGREED design 2026-08-15: per class, x = view-difficulty proxy (best single-view
    predictor error across ALL fitted spaces = the E_A(y) bound proxy, space-independent),
    y = achieved center separation d in spaces of different thickness (pairs100/C8
    artifacts, quintile-binned). Thin extreme = lejepa z.proj.out (16-d)."""
    stores = ["in100.floorssl.s0.d256vm4.extL", "in100.floorssl.s0.d256vm4zonly.extL",
              "in100.lejepa.s0.ext", "in100.vicreg.s0.ext", "in100.simclr.s0.ext"]
    frames = {s: _load(s, ".classes") for s in stores}
    frames = {s: c for s, c in frames.items() if c is not None and not c.d_h.isna().all()}
    if len(frames) < 3:
        print("  [skip] C11"); return
    diff = pd.concat([c[["cls", "probe_err"]] for c in frames.values()]) \
             .groupby("cls").probe_err.min()  # best single-view predictor across spaces
    q = pd.qcut(diff, 5, labels=[f"Q{i}" for i in range(1, 6)])

    def thick(stem, space):
        s = _load(stem)
        if s is None: return np.nan
        r = s[(s.kind == "capacity") & (s.space == space) & (s.manifest.str.contains("audit_v1"))]
        return float(r.theta_tr_over_r.iloc[0]) if len(r) else np.nan

    lines = [  # (stem, side 0=declared-h/1=z, label, color, ls, lw)
        ("in100.floorssl.s0.d256vm4.extL", 0, "ours h.cls (floored)", TREAT, "-", 2.2),
        ("in100.floorssl.s0.d256vm4.extL", 1, "ours z (256)", TREAT, "--", 1.4),
        ("in100.floorssl.s0.d256vm4zonly.extL", 0, "ours-zonly h.cls (no h floor)", "#7ec87e", "-", 1.4),
        ("in100.lejepa.s0.ext", 1, "lejepa z.proj.out (16-d, thin extreme)", Z, "-", 2.2),
        ("in100.lejepa.s0.ext", 0, "lejepa z.embed (512)", Z, "--", 1.4),
        ("in100.vicreg.s0.ext", 0, "vicreg h.cls", H, "-", 1.4),
    ]
    fig, (ax, axn) = plt.subplots(1, 2, figsize=(8.6, 3.5))
    for stem, side, lab, col, ls, lw in lines:
        c = frames.get(stem)
        if c is None: continue
        pair = c.space.iloc[0].split("~")
        d = c.set_index("cls")["d_h" if side == 0 else "d_z"]
        m = d.groupby(q).mean()
        # organization price beyond the irreducible: d² − (best cross-space single-view err)
        ex = (d.pow(2) - diff).groupby(q).mean()
        t = thick(stem, pair[side])
        full = f"{lab} · trΘ/r={t:.1f}" if np.isfinite(t) else lab
        ax.plot(m.index, m.values, color=col, ls=ls, lw=lw, marker="o", ms=3, label=full)
        axn.plot(ex.index, ex.values, color=col, ls=ls, lw=lw, marker="o", ms=3)
    ax.set_ylabel("center residual d (C8 org. term; low = resolved)")
    axn.set_ylabel("excess d² over the cross-space bound proxy")
    fig.supxlabel("view-difficulty quintile (best cross-space single-view error; Q5 = hardest)",
                  fontsize=8.5)
    for a in (ax, axn):
        a.set_xlabel("")
    axn.axhline(0, color=MUT, lw=.8, ls=":")
    ax.legend(frameon=False, fontsize=6.4, loc="lower right")
    fig.suptitle("C11 — too-thin (Thm 5.2i): per-class center resolution vs view difficulty,"
                 " across spaces of different thickness (IN-100 pairs100)", y=1.02, fontsize=10)
    _save(fig, "C11_too_thin")


if __name__ == "__main__":
    print("[checklist figs]")
    c1_exhibit(); c2_capacity(); c2b_capacity_traj(); c3_accessibility()
    c3b_accessibility_nulled(); c3b_capacity(); c3b_profile(); c4_fidelity(); c5_gs()
    c6_trajectory(); c7_depth(); c8_term_shift(); c8_decomposition(); c9_external_anchors()
    c11_too_thin()
    print("[done] C9 fills further as anchor columns land (lmc rerun, lmcse/lmcs5, B/L rungs)")
