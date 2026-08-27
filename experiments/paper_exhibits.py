"""Paper exhibits (Berker 2026-08-24): presentation-grade rebuilds of existing data.

1. fig_discrepancy — the C1 content (stated desideratum at z vs h, IN-100 controls,
   results/compare/thm21_exhibit.csv) as per-panel VALUE bars (the old dumbbell put both
   dots on a dummy y-row). No suptitle — the LaTeX caption carries the claim.
2. fig_treatment_main (lejepa/vicreg/simclr/dino × linear/kNN/RankMe/Θ/cos-margin) and
   fig_treatment_appendix (all 8 rows × every metric) — the E20 zoo data (probes CSVs +
   results/diag/*_depth_metrics.csv + e23_retro_spaces.csv W/B), restyled: tight per-row
   y on linear/kNN, pos−rand merged into one cos-margin column, Ω renamed Θ (paper
   vocabulary), control grey vs one accent per method.
3. placement table (results/compare/placement_table.csv + docs/paper/placement_table.tex)
   — cls-level metric dump: Lightly-anchor row + our winner (v10u), grows as the public
   fleet lands.
"""
import csv
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIG = os.path.join(ROOT, "docs/paper/figures")   # the paper's own figure dir (Berker 08-24)
INK, MUT, GRID = "#1a1a1a", "#666666", "#e3e3e3"
CTRL = "#9a9a9a"
Z_COL, H_COL = "#3d65d0", "#d4820a"
plt.rcParams.update({"font.size": 10, "axes.edgecolor": MUT, "axes.labelcolor": INK,
                     "xtick.color": INK, "ytick.color": INK, "axes.titlesize": 10,
                     "figure.dpi": 150, "savefig.bbox": "tight"})


def _spines(ax):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)


# ---------------------------------------------------------------- fig 1: discrepancy
def fig_discrepancy():
    df = pd.read_csv(f"{ROOT}/results/compare/thm21_exhibit.csv")
    panels = [  # (method, metric, panel title, log-y, target line or None)
        ("lejepa", "gauss_kl_full.total", "LeJEPA\nmoment KL to $\\mathcal{N}(0,I)$  ($\\downarrow$ = met)", True, None),
        ("vicreg", "variance_floor.min_over_mean_std", "VICReg\nvariance floor  (1 = met)", False, 1.0),
        ("simclr", "cos_margin[own_simclr]", "SimCLR\nalignment margin  ($\\uparrow$ = met)", False, None),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(8.2, 2.9))
    for ax, (meth, metric, title, log, target) in zip(axes, panels):
        r = df[(df.method == meth) & (df.run.str.contains("in100")) & (df.metric == metric)]
        if r.empty:
            ax.axis("off")
            continue
        zv, hv = float(r["z.proj.out"].iloc[0]), float(r["h.cls"].iloc[0])
        bars = ax.bar([0, 1], [zv, hv], width=0.55, color=[Z_COL, H_COL], zorder=2)
        for x, v in ((0, zv), (1, hv)):
            ax.annotate(f"{v:.3g}", (x, v), xytext=(0, 4), textcoords="offset points",
                        ha="center", fontsize=9.5, color=INK)
        if log:
            ax.set_yscale("log")
            ax.set_ylim(1e-3, 30)
        else:
            ax.set_ylim(0, max(zv, hv, target or 0) * 1.25)
        if target is not None:
            ax.axhline(target, color=MUT, lw=1, ls=(0, (4, 3)), zorder=1)
            ax.annotate("target", (1.45, target), fontsize=8.5, color=MUT,
                        va="bottom", ha="right")
        ax.set_xticks([0, 1])
        ax.set_xticklabels(["loss space $z$", "backbone $h$"], fontsize=9.5)
        ax.set_xlim(-0.55, 1.55)
        ax.set_title(title, fontsize=9.5, pad=8)
        ax.grid(axis="y", color=GRID, lw=0.6, zorder=0)
        _spines(ax)
    fig.tight_layout(w_pad=2.2)
    fig.savefig(f"{FIG}/fig_discrepancy.png")
    plt.close(fig)
    print(f"[exhibits] wrote {FIG}/fig_discrepancy.png")


# ------------------------------------------------- fig 2: treatment (main + appendix)
STATIONS = ["L03", "L06", "L09", "cls", "embed", "tap1", "tap2", "z.out"]
XLABELS = ["L3", "L6", "L9", "CLS", "emb", "p1", "p2", "z"]
FAMILIES = [
    ("LeJEPA", [("control", "lejepa_ctrl", "in100.lejepa.s0.e17c.ext"),
                ("treated", "lejepa_e20f", "in100.lejepa.s0.e20f.ep100.extL")], "#2e8b57"),
    ("VICReg", [("control", "vicreg_ctrl", "in100.vicreg.s0.e17c.ext"),
                ("treated", "vicreg_e20f", "in100.vicreg.s0.e20f.extL")], "#c23b3b"),
    ("SimCLR", [("control", "simclr_ctrl", "in100.simclr.s0.e17c.ext"),
                ("treated", "simclr_e20f", "in100.simclr.s0.e20f.extL")], "#3d65d0"),
    ("DINO", [("control", "dino_ctrl", "in100.dino.s0.e17c.ext"),
              # e20fwlo λ=.01 = the ruled winner dose (Berker 2026-08-26) — replaces the
              # λ=.258 overshoot arm (e20f) in every paper exhibit; the zoo keeps both.
              ("treated", "dino_e20fwlo", "in100.dino.s0.e20fwlo.extL")], "#d4820a"),
    ("BYOL", [("control", "byol_ctrl", "in100.byol.s0.e17c.ext"),
              ("treated", "byol_e20f", "in100.byol.s0.e20f.extL")], "#6a3fb5"),
    # house VISReg pair (E29, Berker 2026-08-26 "do the same for visreg treated +
    # untreated"): control + zoo-6%-share floor at the 512-d embedding (λ=.0123)
    ("VISReg", [("control", "visreg_ctrl", "in100.visreg.s0.extL"),
                ("treated", "visreg_f", "in100.visreg.s0.visregf.extL")], "#b3477d"),
    ("MAE", [("control", "mae_ctrl", "in100.mae.s0.extL"),
             ("treated", "mae_e20f", "in100.mae.s0.e20f.extL")], "#8a5cb8"),
    ("I-JEPA", [("control", "ijepa_ctrl", "in100.ijepa.s0.extL"),
                ("treated", "ijepa_e20f", "in100.ijepa.s0.e20f.extL")], "#0f7b8a"),
    ("Ours", [("z-only", "ours_zonly", "in100.floorssl.s0.d256vm4zonly.extL"),
              ("h+z", "ours_vm4", "in100.floorssl.s0.d256vm4.extL")], "#008300"),
]


def station_of(s):
    for k in ("L03", "L06", "L09"):
        if s.endswith(f".{k}"):
            return k if ".h.cls." in s else None
    m = {".z.dec.tap2": "tap1", ".z.dec.tap5": "tap2", ".z.dec.tap8": "z.out",
         ".h.gap": "gap", ".h.cls": "cls", ".z.embed": "embed"}
    for suf, st in m.items():
        if s.endswith(suf):
            return st
    if s.endswith("tap1"):
        return "tap1"
    if s.endswith("tap2"):
        return "tap2"
    if s.endswith(".z.proj.out") or s.endswith(".bottleneck") or s.endswith(".z.pred.out"):
        return "z.out"
    return None


def load_zoo_data():
    P, M, WB, WBo = {}, {}, {}, {}
    for _, members, _ in FAMILIES:
        for _, mlab, run in members:
            p = f"{ROOT}/results/probes/{run}.csv"
            if os.path.exists(p):
                for r in csv.DictReader(open(p)):
                    st = station_of(r["space"]) if r["space"].startswith("student") else None
                    if st:
                        P[(mlab, st, r["probe"])] = float(r["val_acc"])
    for mp in ("e17_depth_metrics", "e20_zoo_depth_metrics", "e20_ours_depth_metrics",
               "e20_new_arms_depth_metrics"):
        fp = f"{ROOT}/results/diag/{mp}.csv"
        if os.path.exists(fp):
            for r in csv.DictReader(open(fp)):
                st = station_of(r["space"])
                if st:
                    M[(r["ep"], st)] = r
    pref = {"dino_ctrl": "teacher", "dino_e20fwlo": "teacher"}
    rows = list(csv.DictReader(open(f"{ROOT}/results/diag/e23_retro_spaces.csv")))
    for _, members, _ in FAMILIES:
        for _, mlab, run in members:
            want = pref.get(mlab, "student")
            for r in rows:
                if r["run"] != run or r["framing"] != "raw":
                    continue
                st = station_of(r["space"])
                if st and (r["space"].startswith(want) or (mlab, st) not in WB):
                    WB[(mlab, st)] = (float(r["W"]), float(r["B"]))
            if (mlab, "z.out") in WB:
                WBo[mlab] = WB[(mlab, "z.out")]
    return P, M, WB, WBo


def make_treatment_fig(rows, quants, out, legend_note=True):
    P, M, WB, WBo = load_zoo_data()

    def mget(key, per_d=False):
        def g(mlab, st):
            r = M.get((mlab, st))
            v = None if r is None else r.get(key)
            if v in (None, "", "None"):
                return None
            return float(v) / float(r["d"]) if per_d else float(v)
        return g

    getters = {
        "Linear": lambda m, s: P.get((m, s, "linear_raw_v2")),
        "kNN-200": lambda m, s: P.get((m, s, "knn_v1_k200")),
        "RankMe / d": mget("rankme", per_d=True),
        "EffRank / d": mget("effective_rank", per_d=True),
        "Gaussian KL": mget("gauss_kl_total"),
        "cos margin (pos$-$rand)": lambda m, s: (
            lambda a, b: None if a is None or b is None else a - b)(
            mget("pos_cos")(m, s), mget("rand_cos")(m, s)),
        "class margin": lambda m, s: (
            lambda a, b: None if a is None or b is None else a - b)(
            mget("class_cos_same")(m, s), mget("class_cos_diff")(m, s)),
        "$\\Theta$ (W/B)": lambda m, s: (
            None if (m, s) not in WB else WB[(m, s)][0] / WB[(m, s)][1]),
        "a (s$\\to$z)": lambda m, s: (
            None if (m, s) not in WB or m not in WBo
            else (WBo[m][0] / WB[(m, s)][0]) ** 0.5),
        "b (s$\\to$z)": lambda m, s: (
            None if (m, s) not in WB or m not in WBo
            else (WBo[m][1] / WB[(m, s)][1]) ** 0.5),
        "$\\Lambda$ (s$\\to$z)": lambda m, s: (
            None if (m, s) not in WB or m not in WBo
            else ((WBo[m][1] / WB[(m, s)][1]) / (WBo[m][0] / WB[(m, s)][0])) ** 0.5),
    }
    fams = [f for f in FAMILIES if f[0] in rows]
    nr, nc = len(fams), len(quants)
    fig, axes = plt.subplots(nr, nc, figsize=(2.15 * nc, 1.55 * nr), squeeze=False)
    xs = np.arange(len(STATIONS))
    for i, (name, members, accent) in enumerate(fams):
        for j, q in enumerate(quants):
            ax = axes[i][j]
            vals = []
            for k, (lab, mlab, _) in enumerate(members):
                ys = [getters[q](mlab, st) for st in STATIONS]
                pts = [(x, y) for x, y in zip(xs, ys) if y is not None]
                if pts:
                    px, py = zip(*pts)
                    ax.plot(px, py, "-o", ms=2.6, lw=1.5,
                            color=CTRL if k == 0 else accent,
                            label=lab if j == 0 else None, zorder=2 + k)
                    vals += list(py)
            if vals:  # tight per-cell y so treatment separations stay visible
                lo, hi = min(vals), max(vals)
                pad = 0.10 * (hi - lo or abs(hi) or 1.0)
                ax.set_ylim(lo - pad, hi + pad)
            ax.set_xlim(-0.4, len(STATIONS) - 0.6)
            ax.set_xticks(xs)
            ax.set_xticklabels(XLABELS if i == nr - 1 else [], fontsize=6.5, rotation=45)
            ax.tick_params(axis="y", labelsize=7)
            ax.axvline(3.5, color=GRID, lw=0.8, zorder=1)      # trunk | head boundary
            ax.grid(color=GRID, lw=0.5, zorder=0)
            _spines(ax)
            if i == 0:
                ax.set_title(q, fontsize=9.5)
            if j == 0:
                ax.set_ylabel(name, fontsize=10)
                ax.legend(fontsize=6.5, frameon=False, loc="best",
                          handlelength=1.2, borderpad=0.2)
    if legend_note:
        fig.text(0.005, -0.012,
                 "DINO treated = winner-dose re-dose (λ=.01, E20 addendum 2026-08-26). "
                 "Vertical rule = backbone | projector boundary.",
                 fontsize=7.5, color=MUT)
    fig.tight_layout(h_pad=0.7, w_pad=0.9)
    fig.savefig(out)
    plt.close(fig)
    print(f"[exhibits] wrote {out}")


# ------------------------------------------------------------- placement table (cls)
def placement_table():
    # Linear = MAX over bench epochs (the Lightly protocol's own reported number and the
    # anchor's 64.11 convention; was the ep-90 row before 08-25 — Ours moves 68.41→68.49).
    ROWS = [("Ours (ViT-S/16)", "in1k.floorssl.s0.d256vm4.extL", None, 100),
            ("Ours (ViT-S/16, 10 views)", "in1k.floorssl.s0.e27v10u.extL", None, 100),
            ("LeJEPA (Lightly benchmark repro.)", "in1k.lejepa.s0.lightly.extL",
             (0.6411, 0.4706), 100),
            # D-102 public placement fleet (cls-only lean lifts; a row appears when its
            # bench + battery have landed — missing artifacts are skipped LOUDLY below)
            ("LeJEPA (OK-AI, ViT-S/16)", "in1k.pub.oklejepas300.ext", None, 300),
            ("DINO (OK-AI, ViT-S/16)", "in1k.pub.okdinos300.ext", None, 300),
            ("iBOT (OK-AI, ViT-S/16)", "in1k.pub.okibots300.ext", None, 300),
            ("MoCo v3 (ViT-S/16)", "in1k.pub.mocov3s300.ext", None, 300),
            ("LeJEPA (OK-AI, ViT-B/16)", "in1k.pub.oklejepab300.ext", None, 300),
            ("MoCo v3 (ViT-B/16)", "in1k.pub.mocov3b300.ext", None, 300),
            ("DINO (ViT-B/16)", "in1k.pub.dinob400.ext", None, 400),
            ("iBOT (ViT-B/16)", "in1k.pub.ibotb400.ext", None, 400),
            ("VISReg (ViT-B/16)", "in1k.pub.visregb400.ext", None, 400),
            ("iBOT (ViT-L/16)", "in1k.pub.ibotl250.ext", None, 250),
            ("VISReg (ViT-L/16)", "in1k.pub.visregl400.ext", None, 400)]
    comp = pd.read_csv(f"{ROOT}/results/compare/svit_zoo_computed_v2.csv")
    # store-computed cells (pos/rand cos, class margin, omega) for rows outside the zoo
    # cache: same computation (svit_zoo.compute_store_metrics), own compute-if-missing
    # cache — the pub stores are lean (cls station only; per-station guards skip the rest)
    from experiments.svit_zoo import compute_store_metrics
    pcache_f = f"{ROOT}/results/compare/placement_cells.csv"
    pcache = pd.read_csv(pcache_f) if os.path.exists(pcache_f) else pd.DataFrame(
        columns=["run_id", "station", "metric", "value", "manifest"])
    fresh = []
    for _, rid, _, _ in ROWS:
        if not (comp.run_id == rid).any() and not (pcache.run_id == rid).any() \
                and os.path.isdir(f"{ROOT}/features/{rid}"):
            fresh += compute_store_metrics(rid)
    if fresh:
        pcache = pd.concat([pcache, pd.DataFrame(fresh)], ignore_index=True)
        pcache.to_csv(pcache_f, index=False)
    comp = pd.concat([comp, pcache], ignore_index=True)
    out = []
    for label, rid, cert, ep in ROWS:
        bench_f = f"{ROOT}/results/probes/{rid}.bench.csv"
        knn_f = f"{ROOT}/results/probes/{rid}.bench_knn.csv"
        batt_f = f"{ROOT}/results/battery/{rid}.csv"
        # a row enters only on a COMPLETE bench (ep-90 row present) — a mid-bench CSV
        # would quote a partial curve as the final (the mocov3s300 re-bench race, 08-25)
        bench_done = os.path.exists(bench_f) and (pd.read_csv(bench_f).epoch == 90).any()
        if not cert and not (bench_done and os.path.exists(batt_f)):
            print(f"[exhibits] placement row SKIPPED (artifacts pending): {label} ({rid})")
            continue
        r = {"Method": label, "Ep.": ep}
        if cert:
            r["Linear"], r["kNN-200"] = cert
        else:
            b = pd.read_csv(bench_f)
            r["Linear"] = float(b.val_top1.max())
            if os.path.exists(knn_f):
                k = pd.read_csv(knn_f)
                r["kNN-200"] = float(k[k.probe == "knn_k200_t0.07"].val_acc.iloc[0])
            else:
                print(f"[exhibits] kNN rider missing for {rid} — cell left blank")
                r["kNN-200"] = np.nan
        bd = pd.read_csv(f"{ROOT}/results/battery/{rid}.csv")
        man = next((m for m in ("in1k.train.v1L", "in1k.train.v1",
                                "in1k.val.v1L", "in1k.val.v1")
                    if (bd.manifest == m).any()), None)
        bd = bd[(bd.manifest == man) & (bd.space == "student.h.cls")
                & (bd.variant == "raw|full")]
        g = lambda m: float(bd[bd.metric == m].value.iloc[0])
        d = float(bd[bd.metric == "rankme"].d.iloc[0])
        r.update({"RankMe/d": g("rankme") / d, "EffRank/d": g("effective_rank") / d,
                  "Gauss. KL": g("gauss_kl_full.total"), "Epps–Pulley": g("epps_pulley"),
                  "Kurt. (worst)": g("kurt_topeig.worst")})
        c = comp[(comp.run_id == rid) & (comp.station == "cls")]
        cg = lambda m: float(c[c.metric == m].value.iloc[0]) if (c.metric == m).any() else np.nan
        r.update({"cos margin": cg("pos_cos") - cg("rand_cos"),
                  "class margin": cg("class_margin"), "$\\Theta$": cg("omega")})
        out.append(r)
    df = pd.DataFrame(out)
    df.to_csv(f"{ROOT}/results/compare/placement_table.csv", index=False)
    cols = list(df.columns)
    # adjustbox wrap (Berker 08-25): the metrics table is wider than \textwidth
    lines = ["\\begin{adjustbox}{max width=\\textwidth}",
             "\\begin{tabular}{l" + "r" * (len(cols) - 1) + "}", "\\toprule",
             " & ".join(cols) + " \\\\", "\\midrule"]
    for _, r in df.iterrows():
        cells = [r["Method"]] + ["—" if pd.isna(r[c]) else f"{int(r[c])}" if c == "Ep."
                                 else f"{r[c]:.3f}" if abs(r[c]) < 10 else f"{r[c]:.1f}"
                                 for c in cols[1:]]
        lines.append(" & ".join(cells) + " \\\\")
    lines += ["\\bottomrule", "\\end{tabular}", "\\end{adjustbox}"]
    _write_tex_block("placement-table", lines)
    print("[exhibits] wrote placement_table.csv + updated main.tex placement-table block")
    print(df.to_string(index=False))


def _write_tex_block(name, lines):
    """Replace the marked auto-gen block in main.tex (tables live INLINE, Berker 08-24).
    Comment lines directly under the BEGIN marker are documentation and are preserved;
    generated content never starts with '%', so rewrites are idempotent."""
    mt = f"{ROOT}/docs/paper/main.tex"
    src = open(mt).read()
    j = src.index(f"% END {name}")
    i = src.index("\n", src.index(f"% BEGIN {name}")) + 1
    while i < j:
        nl = src.index("\n", i)
        if not src[i:nl].lstrip().startswith("%"):
            break
        i = nl + 1
    open(mt, "w").write(src[:i] + "\n".join(lines) + "\n" + src[j:])


# --------------------------------------------- main B/L comparison + OOD + seg skeletons
def comparison_tables():
    """The paper's main comparison rows (Berker 2026-08-25: "our main comparisons will be
    b/l models but keep the performances as placeholders"): externals under our
    bench_linear_v1 yardstick, Ours 400-ep cells = placeholders until the D-103 runs
    land; OOD-shift + ADE20k segmentation tables share the same row set as skeletons."""
    # ep100 OK-AI revisions (wave 2, benches in flight 08-25) = the only epoch-matched
    # externals in existence; rows fill on landing. Ours-S = vm4 (the declared small
    # model) so the S block reads against something — drop if the S block is unwanted.
    ROWS = [("LeJEPA (OK-AI)", "ViT-S/16", 100, "in1k.pub.oklejepas100.ext"),
            ("DINO (OK-AI)", "ViT-S/16", 100, "in1k.pub.okdinos100.ext"),
            ("iBOT (OK-AI)", "ViT-S/16", 100, "in1k.pub.okibots100.ext"),
            # the certified anchor (their code end-to-end, D-094), rescued from the
            # discarded tab:lightly-benchmark (Berker 08-25); values = our yardstick
            ("LeJEPA (Lightly repro.)", "ViT-S/16", 100, ("cite", 0.6411, 0.4706)),
            ("Ours", "ViT-S/16", 100, "in1k.floorssl.s0.d256vm4.extL"),
            "MIDRULE",
            ("LeJEPA (OK-AI)", "ViT-B/16", 100, "in1k.pub.oklejepab100.ext"),
            ("DINO (OK-AI)", "ViT-B/16", 100, "in1k.pub.okdinob100.ext"),
            ("iBOT (OK-AI)", "ViT-B/16", 100, "in1k.pub.okibotb100.ext"),
            ("LeJEPA (OK-AI)", "ViT-B/16", 300, "in1k.pub.oklejepab300.ext"),
            ("MoCo v3", "ViT-B/16", 300, "in1k.pub.mocov3b300.ext"),
            ("DINO", "ViT-B/16", 400, "in1k.pub.dinob400.ext"),
            ("iBOT", "ViT-B/16", 400, "in1k.pub.ibotb400.ext"),
            ("VISReg", "ViT-B/16", 400, "in1k.pub.visregb400.ext"),
            # the E30 matched anchor (Berker 08-27): their code end-to-end on the
            # shipped 4g+6l ViT-B config, same arch/epochs/batch as Ours B-100 —
            # the epoch-matched external our B row is actually paired against
            ("VISReg (repro.)", "ViT-B/16", 100, "in1k.visreg.s0.vrb.extL"),
            ("Ours", "ViT-B/16", 100, "in1k.floorssl.s0.e27lm4sbe.extL"),
            ("Ours", "ViT-B/16", 400, None),
            "MIDRULE",
            # L/H field block + separated Ours block: Berker's hand arrangement
            # 08-25 ("keep vit l vit b etc comparing the field") — midrules are these
            # explicit markers, NOT arch transitions (I-JEPA ViT-H sits inside the field)
            ("iBOT", "ViT-L/16", 250, "in1k.pub.ibotl250.ext"),
            ("VISReg", "ViT-L/14", 400, "in1k.pub.visregl400.ext"),
            ("MAE", "ViT-L/16", 1600, "in1k.pub.mael1600.ext"),
            ("data2vec", "ViT-L/16", 1600, "in1k.pub.d2vl1600.ext"),
            # cite-only (Berker: "let lejepa vit-l enter the table with a footnote"):
            # NO public ckpt exists (HF-wide search) — the LeJEPA paper's own number
            ("LeJEPA$^\\dagger$", "ViT-L/14", 100, ("cite", 0.756, None)),
            ("I-JEPA", "ViT-H/14", 300, "in1k.pub.ijepah300.ext"),
            "MIDRULE",
            ("Ours", "ViT-L/16", 100, "in1k.floorssl.s0.e27lm4Ls5b.extL"),
            ("Ours", "ViT-L/16", 400, None)]

    def bench_cells(rid):
        if isinstance(rid, tuple) and rid[0] == "cite":
            return rid[1], rid[2]
        bench_f = f"{ROOT}/results/probes/{rid}.bench.csv"
        if rid is None or not os.path.exists(bench_f):
            return None, None
        b = pd.read_csv(bench_f)
        if not (b.epoch == 90).any():              # mid-bench = not a final
            return None, None
        lin = float(b.val_top1.max())
        knn_f = f"{ROOT}/results/probes/{rid}.bench_knn.csv"
        knn = None
        if os.path.exists(knn_f):
            k = pd.read_csv(knn_f)
            t7 = k[k.probe == "knn_k200_t0.07"].val_acc
            knn = float(t7.iloc[0]) if len(t7) else None
        return lin, knn

    fmt = lambda v: "---" if v is None else f"{100 * v:.1f}"
    main_lines = ["\\begin{tabular}{llrrr}", "\\toprule",
                  "Method & Backbone & Ep. & Linear & kNN \\\\", "\\midrule"]
    # OOD suite = the VISReg paper's six domain-shift sets (Berker 08-25: "we will
    # follow"; their protocol: frozen encoder, concat CLS of last 4 layers, linear+BN)
    ood_lines = ["\\begin{tabular}{llrrrrrrr}", "\\toprule",
                 "Method & Backbone & Ep. & ChestX & RetinaMNIST & OrganAMNIST & "
                 "Galaxy10 & AID & DTD \\\\", "\\midrule"]
    for row in ROWS:
        if row == "MIDRULE":
            for ls in (main_lines, ood_lines):
                ls.append("\\midrule")
            continue
        label, arch, ep, rid = row
        lin, knn = bench_cells(rid)
        main_lines.append(f"{label} & {arch} & {ep} & {fmt(lin)} & {fmt(knn)} \\\\")
        ood_lines.append(f"{label} & {arch} & {ep}" + " & ---" * 6 + " \\\\")
    for ls in (main_lines, ood_lines):
        ls += ["\\bottomrule", "\\end{tabular}"]
    _write_tex_block("in1k-main-table", main_lines)
    _write_tex_block("ood-table", ood_lines)
    _transfer_seg_tables()
    print("[exhibits] wrote in1k-main + ood + seg table blocks")
    for row in ROWS:
        if row == "MIDRULE":
            continue
        label, arch, ep, rid = row
        lin, knn = bench_cells(rid)
        print(f"  {label:16} {arch} {ep:4d}  lin {fmt(lin):>5}  knn {fmt(knn):>5}")


def _transfer_seg_tables():
    """tab:transfer + tab:seg (Berker 08-25: cite-where-published, matched-impl
    elsewhere). Cited rows (ddag) = the VISReg paper's Table 5 / Table 8 numbers
    verbatim (arXiv HTML read 2026-08-25) under THEIR protocols (transfer: concat-4-CLS
    LP 10 ep, 13-LR grid; seg: 1x1 conv on frozen last-layer patch features, 40 ep,
    ViT-B/16 rows only). Self-run rows auto-fill from results/transfer/<tag>.visreg_lp.csv
    (E30 evaluator, port-validated on the printed DINO-B row to |d|<=.4 per set — the
    license for cited/self-run adjacency); the VISReg (repro.) row = the E30 matched
    anchor (same arch/epochs/geometry as Ours B-100)."""
    D = "$^\\ddag$"
    TRANSFER_CITED = [  # (label, arch, ep, [DTD,Airc,Cars,C10,C100,Flow,Food,Pets], avg)
        ("MoCo v3" + D, "ViT-B/16", 300, [73.7, 57.9, 67.5, 96.9, 85.2, 91.5, 81.8, 89.8], 80.5),
        ("DINO" + D, "ViT-B/16", 400, [74.3, 63.6, 73.9, 96.5, 85.0, 94.6, 83.1, 93.6], 83.1),
        ("iBOT" + D, "ViT-B/16", 400, [74.1, 63.5, 73.8, 97.1, 85.9, 93.7, 84.2, 93.6], 83.2),
        ("iBOT" + D, "ViT-L/16", 250, [75.3, 66.0, 76.1, 97.5, 87.2, 94.0, 86.1, 94.0], 84.5),
        ("I-JEPA" + D, "ViT-H/14", 300, [69.9, 55.4, 59.2, 97.2, 85.5, 86.8, 83.3, 92.8], 78.7),
        ("data2vec" + D, "ViT-L/14", 1600, [69.7, 43.9, 38.7, 96.9, 83.7, 81.4, 79.6, 83.0], 72.1),
        ("MAE" + D, "ViT-L/16", 1600, [72.8, 61.9, 61.5, 93.3, 78.0, 85.4, 78.6, 91.3], 77.8),
        ("VISReg" + D, "ViT-B/16", 400, [75.7, 57.1, 64.8, 94.6, 78.8, 90.4, 82.9, 88.3], 79.1),
        ("VISReg" + D, "ViT-L/14", 400, [76.5, 56.6, 66.2, 94.1, 71.9, 90.2, 83.3, 89.2], 78.5),
        ("LeJEPA" + D, "ViT-L/14", 100, [78.3, 57.0, 57.3, 96.5, 83.7, 91.2, 82.1, 89.7], 79.5),
    ]
    TRANSFER_ORDER = ["dtd", "aircraft", "cars", "cifar10", "cifar100", "flowers",
                      "food", "pets"]

    def transfer_cells(tag):
        f = f"{ROOT}/results/transfer/{tag}.visreg_lp.csv"
        if tag is None or not os.path.exists(f):
            return None
        t = pd.read_csv(f, header=None,
                        names=["tag", "ds", "ep", "ntr", "nte", "lr", "acc"])
        vals = {r.ds: 100 * r.acc for r in t.itertuples()}
        out = [vals.get(d) for d in TRANSFER_ORDER]
        return out if all(v is not None for v in out) else None

    SELF_RUN = [("VISReg (repro.)", "ViT-B/16", 100, "in1k.visreg.s0.vrb"),
                ("Ours", "ViT-S/16", 100, None),
                ("Ours", "ViT-B/16", 100, "in1k.floorssl.s0.e27lm4sbe"),
                ("Ours", "ViT-B/16", 400, None),
                ("Ours", "ViT-L/16", 100, "in1k.floorssl.s0.e27lm4Ls5b"),
                ("Ours", "ViT-L/16", 400, None)]
    tl = ["\\begin{adjustbox}{max width=\\textwidth}", "\\begin{tabular}{llrrrrrrrrrr}",
          "\\toprule", "Method & Backbone & Ep. & DTD & Aircraft & Cars & CIFAR10 & "
          "CIFAR100 & Flowers & Food & Pets & Avg. \\\\", "\\midrule"]
    for label, arch, ep, vals, avg in TRANSFER_CITED:
        tl.append(f"{label} & {arch} & {ep} & " + " & ".join(f"{v:.1f}" for v in vals)
                  + f" & {avg:.1f} \\\\")
    tl.append("\\midrule")
    for label, arch, ep, tag in SELF_RUN:
        vals = transfer_cells(tag)
        if vals is None:
            tl.append(f"{label} & {arch} & {ep}" + " & ---" * 9 + " \\\\")
        else:
            tl.append(f"{label} & {arch} & {ep} & "
                      + " & ".join(f"{v:.1f}" for v in vals)
                      + f" & {sum(vals) / len(vals):.1f} \\\\")
    tl += ["\\bottomrule", "\\end{tabular}", "\\end{adjustbox}"]
    _write_tex_block("transfer-table", tl)

    SEG_CITED = [("MoCo v3" + D, "ViT-B/16", 300, 31.69), ("DINO" + D, "ViT-B/16", 400, 29.40),
                 ("data2vec" + D, "ViT-B/16", None, 21.99), ("MAE" + D, "ViT-B/16", None, 23.60),
                 ("VISReg" + D, "ViT-B/16", 400, 30.16)]
    sl = ["\\begin{tabular}{llrr}", "\\toprule", "Method & Backbone & Ep. & mIoU \\\\",
          "\\midrule"]
    for label, arch, ep, miou in SEG_CITED:
        sl.append(f"{label} & {arch} & {ep if ep else '---'} & {miou:.2f} \\\\")
    sl.append("\\midrule")
    for label, arch, ep in (("Ours", "ViT-B/16", 400), ("Ours", "ViT-L/16", 400)):
        sl.append(f"{label} & {arch} & {ep} & --- \\\\")
    sl += ["\\bottomrule", "\\end{tabular}"]
    _write_tex_block("seg-table", sl)
    print("[exhibits] wrote transfer (cited) + seg (cited) table blocks")


# ------------------------------------------------- fig: placement scatter (main paper)
def placement_scatter():
    """Berker 08-25: accuracy vs structure for every placement row, main-paper grade.
    y = bench linear top-1; x = Theta (retained view-sensitivity at cls) and EffRank/d.
    Ours = accent, field = gray; marker = arch (S circle, B square, L triangle, H
    diamond); rows appear as their bench+battery land (reads placement_table.csv)."""
    df = pd.read_csv(f"{ROOT}/results/compare/placement_table.csv")
    df = df.dropna(subset=["Linear", "$\\Theta$", "EffRank/d"])

    def short(m):
        arch = "S" if "ViT-S" in m else "B" if "ViT-B" in m else \
               "L" if "ViT-L" in m else "H" if "ViT-H" in m else "S"
        fam = m.split(" (")[0]
        if "Lightly" in m:
            return "LeJEPA-S (repro)", "S"
        if "10 views" in m:
            return "Ours-S (10v)", "S"
        return f"{fam}-{arch}", arch

    MARK = {"S": "o", "B": "s", "L": "^", "H": "D"}
    fig, axes = plt.subplots(1, 2, figsize=(8.6, 3.6), sharey=True)
    for ax, xcol, xlab in ((axes[0], "$\\Theta$", "retained view-sensitivity $\\Theta_h$"),
                           (axes[1], "EffRank/d", "effective rank / $d$")):
        placed = []
        for _, r in df.iterrows():
            lab, arch = short(r["Method"])
            ours = r["Method"].startswith("Ours")
            ax.scatter(r[xcol], 100 * r["Linear"], s=64 if ours else 52,
                       marker=MARK.get(arch, "o"),
                       color=Z_COL if ours else CTRL, edgecolor="white",
                       linewidth=0.9, zorder=3)
            placed.append((r[xcol], 100 * r["Linear"], lab, ours))
        # collision-dodged labels: near-vertical neighbors alternate above/below
        xr = ax.dataLim.width or 1.0
        done = []
        for x, y, lab, ours in sorted(placed, key=lambda p: p[1]):
            dy = 3
            for px, py, pdy in done:
                if abs(x - px) < 0.14 * xr and abs(y - py) < 1.4 and pdy > 0:
                    dy = -11
            done.append((x, y, dy))
            ax.annotate(lab, (x, y), xytext=(5, dy), textcoords="offset points",
                        fontsize=7.3, color=INK if ours else MUT, zorder=4)
        ax.set_xlabel(xlab, fontsize=9.5)
        ax.grid(color=GRID, lw=0.6, zorder=0)
        _spines(ax)
    axes[0].set_ylabel("ImageNet-1k linear top-1 (%)", fontsize=9.5)
    from matplotlib.lines import Line2D
    handles = [Line2D([], [], marker="o", ls="", color=Z_COL, label="Ours"),
               Line2D([], [], marker="o", ls="", color=CTRL, label="Public checkpoints"),
               Line2D([], [], marker="o", ls="", color=MUT, mfc="none", label="ViT-S"),
               Line2D([], [], marker="s", ls="", color=MUT, mfc="none", label="ViT-B"),
               Line2D([], [], marker="^", ls="", color=MUT, mfc="none", label="ViT-L"),
               Line2D([], [], marker="D", ls="", color=MUT, mfc="none", label="ViT-H")]
    axes[1].legend(handles=handles, fontsize=7.5, frameon=False, loc="lower right",
                   ncol=2, handletextpad=0.2, columnspacing=0.8)
    fig.tight_layout(w_pad=1.6)
    fig.savefig(f"{FIG}/fig_placement_scatter.png")
    plt.close(fig)
    print(f"[exhibits] wrote {FIG}/fig_placement_scatter.png")


# --------------------------- fig: in-house IN-100 scatter + the C8 org-vs-price recast
def fig_treatment_arrows():
    """Berker 08-25: linear + kNN as panels 1/2 of ONE figure (renamed — arrows, not a
    scatter). Per-family control -> +floor arrows in (Theta_h, accuracy) space on the
    matched IN-100 zoo; Ours = black emphasis; DINO = the e20fwlo winner-dose arm
    (λ=.01, replaces the overshoot arm per Berker 2026-08-26)."""
    import matplotlib.patheffects as pe
    P, M, WB, WBo = load_zoo_data()
    halo = [pe.withStroke(linewidth=2.6, foreground="white")]
    PANELS = [("linear_raw_v2", "IN-100 linear top-1 (%)",
               {"Ours": (7, 4), "LeJEPA": (-52, -3), "DINO": (8, -3),
                "VICReg": (-48, 2), "SimCLR": (8, -9), "BYOL": (8, 1),
                "VISReg": (8, 5)}),
              ("knn_v1_k200", "IN-100 kNN-200 top-1 (%)",
               {"Ours": (7, 4), "LeJEPA": (-52, -12), "DINO": (8, -3),
                "VICReg": (7, -11), "SimCLR": (8, -9), "BYOL": (9, 5),
                "VISReg": (8, 5)})]
    fig, axes = plt.subplots(1, 2, figsize=(8.8, 3.6))
    for ax, (probe, ylab, OFF) in zip(axes, PANELS):
        for name, members, accent in FAMILIES:
            if name in ("MAE", "I-JEPA"):
                continue
            if name == "Ours":
                accent = INK
            pts = []
            for lab, mlab, _ in members:
                th = None if (mlab, "cls") not in WB else \
                    WB[(mlab, "cls")][0] / WB[(mlab, "cls")][1]
                v = P.get((mlab, "cls", probe))
                if th is not None and v is not None:
                    pts.append((th, 100 * v))
            if len(pts) == 2:
                (x0, y0), (x1, y1) = pts
                ax.annotate("", (x1, y1), (x0, y0), zorder=2,
                            arrowprops=dict(arrowstyle="-|>", color=accent,
                                            lw=2.2 if name == "Ours" else 1.5,
                                            shrinkA=4.5, shrinkB=4.5,
                                            mutation_scale=13))
            for k, (x, y) in enumerate(pts):
                ax.scatter(x, y, s=52 if name == "Ours" else 44, color=accent,
                           zorder=3, facecolor="white" if k == 0 else accent,
                           linewidth=1.5)
            if pts:
                x1, y1 = pts[-1]
                dx, dy = OFF.get(name, (6, 3))
                lab = name
                ax.annotate(lab, (x1, y1), xytext=(dx, dy),
                            textcoords="offset points", fontsize=8.6, color=accent,
                            zorder=5, path_effects=halo,
                            fontweight="bold" if name == "Ours" else "normal")
        ax.set_xlabel("retained view-sensitivity $\\Theta_h$ at CLS", fontsize=9.5)
        ax.set_ylabel(ylab, fontsize=9.5)
        ax.grid(color=GRID, lw=0.6, zorder=0)
        _spines(ax)
    from matplotlib.lines import Line2D
    axes[0].legend(handles=[Line2D([], [], marker="o", ls="", mfc="white", color=MUT,
                                   label="control"),
                            Line2D([], [], marker="o", ls="", color=MUT,
                                   label="+ moment floor")],
                   fontsize=8.2, frameon=False, loc="lower right", borderaxespad=0.3)
    fig.tight_layout(w_pad=1.6)
    fig.savefig(f"{FIG}/fig_treatment_arrows.png")
    plt.close(fig)
    print(f"[exhibits] wrote {FIG}/fig_treatment_arrows.png")


C8_PAIRS = [  # (family, control classes csv, treated classes csv, accent)
    ("LeJEPA", "in100.lejepa.s0.ext", "in100.lejepa.s0.e20f.ext", "#2e8b57"),
    ("VICReg", "in100.vicreg.s0.ext", "in100.vicreg.s0.e20f.ext", "#c23b3b"),
    ("SimCLR", "in100.simclr.s0.ext", "in100.simclr.s0.e20f.ext", "#3d65d0"),
    ("DINO", "in100.dino.s0.ext", "in100.dino.s0.e20fwlo.extL", "#d4820a"),
    ("BYOL", "in100.byol.s0.ext", "in100.byol.s0.e20f.ext", "#6a3fb5"),
    ("VISReg", "in100.visreg.s0.extL", "in100.visreg.s0.visregf.extL", "#b3477d"),
    ("Ours", "in100.floorssl.s0.d256vm4zonly.extL", "in100.floorssl.s0.d256vm4.extL",
     "#008300"),
]


def fig_org_vs_sensitivity():
    """Redesigned 08-25 (Berker: neutral wording — the sensitivity term is measured, not
    graded; no error/error geometry; per-class dots must MEAN one class). Panel A: per-
    model mean arrows, control -> +floor, in (class-direction view-sensitivity,
    center separability) space — up = better organized. Panel B: our pair per class —
    one arrow per class (96/100 up = the per-class version of the claim)."""
    import matplotlib.patheffects as pe
    halo = [pe.withStroke(linewidth=2.6, foreground="white")]
    OFF = {"Ours": (8, 2), "LeJEPA": (-52, -14), "VICReg": (9, -11), "DINO": (8, -2),
           "SimCLR": (-24, 12), "BYOL": (9, 7), "VISReg": (8, -6)}

    def cls_frame(run):
        p = f"{ROOT}/results/twospace/{run}.classes.csv"
        if not os.path.exists(p):
            return None
        d = pd.read_csv(p).drop_duplicates("cls").set_index("cls")
        x = (d["decomp_pred"] - d["d_h"] ** 2).clip(lower=0)
        return pd.DataFrame({"x": x, "sep": 1 - d["d_h"] ** 2})

    fig, (axA, axB) = plt.subplots(1, 2, figsize=(8.8, 3.6), sharey=True)
    for name, ctrl, treat, accent in C8_PAIRS:
        if name == "Ours":
            accent = INK
        fc, ft = cls_frame(ctrl), cls_frame(treat)
        if fc is None or ft is None:
            print(f"[exhibits] org-vs-sens: missing arm for {name}")
            continue
        (x0, y0) = float(fc.x.mean()), float(fc.sep.mean())
        (x1, y1) = float(ft.x.mean()), float(ft.sep.mean())
        axA.annotate("", (x1, y1), (x0, y0), zorder=3,
                     arrowprops=dict(arrowstyle="-|>", color=accent,
                                     lw=2.2 if name == "Ours" else 1.6,
                                     shrinkA=3, shrinkB=3, mutation_scale=13))
        axA.scatter([x0], [y0], s=38, facecolor="white", edgecolor=accent,
                    linewidth=1.5, zorder=4)
        axA.scatter([x1], [y1], s=38, color=accent, zorder=4)
        dx, dy = OFF.get(name, (7, 2))
        axA.annotate(name, (x1, y1), xytext=(dx, dy), textcoords="offset points",
                     fontsize=8.6, color=accent, zorder=5, path_effects=halo,
                     fontweight="bold" if name == "Ours" else "normal")
        if name == "Ours":
            up = (ft.sep > fc.sep)
            for i in fc.index:
                col = Z_COL if up.loc[i] else "#b0731d"
                axB.annotate("", (ft.x.loc[i], ft.sep.loc[i]),
                             (fc.x.loc[i], fc.sep.loc[i]), zorder=2,
                             arrowprops=dict(arrowstyle="->",
                                             color=col, lw=0.7, alpha=0.5,
                                             shrinkA=0, shrinkB=0))
            axB.annotate(f"{int(up.sum())}/100 classes improve", (0.97, 0.05),
                         xycoords="axes fraction", ha="right", fontsize=9,
                         color=INK, path_effects=halo)
    for ax, ttl in ((axA, "per-model mean over 100 classes"),
                    (axB, "our pair, one arrow per class")):
        ax.set_xlabel("view-sensitivity along the class direction\n"
                      "$a_c^\\top\\Theta_h(I+\\Theta_h)^{-1}a_c$", fontsize=9)
        ax.grid(color=GRID, lw=0.6, zorder=0)
        ax.set_title(ttl, fontsize=9.5)
        _spines(ax)
    axA.set_ylabel("class-center separability  $1-d_h^2$  ($\\uparrow$)", fontsize=9.5)
    axA.set_ylim(-0.55, 1.02)
    for ax in (axA, axB):
        ax.set_xlim(0.0, 0.16)
    from matplotlib.lines import Line2D
    axA.legend(handles=[Line2D([], [], marker="o", ls="", mfc="white", color=MUT,
                               label="control"),
                        Line2D([], [], marker="o", ls="", color=MUT,
                               label="+ moment floor")],
               fontsize=8, frameon=False, loc="lower right", borderaxespad=0.3)
    fig.tight_layout(w_pad=1.4)
    fig.savefig(f"{FIG}/fig_org_vs_sensitivity.png")
    plt.close(fig)
    print(f"[exhibits] wrote {FIG}/fig_org_vs_sensitivity.png")


if __name__ == "__main__":
    os.makedirs(FIG, exist_ok=True)
    fig_discrepancy()
    MAIN_Q = ["Linear", "kNN-200", "RankMe / d", "$\\Theta$ (W/B)",
              "cos margin (pos$-$rand)"]
    ALL_Q = ["Linear", "kNN-200", "RankMe / d", "EffRank / d", "Gaussian KL",
             "cos margin (pos$-$rand)", "class margin", "$\\Theta$ (W/B)",
             "a (s$\\to$z)", "b (s$\\to$z)", "$\\Lambda$ (s$\\to$z)"]
    make_treatment_fig(["LeJEPA", "VICReg", "SimCLR", "DINO", "VISReg"], MAIN_Q,
                       f"{FIG}/fig_treatment_main.png")
    make_treatment_fig([f[0] for f in FAMILIES], ALL_Q,
                       f"{FIG}/fig_treatment_appendix.png")
    placement_table()
    comparison_tables()
    placement_scatter()
    fig_treatment_arrows()
    fig_org_vs_sensitivity()
