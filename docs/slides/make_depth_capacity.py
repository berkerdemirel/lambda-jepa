"""Slide-local exhibit: stable source-center capacity ALONG THE WHOLE STACK — trunk
taps, projector taps, and the loss space z (Berker 2026-08-26: "across layers instead
of epochs", then "show projector layers and z too").

Reads only existing audit CSVs (results/twospace/*.csv `capacity` rows: b_rank = kept
eigenvalues of B_hat at that station; results/diag/*depth_metrics.csv for each
station's width d) — no compute, no new measurement. Store resolution copies C7's
ladder (.extL -> .ep100.extL -> .ep100.o8 -> .ext) so control arms resolve to their
layered ep100 stores.

Stations have different widths (trunk 384, projector 2048/4096, z 256/512/2048), so
the plotted quantity is b_rank / d: the fraction of that layer's own width carrying
stable center capacity. Same normalization convention as the paper's RankMe/d and
EffRank/d panels. Pairs whose h is declared at a head tap (lejepa, visreg: z.embed)
and dino's ruled winner arm (no layer stations audited yet) are left out.

  .venv/bin/python docs/slides/make_depth_capacity.py
"""
import csv
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUT = os.path.join(ROOT, "docs/slides/assets/depth_capacity.png")
CTRL, TREAT, MUT, GRID = "#4A3AA7", "#008300", "#6B6B6B", "#E3E3E3"
ORDER = ["L03", "L06", "L09", "cls", "tap1", "tap2", "z.out"]
XLAB = ["L3", "L6", "L9", "CLS", "p1", "p2", "$z$"]
BOUNDARY = 3.5                                  # backbone | projector
BRANCH = {"DINO": "teacher"}                    # declared branch (D-036/paper convention)
PAIRS = [("VICReg", "in100.vicreg.s0.ext", "in100.vicreg.s0.e20f.ext"),
         ("SimCLR", "in100.simclr.s0.ext", "in100.simclr.s0.e20f.ext"),
         ("BYOL", "in100.byol.s0.ext", "in100.byol.s0.e20f.ext"),
         ("Ours", "in100.floorssl.s0.d256vm4zonly.extL",
          "in100.floorssl.s0.d256vm4.extL")]
DLABEL = {"in100.vicreg.s0.ext": "vicreg_ctrl", "in100.vicreg.s0.e20f.ext": "vicreg_e20f",
          "in100.simclr.s0.ext": "simclr_ctrl", "in100.simclr.s0.e20f.ext": "simclr_e20f",
          "in100.byol.s0.ext": "byol_ctrl", "in100.byol.s0.e20f.ext": "byol_e20f",
          "in100.floorssl.s0.d256vm4zonly.extL": "ours_zonly",
          "in100.floorssl.s0.d256vm4.extL": "ours_vm4"}


def station(s):
    for k in ("L03", "L06", "L09"):
        if s.endswith(f".{k}"):
            return k
    for suf, st in ((".z.dec.tap2", "tap1"), (".z.dec.tap5", "tap2"),
                    (".z.dec.tap8", "z.out"), (".h.cls", "cls"), (".z.embed", "embed")):
        # NB: .h.gap.* also ends in .L03/.L06/.L09 — callers filter it out (CLS tap only)
        if s.endswith(suf):
            return st
    if s.endswith("tap1"):
        return "tap1"
    if s.endswith("tap2"):
        return "tap2"
    if s.endswith(".z.proj.out") or s.endswith(".bottleneck"):
        return "z.out"                          # BYOL: projector output is z, not .z.pred.out
    return None


WIDTH = {}
for f in ("e17_depth_metrics", "e20_zoo_depth_metrics", "e20_ours_depth_metrics",
          "e20_new_arms_depth_metrics"):
    for r in csv.DictReader(open(f"{ROOT}/results/diag/{f}.csv")):
        st = station(r["space"])
        if st:
            WIDTH[(r["ep"], st)] = int(float(r["d"]))


def load(run):
    p = f"{ROOT}/results/twospace/{run}.csv"
    return list(csv.DictReader(open(p))) if os.path.exists(p) else None


def layered(run):                               # C7's resolution ladder
    for c in (run.replace(".ext", ".extL"), run.replace(".ext", ".ep100.extL"),
              run.replace(".ext", ".ep100.o8"), run):
        if load(c) is not None:
            return c
    return run


def profile(run, branch="student"):
    """b_rank / d per station on the SHARED audit_v1 view stack (never the method's
    own-aug store). Ambiguity is an error, not a silent first match."""
    rows, lab = load(layered(run)), DLABEL[run]
    out = []
    for st in ORDER:
        hit = {float(r["b_rank"]) for r in rows
               if r["kind"] == "capacity" and r["b_rank"] and station(r["space"]) == st
               and r["space"].startswith(f"{branch}.") and ".h.gap" not in r["space"]
               and "audit_v1" in r["manifest"]}
        assert len(hit) <= 1, f"{run} {st}: {hit}"
        d = WIDTH.get((lab, st))
        out.append(hit.pop() / d if hit and d else None)
    return out


plt.rcParams.update({"font.size": 9, "axes.linewidth": 0.8, "figure.dpi": 150,
                     "savefig.bbox": "tight"})
fig, axes = plt.subplots(1, len(PAIRS), figsize=(10.4, 3.3), sharey=True)
x = range(len(ORDER))
for ax, (name, ctrl, trt) in zip(axes, PAIRS):
    ax.axvline(BOUNDARY, color=MUT, lw=0.9, zorder=1)
    ax.axhline(1.0, color=MUT, lw=0.8, ls=(0, (2, 3)), zorder=1)
    for run, col, ls, lab in ((ctrl, CTRL, "--", "control"),
                              (trt, TREAT, "-", "+ the $h$ term")):
        v = profile(run, BRANCH.get(name, "student"))
        pts = [(k, y) for k, y in zip(x, v) if y is not None]
        ax.plot([k for k, _ in pts], [y for _, y in pts], marker="o", ms=4, lw=1.6,
                color=col, ls=ls, label=lab, zorder=3)
    ax.set_title(name, fontsize=9.5)
    ax.set_xticks(list(x), XLAB, fontsize=8.5)
    ax.set_ylim(0, 1.10)
    ax.set_xlim(-0.4, len(ORDER) - 0.6)
    ax.grid(color=GRID, lw=0.6, zorder=0)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
axes[0].set_ylabel("stable center rank\n/ layer width", fontsize=9)
for ax in axes:
    ax.annotate("backbone", (BOUNDARY - 0.15, 1.05), fontsize=7, color=MUT, ha="right")
    ax.annotate("projector", (BOUNDARY + 0.15, 1.05), fontsize=7, color=MUT, ha="left")
axes[0].legend(fontsize=7.5, frameon=False, loc="lower left", borderaxespad=0.3)
fig.tight_layout(w_pad=1.0)
fig.savefig(OUT)
print(f"[slides] wrote {OUT}")
hdr = " ".join(f"{s:>6s}" for s in XLAB).replace("$z$", "z")
print(f"{'pair':8s} {'arm':6s} {hdr}")
for name, c, t in PAIRS:
    b = BRANCH.get(name, "student")
    for lab, run in (("ctrl", c), ("treat", t)):
        v = profile(run, b)
        print(f"{name:8s} {lab:6s} " + " ".join("     -" if y is None else f"{y:6.2f}" for y in v))
