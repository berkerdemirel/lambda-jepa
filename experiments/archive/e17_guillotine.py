"""Guillotine curves for the PREV cells (Berker 2026-07-21: "where're the guillotine
curves … for the prev cells") — the E17 h-pull program roster, whose stores carry the full
L-tap grid. Probe accuracy vs depth station per method family: ctrl (e17c) vs the own-term
arms; stations L03 · L06 · L09 (trunk, at the cls readout) · gap · cls · [embed] · tap1 ·
tap2 · z-out. 2 rows (lin_v2 / knn200) × 5 families. Declared h in E17 = student cls for
every method (dino included — gradients flow through the student). E02 remains the
pre-registered full experiment; this renders the existing record. RAW, no takeaway."""
import csv
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = "/nfs/scistore19/locatgrp/bdemirel/ssl_project"
FAMILIES = [
    ("lejepa", [("ctrl", "in100.lejepa.s0.e17c.ext", "#8c8c8c"),
                ("sigreg", "in100.lejepa.s0.hpull_sigreg.ext", "#2e8b57"),
                ("sigreg_inv", "in100.lejepa.s0.hpull_sigreg_inv.ext", "#1a5c38"),
                ("sigreg_t (E18)", "in100.lejepa.s0.hpull_sigreg_t.ext", "#d4820a"),
                ("sigreg3 (E18)", "in100.lejepa.s0.hpull_sigreg3.ext", "#b8608a")]),
    ("simclr", [("ctrl", "in100.simclr.s0.e17c.ext", "#8c8c8c"),
                ("uniform", "in100.simclr.s0.hpull_uniform.ext", "#3d65d0"),
                ("uniform_align", "in100.simclr.s0.hpull_uniform_align.ext", "#1b2c60")]),
    ("byol", [("ctrl", "in100.byol.s0.e17c.ext", "#8c8c8c"),
              ("align", "in100.byol.s0.hpull_align.ext", "#6a3fb5")]),
    ("dino", [("ctrl", "in100.dino.s0.e17c.ext", "#8c8c8c"),
              ("protoce", "in100.dino.s0.hpull_protoce.ext", "#0f7b8a")]),
    ("vicreg", [("ctrl", "in100.vicreg.s0.e17c.ext", "#8c8c8c"),
                ("varcov@6x (c015)", "in100.vicreg.s0.hpull_varcov.c015.ext", "#c23b3b")]),
]
STATIONS = ["L03", "L06", "L09", "gap", "cls", "embed", "tap1", "tap2", "z.out"]


def probes(run):
    out = {}
    for path in (f"{ROOT}/results/probes/{run}.csv",
                 f"{ROOT}/results/probes/{run.replace('.ext', '.extL')}.csv"):
        if os.path.exists(path):
            for r in csv.DictReader(open(path)):
                out[(r["space"], r["probe"])] = float(r["val_acc"])
    return out


def station_of(space, branch):
    """Map a stored space name to its depth station (None = not on this run's path)."""
    s = space
    if not s.startswith(branch):
        return None
    for k in ("L03", "L06", "L09"):
        if s.endswith(f".{k}"):
            return k
    if s.endswith(".h.gap"):
        return "gap"
    if s.endswith(".h.cls"):
        return "cls"
    if s.endswith(".z.embed"):
        return "embed"
    if s.endswith("tap1"):
        return "tap1"
    if s.endswith("tap2"):
        return "tap2"
    if s.endswith(".z.proj.out") or s.endswith(".bottleneck"):
        return "z.out"
    return None


fig, axes = plt.subplots(2, 5, figsize=(19, 7), facecolor="white")
for col, (fam, members) in enumerate(FAMILIES):
    for row, probe in ((0, "linear_raw_v2"), (1, "knn_v1_k200")):
        ax = axes[row, col]
        for lab, run, c in members:
            P = probes(run)
            pts = {}
            for (space, pr), v in P.items():
                if pr != probe:
                    continue
                st = station_of(space, "student")
                if st:
                    pts[st] = v
            xs = [i for i, st in enumerate(STATIONS) if st in pts]
            ys = [pts[STATIONS[i]] for i in xs]
            if xs:
                ax.plot(xs, ys, "-o", color=c, ms=3.5, lw=1.2,
                        label=lab if row == 0 else None)
        ax.set_xticks(range(len(STATIONS)))
        ax.set_xticklabels(STATIONS, fontsize=5.8, rotation=45)
        ax.axvline(4.5, color="#bbbbbb", lw=0.8, ls=":")
        ax.set_title(f"{fam} — {'lin_v2' if row == 0 else 'knn200'}", fontsize=8.5)
        ax.tick_params(length=0, labelsize=6.5)
        ax.margins(y=0.12)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        if row == 0:
            ax.legend(fontsize=5.4, frameon=False, loc="lower right")
fig.suptitle("E17-program guillotine (prev cells): converged v2 probes at every stored depth station — L03/06/09 = trunk cls "
             "readouts · dotted line = trunk|head boundary · ctrl (grey) vs own-term arms per family — RAW",
             fontsize=10, y=0.995)
fig.tight_layout(rect=(0, 0.02, 1, 0.955))
fig.text(0.01, 0.005, "stations categorical; lejepa carries the extra embed station; dino head = bottleneck path; "
         "E17 declared h = student cls all methods; probes = converged v2 (D-020); E02 remains the pre-registered full experiment",
         fontsize=6, color="#555555")
out = f"{ROOT}/results/figures/e17/e17_guillotine.png"
os.makedirs(os.path.dirname(out), exist_ok=True)
fig.savefig(out, dpi=160)
print("wrote", out)
