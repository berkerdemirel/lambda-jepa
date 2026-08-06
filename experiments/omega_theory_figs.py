"""Figures + tables for docs/theory/OMEGA_CONNECTIVITY.md (the touch-law proposal):
F1 touch law M = 1 - c/sqrt(Omega) on the 9 view-mean toy cells (h and z spaces;
deprecated pooled-anatomy cells as a grey stability layer), F2 probe accuracy vs
threshold-normalized Omega_h. Label-free aggregates are reconstructed from the census
class split (balanced 10-class pool: T = .1*p_pos + .9*p_neg; M ~= cross-class median,
90% of pair mass); the natively label-free recomputation is a registered validation."""
import csv
import os
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIGS = os.path.join(ROOT, "results/figures/theory")

VM9 = ["s0.e24va", "s0.e24vb", "s0.e24vc", "s0.e24vac", "s0.e24v4oas", "s0.e24vh0",
       "s0.e24v1", "s0.e24v2lr", "s0.e24v3nq"]
BLUE, GREEN, GREY = "#2a78d6", "#008300", "#b5b4ad"
INK, INK2, SURF = "#0b0b0b", "#52514e", "#fcfcfb"


def load(path):
    with open(os.path.join(ROOT, path)) as fh:
        return list(csv.DictReader(fh))


def spearman(x, y):
    rx = np.argsort(np.argsort(x)).astype(float)
    ry = np.argsort(np.argsort(y)).astype(float)
    rx -= rx.mean(); ry -= ry.mean()
    return float((rx * ry).sum() / np.sqrt((rx**2).sum() * (ry**2).sum()))


cen = defaultdict(dict)
for r in load("results/diag/e24_toy_census.csv"):
    if int(r["class"]) == -1:                       # run-level summary rows
        tag = r["run"].replace("toy.floorssl.", "").replace(".extL", "")
        cen[tag][r["space"]] = {k: float(r[k]) for k in
                                ("omega_diff_med", "p_pos", "p_neg")}
cells = {r["tag"]: r for r in load("results/diag/e24_omega_sort_cells.csv")}
H, Z = "student.h.cls", "student.z.proj.out"


def stats(tags, space, omcol):
    om = np.array([float(cells[t][omcol]) for t in tags])
    M = np.array([cen[t][space]["omega_diff_med"] for t in tags])
    T = np.array([.1 * cen[t][space]["p_pos"] + .9 * cen[t][space]["p_neg"]
                  for t in tags])
    return om, M, T


def fit_c(om, M):
    x = 1.0 / np.sqrt(om)
    c = float(np.sum(x * (1 - M)) / np.sum(x * x))
    r2 = 1 - np.sum((M - (1 - c * x))**2) / np.sum((M - M.mean())**2)
    return c, float(r2)


POOL = [t for t in cells if t in cen and t not in VM9]
acc = {t: float(cells[t]["acc_best"]) for t in cells}

base = dict(lw=0, zorder=3)
plt.rcParams.update({"font.size": 9.5, "text.color": INK, "axes.labelcolor": INK,
                     "xtick.color": INK2, "ytick.color": INK2, "axes.edgecolor": INK2})

# ---------------- F1: the touch law ----------------
fig, ax = plt.subplots(figsize=(5.4, 3.8), dpi=200)
fig.patch.set_facecolor(SURF); ax.set_facecolor(SURF)
oh, Mh, Th = stats(VM9, H, "omega_h_o8_raw")
oz, Mz, Tz = stats(VM9, Z, "omega_z_o8_raw")
ch, r2h = fit_c(oh, Mh)
cz, r2z = fit_c(oz, Mz)
for tags, sp, omc in [(POOL, H, "omega_h_o8_raw"), (POOL, Z, "omega_z_o8_raw")]:
    o_, M_, _ = stats(tags, sp, omc)
    ax.plot(o_, M_, "x", color=GREY, ms=5, mew=1.2, zorder=2,
            label="deprecated-anatomy runs" if sp == H else None)
xx = np.linspace(0.09, 1.35, 300)
ax.plot(xx, 1 - ch / np.sqrt(xx), color=BLUE, lw=1.8, zorder=2)
ax.plot(xx, 1 - cz / np.sqrt(xx), color=GREEN, lw=1.8, ls=(0, (5, 2)), zorder=2)
ax.plot(oh, Mh, "o", color=BLUE, ms=7, label="h space (trunk)", **base)
ax.plot(oz, Mz, "s", color=GREEN, ms=6.5, label="z space (loss)", **base)
ax.axhline(0, color=INK2, lw=0.7, alpha=.5, zorder=1)
thr = ch * ch
ax.axvline(thr, color=INK2, lw=0.9, ls=":", zorder=1)
ax.annotate("median clouds touch\n$\\Omega = c^2 \\approx {:.2f}$".format(thr),
            (thr + .03, -0.62), fontsize=8.5, color=INK2)
ax.annotate(f"$c_h$={ch:.3f}, $R^2$={r2h:.2f}", (.13, .21), color=BLUE, fontsize=8.5)
ax.annotate(f"$c_z$={cz:.3f}, $R^2$={r2z:.2f}", (.13, .10), color=GREEN, fontsize=8.5)
ax.set_xlabel(r"$\Omega$  (orbit energy / center energy)")
ax.set_ylabel("M  (median pair overlap)")
ax.set_title(r"Touch law  $M = 1 - c/\sqrt{\Omega}$  — 9 view-mean runs, both spaces",
             fontsize=10, loc="left")
ax.grid(color=INK2, alpha=.15, lw=.5); ax.set_axisbelow(True)
for s in ("top", "right"):
    ax.spines[s].set_visible(False)
ax.legend(frameon=False, fontsize=8.5, loc="lower right")
fig.tight_layout()
fig.savefig(os.path.join(FIGS, "omega_touchlaw.png"), facecolor=SURF)

# ---------------- F2: accuracy vs normalized Omega ----------------
fig, ax = plt.subplots(figsize=(5.4, 3.6), dpi=200)
fig.patch.set_facecolor(SURF); ax.set_facecolor(SURF)
x9 = oh / thr
y9 = np.array([acc[t] for t in VM9])
ax.axvspan(.5, .7, color=BLUE, alpha=.08, zorder=1)
ax.annotate("operating band\nof the 6 map cells", (.505, .782), fontsize=8.5,
            color=INK2)
ax.axvline(1.0, color=INK2, lw=0.9, ls=":", zorder=1)
ax.annotate("median clouds touch", (1.02, .868), fontsize=8.5, color=INK2)
ax.plot(x9, y9, "o", color=BLUE, ms=8, **base)
for t, lbl in [("s0.e24vc", "vc (best)"), ("s0.e24v3nq", "fresh estimator"),
               ("s0.e24v2lr", "stale ring, half lr"), ("s0.e24v1", "stale ring")]:
    i = VM9.index(t)
    ax.annotate(lbl, (x9[i] + .025, y9[i] - .0035), fontsize=8.5, color=INK2)
ax.set_xlabel(r"$\Omega_h\, /\, c_h^2$   (position relative to the touching threshold)")
ax.set_ylabel("probe accuracy (best)")
ax.set_title(r"Accuracy vs threshold-normalized $\Omega_h$ — 9 view-mean runs",
             fontsize=10, loc="left")
ax.grid(color=INK2, alpha=.15, lw=.5); ax.set_axisbelow(True)
for s in ("top", "right"):
    ax.spines[s].set_visible(False)
fig.tight_layout()
fig.savefig(os.path.join(FIGS, "omega_band.png"), facecolor=SURF)

# ---------------- doc tables ----------------
print("law fits (n9): h c=%.3f R2=%.3f | z c=%.3f R2=%.3f | threshold c_h^2=%.3f"
      % (ch, r2h, cz, r2z, thr))
print("rho(acc, Omega)=%+.3f  rho(acc,T)=%+.3f  rho(acc,M)=%+.3f"
      % (spearman(oh, y9), spearman(Th, y9), spearman(Mh, y9)))
print("\nrun            acc     Om_h    Om/c2   T       M       Om_z")
for i, t in enumerate(sorted(VM9, key=lambda t: -acc[t])):
    j = VM9.index(t)
    print("%-14s %.4f  %.3f   %.2f    %.3f   %+.3f  %.3f"
          % (t, acc[t], oh[j], x9[j], Th[j], Mh[j], oz[j]))
