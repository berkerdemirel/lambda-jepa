"""E20f battery-metric figure (Berker 2026-07-20: "create the metric figures for both e20f and
e21, separately comparing the controls"; pair panels per his same-day correction: alignment ≡
2−2·cos_invariance on normalized features — one number rendered twice — so the pair block shows
POSITIVE-pair cos and RANDOM-pair cos separately instead, from results/diag/e2x_posneg.csv).
The zoo floor arm (E20-T3) vs its own control per lane, on the desiderata the floor targets:
isotropy (effective_rank / kurt_topeig.worst), uniformity, gaussianity (gauss_kl_full / EP —
EP shown beside kurt/KL, never alone, house rule), and the audit_v1-frame pair cosines. View
lanes carry the floor's aug-coupled story; mae/ijepa are the aug-less negative-space pair.
Reads results/compare/e20_battery_vs_ctrl.csv (audit-frame regen 2026-07-20) ->
results/figures/e20/e20f_metric_panel.png. RAW, no takeaway."""
import csv

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = "/nfs/scistore19/locatgrp/bdemirel/ssl_project"
ORDER = ["lejepa", "byol", "simclr", "vicreg", "dino", "ijepa", "mae"]  # view lanes | aug-less
N_VIEW = 5
PANELS = [  # (metric key, source, title, sub-label, log-y)
    ("effective_rank|raw|full", "h", "effective_rank", "exp spectral entropy (isotropy → d)", False),
    ("kurt_topeig.worst|raw|full", "h", "kurt_topeig.worst", "max |excess kurt| top-10 eigdirs (gauss = 0)", False),
    ("gauss_kl_full.total|raw|full", "h", "gauss_kl_full.total", "moment-KL to N(0,I), per dim", False),
    ("epps_pulley|raw|full", "h", "epps_pulley", "sliced EP, standardized (log y)", True),
    ("uniformity|raw|full", "h", "uniformity", "Wang–Isola on sphere (lower = more spread)", False),
    ("pos_cos", "posneg", "pair pos-cos", "mean cos of positive pairs (view invariance)", False),
    ("rand_cos", "posneg", "pair rand-cos", "mean cos of cross-view different-image pairs (cone)", False),
]

rows = list(csv.DictReader(open(f"{ROOT}/results/compare/e20_battery_vs_ctrl.csv")))
val = {(r["lane"], r["space_role"], r["metric"]): (float(r["arm"]), float(r["ctrl"]))
       for r in rows if r["arm"] and r["ctrl"]}
space_of = {r["lane"]: r["space"] for r in rows if r["space_role"] == "h"}
pn = {(r["run"], r["space"]): r for r in csv.DictReader(open(f"{ROOT}/results/diag/e2x_posneg.csv"))}


def get(lane, key, src):
    if src == "h":
        return val.get((lane, "h", key))
    arm = pn.get((f"in100.{lane}.s0.e20f.ext", space_of[lane]))
    ctl = pn.get((f"in100.{lane}.s0.ext", space_of[lane]))
    return None if arm is None or ctl is None else (float(arm[key]), float(ctl[key]))


fig, axes = plt.subplots(2, 4, figsize=(15.5, 6.4), facecolor="white")
axes = axes.ravel()
for ax, (met, src, title, sub, logy) in zip(axes, PANELS):
    for li, lane in enumerate(ORDER):
        got = get(lane, met, src)
        if got is None:
            continue
        a, c = got
        ax.bar(li - 0.21, c, width=0.4, color="#8c8c8c")
        ax.bar(li + 0.21, a, width=0.4, color="#3d65d0")
        lo, hi = min(a, c), max(a, c)
        if not logy or lo > 0:  # hatched delta segment on the arm bar
            ax.bar(li + 0.21, hi - lo, bottom=lo, width=0.4, fill=False,
                   hatch="///", edgecolor="#1b2c60", linewidth=0)
        for x, v in ((li - 0.21, c), (li + 0.21, a)):
            ax.text(x, v, f"{v:.3g}", ha="center",
                    va="bottom" if v >= 0 else "top", fontsize=5.6, rotation=90)
    if logy:
        ax.set_yscale("log")
    ax.axvline(N_VIEW - 0.5, color="#bbbbbb", lw=0.8, ls=":")
    ax.set_xticks(range(len(ORDER)))
    ax.set_xticklabels(ORDER, fontsize=7.5, rotation=25)
    ax.set_title(f"{title}\n{sub}", fontsize=8.5)
    ax.margins(y=0.30)
    ax.tick_params(length=0, labelsize=7)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.axhline(0, color="#666666", lw=0.6)
ax = axes[-1]
ax.axis("off")
ax.legend(handles=[plt.Rectangle((0, 0), 1, 1, color="#8c8c8c"),
                   plt.Rectangle((0, 0), 1, 1, color="#3d65d0"),
                   plt.Rectangle((0, 0), 1, 1, fill=False, hatch="///", edgecolor="#1b2c60")],
          labels=["control", "e20f floor arm", "arm−ctrl gap"], loc="upper left", fontsize=8.5,
          frameon=False)
ax.text(0, 0.52, "left of dotted divider: view lanes (aug pipeline in the loss)\n"
        "right: aug-less negative-space pair (mae/ijepa)\n\n"
        "declared h per lane:\n"
        + "\n".join(f"  {l}: {space_of.get(l, '?')}" for l in ORDER)
        + "\n\npairs = fixed audit_v1 stack; rand pair = cross-view\n"
        "different-image pair (same pipeline, identity differs)\n"
        "EP beside kurt/KL per house rule (sliced stats foolable)",
        fontsize=7, va="top", family="monospace", transform=ax.transAxes)
fig.suptitle("E20f: battery metrics at declared h — floor arm vs control per lane (RAW)",
             fontsize=11, y=0.995)
fig.tight_layout(rect=(0, 0, 1, 0.97))
out = f"{ROOT}/results/figures/e20/e20f_metric_panel.png"
fig.savefig(out, dpi=160)
print("wrote", out)
