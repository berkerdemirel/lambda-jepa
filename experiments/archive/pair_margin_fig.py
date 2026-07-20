"""Dumbbell figure for the pair-margin verify pass (Berker 2026-07-08: figures over prose).
Per method x {h, z}: segment from mean random-pair cosine to mean positive-pair cosine —
segment length IS the invariance margin. Trained run vs random-init null overlaid; the M1
question ("is h really more view-invariant than z?") is answered by comparing segment lengths,
not endpoint positions (endpoints move with compactness).

  python experiments/pair_margin_fig.py   # reads results/M1/pair_margin.csv
"""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from pair_margin_m1 import H_SPACE, NULL_ID, RES, RUN_ID, Z_FINAL

ORDER = ["simclr", "byol", "vicreg", "dino", "mae", "ijepa", "lejepa"]  # matrix order
BLUE, GRAY, INK, MUT = "#2a78d6", "#8a8a85", "#333333", "#767471"


def _row(df, rid, space):
    r = df[(df["run_id"] == rid) & (df["stack"] == "audit_v1") & (df["space"] == space)]
    return r.iloc[0] if len(r) else None


def _dumbbell(ax, y, row, color, solid, label_margin):
    ax.plot([row.rand_cos, row.pos_cos], [y, y], color=color, lw=2 if solid else 1.4,
            ls="-" if solid else (0, (4, 2)), zorder=2, solid_capstyle="round")
    ax.plot(row.rand_cos, y, "o", ms=6, mfc="white", mec=color, mew=1.4, zorder=3)
    ax.plot(row.pos_cos, y, "o", ms=7, mfc=color, mec=color, zorder=3)
    if label_margin:
        ax.annotate(f"+{row.cos_margin:.2f}", (row.pos_cos, y), xytext=(6, -2.5),
                    textcoords="offset points", fontsize=8, color=INK)


def main():
    df = pd.read_csv(os.path.join(RES, "M1", "pair_margin.csv"))
    fig, axes = plt.subplots(1, 3, figsize=(14, 5.2), sharey=True,
                             gridspec_kw={"width_ratios": [1, 1, 0.85]})
    for ax, side, title in ((axes[0], "h", "h — probed representation"),
                            (axes[1], "z", "z — loss space (z.final)")):
        for i, m in enumerate(ORDER):
            y = len(ORDER) - 1 - i
            space = H_SPACE[m] if side == "h" else Z_FINAL[m]
            if space is None:
                ax.text(0.5, y, "— (pixel loss)", ha="center", va="center",
                        fontsize=8, color=MUT)
                continue
            tr = _row(df, RUN_ID[m], space)
            nu = _row(df, NULL_ID[m], space)
            if nu is not None:
                _dumbbell(ax, y - 0.18, nu, GRAY, solid=False, label_margin=False)
            if tr is not None:
                _dumbbell(ax, y + 0.18, tr, BLUE, solid=True, label_margin=True)
        ax.set_title(title, fontsize=11, color=INK, loc="left")
        ax.set_xlim(-0.05, 1.12)
        ax.set_xlabel("mean cosine similarity", fontsize=9, color=MUT)
        ax.grid(axis="x", color="#e6e4dd", lw=0.7, zorder=0)
        for s in ("top", "right", "left"):
            ax.spines[s].set_visible(False)
        ax.tick_params(axis="both", labelsize=9, colors=INK, length=0)
    # panel 3 — the derived quantity itself: within-network margin (pos − rand), h vs z,
    # trained and untrained. This is THE comparison; panels 1–2 are its evidence.
    ax = axes[2]
    for i, m in enumerate(ORDER):
        y = len(ORDER) - 1 - i
        for rid, color, solid in ((RUN_ID[m], BLUE, True), (NULL_ID[m], GRAY, False)):
            yy = y + (0.18 if solid else -0.18)
            rh = _row(df, rid, H_SPACE[m])
            rz = _row(df, rid, Z_FINAL[m]) if Z_FINAL[m] else None
            if rh is not None and rz is not None:
                ax.plot([rh.cos_margin, rz.cos_margin], [yy, yy], color=color,
                        lw=2 if solid else 1.4, ls="-" if solid else (0, (4, 2)), zorder=2)
            if rh is not None:
                ax.plot(rh.cos_margin, yy, "o", ms=6.5, mfc="white", mec=color, mew=1.6, zorder=3)
            if rz is not None:
                ax.plot(rz.cos_margin, yy, "D", ms=6, mfc=color, mec=color, zorder=3)
    ax.set_title("margin only: h ○ vs z ◆", fontsize=11, color=INK, loc="left")
    ax.set_xlim(-0.06, 0.9)
    ax.set_xlabel("cos margin (pos − rand)", fontsize=9, color=MUT)
    ax.grid(axis="x", color="#e6e4dd", lw=0.7, zorder=0)
    ax.axvline(0, color="#c9c7bf", lw=0.9, zorder=1)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.tick_params(axis="both", labelsize=9, colors=INK, length=0)

    labels = [m + " (h=z.embed)" if m == "lejepa" else m for m in ORDER]
    axes[0].set_yticks(range(len(ORDER))[::-1], labels)
    hnd = [plt.Line2D([], [], color=BLUE, lw=2, marker="o", ms=6, mfc=BLUE, label="trained net"),
           plt.Line2D([], [], color=GRAY, lw=1.4, ls=(0, (4, 2)), marker="o", ms=6,
                      mfc="white", label="untrained net (random init)"),
           plt.Line2D([], [], color=INK, lw=0, marker="o", ms=6, mfc="white", mec=INK,
                      label="margin at h"),
           plt.Line2D([], [], color=INK, lw=0, marker="D", ms=5.5, mfc=INK,
                      label="margin at z")]
    fig.legend(handles=hnd, loc="upper right", ncols=4, fontsize=8.5, frameon=False,
               bbox_to_anchor=(0.995, 0.985))
    fig.suptitle("Within-network view-invariance: positive-pair vs random-pair cosine, "
                 "per space", fontsize=12.5, color=INK, x=0.02, ha="left")
    fig.text(0.02, 0.015, "each segment (panels 1–2) = ONE network, ONE space: open dot = mean "
             "random-pair cos, filled dot = mean positive-pair cos, length = the invariance "
             "margin · panel 3 plots those margins directly (h ○ → z ◆) · audit_v1 stack, "
             "N=9469 · source: results/M1/pair_margin.csv", fontsize=7.5, color=MUT)
    fig.tight_layout(rect=(0, 0.035, 1, 0.94))
    out = os.path.join(RES, "figures", "geometry", "pair_margin.png")
    fig.savefig(out, dpi=200, facecolor="white")
    print(f"[pair_margin_fig] wrote {out}")


if __name__ == "__main__":
    main()
