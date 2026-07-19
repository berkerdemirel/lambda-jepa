"""E18 deeper analysis, leg 2 — the inv-vs-val PHASE read made durable (E18-T1's candidate
mechanism said "h-CF-shape terms impede inv" but also "insufficient alone: at inv-matched
epochs val still favors e20f" — this instrument pins both halves on one figure): per cell
(sigreg_t · sigreg3 · e20f · f2 · control), online-probe acc (y) vs train/inv EMA (x) with
epoch markers; plus val interpolated at matched inv levels. Monitor caveat E12-T9 rides (the
online probe, not offline v2, is the y-axis — trajectory shape is the read, not absolutes).
Figure: results/figures/e18/e18_inv_phase.png; matched-inv table -> stdout. RAW; no takeaway.

  python experiments/e18_inv_phase.py
"""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import wandb

ROOT = "/nfs/scistore19/locatgrp/bdemirel/ssl_project"
RUNS = {"in100.lejepa.s0.hpull_sigreg_t": ("#c04f4f", "sigreg_t (t_8.2 CF)"),
        "in100.lejepa.s0.hpull_sigreg3": ("#e07b39", "sigreg3 (gauss CF, 3%)"),
        "in100.lejepa.s0.hpull_sigreg": ("#8c6d31", "sigreg (gauss CF, 10%)"),
        "in100.lejepa.s0.e20f": ("#3d65d0", "e20f (floor, calibrated)"),
        "in100.lejepa.s0.e12f2": ("#4a9d9d", "f2 (floor, .02)"),
        "in100.lejepa.s0": ("#9d9d9d", "control")}
EPM = [10, 25, 50, 75, 100]
INV_LEVELS = [0.10, 0.08, 0.06, 0.05]


def ema(x, a=0.9):
    out, m = np.empty_like(x), x[0]
    for i, v in enumerate(x):
        m = a * m + (1 - a) * v
        out[i] = m
    return out


def run_of(api, name):
    found = None
    for r in api.runs("causal-learning-ai-ista/sslgap", filters={"displayName": name}):
        if found is None or r.lastHistoryStep > found.lastHistoryStep:
            found = r
    return found


def main():
    api = wandb.Api(timeout=60)
    fig, ax = plt.subplots(figsize=(7.5, 5.5), facecolor="white")
    print(f"{'cell':26s} " + " ".join(f"acc@inv={l:.2f}" for l in INV_LEVELS))
    for name, (col, label) in RUNS.items():
        r = run_of(api, name)
        if r is None:
            print(f"{name}: MISSING"); continue
        h = r.history(samples=2000, pandas=True)
        t = r.history(samples=300, keys=["test/acc", "test/epoch"], pandas=True)
        if "train/inv" not in h.columns or not len(t):
            print(f"{name}: no inv/test keys"); continue
        d = h.dropna(subset=["train/inv"])
        s, v = d["_step"].to_numpy(float), ema(d["train/inv"].to_numpy(float))
        spe = float(t["_step"].max()) / (float(t["test/epoch"].max()) + 1)
        te, ta = (t["test/epoch"].to_numpy(float) + 1), t["test/acc"].to_numpy(float)
        inv_at_ep = np.interp(te * spe, s, v)
        ax.plot(inv_at_ep, 100 * ta, "-", color=col, lw=1.6, label=label)
        for e in EPM:
            i = np.argmin(np.abs(te - e))
            if abs(te[i] - e) < 1.5:
                ax.plot(inv_at_ep[i], 100 * ta[i], "o", color=col, ms=4)
                if name in ("in100.lejepa.s0.hpull_sigreg_t", "in100.lejepa.s0.e20f"):
                    ax.annotate(f"e{e}", (inv_at_ep[i], 100 * ta[i]), fontsize=6.5,
                                textcoords="offset points", xytext=(3, 3), color=col)
        # val at matched inv (only readable where inv passes through the level going down)
        cells = []
        for lv in INV_LEVELS:
            k = np.where(inv_at_ep <= lv)[0]
            cells.append(f"{100 * ta[k[0]]:9.1f}" if len(k) else f"{'—':>9s}")
        print(f"{label:26s} " + " ".join(cells))
    ax.invert_xaxis()
    ax.set_xlabel("train/inv (EMA; loss-space alignment residual) — training moves right→left")
    ax.set_ylabel("online probe acc (%)  [E12-T9 caveat]")
    ax.set_title("E18: val vs inv phase plot — same inv, different val = the non-inv channel",
                 fontsize=10)
    ax.grid(alpha=0.15)
    ax.legend(fontsize=7.5)
    os.makedirs(f"{ROOT}/results/figures/e18", exist_ok=True)
    fig.tight_layout()
    fig.savefig(f"{ROOT}/results/figures/e18/e18_inv_phase.png", dpi=160)
    print("wrote results/figures/e18/e18_inv_phase.png", flush=True)


if __name__ == "__main__":
    main()
