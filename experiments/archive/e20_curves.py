"""E20 mid-training/landing reader: for each zoo lane, the arm's h_moment_kl trajectory
(divergence-window protocol from the E19 ladder: bottom, divergence point if any, ep marks)
+ online-probe delta vs the lane's own control at matched epochs (monitor caveat E12-T9
standing — offline v2 probes remain the arbiter at landing). Figure:
results/figures/e20/e20_curves.png. RAW; no takeaway.

  python experiments/e20_curves.py
"""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import wandb

ROOT = "/nfs/scistore19/locatgrp/bdemirel/ssl_project"
SPE = {"simclr": 495, "byol": 495, "vicreg": 495, "mae": 495, "ijepa": 495,
       "dino": 990, "lejepa": 990}
LAM = {"simclr": 0.098, "byol": 0.0205, "vicreg": 1.548, "dino": 0.258,
       "lejepa": 0.0146, "mae": 0.0098, "ijepa": 0.0025}
COLS = dict(zip(SPE, ["#3a8c5c", "#8c6d31", "#e07b39", "#8a5cb8", "#3d65d0", "#c04f4f",
                      "#4a9d9d"]))
EPM = [2, 5, 10, 25, 50, 75, 100]


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


def hist(api, name, keys, n=2000):
    found = run_of(api, name)
    if found is None:
        return None
    h = found.history(samples=n, pandas=True)
    return h[[c for c in ["_step", *keys] if c in h.columns]]


def test_hist(api, name):
    # test rows are 1/epoch — a keyed query keeps them from being sampled out (e19 lesson)
    found = run_of(api, name)
    if found is None:
        return None
    return found.history(samples=300, keys=["test/acc", "test/epoch"], pandas=True)


def main():
    api = wandb.Api(timeout=60)
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(13.5, 5.0))
    print(f"{'lane':8s} {'lam':>7s} {'ep':>5s} {'h_kl bottom@ep':>15s} {'now':>5s} "
          f"{'div?':>5s}  probe arm-vs-ctrl at matched ep")
    for m, spe in SPE.items():
        arm = hist(api, f"in100.{m}.s0.e20f", ["train/h_moment_kl"])
        pa = test_hist(api, f"in100.{m}.s0.e20f")
        pc = test_hist(api, f"in100.{m}.s0")
        if arm is None or "train/h_moment_kl" not in arm.columns:
            print(f"{m:8s} MISSING")
            continue
        d = arm.dropna(subset=["train/h_moment_kl"])
        s, v = d["_step"].to_numpy(float), d["train/h_moment_kl"].to_numpy(float)
        ep, sm = s / spe, ema(v)
        ib = int(sm.argmin())
        # ACTIVE divergence only (the arm-4 signature): current value sustained above the
        # running minimum — early recovered transients don't flag.
        div = float(ep[-1]) if (sm[-5:] > sm.min() + 0.15).all() else None
        a1.plot(ep, sm, color=COLS[m], lw=1.7, label=f"{m} (λ={LAM[m]})")
        deltas = []
        if pa is not None and pc is not None and len(pa) and len(pc):
            ae = dict(zip((pa["test/epoch"] + 1).astype(int), pa["test/acc"]))
            ce = dict(zip((pc["test/epoch"] + 1).astype(int), pc["test/acc"]))
            for e in EPM:
                if e in ae and e in ce:
                    deltas.append((e, 100 * (ae[e] - ce[e])))
            if deltas:
                a2.plot([e for e, _ in deltas], [x for _, x in deltas], "o-", color=COLS[m],
                        lw=1.5, ms=5)
        dv = f"{div:.2f}" if div else "—"
        dl = " ".join(f"e{e}:{x:+.1f}" for e, x in deltas)
        print(f"{m:8s} {LAM[m]:7.4f} {ep[-1]:5.1f} {sm[ib]:9.2f}@{ep[ib]:5.1f} {sm[-1]:5.2f} "
              f"{dv:>5s}  {dl}")
    a1.set_xlabel("epoch")
    a1.set_ylabel("train/h_moment_kl (EMA .9)")
    a1.set_title("E20: the calibrated floor at h, all seven lanes", fontsize=10)
    a1.legend(fontsize=7.5)
    a2.axhline(0, color="#444444", lw=0.8)
    a2.set_xlabel("epoch")
    a2.set_ylabel("online probe: arm − control (pts)")
    a2.set_title("monitor delta at matched epochs (E12-T9 caveat; offline v2 = arbiter)",
                 fontsize=10)
    for ax in (a1, a2):
        ax.grid(alpha=0.15)
    os.makedirs(f"{ROOT}/results/figures/e20", exist_ok=True)
    fig.tight_layout()
    fig.savefig(f"{ROOT}/results/figures/e20/e20_curves.png", dpi=160)
    print("wrote results/figures/e20/e20_curves.png", flush=True)


if __name__ == "__main__":
    main()
