"""E19 divergence-diagnostic ladder reader (Berker /goal 2026-07-17c): pull the four embdiag
cells + the two failure comparators (arm 5 cal trace, arm 4 live) + the f2 hold template from
wandb; epoch-align h_moment_kl; per run report bottom (value/step/ep), the DIVERGENCE POINT
(first sustained rise ≥ .15 above the running bottom on the EMA-smoothed curve), values at
ep marks, and end slope. Figure: results/figures/e19/e19_diag_curves.png. Raw numbers only —
the hold/re-inflate verdict is read off the printed table in-conversation.

  python experiments/e19_diag_curves.py pull | figs | both
"""
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = "/nfs/scistore19/locatgrp/bdemirel/ssl_project"
CACHE = ("/tmp/claude-1104964/-nfs-scistore19-locatgrp-bdemirel-ssl-project/"
         "5b63361b-7953-4e98-8ab6-a7a1b463869f/scratchpad/diag_cache")
BLUE, ORANGE, GREEN, RED, GRAY, PURPLE, BROWN = ("#3d65d0", "#e07b39", "#3a8c5c", "#c04f4f",
                                                 "#8a8a8a", "#8a5cb8", "#8c6d31")
RUNS = {  # name -> (color, dash, label, steps/ep fallback)
    "in100.lejepa.s0.e12f2": (BLUE, "-", "f2 (HOLD template; lejepa lane, ~990 st/ep)", 990),
    "in100.vicreg.s0.floorssl_hz_emb_cal": (RED, "-", "arm 5 cal (FAILURE trace, killed ep3)", 495),
    "in100.vicreg.s0.floorssl_hz_emb": (RED, ":", "arm 4 uncal (live twin)", 495),
    "in100.vicreg.s0.embdiag_lr3": (GREEN, "-", "lr3: lr 1e-3->3e-4 (C5)", 495),
    "in100.vicreg.s0.embdiag_d128": (ORANGE, "-", "d128: expander out 2048->128 (C3)", 495),
    "in100.vicreg.s0.embdiag_spec": (PURPLE, "-", "spec: spectral-norm expander (C4)", 495),
    "in100.vicreg.s0.embdiag_wd0": (BROWN, "-", "wd0: emb Linear wd=0 (C2)", 495),
    "in100.vicreg.s0.embdiag_lam5x": (GREEN, "--", "lam5x: h_lamb .02->.1 (C1 dose)", 495),
    "in100.vicreg.s0.embdiag_lam50x": (ORANGE, "--", "lam50x: h_lamb .02->1.0 (C1 dose ~f2 share)", 495)}
EP_MARKS = [0.5, 1, 2, 3, 5, 8, 12, 16, 20]


def pull():
    import wandb
    os.makedirs(CACHE, exist_ok=True)
    api = wandb.Api(timeout=60)
    found = {}
    for r in api.runs("causal-learning-ai-ista/sslgap",
                      filters={"displayName": {"$in": list(RUNS)}}):
        if r.name not in found or r.lastHistoryStep > found[r.name].lastHistoryStep:
            found[r.name] = r
    for name in RUNS:
        r = found.get(name)
        if r is None:
            print(f"[pull] MISSING {name}", flush=True)
            continue
        hist = r.history(samples=4000, pandas=True)
        cols = [c for c in ["_step", "train/h_moment_kl", "train/grad_norm"] if c in hist.columns]
        hist[cols].to_csv(f"{CACHE}/{name}.csv", index=False)
        te = r.history(samples=200, keys=["test/acc", "test/epoch"], pandas=True)
        te.to_csv(f"{CACHE}/{name}.test.csv", index=False)
        print(f"[pull] {name}: laststep={r.lastHistoryStep} rows={len(hist)}", flush=True)


def ema(x, a=0.9):
    out, m = np.empty_like(x), x[0]
    for i, v in enumerate(x):
        m = a * m + (1 - a) * v
        out[i] = m
    return out


def load(name):
    import pandas as pd
    df = pd.read_csv(f"{CACHE}/{name}.csv").dropna(subset=["train/h_moment_kl"])
    spe = RUNS[name][3]
    try:
        te = pd.read_csv(f"{CACHE}/{name}.test.csv")
        if len(te) >= 2:
            spe = float(np.polyfit(te["test/epoch"] + 1, te["_step"], 1)[0])
    except Exception:
        pass
    s, v = df["_step"].to_numpy(float), df["train/h_moment_kl"].to_numpy(float)
    return s, s / spe, v, ema(v)


def stats(name):
    s, ep, v, sm = load(name)
    ib = int(np.argmin(sm))
    bot, bstep, bep = float(sm[ib]), float(s[ib]), float(ep[ib])
    div = None                      # first sustained (>=5 consecutive samples) rise above bottom+.15
    run_min = np.minimum.accumulate(sm)
    above = sm > run_min + 0.15
    for i in range(len(sm) - 4):
        if above[i:i + 5].all():
            div = (float(s[i]), float(ep[i]), float(run_min[i]))
            break
    marks = {m: float(sm[np.searchsorted(ep, m)]) for m in EP_MARKS if ep[-1] >= m}
    tail = sm[-max(5, len(sm) // 20):]
    return dict(name=name, last_ep=float(ep[-1]), bottom=bot, bottom_step=bstep, bottom_ep=bep,
                div=div, marks=marks, end=float(sm[-1]),
                slope=float((tail[-1] - tail[0]) / max(len(tail) - 1, 1)))


def figs():
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(13.5, 5.0))
    table = []
    for name, (col, dash, label, _) in RUNS.items():
        if not os.path.exists(f"{CACHE}/{name}.csv"):
            continue
        s, ep, v, sm = load(name)
        for ax, xmax in ((a1, None), (a2, 3.2)):
            m = np.ones_like(ep, bool) if xmax is None else ep <= xmax
            if m.any():
                ax.plot(ep[m], sm[m], color=col, ls=dash, lw=1.8, label=label if ax is a1 else None)
        st = stats(name)
        table.append(st)
        if st["div"]:
            a2.plot(st["div"][1], sm[np.searchsorted(s, st["div"][0])], "v", color=col, ms=7)
    for ax, t in ((a1, "h_moment_kl (EMA .9), epoch-aligned — full horizon"),
                  (a2, "first 3 epochs — the divergence window (▼ = divergence point)")):
        ax.set_xlabel("epoch")
        ax.set_ylabel("train/h_moment_kl")
        ax.set_title(t, fontsize=10)
        ax.grid(alpha=0.15)
    a1.legend(fontsize=7.5, loc="upper right")
    fig.tight_layout()
    out = f"{ROOT}/results/figures/e19/e19_diag_curves.png"
    fig.savefig(out, dpi=160)
    print(f"wrote {out}", flush=True)
    hdr = f"{'run':38s} {'lastep':>6s} {'bottom':>7s} {'@ep':>5s} {'diverge@ep':>10s} " \
          f"{'end':>6s} {'slope':>8s}  marks"
    print(hdr)
    for st in table:
        dv = f"{st['div'][1]:.2f}" if st["div"] else "—"
        mk = " ".join(f"e{m}:{v:.2f}" for m, v in st["marks"].items())
        print(f"{st['name'][12:]:38s} {st['last_ep']:6.1f} {st['bottom']:7.3f} "
              f"{st['bottom_ep']:5.2f} {dv:>10s} {st['end']:6.2f} {st['slope']:+8.4f}  {mk}",
              flush=True)


if __name__ == "__main__":
    arg = sys.argv[1] if len(sys.argv) > 1 else "both"
    if arg in ("pull", "both"):
        pull()
    if arg in ("figs", "both"):
        figs()
