"""E21/E20-fix mid-training reader — the four 2026-07-19 cells against their NAMED comparators
(HANDOVER priority 1; divergence-window protocol: bottom / divergence point / ep marks, FULL
curves never heads — house incident rule):
  hz cells   in100.floorssl.s0.lejepa_augs (PRIMARY) + in100.floorssl.s0.nobn
             vs the f2 / lejepa-e20f h_moment_kl hold template (+ own z-floor & inv terms;
             the byol-pair v2 fork's ep6 stub dotted for lineage); grad_norm envelope = the
             nobn no-BN-pin watch (K1 standing).
  dino cells in100.dino.s0.e20f_lam06 / _lam008 vs gd (lam=.02) and e20f (lam=.258) at
             matched epochs + monitor deltas vs the dino control.
Figure: results/figures/e21/e21_curves.png. RAW; no takeaway.

  python experiments/e21_curves.py
"""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import wandb

ROOT = "/nfs/scistore19/locatgrp/bdemirel/ssl_project"
SPE_FALLBACK = {"in100.floorssl.s0.d16": 990, "in100.floorssl.s0.d32": 990,
                "in100.floorssl.s0.d64": 990, "in100.floorssl.s0.d128": 990,
                "in100.floorssl.s0.d256": 990, "in100.floorssl.s0.d512": 990,
                "in100.floorssl.s0.laug_eps": 990, "in100.floorssl.s0.laug_hinge": 990,
                "in100.floorssl.s0.lejepa_augs": 390, "in100.floorssl.s0.lejepa_augs2": 390,
                "in100.vicreg.s0.floorssl_hz_v2": 390, "in100.dino.s0.e20f_lam06": 990,
                "in100.dino.s0.e20f_lam008": 990, "in100.dino.s0.e12gd": 990,
                "in100.dino.s0.e20f": 990, "in100.lejepa.s0.e12f2": 990,
                "in100.lejepa.s0.e20f": 990, "in100.dino.s0": 990}
EPM = [2, 5, 10, 25, 50, 75, 100]
# fix-session live cells: the D-050 dim bracket (+ the D-049 hinge diagnostic until its
# fork datum) vs the collapsed refs
LIVE = [("in100.floorssl.s0.d16", "#7bb3d9"), ("in100.floorssl.s0.d32", "#3d65d0"),
        ("in100.floorssl.s0.d64", "#2e4a9e"), ("in100.floorssl.s0.d128", "#1a2f6e"),
        ("in100.floorssl.s0.d256", "#8a5cb8"), ("in100.floorssl.s0.d512", "#b8608a"),
        ("in100.floorssl.s0.laug_hinge", "#2e8b57")]
DEAD = [("in100.floorssl.s0.lejepa_augs", "#c9c9c9"), ("in100.floorssl.s0.lejepa_augs2", "#d4a0a0"),
        ("in100.floorssl.s0.laug_eps", "#e0c0c0")]


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


class Pull:
    def __init__(self, api, name):
        self.name = name
        r = run_of(api, name)
        self.hist = None if r is None else r.history(samples=2000, pandas=True)
        # test rows are 1/epoch — keyed query keeps them from being sampled out (house gotcha)
        self.test = None if r is None else r.history(samples=300, keys=["test/acc", "test/epoch"],
                                                     pandas=True)
        self.spe = SPE_FALLBACK.get(name)
        if self.test is not None and len(self.test) and self.hist is not None and len(self.hist):
            e_last = float(self.test["test/epoch"].max()) + 1
            s_at = float(self.test.loc[self.test["test/epoch"].idxmax(), "_step"])
            if e_last > 0 and s_at > 0:
                self.spe = s_at / e_last

    def series(self, key):
        if self.hist is None or key not in self.hist.columns:
            return None, None
        d = self.hist.dropna(subset=[key])
        if not len(d):
            return None, None
        return d["_step"].to_numpy(float) / self.spe, d[key].to_numpy(float)

    def probe_by_ep(self):
        if self.test is None or not len(self.test):
            return {}
        return dict(zip((self.test["test/epoch"] + 1).astype(int), self.test["test/acc"]))


def div_flag(sm):
    return bool((sm[-5:] > sm.min() + 0.15).all()) if len(sm) >= 5 else False


def main():
    api = wandb.Api(timeout=60)
    P = {n: Pull(api, n) for n in SPE_FALLBACK}
    fig, ((a1, a2), (a3, a4)) = plt.subplots(2, 2, figsize=(13.5, 9.0))

    # A: hz h_moment_kl vs hold template
    for n, c, lw in [("in100.lejepa.s0.e12f2", "#9d9d9d", 1.2),
                     ("in100.lejepa.s0.e20f", "#b0c4de", 1.2)]:
        ep, v = P[n].series("train/h_moment_kl")
        if ep is not None:
            a1.plot(ep, ema(v), color=c, lw=lw, ls="--", label=f"{n.split('.')[-1]} (template)")
    ep, v = P["in100.vicreg.s0.floorssl_hz_v2"].series("train/h_moment_kl")
    if ep is not None:
        a1.plot(ep, ema(v), color="#c9a227", lw=1.1, ls=":", label="hz_v2 stub (byol pair)")
    for n, c in DEAD:
        ep, v = P[n].series("train/h_moment_kl")
        if ep is not None:
            a1.plot(ep, ema(v), color=c, lw=1.0, ls="--", label=f"{n.split('.')[-1]} (killed)")
    for n, c in LIVE:
        ep, v = P[n].series("train/h_moment_kl")
        if ep is not None:
            a1.plot(ep, ema(v), color=c, lw=1.9, label=n.split(".")[-1])
    a1.set_title("hz cells: h_moment_kl (cls floor) vs f2/lejepa-e20f hold template", fontsize=9.5)

    # B: hz z-side terms (laug_hinge's moment_kl is the HINGE value — semantics per-arm, E21 card)
    for n, c in LIVE + DEAD[1:]:
        for key, ls, lw in [("train/moment_kl", "-", 1.9), ("train/inv", "-", 0.9)]:
            ep, v = P[n].series(key)
            if ep is not None:
                a2.plot(ep, ema(v), color=c, ls=ls, lw=lw if n.endswith(("eps", "hinge")) else lw * 0.5,
                        label=f"{n.split('.')[-1]} {key.split('/')[-1]}")
    ep, v = P["in100.vicreg.s0.floorssl_hz_v2"].series("train/moment_kl")
    if ep is not None:
        a2.plot(ep, ema(v), color="#c9a227", lw=1.1, ls=":", label="hz_v2 moment_kl")
    a2.set_yscale("log")
    a2.set_title("hz cells: z-floor (thick) + inv (thin), log scale", fontsize=9.5)

    # C: dino dose cells vs gd / e20f
    for n, c, ls in [("in100.dino.s0.e12gd", "#9d9d9d", "--"), ("in100.dino.s0.e20f", "#b39ddb", "--"),
                     ("in100.dino.s0.e20f_lam06", "#8a5cb8", "-"),
                     ("in100.dino.s0.e20f_lam008", "#4a9d9d", "-")]:
        ep, v = P[n].series("train/h_moment_kl")
        if ep is not None:
            a3.plot(ep, ema(v), color=c, ls=ls, lw=1.9 if ls == "-" else 1.2,
                    label=n.split(".")[-1])
    a3.set_title("dino dose cells: h_moment_kl vs gd (.02) / e20f (.258)", fontsize=9.5)

    # D: grad_norm envelope, live hz cells (incident watch)
    for n, c in LIVE:
        ep, v = P[n].series("train/grad_norm")
        if ep is not None:
            a4.plot(ep, v, color=c, lw=0.6, alpha=0.55, label=f"{n.split('.')[-1]} grad_norm")
            a4.plot(ep, ema(v, 0.98), color=c, lw=1.8)
    a4.set_yscale("log")
    a4.set_title("hz cells: grad_norm (raw + EMA .98) — incident watch", fontsize=9.5)

    for ax in (a1, a2, a3, a4):
        ax.grid(alpha=0.15)
        ax.legend(fontsize=7)
        ax.set_xlabel("epoch")
    os.makedirs(f"{ROOT}/results/figures/e21", exist_ok=True)
    fig.tight_layout()
    fig.savefig(f"{ROOT}/results/figures/e21/e21_curves.png", dpi=160)

    # THE fork read (menu item 1): monitor probe_acc by epoch, live cells vs collapsed refs
    print(f"\nmonitor fork table (collapse signature = ep3->4 fall; v1 peak .0868@3, laug2 "
          f".0944@3->.0728@4):")
    for n, _ in LIVE + DEAD:
        pe = P[n].probe_by_ep()
        cells = " ".join(f"e{e}:{pe[e]:.4f}" for e in sorted(pe) if e <= 12)
        print(f"  {n.split('.')[-1]:14s} {cells if cells else 'no test rows yet'}")

    print(f"\n{'run':34s} {'ep':>6s} {'h_kl bot@ep':>13s} {'now':>6s} {'div':>4s} "
          f"{'z_kl now':>9s} {'inv now':>8s} {'gnorm med/max':>14s}")
    for n in [nm for nm, _ in LIVE] + ["in100.dino.s0.e20f_lam06", "in100.dino.s0.e20f_lam008"]:
        ep, v = P[n].series("train/h_moment_kl")
        if ep is None:
            print(f"{n:34s} MISSING"); continue
        sm = ema(v); ib = int(sm.argmin())
        zep, zv = P[n].series("train/moment_kl")
        iep, iv = P[n].series("train/inv")
        gep, gv = P[n].series("train/grad_norm")
        print(f"{n:34s} {ep[-1]:6.2f} {sm[ib]:8.3f}@{ep[ib]:4.1f} {sm[-1]:6.3f} "
              f"{'DIV' if div_flag(sm) else '—':>4s} "
              f"{(ema(zv)[-1] if zv is not None else float('nan')):9.4f} "
              f"{(ema(iv)[-1] if iv is not None else float('nan')):8.4f} "
              f"{(np.median(gv) if gv is not None else float('nan')):6.2f}/"
              f"{(np.max(gv) if gv is not None else float('nan')):.2f}")
    # dino monitor deltas vs control at matched epochs, next to gd's own deltas
    ctrl = P["in100.dino.s0"].probe_by_ep()
    gd = P["in100.dino.s0.e12gd"].probe_by_ep()
    print("\ndino monitor deltas vs control (E12-T9 caveat riding):")
    for n in ["in100.dino.s0.e20f_lam06", "in100.dino.s0.e20f_lam008", "in100.dino.s0.e12gd"]:
        pe = P[n].probe_by_ep() if n != "in100.dino.s0.e12gd" else gd
        cells = [f"e{e}:{100 * (pe[e] - ctrl[e]):+.1f}" for e in EPM if e in pe and e in ctrl]
        print(f"  {n.split('.')[-1]:14s} {' '.join(cells) if cells else 'no matched test rows yet'}")
    print("\nwrote results/figures/e21/e21_curves.png", flush=True)


if __name__ == "__main__":
    main()
