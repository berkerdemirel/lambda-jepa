"""E24 acc↔Ω_h training-trajectory figure (Berker 2026-08-03: "i will mainly check
omega_h … present omega_hs and the test accuracies … on a single figure across the
training, left y acc, right y omega_h, x step"). One panel per cell — wave-1 originals +
corrected (INCLUDING killed cells — the Ω_h blowup ↔ collapse coupling is the diagnosis
view) + the view-mean v-cells (OAS map + ring/bs512 variants; in100 mirrors + twins) —
acc solid left (0–1), Ω_h orange right (log; in-training instrument = train-aug
stack, trajectory reading). Source: wandb histories (test/acc, orbit/omega_h). One PNG
per frame → results/figures/e24/."""
import os

import matplotlib
import wandb

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = "/nfs/scistore19/locatgrp/bdemirel/ssl_project"
LAT = [f"s{h}{r}" for h in "0123" for r in "abcd"]
VM = ["v1", "v2lr", "v3nq", "v4oas", "va", "vb", "vc", "vh0", "vac"]
FRAMES = [
    ("toy", LAT + [t + "c" for t in LAT if t not in ("s0d", "s1d")] + VM, 6, 7),
    ("in100", ["s0a", "s0b", "s0c", "s0d", "s1a", "s1b", "s1c", "s1d",
               "s0ac", "s0cc", "s0dc", "s1ac", "s1bc", "s1dc",
               "va", "vb", "vc", "vh0", "vac", "vcc"], 4, 5),
]


def hist(run, key):
    rows = run.history(keys=[key], samples=1000, pandas=False)
    pts = sorted((r["_step"], r[key]) for r in rows if r.get(key) is not None)
    return [p[0] for p in pts], [p[1] for p in pts]


def pick(runs):
    # a display name can match dead relaunch stubs (zero-step OOM/parse casualties);
    # plot the attempt that actually trained
    best, n_best = None, -1
    for r in runs:
        n = len(r.history(keys=["test/acc"], samples=4, pandas=False))
        if n > n_best:
            best, n_best = r, n
    return best


def main():
    api = wandb.Api(timeout=120)
    os.makedirs(f"{ROOT}/results/figures/e24", exist_ok=True)
    for frame, tags, nr, nc in FRAMES:
        fig, axes = plt.subplots(nr, nc, figsize=(2.6 * nc, 2.1 * nr), facecolor="white")
        for i, tag in enumerate(tags):
            ax = axes.flat[i]
            run = pick(api.runs(f"causal-learning-ai-ista/sslgap",
                                filters={"display_name": f"{frame}.floorssl.s0.e24{tag}"}))
            if run is None:
                ax.set_title(f"{tag} (no run)", fontsize=7)
                continue
            sa, va = hist(run, "test/acc")
            so, vo = hist(run, "orbit/omega_h")
            ax.plot(sa, va, color="#3d65d0", lw=1.2)
            ax.set_ylim(0, 1)
            ax2 = ax.twinx()
            ax2.plot(so, vo, color="#e07b39", lw=1.0, alpha=0.85)
            ax2.set_yscale("log")
            ax2.set_ylim(0.2, 200)
            ax2.axhline(1.0, color="#e07b39", lw=0.5, ls=":", alpha=0.5)
            best = max(va) if va else float("nan")
            ax.set_title(f"{tag}  best={best:.3f}", fontsize=7.5)
            ax.tick_params(labelsize=6, length=0)
            ax2.tick_params(labelsize=6, length=0, colors="#b06020")
            for s in ("top",):
                ax.spines[s].set_visible(False)
        for j in range(len(tags), nr * nc):
            axes.flat[j].axis("off")
        fig.suptitle(f"E24 {frame} wave 1 + corrected + view-mean (v*) — test acc (blue, "
                     f"left, 0–1) vs in-training Ω_h (orange, right, LOG, dotted=1) vs "
                     f"step; killed cells show truncated curves — RAW", fontsize=10, y=0.995)
        fig.tight_layout(rect=(0, 0.01, 1, 0.96))
        out = f"{ROOT}/results/figures/e24/e24_acc_omega_{frame}.png"
        fig.savefig(out, dpi=150)
        print("wrote", out, flush=True)


if __name__ == "__main__":
    main()
