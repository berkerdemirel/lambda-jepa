"""E19 session 2026-07-17b (HANDOVER q2, Berker's parked question): the f2 vs floorssl vs
floorssl_hz curve comparison, commensurability machinery explicit. Raw wandb histories are
NON-commensurable across lanes (different inv functionals/spaces; floors sharing a name but
differing in tap/dose/calibration — the on-figure annotations carry those caveats); the
companion instrument experiments/e19_floor_anatomy.py computes the commensurable objects
(floor-KL decomposition cone/scale/aniso per tap, within/total variance ratio at the loss
space, pos/rand) from checkpoints. This script: `pull` caches histories+configs from wandb
(login-safe: network-bound), `figs` renders FS-only figures to results/figures/e19/.

  python experiments/e19_curves.py pull | figs
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
         "9912c025-c88b-4fe8-91e0-27ae461f97e7/scratchpad/wandb_cache")
RUNS = ["in100.lejepa.s0.e12f2", "in100.lejepa.s0",
        "in100.vicreg.s0.e12gv2", "in100.vicreg.s0",
        "in100.vicreg.s0.floorssl", "in100.vicreg.s0.floorssl_hz"]
TRAIN_KEYS = ["train/inv", "train/sigreg", "train/moment_kl", "train/h_moment_kl",
              "train/var", "train/cov", "train/grad_norm", "train/loss"]
BLUE, ORANGE, GREEN, RED, GRAY, PURPLE = "#3d65d0", "#e07b39", "#3a8c5c", "#c04f4f", "#8a8a8a", "#8a5cb8"


def pull():
    import pandas as pd
    import wandb
    os.makedirs(CACHE, exist_ok=True)
    api = wandb.Api(timeout=60)
    found = {}
    for r in api.runs("causal-learning-ai-ista/sslgap",
                      filters={"displayName": {"$in": RUNS}}):
        if r.name not in found or r.lastHistoryStep > found[r.name].lastHistoryStep:
            found[r.name] = r
    for name in RUNS:
        r = found.get(name)
        if r is None:
            print(f"[pull] MISSING run {name}", flush=True)
            continue
        # no keys= filter: wandb returns only ALL-keys-complete rows under it, and no run
        # logs the full cross-run union — pull the sampled union and subset instead.
        hist = r.history(samples=2500, pandas=True)
        hist = hist[[c for c in ["_step", *TRAIN_KEYS] if c in hist.columns]]
        test = r.history(samples=300, keys=["test/acc", "test/epoch"], pandas=True)
        hist.to_csv(f"{CACHE}/{name}.train.csv", index=False)
        test.to_csv(f"{CACHE}/{name}.test.csv", index=False)
        json.dump({k: v for k, v in r.config.items() if k != "frame"} | {"frame": r.config.get("frame")},
                  open(f"{CACHE}/{name}.config.json", "w"), indent=1, default=str)
        print(f"[pull] {name}: laststep={r.lastHistoryStep} train_rows={len(hist)} "
              f"test_rows={len(test)}", flush=True)


def load(name):
    import pandas as pd
    tr = pd.read_csv(f"{CACHE}/{name}.train.csv")
    te = pd.read_csv(f"{CACHE}/{name}.test.csv")
    # step->epoch from the per-epoch test log (robust to resume gaps): epoch = step/spe
    spe = float(np.polyfit(te["test/epoch"] + 1, te["_step"], 1)[0])
    tr["ep"] = tr["_step"] / spe
    te["ep"] = te["_step"] / spe
    return tr, te


# (name, color, dash, short label) — lanes: lejepa solid-family blue/gray, vicreg warm
STYLE = {"in100.lejepa.s0.e12f2": (BLUE, "-", "f2  (lejepa+floor@embed λ=.02, calibrated)"),
         "in100.lejepa.s0": (GRAY, "-", "lejepa ctrl"),
         "in100.vicreg.s0.e12gv2": (PURPLE, "-", "gv2 (vicreg+floor@cls λ=.02)"),
         "in100.vicreg.s0": (GRAY, "--", "vicreg ctrl"),
         "in100.vicreg.s0.floorssl": (ORANGE, "-", "floorssl (floor@z w=19.1, SOLE anti-collapse)"),
         "in100.vicreg.s0.floorssl_hz": (RED, "-", "floorssl_hz (= + floor@cls λ=.02)")}


def _plot(ax, tr, key, name, ema=0.98):
    c, ls, lab = STYLE[name]
    d = tr.dropna(subset=[key])
    if not len(d):
        return
    y = d[key].to_numpy()
    s = np.empty_like(y)
    acc = y[0]
    for i, v in enumerate(y):
        acc = ema * acc + (1 - ema) * v
        s[i] = acc
    ax.plot(d["ep"], s, color=c, ls=ls, lw=1.8, label=lab)
    ax.plot(d["ep"], y, color=c, ls=ls, lw=0.5, alpha=0.18)


def figs():
    data = {n: load(n) for n in RUNS if os.path.exists(f"{CACHE}/{n}.train.csv")}
    os.makedirs(f"{ROOT}/results/figures/e19", exist_ok=True)

    fig, axes = plt.subplots(2, 2, figsize=(14.5, 9.2))
    ax = axes[0, 0]
    for n in ["in100.lejepa.s0.e12f2", "in100.lejepa.s0", "in100.vicreg.s0.floorssl",
              "in100.vicreg.s0.floorssl_hz", "in100.vicreg.s0"]:
        _plot(ax, data[n][0], "train/inv", n)
    ax.set_yscale("log")
    ax.set_title("inv terms, RAW — shapes only, values non-commensurable:\n"
                 "lejepa = mean-of-4-views MSE @16-d proj (w=.98) · vicreg = pair MSE @2048-d z (w=25)",
                 fontsize=9.5)
    ax.legend(fontsize=7.5, loc="upper right")

    ax = axes[0, 1]
    for n, key in [("in100.lejepa.s0.e12f2", "train/h_moment_kl"),
                   ("in100.vicreg.s0.e12gv2", "train/h_moment_kl"),
                   ("in100.vicreg.s0.floorssl_hz", "train/h_moment_kl"),
                   ("in100.vicreg.s0.floorssl", "train/moment_kl"),
                   ("in100.vicreg.s0.floorssl_hz", "train/moment_kl")]:
        if n in data:
            _plot(ax, data[n][0], key, n)
    h, l = ax.get_legend_handles_labels()
    seen, hh, ll = set(), [], []
    for a, b in zip(h, l):
        if b not in seen:
            seen.add(b), hh.append(a), ll.append(b)
    ax.legend(hh, ll, fontsize=7.5, loc="upper right")
    ax.set_yscale("log")
    ax.set_title("the floors, RAW — one name, three objects:\n"
                 "f2 @CALIBRATED 512-d embed (ε-pull 1.1% of lane) · hz @raw 384-d CLS (0.2%) · "
                 "floorssl @raw 2048-d z (83%, destination)", fontsize=9.5)

    ax = axes[1, 0]
    for n in RUNS:
        if n in data:
            _plot(ax, data[n][0], "train/grad_norm", n)
    ax.set_yscale("log")
    ax.set_title("grad_norm (total, per-lane loss scales differ)", fontsize=9.5)
    ax.set_xlabel("epoch")

    ax = axes[1, 1]
    for n in RUNS:
        if n not in data:
            continue
        c, ls, lab = STYLE[n]
        te = data[n][1]
        ax.plot(te["ep"], te["test/acc"], color=c, ls=ls, lw=1.8, label=lab)
    ax.set_title("online probe acc @ declared h — the one directly commensurable live scalar\n"
                 "(same probe protocol/arch/data; aug families still per-method, D-004)", fontsize=9.5)
    ax.set_xlabel("epoch")
    ax.legend(fontsize=7.5, loc="lower right")
    for ax in axes.flat:
        ax.grid(alpha=0.15)
    fig.suptitle("E19 side-by-side, raw curves with the commensurability caveats — "
                 "f2 vs floorssl vs floorssl_hz (+ gv2, controls)", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(f"{ROOT}/results/figures/e19/e19_curves_raw.png", dpi=160)
    plt.close(fig)
    print("wrote e19_curves_raw.png", flush=True)

    # h_kl question figure: trajectories -> exact decomposition -> the reach-back contrast
    import csv as _csv
    from matplotlib.patches import Patch
    A = {(r["run"], int(r["ep"]), r["tap"]): {k: float(r[k]) for k in
         ["mu2_D", "mbar", "R", "pos", "rand", "cone", "scale", "aniso", "kl"]}
         for r in _csv.DictReader(open(f"{ROOT}/results/diag/e19_floor_anatomy.csv"))}
    comps = [("cone", BLUE), ("scale", ORANGE), ("aniso", GREEN)]
    fig, axes = plt.subplots(1, 3, figsize=(19, 5.0))

    ax = axes[0]
    for n, key in [("in100.lejepa.s0.e12f2", "train/h_moment_kl"),
                   ("in100.vicreg.s0.e12gv2", "train/h_moment_kl"),
                   ("in100.vicreg.s0.floorssl_hz", "train/h_moment_kl")]:
        _plot(ax, data[n][0], key, n, ema=0.995)
    eps = [25, 50, 75, 100]
    ax.plot(eps, [A["in100.vicreg.s0", e, "cls"]["kl"] for e in eps], "o:", color=GRAY,
            lw=1.4, label="vicreg ctrl @cls — UNfloored counterfactual (anatomy monitor)")
    ax.plot([18], [A["in100.vicreg.s0.floorssl", 18, "cls"]["kl"]], "s", color=ORANGE, ms=8,
            label="floorssl @cls monitor (z-only floor): cone intact")
    ax.set_title("h_moment_kl, same λ=.02 three ways — hz rides gv2's hump\n"
                 "(rise→crest→slow fall is the LANE's CLS trajectory, not a z-floor interaction)",
                 fontsize=9.5)
    ax.set_xlabel("epoch")
    ax.legend(fontsize=7.5)
    ax.grid(alpha=0.15)

    ax = axes[1]
    groups = [("f2@embed", "in100.lejepa.s0.e12f2", "embed", [25, 50, 75, 100]),
              ("gv2@cls", "in100.vicreg.s0.e12gv2", "cls", [25, 50, 75, 100]),
              ("hz@cls", "in100.vicreg.s0.floorssl_hz", "cls", [20]),
              ("floorssl@z", "in100.vicreg.s0.floorssl", "z", [18]),
              ("hz@z", "in100.vicreg.s0.floorssl_hz", "z", [20])]
    xpos, xticks, x = [], [], 0.0
    for lab, run, tap, es in groups:
        for e in es:
            r, bottom = A[run, e, tap], 0.0
            for comp, col in comps:
                ax.bar(x, r[comp], bottom=bottom, color=col, width=0.8)
                bottom += r[comp]
            xpos.append(x)
            xticks.append(f"ep{e}")
            x += 1
        ax.annotate(lab, ((2 * x - len(es) - 1) / 2, -0.32), ha="center", fontsize=8.5,
                    annotation_clip=False)
        x += 0.9
    ax.set_xticks(xpos, xticks, fontsize=7)
    ax.set_title("the same logged value, decomposed (KL = cone + scale + aniso; recomputed kl\n"
                 "matches each live curve): f2 pays ONLY aniso .19 — its affine tap absorbs scale, "
                 "calib killed the cone;\ncls floors pay frozen scale (m≈.15) + an aniso hump; "
                 "z floors sit low and falling", fontsize=9)
    ax.legend(handles=[Patch(color=c, label=f"{l}: " + {"cone": "½‖μ_Q‖²/d′", "scale": "½(m−1−ln m)",
                       "aniso": "½·logdet Jensen gap"}[l]) for l, c in comps], fontsize=8)
    ax.grid(alpha=0.15, axis="y")

    ax = axes[2]
    pairs = [("lejepa ctrl\n@cls ep100", "in100.lejepa.s0", 100),
             ("f2 @cls ep100\n(floor@embed,\n1 affine away)", "in100.lejepa.s0.e12f2", 100),
             ("vicreg ctrl\n@cls ep100", "in100.vicreg.s0", 100),
             ("floorssl @cls\nep18 (floor@z,\nbeyond BN-MLP)", "in100.vicreg.s0.floorssl", 18),
             ("gv2 @cls ep100\n(floor AT cls)", "in100.vicreg.s0.e12gv2", 100)]
    for i, (lab, run, e) in enumerate(pairs):
        r, bottom = A[run, e, "cls"], 0.0
        for comp, col in comps:
            ax.bar(i, r[comp], bottom=bottom, color=col, width=0.72)
            bottom += r[comp]
        ax.annotate(f"rand={r['rand']:.2f}\nvar={r['mbar']:.3f}", (i, bottom + 0.05),
                    ha="center", fontsize=7.5)
    ax.set_ylim(0, 3.55)
    ax.set_xticks(range(len(pairs)), [p[0] for p in pairs], fontsize=7.5)
    ax.set_title("reach-back at CLS (the KL monitor on the h nobody floored directly):\n"
                 "floor@embed transmits through lejepa's affine emb (cone+scale FIXED at cls);\n"
                 "floor@z transmits NOTHING through vicreg's BN expander — BN is a moment firewall",
                 fontsize=9)
    ax.grid(alpha=0.15, axis="y")

    fig.tight_layout()
    fig.savefig(f"{ROOT}/results/figures/e19/e19_hkl_question.png", dpi=160)
    plt.close(fig)
    print("wrote e19_hkl_question.png", flush=True)

    # realized pull-share trajectories from the e12h_pull ledger (init rows + e19tp ckpt rows):
    # "dose matching is init-only", quantified — share = weight*g_enc(floor) / lane total.
    G = {}
    for r in _csv.DictReader(open(f"{ROOT}/results/diag/e12h_pull.csv")):
        G[r["tag"], r["term"]] = float(r["g_enc"])
    LEJ = {"sigreg": 0.02, "inv": 0.98, "h_moment_kl": 0.02}
    GV2 = {"inv": 25, "var": 25, "cov": 1, "h_moment_kl": 0.02}
    FLR = {"inv": 25, "moment_kl": 19.10}
    HZ = {**FLR, "h_moment_kl": 0.02}

    def share(tag, weights, floor_term, borrow=()):
        g = {t: G[src, t] for t, src in borrow}
        g |= {t: G[tag, t] for t in weights if t not in g}
        tot = sum(weights[t] * g[t] for t in weights)
        return weights[floor_term] * g[floor_term] / tot

    fig, ax = plt.subplots(figsize=(7.6, 4.8))
    eps = [25, 50, 75, 100]
    for lab, col, xs, ys in [
        ("f2 floor@embed (λ=.02)", BLUE, [0] + eps,
         [share("e12f7pull", LEJ, "h_moment_kl")] +
         [share(f"e19tp.f2.ep{e}", LEJ, "h_moment_kl") for e in eps]),
        ("gv2 floor@cls (λ=.02)", PURPLE, [0] + eps,
         [share("e12gvclspull", GV2, "h_moment_kl")] +
         [share(f"e19tp.gv2.ep{e}", GV2, "h_moment_kl") for e in eps]),
        ("floorssl floor@z (w=19.1)", ORANGE, [0, 18],
         [share("e19pull", FLR, "moment_kl"), share("e19tp.floorssl.last", FLR, "moment_kl")]),
        ("hz floor@cls (λ=.02)", RED, [0, 20],
         [share("e19pull", HZ, "h_moment_kl",
                borrow=[("h_moment_kl", "e12gvclspull")]),
          share("e19tp.hz.last", HZ, "h_moment_kl")])]:
        ax.plot(xs, ys, "-o", color=col, lw=2, ms=5, label=lab)
    ax.set_yscale("log")
    ax.set_ylabel("floor share of lane weighted pull (w·g_enc)")
    ax.set_xlabel("epoch of loaded checkpoint (0 = init record)")
    ax.set_title("dose matching is init-only, quantified: the three floors' REALIZED shares\n"
                 "drift in different directions (f2 ×9 up by ep25; gv2 ×6 down; floorssl "
                 "destination→parity with inv)", fontsize=9.5)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.15)
    fig.tight_layout()
    fig.savefig(f"{ROOT}/results/figures/e19/e19_pull_traj.png", dpi=160)
    plt.close(fig)
    print("wrote e19_pull_traj.png", flush=True)


if __name__ == "__main__":
    {"pull": pull, "figs": figs}[sys.argv[1]]()
