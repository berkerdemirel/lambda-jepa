"""E18 per-knot CF residual instrument (card requirement, D-041: "per-knot CF residual
instrument for E18 still TO BUILD"). At the embed tap of each run, the sliced empirical CF is
compared to BOTH declared targets at the 17 knots: err(t_k | φ) = mean over slices of
(E cos(t_k·s) − φ(t_k))² + (E sin(t_k·s))², the exact quantity SIGReg integrates (before the
Gaussian-window quadrature weights). Low-t knots read the moment region, high-t the tails —
WHERE the residual lives is the P-A/P-B/P-C interpretive key, alongside the slice
excess-kurtosis (the sign-flip datum: does the t_8.2 arm push slice tails PAST Gaussian
toward the declared prior, or just reduce the super-Gaussian excess?). Slices are unseeded
fresh draws (SIGReg convention); batches = seed-0 loader first K (pull/anatomy parity).
Rows → results/diag/e18_knot_residual.csv · figure → results/figures/e18/e18_knot_residual.png.
RAW; no takeaway."""
import csv
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
from omegaconf import OmegaConf
from torch.amp import autocast
from torch.utils.data import DataLoader

from sslgap.data import seed_everything
from sslgap.methods import METHODS
from sslgap.methods.base import Frame
from sslgap.methods.lejepa import t_nu_cf

ROOT = "/nfs/scistore19/locatgrp/bdemirel/ssl_project"
RUNS = [("in100.lejepa.s0", "ctrl"), ("in100.lejepa.s0.hpull_sigreg", "sigreg 10% (gauss)"),
        ("in100.lejepa.s0.hpull_sigreg_t", "sigreg_t 10% (t_8.2)"),
        ("in100.lejepa.s0.hpull_sigreg3", "sigreg 3% (gauss)")]
K_BATCHES, N_SLICES, KNOTS, T_MAX, NU = 4, 1024, 17, 3.0, 8.2
BLUE, ORANGE, GREEN, RED = "#3d65d0", "#e07b39", "#3a8c5c", "#c04f4f"
COL = {"ctrl": BLUE, "sigreg 10% (gauss)": GREEN, "sigreg_t 10% (t_8.2)": ORANGE,
       "sigreg 3% (gauss)": RED}


def main():
    dev = "cuda"
    t = torch.linspace(0, T_MAX, KNOTS)
    phi_g = torch.exp(-t.square() / 2.0)
    phi_t = t_nu_cf(t, NU)
    rows, prof = [], {}
    for run, label in RUNS:
        pay = torch.load(f"{ROOT}/outputs/{run}_ep100.pt", map_location="cpu",
                         weights_only=False)
        cfg = OmegaConf.create(pay["cfg"])
        frame = Frame(name=cfg.frame.name, model_name=cfg.frame.model_name,
                      img_size=cfg.frame.img_size, dataset=cfg.frame.dataset,
                      data_root=cfg.frame.get("data_root"), epochs=cfg.frame.epochs,
                      seed=cfg.seed, grad_clip=cfg.frame.grad_clip, num_workers=0, device=dev)
        seed_everything(0)
        method = METHODS[cfg.method.name](cfg.method, frame)
        modules = method.build_modules().to(dev)
        for role, sd in pay["modules"].items():
            if role != "probe":
                modules[role].load_state_dict(sd)
        method.load_extras(pay.get("extras", {}))
        loader = DataLoader(method.build_train_dataset(), batch_size=cfg.bs, shuffle=True,
                            drop_last=True, num_workers=0,
                            generator=torch.Generator().manual_seed(0))
        it = iter(loader)
        err_g = torch.zeros(KNOTS)
        err_t = torch.zeros(KNOTS)
        kurt = []
        tk = t.to(dev)
        for _ in range(K_BATCHES):
            views, _y = next(it)
            x = views.flatten(0, 1).to(dev)
            with torch.no_grad(), autocast(dev, dtype=torch.bfloat16):
                emb = modules["encoder"](x).float()
            A = torch.randn(emb.size(1), N_SLICES, device=dev)
            A = A.div_(A.norm(p=2, dim=0))
            s = (emb @ A).float()                                     # [n, S] raw slices
            x_t = s.unsqueeze(-1) * tk                                # [n, S, K]
            e_g = (x_t.cos().mean(0) - phi_g.to(dev)).square() + x_t.sin().mean(0).square()
            e_t = (x_t.cos().mean(0) - phi_t.to(dev)).square() + x_t.sin().mean(0).square()
            err_g += e_g.mean(0).cpu()
            err_t += e_t.mean(0).cpu()
            sz = (s - s.mean(0)) / s.std(0).clamp_min(1e-6)
            kurt.append((sz.pow(4).mean(0) - 3).mean().item())
        err_g /= K_BATCHES
        err_t /= K_BATCHES
        kbar = sum(kurt) / len(kurt)
        prof[label] = (err_g.tolist(), err_t.tolist(), kbar)
        for k in range(KNOTS):
            rows.append({"run": run, "knot_t": round(float(t[k]), 4),
                         "err_vs_gauss": err_g[k].item(), "err_vs_t82": err_t[k].item(),
                         "slice_kurt": round(kbar, 5)})
        print(f"{label:24s} slice-kurt {kbar:+.4f}  err@t3 gauss {err_g[-1]:.3e} "
              f"t82 {err_t[-1]:.3e}", flush=True)
    out = f"{ROOT}/results/diag/e18_knot_residual.csv"
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(12.5, 4.6))
    for label, (eg, et, kb) in prof.items():
        a1.semilogy(t, eg, color=COL[label], lw=1.8, label=f"{label} (κ {kb:+.2f})")
        a2.semilogy(t, et, color=COL[label], lw=1.8)
    tgt = (phi_g - phi_t).square()
    for ax, ttl in ((a1, "per-knot CF residual vs GAUSSIAN target"),
                    (a2, "per-knot CF residual vs t_8.2 target")):
        ax.semilogy(t, tgt.clamp_min(1e-12), color="#777777", ls=":", lw=1.2,
                    label="(φ_g − φ_t)² — the target gap itself" if ax is a1 else None)
        ax.set_xlabel("knot t (low = moments · high = tails)")
        ax.set_ylabel("err(t) = ΔRe² + ΔIm²")
        ax.set_title(ttl, fontsize=10)
        ax.grid(alpha=0.15)
    a1.legend(fontsize=7.5)
    fig.tight_layout()
    os.makedirs(f"{ROOT}/results/figures/e18", exist_ok=True)
    fig.savefig(f"{ROOT}/results/figures/e18/e18_knot_residual.png", dpi=160)
    print("wrote e18_knot_residual.{csv,png}", flush=True)


if __name__ == "__main__":
    main()
