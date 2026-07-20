"""E19 curve-commensurability instrument (HANDOVER 2026-07-17 q2, the parked h_kl question).
The three "floor" curves (f2 h_moment_kl @calibrated embed, floorssl moment_kl @raw z,
floorssl_hz h_moment_kl @raw cls) share a functional but not a state: the logged value is
KL(N(μ_Q,Σ_Q)‖N(0,I))/d' on a fresh 128-d slice, and it decomposes EXACTLY into three
dimensionless, cross-tap-commensurable parts
    cone  = ½‖μ_Q‖²/d'                       (mean offset; ≈ ½‖μ‖²/D)
    scale = ½(m − 1 − ln m),  m = trΣ_Q/d'   (global per-dim variance vs 1)
    aniso = ½(ln m − logdetΣ_Q/d')           (Jensen gap; scale-invariant spectral spread)
so WHERE each lane's floor value lives (and which part moves) is measurable per checkpoint.
Alongside, per tap: R = within-instance/total variance (the lane-free alignment readout — each
lane's inv is R at its loss space up to a fixed view-count factor), pos/rand view cosines
(cone readout), and raw per-dim variance. Batches are the seed-0 loader's first K — identical
across every checkpoint of a lane and with the e12h_pull rows (num_workers=0 preserves the
augmentation stream). Checkpoints are self-describing (pay["cfg"] rebuilds the method; extras
restore embed_calib state). Rows → results/diag/e19_floor_anatomy.csv; RAW, no takeaway."""
import csv
import os

import torch
from omegaconf import OmegaConf
from torch.amp import autocast
from torch.utils.data import DataLoader

from sslgap.data import seed_everything
from sslgap.methods import METHODS
from sslgap.methods.base import Frame

ROOT = "/nfs/scistore19/locatgrp/bdemirel/ssl_project"
OUT = f"{ROOT}/results/diag/e19_floor_anatomy.csv"
CKPTS = [(run, ep) for run in
         ["in100.lejepa.s0.e12f2", "in100.lejepa.s0", "in100.vicreg.s0.e12gv2",
          "in100.vicreg.s0"] for ep in ["ep25", "ep50", "ep75", "ep100"]] + \
        [("in100.vicreg.s0.floorssl", "last"), ("in100.vicreg.s0.floorssl_hz", "last"),
         ("in100.vicreg.s0.floorssl_emb", "last"), ("in100.vicreg.s0.floorssl_hz_emb", "last")]
K_BATCHES, N_SLICES, D_SLICE, EPS = 4, 8, 128, 1e-4


def slice_decomp(x):
    """x [n, D] fp32: (cone, scale, aniso, total) averaged over fresh slices. The slice dim
    is min(D_SLICE, D) — the D-050 small-z taps (D < 128) get the EXACT full-cov decomp
    (rotation-invariant ⇒ the Q is a value-level no-op there); D >= 128 unchanged."""
    ds = min(D_SLICE, x.size(1))
    mu_f, xc = x.mean(0), x - x.mean(0)
    parts = []
    for _ in range(N_SLICES):
        Q, _ = torch.linalg.qr(torch.randn(x.size(1), ds, device=x.device))
        mu, pc = mu_f @ Q, xc @ Q
        cov = pc.T @ pc / (x.size(0) - 1) + EPS * torch.eye(ds, device=x.device)
        m = cov.diagonal().mean()
        logdet = 2 * torch.linalg.cholesky(cov).diagonal().log().sum() / ds
        cone = 0.5 * mu.square().sum() / ds
        scale = 0.5 * (m - 1 - m.log())
        aniso = 0.5 * (m.log() - logdet)
        parts.append(torch.stack([cone, scale, aniso, cone + scale + aniso]))
    return torch.stack(parts).mean(0).tolist()


def tap_stats(x, N, V):
    """x [N*V, D] fp32. Moment state + view geometry, all dimensionless except mbar/w_raw."""
    D = x.size(1)
    mu = x.mean(0)
    var = x.var(0)
    xv = x.reshape(N, V, D)
    w_raw = (xv - xv.mean(1, keepdim=True)).square().mean()
    R = w_raw * V / (V - 1) / var.mean()
    xn = torch.nn.functional.normalize(xv, dim=-1)
    pos = (torch.einsum("nvd,nwd->nvw", xn, xn).sum((1, 2)) - V) / (V * (V - 1))
    rand = (xn[:, 0] * xn[:, 0].roll(1, 0)).sum(-1)
    row = {"D": D, "n": N * V, "mu2_D": (mu @ mu / D).item(), "mbar": var.mean().item(),
           "w_raw": w_raw.item(), "R": R.item(), "pos": pos.mean().item(),
           "rand": rand.mean().item()}
    row["cone"], row["scale"], row["aniso"], row["kl"] = slice_decomp(x)
    return row


def main():
    # argv mode (divergence ladder, 2026-07-17c): "run:ck" pairs run only those checkpoints
    # and APPEND to the csv; no args = the original full-list overwrite, byte-same.
    import sys
    args = [tuple(a.rsplit(":", 1)) for a in sys.argv[1:]]
    dev = "cuda"
    rows = []
    for run, ck in (args or CKPTS):
        pay = torch.load(f"{ROOT}/outputs/{run}_{ck}.pt", map_location="cpu",
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
        ep = pay["epoch"] + 1
        loader = DataLoader(method.build_train_dataset(), batch_size=cfg.bs, shuffle=True,
                            drop_last=True, num_workers=0,
                            generator=torch.Generator().manual_seed(0))
        it = iter(loader)
        acc = {}
        for _ in range(K_BATCHES):
            views, _y = next(it)
            if isinstance(views, (list, tuple)):  # dino multi-crop: the floor's tap lives on
                views = views[0]                  # the two GLOBAL crops; locals dropped
            N, V = views.shape[:2]
            x = views.flatten(0, 1).to(dev)
            with torch.no_grad(), autocast(dev, dtype=torch.bfloat16):
                if cfg.method.name == "lejepa":
                    feats = modules["encoder"].forward_features(x)
                    taps = {"cls": feats[:, 0]}
                    taps["embed"] = modules["encoder"].forward_head(feats)
                    taps["proj"] = modules["projector"](taps["embed"])
                else:
                    tok = modules["backbone"].forward_features(x)
                    taps = {"cls": tok[:, 0]}
                    z_in = modules["embed"](taps["cls"]) if "embed" in modules else taps["cls"]
                    if "embed" in modules:
                        taps["emb"] = z_in
                    taps["z"] = modules["projector"](z_in)
            for tap, xt in taps.items():
                st = tap_stats(xt.float(), N, V)
                acc.setdefault(tap, []).append(st)
        for tap, sts in acc.items():
            mean = {k: sum(s[k] for s in sts) / len(sts) for k in sts[0]}
            rows.append({"run": run, "ep": ep, "tap": tap,
                         **{k: (v if k in ("D", "n") else round(v, 5))
                            for k, v in mean.items()}})
            print(rows[-1], flush=True)
    mode = "a" if args and os.path.exists(OUT) else "w"
    with open(OUT, mode, newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        if mode == "w":
            w.writeheader()
        w.writerows(rows)
    print("wrote", OUT, flush=True)


if __name__ == "__main__":
    main()
