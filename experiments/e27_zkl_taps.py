"""Where along the projector does the spike-block's variance appear? (Berker 2026-09-10.)

The forensic (e27_zkl_forensics.py) showed the z conditioner's spikes are TRACE excursions, not
collapse: on spike steps lambda_min and logdet are unchanged while tr(Sigma) goes 111 -> 177. Each
spike run traces to one physical 128-row block ageing through the 4-step ring (14/14 runs).

This walks the same per-image view-mean block through the projector and reports the per-dimension
variance at each tap -- h (trunk CLS), after BN1+ReLU, after BN2+ReLU, and z -- so an amplification
introduced by one layer is visible as a jump in the ratio between consecutive taps. W6 is the only
weight whose scale is NOT cancelled by a following BatchNorm, so its singular spectrum is reported
too, together with the share of each block's z variance lying in W6's top left-singular directions.

Usage: python experiments/e27_zkl_taps.py <ckpt> <out csv> [steps]
"""
import sys
import numpy as np
import torch
from omegaconf import OmegaConf
from torch.utils.data import DataLoader

sys.path.insert(0, "/nfs/scistore19/locatgrp/bdemirel/ssl_project")
from sslgap.methods import METHODS
from sslgap.methods.base import Frame
from sslgap.ckpt.schema import load_payload

CKPT, OUT = sys.argv[1], sys.argv[2]
STEPS = int(sys.argv[3]) if len(sys.argv) > 3 else 250
dev = "cuda"
ck = load_payload(CKPT, map_location="cpu")
cfg = OmegaConf.create(ck["cfg"])
frame = Frame(name=cfg.frame.name, model_name=cfg.frame.model_name, img_size=cfg.frame.img_size,
              dataset=cfg.frame.dataset, data_root=cfg.frame.get("data_root"),
              epochs=cfg.frame.epochs, seed=cfg.seed,
              grad_clip=cfg.method.get("grad_clip", cfg.frame.grad_clip), num_workers=16, device=dev)
method = METHODS[cfg.method.name](cfg.method, frame)
modules = method.build_modules().to(dev)
for k, sd in ck["modules"].items():
    if k in modules:
        modules[k].load_state_dict(sd)
modules.train()
proj = modules["projector"]
print(f"[taps] {CKPT} epoch {ck['epoch']}", flush=True)

# W6: the one weight whose scale a BatchNorm does not cancel
W6 = proj[6].weight.detach().float()
U, S6, Vt = torch.linalg.svd(W6, full_matrices=False)
print(f"[taps] W6 {tuple(W6.shape)} |W6|_F {W6.norm():.1f}  singular values: "
      f"s1 {S6[0]:.3f}  s2 {S6[1]:.3f}  s5 {S6[4]:.3f}  s_med {S6.median():.3f}  "
      f"s_min {S6[-1]:.4f}  s1/s_med {S6[0]/S6.median():.1f}  "
      f"top16 energy share {(S6[:16].square().sum()/S6.square().sum()):.3f}", flush=True)
for nm, W in (("W0", proj[0].weight), ("W3", proj[3].weight)):
    s = torch.linalg.svdvals(W.detach().float())
    print(f"[taps] {nm} s1 {s[0]:.3f}  s_med {s.median():.3f}  s1/s_med {s[0]/s.median():.1f}", flush=True)
for nm, bn in (("BN1", proj[1]), ("BN2", proj[4])):
    g = bn.weight.detach().float().abs()
    print(f"[taps] {nm} gamma  median {g.median():.4f}  max {g.max():.4f}  min {g.min():.5f}", flush=True)

ds = method.build_train_dataset()
dl = DataLoader(ds, batch_size=cfg.bs, shuffle=True, drop_last=True, num_workers=16,
                pin_memory=True, persistent_workers=True)
d = method.cond_z.d_slice; eps = method.cond_z.eps
q = int(cfg.method.get("queue_steps", 0) or 0)

def pervar(a, N, V):
    """per-dimension variance of the per-image view-mean at this tap"""
    m = a.reshape(N, V, -1).mean(1).float()
    return float(m.var(0, unbiased=True).sum() / m.size(1)), m

rows, ring = [], []
with torch.no_grad():
    it = iter(dl)
    for s in range(STEPS):
        v = next(it); v = v.to(dev, non_blocking=True) if torch.is_tensor(v) else v[0].to(dev)
        N, V = v.shape[:2]
        cls = modules["backbone"].forward_features(v.flatten(0, 1))[:, 0]
        a = cls
        taps = {}
        taps["h"], _ = pervar(a, N, V)
        for i in range(7):
            a = proj[i](a)
            if i == 2: taps["bn1_relu"], _ = pervar(a, N, V)
            if i == 5: taps["bn2_relu"], _ = pervar(a, N, V)
        taps["z"], zm = pervar(a, N, V)
        # share of this block's z variance inside W6's top-16 left-singular directions
        zc = zm - zm.mean(0)
        top = float((zc @ U[:, :16]).square().sum() / zc.square().sum())
        ring = [zm] + ring[:q]
        x = torch.cat(ring)
        Q, _ = torch.linalg.qr(torch.randn(x.size(1), d, device=dev))
        p = x @ Q[:, :d]; mu = p.mean(0); pc = p - mu
        cov = pc.T @ pc / (p.size(0) - 1) + eps * torch.eye(d, device=dev)
        kl = float(0.5 * (cov.diagonal().sum() + mu.square().sum() - d
                          - torch.linalg.eigvalsh(cov).clamp_min(1e-30).log().sum()) / d)
        rows.append(dict(step=s, kl=kl, blocks=len(ring), z_top16=top, **taps))
        if s % 50 == 0:
            print(f"  step {s:4d} kl {kl:.3f} | h {taps['h']:.4f} bn1 {taps['bn1_relu']:.4f} "
                  f"bn2 {taps['bn2_relu']:.4f} z {taps['z']:.4f}", flush=True)

import pandas as pd
df = pd.DataFrame(rows); df.to_csv(OUT, index=False)
full = df[df.blocks == q + 1]
hot = full[full.kl > 0.3]; cold = full[full.kl <= 0.3]
print(f"\n[taps] {len(hot)} spike steps vs {len(cold)} normal, per-dimension variance at each tap:")
print(f"{'tap':>10}{'normal':>10}{'spike':>10}{'ratio':>8}")
for t in ("h", "bn1_relu", "bn2_relu", "z"):
    a, b = cold[t].median(), hot[t].median()
    print(f"{t:>10}{a:>10.4f}{b:>10.4f}{b/a:>8.2f}")
print(f"{'z_top16':>10}{cold.z_top16.median():>10.3f}{hot.z_top16.median():>10.3f}"
      f"{hot.z_top16.median()/cold.z_top16.median():>8.2f}")
print(f"[taps] wrote {OUT}")
