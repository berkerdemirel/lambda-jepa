"""Does the z conditioner's trace excursion live in the block MEANS rather than in any block's own
variance? (Berker 2026-09-10, following e27_zkl_taps.py.)

The conditioner pools 4 x 128 rows and subtracts one shared mean, so
    tr(S_pooled) ~= mean_i tr(S_i)            (within-block variance)
                  + mean_i ||mu_i - mu||^2    (scatter of the four block means)
A displaced block mean inflates the pooled trace while every block's own variance stays normal, and
adds variance in ONE direction -- which would leave lambda_min and logdet untouched, as observed.

Usage: python experiments/e27_zkl_within_between.py <ckpt> <out csv> [steps]
"""
import sys
import numpy as np, torch
from omegaconf import OmegaConf
from torch.utils.data import DataLoader
sys.path.insert(0, "/nfs/scistore19/locatgrp/bdemirel/ssl_project")
from sslgap.methods import METHODS
from sslgap.methods.base import Frame
from sslgap.ckpt.schema import load_payload

CKPT, OUT = sys.argv[1], sys.argv[2]
STEPS = int(sys.argv[3]) if len(sys.argv) > 3 else 250
dev = "cuda"
ck = load_payload(CKPT, map_location="cpu"); cfg = OmegaConf.create(ck["cfg"])
frame = Frame(name=cfg.frame.name, model_name=cfg.frame.model_name, img_size=cfg.frame.img_size,
              dataset=cfg.frame.dataset, data_root=cfg.frame.get("data_root"), epochs=cfg.frame.epochs,
              seed=cfg.seed, grad_clip=cfg.method.get("grad_clip", cfg.frame.grad_clip),
              num_workers=16, device=dev)
method = METHODS[cfg.method.name](cfg.method, frame); modules = method.build_modules().to(dev)
for k, sd in ck["modules"].items():
    if k in modules: modules[k].load_state_dict(sd)
modules.train()
ds = method.build_train_dataset()
dl = DataLoader(ds, batch_size=cfg.bs, shuffle=True, drop_last=True, num_workers=16,
                pin_memory=True, persistent_workers=True)
d = method.cond_z.d_slice; eps = method.cond_z.eps; q = int(cfg.method.get("queue_steps", 0) or 0)
rows, ring = [], []
with torch.no_grad():
    it = iter(dl)
    for s in range(STEPS):
        v = next(it); v = v.to(dev, non_blocking=True) if torch.is_tensor(v) else v[0].to(dev)
        N, V = v.shape[:2]
        cls = modules["backbone"].forward_features(v.flatten(0, 1))[:, 0]
        zm = modules["projector"](cls).reshape(N, V, -1).mean(1).float()
        ring = [zm] + ring[:q]
        x = torch.cat(ring); nb = len(ring)
        Q, _ = torch.linalg.qr(torch.randn(x.size(1), d, device=dev)); p = x @ Q[:, :d]
        mu = p.mean(0); pc = p - mu
        cov = pc.T @ pc / (p.size(0) - 1) + eps * torch.eye(d, device=dev)
        ev = torch.linalg.eigvalsh(cov)
        kl = float(0.5 * (cov.diagonal().sum() + mu.square().sum() - d - ev.clamp_min(1e-30).log().sum()) / d)
        bs = zm.size(0)
        blocks = [p[i * bs:(i + 1) * bs] for i in range(nb)]
        within = float(np.mean([float(b.var(0, unbiased=True).sum()) for b in blocks]))
        mus = torch.stack([b.mean(0) for b in blocks])
        between = float((mus - mu).square().sum(1).mean())
        offs = (mus - mu).square().sum(1)                      # per-block mean displacement
        rows.append(dict(step=s, kl=kl, blocks=nb, trace=float(cov.diagonal().sum()),
                         within=within, between=between, lmin=float(ev[0]),
                         off_max=float(offs.max()), off_min=float(offs.min()),
                         off_argmax=int(offs.argmax())))
        if s % 50 == 0:
            print(f"  step {s:4d} kl {kl:.3f} trace {float(cov.diagonal().sum()):.1f} "
                  f"within {within:.1f} between {between:.1f}", flush=True)
import pandas as pd
df = pd.DataFrame(rows); df.to_csv(OUT, index=False)
f = df[df.blocks == q + 1]; hot = f[f.kl > 0.3]; cold = f[f.kl <= 0.3]
print(f"\n[wb] {len(hot)} spike vs {len(cold)} normal steps (medians)")
print(f"{'':>12}{'normal':>10}{'spike':>10}{'delta':>10}")
for c in ("trace", "within", "between", "off_max", "off_min", "lmin"):
    a, b = cold[c].median(), hot[c].median()
    print(f"{c:>12}{a:>10.3f}{b:>10.3f}{b-a:>10.3f}")
print(f"\nshare of the trace excursion carried by the block-mean scatter: "
      f"{(hot.between.median()-cold.between.median())/(hot.trace.median()-cold.trace.median())*100:.0f}%")
print(f"[wb] wrote {OUT}")
