"""Is the growing tail of the z conditioner caused by one pathological 128-row block entering the
4-step ring? (Berker 2026-09-10.)

The conditioner (sslgap/methods/_common.py SACReg) is
    KL = 0.5 * ( tr(cov) + ||mu||^2 - d' - logdet(cov) ) / d',   cov = S + eps I,  eps = 1e-4
on a FRESH random d'-dim orthonormal slice per step. Its input is the ring
`[current, prev1, prev2, prev3]` of per-image view-means: 4 x bs = 512 rows for a d' = 128 slice,
so n/d' = 4 and (floor_shrink=null) there is no shrinkage. A block lives in the ring for exactly
four steps, which is the length of the observed spike runs.

Per step this records the three KL terms separately, the covariance spectrum, and — for every step —
the variance of each of the four blocks along the ring covariance's smallest eigenvector. For the
spike steps it also recomputes the covariance leaving out each block in turn, against a matched-n
control that drops the same number of rows spread evenly over all four blocks (leave-one-out alone
lowers n/d' from 4 to 3, which moves lambda_min by itself).

Usage: python experiments/e27_zkl_forensics.py <ckpt> <out csv> [steps]
"""
import sys, json
import numpy as np
import torch
from omegaconf import OmegaConf
from torch.utils.data import DataLoader

sys.path.insert(0, "/nfs/scistore19/locatgrp/bdemirel/ssl_project")
from sslgap.methods import METHODS
from sslgap.methods.base import Frame
from sslgap.ckpt.schema import load_payload

CKPT, OUT = sys.argv[1], sys.argv[2]
STEPS = int(sys.argv[3]) if len(sys.argv) > 3 else 200
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
modules.train()                      # BatchNorm on batch statistics, as in training
print(f"[forensics] {CKPT} epoch {ck['epoch']} step {ck['step']} | d'={method.cond_z.d_slice} "
      f"eps={method.cond_z.eps} shrink={method.cond_z.shrink} q={cfg.method.get('queue_steps')}", flush=True)

ds = method.build_train_dataset()
dl = DataLoader(ds, batch_size=cfg.bs, shuffle=True, drop_last=True, num_workers=16,
                pin_memory=True, persistent_workers=True)
q = int(cfg.method.get("queue_steps", 0) or 0)
d = method.cond_z.d_slice; eps = method.cond_z.eps

def terms(p):
    """The conditioner's own algebra, split into its three parts."""
    mu = p.mean(0); pc = p - mu
    S = pc.T @ pc / (p.size(0) - 1)
    cov = S + eps * torch.eye(d, device=p.device, dtype=p.dtype)
    ev = torch.linalg.eigvalsh(cov)
    logdet = ev.clamp_min(1e-30).log().sum()
    kl = 0.5 * (cov.diagonal().sum() + mu.square().sum() - d - logdet) / d
    return kl.item(), mu.square().sum().item(), cov.diagonal().sum().item(), logdet.item(), ev, cov

rows, ring, ring_ids = [], [], []
with torch.no_grad():
    it = iter(dl)
    for s in range(STEPS):
        views = next(it)
        views = views.to(dev, non_blocking=True) if torch.is_tensor(views) else views[0].to(dev)
        N, V = views.shape[:2]
        cls = modules["backbone"].forward_features(views.flatten(0, 1))[:, 0].reshape(N, V, -1)
        z = modules["projector"](cls.flatten(0, 1)).reshape(N, V, -1)
        blk = z.mean(1).float()                                   # per-image view-mean: bs rows
        ring = [blk] + ring[:q]; ring_ids = [s] + ring_ids[:q]
        x = torch.cat(ring)
        Q, _ = torch.linalg.qr(torch.randn(x.size(1), d, device=dev))
        p = x @ Q[:, :d]
        kl, mean_t, trace_t, logdet, ev, cov = terms(p)
        # variance of each block along the smallest eigenvector of the ring covariance
        vmin = torch.linalg.eigh(cov).eigenvectors[:, 0]
        nb = len(ring); bs = blk.size(0)
        blockvar = [p[i * bs:(i + 1) * bs] @ vmin for i in range(nb)]
        blockvar = [float(b.var(unbiased=True)) for b in blockvar]
        r = dict(step=s, kl=kl, mean_term=mean_t, trace_term=trace_t, logdet=logdet,
                 lmin=float(ev[0]), l2=float(ev[1]), l5=float(ev[4]), lmax=float(ev[-1]),
                 n=int(p.size(0)), blocks=nb, ring_ids=json.dumps(ring_ids),
                 blockvar=json.dumps([round(v, 5) for v in blockvar]))
        if kl > 0.3 and nb == q + 1:                              # spike, full ring
            for i in range(nb):                                   # leave one block out
                keep = torch.cat([p[j * bs:(j + 1) * bs] for j in range(nb) if j != i])
                r[f"lmin_drop{i}"], r[f"kl_drop{i}"] = terms(keep)[4][0].item(), terms(keep)[0]
            idx = torch.cat([torch.arange(i * bs, i * bs + bs - bs // nb) for i in range(nb)])
            r["lmin_ctrl"], r["kl_ctrl"] = terms(p[idx])[4][0].item(), terms(p[idx])[0]  # matched n
        rows.append(r)
        if s % 25 == 0:
            print(f"  step {s:4d} kl {kl:.4f} lmin {ev[0]:.4f} logdet {logdet:.1f} "
                  f"mean {mean_t:.3f} trace {trace_t:.1f}", flush=True)

import pandas as pd
pd.DataFrame(rows).to_csv(OUT, index=False)
df = pd.DataFrame(rows)
sp = df[df.kl > 0.3]
print(f"\n[forensics] {len(sp)}/{len(df)} steps with KL > 0.3")
if len(sp):
    print(f"  typical step: kl {df.kl.median():.4f} lmin {df.lmin.median():.4f} "
          f"logdet {df.logdet.median():.1f} mean {df.mean_term.median():.3f} trace {df.trace_term.median():.1f}")
    print(f"  spike steps : kl {sp.kl.median():.4f} lmin {sp.lmin.median():.4f} "
          f"logdet {sp.logdet.median():.1f} mean {sp.mean_term.median():.3f} trace {sp.trace_term.median():.1f}")
print(f"[forensics] wrote {OUT}")
