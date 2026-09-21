"""WHERE does the displaced block mean first appear -- h, BN1+ReLU, BN2+ReLU, or z?
(Berker 2026-09-10, step 2 of the z-conditioner forensics.)

e27_zkl_taps.py showed no tap amplifies a spike block's own variance (all ratios 0.99-1.04) while
the ring's trace inflates 60%, which points at the scatter of the four block MEANS rather than any
block's internal variance. This keeps a 4-block ring at EVERY tap and splits the pooled trace the
same way at each:
    within_t  = mean_i tr(cov(B_i))          between_t = mean_i ||mu_i - mu||^2
Scales differ by three orders of magnitude between taps, so the comparable quantity is the RATIO
between/within, and the displacement of the guilty block relative to its siblings
(off_max/off_med). If the ratio is normal at h and BN1 and jumps at BN2+ReLU, the layer in front of
it (W3, s1/s_med = 31 at ep302) is implicated; if it is already large at h the trunk is handing the
projector an off-centre batch.

Usage: python experiments/e27_zkl_where.py <ckpt> <out csv> [steps]
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
modules.train(); proj = modules["projector"]
ds = method.build_train_dataset()
dl = DataLoader(ds, batch_size=cfg.bs, shuffle=True, drop_last=True, num_workers=16,
                pin_memory=True, persistent_workers=True)
d = method.cond_z.d_slice; eps = method.cond_z.eps; q = int(cfg.method.get("queue_steps", 0) or 0)
TAPS = ("h", "bn1_relu", "bn2_relu", "z")
rings = {t: [] for t in TAPS}; rows = []

def split(blocks):
    """within-block variance, between-block mean scatter, and the per-block displacements"""
    within = float(np.mean([float(b.var(0, unbiased=True).sum()) for b in blocks]))
    mus = torch.stack([b.mean(0) for b in blocks]); mu = torch.cat(blocks).mean(0)
    off = (mus - mu).square().sum(1)
    return within, float(off.mean()), off

with torch.no_grad():
    it = iter(dl)
    for s in range(STEPS):
        v = next(it); v = v.to(dev, non_blocking=True) if torch.is_tensor(v) else v[0].to(dev)
        N, V = v.shape[:2]
        cls = modules["backbone"].forward_features(v.flatten(0, 1))[:, 0]
        cur = {}; a = cls
        cur["h"] = a.reshape(N, V, -1).mean(1).float()
        for i in range(7):
            a = proj[i](a)
            if i == 2: cur["bn1_relu"] = a.reshape(N, V, -1).mean(1).float()
            if i == 5: cur["bn2_relu"] = a.reshape(N, V, -1).mean(1).float()
        cur["z"] = a.reshape(N, V, -1).mean(1).float()
        for t in TAPS: rings[t] = [cur[t]] + rings[t][:q]
        nb = len(rings["z"])
        # the conditioner's own KL on the z ring, to label spikes
        x = torch.cat(rings["z"]); Q, _ = torch.linalg.qr(torch.randn(x.size(1), d, device=dev))
        p = x @ Q[:, :d]; mu = p.mean(0); pc = p - mu
        cov = pc.T @ pc / (p.size(0) - 1) + eps * torch.eye(d, device=dev)
        kl = float(0.5 * (cov.diagonal().sum() + mu.square().sum() - d
                          - torch.linalg.eigvalsh(cov).clamp_min(1e-30).log().sum()) / d)
        r = dict(step=s, kl=kl, blocks=nb)
        offs = {}
        for t in TAPS:
            w, b, off = split(rings[t]); offs[t] = off
            r[f"{t}_within"], r[f"{t}_between"] = w, b
            r[f"{t}_ratio"] = b / w if w else float("nan")
            r[f"{t}_offmax_over_med"] = float(off.max() / off.median()) if nb > 2 else float("nan")
        # the guilty block is the one with the largest WITHIN-block variance at z: the
        # within/between split (e27_zkl_within_between.py) put 99% of the trace excursion there,
        # and 1% in the block means. Report its variance against its siblings' median at each tap.
        g = int(np.argmax([float(b.var(0, unbiased=True).sum()) for b in rings["z"]]))
        r["guilty"] = g
        for t in TAPS:                                  # is that same block the outlier upstream?
            tv = np.array([float(b.var(0, unbiased=True).sum()) for b in rings[t]])
            r[f"{t}_guilty_rank"] = int((tv > tv[g]).sum())             # 0 = it is the largest
            sib = np.median(np.delete(tv, g)) if len(tv) > 1 else np.nan
            r[f"{t}_guilty_over_sib"] = float(tv[g] / sib) if sib else float("nan")
        rows.append(r)
        if s % 50 == 0: print(f"  step {s:4d} kl {kl:.3f}", flush=True)

import pandas as pd
df = pd.DataFrame(rows); df.to_csv(OUT, index=False)
f = df[df.blocks == q + 1]; hot = f[f.kl > 0.3]; cold = f[f.kl <= 0.3]
print(f"\n[where] {len(hot)} spike vs {len(cold)} normal steps (medians)\n")
print(f"{'tap':>10}{'guilty var / siblings':>20}{'':>4}{'between/within':>15}{'':>4}{'guilty is #1':>14}")
print(f"{'':>10}{'normal':>8}{'spike':>12}{'':>4}{'normal':>7}{'spike':>8}{'':>4}{'spike':>14}")
for t in TAPS:
    print(f"{t:>10}{cold[f'{t}_guilty_over_sib'].median():>8.2f}{hot[f'{t}_guilty_over_sib'].median():>8.2f}{'':>4}"
          f"{cold[f'{t}_ratio'].median():>7.3f}{hot[f'{t}_ratio'].median():>8.3f}{'':>4}"
          f"{(hot[f'{t}_guilty_rank'] == 0).mean() * 100:>13.0f}%")
print(f"\ntrace split at z: normal within {cold.z_within.median():.2f} between {cold.z_between.median():.2f}"
      f"  |  spike within {hot.z_within.median():.2f} between {hot.z_between.median():.2f}")
print(f"[where] wrote {OUT}")
