"""E27 §(m)/D-103(5): the mean-washing readout — does the view-mean satisfy the moment
target by AVERAGING rather than by the per-view distribution being right?

The jointly agreed hypothesis (Berker+Fable 2026-08-25, from the q2-vs-vm4 loss reads):
means over weakly-correlated locals Gaussianize by CLT, so the mc cells' z-moment
constraint self-satisfies (q2 mkl .03 from ep10, w-unreachable) while all-global means
(vm4 V=4, v10u V=10 — correlated views) stay structured and keep the channel live.
Falsification: if mc PER-VIEW moments are as close to target as the mean's, the quench is
genuine convergence, not washing.

Instrument (declared): exact-scatter moment KL to N(0,I) on a fixed seeded 128-d
orthonormal slice (n=512 images, n/d'=4 house rule; no ring/OAS — this is a readout, not
the training estimator; values are comparable WITHIN a run across anatomies, cross-run
levels carry scale). Per run x tap (z=proj.out, h=cls): mkl per view (globals/locals
split), mkl of the all-view mean, mkl of the globals-only mean, mean pairwise
centered-cosine between views (correlation proxy) + n_eff = V/(1+(V-1)r).

  python experiments/e27_meanwash.py            # -> results/compare/e27_meanwash.csv
"""
import os

import numpy as np
import pandas as pd
import torch
from torch.amp import autocast
from torch.utils.data import DataLoader

from sslgap.ckpt import adapters
from sslgap.data import LightlyLejepaMultiCropDataset, ViewsDataset
from sslgap.models.backbones import trunk_features

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
N_IMG, BS, D_SLICE = 512, 64, 128
RUNS = [  # (label, ckpt file) — ep100 finals + the mid-run q2 (ep~44, declared)
    ("vm4",   "in1k.floorssl.s0.d256vm4_ep100.pt"),
    ("v10u",  "in1k.floorssl.s0.e27v10u_ep100.pt"),
    ("lmc",   "in1k.floorssl.s0.e27lmc_ep100.pt"),
    ("lmcs5", "in1k.floorssl.s0.e27lmcs5_ep100.pt"),
    ("lmcse", "in1k.floorssl.s0.e27lmcse_ep100.pt"),
    ("sbe",   "in1k.floorssl.s0.e27lm4sbe_ep100.pt"),
    ("q2",    "in1k.floorssl.s0.e27lmcs5q2_last.pt"),
]


def slice_mat(d, device):
    g = torch.Generator().manual_seed(0)
    q, _ = torch.linalg.qr(torch.randn(d, D_SLICE, generator=g).double())
    return q.to(device)


def mkl(X, P):
    """Shape term of KL( N(m,S) || N(0,I) ) on the CENTERED batch: .5(tr S - d - logdet S).
    The raw-z mean offset (m'm/2, hundreds here) is a per-run scale artifact that swamps
    the covariance-shape contrast the washing hypothesis is about — centered KL isolates
    the shape; the wash question = mean-batch shape vs per-view shape WITHIN a run."""
    Y = (X.double() @ P)
    Y = Y - Y.mean(0)
    S = torch.cov(Y.T)
    sign, logdet = torch.slogdet(S)
    return 0.5 * (torch.trace(S) - D_SLICE - logdet).item()


def pair_cos(F):
    """F [V,N,D] -> mean pairwise per-image cosine of centered features."""
    Fc = F - F.mean(dim=1, keepdim=True)
    Fn = torch.nn.functional.normalize(Fc, dim=-1)
    V = F.shape[0]
    vals = [(Fn[i] * Fn[j]).sum(-1).mean().item()
            for i in range(V) for j in range(i + 1, V)]
    return float(np.mean(vals)) if vals else float("nan")


def build_views(cfg, frame):
    m = cfg["method"]
    if m.get("aug") == "lightly_mc":
        ds = LightlyLejepaMultiCropDataset(
            frame["dataset"], "train", img_size=frame["img_size"],
            data_root=frame.get("data_root"), n_g=m.get("Vg", 2), n_l=m.get("Vl", 6),
            local_size=m.get("local_size", 96),
            global_scale=tuple(m.get("global_scale", (0.3, 1.0))),
            local_scale=tuple(m.get("local_scale", (0.05, 0.3))))
        return ds, m.get("Vg", 2), m.get("Vl", 6)
    # aug="lejepa" = ViewsDataset's default orbit stack (the builder passes no aug)
    ds = ViewsDataset(frame["dataset"], "train", V=m.get("V", 4),
                      img_size=frame["img_size"], data_root=frame.get("data_root"))
    return ds, m.get("V", 4), 0


def main():
    device = "cuda"
    rows = []
    for label, fname in RUNS:
        torch.manual_seed(0)
        np.random.seed(0)
        lc = adapters.load("native", os.path.join(ROOT, "outputs", fname), label)
        br = lc.branches[lc.probed_branch]
        trunk, head = br.trunk.to(device).eval(), br.heads.to(device).eval()
        ds, n_g, n_l = build_views(lc.cfg, lc.frame)
        dl = DataLoader(ds, batch_size=BS, shuffle=False, num_workers=8,
                        persistent_workers=False, drop_last=True)
        hs, zs = [], []                        # per-batch lists of [V,b,D]
        with torch.no_grad(), autocast("cuda", dtype=torch.bfloat16):
            for bi, (x, _) in enumerate(dl):
                if bi * BS >= N_IMG:
                    break
                views = list(x[0].transpose(0, 1)) + list(x[1].transpose(0, 1)) \
                    if isinstance(x, (tuple, list)) else list(x.transpose(0, 1))
                hb, zb = [], []
                for v in views:
                    h = trunk_features(trunk, v.to(device, non_blocking=True))["cls"].float()
                    hb.append(h)
                    zb.append(head(h)["proj.out"].float())
                hs.append(torch.stack(hb))
                zs.append(torch.stack(zb))
        H = torch.cat(hs, dim=1)               # [V, N, Dh]
        Z = torch.cat(zs, dim=1)
        V = Z.shape[0]
        ep = lc.provenance.get("epoch")
        for tap, F in (("z", Z), ("h", H)):
            P = slice_mat(F.shape[-1], device)
            per_view = [mkl(F[v], P) for v in range(V)]
            g_idx = list(range(n_g)) if n_l else list(range(V))
            l_idx = list(range(n_g, V)) if n_l else []
            row = {"run": label, "epoch": ep, "tap": tap, "V": V, "n_g": n_g, "n_l": n_l,
                   "mkl_perview_g": float(np.mean([per_view[i] for i in g_idx])),
                   "mkl_perview_l": float(np.mean([per_view[i] for i in l_idx])) if l_idx
                   else float("nan"),
                   "mkl_allmean": mkl(F.mean(0), P),
                   "mkl_gmean": mkl(F[g_idx].mean(0), P),
                   "r_pair_all": pair_cos(F),
                   "r_pair_gg": pair_cos(F[g_idx]),
                   "r_pair_ll": pair_cos(F[l_idx]) if len(l_idx) > 1 else float("nan")}
            row["wash_ratio"] = row["mkl_allmean"] / (
                (row["mkl_perview_g"] + (row["mkl_perview_l"] if l_idx else
                                         row["mkl_perview_g"])) / 2)
            row["n_eff"] = V / (1 + (V - 1) * max(row["r_pair_all"], 0.0))
            rows.append(row)
            print(f"[meanwash] {label} {tap}: per-view g {row['mkl_perview_g']:.3f} "
                  f"l {row['mkl_perview_l']:.3f} | mean(all) {row['mkl_allmean']:.3f} "
                  f"mean(g) {row['mkl_gmean']:.3f} | r_all {row['r_pair_all']:.3f} "
                  f"n_eff {row['n_eff']:.2f}", flush=True)
    out = os.path.join(ROOT, "results/compare/e27_meanwash.csv")
    pd.DataFrame(rows).to_csv(out, index=False)
    print(f"[meanwash] -> {out}", flush=True)


if __name__ == "__main__":
    main()
