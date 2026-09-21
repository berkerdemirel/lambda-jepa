"""E33 rich-vs-lazy diagnostic: does training keep moving features/kernels (rich) or freeze
them near init (lazy), and does the h floor change that?

Per checkpoint vs the run's TRUE init — from_native(random_init, seed = the run's training
seed), verified bit-exact against the trainer's seed_everything→build_modules path for all
four E33 cells (2026-08-31); both members of a pair share arch+seed, hence ONE common init:
- per-layer linear CKA of h.cls / h.gap on a fixed val subset (n_cka, seeded permutation);
- empirical-NTK alignment on the nested first n_ntk of the same permutation.
Model-in-the-loop by necessity (NTK grads; per-layer taps at epochs never extracted); writes
tidy CSV + a kernels .npz only — no feature stores (D-005 untouched).

  sbatch slurm/feature_drift.sbatch cell=in1k.floorssl.s0.e27v6b100 'epochs=[10,25,50,75,100]'
"""
import csv
import os

import hydra
import numpy as np
import torch
from omegaconf import DictConfig
from torch.amp import autocast
from torch.utils.data import DataLoader, Subset

from sslgap.ckpt import adapters
from sslgap.data import EvalDataset, _Source
from sslgap.extract.ntk import empirical_ntk
from sslgap.metrics.cka import frob_alignment, linear_cka
from sslgap.models.backbones import trunk_features
from sslgap.paths import DIAG, OUTPUTS
from sslgap.ckpt.schema import load_payload


def _val_subset(frame, manifest_dir, n_cka, seed):
    tag = {"imagenet100": "in100", "imagenet1k": "in1k"}[frame["dataset"]]
    root = os.path.expanduser(frame["data_root"])
    assert not root.rstrip("/").endswith("imagenet-100"), "imagenet-100 is the WRONG split (D-002)"
    ds = EvalDataset(os.path.join(manifest_dir, f"{tag}.val.v1.csv"),
                     _Source("imagefolder", root=os.path.join(root, "val")), frame["img_size"])
    idx = torch.randperm(len(ds), generator=torch.Generator().manual_seed(seed))[:n_cka].tolist()
    return ds, idx


def _layer_feats(trunk, ds, idx, h_layers, bs, workers, device):
    """{space: [n, D] fp32 numpy} for cls/gap at h_layers + final; extractor-parity bf16."""
    trunk = trunk.to(device).eval()
    acc = {}
    for x, _ in DataLoader(Subset(ds, idx), batch_size=bs, num_workers=workers, shuffle=False):
        with autocast(device, dtype=torch.bfloat16):
            f = trunk_features(trunk, x.to(device, non_blocking=True), h_layers=tuple(h_layers))
        for kind in ("cls", "gap"):
            acc.setdefault(f"h.{kind}", []).append(f[kind].float().cpu())
            for l in h_layers:
                acc.setdefault(f"h.{kind}.L{l:02d}", []).append(f[f"{kind}.L{l:02d}"].float().cpu())
    return {k: torch.cat(v).numpy() for k, v in acc.items()}


@hydra.main(version_base=None, config_path="configs", config_name="feature_drift")
def main(cfg: DictConfig):
    ref_path = os.path.join(cfg.ckpt_dir, f"{cfg.cell}_ep{max(cfg.epochs)}.pt")
    ck = load_payload(ref_path, map_location="cpu")
    assert ck["cfg"]["seed"] == cfg.init_seed, \
        f"init_seed={cfg.init_seed} is not the run's training seed {ck['cfg']['seed']}"

    init = adapters.load("native", ref_path, f"{cfg.cell}.e33init",
                         random_init=True, seed=cfg.init_seed)
    ds, idx = _val_subset(init.frame, cfg.manifest_dir, cfg.n_cka, cfg.subset_seed)
    ntk_idx = idx[:cfg.n_ntk]
    ntk_imgs = torch.stack([ds[i][0] for i in ntk_idx])
    print(f"[e33] {cfg.cell}: {init.frame['model_name']}@{init.frame['img_size']} "
          f"n_cka={len(idx)} n_ntk={len(ntk_idx)} probes={cfg.n_probes}", flush=True)

    rows = []          # cell, epoch, metric, space, value
    trunk0 = init.branches[init.probed_branch].trunk
    feats0 = _layer_feats(trunk0, ds, idx, cfg.h_layers, cfg.bs, cfg.num_workers, cfg.device)
    if cfg.init_check_store:   # house random-init null store of the same arch+seed, same rows
        stored = np.load(os.path.expanduser(cfg.init_check_store), mmap_mode="r")[idx].astype(np.float64)
        cos = (np.einsum("nd,nd->n", stored, feats0["h.cls"].astype(np.float64))
               / (np.linalg.norm(stored, axis=1) * np.linalg.norm(feats0["h.cls"], axis=1)))
        rows.append((cfg.cell, 0, "initcheck_cos_min", "h.cls", float(cos.min())))
        print(f"[e33] init-vs-null-store cos: min={cos.min():.6f} mean={cos.mean():.6f}", flush=True)
    K0 = empirical_ntk(trunk0, ntk_imgs, cfg.n_probes, cfg.probe_seed, cfg.device).numpy()
    rows.append((cfg.cell, 0, "ntk_fro", "h.cls", float(np.linalg.norm(K0))))
    kernels = {"ep000": K0}
    del trunk0, init

    K_prev = K0
    for ep in cfg.epochs:
        loaded = adapters.load("native", os.path.join(cfg.ckpt_dir, f"{cfg.cell}_ep{ep}.pt"),
                               f"{cfg.cell}.e33", random_init=False)
        trunk = loaded.branches[loaded.probed_branch].trunk
        feats = _layer_feats(trunk, ds, idx, cfg.h_layers, cfg.bs, cfg.num_workers, cfg.device)
        for space in feats:
            rows.append((cfg.cell, ep, "cka_init", space, linear_cka(feats0[space], feats[space])))
        K = empirical_ntk(trunk, ntk_imgs, cfg.n_probes, cfg.probe_seed, cfg.device).numpy()
        rows.append((cfg.cell, ep, "ntk_align_init", "h.cls", frob_alignment(K, K0)))
        rows.append((cfg.cell, ep, "ntk_cka_init", "h.cls", frob_alignment(K, K0, centered=True)))
        rows.append((cfg.cell, ep, "ntk_align_prev", "h.cls", frob_alignment(K, K_prev)))
        rows.append((cfg.cell, ep, "ntk_fro", "h.cls", float(np.linalg.norm(K))))
        kernels[f"ep{ep:03d}"] = K
        K_prev = K
        cka_final = [r for r in rows if r[1] == ep and r[2] == "cka_init" and r[3] == "h.cls"]
        print(f"[e33] {cfg.cell} ep{ep}: cka(h.cls)={cka_final[0][4]:.4f} "
              f"ntk_align={rows[-4][4]:.4f}", flush=True)
        del trunk, loaded

    kdir = OUTPUTS / "feature_drift"
    kdir.mkdir(exist_ok=True)
    np.savez(kdir / f"{cfg.cell}.ntk.npz", **kernels,
             ntk_idx=np.array(ntk_idx), probe_seed=cfg.probe_seed, n_probes=cfg.n_probes)
    out = DIAG / f"feature_drift.{cfg.cell}.csv"
    with open(out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["cell", "epoch", "metric", "space", "value",
                    "n_cka", "n_ntk", "n_probes", "subset_seed", "probe_seed"])
        for r in rows:
            w.writerow([*r, len(idx), len(ntk_idx), cfg.n_probes, cfg.subset_seed, cfg.probe_seed])
    print(f"[e33] done: {out} ({len(rows)} rows) + {kdir / (cfg.cell + '.ntk.npz')}", flush=True)


if __name__ == "__main__":
    main()
