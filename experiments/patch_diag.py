"""Patch-token diagnostic on ADE20k (Berker 2026-09-07: why do we trail only on segmentation?). Same trunk loading and feature
path as the seg protocol (experiments/seg_visreg.py: forward_intermediates(indices=[-1], norm=True) at 512^2) + the CLS token;
metrics = sslgap/metrics/patch.py. One row per checkpoint appended to results/diag/patch_diag.csv.
Usage: python experiments/patch_diag.py --ckpt <path> --tag <tag> [--adapter native|pubvit] [--n-stats 200 --n-train 300 --n-test 200]"""
import argparse, os, sys, time
import numpy as np, pandas as pd, torch
from torch.utils.data import DataLoader, Subset

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT); sys.path.insert(0, os.path.join(ROOT, "experiments"))
from seg_visreg import ADE20K, load_trunk, IMG_SIZE, NUM_CLASSES  # noqa: E402
from sslgap.metrics.patch import patch_labels, patch_stats, ncm_views  # noqa: E402


def extract(trunk, ds, idx, device, bs=16):
    P, C, Y = [], [], []
    for x, m in DataLoader(Subset(ds, idx), batch_size=bs, num_workers=6):
        x = x.to(device)
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            _, inter = trunk.forward_intermediates(x, indices=[-1], norm=True, return_prefix_tokens=False)
            cls = trunk.forward_features(x)[:, 0]
        p = inter[0]
        if p.ndim == 4:
            p = p.flatten(2).transpose(1, 2)            # (B, L, C)
        P.append(p.float().cpu().numpy()); C.append(cls.float().cpu().numpy())
        Y.append(np.stack([patch_labels(mm.numpy()) for mm in m]))
    return np.concatenate(P), np.concatenate(C), np.concatenate(Y)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True); ap.add_argument("--tag", required=True); ap.add_argument("--adapter", default="native")
    ap.add_argument("--n-stats", type=int, default=200); ap.add_argument("--n-train", type=int, default=300); ap.add_argument("--n-test", type=int, default=200)
    ap.add_argument("--res", type=int, default=IMG_SIZE, help="input side; the seg protocol's 512, or 224 = the pretraining scale (a scale-sensitivity read)")
    ap.add_argument("--out", default=None)
    a = ap.parse_args(); t0 = time.time(); dev = torch.device("cuda")
    import seg_visreg; seg_visreg.IMG_SIZE = a.res            # the dataset reads the module constant
    trunk = load_trunk(a.ckpt, dev, a.adapter)
    val, train = ADE20K("val", False), ADE20K("train", False)      # both without augmentation (the val transform)
    rng = np.random.default_rng(0)
    vi = rng.choice(len(val), a.n_stats + a.n_test, replace=False); ti = rng.choice(len(train), a.n_train, replace=False)
    Pv, Cv, Yv = extract(trunk, val, vi, dev); Pt, Ct, Yt = extract(trunk, train, ti, dev)
    grid = int(Pv.shape[1] ** 0.5)
    row = {"tag": a.tag, "res": a.res, "n_stats": a.n_stats, "grid": grid, "d": Pv.shape[-1]}
    row.update(patch_stats(Pv[:a.n_stats], Cv[:a.n_stats], grid, rng))
    row.update(ncm_views(Pt, Yt, Pv[a.n_stats:], Yv[a.n_stats:], NUM_CLASSES, train_cls=Ct, test_cls=Cv[a.n_stats:]))
    row["seconds"] = round(time.time() - t0)
    out = a.out or f"{ROOT}/results/diag/patch_diag.csv"
    pd.DataFrame([row]).to_csv(out, mode="a", header=not os.path.exists(out), index=False)
    print("[patch-diag] " + " ".join(f"{k}={v:.4f}" if isinstance(v, float) else f"{k}={v}" for k, v in row.items()), flush=True)


if __name__ == "__main__":
    main()
