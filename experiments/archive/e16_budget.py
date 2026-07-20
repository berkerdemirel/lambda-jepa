"""E16 (D-033) target-budget gate: measure the dense objective's information budget BEFORE any
training (the standing METHOD RULE born from E15's starvation death). Per train500/val image:
phi-sketch (E15 freeze) of frozen mae h.gap at the fixed 3x3 slot grid -> stores `phi.mean9`
(gist ceiling C_gist) and `phi.concat9` (position-aware range C_pos) under pseudo-run
`in100.mae.s0.e16grid`; plus the x4-lowres whole-image descriptor -> global-channel stats
(`outputs/e16_global_v1.npz`, seed 16) and `phi.global` features (C_glob). Variance decomposition
printed for the card. GPU job (~15 min H100); probes via the standard driver afterwards.
GATE: C_pos >= C_gist + .05 linear."""
import hashlib
import os

import numpy as np
import torch
from scipy.spatial.distance import pdist
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import v2

from sslgap.data import _TAIL, _Source, foveal_ctx, foveal_event, read_manifest, seed_worker
from sslgap.extract import FeatureStore

ROOT = "/nfs/scistore19/locatgrp/bdemirel/ssl_project"
RUN, F_, BLOCK, SEED = "in100.mae.s0.e16grid", 96, 192, 16
GRID = [(x, y) for y in (0, 64, 128) for x in (0, 64, 128)]
PREP = np.load(f"{ROOT}/outputs/e15_target_v1.npz")
GOUT = f"{ROOT}/outputs/e16_global_v1.npz"


class GridDataset(Dataset):
    def __init__(self, csv_path, source):
        self.items = read_manifest(csv_path)
        self.source = source
        self.base = v2.Compose([v2.Resize(224), v2.CenterCrop(224)])
        self.tail = v2.Compose(_TAIL)

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        ref, y = self.items[i]
        base = self.base(self.source(ref))
        crops = torch.stack([self.tail(foveal_ctx(base, q, F_)) for q in GRID])
        low = self.tail(foveal_event(base, None, None))
        return crops, low, y


def main():
    dev = "cuda"
    ck = torch.load(f"{ROOT}/outputs/in100.mae.s0_ep100.pt", map_location="cpu", weights_only=False)
    from sslgap.ckpt.adapters import _resolve
    spec = ck["arch"]["backbone"]
    trunk = _resolve(spec["class"])(**spec["kwargs"])
    trunk.load_state_dict(ck["modules"]["backbone"])
    trunk.to(dev).eval()
    mu = torch.tensor(PREP["mu"], device=dev)
    sd = torch.tensor(PREP["sd"], device=dev)
    W = torch.tensor(PREP["W"], device=dev)
    b = torch.tensor(PREP["b"], device=dev)
    scale = float(np.sqrt(2.0 / int(PREP["block"])))
    store = FeatureStore(f"{ROOT}/features")

    out = {}
    for man, sub in (("in100.train500.v1", "train"), ("in100.val.v1", "val")):
        ds = GridDataset(f"{ROOT}/features/manifests/{man}.csv",
                         _Source("imagefolder", root=os.path.expanduser(f"~/data/imagenet100/{sub}")))
        dl = DataLoader(ds, batch_size=64, num_workers=12, shuffle=False,
                        persistent_workers=True, worker_init_fn=seed_worker)
        phis, lows, ys = [], [], []
        with torch.inference_mode():
            for crops, low, y in dl:
                B, G = crops.shape[:2]
                with torch.autocast(dev, dtype=torch.bfloat16):
                    hc = trunk.forward_features(crops.flatten(0, 1).to(dev, non_blocking=True))[:, 1:].mean(1)
                    hl = trunk.forward_features(low.to(dev, non_blocking=True))[:, 1:].mean(1)
                z = (scale * torch.cos(((hc.float() - mu) / sd) @ W + b)).reshape(B, G, -1)
                phis.append(z.cpu().numpy())
                lows.append(hl.float().cpu().numpy())
                ys.append(y.numpy())
        Z = np.concatenate(phis)                        # [N, 9, 384]
        L = np.concatenate(lows)                        # [N, 384] raw lowres descriptors
        yy = np.concatenate(ys)
        store.put(RUN, man, "phi.mean9", Z.mean(1))
        store.put(RUN, man, "phi.concat9", Z.reshape(len(Z), -1))
        store.put_labels(RUN, man, yy)
        out[man] = (Z, L, yy)
        print(f"[e16budget] {man}: grid {Z.shape}, lowres {L.shape}", flush=True)

    # variance decomposition (train): total across (img,pos) vs within-image (position) share
    Z, Ltr, _ = out["in100.train500.v1"]
    tot = Z.reshape(-1, Z.shape[-1]).var(0).mean()
    within = Z.var(1).mean()
    print(f"[e16budget] Var(z) total={tot:.6f}  within-image(position)={within:.6f} "
          f"({within / tot:.1%} of total = the signal E15 averaged away)", flush=True)

    # global-channel stats (train lowres descriptors) + phi.global features
    rng = np.random.default_rng(SEED)
    mu_g, sd_g = Ltr.mean(0), Ltr.std(0).clip(1e-6)
    Ls = (Ltr - mu_g) / sd_g
    sub = rng.choice(len(Ls), size=2048, replace=False)
    sigma_g = float(np.median(pdist(Ls[sub].astype(np.float64))))
    Wg = np.concatenate([rng.standard_normal((384, BLOCK)) / (m * sigma_g) for m in (1.0, 2.0)], 1)
    bg = rng.uniform(0, 2 * np.pi, 2 * BLOCK)
    np.savez(GOUT, mu=mu_g.astype(np.float32), sd=sd_g.astype(np.float32),
             W=Wg.astype(np.float32), b=bg.astype(np.float32), block=BLOCK,
             sigma_med=sigma_g, seed=SEED)
    print(f"[e16budget] global prep: sigma_med={sigma_g:.3f} -> {GOUT} "
          f"sha={hashlib.sha256(open(GOUT, 'rb').read()).hexdigest()[:16]}", flush=True)
    for man in ("in100.train500.v1", "in100.val.v1"):
        _, L, yy = out[man]
        pg = np.sqrt(2.0 / BLOCK) * np.cos(((L - mu_g) / sd_g) @ Wg + bg)
        store.put(RUN, man, "phi.global", pg)
        store.put_meta(RUN, man, {"kind": "eval", "n": len(yy),
                                  "spaces": ["phi.mean9", "phi.concat9", "phi.global"],
                                  "method": "pivot-budget", "frame": {"name": "in100"},
                                  "ckpt_provenance": {"source": "in100.mae.s0_ep100.pt",
                                                      "note": "D-033 budget gate", "grid": GRID}})
    print("[e16budget] done")


if __name__ == "__main__":
    main()
