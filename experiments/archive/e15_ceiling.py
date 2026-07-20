"""E15 (D-032) capped-distillation ceiling C: the position-marginal training target as a
representation. For every train500/val image, phi-sketch (e15_target_v1.npz) of frozen mae h.gap
on K=2 ref-seeded UNIFORM ctx draws, averaged -> pseudo-run `in100.mae.s0.e15phi`, space
`phi.marginal`; the standard probe driver then yields C = linear_raw_v2 / knn on the target
itself. GPU job (~5 min H100): sbatch wrapper reuses train env.

  sbatch --partition=gpu100 --constraint=H100 --job-name=h100-slotA --dependency=singleton \
      slurm/e15_ceiling.sbatch
"""
import hashlib
import os

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import v2

from sslgap.data import _TAIL, _Source, foveal_ctx, read_manifest, seed_worker
from sslgap.extract import FeatureStore

ROOT = "/nfs/scistore19/locatgrp/bdemirel/ssl_project"
RUN, SPACE, F_ = "in100.mae.s0.e15phi", "phi.marginal", 96
PREP = f"{ROOT}/outputs/e15_target_v1.npz"
SLOTS = list(range(0, 224 - F_ + 1, 16))


class CtxDrawsDataset(Dataset):
    """K=2 uniform ctx positions per image, md5(ref)-seeded (deterministic, uniform over slots —
    NOT the E14 strata-coupled boxes)."""

    def __init__(self, csv_path, source):
        self.items = read_manifest(csv_path)
        self.source = source
        self.base = v2.Compose([v2.Resize(224), v2.CenterCrop(224)])
        self.tail = v2.Compose(_TAIL)

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        ref, y = self.items[i]
        rng = np.random.default_rng(int.from_bytes(
            hashlib.md5(f"e15ceil:{ref}".encode()).digest()[:8], "little"))
        base = self.base(self.source(ref))
        crops = [self.tail(foveal_ctx(base, (int(rng.choice(SLOTS)), int(rng.choice(SLOTS))), F_))
                 for _ in range(2)]
        return crops[0], crops[1], y


def main():
    dev = "cuda"
    ck = torch.load(f"{ROOT}/outputs/in100.mae.s0_ep100.pt", map_location="cpu", weights_only=False)
    from sslgap.ckpt.adapters import _resolve
    spec = ck["arch"]["backbone"]
    trunk = _resolve(spec["class"])(**spec["kwargs"])
    trunk.load_state_dict(ck["modules"]["backbone"])
    trunk.to(dev).eval()
    d = np.load(PREP)
    mu, sd = torch.tensor(d["mu"], device=dev), torch.tensor(d["sd"], device=dev)
    W, b = torch.tensor(d["W"], device=dev), torch.tensor(d["b"], device=dev)
    scale = float(np.sqrt(2.0 / int(d["block"])))
    store = FeatureStore(f"{ROOT}/features")

    for man, sub in (("in100.train500.v1", "train"), ("in100.val.v1", "val")):
        ds = CtxDrawsDataset(f"{ROOT}/features/manifests/{man}.csv",
                             _Source("imagefolder", root=os.path.expanduser(f"~/data/imagenet100/{sub}")))
        dl = DataLoader(ds, batch_size=256, num_workers=12, shuffle=False,
                        persistent_workers=True, worker_init_fn=seed_worker)
        feats, ys = [], []
        with torch.inference_mode():
            for c1, c2, y in dl:
                zs = []
                for c in (c1, c2):
                    with torch.autocast(dev, dtype=torch.bfloat16):
                        h = trunk.forward_features(c.to(dev, non_blocking=True))[:, 1:].mean(1)
                    z = scale * torch.cos(((h.float() - mu) / sd) @ W + b)
                    zs.append(z)
                feats.append(((zs[0] + zs[1]) / 2).cpu().numpy())
                ys.append(y.numpy())
        X, y_all = np.concatenate(feats), np.concatenate(ys)
        store.put(RUN, man, SPACE, X)
        store.put_labels(RUN, man, y_all)
        store.put_meta(RUN, man, {"kind": "eval", "n": len(y_all), "spaces": [SPACE],
                                  "method": "pivot-ceiling", "frame": {"name": "in100"},
                                  "ckpt_provenance": {"source": "in100.mae.s0_ep100.pt",
                                                      "prep": PREP, "draws": 2, "note": "D-032 C"}})
        print(f"[e15ceil] {man}: {X.shape}", flush=True)
    print("[e15ceil] done")


if __name__ == "__main__":
    main()
