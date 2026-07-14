"""Augmentation-orbit extraction for the orbit-cloud figures: Q seeded train images × K draws of
the FIXED audit stack (orbit_stack), features at h and z.final per method -> one npz per method
under features/orbits/. Tiny GPU job; images and draws are seeded so every method sees the SAME
orbit samples (paired by construction, like the extractor).

  sbatch slurm/orbits.sbatch
"""
import os

import hydra
import numpy as np
import torch
from omegaconf import DictConfig
from torch.amp import autocast

from sslgap.ckpt import adapters
from sslgap.data import _Source, orbit_stack, read_manifest, seed_everything
from sslgap.extract.extractor import _batch_spaces

H = {"simclr": "student.h.cls", "byol": "student.h.cls", "vicreg": "student.h.cls",  # D-036: projector input (was GAP)
     "dino": "teacher.h.cls", "mae": "student.h.gap", "ijepa": "teacher.h.gap",
     "lejepa": "student.z.embed"}
Z = {"simclr": "student.z.proj.out", "byol": "student.z.pred.out",
     "vicreg": "student.z.proj.out", "dino": "teacher.z.dino.bottleneck",
     "mae": "student.z.dec.tap8", "ijepa": "student.z.pred.out", "lejepa": "student.z.proj.out"}


@hydra.main(version_base=None, config_path="configs", config_name="orbits")
def main(cfg: DictConfig):
    src = _Source("hf-imagenette", split="train")
    items = read_manifest(os.path.join(os.path.expanduser(cfg.manifest_dir),
                                       cfg.train_manifest + ".csv"))
    rng = np.random.default_rng(cfg.seed)
    q_idx = rng.choice(len(items), cfg.n_images, replace=False)
    tfm = orbit_stack(cfg.img_size)
    seed_everything(cfg.seed)                                     # aug draws: same for all methods
    views, img_ids, ys = [], [], []
    for qi in q_idx:
        ref, y = items[qi]
        pil = src(ref).convert("RGB")
        for _ in range(cfg.n_augs):
            views.append(tfm(pil))
            img_ids.append(qi)
            ys.append(y)
    x = torch.stack(views)                                        # [Q*K, C, H, W]
    out_dir = os.path.expanduser(cfg.out_dir)
    os.makedirs(out_dir, exist_ok=True)

    overrides = cfg.get("run_id_overrides") or {}
    for m in cfg.methods:
        ckpt = os.path.expanduser(overrides.get(m) or cfg.ckpt_pattern.format(method=m))
        loaded = adapters.load("native", ckpt, run_id=f"orbits.{m}").eval_(cfg.device)
        feats = {}
        with torch.inference_mode(), autocast(cfg.device, dtype=torch.bfloat16):
            for i in range(0, len(x), cfg.bs):
                sp = _batch_spaces(loaded, x[i:i + cfg.bs].to(cfg.device), h_layers=())
                for key, name in (("h", H[m]), ("z", Z[m])):
                    feats.setdefault(key, []).append(sp[name].float().cpu().numpy())
        np.savez(os.path.join(out_dir, f"toy.{m}.s0.npz"),
                 h=np.concatenate(feats["h"]), z=np.concatenate(feats["z"]),
                 img_id=np.array(img_ids), y=np.array(ys),
                 h_space=H[m], z_space=Z[m], ckpt=ckpt)
        print(f"[orbits] {m}: {len(img_ids)} orbit points ({cfg.n_images}x{cfg.n_augs})")
    print("[orbits] done")


if __name__ == "__main__":
    main()
