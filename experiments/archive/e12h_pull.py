"""H-wave (D-035) pull measurement — the p3/equal-pull convention generalized: per-term encoder
gradient norms for EVERY logged term on a real first train batch at seed 0, under the training
autocast, after any embed_calib fold (mirrors training_step order). Encoder = the shared trunk
(lejepa "encoder", dino "backbone"); norms are of the UNWEIGHTED terms — λ recommendations are
computed in-conversation from these raw pulls and recorded on the card before launch (no
mid-flight discretion once launched). Appends to results/diag/e12h_pull.csv.

Trajectory extension (E18/E19 cards, "dose matching is init-only"): `+ckpt=<path>` loads a saved
checkpoint's module weights (arch-asserted, extras restored) before the measurement — the same
first batch is then pulled AT that training state, giving per-term realized pressure along the
run. Encode the checkpoint in tag= (e.g. tag=floorssl.ep25); schema unchanged.

  sbatch slurm/e12h_pull.sbatch <train.py overrides> tag=<tag> [+ckpt=outputs/<run>_ep25.pt]
"""
import csv
import os

import hydra
import torch
from omegaconf import DictConfig
from torch.amp import autocast
from torch.utils.data import DataLoader

from sslgap.data import seed_everything
from sslgap.methods import METHODS
from sslgap.methods.base import Frame

OUT = "/nfs/scistore19/locatgrp/bdemirel/ssl_project/results/diag/e12h_pull.csv"


def to_dev(x, d):
    return tuple(to_dev(t, d) for t in x) if isinstance(x, (tuple, list)) \
        else x.to(d, non_blocking=True)


@hydra.main(version_base=None, config_path="configs", config_name="train")
def main(cfg: DictConfig):
    frame = Frame(name=cfg.frame.name, model_name=cfg.frame.model_name,
                  img_size=cfg.frame.img_size, dataset=cfg.frame.dataset,
                  data_root=cfg.frame.get("data_root"), epochs=cfg.frame.epochs,
                  seed=cfg.seed, grad_clip=cfg.method.get("grad_clip", cfg.frame.grad_clip),
                  num_workers=0, device=cfg.device)
    seed_everything(cfg.seed)
    method = METHODS[cfg.method.name](cfg.method, frame)
    modules = method.build_modules().to(frame.device)
    if cfg.get("ckpt"):
        pay = torch.load(cfg.ckpt, map_location=frame.device, weights_only=False)
        saved_arch = {k: v for k, v in pay["arch"].items() if k != "probe"}
        assert saved_arch == method.arch(), (
            f"pull-at-ckpt refused: {cfg.ckpt} arch != current method.arch()")
        for role, sd in pay["modules"].items():
            if role != "probe":
                modules[role].load_state_dict(sd)
        method.load_extras(pay.get("extras", {}))
        print(f"[pull] loaded {cfg.ckpt} (epoch {pay['epoch'] + 1})", flush=True)
    loader = DataLoader(method.build_train_dataset(), batch_size=cfg.bs, shuffle=True,
                        drop_last=True, num_workers=0,
                        generator=torch.Generator().manual_seed(cfg.seed))
    batch_x, y = next(iter(loader))
    batch_x, y = to_dev(batch_x, frame.device), y.to(frame.device, non_blocking=True)

    enc_key = "encoder" if "encoder" in modules else "backbone"
    enc_params = [p for p in modules[enc_key].parameters() if p.requires_grad]
    with autocast(frame.device, dtype=torch.bfloat16):
        terms, _, _ = method.training_step(modules, batch_x, frame.device, y=y)
    names = [k for k in terms if k != "loss"]

    rows = []
    for i, k in enumerate(names):
        grads = torch.autograd.grad(terms[k], enc_params, retain_graph=i < len(names) - 1,
                                    allow_unused=True)
        g = torch.cat([gr.reshape(-1).float() for gr in grads if gr is not None]).norm().item()
        rows.append({"tag": cfg.tag, "method": cfg.method.name, "term": k, "g_enc": g,
                     "loss_value": float(terms[k]), "seed": cfg.seed,
                     "slurm_job": os.environ.get("SLURM_JOB_ID", "")})
        print(f"[pull] tag={cfg.tag} term={k}: g_enc={g:.6g} loss={float(terms[k]):.6g}",
              flush=True)
    new = not os.path.exists(OUT)
    with open(OUT, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        if new:
            w.writeheader()
        w.writerows(rows)


if __name__ == "__main__":
    main()
