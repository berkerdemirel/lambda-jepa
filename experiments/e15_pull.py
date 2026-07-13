"""E15 (D-032) equal-pull-at-init measurement (D-021/D-026 lineage) + step dry-run: build the
pivot method exactly as the trainer would, run ONE real batch, and report per-term losses and
per-term grad norms over backbone params at init. lambda_view / lambda_t := |g_pred| / |g_term|
(recorded on the E15 card, then frozen into configs — no mid-flight discretion).

  python experiments/e15_pull.py --device cpu --bs 4          # CPU dry-run (step correctness)
  sbatch ... slurm/e15_pull.sbatch                            # H100 real measurement (bs 128)
"""
import argparse

import torch
from omegaconf import OmegaConf
from torch.utils.data import DataLoader

from sslgap.data import seed_everything, seed_worker
from sslgap.methods import METHODS
from sslgap.methods.base import Frame

ROOT = "/nfs/scistore19/locatgrp/bdemirel/ssl_project"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--bs", type=int, default=128)
    args = ap.parse_args()
    seed_everything(0)

    cfg = OmegaConf.load(f"{ROOT}/experiments/configs/method/pivot.yaml")
    cfg.global_channel = True                     # E16: measure all terms in one pass
    frame = Frame(name="in100", model_name="vit_small_patch16_224", img_size=224,
                  dataset="imagenet100", data_root="~/data/imagenet100", epochs=100, seed=0,
                  grad_clip=1.0, num_workers=4, device=args.device)
    method = METHODS["pivot"](cfg, frame)
    modules = method.build_modules().to(args.device)
    dl = DataLoader(method.build_train_dataset(), batch_size=args.bs, shuffle=True,
                    num_workers=2, worker_init_fn=seed_worker,
                    generator=torch.Generator().manual_seed(0))
    batch_x, y = next(iter(dl))
    batch_x = tuple(t.to(args.device) for t in batch_x)

    params = [p for p in modules["backbone"].parameters() if p.requires_grad]
    amp = torch.autocast(args.device, dtype=torch.bfloat16, enabled=args.device == "cuda")
    with amp:
        terms, feats, k = method.training_step(modules, batch_x, args.device, y=y)
    print({t: round(float(v), 6) for t, v in terms.items() if not t.startswith("loss")})
    assert torch.isfinite(terms["loss"]), "non-finite loss at init"
    assert feats.shape == (args.bs * k, 384), feats.shape

    norms = {}
    for name in ("pred", "view", "var", "transport", "global"):
        if name not in terms:
            continue
        for p in params:
            p.grad = None
        terms[name].backward(retain_graph=True)
        g = torch.norm(torch.stack([p.grad.norm() for p in params if p.grad is not None]))
        norms[name] = float(g)
    print("grad norms:", {n: round(v, 6) for n, v in norms.items()})
    for t in ("view", "transport", "global"):
        if norms.get(t, 0) > 0:
            print(f"lamb_{'t' if t == 'transport' else ('g' if t == 'global' else t)} (equal-pull) = "
                  f"{norms['pred'] / norms[t]:.4f}")
    if norms.get("var", 0) == 0:
        print("var floor inactive at init (as constructed) — lamb_var stays 1.0 on the hinge")


if __name__ == "__main__":
    main()
