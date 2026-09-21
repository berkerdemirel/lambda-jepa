"""E12 (D-026) equal-pull λ measurement — protocol premise p3, D-021 lineage.

λ·‖∇_enc reg‖ = (1−λ)·‖∇_enc inv‖ at init ⟹ λ = g_inv / (g_inv + g_reg), measured on a real
first train batch at seed 0, AFTER embed_calib folds (mirrors the training_step order), under the
training autocast (bf16). Encoder = the shared parameters (trunk + emb Linear); the projector is
alignment-only and excluded. num_workers=0 (one-shot script: no throwaway worker pools — the
main-process aug RNG makes this batch not byte-identical to training's first batch; the measured
ratio is a magnitude calibration, not a reproduction). Launch rule (card): the full arm runs with
the measured value, no mid-flight discretion. Appends to results/diag/e12_equal_pull.csv.

    python experiments/equal_pull.py frame=in100_vits16 bs=128 num_classes=100 tag=e12a2 \
        method.lr=3e-4 +method.sigreg_at=embed +method.spec_norm=true +method.embed_calib=true
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

OUT = "/nfs/scistore19/locatgrp/bdemirel/ssl_project/results/diag/e12_equal_pull.csv"


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
    train_ds = method.build_train_dataset()
    g = torch.Generator().manual_seed(cfg.seed)
    loader = DataLoader(train_ds, batch_size=cfg.bs, shuffle=True, drop_last=True,
                        num_workers=0, generator=g)
    views, y = next(iter(loader))
    views = views.to(frame.device, non_blocking=True)
    y = y.to(frame.device, non_blocking=True)

    enc_params = [p for p in modules["encoder"].parameters() if p.requires_grad]
    with autocast(frame.device, dtype=torch.bfloat16):
        terms, _, _ = method.training_step(modules, views, frame.device, y=y)
    if cfg.method.get("h_reg"):                       # F-wave (D-027): measure the ADDITIVE h-term
        from sslgap.methods.lejepa import H_KEYS
        reg_key = H_KEYS[cfg.method.h_reg]
    else:
        reg_key = "moment_kl" if cfg.method.get("floor", "sigreg") == "sacreg" else "sigreg"

    def enc_norm(loss, retain):
        grads = torch.autograd.grad(loss, enc_params, retain_graph=retain, allow_unused=True)
        return torch.cat([gr.reshape(-1).float() for gr in grads if gr is not None]).norm().item()

    g_inv = enc_norm(terms["inv"], retain=True)
    g_reg = enc_norm(terms[reg_key], retain=False)
    lam = g_inv / (g_inv + g_reg)
    degen = " DEGENERATE(barrier-type term ~unbound at init: use the card's declared fallback)" \
        if (lam > 0.9 or g_reg < 0.01 * g_inv) else ""
    print(f"[equal_pull] tag={cfg.tag} reg={reg_key} g_inv={g_inv:.6g} g_reg={g_reg:.6g} "
          f"lambda={lam:.6g}  (fixed-0.02 distance: x{lam / 0.02:.2f}){degen}", flush=True)
    new = not os.path.exists(OUT)
    with open(OUT, "a", newline="") as f:
        w = csv.writer(f)
        if new:
            w.writerow(["tag", "reg", "g_inv", "g_reg", "lambda", "seed",
                        "inv_loss", "reg_loss", "slurm_job"])
        w.writerow([cfg.tag, reg_key, g_inv, g_reg, lam, cfg.seed,
                    terms["inv"].item(), terms[reg_key].item(),
                    os.environ.get("SLURM_JOB_ID", "")])


if __name__ == "__main__":
    main()
