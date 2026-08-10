#!/bin/bash
#SBATCH --job-name=e27lejl
#SBATCH --partition=gpu100
#SBATCH --constraint=H100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=28
#SBATCH --mem=128G
#SBATCH --time=08:00:00
#SBATCH --requeue
#SBATCH --output=/nfs/scistore19/locatgrp/bdemirel/ssl_project/outputs/%x_%j.out

# E27 §(j.9)/D-092: the Lightly-benchmark LeJEPA replication at ViT-S/16 (Berker 2026-08-10:
# "change and relaunch the lejepa so that we perfectly match (at least the lightly case)
# reproduction … out dim … 64 … match n_slices … nof locals, do 6 … photometrics … bs 512
# we will keep, i want lightly numbers"). Their recipe verbatim: DINOTransform views
# 2g@224 (0.3,1) + 6l@96 (0.05,0.3), solarize on global-2 only; loss = .05·SIGReg(locals,
# 1024 slices) + .95·MSE(locals → globals-mean, no stop-grad); proj 384→[2048,2048,64] BN
# (NO embed stage); AdamW 5e-4 flat at GLOBAL bs 512, wd 5e-2, warmup 1 ep from .01×,
# cosine to .001×lr; no grad clip, no EMA. Single-GPU bs=512 preserves their loss
# semantics exactly (their SIGReg/BN all-reduce over the global batch ≡ one 512-batch);
# grad_ckpt for the fit (e24v3nq bs-512 precedent). Declared deviations (card §(j.9)):
# bf16 vs their fp16-mixed; house k9 blur approximating their radius-sampled PIL blur;
# seed 0 vs their unseeded default; house online monitor (CLS, all views). Selftest heads
# every segment; resume via _last.pt.
source /nfs/scistore19/locatgrp/bdemirel/ssl_project/slurm/_env.sh
cd /nfs/scistore19/locatgrp/bdemirel/ssl_project
python experiments/e27_selftest_lejepa.py || exit 1
python experiments/train.py method=lejepa frame=in1k_vits16 bs=512 num_classes=1000 \
  method.lamb=0.05 +method.aug=lightly_mc +method.mc_form=lightly method.proj_dim=64 \
  method.emb_dim=0 +method.n_slices=1024 method.Vl=6 +method.grad_ckpt=true \
  method.lr=5e-4 method.eta_min=5e-7 '+extra_cadence=[10]' tag=e27lejl "$@"
