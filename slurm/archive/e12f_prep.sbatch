#!/bin/bash
#SBATCH --job-name=e12f-prep
#SBATCH --partition=gpu
#SBATCH --constraint=L40S|A40|A100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=00:50:00
#SBATCH --output=/nfs/scistore19/locatgrp/bdemirel/ssl_project/outputs/%x_%j.out

# E12 F-wave (D-027) prep: equal-pull λ for the new h-terms. f3 (spec_floor) is expected
# DEGENERATE at init (barrier unbound) → card fallback 0.4775; f4/f5 use measured values.
source /nfs/scistore19/locatgrp/bdemirel/ssl_project/slurm/_env.sh
cd /nfs/scistore19/locatgrp/bdemirel/ssl_project
BASE="frame=in100_vits16 bs=128 num_classes=100 method.lr=3e-4 method.warmup_ep=10
      method.eta_min=1e-5 method.grad_clip=1.0 method.lamb=0.02
      +method.spec_norm=true +method.embed_calib=true +method.h_lamb=1.0"
python experiments/equal_pull.py $BASE tag=e12f4 +method.h_reg=sigreg_std
python experiments/equal_pull.py $BASE tag=e12f5 +method.h_reg=moment_diag
python experiments/equal_pull.py $BASE tag=e12f3 +method.h_reg=spec_floor
