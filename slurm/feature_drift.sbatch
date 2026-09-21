#!/bin/bash
#SBATCH --job-name=gap-e33drift
#SBATCH --partition=gpu
#SBATCH --constraint=L40S|A40|A100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=/nfs/scistore19/locatgrp/bdemirel/ssl_project/outputs/%x_%j.out

# E33 rich-vs-lazy diagnostic (CKA drift + empirical-NTK alignment vs true init). ViT-S
# cells fit anywhere (per-sample grads [n_ntk, 22M] bf16 ≈ 11 GB); a ViT-B cell would need
# A100-80 (≈ 44 GB) — pass --constraint=A100 on the CLI then. Example:
#   sbatch slurm/feature_drift.sbatch cell=in1k.floorssl.s0.e27v6b100 'epochs=[10,25,50,75,100]'
source /nfs/scistore19/locatgrp/bdemirel/ssl_project/slurm/_env.sh
cd /nfs/scistore19/locatgrp/bdemirel/ssl_project
python experiments/feature_drift.py "$@"
