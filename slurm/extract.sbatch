#!/bin/bash
#SBATCH --job-name=gap-extract
#SBATCH --partition=gpu
#SBATCH --constraint=L40S|A40|A100|RTX3090Ti
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=04:00:00
#SBATCH --output=/nfs/scistore19/locatgrp/bdemirel/ssl_project/outputs/%x_%j.out

# Feature extraction to the store. Example:
#   sbatch slurm/extract.sbatch ckpt=/nfs/.../lejepa/ckpt_lamb002.pt adapter=lejepa_minimal run_id=toy.lejepa-lamb002.ext
source /nfs/scistore19/locatgrp/bdemirel/ssl_project/slurm/_env.sh
cd /nfs/scistore19/locatgrp/bdemirel/ssl_project
python experiments/extract.py "$@"
