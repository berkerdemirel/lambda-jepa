#!/bin/bash
#SBATCH --job-name=gap-transfer
#SBATCH --partition=gpu
#SBATCH --constraint=L40S|A40|A100|RTX3090Ti
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH --output=/nfs/scistore19/locatgrp/bdemirel/ssl_project/outputs/%x_%j.out

# E25 ssl-transfer linear benchmark, one dataset per job:
#   sbatch slurm/transfer.sbatch cifar10 [ckpt] [out_csv]
# Dataset downloads happen in-job (compute nodes have internet, D-076).
source /nfs/scistore19/locatgrp/bdemirel/ssl_project/slurm/_env.sh
export HF_HUB_OFFLINE=0          # cars rides an HF mirror (E25 deviation 3); one-time fetch, then cached
export HF_DATASETS_OFFLINE=0
cd /nfs/scistore19/locatgrp/bdemirel/ssl_project
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
python experiments/transfer_probe.py "$@"
