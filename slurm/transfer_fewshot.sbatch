#!/bin/bash
#SBATCH --job-name=gap-fewshot
#SBATCH --partition=gpu
#SBATCH --constraint=L40S|A40|A100|RTX3090Ti
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH --output=/nfs/scistore19/locatgrp/bdemirel/ssl_project/outputs/%x_%j.out

# E25 few-shot track (LeJEPA-style k-shot linear), all 8 datasets in one job:
#   sbatch slurm/transfer_fewshot.sbatch [dataset ...]
source /nfs/scistore19/locatgrp/bdemirel/ssl_project/slurm/_env.sh
export HF_HUB_OFFLINE=0
export HF_DATASETS_OFFLINE=0
cd /nfs/scistore19/locatgrp/bdemirel/ssl_project
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
python experiments/transfer_fewshot.py "$@"
