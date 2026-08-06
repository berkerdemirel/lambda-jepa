#!/bin/bash
#SBATCH --job-name=gap-pubzoo-extract
#SBATCH --partition=gpu
#SBATCH --constraint=L40S|A40|A100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=10:00:00
#SBATCH --output=/nfs/scistore19/locatgrp/bdemirel/ssl_project/outputs/%x_%j.out

# E26 zoo wave 2: one model per job:
#   sbatch slurm/pubzoo_extract.sbatch dinob16
# Weights come from the HF cache primed by the selftest job; HF stays reachable for
# cold caches (D-076).
source /nfs/scistore19/locatgrp/bdemirel/ssl_project/slurm/_env.sh
export HF_HUB_OFFLINE=0
cd /nfs/scistore19/locatgrp/bdemirel/ssl_project
python experiments/pubzoo_extract.py "$@"
