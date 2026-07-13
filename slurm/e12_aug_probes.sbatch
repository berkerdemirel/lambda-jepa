#!/bin/bash
#SBATCH --job-name=e12-augpr
#SBATCH --partition=gpu
#SBATCH --constraint=L40S|A40|A100|RTX3090Ti
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH --output=/nfs/scistore19/locatgrp/bdemirel/ssl_project/outputs/%x_%j.out

# Offline probes on augmented-view features (monitor-vs-offline instrument check, all six
# calibration checkpoints). Example:
#   sbatch slurm/e12_aug_probes.sbatch
source /nfs/scistore19/locatgrp/bdemirel/ssl_project/slurm/_env.sh
cd /nfs/scistore19/locatgrp/bdemirel/ssl_project
python experiments/e12_aug_probes.py
