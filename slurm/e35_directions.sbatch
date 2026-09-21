#!/bin/bash
#SBATCH --job-name=gap-e35dir
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=03:00:00
#SBATCH --output=/nfs/scistore19/locatgrp/bdemirel/ssl_project/outputs/%x_%j.out

# E35 (docs/experiments/E35_direction_ablation.md): delete h's directions in variance order and
# read the run's own loss (frozen projector, augmented views) against the probes. Stored features
# only + the checkpoint's head, no trunk forward: peak VRAM < 4 GB (the V=4 cloud store is
# 389 MB fp16), so no --constraint — any card in the gpu partition takes it.
#   sbatch slurm/e35_directions.sbatch
#   sbatch slurm/e35_directions.sbatch n_images=4096 'keep_grid=[384,128,16]' probes=[linear]
source /nfs/scistore19/locatgrp/bdemirel/ssl_project/slurm/_env.sh
cd /nfs/scistore19/locatgrp/bdemirel/ssl_project
python experiments/e35_directions.py "$@"
