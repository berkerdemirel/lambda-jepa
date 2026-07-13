#!/bin/bash
#SBATCH --job-name=e12g-ladder
#SBATCH --partition=gpu
#SBATCH --constraint=L40S|A40|A100|RTX3090Ti
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=05:00:00
#SBATCH --output=/nfs/scistore19/locatgrp/bdemirel/ssl_project/outputs/%x_%j.out

# Depth-ladder 3x3 grid: L-tap probes + diag-KL + figure rebuild. Example:
#   sbatch slurm/e12g_ladder.sbatch
source /nfs/scistore19/locatgrp/bdemirel/ssl_project/slurm/_env.sh
cd /nfs/scistore19/locatgrp/bdemirel/ssl_project
python experiments/e12g_ladder_fig.py
