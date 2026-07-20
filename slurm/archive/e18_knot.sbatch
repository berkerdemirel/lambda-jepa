#!/bin/bash
#SBATCH --job-name=e18-knot
#SBATCH --partition=gpu
#SBATCH --constraint=L40S|A40|A100|RTX3090Ti
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=01:30:00
#SBATCH --output=/nfs/scistore19/locatgrp/bdemirel/ssl_project/outputs/%x_%j.out

# Floor-KL decomposition + commensurable tap readouts at checkpoints (E19 curve question).
source /nfs/scistore19/locatgrp/bdemirel/ssl_project/slurm/_env.sh
cd /nfs/scistore19/locatgrp/bdemirel/ssl_project
python experiments/e18_knot_residual.py
