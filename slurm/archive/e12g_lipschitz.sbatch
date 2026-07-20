#!/bin/bash
#SBATCH --job-name=e12g-lip
#SBATCH --partition=defaultp
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=02:00:00
#SBATCH --output=/nfs/scistore19/locatgrp/bdemirel/ssl_project/outputs/%x_%j.out

# Head-MLP Lipschitz instrument over stored orbit features + final ckpts (CPU). Example:
#   sbatch slurm/e12g_lipschitz.sbatch
source /nfs/scistore19/locatgrp/bdemirel/ssl_project/slurm/_env.sh
cd /nfs/scistore19/locatgrp/bdemirel/ssl_project
python experiments/e12g_head_lipschitz.py
