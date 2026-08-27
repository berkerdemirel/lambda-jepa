#!/bin/bash
#SBATCH --job-name=gap-probe
#SBATCH --partition=gpu
#SBATCH --constraint=L40S|A40|A100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=/nfs/scistore19/locatgrp/bdemirel/ssl_project/outputs/%x_%j.out

# Frozen probes (linear_house_v1, linear_l2_v1, knn_v1) over stored features. Example:
#   sbatch slurm/probe.sbatch run_id=toy.lejepa-lamb002.ext
source /nfs/scistore19/locatgrp/bdemirel/ssl_project/slurm/_env.sh
cd /nfs/scistore19/locatgrp/bdemirel/ssl_project
python experiments/probe.py "$@"
