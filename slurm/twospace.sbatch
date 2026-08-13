#!/bin/bash
#SBATCH --job-name=gap-twospace
#SBATCH --partition=defaultp
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=08:00:00
#SBATCH --output=/nfs/scistore19/locatgrp/bdemirel/ssl_project/outputs/%x_%j.out

# Two-space program estimators over stored V-view clouds (CPU float64; do_mlp=true
# wants a GPU — submit with --partition=gpu --gres=gpu:1 then). Example:
#   sbatch slurm/twospace.sbatch run_id=in1k.lejepa.s0.e27lej.extL
source /nfs/scistore19/locatgrp/bdemirel/ssl_project/slurm/_env.sh
cd /nfs/scistore19/locatgrp/bdemirel/ssl_project
python experiments/twospace.py "$@"
