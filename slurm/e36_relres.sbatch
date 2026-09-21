#!/bin/bash
#SBATCH --job-name=gap-e36
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=08:00:00
#SBATCH --output=/nfs/scistore19/locatgrp/bdemirel/ssl_project/outputs/%x_%j.out

# E36 (docs/experiments/E36_invariant_vs_residual.md): the whitened shared-anchor residual between
# the treated and untreated cells, and the view-invariant / view-residual decomposition of where
# the downstream gain lives. Stored features only, no trunk forward; the view store is the size
# dial (126,689 x 8 x 384 fp16 = 778 MB per cell).
#   sbatch slurm/e36_relres.sbatch
#   sbatch slurm/e36_relres.sbatch 'stages=[preflight]'
source /nfs/scistore19/locatgrp/bdemirel/ssl_project/slurm/_env.sh
cd /nfs/scistore19/locatgrp/bdemirel/ssl_project
python experiments/e36_relres.py "$@"
