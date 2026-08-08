#!/bin/bash
#SBATCH --job-name=gap-gridmet
#SBATCH --partition=defaultp
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=/nfs/scistore19/locatgrp/bdemirel/ssl_project/outputs/%x_%j.out

# The grid landing instrument (replaces e23/e23c/e24 grid_metrics wrappers). Examples:
#   sbatch slurm/grid_metrics.sbatch e23c_grid
#   sbatch slurm/grid_metrics.sbatch my_grid --tags s0.aa,s0.bb --lane toy.floorssl
source /nfs/scistore19/locatgrp/bdemirel/ssl_project/slurm/_env.sh
cd /nfs/scistore19/locatgrp/bdemirel/ssl_project
python experiments/grid_metrics.py "$@"
