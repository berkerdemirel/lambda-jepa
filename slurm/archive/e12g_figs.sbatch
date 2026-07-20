#!/bin/bash
#SBATCH --job-name=e12g-figs
#SBATCH --partition=defaultp
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=/nfs/scistore19/locatgrp/bdemirel/ssl_project/outputs/%x_%j.out

# h-vs-z metric/invariance/probe figures for the calibration pairs (CPU float64). Example:
#   sbatch slurm/e12g_figs.sbatch
source /nfs/scistore19/locatgrp/bdemirel/ssl_project/slurm/_env.sh
cd /nfs/scistore19/locatgrp/bdemirel/ssl_project
python experiments/e12g_hz_figs.py
