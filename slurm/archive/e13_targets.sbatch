#!/bin/bash
#SBATCH --job-name=e13-targets
#SBATCH --partition=defaultp
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=04:00:00
#SBATCH --output=/nfs/scistore19/locatgrp/bdemirel/ssl_project/outputs/%x_%j.out

# Every zoo method as D_read target -> 7 scatters vs linear_raw_v2 (CPU float64). Example:
#   sbatch slurm/e13_targets.sbatch
source /nfs/scistore19/locatgrp/bdemirel/ssl_project/slurm/_env.sh
cd /nfs/scistore19/locatgrp/bdemirel/ssl_project
python experiments/e13_all_targets.py
