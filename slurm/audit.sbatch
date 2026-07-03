#!/bin/bash
#SBATCH --job-name=gap-audit
#SBATCH --partition=defaultp
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=08:00:00
#SBATCH --output=/nfs/scistore19/locatgrp/bdemirel/ssl_project/outputs/%x_%j.out

# Metric battery over stored features (CPU float64; no GPU needed). Example:
#   sbatch slurm/audit.sbatch run_id=toy.lejepa-lamb002.ext
source /nfs/scistore19/locatgrp/bdemirel/ssl_project/slurm/_env.sh
cd /nfs/scistore19/locatgrp/bdemirel/ssl_project
python experiments/audit.py "$@"
