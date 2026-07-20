#!/bin/bash
#SBATCH --job-name=e17-reach
#SBATCH --partition=defaultp
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=/nfs/scistore19/locatgrp/bdemirel/ssl_project/outputs/%x_%j.out
# pilot: sbatch slurm/e17_reach.sbatch pilot [class]   full sweep: sbatch slurm/e17_reach.sbatch
source /nfs/scistore19/locatgrp/bdemirel/ssl_project/slurm/_env.sh
cd /nfs/scistore19/locatgrp/bdemirel/ssl_project
python experiments/e17_reach.py "$@"
