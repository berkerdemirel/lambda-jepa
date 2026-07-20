#!/bin/bash
#SBATCH --job-name=gap-viz
#SBATCH --partition=defaultp
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=/nfs/scistore19/locatgrp/bdemirel/ssl_project/outputs/%x_%j.out

# Results gallery: feature store + CSVs -> results/figures/*.png (+ wandb run m1-viz).
#   sbatch slurm/viz.sbatch                        # all panels
#   sbatch slurm/viz.sbatch 'panels=[pca,relrep]'
source /nfs/scistore19/locatgrp/bdemirel/ssl_project/slurm/_env.sh
cd /nfs/scistore19/locatgrp/bdemirel/ssl_project
python experiments/viz.py "$@"
