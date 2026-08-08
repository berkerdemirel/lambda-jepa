#!/bin/bash
#SBATCH --job-name=e21-figs
#SBATCH --partition=defaultp
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=00:20:00
#SBATCH --output=/nfs/scistore19/locatgrp/bdemirel/ssl_project/outputs/%x_%j.out

# Figure-only re-render (readers over landed CSVs).
source /nfs/scistore19/locatgrp/bdemirel/ssl_project/slurm/_env.sh
cd /nfs/scistore19/locatgrp/bdemirel/ssl_project
set -e
python experiments/e21_metric_fig.py
python experiments/e21_quad_fig.py
echo FIGS-DONE
