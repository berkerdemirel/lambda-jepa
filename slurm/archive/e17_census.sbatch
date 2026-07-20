#!/bin/bash
#SBATCH --job-name=e17-census
#SBATCH --partition=defaultp
#SBATCH --cpus-per-task=16
#SBATCH --mem=48G
#SBATCH --time=00:40:00
#SBATCH --output=/nfs/scistore19/locatgrp/bdemirel/ssl_project/outputs/%x_%j.out

# Berker's full touch census (deg_pos/deg_neg over ALL candidates, probabilities + enrichment).
source /nfs/scistore19/locatgrp/bdemirel/ssl_project/slurm/_env.sh
cd /nfs/scistore19/locatgrp/bdemirel/ssl_project
python experiments/e17_touch_census.py
