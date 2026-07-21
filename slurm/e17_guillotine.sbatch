#!/bin/bash
#SBATCH --job-name=e17-guillotine
#SBATCH --partition=defaultp
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=00:20:00
#SBATCH --output=/nfs/scistore19/locatgrp/bdemirel/ssl_project/outputs/%x_%j.out

# Guillotine curves for the E17-program roster (reads probe CSVs only).
source /nfs/scistore19/locatgrp/bdemirel/ssl_project/slurm/_env.sh
cd /nfs/scistore19/locatgrp/bdemirel/ssl_project
set -e
python experiments/e17_guillotine.py
echo FIGS-DONE
