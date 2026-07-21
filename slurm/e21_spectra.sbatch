#!/bin/bash
#SBATCH --job-name=e21-spectra
#SBATCH --partition=defaultp
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=00:20:00
#SBATCH --output=/nfs/scistore19/locatgrp/bdemirel/ssl_project/outputs/%x_%j.out

# Spectrum decomposition figure (reads feature stores; CPU eigendecomp d<=384).
source /nfs/scistore19/locatgrp/bdemirel/ssl_project/slurm/_env.sh
cd /nfs/scistore19/locatgrp/bdemirel/ssl_project
set -e
python experiments/e21_spectra_fig.py
echo FIGS-DONE
