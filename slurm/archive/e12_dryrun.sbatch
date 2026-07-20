#!/bin/bash
#SBATCH --job-name=e12-dryrun
#SBATCH --partition=defaultp
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --output=/nfs/scistore19/locatgrp/bdemirel/ssl_project/outputs/%x_%j.out

# E12 (D-026) CPU dry-run: step+backward+arch-rebuild round-trip for all 4 arms (D-022 precedent).
source /nfs/scistore19/locatgrp/bdemirel/ssl_project/slurm/_env.sh
cd /nfs/scistore19/locatgrp/bdemirel/ssl_project
python experiments/e12_dryrun.py
