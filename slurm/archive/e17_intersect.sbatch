#!/bin/bash
#SBATCH --job-name=e17-intersect
#SBATCH --partition=defaultp
#SBATCH --cpus-per-task=16
#SBATCH --mem=48G
#SBATCH --time=01:30:00
#SBATCH --output=/nfs/scistore19/locatgrp/bdemirel/ssl_project/outputs/%x_%j.out

# Cloud-intersection AUC vs nearest negatives (Berker's organization hypothesis, direct form).
source /nfs/scistore19/locatgrp/bdemirel/ssl_project/slurm/_env.sh
cd /nfs/scistore19/locatgrp/bdemirel/ssl_project
python experiments/e17_intersect.py
