#!/bin/bash
#SBATCH --job-name=e22-bench2
#SBATCH --partition=defaultp
#SBATCH --cpus-per-task=32
#SBATCH --mem=64G
#SBATCH --time=00:40:00
#SBATCH --output=/nfs/scistore19/locatgrp/bdemirel/ssl_project/outputs/%x_%j.out

# E22 bench round 2: worker scaling 24/28/32 + the pre-resized-dataset lever (CPU-only).
source /nfs/scistore19/locatgrp/bdemirel/ssl_project/slurm/_env.sh
cd /nfs/scistore19/locatgrp/bdemirel/ssl_project
python experiments/e22_bench2.py
