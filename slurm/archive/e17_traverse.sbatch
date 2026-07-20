#!/bin/bash
#SBATCH --job-name=e17-traverse
#SBATCH --partition=defaultp
#SBATCH --cpus-per-task=16
#SBATCH --mem=48G
#SBATCH --time=00:40:00
#SBATCH --output=/nfs/scistore19/locatgrp/bdemirel/ssl_project/outputs/%x_%j.out
source /nfs/scistore19/locatgrp/bdemirel/ssl_project/slurm/_env.sh
cd /nfs/scistore19/locatgrp/bdemirel/ssl_project
python experiments/e17_traverse.py "$@"
