#!/bin/bash
#SBATCH --job-name=gap-headlin
#SBATCH --partition=defaultp
#SBATCH --cpus-per-task=16
#SBATCH --mem=192G
#SBATCH --time=04:00:00
#SBATCH --output=/nfs/scistore19/locatgrp/bdemirel/ssl_project/outputs/%x_%j.out

# Linear accessibility h -> z (D-015 head-linearity index). Variants: centers | views | eval.
#   sbatch slurm/head_linearity.sbatch
#   sbatch slurm/head_linearity.sbatch --runs toy.floorssl.s0.e23cK3.extL
source /nfs/scistore19/locatgrp/bdemirel/ssl_project/slurm/_env.sh
cd /nfs/scistore19/locatgrp/bdemirel/ssl_project
python experiments/head_linearity.py "$@"
