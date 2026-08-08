#!/bin/bash
#SBATCH --job-name=e22-bench
#SBATCH --partition=gpu
#SBATCH --constraint=L40S|A40
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=24
#SBATCH --mem=96G
#SBATCH --time=00:50:00
#SBATCH --output=/nfs/scistore19/locatgrp/bdemirel/ssl_project/outputs/%x_%j.out

# E22 input-pipeline benchmark: loader grid + stage decomposition + GPU step time.
source /nfs/scistore19/locatgrp/bdemirel/ssl_project/slurm/_env.sh
cd /nfs/scistore19/locatgrp/bdemirel/ssl_project
python experiments/e22_bench.py
