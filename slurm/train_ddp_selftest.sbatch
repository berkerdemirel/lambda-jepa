#!/bin/bash
#SBATCH --job-name=ddp-selftest
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --output=/nfs/scistore19/locatgrp/bdemirel/ssl_project/outputs/%x_%j.out

# W=1 faithfulness gate for the DDP add-on (experiments/train_ddp_selftest.py).
source /nfs/scistore19/locatgrp/bdemirel/ssl_project/slurm/_env.sh
cd /nfs/scistore19/locatgrp/bdemirel/ssl_project
echo "node: $(hostname)"
.venv/bin/python experiments/train_ddp_selftest.py
