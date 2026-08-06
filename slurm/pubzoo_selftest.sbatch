#!/bin/bash
#SBATCH --job-name=gap-pubzoo-selftest
#SBATCH --partition=gpu
#SBATCH --constraint=L40S|A40|A100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=/nfs/scistore19/locatgrp/bdemirel/ssl_project/outputs/%x_%j.out

# D-076 zoo wave 1: fetch + adapter selftests (no extraction). Needs the network:
# the _env.sh offline default is overridden AFTER sourcing (dataset mode stays offline
# for HF datasets; model weights come from the HF hub / torch.hub).
source /nfs/scistore19/locatgrp/bdemirel/ssl_project/slurm/_env.sh
export HF_HUB_OFFLINE=0
cd /nfs/scistore19/locatgrp/bdemirel/ssl_project
python experiments/pubzoo_selftest.py
