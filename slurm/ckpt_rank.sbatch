#!/bin/bash
#SBATCH --job-name=gap-ckptrank
#SBATCH --partition=gpu
#SBATCH --constraint=A100|L40S|A40
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=08:00:00
#SBATCH --output=/nfs/scistore19/locatgrp/bdemirel/ssl_project/outputs/%x_%j.out

# Effective rank at h per checkpoint vs the conditioner's rows-per-slice ratio. Example:
#   sbatch slurm/ckpt_rank.sbatch outputs/in1k.floorssl.s0.e27v6b100_ep25.pt ...
source /nfs/scistore19/locatgrp/bdemirel/ssl_project/slurm/_env.sh
cd /nfs/scistore19/locatgrp/bdemirel/ssl_project
python experiments/ckpt_rank.py "$@"
