#!/bin/bash
#SBATCH --job-name=gap-e35tr
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=03:00:00
#SBATCH --output=/nfs/scistore19/locatgrp/bdemirel/ssl_project/outputs/%x_%j.out

# E35 follow-up: split h by the projector's own linear gain and probe each half.
# Stored features + the checkpoint's head only, no trunk forward.
#   sbatch slurm/e35_transmit.sbatch cell=in100.floorssl.s0.d256vm4
source /nfs/scistore19/locatgrp/bdemirel/ssl_project/slurm/_env.sh
cd /nfs/scistore19/locatgrp/bdemirel/ssl_project
python experiments/e35_transmit.py "$@"
