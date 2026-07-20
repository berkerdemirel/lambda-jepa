#!/bin/bash
#SBATCH --job-name=gap-train
#SBATCH --partition=gpu
#SBATCH --constraint=L40S|A40|A100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=08:00:00
#SBATCH --requeue
#SBATCH --output=/nfs/scistore19/locatgrp/bdemirel/ssl_project/outputs/%x_%j.out

# Frame trainer (resume-safe: --requeue restarts from _last.pt). Examples:
#   sbatch slurm/train.sbatch method=lejepa frame=toy_vits8                       # frame-standard
#   sbatch slurm/train.sbatch method=lejepa frame=toy_vits8 frame.epochs=800 tag=portval
source /nfs/scistore19/locatgrp/bdemirel/ssl_project/slurm/_env.sh
cd /nfs/scistore19/locatgrp/bdemirel/ssl_project
python experiments/train.py "$@"
