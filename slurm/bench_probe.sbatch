#!/bin/bash
#SBATCH --job-name=gap-bench
#SBATCH --partition=gpu
#SBATCH --constraint=A100|L40S
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=24
#SBATCH --mem=128G
#SBATCH --time=24:00:00
#SBATCH --output=/nfs/scistore19/locatgrp/bdemirel/ssl_project/outputs/%x_%j.out

# E27 §(j): Lightly-benchmark-faithful linear eval (MAE recipe) on a project ckpt. Example:
#   sbatch slurm/bench_probe.sbatch outputs/in1k.lejepa.s0.e27lej_best.pt in1k.lejepa.s0.e27lej.extL
source /nfs/scistore19/locatgrp/bdemirel/ssl_project/slurm/_env.sh
cd /nfs/scistore19/locatgrp/bdemirel/ssl_project
python experiments/bench_probe.py "$@"
