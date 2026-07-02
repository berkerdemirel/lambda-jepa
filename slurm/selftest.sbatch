#!/bin/bash
#SBATCH --job-name=gap-selftest
#SBATCH --partition=gpu
#SBATCH --constraint=L40S|A40|A100|RTX3090Ti
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:20:00
#SBATCH --output=/nfs/scistore19/locatgrp/bdemirel/ssl_project/outputs/%x_%j.out

# Adapter↔trainer coherence self-test. Args = native checkpoint paths.
#   sbatch slurm/selftest.sbatch outputs/toy.lejepa.s0_ep150.pt outputs/toy.vicreg.s0_ep150.pt
source /nfs/scistore19/locatgrp/bdemirel/ssl_project/slurm/_env.sh
cd /nfs/scistore19/locatgrp/bdemirel/ssl_project
python experiments/adapter_selftest.py "$@"
