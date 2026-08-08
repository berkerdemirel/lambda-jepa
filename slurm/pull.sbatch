#!/bin/bash
#SBATCH --job-name=gap-pull
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --output=/nfs/scistore19/locatgrp/bdemirel/ssl_project/outputs/%x_%j.out

# THE pull instrument (replaces the e12h/e21/e22/e23/e24 pull wrappers). Triples repeat:
#   sbatch slurm/pull.sbatch vm4_in1k outputs/in1k.floorssl.s0.d256vm4_ep25.pt ckpt
#   sbatch slurm/pull.sbatch voas outputs/in1k.floorssl.s0.e24voas_ep100.pt ckpt --batches 6
# The ring warms across batches (D-072): read the WARM rows, not batch 0.
source /nfs/scistore19/locatgrp/bdemirel/ssl_project/slurm/_env.sh
cd /nfs/scistore19/locatgrp/bdemirel/ssl_project
python experiments/pull.py "$@"
