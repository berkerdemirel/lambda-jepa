#!/bin/bash
#SBATCH --job-name=e24-oas-cmp
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=01:00:00
#SBATCH --output=/nfs/scistore19/locatgrp/bdemirel/ssl_project/outputs/%x_%j.out
source /nfs/scistore19/locatgrp/bdemirel/ssl_project/slurm/_env.sh
cd /nfs/scistore19/locatgrp/bdemirel/ssl_project
python experiments/e24_oas_cmp.py \
  in100_vm4  outputs/in100.floorssl.s0.d256vm4_ep25.pt \
  in1k_vm4   outputs/in1k.floorssl.s0.d256vm4_ep25.pt
