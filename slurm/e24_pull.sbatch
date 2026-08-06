#!/bin/bash
#SBATCH --job-name=e24-pull
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=00:40:00
#SBATCH --output=/nfs/scistore19/locatgrp/bdemirel/ssl_project/outputs/%x_%j.out
source /nfs/scistore19/locatgrp/bdemirel/ssl_project/slurm/_env.sh
cd /nfs/scistore19/locatgrp/bdemirel/ssl_project
python experiments/e24_pull.py \
  toy_Llr       outputs/toy.floorssl.s0.e23Llr_ep38.pt      ckpt \
  toy_Llr       outputs/toy.floorssl.s0.e23Llr_ep38.pt      init \
  in100_d256    outputs/in100.floorssl.s0.d256_ep25.pt      ckpt \
  in100_d256    outputs/in100.floorssl.s0.d256_ep25.pt      init
