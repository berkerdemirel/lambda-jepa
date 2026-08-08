#!/bin/bash
#SBATCH --job-name=e24-pull-1k
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=00:50:00
#SBATCH --output=/nfs/scistore19/locatgrp/bdemirel/ssl_project/outputs/%x_%j.out
source /nfs/scistore19/locatgrp/bdemirel/ssl_project/slurm/_env.sh
cd /nfs/scistore19/locatgrp/bdemirel/ssl_project
python experiments/e24_pull.py \
  in1k_vm2  outputs/in1k.floorssl.s0.d256vm_ep25.pt   ckpt \
  in1k_vm3  outputs/in1k.floorssl.s0.d256vm3_ep25.pt  ckpt \
  in1k_vm4  outputs/in1k.floorssl.s0.d256vm4_ep25.pt  ckpt
