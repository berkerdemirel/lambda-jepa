#!/bin/bash
#SBATCH --job-name=gap-seg
#SBATCH --partition=gpu
#SBATCH --constraint=A100|L40S
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=12:00:00
#SBATCH --output=/nfs/scistore19/locatgrp/bdemirel/ssl_project/outputs/%x_%j.out

# ADE20k linear seg, VISReg protocol (D-108.5). Example:
#   sbatch slurm/seg_visreg.sbatch --ckpt timm:vit_base_patch16_224.dino --tag in1k.pub.dinob400.seg
# timm: sources may need the hub online (pub weights fetch, D-076 pattern).
source /nfs/scistore19/locatgrp/bdemirel/ssl_project/slurm/_env.sh
export HF_HUB_OFFLINE=0
cd /nfs/scistore19/locatgrp/bdemirel/ssl_project
python experiments/seg_visreg.py "$@"
