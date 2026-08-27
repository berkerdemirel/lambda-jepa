#!/bin/bash
#SBATCH --job-name=tvlp
#SBATCH --partition=gpu
#SBATCH --constraint=A100|L40S|A40
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=12:00:00
#SBATCH --output=/nfs/scistore19/locatgrp/bdemirel/ssl_project/outputs/%x_%j.out

# VISReg Table-5 transfer linear probe (their protocol, 10 ep; experiments/transfer_visreg.py).
# Port-validation cell = the public DINO-B their table row uses, vs their printed numbers:
#   sbatch -J tvlp-dinob slurm/transfer_visreg.sbatch timm:vit_base_patch16_224.dino dinob.pub
#   sbatch -J tvlp-sbe   slurm/transfer_visreg.sbatch native:outputs/in1k.floorssl.s0.e27lm4sbe_ep100.pt in1k.floorssl.s0.e27lm4sbe
#   sbatch -J tvlp-vrb   slurm/transfer_visreg.sbatch visreg:<anchor latest.pt> in1k.visreg.s0.vrb
source /nfs/scistore19/locatgrp/bdemirel/ssl_project/slurm/_env.sh
cd /nfs/scistore19/locatgrp/bdemirel/ssl_project
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
python experiments/transfer_visreg.py "$@"
