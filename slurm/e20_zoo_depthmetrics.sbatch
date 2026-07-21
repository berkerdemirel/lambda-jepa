#!/bin/bash
#SBATCH --job-name=e20zoo-depthm
#SBATCH --partition=defaultp
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=/nfs/scistore19/locatgrp/bdemirel/ssl_project/outputs/%x_%j.out
source /nfs/scistore19/locatgrp/bdemirel/ssl_project/slurm/_env.sh
cd /nfs/scistore19/locatgrp/bdemirel/ssl_project
python experiments/e20f_depth_metrics.py e20_zoo_depth_metrics.csv \
  simclr_e20f=in100.simclr.s0.e20f.extL byol_e20f=in100.byol.s0.e20f.extL \
  vicreg_e20f=in100.vicreg.s0.e20f.extL dino_e20f=in100.dino.s0.e20f.extL \
  lejepa_e20f=in100.lejepa.s0.e20f.ep100.extL mae_e20f=in100.mae.s0.e20f.extL \
  ijepa_e20f=in100.ijepa.s0.e20f.extL mae_ctrl=in100.mae.s0.extL ijepa_ctrl=in100.ijepa.s0.extL
