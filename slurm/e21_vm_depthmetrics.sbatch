#!/bin/bash
#SBATCH --job-name=e21vm-depthm
#SBATCH --partition=defaultp
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=/nfs/scistore19/locatgrp/bdemirel/ssl_project/outputs/%x_%j.out
source /nfs/scistore19/locatgrp/bdemirel/ssl_project/slurm/_env.sh
cd /nfs/scistore19/locatgrp/bdemirel/ssl_project
python experiments/e20f_depth_metrics.py e21_vm_depth_metrics.csv \
  vm3zonly=in100.floorssl.s0.d256vm3zonly.extL vm3x2=in100.floorssl.s0.d256vm3x2.extL \
  vm3=in100.floorssl.s0.d256vm3.extL vm2=in100.floorssl.s0.d256vm2.extL \
  d256=in100.floorssl.s0.d256.extL d128=in100.floorssl.s0.d128.extL d64=in100.floorssl.s0.d64.extL
