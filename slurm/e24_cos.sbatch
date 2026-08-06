#!/bin/bash
#SBATCH --job-name=e24-cos
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=02:30:00
#SBATCH --output=/nfs/scistore19/locatgrp/bdemirel/ssl_project/outputs/%x_%j.out
source /nfs/scistore19/locatgrp/bdemirel/ssl_project/slurm/_env.sh
cd /nfs/scistore19/locatgrp/bdemirel/ssl_project
# force-geometry ladder at matched formation states: pooled -> vm2 -> vm3 -> vm4,
# three frames; toy_v1 = the live vm4 canary (_last, ep stamped in the row).
python experiments/e24_cos.py \
  toy_Llr     outputs/toy.floorssl.s0.e23Llr_ep38.pt \
  toy_s0ac    outputs/toy.floorssl.s0.e24s0ac_ep38.pt \
  toy_v1      outputs/toy.floorssl.s0.e24v1_last.pt \
  in100_d256  outputs/in100.floorssl.s0.d256_ep25.pt \
  in100_vm2   outputs/in100.floorssl.s0.d256vm2_ep25.pt \
  in100_vm3   outputs/in100.floorssl.s0.d256vm3_ep25.pt \
  in100_vm4   outputs/in100.floorssl.s0.d256vm4_ep25.pt \
  in1k_vm2    outputs/in1k.floorssl.s0.d256vm_ep25.pt \
  in1k_vm3    outputs/in1k.floorssl.s0.d256vm3_ep25.pt \
  in1k_vm4    outputs/in1k.floorssl.s0.d256vm4_ep25.pt
