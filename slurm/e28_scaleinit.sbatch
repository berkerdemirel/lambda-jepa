#!/bin/bash
#SBATCH --job-name=e28x10
#SBATCH --partition=gpu100
#SBATCH --constraint=H100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=08:00:00
#SBATCH --requeue
#SBATCH --output=/nfs/scistore19/locatgrp/bdemirel/ssl_project/outputs/%x_%j.out

# E28 (docs/experiments/E28_init_scale.md): the d256vm4zonly lane with the ViT's final
# LayerNorm moved into the projector (h = unnormalized) and the block weight matrices ×10 at
# init. cpus/mem/time mirror slurm/train.sbatch so the loader geometry (num_workers =
# SLURM_CPUS_PER_TASK) matches the ×1 reference lane; partition is H100 by D-093's budget.
#
# The selftest gates EVERY segment (house pattern, e27_wavefinal.sbatch): it re-proves the
# relocation is numerically a no-op for z and that the ×10 touched exactly the 48 block weight
# tensors. E28_MODE=control runs the same diagnostics over the landed ×1 checkpoints instead.
source /nfs/scistore19/locatgrp/bdemirel/ssl_project/slurm/_env.sh
cd /nfs/scistore19/locatgrp/bdemirel/ssl_project
python experiments/e28_scaleinit.py "$@" +e28.mode=selftest || exit 1
python experiments/e28_scaleinit.py "$@" +e28.mode=${E28_MODE:-train}
