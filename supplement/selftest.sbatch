#!/bin/bash
#SBATCH --job-name=gap-selftest
#SBATCH --partition=gpu
#SBATCH --constraint=L40S|A40|A100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --output=outputs/%x_%j.out

# The two gates that every ImageNet-1k DDP job runs before training (slurm/in1k_ddp.sbatch):
#   lambdajepa_selftest.py   — resolution parity of the dynamic-size trunk, the view-to-mean == all-pairs
#                              invariance identity, the ring buffer, one full training step
#   train_ddp_selftest.py    — at world_size 1 the DDP loss reproduces the single-GPU training_step
#                              byte-for-byte (same RNG state, same slice)
# Run it once after `uv sync` to check the installation:  sbatch slurm/selftest.sbatch
source "$SLURM_SUBMIT_DIR/slurm/_env.sh"
cd "$SLURM_SUBMIT_DIR"
python experiments/lambdajepa_selftest.py && python experiments/train_ddp_selftest.py && echo "SELFTEST_OK"
