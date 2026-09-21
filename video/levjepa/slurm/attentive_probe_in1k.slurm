#!/bin/bash
#SBATCH --job-name=attn_probe_in1k
#SBATCH --partition=gpu100
#SBATCH --constraint=H100
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=64
#SBATCH --mem=400G
#SBATCH --time=1-00:00:00
#SBATCH --output=/nfs/scistore19/locatgrp/bdemirel/ssl_project/outputs/%x_%j.out
# Frozen attentive probe on ImageNet-1k (V-JEPA protocol; scripts/attentive_probe.py docstring), the
# E34 bench read. 8 GPUs x bs 16 = 128 global (V-JEPA ran 64 GPUs x 16 = 1,024 at lr 1e-3; the lr is
# kept, the batch is what 8 cards give — recorded on the card as a protocol note).
# Usage: sbatch slurm/attentive_probe_in1k.slurm <lightning .ckpt> <out csv> [ema|student]
cd /nfs/scistore19/locatgrp/bdemirel/ssl_project/video/levjepa || exit 1
export PYTHONPATH=/nfs/scistore19/locatgrp/bdemirel/ssl_project/video/levjepa
CKPT=$1; OUT=$2; W=${3:-ema}
nvidia-smi -L | head -2
.venv/bin/torchrun --standalone --nproc_per_node=8 scripts/attentive_probe.py --ckpt "$CKPT" --out "$OUT" --weights "$W" --bs 16 --workers 8
echo "ATTN_PROBE_DONE rc=$?"
