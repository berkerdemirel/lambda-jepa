#!/bin/bash
#SBATCH --job-name=visreg-b
#SBATCH --partition=gpu100
#SBATCH --constraint=H100
#SBATCH --nodes=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=48
#SBATCH --mem=400G
#SBATCH --time=08:00:00
#SBATCH --requeue
#SBATCH --output=/nfs/scistore19/locatgrp/bdemirel/ssl_project/outputs/%x_%j.out

# D-092/D-097 VISReg-B faithful reproduction — the B-rung external anchor (4g@224+6l@96,
# lambda=.9 SWD sketch, their shipped ViT-B config verbatim; ~/visreg_repro on THEIR
# pins, one declared data patch). 4xH100 x bs 128/rank = their effective 512 (their reg
# gathers projections across processes, so only the product matters); bf16 via
# accelerate per their documented launch. Segments resume via their own latest.pt +
# wandb_id (native support); wandb ONLINE under the team entity — the run appears in
# project SSL-ImageNet1K-VIT-B-16 (their code names it; entity via env).
export WANDB_ENTITY=causal-learning-ai-ista
export HF_HUB_OFFLINE=1
cd /nfs/scistore19/locatgrp/bdemirel/visreg_repro

CKPT=$(ls -t checkpoints/*/*/checkpoints/latest.pt 2>/dev/null | head -n 1)
WID=""
if [[ -n "$CKPT" && -L wandb/latest-run ]]; then
  WID=$(basename "$(readlink wandb/latest-run)" | awk -F- '{print $NF}')
fi
echo "[segment] resume=${CKPT:-fresh} wandb_id=${WID:-new} CVD=${CUDA_VISIBLE_DEVICES:-unset} ngpu=$(nvidia-smi -L | wc -l)"
.venv/bin/accelerate launch --multi_gpu --num_processes 4 --gpu_ids 0,1,2,3 \
  --mixed_precision bf16 train.py \
  bs=128 +num_workers=12 ${CKPT:+resume="$CKPT"} ${WID:+wandb_id="$WID"}
