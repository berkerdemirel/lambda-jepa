#!/bin/bash
#SBATCH --job-name=e34_vits_k710_h100x8
#SBATCH --partition=gpu100
#SBATCH --constraint=H100
#SBATCH --exclude=gpu269,gpu273,gpu267,gpu272,gpu275
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=8
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=24
#SBATCH --mem=1200G
#SBATCH --time=2-00:00:00
#SBATCH --output=/nfs/scistore19/locatgrp/bdemirel/ssl_project/outputs/%x_%j.out
# E34 first cell: the lambda-JEPA recipe on LeVJEPA's ViT-S/16, K710-20% (three stores), 240 epochs.
# One node x 8 H100, batch 96 per GPU x accumulation 4 = the donor's effective batch 3,072
# (their config: 2 nodes x 8 x 96 x 2). Run dir, checkpoints (every epoch) and logs on BeeGFS.
# NOT launched without Berker's word (card predictions confirmed first). Usage: sbatch slurm/train_k710_lambdajepa_vits.slurm
cd /nfs/scistore19/locatgrp/bdemirel/ssl_project/video/levjepa || exit 1
export LEVJEPA_DATA_ROOT=/mnt/beegfs/locatgrp/shared/datasets/kinetics
export HYDRA_FULL_ERROR=1
export WANDB_ENTITY=causal-learning-ai-ista
RUN=vid.floorssl.s0.k710s
nvidia-smi -L | head -8
srun .venv/bin/python -u main.py --config-name lambdajepa_vits data=k710_20pct \
  trainer.devices=8 trainer.num_nodes=1 accumulate_grad_batches=4 trainer.max_epochs=240 \
  loader.num_workers=22 checkpoint.every_n_epochs=1 checkpoint.every_n_train_steps=0 \
  wandb.enabled=true wandb.config.project=sslgap wandb.config.name=$RUN \
  hydra.run.dir=/mnt/beegfs/locatgrp/shared/bdemirel/levjepa_runs/$RUN
echo "E34_TRAIN_EXIT rc=$?"
