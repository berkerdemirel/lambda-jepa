#!/bin/bash
#SBATCH --job-name=vid_smoke_wt2
#SBATCH --partition=gpu
#SBATCH --constraint=RTX3090Ti|A40
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=00:45:00
#SBATCH --output=/nfs/scistore19/locatgrp/bdemirel/ssl_project/outputs/%x_%j.out
# Real-data smoke of the lambda-JEPA recipe (E34) on the Walking Tours store, ONE GPU, ViT-S, a few
# hundred optimizer steps: exercises the Lance loader with six views, the DDP-free path, the
# share logger and checkpointing. Not an experiment; no numbers from it are read as results.
# Launch (from video/levjepa): sbatch slurm/lambdajepa_smoke_wt.slurm
cd /nfs/scistore19/locatgrp/bdemirel/ssl_project/video/levjepa || exit 1
export LEVJEPA_DATA_ROOT=/mnt/beegfs/locatgrp/shared/datasets
export HYDRA_FULL_ERROR=1
nvidia-smi -L
.venv/bin/python -u main.py --config-name lambdajepa_vits data=walking_tours \
  trainer.devices=1 trainer.num_nodes=1 accumulate_grad_batches=1 loader.batch_size=16 loader.num_workers=6 \
  loader.clips_per_video=20 trainer.max_epochs=1 trainer.limit_train_batches=150 \
  checkpoint.every_n_train_steps=100 wandb.enabled=false loss.lambdajepa.share_log_bs=32
echo "SMOKE_WT_DONE rc=$?"
