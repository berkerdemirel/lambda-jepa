#!/bin/bash
#SBATCH --job-name=e34cf2_smoke_vitb
#SBATCH --partition=gpu100
#SBATCH --constraint=H100
#SBATCH --exclude=gpu274
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=24
#SBATCH --mem=200G
#SBATCH --time=00:50:00
#SBATCH --output=/nfs/scistore19/locatgrp/bdemirel/ssl_project/outputs/%x_%j.out
# Smoke of the E34 ViT-B cell on one H100 before it is queued: memory at bs 64 x 6 views x 16 frames with the GPU
# photometrics, the share logger at share_log_bs 64 (its fixed batch runs WITH gradients for the pull instrument),
# the online K710 probe callback under Lightning (first scoring at the start of epoch 1), the step-50 rate line.
cd /nfs/scistore19/locatgrp/bdemirel/ssl_project/video/levjepa || exit 1
export LEVJEPA_DATA_ROOT=/mnt/beegfs/locatgrp/shared/datasets/kinetics HYDRA_FULL_ERROR=1
nvidia-smi -L | head -n 1
( while true; do echo "[gpumem] $(date '+%H:%M:%S') $(nvidia-smi --query-gpu=memory.used --format=csv,noheader)"; sleep 30; done ) &
MEMPID=$!
.venv/bin/python -u main.py --config-name lambdajepa_vitb data=k710_clipfiles \
  trainer.devices=1 trainer.num_nodes=1 accumulate_grad_batches=4 trainer.max_epochs=2 trainer.limit_train_batches=60 \
  loader.num_workers=22 loader.batch_size=64 augmentation.photometrics_on_gpu=true loss.lambdajepa.share_log_bs=64 +probe_k710=256 \
  checkpoint.every_n_epochs=0 checkpoint.every_n_train_steps=0 wandb.enabled=false \
  hydra.run.dir=/mnt/beegfs/locatgrp/shared/bdemirel/levjepa_runs/smoke_vitb_$SLURM_JOB_ID 2>&1 | grep -v -E "UserWarning|warnings.warn|return func"
echo "SMOKE_VITB_EXIT rc=${PIPESTATUS[0]}"
kill $MEMPID 2>/dev/null
