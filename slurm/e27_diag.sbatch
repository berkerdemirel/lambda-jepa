#!/bin/bash
#SBATCH --job-name=e27diag
#SBATCH --partition=gpu100
#SBATCH --constraint=H100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=28
#SBATCH --mem=128G
#SBATCH --time=08:00:00
#SBATCH --output=/nfs/scistore19/locatgrp/bdemirel/ssl_project/outputs/%x_%j.out

# E27 v4->v10 diagnostic sweep cell (D-087; Berker 2026-08-08: "make a good breadth and
# do many experiments (no h100 cap for this diagnosis) … my initial attempt was to scale
# the thing from v=4 to v=10"). One 5-epoch single-GPU cell per delta on the v4->v10
# bridge; fixed KNOWN-GOOD doses (vm4's w for the uniform-V lane, voas's w for the
# multicrop lane) — no derived doses in this round. Selftest gates every cell. Full cell
# table: E27 card §(h). Caller passes the cell's overrides + tag; job name set at submit.
source /nfs/scistore19/locatgrp/bdemirel/ssl_project/slurm/_env.sh
cd /nfs/scistore19/locatgrp/bdemirel/ssl_project
python experiments/e27_selftest.py || exit 1
python experiments/train.py method=lambdajepa frame=in1k_vits16 frame.epochs=5 bs=128 \
  num_classes=1000 share_log_every=1 resume=false \
  method.head_layers=2 method.expander_dim=256 method.mlp_wd=0.05 \
  method.z_floor_batch=view_mean method.h_floor_batch=view_mean method.queue_steps=3 \
  "$@"
