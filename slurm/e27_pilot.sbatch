#!/bin/bash
#SBATCH --job-name=gap-e27pilot
#SBATCH --partition=gpu100
#SBATCH --constraint=H100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=28
#SBATCH --mem=128G
#SBATCH --time=02:30:00
#SBATCH --output=/nfs/scistore19/locatgrp/bdemirel/ssl_project/outputs/%x_%j.out

# E27 smoke+pilot in one job (D-079a): selftest gates the train; the 2-ep train IS the dose
# pilot — the [share] ep1 line carries the formation-state per-term trunk g's under the
# Recipe v2 multicrop aug, measured at the incumbent family weights (voas 21.4/49.6/1.89,
# the held-state convention). Doses for the S pair derive as w_i = s*_i * T / g_i at
# s* = (.63, .34, .03), T = 9.0 (the certified in1k family total at S).
source /nfs/scistore19/locatgrp/bdemirel/ssl_project/slurm/_env.sh
cd /nfs/scistore19/locatgrp/bdemirel/ssl_project
# Trailing args override (hydra last-wins): the large-locals pilot passes
#   'method.local_scale=[0.7,1.0]' tag=e27slg_pilot
python experiments/e27_selftest.py || exit 1
python experiments/train.py method=lambdajepa frame=in1k_vits16 frame.epochs=2 bs=128 \
  num_classes=1000 share_log_every=1 resume=false \
  +method.aug=lejepa_mc method.head_layers=2 method.expander_dim=256 method.mlp_wd=0.05 \
  method.z_floor_batch=view_mean method.h_floor_batch=view_mean method.queue_steps=3 \
  method.w_inv=21.4 method.w_floor=49.6 method.h_lamb=1.89 \
  tag=e27s_pilot "$@"
