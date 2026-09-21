#!/bin/bash
# RESUME vm4-1k (paused 2026-07-24 14:08 at ep74/100, best_acc .5847, for Berker's rebuttal H100s).
# train.py auto-resumes from outputs/in1k.floorssl.s0.d256vm4_last.pt (epoch 74) — SAME args = SAME
# run_id. 26 epochs left ~= 16h ~= 3 links; 4 submitted for slack. Queue re-warms on resume (declared).
set -e
cd /nfs/scistore19/locatgrp/bdemirel/ssl_project
ARGS="method=lambdajepa frame=in1k_vits16 num_classes=1000 bs=128 +method.aug=lejepa +method.V=4 \
method.w_inv=26.9 method.w_floor=129.2 method.h_lamb=1.679 method.expander_dim=256 \
method.z_floor_batch=view_mean method.z_d_slice=128 method.h_floor_batch=view_mean method.h_d_slice=128 \
method.queue_steps=3 num_workers=28 pin_memory=true persistent_workers=true eval_every=2 tag=d256vm4"
N=${1:-4}
for i in $(seq 1 $N); do
  sbatch --job-name=h100-slotA --partition=gpu100 --constraint=H100 \
         --dependency=singleton --cpus-per-task=28 --mem=96G --time=08:00:00 slurm/train.sbatch $ARGS
done
echo "resubmitted $N slotA links; verify: squeue -u bdemirel -h -n h100-slotA | wc -l  (expect $N)"
