#!/bin/bash
# E27 S-wave launcher (D-081/D-082): the 4 floorssl chains — {mc,lg} x {anchor,band}.
#   Usage: bash slurm/e27_wave_s.sh mc W_INV W_FLOOR H_LAMB
#          bash slurm/e27_wave_s.sh lg W_INV W_FLOOR H_LAMB
# Doses come from the version's OWN pilot ep1 [share] g's: w_i = s*_i * 9.0 / g_i at
# s* = (.63, .34, .03). Anchor and band launch byte-identical — the band cell's steering
# is procedural (ep25/ep50 reads on the audit-channel Ω_h vs [.35,.50], corrections as
# D-rows). 8 x 8h singleton segments per cell, single-GPU H100, 28 workers (D-080/D-057).
set -e
V=$1; WI=$2; WF=$3; WH=$4
[ "$V" = mc ] || [ "$V" = lg ] || { echo "version must be mc|lg"; exit 1; }
[ -n "$WH" ] || { echo "usage: e27_wave_s.sh mc|lg W_INV W_FLOOR H_LAMB"; exit 1; }
EXTRA=()
[ "$V" = lg ] && EXTRA=("method.local_scale=[0.7,1.0]")
for CELL in "" b; do
  NAME=e27s${V}${CELL}
  for SEG in 1 2 3 4 5 6 7 8; do
    sbatch --partition=gpu100 --constraint=H100 --job-name="$NAME" \
      --dependency=singleton --cpus-per-task=28 --mem=128G --begin=now+120 \
      slurm/train.sbatch method=floorssl frame=in1k_vits16 bs=128 num_classes=1000 \
      share_log_every=1 +method.aug=lejepa_mc method.head_layers=2 \
      method.expander_dim=256 method.mlp_wd=0.05 method.z_floor_batch=view_mean \
      method.h_floor_batch=view_mean method.queue_steps=3 \
      method.w_inv="$WI" method.w_floor="$WF" method.h_lamb="$WH" \
      "+extra_cadence=[10]" "${EXTRA[@]}" tag="$NAME"
  done
done
echo "[e27_wave_s] $V pair launched: e27s${V} (anchor) + e27s${V}b (band), 8 segments each"
