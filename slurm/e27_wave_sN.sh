#!/bin/bash
# E27 S-wave launcher, parametric over wave (D-084/D-085). One script per wave was turning
# into a third near-copy; the per-wave RECORD lives in DECISIONS + the E27 card, not here.
#   Usage: bash slurm/e27_wave_sN.sh mc|lg W_INV W_FLOOR H_LAMB SUFFIX
#   e.g.   bash slurm/e27_wave_sN.sh mc 21.4 49.6 1.89 3      -> e27smc3 + e27smcb3
#   Placement via env (D-080 sanctions A100 for the in1k program; SLURM's native partition
#   OR is disabled on this cluster, so it is one partition per submit):
#     PART=gpu CONSTR=A100 bash slurm/e27_wave_sN.sh ...        # default: gpu100/H100
#
# Two changes from e27_wave_s.sh, both measured (E27 card §(d), results/diag/pull.csv):
#  1. cond_stream=globals — the conditioner reads the mean over the 2 GLOBALS; inv keeps
#     every view. The mc cell's view-mean z had retained only 63% of its pooled variance
#     (trace/d .170 against a target of 1.0) because averaging over 8 aggressive locals
#     estimates the per-image centre with crop noise; the conditioner was fighting an
#     averaging artifact. Also makes the stream V-independent.
#  2. Doses re-derived at the ep10 HELD STATE under that stream, w_i = s*_i·T/g_i at
#     s* = (.63, .34, .03) and T = 10.0 — d256vm4's realized total at ep25. The first wave
#     used T = 9.0 measured at a 2-epoch pilot's ep1; the g's then fell ~10x, landing the
#     cells at Sum(w.g) ~ 1.2 against vm4's 9.98.
# Everything else is byte-matched to wave 1. 8 x 8h singleton segments per cell.
set -e
V=$1; WI=$2; WF=$3; WH=$4; SUF=${5:-2}
PART=${PART:-gpu100}; CONSTR=${CONSTR:-H100}
[ "$V" = mc ] || [ "$V" = lg ] || { echo "version must be mc|lg"; exit 1; }
[ -n "$WH" ] || { echo "usage: e27_wave_sN.sh mc|lg W_INV W_FLOOR H_LAMB [SUFFIX]"; exit 1; }
EXTRA=()
[ "$V" = lg ] && EXTRA=("method.local_scale=[0.7,1.0]")
for CELL in "" b; do
  NAME=e27s${V}${CELL}${SUF}
  for SEG in 1 2 3 4 5 6 7 8; do
    sbatch --partition="$PART" --constraint="$CONSTR" --job-name="$NAME" \
      --dependency=singleton --cpus-per-task=28 --mem=128G --begin=now+120 \
      slurm/train.sbatch method=floorssl frame=in1k_vits16 bs=128 num_classes=1000 \
      share_log_every=1 +method.aug=lejepa_mc method.head_layers=2 \
      method.expander_dim=256 method.mlp_wd=0.05 method.z_floor_batch=view_mean \
      method.h_floor_batch=view_mean method.queue_steps=3 method.cond_stream=globals \
      method.w_inv="$WI" method.w_floor="$WF" method.h_lamb="$WH" \
      "+extra_cadence=[10]" "${EXTRA[@]}" tag="$NAME"
  done
done
echo "[e27_wave_sN] ($PART/$CONSTR) $V pair launched: e27s${V}${SUF} (anchor) + e27s${V}b${SUF} (band), 8 segments each"
