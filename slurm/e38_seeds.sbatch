#!/bin/bash
#SBATCH --job-name=e38
#SBATCH --partition=gpu100
#SBATCH --constraint=H100
#SBATCH --exclude=gpu277,gpu274
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=28
#SBATCH --mem=128G
#SBATCH --time=08:00:00
#SBATCH --requeue
#SBATCH --open-mode=append
#SBATCH --output=/nfs/scistore19/locatgrp/bdemirel/ssl_project/outputs/%x_%j.out

# E38 (docs/experiments/E38_in100_seeds.md): seed repeats of the paper's IN-100 controlled pairs.
# Plain `experiments/train.py` with the per-cell override set recovered from the seed-0
# checkpoint's stored cfg (slurm/e38_launch.sh); only `seed=` differs. Resume-safe via _last.pt.
# gpu277 excluded: CUDA init error 802 re-confirmed 2026-09-19 (its idle cards are not capacity);
# gpu274 excluded: input-bound there (GPU 0 % with every loader worker pegged; the two LeJEPA smokes
# had not finished one epoch after 12 min where other nodes took 4) — the E34 "gpu274 slow" anomaly.
source /nfs/scistore19/locatgrp/bdemirel/ssl_project/slurm/_env.sh
cd /nfs/scistore19/locatgrp/bdemirel/ssl_project

# Driver-health gate (2026-09-03 pattern, slurm/e28_scaleinit.sbatch): a node whose CUDA driver is
# unusable kills the job at the first .to(cuda); wait, requeue under the same JobId so the chain's
# dependencies hold, bounded so a dead fleet cannot spin forever.
if ! nvidia-smi -L >/dev/null 2>&1; then
  echo "[e38] $(date '+%F %T') $(hostname -s): CUDA driver unusable (life ${SLURM_RESTART_COUNT:-0}) — requeue in 3 min"
  if [ "${SLURM_RESTART_COUNT:-0}" -ge 160 ]; then echo "[e38] giving up after 160 lives"; exit 1; fi
  sleep 180
  scontrol requeue "$SLURM_JOB_ID"
  sleep 60
  exit 0
fi
python experiments/train.py "$@"
