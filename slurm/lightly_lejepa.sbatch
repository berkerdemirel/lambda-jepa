#!/bin/bash
#SBATCH --job-name=lightly-lejepa
#SBATCH --partition=gpu100
#SBATCH --constraint=H100
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=16
#SBATCH --mem=256G
#SBATCH --time=08:00:00
#SBATCH --requeue
#SBATCH --output=/nfs/scistore19/locatgrp/bdemirel/ssl_project/outputs/%x_%j.out

# The zero-diff Lightly LeJEPA reproduction (Berker 2026-08-10: "do what is needed to get
# the lightly numbers, i want no diff"): THEIR code verbatim — lightly @ f444cf36,
# benchmarks/imagenet/vitb16/main.py --methods lejepa, their defaults (4x128 = global
# bs 512, 16-mixed, unseeded, their DINOTransform/collate/schedule/eval chain).
# Runnable copy ~/lightly_repro (own venv: torch 2.13.0+cu130, PL 2.6.5, timm 1.0.28);
# pristine reference third_party/lightly. Zero code patches — data/log paths are CLI args.
# Phase logic (ours, outside their code): pretrain segments skip the evals so wall cuts
# never land mid-eval; once epoch=99 exists, the segment runs eval-only (their
# --epochs 0 load path): kNN + MAE-protocol linear. finetune eval skipped (declared).
REPRO=/nfs/scistore19/locatgrp/bdemirel/lightly_repro
PY=$REPRO/.venv/bin/python
LOGROOT=$REPRO/benchmark_logs/lejepa
mkdir -p "$LOGROOT"
cd $REPRO/benchmarks/imagenet/vitb16

CKPT=$(ls -t "$LOGROOT"/*/pretrain/*/checkpoints/*.ckpt 2>/dev/null | head -n 1)
ARGS=(--train-dir ~/data/imagenet/train --val-dir ~/data/imagenet/val
      --log-dir "$REPRO/benchmark_logs" --methods lejepa
      --devices 4 --num-workers 12)

if [[ -n "$CKPT" && "$CKPT" == *"epoch=99"* ]]; then
  echo "[phase] EVAL from $CKPT"
  srun $PY main.py "${ARGS[@]}" --epochs 0 --ckpt-path "$CKPT" --skip-finetune-eval
else
  echo "[phase] PRETRAIN resume=${CKPT:-none}"
  srun $PY main.py "${ARGS[@]}" --skip-knn-eval --skip-linear-eval --skip-finetune-eval \
    ${CKPT:+--ckpt-path "$CKPT"}
fi
