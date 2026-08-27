#!/bin/bash
#SBATCH --job-name=visreg-lin
#SBATCH --partition=gpu
#SBATCH --constraint=A100|L40S
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=200G
#SBATCH --time=24:00:00
#SBATCH --output=/nfs/scistore19/locatgrp/bdemirel/ssl_project/outputs/%x_%j.out

# C9 B-anchor "their version" in1k linear (Berker 2026-08-23: both evals): VISReg's own
# downstream/linear_prob suite (DINO-style concat last-4-layer CLS features, MultiLR
# classifier sweep) run in THEIR venv on the repro ckpt, per their README. The in1k data
# path rides repro patch #2 (PATCHES.md — offline ImageFolder arrow, data access only).
#   sbatch slurm/visreg_linear.sbatch --checkpoint <ckpt.pt> --model vit_b \
#     --datasets imagenet --shots -1 --output downstream/<name>.json [--quick]
export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export VISREG_EVAL_WORKERS=24   # repro patch #3 (PATCHES.md): loader perf only
cd /nfs/scistore19/locatgrp/bdemirel/visreg_repro
.venv/bin/python downstream/linear_prob/run_evaluation.py "$@"
