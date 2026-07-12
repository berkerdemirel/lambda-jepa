#!/bin/bash
# E12 (D-026) launcher — docs/experiments/E12_moment_floor.md.
#   slurm/e12_launch.sh smoke <lam_a2> <lam_a3>   # 2-ep GPU smoke of the two NEW code paths (A2, A3)
#   slurm/e12_launch.sh full  <lam_a2> <lam_a3>   # all 4 arms: A1/A2 on h100 slots, A3/C1 on gpu
# λ values come from results/diag/e12_equal_pull.csv (premise p3) — launched verbatim, no discretion.
set -euo pipefail
MODE=$1; LAM_A2=$2; LAM_A3=$3
cd /nfs/scistore19/locatgrp/bdemirel/ssl_project

COMMON="method=lejepa frame=in100_vits16 bs=128 num_classes=100 num_workers=10
        method.lr=3e-4 method.warmup_ep=10 method.eta_min=1e-5 method.grad_clip=1.0"
# Amended arms (E12 card §Amendment, 2026-07-11): shipped SIGReg@proj (lamb 0.02) retained in
# A2/A3 — the h-side term is ADDITIVE (h_reg/h_lamb); moved placement admitted a lazy projector.
A1="$COMMON method.lamb=0.02 +method.proj_depth=0 +method.embed_calib=true"
A2="$COMMON method.lamb=0.02 +method.spec_norm=true +method.embed_calib=true +method.h_reg=sigreg +method.h_lamb=$LAM_A2"
A3="$COMMON method.lamb=0.02 +method.spec_norm=true +method.embed_calib=true +method.h_reg=moment +method.h_lamb=$LAM_A3"
C1="$COMMON method.lamb=0.02 +method.spec_norm=true +method.embed_calib=true"

if [ "$MODE" = smoke ]; then
    sbatch slurm/train.sbatch $A2 tag=e12a2smoke2 frame.epochs=2
    sbatch slurm/train.sbatch $A3 tag=e12a3smoke2 frame.epochs=2
elif [ "$MODE" = full ]; then
    sbatch --partition=gpu100 --constraint=H100 --job-name=h100-slotA --dependency=singleton slurm/train.sbatch $A1 tag=e12a1
    sbatch --partition=gpu100 --constraint=H100 --job-name=h100-slotB --dependency=singleton slurm/train.sbatch $A2 tag=e12a2
    sbatch slurm/train.sbatch $A3 tag=e12a3
    sbatch slurm/train.sbatch $C1 tag=e12c1
else
    echo "usage: e12_launch.sh smoke|full <lam_a2> <lam_a3>"; exit 1
fi
