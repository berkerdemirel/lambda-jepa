#!/bin/bash
# E12 F-wave launcher (D-027) — docs/experiments/E12_moment_floor.md §F-wave.
#   slurm/e12f_launch.sh prep                       # equal-pull measurement for f4/f5 (+f3 attempt)
#   slurm/e12f_launch.sh fullA                      # f1/f2/f6 (validated moment term) + 3 chain links each
#   slurm/e12f_launch.sh smoke <lam_f4> <lam_f5>    # 2-ep smokes of the NEW terms: f3/f4/f5
#   slurm/e12f_launch.sh fullB <lam_f4> <lam_f5>    # f3/f4/f5 full + 3 chain links each
# λ rules (pre-declared on the card): f1=0.1 f2=0.02 f3=0.4775-matched f6=0.4775-matched(delay ep10);
# f4/f5 = equal-pull measured (fallbacks f4=0.1, f5=0.4775 if DEGENERATE per equal_pull.py flag).
set -euo pipefail
MODE=$1
cd /nfs/scistore19/locatgrp/bdemirel/ssl_project

COMMON="method=lejepa frame=in100_vits16 bs=128 num_classes=100 num_workers=10
        method.lr=3e-4 method.warmup_ep=10 method.eta_min=1e-5 method.grad_clip=1.0
        method.lamb=0.02 +method.spec_norm=true +method.embed_calib=true"
F1="$COMMON +method.h_reg=sacreg +method.h_lamb=0.1"
F2="$COMMON +method.h_reg=sacreg +method.h_lamb=0.02"
F3="$COMMON +method.h_reg=spec_floor +method.h_lamb=0.477532"
F6="$COMMON +method.h_reg=sacreg +method.h_lamb=0.477532 +method.h_start_ep=10"

chain() {  # chain <n_links> <first_dep_jobid> <arm_string...> — afterany continuation links
    local n=$1 p=$2; shift 2
    for i in $(seq 1 "$n"); do p=$(sbatch --parsable --dependency=afterany:$p slurm/train.sbatch "$@"); echo "  link $p"; done
}

if [ "$MODE" = prep ]; then
    sbatch slurm/e12f_prep.sbatch
elif [ "$MODE" = fullA ]; then
    for arm in F1 F2 F6; do
        tag=$(echo $arm | tr 'F' 'f'); tag="e12$tag"
        j=$(sbatch --parsable slurm/train.sbatch ${!arm} tag=$tag); echo "$tag primary $j"
        chain 3 "$j" ${!arm} tag=$tag
    done
elif [ "$MODE" = smoke ]; then
    F4="$COMMON +method.h_reg=sigreg_std +method.h_lamb=$2"
    F5="$COMMON +method.h_reg=sacreg_diag +method.h_lamb=$3"
    sbatch slurm/train.sbatch $F3 tag=e12f3smoke frame.epochs=2
    sbatch slurm/train.sbatch $F4 tag=e12f4smoke frame.epochs=2
    sbatch slurm/train.sbatch $F5 tag=e12f5smoke frame.epochs=2
elif [ "$MODE" = fullB ]; then
    F4="$COMMON +method.h_reg=sigreg_std +method.h_lamb=$2"
    F5="$COMMON +method.h_reg=sacreg_diag +method.h_lamb=$3"
    for arm in F3 F4 F5; do
        tag=$(echo $arm | tr 'F' 'f'); tag="e12$tag"
        j=$(sbatch --parsable slurm/train.sbatch ${!arm} tag=$tag); echo "$tag primary $j"
        chain 3 "$j" ${!arm} tag=$tag
    done
else
    echo "usage: e12f_launch.sh prep | fullA | smoke <lam_f4> <lam_f5> | fullB <lam_f4> <lam_f5>"; exit 1
fi
