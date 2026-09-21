#!/bin/bash
# E28 launch (docs/experiments/E28_init_scale.md).   usage: bash slurm/e28_launch.sh <arm> [gate_job]
#
#   sweep     init-scale sweep + per-block rank profile (2026-08-13, job 63437775 — LANDED)
#   x10       the ×10 arm, 3×8h H100 chain            (2026-08-13, 63437776→78 — LANDED, cancelled ~ep47)
#   lowsweep  the reopening's step-0 read: ×0.1/×0.3/×0.5 geometry + depth profile at ×0.1, and
#             Ω at ×0.1 per weight family — two small `gpu` jobs, each also the selftest gate for
#             one chain below (prints "gate_x0.1=<job> gate_x1=<job>")
#   x0.1, x1  the reopening pair (2026-09-03): the reference recipe VERBATIM, only the init
#             scale differs; pass the matching gate job so the chain waits on a green selftest
#
# ARGS = the d256vm4zonly override set VERBATIM (recovered from the reference checkpoint's stored
# cfg, not retyped from memory) — note there is deliberately NO +method.V=4: that lane rode the
# code default, and the startup resolved-config guard fails the job on any key that differs from
# the reference beyond `tag` and `e28.*`. The diagnostic CSVs are APPEND-only: never re-run a
# landed arm under the same tag.
set -e
cd /nfs/scistore19/locatgrp/bdemirel/ssl_project

ARGS="method=lambdajepa frame=in100_vits16 bs=128 num_classes=100 +method.aug=lejepa \
method.expander_dim=256 method.w_inv=32.8 method.w_floor=157.8 method.h_lamb=0.0 \
method.z_floor_batch=view_mean method.z_d_slice=128 \
method.h_floor_batch=view_mean method.h_d_slice=128 method.queue_steps=3"
# E28_EXCLUDE=<nodelist> keeps the short gate jobs off nodes known to be bad right now (the
# sbatch's own driver gate would requeue them anyway, just more slowly); the H100 chains take no
# exclude list — with the whole fleet in one state that would make them unschedulable.
SMALL="--partition=gpu --constraint=A100|A40|L40S ${E28_EXCLUDE:+--exclude=$E28_EXCLUDE} \
--time=01:00:00 --mem=64G --cpus-per-task=8"

chain() {   # chain TAG SCALE [GATE]: 3×8h H100 segments (~6 min/epoch ⇒ 100 ep ≈ 10 h),
            # resume-safe via _last.pt; every segment re-runs the selftest at its own scale.
  local D=${3:+--dependency=afterok:$3}
  for i in 1 2 3; do
    J=$(sbatch --parsable $D --job-name="$1" slurm/e28_scaleinit.sbatch $ARGS tag="$1" \
               +e28.scale="$2")
    echo "$1 seg$i=$J"
    D="--dependency=afterany:$J"
  done
}

case "$1" in
  sweep)
    sbatch --parsable --job-name=e28-sweep $SMALL --export=ALL,E28_MODE=initsweep \
           slurm/e28_scaleinit.sbatch $ARGS tag=d256vm4zonly ;;
  x10)
    chain e28x10 10.0 "$2" ;;
  lowsweep)
    G1=$(sbatch --parsable --job-name=e28-lowsweep $SMALL --export=ALL,E28_MODE=initsweep \
                slurm/e28_scaleinit.sbatch $ARGS tag=d256vm4zonly +e28.scale=0.1 \
                '+e28.scales=[0.1,0.3,0.5]')
    G2=$(sbatch --parsable --job-name=e28-lowviews $SMALL --export=ALL,E28_MODE=viewsweep \
                slurm/e28_scaleinit.sbatch $ARGS tag=d256vm4zonly +e28.scale=1.0 \
                '+e28.scales=[0.1]')
    echo "gate_x0.1=$G1 gate_x1=$G2" ;;
  x0.1)
    chain e28x0.1 0.1 "$2" ;;
  x1)
    chain e28x1 1.0 "$2" ;;
  *)
    sed -n '2,13p' "$0"; exit 1 ;;
esac
