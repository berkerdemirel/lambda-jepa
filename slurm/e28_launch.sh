#!/bin/bash
# E28 launch (docs/experiments/E28_init_scale.md). ARGS = the d256vm4zonly override set
# VERBATIM (recovered from the reference checkpoint's stored cfg, not retyped from memory) —
# note there is deliberately NO +method.V=4: that lane rode the code default, and the startup
# resolved-config guard fails the job on any key that differs from the reference beyond
# `tag` and `e28.*`.
#
# ALREADY LANDED, DO NOT RE-RUN (the diagnostic CSVs are append-only):
#   job 63437128 = selftest + `E28_MODE=control` (the ×1 reference curve at ep0/25/50/75/100).
# Every training segment re-runs the selftest as its own gate, so the chain is self-gating.
set -e
cd /nfs/scistore19/locatgrp/bdemirel/ssl_project

ARGS="method=floorssl frame=in100_vits16 bs=128 num_classes=100 +method.aug=lejepa \
method.expander_dim=256 method.w_inv=32.8 method.w_floor=157.8 method.h_lamb=0.0 \
method.z_floor_batch=view_mean method.z_d_slice=128 \
method.h_floor_batch=view_mean method.h_d_slice=128 method.queue_steps=3"

# (1) init-scale sweep — h/z geometry at step 0 vs scale, both patch_embed variants, plus the
# per-block rank profile. No training, minutes, any GPU. Berker's "why only 180" question.
S=$(sbatch --parsable --job-name=e28-sweep --partition=gpu --constraint="A100|A40|L40S" \
           --time=01:00:00 --mem=64G --cpus-per-task=8 --export=ALL,E28_MODE=initsweep \
           slurm/e28_scaleinit.sbatch $ARGS tag=d256vm4zonly)
echo "sweep=$S"

# (2) the run: ×10 on the 48 block weight matrices AND patch_embed.proj (Berker 2026-08-13),
# 3×8h H100 segments (~6 min/epoch ⇒ 100 epochs ≈ 10 h), resume-safe via _last.pt.
D=""
for i in 1 2 3; do
  J=$(sbatch --parsable $D slurm/e28_scaleinit.sbatch $ARGS tag=e28x10 \
             +e28.scale=10.0 +e28.patch_embed=true)
  echo "seg$i=$J"
  D="--dependency=afterany:$J"
done
