#!/bin/bash
# E23 stage-C landing (one command, run when cells reach ep150): per grid cell with a
# _ep150.pt checkpoint and no .extL store yet — extraction (eval train+val, pairs, o8
# orbits, L-taps 3/6/9 in ONE pass) -> frozen probes + battery (afterany, house gotcha).
# Cells: the 13 stage-C grid tags + e23Llr (the 2048/K2 lane cell) + the 12 stage-C′
# vm-OAS tags (e23c*, launched 2026-08-04 — D-069 as amended; their 2048/K2 refs are
# the landed E24 v-cells). A cell with an existing .extL store dir is SKIPPED (delete
# the dir to re-land).
set -e
cd /nfs/scistore19/locatgrp/bdemirel/ssl_project
CELLS="e23jW32 e23jW64 e23jW128 e23jW256 e23jW512 e23jK0 e23jK1 e23jK3 e23jK4 e23jK6 \
       e23jW64h0 e23jK2h0 e23jK6h0 e23Llr \
       e23cW32 e23cW64 e23cW128 e23cW256 e23cW512 e23cK0 e23cK1 e23cK3 e23cK4 e23cK6 \
       e23cW64h0 e23cK6h0"
for T in $CELLS; do
  R=toy.floorssl.s0.$T
  if [ ! -f outputs/${R}_ep150.pt ]; then echo "[skip] $R: no ep150 ckpt"; continue; fi
  if [ -d features/${R}.extL ]; then echo "[skip] $R: .extL store exists"; continue; fi
  EX=$(sbatch --parsable slurm/extract.sbatch ckpt=outputs/${R}_ep150.pt adapter=native \
       run_id=${R}.extL 'h_layers=[3,6,9]' orbit_v=8)
  PR=$(sbatch --parsable --dependency=afterany:$EX slurm/probe.sbatch run_id=${R}.extL)
  AU=$(sbatch --parsable --dependency=afterany:$EX slurm/audit.sbatch run_id=${R}.extL)
  echo "$R: extract=$EX probe=$PR audit=$AU"
done
echo "verify counts: squeue -u bdemirel | grep -cE 'gap-(extract|probe|audit)'"
