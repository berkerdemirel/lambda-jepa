#!/bin/bash
# E24 landing (adapted from e23_land_grid.sh): per completed cell with an _ep150.pt (toy)
# or _ep100.pt (in100) checkpoint and no .extL store yet — extraction (eval train+val,
# pairs, o8 orbits, L-taps 3/6/9 in ONE pass) -> frozen probes + battery (afterany).
# A cell with an existing .extL store dir is SKIPPED (delete the dir to re-land).
set -e
cd /nfs/scistore19/locatgrp/bdemirel/ssl_project
TOY="s0.e24s0a s0.e24s0b s0.e24s0c s0.e24s0d s0.e24s1a s0.e24s1b s0.e24s1c \
     s0.e24s0ac s0.e24s0bc s0.e24s0cc s0.e24s1ac s0.e24s1bc s0.e24s1cc \
     s0.e24wz05 s0.e24wz12 s0.e24wz18 s0.e24wz25 s0.e24wt05 s0.e24wt2 s1.e24wrep \
     s0.e24v1 s0.e24v2lr s0.e24v3nq \
     s0.e24v4oas s0.e24va s0.e24vb s0.e24vc s0.e24vh0 s0.e24vac"
IN100="s0.e24s0a s0.e24s0b s0.e24s0c s0.e24s0d s0.e24s1a s0.e24s1b s0.e24s1c s0.e24s1d \
       s0.e24s0ac s0.e24s0cc s0.e24s0dc s0.e24s1ac s0.e24s1bc s0.e24s1dc \
       s0.e24va s0.e24vb s0.e24vc s0.e24vh0 s0.e24vac s0.e24vcc"
for T in $TOY; do
  R=toy.floorssl.$T
  if [ ! -f outputs/${R}_ep150.pt ]; then echo "[skip] $R: no ep150 ckpt"; continue; fi
  if [ -d features/${R}.extL ]; then echo "[skip] $R: .extL store exists"; continue; fi
  EX=$(sbatch --parsable --begin=now+120 slurm/extract.sbatch ckpt=outputs/${R}_ep150.pt \
       adapter=native run_id=${R}.extL 'h_layers=[3,6,9]' orbit_v=8)
  PR=$(sbatch --parsable --dependency=afterany:$EX slurm/probe.sbatch run_id=${R}.extL)
  AU=$(sbatch --parsable --dependency=afterany:$EX slurm/audit.sbatch run_id=${R}.extL)
  echo "$R: extract=$EX probe=$PR audit=$AU"
done
for T in $IN100; do
  R=in100.floorssl.$T
  if [ ! -f outputs/${R}_ep100.pt ]; then echo "[skip] $R: no ep100 ckpt"; continue; fi
  if [ -d features/${R}.extL ]; then echo "[skip] $R: .extL store exists"; continue; fi
  EX=$(sbatch --parsable --begin=now+120 slurm/extract.sbatch ckpt=outputs/${R}_ep100.pt \
       adapter=native run_id=${R}.extL 'h_layers=[3,6,9]' orbit_v=8)
  PR=$(sbatch --parsable --dependency=afterany:$EX slurm/probe.sbatch run_id=${R}.extL)
  AU=$(sbatch --parsable --dependency=afterany:$EX slurm/audit.sbatch run_id=${R}.extL)
  echo "$R: extract=$EX probe=$PR audit=$AU"
done
echo "verify counts: squeue -u bdemirel | grep -cE 'gap-(extract|probe|audit)'"
