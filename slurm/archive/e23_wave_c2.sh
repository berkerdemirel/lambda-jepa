#!/bin/bash
# E23 stage-C′ (D-069 USER-APPROVED-AS-AMENDED, Berker 2026-08-04: "redo our grid search
# on toy: mainly playing with width and depth … always using the mean of the views for
# the regularization on both h and z"): the capacity grid on the vm-OAS anatomy (D-075
# toy estimator policy). WIDTH ladder K=2 hidden {32,64,128,256,512} + DEPTH ladder
# hidden 2048 K {0,1,3,4,6} + h0 twins {W64h0, K6h0}; the 2048/K2 references are the
# landed E24 v-cells (vc/v4oas/vb; vh0 = h0 ref). Uniform prior dose = vc's measured
# basis (16.2/46.7/0.73 — the profile-pinned (.47/.50/.03) W2048K2 cell); per-epoch
# share/ρ̂ logging from birth; ep10 confirm reads; corrections per-case (completeness
# rule — originals run on as map points). ep150, 2×8h links, gpu partition.
# Usage: bash slurm/e23_wave_c2.sh [smoke|wave]
set -e
cd /nfs/scistore19/locatgrp/bdemirel/ssl_project

COMMON="method=floorssl frame=toy_vits8 bs=128 seed=0 share_log_every=1 +method.aug=lejepa +method.V=4 method.lr=5e-4 method.expander_dim=256 method.mlp_wd=0.05 method.z_floor_batch=view_mean method.h_floor_batch=view_mean method.floor_shrink=oas method.w_inv=16.2 method.w_floor=46.7"
H="method.h_lamb=0.73"
H0="method.h_lamb=0.0"

declare -A CELLS=(
  [e23cW32]="method.head_layers=2 method.expander_hidden=32 $H"
  [e23cW64]="method.head_layers=2 method.expander_hidden=64 $H"
  [e23cW128]="method.head_layers=2 method.expander_hidden=128 $H"
  [e23cW256]="method.head_layers=2 method.expander_hidden=256 $H"
  [e23cW512]="method.head_layers=2 method.expander_hidden=512 $H"
  [e23cK0]="method.head_layers=0 method.expander_hidden=2048 $H"
  [e23cK1]="method.head_layers=1 method.expander_hidden=2048 $H"
  [e23cK3]="method.head_layers=3 method.expander_hidden=2048 $H"
  [e23cK4]="method.head_layers=4 method.expander_hidden=2048 $H"
  [e23cK6]="method.head_layers=6 method.expander_hidden=2048 $H"
  [e23cW64h0]="method.head_layers=2 method.expander_hidden=64 $H0"
  [e23cK6h0]="method.head_layers=6 method.expander_hidden=2048 $H0"
)

if [ "${1:-wave}" = "smoke" ]; then
  for t in e23cW32 e23cK0 e23cK6; do
    sbatch --begin=now+120 --job-name=e23c-smk-$t slurm/train.sbatch \
      ${CELLS[$t]} $COMMON frame.epochs=2 tag=${t}smk
  done
  echo "3 smokes fired (2 ep each; check [share] lines + rho_{z,h} + ckpt, then: bash $0 wave)"
  exit 0
fi

for t in "${!CELLS[@]}"; do
  j=$(sbatch --parsable --begin=now+120 --job-name=e23c-toy-$t slurm/train.sbatch \
      ${CELLS[$t]} $COMMON tag=$t)
  j=$(sbatch --parsable --dependency=afterany:$j --job-name=e23c-toy-$t slurm/train.sbatch \
      ${CELLS[$t]} $COMMON tag=$t)
  echo "$t tail=$j"
done
