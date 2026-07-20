#!/bin/bash
# E14 addendum no-fovea (blur_v1) fleet: 22 jobs, 2 H100 singleton slots, alternating.
# ckpt/adapter per run copied from the store's ckpt_provenance (generated 2026-07-12).
# dino/randinit/mae are tokenizers -> foveal=both; everyone else foveal=event.
set -e
cd /nfs/scistore19/locatgrp/bdemirel/ssl_project

submit() {  # slot run_id ckpt adapter foveal extra...
  local slot=$1 run=$2 ckpt=$3 adapter=$4 fov=$5; shift 5
  sbatch --partition=gpu100 --constraint=H100 --job-name="h100-slot${slot}" \
         --dependency=singleton --time=01:00:00 slurm/extract.sbatch \
         ckpt="$ckpt" adapter="$adapter" run_id="$run" \
         do_eval=false do_pairs=false foveal="$fov" "$@"
}

i=0
while IFS='|' read -r run ckpt adapter fov extra; do
  slot=$( ((i % 2)) && echo B || echo A )
  submit "$slot" "$run" "$ckpt" "$adapter" "$fov" $extra
  i=$((i + 1))
done << 'EOF'
in100.dino.s0.ext|outputs/in100.dino.s0_ep100.pt|native|blur|
in100.simclr.s0.ext|outputs/in100.simclr.s0_ep100.pt|native|blur|
in100.byol.s0.ext|outputs/in100.byol.s0_ep100.pt|native|blur|
in100.vicreg.s0.ext|outputs/in100.vicreg.s0_ep100.pt|native|blur|
in100.ijepa.s0.ext|outputs/in100.ijepa.s0_ep100.pt|native|blur|
in100.lejepa.s0.ext|outputs/in100.lejepa.s0_ep100.pt|native|blur|
in100.deitlite.s0.ext|outputs/in100.deitlite.s0_ep100.pt|native|blur|
in100.dino-ctrl.ep25.ext|/nfs/scistore19/locatgrp/bdemirel/ssl_explore/outputs/inv_dino-in100_ep25.pt|sslx_dino|blur|
in100.dino-ctrl.ep50.ext|/nfs/scistore19/locatgrp/bdemirel/ssl_explore/outputs/inv_dino-in100_ep50.pt|sslx_dino|blur|
in100.dino-ctrl.ep100.ext|/nfs/scistore19/locatgrp/bdemirel/ssl_explore/outputs/inv_dino-in100_ep100.pt|sslx_dino|blur|
in100.lejepa.s0.e12a1.ext|outputs/in100.lejepa.s0.e12a1_ep100.pt|native|blur|
in100.lejepa.s0.e12a2.ext|outputs/in100.lejepa.s0.e12a2_ep100.pt|native|blur|
in100.lejepa.s0.e12a3.ext|outputs/in100.lejepa.s0.e12a3_ep100.pt|native|blur|
in100.lejepa.s0.e12c1.ext|outputs/in100.lejepa.s0.e12c1_ep100.pt|native|blur|
in100.lejepa.s0.e12f1.ext|outputs/in100.lejepa.s0.e12f1_ep100.pt|native|blur|
in100.lejepa.s0.e12f2.ext|outputs/in100.lejepa.s0.e12f2_ep100.pt|native|blur|
in100.lejepa.s0.e12f3.ext|outputs/in100.lejepa.s0.e12f3_ep100.pt|native|blur|
in100.lejepa.s0.e12f4.ext|outputs/in100.lejepa.s0.e12f4_ep100.pt|native|blur|
in100.lejepa.s0.e12f5.ext|outputs/in100.lejepa.s0.e12f5_ep100.pt|native|blur|
in100.lejepa.s0.e12f6.ext|outputs/in100.lejepa.s0.e12f6_ep100.pt|native|blur|
in100.randinit-s0.ext|/nfs/scistore19/locatgrp/bdemirel/ssl_explore/outputs/inv_dino-in100_ep100.pt|sslx_dino|blur|random_init=true
in100.mae.s0.ext|outputs/in100.mae.s0_ep100.pt|native|blur|
EOF
echo "submitted $i jobs"
