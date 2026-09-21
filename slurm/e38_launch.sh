#!/bin/bash
# E38 launch (docs/experiments/E38_in100_seeds.md) — seed repeats of the 14 IN-100 paper cells.
#   bash slurm/e38_launch.sh smoke                 # 1-epoch smoke of every cell at seed=99 (H100)
#   bash slurm/e38_launch.sh chains <seeds> [gate] # e.g. chains "1 2" — 3×8h H100 chain per run
#                                                  #   + landing (extract → probe + thickness);
#                                                  #   [gate] = "smoke" waits on that cell's smoke
#   bash slurm/e38_launch.sh land <run_id>         # (re)queue only the landing chain of one run
#
# CELL RECIPES = the seed-0 checkpoint's stored cfg, recovered by scratch/e38/overrides.py (a diff
# against today's composed defaults), NOT retyped from memory. Only `seed=` changes. Loader keys
# (D-057 fast defaults) are not part of the recipe. bs is written explicitly even where it equals
# the train.yaml default so the recipe survives a default change: the July lanes (VICReg, SimCLR,
# BYOL — control AND treated) trained at bs=256; LeJEPA, DINO, VISReg and Ours at bs=128.
set -e
cd /nfs/scistore19/locatgrp/bdemirel/ssl_project
OUT=/nfs/scistore19/locatgrp/bdemirel/ssl_project/outputs

declare -A CELL     # run stem (without seed) -> hydra args
declare -A TAG      # run stem -> tag ("" for controls)
LEJ="method=lejepa frame=in100_vits16 bs=128 num_classes=100 method.lr=0.001 method.warmup_ep=10 method.eta_min=1e-05 method.grad_clip=1.0"
CELL[lejepa]="$LEJ";                                   TAG[lejepa]=""
CELL[lejepa.e20f]="$LEJ +method.h_reg=sacreg +method.h_lamb=0.0146";           TAG[lejepa.e20f]=e20f
VIC="method=vicreg frame=in100_vits16 bs=256 num_classes=100"
CELL[vicreg]="$VIC";                                   TAG[vicreg]=""
CELL[vicreg.e20f]="$VIC +method.h_reg=sacreg +method.h_lamb=1.548 +method.h_tap=cls"; TAG[vicreg.e20f]=e20f
SIM="method=simclr frame=in100_vits16 bs=256 num_classes=100"
CELL[simclr]="$SIM";                                   TAG[simclr]=""
CELL[simclr.e20f]="$SIM +method.h_reg=sacreg +method.h_lamb=0.098";            TAG[simclr.e20f]=e20f
DIN="method=dino frame=in100_vits16 bs=128 num_classes=100 method.ema_base=0.996 method.local_size=96"
CELL[dino]="$DIN";                                     TAG[dino]=""
CELL[dino.e20fwlo]="$DIN +method.h_reg=sacreg +method.h_lamb=0.01";            TAG[dino.e20fwlo]=e20fwlo
BYO="method=byol frame=in100_vits16 bs=256 num_classes=100 method.ema_base=0.996"
CELL[byol]="$BYO";                                     TAG[byol]=""
CELL[byol.e20f]="$BYO +method.h_reg=sacreg +method.h_lamb=0.0205";             TAG[byol.e20f]=e20f
VIS="method=visreg frame=in100_vits16 bs=128 num_classes=100"
CELL[visreg]="$VIS";                                   TAG[visreg]=""
CELL[visreg.visregf]="$VIS +method.h_reg=sacreg +method.h_lamb=0.0123";        TAG[visreg.visregf]=visregf
OURS="method=lambdajepa frame=in100_vits16 bs=128 num_classes=100 +method.aug=lejepa +method.V=4 \
method.expander_dim=256 method.w_inv=32.8 method.w_floor=157.8 method.z_floor_batch=view_mean \
method.z_d_slice=128 method.h_floor_batch=view_mean method.h_d_slice=128 method.queue_steps=3"
CELL[floorssl.d256vm4zonly]="$OURS method.h_lamb=0.0";                         TAG[floorssl.d256vm4zonly]=d256vm4zonly
CELL[floorssl.d256vm4]="$OURS method.h_lamb=1.894";                            TAG[floorssl.d256vm4]=d256vm4
ORDER="lejepa lejepa.e20f vicreg vicreg.e20f simclr simclr.e20f dino dino.e20fwlo byol byol.e20f visreg visreg.visregf floorssl.d256vm4zonly floorssl.d256vm4"
ORDER=${E38_CELLS:-$ORDER}   # E38_CELLS="a b" restricts a mode to those cells

run_id() {  # run_id STEM SEED -> in100.<method>.s<seed>[.<tag>]
  local m=${1%%.*} t=${TAG[$1]}
  echo "in100.${m}.s$2${t:+.$t}"
}

land() {  # land RUN_ID [DEP]: extract .extL (h_layers 3/6/9 + 8-view orbits, the paper cells'
          # store layout) -> the three readers every column of the zoo figure needs:
          #   probes (GPU)            -> results/probes/<rid>.extL.csv
          #   depth metrics (CPU)     -> results/diag/e38/<rid>.extL_depth_metrics.csv (own file per run)
          #   thickness rows (CPU)    -> appended to results/diag/e23_retro_spaces.csv — the append is
          #                              serialized through the shared job name (singleton dependency)
          # Same pipeline as the E29/E37 landings; twospace/audit only on request.
  local rid=$1 dep=${2:+--dependency=afterok:$2}
  local EX; EX=$(sbatch --parsable $dep --job-name="e38x-$rid" slurm/extract.sbatch \
        ckpt="$OUT/${rid}_ep100.pt" adapter=native run_id="$rid.extL" 'h_layers=[3,6,9]' orbit_v=8)
  local PR; PR=$(sbatch --parsable --dependency=afterok:$EX --job-name="e38p-$rid" slurm/probe.sbatch run_id="$rid.extL")
  local DM; DM=$(sbatch --parsable --dependency=afterok:$EX --job-name="e38d-$rid" --partition=defaultp \
        --cpus-per-task=8 --mem=64G --time=03:00:00 -o "$OUT/%x_%j.out" \
        --wrap "bash -c 'cd $PWD && PYTHONPATH=$PWD $PWD/.venv/bin/python experiments/e20f_depth_metrics.py e38/$rid.extL_depth_metrics.csv $rid.extL=$rid.extL'")
  local TH; TH=$(sbatch --parsable --dependency=afterok:$EX,singleton --job-name="e38-thick" --partition=defaultp \
        --cpus-per-task=8 --mem=48G --time=01:00:00 -o "$OUT/%x_%j.out" \
        --wrap "bash -c 'cd $PWD && PYTHONPATH=$PWD $PWD/.venv/bin/python experiments/e23_retro_append.py $rid.extL'")
  echo "  land $rid: extract=$EX probe=$PR depth=$DM thickness=$TH"
  echo "$rid $EX $PR $DM $TH" >> scratch/e38/landing_jobs.txt
}

case "$1" in
  smoke)
    for c in $ORDER; do
      J=$(sbatch --parsable --job-name="e38s-$c" --time=00:40:00 slurm/e38_seeds.sbatch \
             ${CELL[$c]} ${TAG[$c]:+tag=${TAG[$c]}} seed=99 frame.epochs=1)
      echo "smoke $c seed=99: $J"
      echo "$c $J" >> scratch/e38/smoke_jobs.txt
    done ;;
  chains)
    for c in $ORDER; do
      G=""
      if [ "$3" = "smoke" ]; then G=$(awk -v c="$c" '$1==c{j=$2} END{print j}' scratch/e38/smoke_jobs.txt); fi
      for s in $2; do
        rid=$(run_id $c $s)
        D=${G:+--dependency=afterok:$G}
        for i in 1 2 3; do   # 3×8h links: link 1 normally finishes (~6 h on an H100); the rest
                             # resume from _last.pt or find the run complete and exit in seconds
          J=$(sbatch --parsable $D --job-name="e38-$rid" slurm/e38_seeds.sbatch \
                 ${CELL[$c]} ${TAG[$c]:+tag=${TAG[$c]}} seed=$s)
          echo "$rid link$i=$J"
          D="--dependency=afterany:$J"
        done
        echo "$rid $J" >> scratch/e38/chain_tail.txt
        land "$rid" "$J"
      done
    done ;;
  land)
    land "$2" "" ;;
  *)
    sed -n '2,8p' "$0"; exit 1 ;;
esac
