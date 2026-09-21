#!/bin/bash
#SBATCH --job-name=e34b_first_wins
#SBATCH --partition=defaultp
#SBATCH --cpus-per-task=1
#SBATCH --mem=1G
#SBATCH --time=3-00:00:00
#SBATCH --output=/nfs/scistore19/locatgrp/bdemirel/ssl_project/outputs/%x_%j.out
# The B cell base is queued in several uniform shapes (8x1 / 4x2 / 2x4; Berker 2026-09-06 21:5x "b start is the priority"; the H100 pool
# is full). Keep the FIRST shape that starts and survives 3 minutes; cancel the others; chain two 8x1 successors afterany the winner
# (MAX_EPOCHS 515, staging); point the S probes (after:) at the winner. Runs as a 1-CPU cluster job so the login node's memory pressure
# cannot kill it (the in-session waiter was killed for that reason). Usage: sbatch e34b_first_wins.sh <shape ids...> -- <probe ids...>
cd /nfs/scistore19/locatgrp/bdemirel/ssl_project || exit 1
L=video/levjepa/slurm/e34cf3_shape_h100x8x1_vitb.slurm; E="--export=ALL,STAGE_LOCAL=1,MAX_EPOCHS=515,NUM_WORKERS=24"
ids=(); probes=(); sep=0
for a in "$@"; do [ "$a" = "--" ] && { sep=1; continue; }; [ $sep -eq 0 ] && ids+=("$a") || probes+=("$a"); done
echo "$(date '+%m-%d %H:%M') watching shapes ${ids[*]}; probes ${probes[*]}"
while true; do
  alive=0
  for j in "${ids[@]}"; do
    st=$(squeue -h -j "$j" -o %T 2>/dev/null | head -1); [ -n "$st" ] && alive=1
    if [ "$st" = "RUNNING" ]; then
      echo "$(date '+%m-%d %H:%M') shape $j ($(squeue -h -j $j -o %j)) RUNNING on $(squeue -h -j $j -o %N) — grace 180 s"; sleep 180
      log=$(ls outputs/e34b_*_$j.out 2>/dev/null | head -1)
      if squeue -h -j "$j" -o %T | grep -q RUNNING && ! grep -q -E "Traceback|E34_TRAIN_EXIT|CUDA error|NCCL error" "$log" 2>/dev/null; then
        for k in "${ids[@]}"; do [ "$k" != "$j" ] && scancel "$k"; done
        S1=$(sbatch --parsable $E -J e34b_8x1 --nodes=8 --ntasks-per-node=1 --gres=gpu:1 --cpus-per-task=24 --mem=300G --dependency=afterany:$j $L)
        S2=$(sbatch --parsable $E -J e34b_8x1 --nodes=8 --ntasks-per-node=1 --gres=gpu:1 --cpus-per-task=24 --mem=300G --dependency=afterany:$S1 $L)
        for p in "${probes[@]}"; do scontrol update JobId=$p Dependency=after:$j; done
        echo "$(date '+%m-%d %H:%M') WINNER $j; others cancelled; successors $S1 -> $S2; probes after:$j"
        grep -E "nodes:|\[stage\]|ClipFileDataset" "$log" | head -12; echo "FIRST_WINS_DONE winner=$j successors=$S1,$S2"; exit 0
      fi
      echo "$(date '+%m-%d %H:%M') shape $j died at start ($log):"; grep -E "Traceback|Error|E34_TRAIN_EXIT" "$log" | tail -3; scancel "$j"
    fi
  done
  [ "$alive" -eq 0 ] && { echo "no B shapes left in the queue"; exit 1; }
  sleep 60
done
