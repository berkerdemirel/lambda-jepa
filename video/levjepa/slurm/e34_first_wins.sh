#!/bin/bash
# The E34 first cell is submitted as several mutually exclusive job shapes (same run dir, same wandb
# name). Keep the FIRST shape that starts AND survives its first 3 minutes (no NVML/driver death, no
# traceback), then cancel the still-pending others. A shape that dies at start is dropped and the
# loop continues with the rest. Usage: e34_first_wins.sh <jobid> [<jobid> ...]
ids=("$@")
while true; do
  alive=0
  for j in "${ids[@]}"; do
    st=$(squeue -h -j "$j" -o "%T" 2>/dev/null | head -1)
    [ -n "$st" ] && alive=1
    if [ "$st" = "RUNNING" ]; then
      echo "$(date '+%m-%d %H:%M') shape $j RUNNING on $(squeue -h -j $j -o %N) — grace check 180 s"
      sleep 180
      log=$(ls outputs/e34_vits_k710_*_$j.out 2>/dev/null | head -1)
      if squeue -h -j "$j" -o "%T" | grep -q RUNNING && ! grep -q -E "Failed to get device count: Uninitialized|version mismatch|Traceback|E34_TRAIN_EXIT" "$log" 2>/dev/null; then
        echo "$(date '+%m-%d %H:%M') shape $j healthy after 3 min — cancelling the other shapes: ${ids[*]/$j/}"
        for k in "${ids[@]}"; do [ "$k" != "$j" ] && scancel "$k" 2>/dev/null; done
        exit 0
      fi
      echo "$(date '+%m-%d %H:%M') shape $j died or is unhealthy at start ($log) — cancelling it, the others stay queued"
      scancel "$j" 2>/dev/null
    fi
  done
  [ "$alive" -eq 0 ] && { echo "no shapes left in the queue"; exit 1; }
  sleep 60
done
