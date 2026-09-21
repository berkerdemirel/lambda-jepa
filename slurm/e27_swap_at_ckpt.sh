#!/bin/bash
#SBATCH --job-name=e27_swap
#SBATCH --partition=defaultp
#SBATCH --cpus-per-task=1
#SBATCH --mem=1G
#SBATCH --time=06:00:00
#SBATCH --output=/nfs/scistore19/locatgrp/bdemirel/ssl_project/outputs/%x_%j.out
# Re-shaping a running IN-1k chain segment (Berker 2026-09-07: "first queue then cancel the ongoing job so that we increase the chance of
# securing resources"): the faster segment is ALREADY queued behind the running one (singleton); this job waits for the next `_last.pt`
# write of the run and cancels the running segment within a minute of it, so at most a minute of the next epoch is lost and the queued
# segment resumes that checkpoint. Runs on the cluster so the login node's memory pressure cannot kill it.
# Usage: sbatch slurm/e27_swap_at_ckpt.sh <running jobid> <path of _last.pt>
J=$1; CK=$2; m0=$(stat -c %Y "$CK"); echo "$(date '+%m-%d %H:%M') watching job $J for a new write of $CK (mtime $(date -d @$m0 '+%H:%M'))"
while true; do
  squeue -h -j "$J" -t RUNNING -o %i | grep -q "$J" || { echo "$(date '+%m-%d %H:%M') job $J is no longer running — nothing to do"; exit 0; }
  m=$(stat -c %Y "$CK")
  if [ "$m" != "$m0" ]; then
    sleep 45   # let the write settle (the trainer writes _last then _best)
    scancel "$J"; echo "$(date '+%m-%d %H:%M') checkpoint written at $(date -d @$m '+%H:%M') -> job $J cancelled; the queued segment takes over. SWAP_DONE"; exit 0
  fi
  sleep 30
done
