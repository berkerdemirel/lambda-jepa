#!/bin/bash
# Stage the K710-20% mp4 clips (3 dirs, ~155 GB, 120k files) from BeeGFS into this node's RAM disk with 16
# parallel copy streams. Run once per node at job start (srun --ntasks-per-node=1). Idempotent: skips a
# set whose file count already matches. Prints the copy rate. Then LEVJEPA_MP4_ROOT=/dev/shm/k710.
SRC=/mnt/beegfs/locatgrp/shared/datasets/kinetics; DST=/dev/shm/k710; mkdir -p "$DST"
t0=$(date +%s)
for s in k710_20pct_videos_k700_2020 k710_20pct_videos_k600 k710_20pct_videos_k400; do
  want=$(find "$SRC/$s" -name '*.mp4' | wc -l); have=$(find "$DST/$s" -name '*.mp4' 2>/dev/null | wc -l)
  if [ "$have" -ge "$want" ]; then echo "$(hostname) $s: already staged ($have files)"; continue; fi
  (cd "$SRC/$s" && find . -type d -exec mkdir -p "$DST/$s/{}" \; && find . -name '*.mp4' -print0 | xargs -0 -P 16 -I{} cp "{}" "$DST/$s/{}")
  echo "$(hostname) $s: $(find "$DST/$s" -name '*.mp4' | wc -l)/$want files staged"
done
gb=$(du -sb "$DST" | awk '{printf "%.0f", $1/1e9}'); dt=$(( $(date +%s) - t0 ))
echo "$(hostname) STAGED ${gb} GB in ${dt}s ($(( gb * 1000 / (dt > 0 ? dt : 1) )) MB/s); free RAM: $(free -g | awk '/Mem:/{print $7}') GB"
