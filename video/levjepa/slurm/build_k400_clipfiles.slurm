#!/bin/bash
#SBATCH --job-name=k400_clipfiles
#SBATCH --partition=defaultp
#SBATCH --cpus-per-task=16
#SBATCH --mem=48G
#SBATCH --time=1-00:00:00
#SBATCH --array=0-19
#SBATCH --output=/nfs/scistore19/locatgrp/bdemirel/ssl_project/outputs/%x_%A_%a.out
# K400 evaluation store (E34; scripts/build_clipfiles.py --kinetics): the CVDF tarballs streamed once each, members decoded from
# memory, stored at 15 fps / short edge 256 / JPEG q90 under <out>/{train,val}/{label}/{id}_{start}_{end}.clip; labels from the
# K400 annotation CSV of the split; 20 shards over the tarballs. Usage: sbatch <this> <train|val> [<out root>]
cd /nfs/scistore19/locatgrp/bdemirel/ssl_project/video/levjepa || exit 1
SPLIT=$1; OUT=${2:-/mnt/beegfs/locatgrp/shared/datasets/kinetics/k400_eval_clipfiles}; K=/mnt/beegfs/locatgrp/shared/datasets/kinetics/k400
echo "$(date) $SPLIT shard $SLURM_ARRAY_TASK_ID/$SLURM_ARRAY_TASK_COUNT on $(hostname)"
.venv/bin/python -u scripts/build_clipfiles.py --kinetics --split $SPLIT --out $OUT --tars $K/$SPLIT/*.tar.gz --annotations $K/annotations/$SPLIT.csv \
  --shard $SLURM_ARRAY_TASK_ID/$SLURM_ARRAY_TASK_COUNT --workers 16
