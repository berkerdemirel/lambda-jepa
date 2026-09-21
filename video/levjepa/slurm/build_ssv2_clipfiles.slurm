#!/bin/bash
#SBATCH --job-name=ssv2_clipfiles
#SBATCH --partition=defaultp
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --array=0-7
#SBATCH --output=/nfs/scistore19/locatgrp/bdemirel/ssl_project/outputs/%x_%A_%a.out
# SSv2 evaluation store (E34; scripts/build_clipfiles.py --ssv2): native 12 fps, native 240p, JPEG q90, one clip file per
# video under <out>/{train,validation}/{template}/{id}.clip; 8 shards per split. Usage: sbatch <this> <split> [<out root>]
cd /nfs/scistore19/locatgrp/bdemirel/ssl_project/video/levjepa || exit 1
SPLIT=$1; OUT=${2:-/mnt/beegfs/locatgrp/shared/datasets/ssv2/ssv2_clipfiles}
echo "$(date) $SPLIT shard $SLURM_ARRAY_TASK_ID/$SLURM_ARRAY_TASK_COUNT on $(hostname)"
.venv/bin/python -u scripts/build_clipfiles.py --ssv2 --split $SPLIT --out $OUT --shard $SLURM_ARRAY_TASK_ID/$SLURM_ARRAY_TASK_COUNT --workers 16
