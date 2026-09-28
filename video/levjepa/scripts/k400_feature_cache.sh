#!/bin/bash
# K400 feature cache for the linear read:  bash scripts/k400_feature_cache.sh <out dir> name=<ckpt> [name=<ckpt> ...]
# One process per GPU (RANK / WORLD_SIZE / LOCAL_RANK from your launcher); then, from the package root:
#   python experiments/k400_features_read.py --cache <out dir>
set -e
cd "$(dirname "$0")/.."
export PYTHONPATH="$PWD"
OUT=${1:?out dir}; shift
python -u scripts/k400_feature_cache.py --data "${K400_CLIPFILES:?set to the K400 clip-file store}" --out-dir "$OUT" --bs ${BS:-8} --workers ${NUM_WORKERS:-20} --ckpts "$@"
