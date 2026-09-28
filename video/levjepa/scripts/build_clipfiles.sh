#!/bin/bash
# Evaluation clip files:  bash scripts/build_clipfiles.sh k400|ssv2 train|val [out dir] [shard i/n]
set -e
cd "$(dirname "$0")/.."
DS=${1:?k400|ssv2}; SPLIT=${2:?train|val}; SHARD=${4:-0/1}
case $DS in
  k400) OUT=${3:-"${K400_CLIPFILES:?set to the K400 clip-file store}"}; K="${LEVJEPA_DATA_ROOT:?set to the Kinetics store root}"/k400
        python -u scripts/build_clipfiles.py --kinetics --split $SPLIT --out "$OUT" --tars $K/$SPLIT/*.tar.gz --annotations $K/annotations/$SPLIT.csv --shard $SHARD --workers 16 ;;
  ssv2) OUT=${3:-"${SSV2_CLIPFILES:?set to the SSv2 clip-file store}"}
        python -u scripts/build_clipfiles.py --ssv2 --split $SPLIT --out "$OUT" --shard $SHARD --workers 16 ;;
  *) echo "bad dataset $DS"; exit 1 ;;
esac
