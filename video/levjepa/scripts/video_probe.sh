#!/bin/bash
# SSv2 / K400 frozen probes:  bash scripts/video_probe.sh ssv2|k400 attentive|linear_mean <ckpt> <out csv> [ema|student] [bs]
# One process per GPU (RANK / WORLD_SIZE / LOCAL_RANK from your launcher). Data roots: $SSV2_CLIPFILES, $K400_CLIPFILES.
set -e
cd "$(dirname "$0")/.."
export PYTHONPATH="$PWD"
DS=${1:?ssv2|k400}; HEAD=${2:?attentive|linear_mean}; CKPT=${3:?ckpt}; OUT=${4:?out csv}; W=${5:-ema}; BS=${6:-4}
case $DS in ssv2) DATA="${SSV2_CLIPFILES:?set to the SSv2 clip-file store}";; k400) DATA="${K400_CLIPFILES:?set to the K400 clip-file store}";; *) echo "bad dataset $DS"; exit 1;; esac
python -u scripts/video_probe.py --dataset "$DS" --head "$HEAD" --ckpt "$CKPT" --out "$OUT" --weights "$W" --bs "$BS" --data "$DATA" --workers ${NUM_WORKERS:-12}
