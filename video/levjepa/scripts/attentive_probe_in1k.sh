#!/bin/bash
# ImageNet-1k attentive probe:  bash scripts/attentive_probe_in1k.sh <lightning .ckpt> <out csv> [ema|student]   (NGPU processes)
set -e
cd "$(dirname "$0")/.."
export PYTHONPATH="$PWD"
torchrun --standalone --nproc_per_node=${NGPU:-8} scripts/attentive_probe.py --ckpt "${1:?ckpt}" --out "${2:?out csv}" --weights "${3:-ema}" --bs 16 --workers 8
