#!/bin/bash
# ImageNet-1k lambda-JEPA:  bash scripts/in1k.sh vits|vitb 100|400
# Two GPUs (one process each), global batch 128. The trainer resumes from outputs/<run>_last.pt, so a
# wall-time-limited scheduler can run this as a chain of segments.
set -e
cd "$(dirname "$0")/.."
ARCH=${1:?vits|vitb}; EP=${2:?100|400}
COMMON="method=lambdajepa num_classes=1000 share_log_every=1 method.head_layers=2 \
method.expander_dim=256 method.mlp_wd=0.05 method.z_floor_batch=view_mean \
method.h_floor_batch=view_mean +method.aug=lejepa +method.V=6 method.floor_shrink=null \
method.queue_steps=3 +method.swa=ema +extra_cadence=[10] bs=128 num_workers=12"
case $ARCH in
  vits) CELL="frame=in1k_vits16 method.h_queue_steps=3 method.h_d_slice=128 method.w_inv=31.07 method.w_floor=225.50 method.h_lamb=1.671";;
  vitb) CELL="frame=in1k_vitb16 method.h_queue_steps=7 method.h_d_slice=256 method.w_inv=22.81 method.w_floor=222.26 method.h_lamb=4.054";;
  *) echo "bad arch $ARCH"; exit 1;;
esac
python experiments/lambdajepa_selftest.py && python experiments/train_ddp_selftest.py
python -m torch.distributed.run --standalone --nproc_per_node=${NGPU:-2} \
  experiments/train_ddp.py $COMMON $CELL frame.epochs=$EP tag="v6${ARCH#vit}${EP}"
