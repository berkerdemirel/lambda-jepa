#!/bin/bash
# ImageNet-1k lambda-JEPA chains: bash slurm/in1k_launch.sh vits|vitb 100|400
set -e
ARCH=${1:?vits|vitb}; EP=${2:?100|400}
COMMON="method=lambdajepa num_classes=1000 share_log_every=1 method.head_layers=2 \
method.expander_dim=256 method.mlp_wd=0.05 method.z_floor_batch=view_mean \
method.h_floor_batch=view_mean +method.aug=lejepa +method.V=6 method.floor_shrink=null \
method.queue_steps=3 +method.swa=ema +extra_cadence=[10] bs=128 num_workers=12"
case $ARCH in
  vits) CELL="frame=in1k_vits16 method.h_queue_steps=3 method.h_d_slice=128 \
method.w_inv=31.07 method.w_floor=225.50 method.h_lamb=1.671";;
  vitb) CELL="frame=in1k_vitb16 method.h_queue_steps=7 method.h_d_slice=256 \
method.w_inv=22.81 method.w_floor=222.26 method.h_lamb=4.054";;
  *) echo "bad arch $ARCH"; exit 1;;
esac
TAG="v6${ARCH#vit}${EP}"
N=$(( EP == 400 ? 3 : 2 ))
for i in $(seq 1 $N); do
  sbatch --gres=gpu:2 --cpus-per-task=28 --mem=200G --time=120:00:00 --dependency=singleton \
    --job-name="$TAG" slurm/in1k_ddp.sbatch $COMMON $CELL frame.epochs=$EP tag=$TAG
done
