#!/bin/bash
# lambda-JEPA ViT-B/16 on K710-20 %: 8 GPUs (one process per GPU; DEVICES x NUM_NODES), batch 64 per GPU.
# MAX_EPOCHS=515 first, then MAX_EPOCHS=1085 resumes the same run; epoch-0239.ckpt is kept for the 240-epoch row.
set -e
LEV="$(cd "$(dirname "$0")/.." && pwd)"
export LEVJEPA_DATA_ROOT="${LEVJEPA_DATA_ROOT:?set to the Kinetics store root}"
RUN=vid.lambdajepa.s0.k710b
RUNDIR="${LEVJEPA_RUNS:-$LEV/outputs/runs}/$RUN"
MAX_EPOCHS=${MAX_EPOCHS:-515}
LAST=$(ls -t "$RUNDIR"/spt_cache/runs/*/*/*/checkpoints/last.ckpt 2>/dev/null | head -n 1)
[ -n "$LAST" ] && RESUME="resume.ckpt_path=$LAST resume.weights_only=false"
mkdir -p "$RUNDIR" && cd "$RUNDIR"
python -u "$LEV/main.py" --config-name lambdajepa_vitb data=k710_clipfiles loader.batch_size=64 loss.lambdajepa.share_log_bs=64 +probe_k710=1024 \
  trainer.devices=${DEVICES:-1} trainer.num_nodes=${NUM_NODES:-8} trainer.max_epochs=$MAX_EPOCHS accumulate_grad_batches=1 \
  loss.lambdajepa.queue_steps=0 loss.lambdajepa.h_queue_steps=1 scheduler.peak_step=6739 \
  loader.num_workers=${NUM_WORKERS:-48} augmentation.photometrics_on_gpu=true \
  checkpoint.every_n_epochs=1 checkpoint.every_n_train_steps=0 checkpoint.save_top_k=0 checkpoint.save_last=true +keep.epochs=[239] +keep.best=true \
  wandb.enabled=true wandb.config.name=$RUN hydra.run.dir="$RUNDIR" $RESUME
