#!/bin/bash
# lambda-JEPA ViT-S/16 on K710-20 %: 240 epochs, 8 GPUs (one process per GPU; DEVICES x NUM_NODES), batch 96 per GPU.
# Resumes the run's newest last.ckpt. Set LEVJEPA_DATA_ROOT; launch one process per GPU under your scheduler.
set -e
LEV="$(cd "$(dirname "$0")/.." && pwd)"
export LEVJEPA_DATA_ROOT="${LEVJEPA_DATA_ROOT:?set to the Kinetics store root}"
RUN=vid.lambdajepa.s0.k710s
RUNDIR="${LEVJEPA_RUNS:-$LEV/outputs/runs}/$RUN"
LAST=$(ls -t "$RUNDIR"/spt_cache/runs/*/*/*/checkpoints/last.ckpt 2>/dev/null | head -n 1)
[ -n "$LAST" ] && RESUME="resume.ckpt_path=$LAST resume.weights_only=false"
mkdir -p "$RUNDIR" && cd "$RUNDIR"
python -u "$LEV/main.py" --config-name lambdajepa_vits data=k710_clipfiles \
  trainer.devices=${DEVICES:-1} trainer.num_nodes=${NUM_NODES:-8} trainer.max_epochs=240 accumulate_grad_batches=1 \
  loss.lambdajepa.queue_steps=0 loss.lambdajepa.h_queue_steps=0 scheduler.peak_step=4493 +probe_k710=1024 \
  loader.num_workers=${NUM_WORKERS:-22} augmentation.photometrics_on_gpu=true \
  checkpoint.every_n_epochs=1 checkpoint.every_n_train_steps=0 \
  wandb.enabled=true wandb.config.name=$RUN hydra.run.dir="$RUNDIR" $RESUME
