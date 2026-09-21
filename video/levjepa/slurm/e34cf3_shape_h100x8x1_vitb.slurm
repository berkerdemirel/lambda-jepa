#!/bin/bash
#SBATCH --job-name=e34cf3_vitb_k710_h100x8x1
#SBATCH --partition=gpu100
#SBATCH --constraint=H100
#SBATCH --nodes=8
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=48
#SBATCH --mem=600G
#SBATCH --time=2-00:00:00
#SBATCH --exclude=gpu274,gpu277,gpu271
#SBATCH --output=/nfs/scistore19/locatgrp/bdemirel/ssl_project/outputs/%x_%j.out
# E34 B CELL on the 8 x 1 H100 shape (Berker 2026-09-06 17:30: "yes lets do that update (and do a chain in case it is necessary)" — the FLOP-matched
# schedule, 515 epochs = LeVJEPA's Table 3 budget (1,085 epochs at V = 10) at our 942 retained tokens per clip; the 240-epoch state stays as
# epoch-0239.ckpt for the grid rows since every epoch's checkpoint is kept). Settings = the 2 x 4 launcher's (batch 64 x 8 ranks = 512 per
# optimizer step, 234 steps per epoch; z ring off, h ring q1; peak_step 6,739; lr 4e-4; doses = e27v6b400; online K710 probe; share logger
# at bs 64; checkpoints = the rolling last.ckpt + epoch-0239.ckpt + best.ckpt by the online K710 probe, Berker 2026-09-06) on eight BeeGFS clients instead of two (the per-node read cap: ~25-35 clips/s per node; the 2 x 4 shape would take ~30 min per
# epoch). MAX_EPOCHS env (default 515). Optional node-local staging: STAGE_LOCAL=1 copies each rank's fixed shard of the clip files to the
# node's local disk once (data.stage_local, trainer.use_distributed_sampler=false). Every launch resumes the newest last.ckpt.
# SHAPE from the sbatch line (21:5x, Berker: B start is the priority; the pool is full): --nodes / --ntasks-per-node / --gres / --cpus-per-task /
# --mem override the directives above; trainer.devices = tasks per node, num_nodes = nodes; NUM_WORKERS env (default 48; 24 for 24-CPU asks).
# Any uniform shape with 8 ranks (8x1, 4x2, 2x4) gives batch 64 x 8 = 512; with staging the per-node read cap no longer matters.
LEV=/nfs/scistore19/locatgrp/bdemirel/ssl_project/video/levjepa
export LEVJEPA_DATA_ROOT=/mnt/beegfs/locatgrp/shared/datasets/kinetics
export HYDRA_FULL_ERROR=1
export WANDB_ENTITY=causal-learning-ai-ista
RUN=vid.floorssl.s0.k710b
RUNDIR=/mnt/beegfs/locatgrp/shared/bdemirel/levjepa_runs/$RUN
MAX_EPOCHS=${MAX_EPOCHS:-515}
LAST=$(ls -t $RUNDIR/spt_cache/runs/*/*/*/checkpoints/last.ckpt 2>/dev/null | head -n 1)
if [ -n "$LAST" ]; then
  RESUME="resume.ckpt_path=$LAST resume.weights_only=false"
fi
STAGE=""
if [ "${STAGE_LOCAL:-0}" = "1" ]; then
  STAGE="+data.stage_local=${SLURM_TMPDIR:-/tmp}/lev_$SLURM_JOB_ID +trainer.use_distributed_sampler=false"
fi
# node-local hygiene: staging dirs of jobs that are no longer running (wall-time kills skip the cleanup below) are removed first
srun --ntasks-per-node=1 bash -c 'for d in ${SLURM_TMPDIR:-/tmp}/lev_* ${SLURM_TMPDIR:-/tmp}/vp_*; do [ -d "$d" ] || continue; id=${d##*_}; squeue -h -j "$id" -o %i 2>/dev/null | grep -q . || rm -rf "$d"; done; df -h ${SLURM_TMPDIR:-/tmp} | tail -1'
mkdir -p $RUNDIR && cd $RUNDIR || exit 1
echo "nodes: $SLURM_JOB_NODELIST | max_epochs $MAX_EPOCHS | resume: ${LAST:-fresh start} | stage: ${STAGE:-none} | wandb sidecar: $(cat wandb_resume.json 2>/dev/null || echo none)"; nvidia-smi -L | head -n 1
srun $LEV/.venv/bin/python -u $LEV/main.py --config-name lambdajepa_vitb data=k710_clipfiles loader.batch_size=64 loss.lambdajepa.share_log_bs=64 +probe_k710=1024 \
  trainer.devices=${SLURM_NTASKS_PER_NODE:-1} trainer.num_nodes=$SLURM_NNODES trainer.max_epochs=$MAX_EPOCHS accumulate_grad_batches=1 loss.lambdajepa.queue_steps=0 loss.lambdajepa.h_queue_steps=1 scheduler.peak_step=6739 \
  loader.num_workers=${NUM_WORKERS:-48} augmentation.photometrics_on_gpu=true \
  checkpoint.every_n_epochs=1 checkpoint.every_n_train_steps=0 checkpoint.save_top_k=0 checkpoint.save_last=true +keep.epochs=[239] +keep.best=true \
  wandb.enabled=true wandb.config.project=sslgap wandb.config.name=$RUN \
  hydra.run.dir=$RUNDIR $RESUME $STAGE
rc=$?; [ "${STAGE_LOCAL:-0}" = "1" ] && srun --ntasks-per-node=1 bash -c "rm -rf ${SLURM_TMPDIR:-/tmp}/lev_$SLURM_JOB_ID"
echo "E34_TRAIN_EXIT rc=$rc"
