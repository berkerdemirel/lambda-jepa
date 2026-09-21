#!/bin/bash
#SBATCH --job-name=e34cf3_vits_k710_h100x8x1
#SBATCH --partition=gpu100
#SBATCH --constraint=H100
#SBATCH --nodes=8
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=24
#SBATCH --mem=300G
#SBATCH --time=2-00:00:00
#SBATCH --exclude=gpu267,gpu269,gpu274
#SBATCH --output=/nfs/scistore19/locatgrp/bdemirel/ssl_project/outputs/%x_%j.out
# E34 THIRD LAUNCH of the S cell (Berker 2026-09-04 ~14:00: "i would cancel the s cell and b cell runs. i think this is more
# faithful to our imagenet chains as it wont be underutilizing the epochs"): the second launch's recipe and data path with NO
# gradient accumulation — batch 768 per optimizer step (96 x 8 ranks), 156 steps per epoch, 37,440 in 240 epochs (the
# second launch: batch 3,072, 39 per epoch); the conditioner rings OFF (768 fresh rows on the 128-d slices = n/d' 6, the
# ImageNet chains' certified ratio is 4 with 3/4 stale ring rows); warm-up kept at the donor's 12 percent of the run
# (peak_step 4,493); lr 4e-4 unchanged; the online K710 probe on 1,024 held-out clips from step 0. Run vid.floorssl.s0.k710s3.
# Every shape resumes the newest last.ckpt of the run; the 8 x 1 (eight BeeGFS clients) is the faster shape when the pool
# has eight free nodes. The second launch (k710s2, batch 3,072) ran to epoch ~100 and stays on the card as the
# their-batch contrast (checkpoints kept).
LEV=/nfs/scistore19/locatgrp/bdemirel/ssl_project/video/levjepa
export LEVJEPA_DATA_ROOT=/mnt/beegfs/locatgrp/shared/datasets/kinetics
export HYDRA_FULL_ERROR=1
export WANDB_ENTITY=causal-learning-ai-ista
RUN=vid.floorssl.s0.k710s3
RUNDIR=/mnt/beegfs/locatgrp/shared/bdemirel/levjepa_runs/$RUN
LAST=$(ls -t $RUNDIR/spt_cache/runs/*/*/*/checkpoints/last.ckpt 2>/dev/null | head -n 1)
if [ -n "$LAST" ]; then
  RESUME="resume.ckpt_path=$LAST resume.weights_only=false"
fi
# Working directory = the run directory: the trainer writes its wandb sidecar (wandb_resume.json) to the working
# directory at every checkpoint and re-injects the run id from there at a resume, so each cell keeps its own.
# (09-04 15:10: a manual wandb.config.id override crashed at start — the key is not in the config — and the shared
# video/levjepa working directory let the cells overwrite each other's sidecar.)
mkdir -p $RUNDIR && cd $RUNDIR || exit 1
echo "nodes: $SLURM_JOB_NODELIST | resume: ${LAST:-fresh start} | wandb sidecar: $(cat wandb_resume.json 2>/dev/null || echo none)"; nvidia-smi -L | head -n 1
srun $LEV/.venv/bin/python -u $LEV/main.py --config-name lambdajepa_vits data=k710_clipfiles \
  trainer.devices=1 trainer.num_nodes=8 trainer.max_epochs=240 accumulate_grad_batches=1 loss.lambdajepa.queue_steps=0 loss.lambdajepa.h_queue_steps=0 scheduler.peak_step=4493 +probe_k710=1024 \
  loader.num_workers=22 augmentation.photometrics_on_gpu=true \
  checkpoint.every_n_epochs=1 checkpoint.every_n_train_steps=0 \
  wandb.enabled=true wandb.config.project=sslgap wandb.config.name=$RUN \
  hydra.run.dir=$RUNDIR $RESUME
echo "E34_TRAIN_EXIT rc=$?"
