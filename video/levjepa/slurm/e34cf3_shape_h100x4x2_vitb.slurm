#!/bin/bash
#SBATCH --job-name=e34cf3_vitb_k710_h100x4x2
#SBATCH --partition=gpu100
#SBATCH --constraint=H100
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=4
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=24
#SBATCH --mem=700G
#SBATCH --time=2-00:00:00
#SBATCH --exclude=gpu267,gpu269,gpu274
#SBATCH --output=/nfs/scistore19/locatgrp/bdemirel/ssl_project/outputs/%x_%j.out
# E34 B CELL, third-launch settings (Berker 2026-09-04 ~14:00): batch 512 per optimizer step (64 x 8 ranks), 234 steps per
# epoch, 56,160 in 240 epochs; z ring off (512 rows / 128 = n/d' 4 fresh), h ring q1 (1,024 rows / 256 = 4); warm-up at the
# donor's 12 percent (peak_step 6,739); lr 4e-4; doses = the e27v6b400 chain (22.81/222.26/4.054); the online K710 probe
# from step 0; share logger at bs 64 (48 GB peak in the smoke 64450968). Run vid.floorssl.s0.k710b. Pends on the 31-GPU
# QOS cap until v6Llr100 lands. Predictions P-B1..B4 on the E34 card.
LEV=/nfs/scistore19/locatgrp/bdemirel/ssl_project/video/levjepa
export LEVJEPA_DATA_ROOT=/mnt/beegfs/locatgrp/shared/datasets/kinetics
export HYDRA_FULL_ERROR=1
export WANDB_ENTITY=causal-learning-ai-ista
RUN=vid.floorssl.s0.k710b
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
srun $LEV/.venv/bin/python -u $LEV/main.py --config-name lambdajepa_vitb data=k710_clipfiles loader.batch_size=64 loss.lambdajepa.share_log_bs=64 +probe_k710=1024 \
  trainer.devices=4 trainer.num_nodes=2 trainer.max_epochs=240 accumulate_grad_batches=1 loss.lambdajepa.queue_steps=0 loss.lambdajepa.h_queue_steps=1 scheduler.peak_step=6739 \
  loader.num_workers=22 augmentation.photometrics_on_gpu=true \
  checkpoint.every_n_epochs=1 checkpoint.every_n_train_steps=0 \
  wandb.enabled=true wandb.config.project=sslgap wandb.config.name=$RUN \
  hydra.run.dir=$RUNDIR $RESUME
echo "E34_TRAIN_EXIT rc=$?"
