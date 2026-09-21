#!/bin/bash
#SBATCH --job-name=e34cf2_vitb_k710_h100x4x2
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
# E34 SECOND CELL, ViT-B (Berker 2026-09-04: "overlap is fine just to be ready to the best case"): the lambda-JEPA recipe on
# LeVJEPA's ViT-B/16 with the e27v6b400 chain's doses (22.81/222.26/4.054; conf/lambdajepa_vitb.yaml: rings z-q3 / h-q7,
# slices 128 / 256), K710-20% clip files, 240 epochs, effective batch 8 ranks x 64 x accum 6 = 3,072 (ViT-B at bs 64 x
# 6 views peaks at 48 GB with the share logger at bs 64 — smoke 64450968), GPU photometrics + span reads, the online K710
# probe on 1,024 held-out clips (+probe_k710), checkpoints under the BeeGFS run dir. Same resume clause as the S shapes:
# a later shape continues the run from its newest last.ckpt. Reference at landing: LeVJEPA ViT-B 50.7 / V-JEPA 2 51.6 /
# VideoMAEv2 47.1 (their Fig. 2, IN-1k attentive probe, matched epochs). Predictions P-B1..B4 on the E34 card.
cd /nfs/scistore19/locatgrp/bdemirel/ssl_project/video/levjepa || exit 1
export LEVJEPA_DATA_ROOT=/mnt/beegfs/locatgrp/shared/datasets/kinetics
export HYDRA_FULL_ERROR=1
export WANDB_ENTITY=causal-learning-ai-ista
RUN=vid.floorssl.s0.k710b
RUNDIR=/mnt/beegfs/locatgrp/shared/bdemirel/levjepa_runs/$RUN
LAST=$(ls -t $RUNDIR/spt_cache/runs/*/*/*/checkpoints/last.ckpt 2>/dev/null | head -n 1)
if [ -n "$LAST" ]; then
  SIDE=$(dirname "$(dirname "$LAST")")/wandb_resume.json; [ -f "$SIDE" ] || SIDE=wandb_resume.json
  WID=$(python3 -c "import json,sys; print(json.load(open(sys.argv[1])).get('id') or '')" "$SIDE" 2>/dev/null)
  RESUME="resume.ckpt_path=$LAST resume.weights_only=false ${WID:+wandb.config.id=$WID wandb.config.resume=allow}"
fi
echo "nodes: $SLURM_JOB_NODELIST | resume: ${LAST:-fresh start} | wandb id: ${WID:-new}"; nvidia-smi -L | head -n 1
srun .venv/bin/python -u main.py --config-name lambdajepa_vitb data=k710_clipfiles loader.batch_size=64 loss.lambdajepa.share_log_bs=64 +probe_k710=1024 \
  trainer.devices=4 trainer.num_nodes=2 accumulate_grad_batches=6 trainer.max_epochs=240 \
  loader.num_workers=22 augmentation.photometrics_on_gpu=true \
  checkpoint.every_n_epochs=1 checkpoint.every_n_train_steps=0 \
  wandb.enabled=true wandb.config.project=sslgap wandb.config.name=$RUN \
  hydra.run.dir=$RUNDIR $RESUME
echo "E34_TRAIN_EXIT rc=$?"
