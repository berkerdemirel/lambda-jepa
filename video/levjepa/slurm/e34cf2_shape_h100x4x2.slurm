#!/bin/bash
#SBATCH --job-name=e34cf2_vits_k710_h100x4x2
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
# E34 second launch (Berker 2026-09-03 "okay queue 8x1 h100s" / "you can do that cancel"): the same cell as job
# 64312322 — the lambda-JEPA recipe on LeVJEPA's ViT-S/16, K710-20% clip files, 240 epochs, effective batch
# 8 ranks x 96 x accum 4 = 3,072 — with the span read (data/clipfile_loader.py) and the photometric ops on the GPU
# (data/gpu_views.py), on TWO nodes x four H100 (the shape that starts on the first cell's freed nodes; two BeeGFS clients) (the first cell was bound by the
# per-node read rate and by CPU photometrics: 53 s per step). Every shape of this run has the same 8 ranks, so
# whichever shape starts first begins the run and the others CONTINUE it from the newest last.ckpt (chain roll):
# the run dir and the wandb run are shared; rings re-warm over q steps after a resume (declared).
cd /nfs/scistore19/locatgrp/bdemirel/ssl_project/video/levjepa || exit 1
export LEVJEPA_DATA_ROOT=/mnt/beegfs/locatgrp/shared/datasets/kinetics
export HYDRA_FULL_ERROR=1
export WANDB_ENTITY=causal-learning-ai-ista
RUN=vid.floorssl.s0.k710s2
RUNDIR=/mnt/beegfs/locatgrp/shared/bdemirel/levjepa_runs/$RUN
LAST=$(ls -t $RUNDIR/spt_cache/runs/*/*/*/checkpoints/last.ckpt 2>/dev/null | head -n 1)
if [ -n "$LAST" ]; then
  SIDE=$(dirname "$(dirname "$LAST")")/wandb_resume.json; [ -f "$SIDE" ] || SIDE=wandb_resume.json
  WID=$(python3 -c "import json,sys; print(json.load(open(sys.argv[1])).get('id') or '')" "$SIDE" 2>/dev/null)
  RESUME="resume.ckpt_path=$LAST resume.weights_only=false ${WID:+wandb.config.id=$WID wandb.config.resume=allow}"
fi
echo "nodes: $SLURM_JOB_NODELIST | resume: ${LAST:-fresh start} | wandb id: ${WID:-new}"; nvidia-smi -L | head -n 1
srun .venv/bin/python -u main.py --config-name lambdajepa_vits data=k710_clipfiles \
  trainer.devices=4 trainer.num_nodes=2 accumulate_grad_batches=4 trainer.max_epochs=240 \
  loader.num_workers=22 augmentation.photometrics_on_gpu=true \
  checkpoint.every_n_epochs=1 checkpoint.every_n_train_steps=0 \
  wandb.enabled=true wandb.config.project=sslgap wandb.config.name=$RUN \
  hydra.run.dir=$RUNDIR $RESUME
echo "E34_TRAIN_EXIT rc=$?"
