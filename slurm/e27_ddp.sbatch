#!/bin/bash
#SBATCH --job-name=e27ddp
#SBATCH --partition=gpu100
#SBATCH --constraint=H100
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=48
#SBATCH --mem=400G
#SBATCH --time=24:00:00
#SBATCH --output=/nfs/scistore19/locatgrp/bdemirel/ssl_project/outputs/%x_%j.out

# Single-node multi-GPU DDP for the LambdaJEPA add-on (experiments/train_ddp.py). torchrun spawns
# one worker per allocated GPU (ntasks-per-node=1 — torchrun does the spawning, not srun). Resume
# is on by default, so segment the L run with --dependency=singleton chains like the other lanes.
#   sbatch slurm/e27_ddp.sbatch method=lambdajepa frame=in1k_vitl16 num_classes=1000 <lm4s5b> tag=...
source /nfs/scistore19/locatgrp/bdemirel/ssl_project/slurm/_env.sh
cd /nfs/scistore19/locatgrp/bdemirel/ssl_project
NGPU=$(nvidia-smi -L | wc -l)
echo "node: $(hostname)  GPUs: ${NGPU}"
.venv/bin/python -m torch.distributed.run --standalone --nproc_per_node="${NGPU}" \
    experiments/train_ddp.py "$@"
