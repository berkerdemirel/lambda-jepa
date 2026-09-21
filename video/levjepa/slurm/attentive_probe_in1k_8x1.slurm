#!/bin/bash
#SBATCH --job-name=attn_probe_in1k
#SBATCH --partition=gpu100
#SBATCH --constraint=H100
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=2
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=24
#SBATCH --mem=200G
#SBATCH --time=12:00:00
#SBATCH --exclude=gpu274,gpu277,gpu271
#SBATCH --output=/nfs/scistore19/locatgrp/bdemirel/ssl_project/outputs/%x_%j.out
# Frozen attentive probe on ImageNet-1k (V-JEPA protocol, scripts/attentive_probe.py) on a UNIFORM 8-rank DDP shape: 4 nodes x 2
# H100 (--nodes / --ntasks-per-node / --gres together; any uniform N x k with N*k = 8 may be given on the sbatch line). Only a
# per-node --gres ask keeps the task count equal to the card count on every node and leaves all of a node's cards visible to its
# tasks, so LOCAL_RANK = SLURM_LOCALID indexes a real device and NCCL 2.28 keeps its intra-node transports -- the flexible
# --gpus / --gpus-per-task / --ntasks-per-gpu asks all break one of those two, see video_probe_nx1.slurm for the smoke evidence.
# env:// rendezvous on the first node, --kill-on-bad-exit=1. Global batch 8 x 16 = 128 at lr 1e-3 (V-JEPA: 64 x 16 = 1,024; lr
# kept -- the card's protocol note), 20 epochs, full IN-1k train, 50k val. IN-1k is read from NFS (no staging).
# Usage: sbatch [--time=..] slurm/attentive_probe_in1k_8x1.slurm <run dir> <out csv> [ema|student] [extra args]
#        (run dir: newest last.ckpt; extra args pass through to the probe, as video_probe_nx1.slurm does --
#         e.g. --lr for a batch-scaled read: V-JEPA sets lr 1e-3 at global batch 1,024, we run 128.)
cd /nfs/scistore19/locatgrp/bdemirel/ssl_project/video/levjepa || exit 1
export PYTHONPATH=/nfs/scistore19/locatgrp/bdemirel/ssl_project/video/levjepa
export NCCL_DEBUG=WARN
RUNDIR=$1; OUT=$2; W=${3:-ema}; shift 3 2>/dev/null; EXTRA="$@"
# a checkpoint PATH may be given instead of a run dir: the newest-last.ckpt glob would otherwise follow a
# continuation of the same run and silently read a later encoder than the one a diagnostic is comparing against.
CKPT=$([ -f "$RUNDIR" ] && echo "$RUNDIR" || ls -t $RUNDIR/spt_cache/runs/*/*/*/checkpoints/last.ckpt | head -n 1)
export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1); export MASTER_PORT=$((29000 + SLURM_JOB_ID % 1000))   # unique per job (as video_probe_nx1.slurm)
echo "nodes: $SLURM_JOB_NODELIST ($SLURM_NNODES nodes, $SLURM_NTASKS ranks) | ckpt: $CKPT | weights: $W | master: $MASTER_ADDR:$MASTER_PORT | extra: $EXTRA"
srun --kill-on-bad-exit=1 bash -c 'export RANK=$SLURM_PROCID WORLD_SIZE=$SLURM_NTASKS LOCAL_RANK=$SLURM_LOCALID; exec .venv/bin/python -u scripts/attentive_probe.py --ckpt "$0" --out "$1" --weights "$2" --bs 16 --workers 8 '"$EXTRA"'' "$CKPT" "$OUT" "$W"
echo "ATTN_PROBE_DONE rc=$?"
