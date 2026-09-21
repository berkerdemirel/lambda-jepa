#!/bin/bash
#SBATCH --job-name=video_probe
#SBATCH --partition=gpu100
#SBATCH --constraint=H100
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=2
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=24
#SBATCH --mem=500G
#SBATCH --time=1-00:00:00
#SBATCH --exclude=gpu274,gpu277,gpu271
#SBATCH --output=/nfs/scistore19/locatgrp/bdemirel/ssl_project/outputs/%x_%j.out
# Frozen video probe (V-JEPA protocol port, scripts/video_probe.py) on a UNIFORM 8-rank DDP shape: 4 nodes x 2 H100
# (--nodes / --ntasks-per-node / --gres together; any uniform N x k with N*k = 8 may be given on the sbatch line).
# WHY UNIFORM (H100 smokes 2026-09-08 15:08-15:22, smoke_videoprobe/ddp_smoke.py): only a per-node --gres ask keeps the task
# count equal to the card count on EVERY node and leaves all of a node's cards visible to all of its tasks, so
# LOCAL_RANK = SLURM_LOCALID indexes a real device and NCCL 2.28 keeps its intra-node transports. Both flexible asks fail:
#   --gpus=8 --ntasks=8 --nodes=1-8 places tasks by CPU, not by card, so a node can get 4 tasks and 2 cards -> 'invalid device
#     ordinal' at set_device (job 65009767 on gpu269; it happened to align in the 2-CPU smoke 65009354, which is why it looked good);
#   --gpus-per-task=1 and --ntasks-per-gpu=1 mask one card per task and NCCL's SHM transport then dies with 'invalid device ordinal'
#     between ranks of one node (65006461 / 65006462), while --gpu-bind=none makes this SLURM hand every task of an H100 node the
#     SAME card ('Duplicate GPU detected', 65008523).
# The one-card-per-node shape this file is named after only served the BeeGFS per-node read cap; every rank stages its fixed shard of
# the store to the node's local disk once and the epochs read local disk, so the layout is free -- but eight DISTINCT nodes could not
# be placed while the free cards sat on five (2026-09-08 14:58, Berker). Staging is per rank into one per-job directory per node
# (disjoint files; K400 ~81 GB and SSv2 ~18 GB per rank, so 2 ranks per node use ~162 GB of the ~675 GB free local disk); the
# directory is removed per node after the step. env:// rendezvous on the first node; --kill-on-bad-exit=1 ends the step when a rank dies.
# Usage: sbatch [--mem=..] [--time=..] slurm/video_probe_nx1.slurm <dataset ssv2|k400> <head attentive|linear_mean> <run dir>
#        <out csv> [ema|student, default ema] [bs per GPU] [extra args]   (run dir: newest last.ckpt; data root by dataset)
cd /nfs/scistore19/locatgrp/bdemirel/ssl_project/video/levjepa || exit 1
export PYTHONPATH=/nfs/scistore19/locatgrp/bdemirel/ssl_project/video/levjepa
export NCCL_DEBUG=WARN
DS=$1; HEAD=$2; RUNDIR=$3; OUT=$4; W=${5:-ema}; BS=${6:-4}; shift 6 2>/dev/null; EXTRA="$@"
case $DS in ssv2) DATA=/mnt/beegfs/locatgrp/shared/datasets/ssv2/ssv2_clipfiles;; k400) DATA=/mnt/beegfs/locatgrp/shared/datasets/kinetics/k400_eval_clipfiles;; *) echo "bad dataset $DS"; exit 1;; esac
# a checkpoint PATH may be given instead of a run dir: the newest-last.ckpt glob would otherwise follow a
# continuation of the same run and silently read a later encoder than the one a diagnostic is comparing against.
CKPT=$([ -f "$RUNDIR" ] && echo "$RUNDIR" || ls -t $RUNDIR/spt_cache/runs/*/*/*/checkpoints/last.ckpt | head -n 1)
export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1); export MASTER_PORT=$((29000 + SLURM_JOB_ID % 1000))   # unique per job: two probes starting together on one master node collided on a fixed port (2026-09-07 00:06)
# node-local hygiene (one task per node): staging dirs of jobs no longer in the queue are removed before staging (wall-time kills skip the in-job cleanup)
srun -N $SLURM_NNODES -n $SLURM_NNODES --ntasks-per-node=1 bash -c 'for d in ${SLURM_TMPDIR:-/tmp}/lev_* ${SLURM_TMPDIR:-/tmp}/vp_*; do [ -d "$d" ] || continue; id=${d##*_}; squeue -h -j "$id" -o %i 2>/dev/null | grep -q . || rm -rf "$d"; done; echo "$(hostname): $(df -h ${SLURM_TMPDIR:-/tmp} | tail -1)"'
echo "nodes: $SLURM_JOB_NODELIST ($SLURM_NNODES nodes, $SLURM_NTASKS ranks) | $DS $HEAD | ckpt: $CKPT | weights: $W | bs $BS | extra: $EXTRA"
STAGE=${SLURM_TMPDIR:-/tmp}/vp_$SLURM_JOB_ID
srun --kill-on-bad-exit=1 bash -c 'export RANK=$SLURM_PROCID WORLD_SIZE=$SLURM_NTASKS LOCAL_RANK=$SLURM_LOCALID; mkdir -p "$7"; exec .venv/bin/python -u scripts/video_probe.py --dataset "$0" --head "$1" --ckpt "$2" --out "$3" --weights "$4" --bs "$5" --data "$6" --stage-local "$7" --workers '"${NUM_WORKERS:-12}"' '"$EXTRA"'' "$DS" "$HEAD" "$CKPT" "$OUT" "$W" "$BS" "$DATA" "$STAGE"
rc=$?
srun -N $SLURM_NNODES -n $SLURM_NNODES --ntasks-per-node=1 bash -c "rm -rf $STAGE"
echo "VIDEO_PROBE_DONE rc=$rc"
