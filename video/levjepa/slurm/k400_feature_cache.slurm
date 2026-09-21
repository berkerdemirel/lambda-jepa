#!/bin/bash
#SBATCH --job-name=k400_feats
#SBATCH --partition=gpu100
#SBATCH --constraint=H100
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=2
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=24
#SBATCH --mem=300G
#SBATCH --time=06:00:00
#SBATCH --exclude=gpu274,gpu277,gpu271
#SBATCH --output=/nfs/scistore19/locatgrp/bdemirel/ssl_project/outputs/%x_%j.out
# K400 feature cache (scripts/k400_feature_cache.py) on the UNIFORM 8-rank shape of video_probe_nx1.slurm (4 nodes x 2 H100; the
# reasons for the uniform ask are documented there). No DDP: ranks are independent and only need RANK / WORLD_SIZE / LOCAL_RANK.
# Every rank stages its fixed shard of the store to the node's local disk once (K400 train ~81 GB per rank, val ~7 GB); the
# per-job staging directory is removed per node afterwards; stale staging dirs of finished jobs are removed first.
# Usage: sbatch slurm/k400_feature_cache.slurm <out_dir> name=ckpt [name=ckpt ...]   (EMA weights; bs 8 per GPU; 20 workers)
cd /nfs/scistore19/locatgrp/bdemirel/ssl_project/video/levjepa || exit 1
export PYTHONPATH=/nfs/scistore19/locatgrp/bdemirel/ssl_project/video/levjepa
OUT=$1; shift; CKPTS="$@"
DATA=/mnt/beegfs/locatgrp/shared/datasets/kinetics/k400_eval_clipfiles
srun -N $SLURM_NNODES -n $SLURM_NNODES --ntasks-per-node=1 bash -c 'for d in ${SLURM_TMPDIR:-/tmp}/lev_* ${SLURM_TMPDIR:-/tmp}/vp_* ${SLURM_TMPDIR:-/tmp}/kf_*; do [ -d "$d" ] || continue; id=${d##*_}; squeue -h -j "$id" -o %i 2>/dev/null | grep -q . || rm -rf "$d"; done; echo "$(hostname): $(df -h ${SLURM_TMPDIR:-/tmp} | tail -1)"'
echo "nodes: $SLURM_JOB_NODELIST ($SLURM_NNODES nodes, $SLURM_NTASKS ranks) | out: $OUT | ckpts: $CKPTS"
STAGE=${SLURM_TMPDIR:-/tmp}/kf_$SLURM_JOB_ID
srun --kill-on-bad-exit=1 bash -c 'export RANK=$SLURM_PROCID WORLD_SIZE=$SLURM_NTASKS LOCAL_RANK=$SLURM_LOCALID; mkdir -p "$2"; exec .venv/bin/python -u scripts/k400_feature_cache.py --data "$0" --out-dir "$1" --stage-local "$2" --bs '"${BS:-8}"' --workers '"${NUM_WORKERS:-20}"' --ckpts '"$CKPTS"'' "$DATA" "$OUT" "$STAGE"
rc=$?
srun -N $SLURM_NNODES -n $SLURM_NNODES --ntasks-per-node=1 bash -c "rm -rf $STAGE"
echo "K400_CACHE_DONE rc=$rc"
