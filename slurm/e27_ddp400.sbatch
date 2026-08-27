#!/bin/bash
#SBATCH --job-name=e27ddp400
#SBATCH --partition=gpu100
#SBATCH --constraint=H100
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=48
#SBATCH --mem=400G
#SBATCH --time=120:00:00
#SBATCH --requeue
#SBATCH --output=/nfs/scistore19/locatgrp/bdemirel/ssl_project/outputs/%x_%j.out

# The 400-epoch round (Berker 2026-08-25): e27_ddp body + the wavefinal selftest gate,
# 120h segments (partition allows 10d; 5d keeps one bad-node re-roll per run while cutting
# chain boundaries vs the 24h/8h eras). B mains override --gres=gpu:2 --cpus-per-task=28
# --mem=200G on the CLI; L mains use the defaults. Chain with --dependency=singleton;
# spare segments are no-ops (resume finds the final epoch).
source /nfs/scistore19/locatgrp/bdemirel/ssl_project/slurm/_env.sh
cd /nfs/scistore19/locatgrp/bdemirel/ssl_project
python experiments/e27_selftest.py || exit 1
python experiments/train_ddp_selftest.py || exit 1
NGPU=$(nvidia-smi -L | wc -l)
echo "node: $(hostname)  GPUs: ${NGPU}"
.venv/bin/python -m torch.distributed.run --standalone --nproc_per_node="${NGPU}" \
    experiments/train_ddp.py "$@"
