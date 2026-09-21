#!/bin/bash
#SBATCH --job-name=k400_read
#SBATCH --partition=gpu
#SBATCH --constraint=A40|L40S|A100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=04:00:00
#SBATCH --output=/nfs/scistore19/locatgrp/bdemirel/ssl_project/outputs/%x_%j.out
# The optimizer-free readers + geometry over the cached K400 features (experiments/e34_k400_features.py); chained afterok the
# cache job. Usage: sbatch [--dependency=afterok:<cache job>] slurm/e34_k400_features.sbatch <cache dir> [out csv]
cd /nfs/scistore19/locatgrp/bdemirel/ssl_project || exit 1
nvidia-smi -L
PYTHONPATH=$PWD .venv/bin/python -u experiments/e34_k400_features.py --cache "$1" ${2:+--out "$2"}
echo "K400_READ_DONE rc=$?"
