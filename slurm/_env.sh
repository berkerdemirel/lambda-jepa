#!/bin/bash
# Sourced by the sbatch scripts: activate the uv venv and reuse caches (port of ssl_explore/_env.sh).
export PYTHONUNBUFFERED=1        # stream prints to the .out live
export HF_HOME=/nfs/scistore19/locatgrp/bdemirel/.cache/huggingface
export HF_HUB_OFFLINE=1          # imagenette is fully cached; don't hit the network on compute nodes
export HF_DATASETS_OFFLINE=1
export WANDB__SERVICE_WAIT=300
export TOKENIZERS_PARALLELISM=false
source /nfs/scistore19/locatgrp/bdemirel/ssl_project/.venv/bin/activate
echo "node: $(hostname)"
nvidia-smi -L || true
python -c "import torch; print('torch', torch.__version__, '| cuda_available', torch.cuda.is_available())"
