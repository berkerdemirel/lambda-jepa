#!/bin/bash
#SBATCH --job-name=e27lej
#SBATCH --partition=gpu100
#SBATCH --constraint=H100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=28
#SBATCH --mem=128G
#SBATCH --time=08:00:00
#SBATCH --requeue
#SBATCH --output=/nfs/scistore19/locatgrp/bdemirel/ssl_project/outputs/%x_%j.out

# E27 lejepa control chain unit (D-082): the matched-frame LeJEPA reference at ViT-S —
# aug/arch/epochs/bs identical to the floorssl S cells. Dials per the D-081 gap-fill
# order: λ=0.05 (paper recommendation, Berker verbatim); lr 5e-4 / final lr = lr/1000
# (eta_min 5e-7) / wd 5e-2 / bf16 (repo README large-scale values); warmup 1 ep /
# proj [2048,2048,16] / emb 512 / no grad clip (repo minimal, port lineage). λ exists
# ONLY in this lane. Selftest heads every segment (~1 min); resume via _last.pt.
source /nfs/scistore19/locatgrp/bdemirel/ssl_project/slurm/_env.sh
cd /nfs/scistore19/locatgrp/bdemirel/ssl_project
python experiments/e27_selftest_lejepa.py || exit 1
python experiments/train.py method=lejepa frame=in1k_vits16 bs=128 num_classes=1000 \
  method.lamb=0.05 +method.aug=lejepa_mc method.lr=5e-4 method.eta_min=5e-7 \
  '+extra_cadence=[10]' tag=e27lej "$@"
