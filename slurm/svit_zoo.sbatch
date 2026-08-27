#!/bin/bash
#SBATCH --job-name=svit-zoo
#SBATCH --partition=defaultp
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=02:00:00
#SBATCH --output=/nfs/scistore19/locatgrp/bdemirel/ssl_project/outputs/%x_%j.out
# ViT-S in1k metric zoo (guillotine layout, Berker 2026-08-24). Store-side compute
# (pairs/margin/Ω) caches to results/compare/svit_zoo_computed.csv; rerun after a new
# model's audit lands to add its row.
source /nfs/scistore19/locatgrp/bdemirel/ssl_project/slurm/_env.sh
cd /nfs/scistore19/locatgrp/bdemirel/ssl_project
.venv/bin/python experiments/svit_zoo.py
