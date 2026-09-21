#!/bin/bash
#SBATCH --job-name=zkl_forensics
#SBATCH --partition=gpu
#SBATCH --constraint=A100|L40S|A40
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=120G
#SBATCH --time=02:00:00
#SBATCH --output=/nfs/scistore19/locatgrp/bdemirel/ssl_project/outputs/%x_%j.out
# One-shot: decompose the z conditioner per step and test whether one 128-row block drives the
# 4-step spike runs (E27, Berker 2026-09-10). Read-only on the checkpoint.
cd /nfs/scistore19/locatgrp/bdemirel/ssl_project || exit 1
.venv/bin/python experiments/e27_zkl_forensics.py "$1" "$2" "${3:-200}"
