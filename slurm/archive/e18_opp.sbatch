#!/bin/bash
#SBATCH --job-name=e18-opp
#SBATCH --partition=gpu
#SBATCH --constraint=L40S|A40|A100|RTX3090Ti
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=01:30:00
#SBATCH --output=/nfs/scistore19/locatgrp/bdemirel/ssl_project/outputs/%x_%j.out

# E18 within-lane opposition read (E18-T1 owed mechanism; matched-state functional grid).
source /nfs/scistore19/locatgrp/bdemirel/ssl_project/slurm/_env.sh
cd /nfs/scistore19/locatgrp/bdemirel/ssl_project
python experiments/e18_opposition.py
