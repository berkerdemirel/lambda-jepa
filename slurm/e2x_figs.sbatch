#!/bin/bash
#SBATCH --job-name=e2x-figs
#SBATCH --partition=defaultp
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=/nfs/scistore19/locatgrp/bdemirel/ssl_project/outputs/%x_%j.out
source /nfs/scistore19/locatgrp/bdemirel/ssl_project/slurm/_env.sh
cd /nfs/scistore19/locatgrp/bdemirel/ssl_project
python experiments/e2x_posneg.py
python experiments/e21_centered.py
python experiments/e20_battery_table.py
python experiments/e20f_metric_fig.py
python experiments/e21_metric_fig.py
python experiments/e21_quad_fig.py
python experiments/e20_dino_dose_fig.py
