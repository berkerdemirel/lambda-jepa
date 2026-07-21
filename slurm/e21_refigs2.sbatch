#!/bin/bash
#SBATCH --job-name=e21-refigs2
#SBATCH --partition=defaultp
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=01:00:00
#SBATCH --output=/nfs/scistore19/locatgrp/bdemirel/ssl_project/outputs/%x_%j.out

# Figure feedback round (Berker 2026-07-21): posneg+classcos extended to vm2/e200 ->
# bar-panel metric fig + quad without the cancelled z-pred panel (D-060).
source /nfs/scistore19/locatgrp/bdemirel/ssl_project/slurm/_env.sh
cd /nfs/scistore19/locatgrp/bdemirel/ssl_project
set -e
python experiments/e2x_posneg.py
python experiments/e2x_classcos.py
python experiments/e21_metric_fig.py
python experiments/e21_quad_fig.py
echo "FIGS ROUND 2 DONE"
