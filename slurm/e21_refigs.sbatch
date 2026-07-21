#!/bin/bash
#SBATCH --job-name=e21-refigs
#SBATCH --partition=defaultp
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=02:00:00
#SBATCH --output=/nfs/scistore19/locatgrp/bdemirel/ssl_project/outputs/%x_%j.out

# E21 figure regeneration with the landed d256vm2 + d256e200 cells (Berker 2026-07-21):
# centered kNN -> metric panel -> quad probe fig -> curves. Runs afterok on both landings.
source /nfs/scistore19/locatgrp/bdemirel/ssl_project/slurm/_env.sh
cd /nfs/scistore19/locatgrp/bdemirel/ssl_project
set -e
# merge per-space probe partials into the canonical CSVs (part files removed after merge)
for RUN in in100.floorssl.s0.d256vm2.ext in100.floorssl.s0.d256e200.ext; do
  P=(results/probes/${RUN}.part_*.csv)
  head -1 "${P[0]}" > results/probes/${RUN}.csv
  tail -n +2 -q "${P[@]}" >> results/probes/${RUN}.csv
  rm results/probes/${RUN}.part_*.csv
  echo "merged $(wc -l < results/probes/${RUN}.csv) rows -> ${RUN}.csv"
done
python experiments/e21_centered.py
python experiments/e21_metric_fig.py
python experiments/e21_quad_fig.py
python experiments/e21_curves.py
echo "ALL FIGURES REGENERATED"
