#!/bin/bash
#SBATCH --job-name=e20f-guillotine
#SBATCH --partition=defaultp
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=00:30:00
#SBATCH --output=/nfs/scistore19/locatgrp/bdemirel/ssl_project/outputs/%x_%j.out

# Merge per-space probe partials for the four .extL stores, then render.
source /nfs/scistore19/locatgrp/bdemirel/ssl_project/slurm/_env.sh
cd /nfs/scistore19/locatgrp/bdemirel/ssl_project
set -e
for N in 25 50 75 100; do
  RUN=in100.lejepa.s0.e20f.ep${N}.extL
  P=(results/probes/${RUN}.part_*.csv)
  head -1 "${P[0]}" > results/probes/${RUN}.csv
  tail -n +2 -q "${P[@]}" >> results/probes/${RUN}.csv
  rm results/probes/${RUN}.part_*.csv
done
python experiments/e20f_guillotine.py
echo E20F-GUILLOTINE-DONE
