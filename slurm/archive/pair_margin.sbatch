#!/bin/bash
#SBATCH --job-name=gap-pairmargin
#SBATCH --partition=defaultp
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=00:45:00
#SBATCH --output=/nfs/scistore19/locatgrp/bdemirel/ssl_project/outputs/%x_%j.out

# M1 verify pass over stored features (CPU): pair margins + scale check, then re-emit the matrix
# with the scale-free variance-floor row. sbatch slurm/pair_margin.sbatch
source /nfs/scistore19/locatgrp/bdemirel/ssl_project/slurm/_env.sh
cd /nfs/scistore19/locatgrp/bdemirel/ssl_project
python experiments/pair_margin_m1.py
python experiments/report_m1.py
python experiments/pair_margin_fig.py
python experiments/sigreg_decomp.py
