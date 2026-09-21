#!/bin/bash
#SBATCH --job-name=e12-prep
#SBATCH --partition=gpu
#SBATCH --constraint=L40S|A40|A100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=00:40:00
#SBATCH --output=/nfs/scistore19/locatgrp/bdemirel/ssl_project/outputs/%x_%j.out

# E12 (D-026) prep: equal-pull λ for arms A2/A3 (premise p3; D-021 lineage). Values land in
# results/diag/e12_equal_pull.csv + the E12 card; full arms launch with the measured λ verbatim.
source /nfs/scistore19/locatgrp/bdemirel/ssl_project/slurm/_env.sh
cd /nfs/scistore19/locatgrp/bdemirel/ssl_project
python experiments/equal_pull.py frame=in100_vits16 bs=128 num_classes=100 tag=e12a2 \
    method.lr=3e-4 method.warmup_ep=10 method.eta_min=1e-5 method.grad_clip=1.0 \
    +method.sigreg_at=embed +method.spec_norm=true +method.embed_calib=true
python experiments/equal_pull.py frame=in100_vits16 bs=128 num_classes=100 tag=e12a3 \
    method.lr=3e-4 method.warmup_ep=10 method.eta_min=1e-5 method.grad_clip=1.0 \
    +method.sigreg_at=embed +method.spec_norm=true +method.embed_calib=true +method.floor=sacreg
