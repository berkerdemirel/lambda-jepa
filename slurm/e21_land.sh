#!/bin/bash
# E21 landing pipeline (one command per landed cell; the E20 convention: standard .ext
# extraction -> v2 probes + battery, then e20_centered rerun + e21 scoring by hand):
#   bash slurm/e21_land.sh in100.floorssl.s0.lejepa_augs [ep100|best|last]
# Bars for the hz runs (E21 card): lejepa e20f .6700 lin / .6288 knn_c; f2 .6578/.6056.
set -e
RUN=$1
CK=${2:-ep100}
cd /nfs/scistore19/locatgrp/bdemirel/ssl_project
EX=$(sbatch --parsable slurm/extract.sbatch ckpt=outputs/${RUN}_${CK}.pt adapter=native run_id=${RUN}.ext)
sbatch --dependency=afterok:$EX slurm/probe.sbatch run_id=${RUN}.ext
sbatch --dependency=afterok:$EX slurm/audit.sbatch run_id=${RUN}.ext
echo "extract $EX -> probes + battery queued for ${RUN}.ext (ckpt ${CK})"
