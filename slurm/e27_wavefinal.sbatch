#!/bin/bash
#SBATCH --output=/nfs/scistore19/locatgrp/bdemirel/ssl_project/outputs/%x_%j.out
#SBATCH --requeue
# E27 final S wave (D-088): 5x100ep on the adopted V=10 frame; selftest gates each segment.
source /nfs/scistore19/locatgrp/bdemirel/ssl_project/slurm/_env.sh
cd /nfs/scistore19/locatgrp/bdemirel/ssl_project
python experiments/e27_selftest.py || exit 1
python experiments/train.py "$@"
