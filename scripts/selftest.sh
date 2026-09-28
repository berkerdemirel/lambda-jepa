#!/bin/bash
# The two gates every training job runs first (one GPU, a few minutes):  bash scripts/selftest.sh  ->  SELFTEST_OK
set -e
cd "$(dirname "$0")/.."
python experiments/lambdajepa_selftest.py && python experiments/train_ddp_selftest.py && echo "SELFTEST_OK"
