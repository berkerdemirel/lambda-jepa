#!/bin/bash
# G-wave (D-028/D-030) post-training scoring pipeline, submitted 2026-07-13 with dependencies on
# the LAST training chain link per run: training done -> extract (gpu) -> audit (defaultp) +
# probe (gpu); after ALL FOUR extracts -> class-align + floor-values in --gwave mode (new _g.csv
# outputs; per-run space map lives in the two scripts). Everything mechanical; no takeaways.
set -e
cd /nfs/scistore19/locatgrp/bdemirel/ssl_project

declare -A LAST=([e12gv]=62237581 [e12gd]=62237585 [e12gvc]=62241907 [e12gdc]=62241911)
declare -A CKPT=([e12gv]=in100.vicreg.s0.e12gv [e12gd]=in100.dino.s0.e12gd
                 [e12gvc]=in100.vicreg.s0.e12gvc [e12gdc]=in100.dino.s0.e12gdc)

extract_ids=()
for arm in e12gv e12gd e12gvc e12gdc; do
  run=${CKPT[$arm]}
  ex=$(sbatch --parsable --dependency=afterany:${LAST[$arm]} slurm/extract.sbatch \
       ckpt=outputs/${run}_ep100.pt adapter=native run_id=${run}.ext)
  extract_ids+=("$ex")
  sbatch --dependency=afterok:${ex} slurm/audit.sbatch run_id=${run}.ext > /dev/null
  sbatch --dependency=afterok:${ex} slurm/probe.sbatch run_id=${run}.ext > /dev/null
  echo "$arm: extract=$ex (+audit,probe chained)"
done

dep=$(IFS=:; echo "${extract_ids[*]}")
sbatch --dependency=afterok:${dep} slurm/e12_align.sbatch --gwave
sbatch --dependency=afterok:${dep} slurm/e12_floorval.sbatch --gwave
