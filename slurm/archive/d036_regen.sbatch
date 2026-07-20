#!/bin/bash
#SBATCH --job-name=d036-regen
#SBATCH --partition=defaultp
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=03:00:00
#SBATCH --output=/nfs/scistore19/locatgrp/bdemirel/ssl_project/outputs/%x_%j.out

# D-036 regeneration: re-run everything the h-redeclaration (F1 GAP->CLS) touches, at the new
# declared tap. NO re-extraction / re-battery / re-probe — all taps are already stored; these
# scripts only re-SELECT the declared-h column via the edited mirror dicts and re-plot/re-tabulate.
# Error-tolerant: one script failing does not abort the rest (each logs its own exit code).
source /nfs/scistore19/locatgrp/bdemirel/ssl_project/slurm/_env.sh
cd /nfs/scistore19/locatgrp/bdemirel/ssl_project

run() { echo "########## RUN: $* ##########"; python "$@"; echo "########## EXIT $?: $* ##########"; }

# --- M2 report family (tables + two-space matrix fig + pair-margin) ---
run experiments/report_m2.py
run experiments/e1_matrix_figs.py
run experiments/pair_margin_m2.py
# --- G-wave in-house figs (h-vs-z set, ladder, class-align at CLS) ---
run experiments/e12g_hz_figs.py
run experiments/e12g_ladder_fig.py
run experiments/e12_class_align.py --gwave
# --- E13 chain (rung-0 ranking CSVs -> 7 all-target scatters -> summary figs) ---
run experiments/e13_pivot_rung0.py
run experiments/e13_all_targets.py
run experiments/e13_figs.py

echo "########## D-036 REGEN COMPLETE ##########"
