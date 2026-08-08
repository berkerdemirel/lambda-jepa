"""Canonical repo paths, derived once.

Sixty-one scripts had hardcoded `ROOT = "/nfs/scistore19/locatgrp/bdemirel/ssl_project"`,
which makes the repo unrelocatable and the constant impossible to grep-verify. The root is
derived from this file's own location (so a clone anywhere works) and can be overridden with
`SSLGAP_ROOT` for out-of-tree runs. Scripts migrate to these as they are touched — a blanket
rewrite of the live path is not worth the risk while training chains are mid-flight.
"""
import os
from pathlib import Path

ROOT = Path(os.environ.get("SSLGAP_ROOT") or Path(__file__).resolve().parents[1])

OUTPUTS = ROOT / "outputs"        # checkpoints + SLURM logs (untracked)
FEATURES = ROOT / "features"      # the feature store (untracked)
RESULTS = ROOT / "results"        # small CSVs + findings + figures (tracked)

DIAG = RESULTS / "diag"           # per-experiment diagnostic CSVs
PROBES = RESULTS / "probes"       # probe results per run_id
BATTERY = RESULTS / "battery"     # metric battery per run_id
FIGURES = RESULTS / "figures"     # figures, one subdir per experiment


def figdir(name):
    """results/figures/<name>/, created on demand."""
    d = FIGURES / name
    d.mkdir(parents=True, exist_ok=True)
    return d
