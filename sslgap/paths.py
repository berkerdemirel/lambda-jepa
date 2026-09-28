"""Package paths derived from this file's location (override with SSLGAP_ROOT)."""
import os
from pathlib import Path

ROOT = Path(os.environ.get("SSLGAP_ROOT") or Path(__file__).resolve().parents[1])

OUTPUTS = ROOT / "outputs"
FEATURES = ROOT / "features"
RESULTS = ROOT / "results"

DIAG = RESULTS / "diag"
PROBES = RESULTS / "probes"
BATTERY = RESULTS / "battery"
FIGURES = RESULTS / "figures"

def figdir(name):
    """results/figures/<name>/, created on demand."""
    d = FIGURES / name
    d.mkdir(parents=True, exist_ok=True)
    return d
