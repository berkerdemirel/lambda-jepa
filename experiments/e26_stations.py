"""E26 zoo stations CSV (D-076: own track, trunk stations only): per model x station
{L03, L06, L09, cls}, orbit calculus from the pub o8 stores — W, B, Omega (raw + l2
rider) via the canonical estimators — plus the probe headline columns when the probe
CSVs exist. Battery columns join at the figure build. Label-free throughout (labels
untouched); siglip's final station rides pub.h.pool (no prefix token; card-noted)."""
import csv
import os
import re

import numpy as np

from sslgap.extract import FeatureStore
from sslgap.metrics.orbit_energy import orbit_energies

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "results/diag/e26_stations.csv")
O8 = "in1k.pairs10.v1@audit_v1.o8"
MODELS = ["dinob16", "clipb16", "siglipb16", "maeb16", "dinov2b14", "dinov3b16"]
STATIONS = ["L03", "L06", "L09", "cls"]


def station_space(model, st):
    if st == "cls":
        return "pub.h.pool" if model == "siglipb16" else "pub.h.cls"
    return f"pub.h.cls.L{st[1:]}" if model != "siglipb16" else f"pub.h.gap.L{st[1:]}"


def load_views(store, run, base, V, l2=False):
    vs = []
    for k in range(V):
        v = np.asarray(store.get(run, O8, f"{base}.view{k}"), dtype=np.float64)
        if l2:
            v /= np.linalg.norm(v, axis=1, keepdims=True) + 1e-12
        vs.append(v)
    return vs


def main():
    store = FeatureStore(os.path.join(ROOT, "features"))
    rows = []
    for m in MODELS:
        run = f"in1k.pub.{m}"
        V = store.meta(run, O8)["v"]
        spaces = set(re.sub(r"\.view\d+$", "", s) for s in store.spaces(run, O8))
        for st in STATIONS:
            base = station_space(m, st)
            if base not in spaces:
                print(f"[skip] {run} {st}: no {base}", flush=True)
                continue
            e = orbit_energies(load_views(store, run, base, V))
            el2 = orbit_energies(load_views(store, run, base, V, l2=True))
            rows.append({"model": m, "station": st, "space": base, "V": V,
                         "N": e["N"], "D": e["D"], "W": e["W"], "B": e["B"],
                         "omega": e["omega"], "omega_l2": el2["omega"],
                         "r_rms": e["r_rms"]})
            print(f"[o8] {m} {st}: omega {e['omega']:.4f} (l2 {el2['omega']:.4f})",
                  flush=True)
    with open(OUT, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"[done] {OUT} ({len(rows)} rows)", flush=True)


if __name__ == "__main__":
    main()
