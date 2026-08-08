"""α-dial cloud-count profiles (Berker 2026-08-06: tunable-inclusion touch graphs) +
the undertraining probe: connectivity fingerprints n_comp/giant/singletons vs α per
run × space over o8 stores, including two TRAINING-TIME trajectories — the landed
in100 e20f cadence (ep25/50/75/100/final) and the toy C′ K3 cadence (ep38/75/112 o8
extracted 2026-08-06 + the ep150 landing) — testing Berker's undertraining hypothesis
for toy one-island connectivity (toy 150 ep ≈ 11k optimizer steps vs in100 ≈ 100k vs
in1k ≈ 1M). Groups via argv. → results/diag/omega_graphprofile_<group>.csv, RAW."""
import csv
import os
import sys

import numpy as np

ROOT = "/nfs/scistore19/locatgrp/bdemirel/ssl_project"
sys.path.insert(0, ROOT)
from sslgap.extract.store import FeatureStore                    # noqa: E402
from sslgap.metrics.census import touch_graph_profile            # noqa: E402
from sslgap.metrics.orbit_energy import orbit_energies           # noqa: E402

TOY_KEY = "imagenette.train.v1@audit_v1.o8"
IN100_KEY = "in100.pairs100.v1@audit_v1.o8"
IN1K_KEY = "in1k.pairs10.v1@audit_v1.o8"
H = ["student.h.cls"]
GROUPS = {
    "toyvm": [(f"toy.floorssl.{t}.extL", TOY_KEY, H) for t in
              ("s0.e24v4oas", "s0.e24va", "s0.e24vb", "s0.e24vc", "s0.e24vh0",
               "s0.e24vac", "s0.e24v1", "s0.e24v2lr", "s0.e24v3nq")],
    "c2": [(f"toy.floorssl.s0.{t}.extL", TOY_KEY, H) for t in
           ("e23cW32", "e23cW64", "e23cW128", "e23cW256", "e23cW512", "e23cK0",
            "e23cK1", "e23cK3", "e23cK4", "e23cK6", "e23cW64h0", "e23cK6h0")],
    "in100": [(f"in100.floorssl.s0.{t}.extL", IN100_KEY, H) for t in
              ("e24va", "e24vb", "e24vc", "e24vh0", "e24vac", "e24vcc", "d256vm4")],
    "in1k": [(f"in1k.floorssl.s0.{t}.extL", IN1K_KEY, H)
             for t in ("d256vm", "d256vm3", "d256vm4")],
    "zoo": [(f"in1k.pub.{m}", IN1K_KEY,
             ["pub.h.pool" if m == "siglipb16" else "pub.h.cls"])
            for m in ("dinob16", "clipb16", "siglipb16", "maeb16", "dinov2b14",
                      "dinov3b16")],
    "cad100": [(f"in100.lejepa.s0.e20f.ep{e}.extL", IN100_KEY, H)
               for e in (25, 50, 75, 100)] + [("in100.lejepa.s0.e20f.ext",
                                               IN100_KEY, H)],
    "ktraj": [(f"toy.floorssl.s0.e23cK3.ep{e}.o8", TOY_KEY, H)
              for e in (38, 75, 112)] + [("toy.floorssl.s0.e23cK3.extL",
                                          TOY_KEY, H)],
}


def load_views(store, run, key, base, V):
    return [np.asarray(store.get(run, key, f"{base}.view{k}"), dtype=np.float64)
            for k in range(V)]


def main():
    store = FeatureStore(os.path.join(ROOT, "features"))
    for group in sys.argv[1].split(","):
        rows = []
        for run, key, spaces in GROUPS[group]:
            if not os.path.isdir(os.path.join(ROOT, "features", run)):
                print(f"[skip] {run}: no store", flush=True)
                continue
            V = store.meta(run, key)["v"]
            for base in spaces:
                vs = load_views(store, run, key, base, V)
                om = orbit_energies(vs)["omega"]
                prof, M = touch_graph_profile(vs)
                for p in prof:
                    rows.append({"group": group, "run": run, "space": base,
                                 "omega": om, "M_med": M, **p})
                a1 = [p for p in prof if p["alpha"] == 1.0][0]
                print(f"[prof] {run} {base}: omega {om:.3f} M {M:+.3f} | a=1 comp "
                      f"{a1['n_comp']} giant {a1['giant_share']:.3f} | a=.7 comp "
                      f"{[p for p in prof if p['alpha'] == 0.7][0]['n_comp']}",
                      flush=True)
        path = os.path.join(ROOT, f"results/diag/omega_graphprofile_{group}.csv")
        with open(path, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
        print(f"[write] {path} ({len(rows)} rows)", flush=True)


if __name__ == "__main__":
    main()
