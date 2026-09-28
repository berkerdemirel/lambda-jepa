"""Augmentation thickness per station (W, B, Theta = W/B) from the 8-view orbit store; appends to results/diag/e23_retro_spaces.csv."""
import csv
import sys

import numpy as np

from sslgap.extract import FeatureStore
from sslgap.metrics.orbit_energy import orbit_energies
from sslgap.paths import DIAG, FEATURES

run = sys.argv[1]
st = FeatureStore(str(FEATURES))
man = "in100.pairs100.v1@audit_v1.o8"
V = int(st.meta(run, man)["v"])
labels = st.labels(run, man)
bases = sorted({s.rsplit(".view", 1)[0] for s in st.spaces(run, man)})
path = str(DIAG / "e23_retro_spaces.csv")
existing = {(r["run"], r["space"], r["framing"]) for r in csv.DictReader(open(path))}
cols = ["run", "V_used", "space", "framing", "V", "N", "D", "W", "B_hat", "B", "omega",
        "omega_hat", "r_rms", "B_within_cls", "B_between_cls", "BW_cls"]
out = []
for base in bases:
    views = [np.asarray(st.get(run, man, f"{base}.view{k}"), dtype=np.float64)
             for k in range(V)]
    for framing in ("raw", "l2"):
        if (run, base, framing) in existing:
            print(f"[skip] {base} {framing} (already present)")
            continue
        vs = ([v / (np.linalg.norm(v, axis=1, keepdims=True) + 1e-12) for v in views]
              if framing == "l2" else views)
        e = orbit_energies(vs, labels)
        out.append({"run": run, "V_used": V, "space": base, "framing": framing,
                    **{k: e[k] for k in ("V", "N", "D", "W", "B_hat", "B", "omega",
                                          "omega_hat", "r_rms", "B_within_cls",
                                          "B_between_cls", "BW_cls")}})
    print(f"[retro-append] {base} done")
with open(path, "a", newline="") as f:
    w = csv.DictWriter(f, fieldnames=cols)
    w.writerows(out)
print(f"appended {len(out)} rows for {run}")
