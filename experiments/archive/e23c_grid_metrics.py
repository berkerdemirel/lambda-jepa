"""E23 stage-C′ landing instruments (D-068) over the 12 vm-OAS grid stores (e23c tags,
D-069-as-amended wave; re-landed 2026-08-06 after the cap incident) — same battery as the
stage-C pooled pass (e23_grid_metrics.py): per cell (a) the orbit calculus (W/B/Ω/r_rms
raw+l2, label splits; a/b/Λ from every h base into every z tap), (b) the touch census at
declared h and z (evaluation-side per the D-068 label-free addendum), (c) the {a_k}
transmission spectrum from declared h into every z tap. Key
`imagenette.train.v1@audit_v1.o8`; z-tap count varies with K — bases discovered per store.
Numbers land RAW (results/diag/e23c_grid_*.csv); interpretation is joint."""
import csv
import os
import re
import sys

import numpy as np

ROOT = "/nfs/scistore19/locatgrp/bdemirel/ssl_project"
sys.path.insert(0, ROOT)
from sslgap.extract.store import FeatureStore                    # noqa: E402
from sslgap.metrics.census import touch_census                   # noqa: E402
from sslgap.metrics.orbit_energy import (                        # noqa: E402
    orbit_energies, transmission, transmission_spectrum)

FEAT = os.path.join(ROOT, "features")
KEY = "imagenette.train.v1@audit_v1.o8"
TAGS = ["e23cW32", "e23cW64", "e23cW128", "e23cW256", "e23cW512",
        "e23cK0", "e23cK1", "e23cK3", "e23cK4", "e23cK6",
        "e23cW64h0", "e23cK6h0"]
DH, DZ = "student.h.cls", "student.z.proj.out"


def load_views(store, run, base, V, l2=False):
    vs = []
    for k in range(V):
        v = np.asarray(store.get(run, KEY, f"{base}.view{k}"), dtype=np.float64)
        if l2:
            v /= np.linalg.norm(v, axis=1, keepdims=True) + 1e-12
        vs.append(v)
    return vs


def main():
    store = FeatureStore(FEAT)
    sp_rows, tr_rows, cen_rows, spec_rows, spec_sum = [], [], [], [], []
    for tag in TAGS:
        run = f"toy.floorssl.s0.{tag}.extL"
        V = store.meta(run, KEY)["v"]
        y = store.labels(run, KEY)
        bases = sorted({re.sub(r"\.view\d+$", "", s) for s in store.spaces(run, KEY)})
        energies, views = {}, {}
        for base in bases:
            vs = load_views(store, run, base, V)
            views[base] = vs
            energies[base] = orbit_energies(vs, labels=y)
            l2 = orbit_energies(load_views(store, run, base, V, l2=True))
            for framing, e in [("raw", energies[base]), ("l2", l2)]:
                sp_rows.append({"run": run, "V_used": V, "space": base,
                                "framing": framing, **{k: e[k] for k in e}})
        froms = [f for f in dict.fromkeys([DH] + [b for b in bases if ".h." in b])
                 if f in bases]
        tos = [b for b in bases if b.startswith("student.z.")]
        for f in froms:
            for t in tos:
                tr_rows.append({"run": run, "V_used": V, "from": f, "to": t,
                                "declared": int(f == DH and t == DZ),
                                **transmission(energies[f], energies[t])})
        for base in (DH, DZ):
            rows, summ = touch_census(views[base], y)
            cen_rows += [{"run": run, "space": base, **r} for r in rows]
            cen_rows.append({"run": run, "space": base, "class": -1, **summ})
            print(f"[census] {tag} {base}: p_pos {summ['p_pos']:.3f} p_neg "
                  f"{summ['p_neg']:.4f} enrich {summ['enrich']:.1f} "
                  f"omega_same {summ['omega_same_med']:+.3f}", flush=True)
        for t in tos:
            rows, summ = transmission_spectrum(views[DH], views[t])
            spec_rows += [{"run": run, "from": DH, "to": t, **r} for r in rows]
            spec_sum.append({"run": run, "from": DH, "to": t,
                             "declared": int(t == DZ), **summ})
        s = [r for r in spec_sum if r["run"] == run and r["declared"]][0]
        print(f"[done] {tag}: a {np.sqrt(s['a2_total']):.3f} r2_lin {s['r2_lin']:.3f} "
              f"cv_ak {s['cv_ak']:.2f}", flush=True)
    for name, rows in [("e23c_grid_spaces.csv", sp_rows), ("e23c_grid_trans.csv", tr_rows),
                       ("e23c_grid_census.csv", cen_rows),
                       ("e23c_grid_spectrum.csv", spec_rows),
                       ("e23c_grid_spectrum_summary.csv", spec_sum)]:
        path = os.path.join(ROOT, "results/diag", name)
        with open(path, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
        print(f"[write] {path} ({len(rows)} rows)", flush=True)


if __name__ == "__main__":
    main()
