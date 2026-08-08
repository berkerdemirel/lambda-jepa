"""E24 landing instruments (D-068) over the healthy toy cells (wave-1 13 + wave-2 w 7 +
view-mean 9: OAS va/vb/vc/vh0/vac/v4oas + variants v1/v2lr/v3nq) — the retro extension
the interpretation session reads: per cell (a) the orbit calculus (W/B/Ω/r_rms raw+l2,
label splits; a/b/Λ from every h base into every z tap — e23_retro conventions),
(b) Berker's touch census at the declared h and z (touch-same/diff, enrichment, ω levels),
(c) the {a_k} transmission spectrum from declared h into every z tap. Key
`imagenette.train.v1@audit_v1.o8`; z-tap count varies with K — bases discovered per store.
Numbers land RAW (results/diag/e24_toy_*.csv); interpretation is joint."""
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
TAGS = ["s0.e24s0a", "s0.e24s0b", "s0.e24s0c", "s0.e24s0d", "s0.e24s1a", "s0.e24s1b",
        "s0.e24s1c", "s0.e24s0ac", "s0.e24s0bc", "s0.e24s0cc", "s0.e24s1ac",
        "s0.e24s1bc", "s0.e24s1cc",
        "s0.e24wz05", "s0.e24wz12", "s0.e24wz18", "s0.e24wz25", "s0.e24wt05",
        "s0.e24wt2", "s1.e24wrep",
        "s0.e24v1", "s0.e24v2lr", "s0.e24v3nq",
        "s0.e24v4oas", "s0.e24va", "s0.e24vb", "s0.e24vc", "s0.e24vh0", "s0.e24vac"]
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
        run = f"toy.floorssl.{tag}.extL"
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
    for name, rows in [("e24_toy_spaces.csv", sp_rows), ("e24_toy_trans.csv", tr_rows),
                       ("e24_toy_census.csv", cen_rows),
                       ("e24_toy_spectrum.csv", spec_rows),
                       ("e24_toy_spectrum_summary.csv", spec_sum)]:
        path = os.path.join(ROOT, "results/diag", name)
        with open(path, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
        print(f"[write] {path} ({len(rows)} rows)", flush=True)


if __name__ == "__main__":
    main()
