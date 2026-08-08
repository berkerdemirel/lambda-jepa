"""THE grid landing instrument (D-068): the cloud calculus + touch census + transmission
spectrum over a set of landed `.extL` stores.

One script replacing e23_grid_metrics / e23c_grid_metrics / e24_grid_metrics, which were
byte-identical apart from a tag list and an output prefix (archived 2026-08-07). Per cell:

  (a) cloud energies — W/B/Ω/r_rms in raw and per-view-ℓ2 framings, plus the label-split
      center energies (evaluation-side per the D-068 label-free addendum);
  (b) transmission a/b/Λ from every h base into every z tap;
  (c) the touch census at declared h and z;
  (d) the {a_k} transmission spectrum from declared h into every z tap.

The z-tap count varies with head depth, so bases are DISCOVERED per store rather than
assumed. Numbers land RAW; interpretation is joint.

Usage:
    python experiments/grid_metrics.py e23c_grid                  # a named preset
    python experiments/grid_metrics.py my_grid --tags s0.aa,s0.bb # an ad-hoc set
    python experiments/grid_metrics.py e24_toy --lane toy.floorssl --key <store key>

Writes results/diag/<prefix>_{spaces,trans,census,spectrum,spectrum_summary}.csv.
"""
import csv
import re
import sys

import numpy as np

from sslgap.extract.store import FeatureStore
from sslgap.metrics.census import touch_census
from sslgap.metrics.orbit_energy import orbit_energies, transmission, transmission_spectrum
from sslgap.paths import DIAG, FEATURES

KEY = "imagenette.train.v1@audit_v1.o8"
LANE = "toy.floorssl"
DH, DZ = "student.h.cls", "student.z.proj.out"

# The landed grids, so re-running any of them is one word. Tags are the run-id segment
# between the lane and `.extL` (they carry their own seed).
PRESETS = {
    "e23_grid": ["s0.e23jW32", "s0.e23jW64", "s0.e23jW128", "s0.e23jW256", "s0.e23jW512",
                 "s0.e23Llr", "s0.e23jK0", "s0.e23jK1", "s0.e23jK3", "s0.e23jK4",
                 "s0.e23jK6", "s0.e23jW64h0", "s0.e23jK2h0", "s0.e23jK6h0"],
    "e23c_grid": ["s0.e23cW32", "s0.e23cW64", "s0.e23cW128", "s0.e23cW256", "s0.e23cW512",
                  "s0.e23cK0", "s0.e23cK1", "s0.e23cK3", "s0.e23cK4", "s0.e23cK6",
                  "s0.e23cW64h0", "s0.e23cK6h0"],
    "e24_toy": ["s0.e24s0a", "s0.e24s0b", "s0.e24s0c", "s0.e24s0d", "s0.e24s1a", "s0.e24s1b",
                "s0.e24s1c", "s0.e24s0ac", "s0.e24s0bc", "s0.e24s0cc", "s0.e24s1ac",
                "s0.e24s1bc", "s0.e24s1cc",
                "s0.e24wz05", "s0.e24wz12", "s0.e24wz18", "s0.e24wz25", "s0.e24wt05",
                "s0.e24wt2", "s1.e24wrep",
                "s0.e24v1", "s0.e24v2lr", "s0.e24v3nq",
                "s0.e24v4oas", "s0.e24va", "s0.e24vb", "s0.e24vc", "s0.e24vh0", "s0.e24vac"],
}


def load_views(store, run, base, V, key, l2=False):
    vs = []
    for k in range(V):
        v = np.asarray(store.get(run, key, f"{base}.view{k}"), dtype=np.float64)
        if l2:
            v /= np.linalg.norm(v, axis=1, keepdims=True) + 1e-12
        vs.append(v)
    return vs


def main():
    args = sys.argv[1:]
    assert args, "usage: grid_metrics.py <prefix|preset> [--tags a,b] [--lane L] [--key K]"
    prefix, lane, key, tags = args[0], LANE, KEY, PRESETS.get(args[0])
    for flag, setter in (("--tags", "tags"), ("--lane", "lane"), ("--key", "key")):
        if flag in args:
            val = args[args.index(flag) + 1]
            if setter == "tags":
                tags = val.split(",")
            elif setter == "lane":
                lane = val
            else:
                key = val
    assert tags, f"no preset named {prefix!r} and no --tags given"

    store = FeatureStore(str(FEATURES))
    sp_rows, tr_rows, cen_rows, spec_rows, spec_sum = [], [], [], [], []
    for tag in tags:
        run = f"{lane}.{tag}.extL"
        V = store.meta(run, key)["v"]
        y = store.labels(run, key)
        bases = sorted({re.sub(r"\.view\d+$", "", s) for s in store.spaces(run, key)})
        energies, views = {}, {}
        for base in bases:
            vs = load_views(store, run, base, V, key)
            views[base] = vs
            energies[base] = orbit_energies(vs, labels=y)
            l2 = orbit_energies(load_views(store, run, base, V, key, l2=True))
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
    for suffix, rows in [("spaces", sp_rows), ("trans", tr_rows), ("census", cen_rows),
                         ("spectrum", spec_rows), ("spectrum_summary", spec_sum)]:
        path = DIAG / f"{prefix}_{suffix}.csv"
        with open(path, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
        print(f"[write] {path} ({len(rows)} rows)", flush=True)


if __name__ == "__main__":
    main()
