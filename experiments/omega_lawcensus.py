"""Touch-law census, NATIVE label-free (theory doc §9: R1 Ω_local, R2/R5/R6 law
transport via per-frame c fits, R3 touch-graph, R7 zoo; also the §10 'reconstructed
label-free aggregates pending native recomputation' debt). Per run × space over o8
orbit stores: the orbit calculus (W/B/Ω, debiased) + touch_law_stats — all-pairs
median signed overlap M, touching fraction T, cloud-local Ω_local(k=20), touch-graph
components/giant share — census conventions (r_mean radii, sample centers), label-free
throughout. Groups (argv, comma-separated): toyvm (9 view-mean toy cells), c2 (the 12
stage-C′ cells, run after their re-landing), in100 (vm-OAS map + ring anchor), in1k
(vm/vm3/vm4 standard-frame), zoo (D-076 public track — own CSV, never mixed).
→ results/diag/omega_lawcensus_<group>.csv, RAW; fits happen at read time."""
import csv
import os
import sys

import numpy as np

ROOT = "/nfs/scistore19/locatgrp/bdemirel/ssl_project"
sys.path.insert(0, ROOT)
from sslgap.extract.store import FeatureStore                    # noqa: E402
from sslgap.metrics.census import touch_law_stats                # noqa: E402
from sslgap.metrics.orbit_energy import orbit_energies           # noqa: E402

TOY_KEY = "imagenette.train.v1@audit_v1.o8"
HZ = ["student.h.cls", "student.z.proj.out"]
GROUPS = {
    "toyvm": [(f"toy.floorssl.{t}.extL", TOY_KEY, HZ) for t in
              ("s0.e24v4oas", "s0.e24va", "s0.e24vb", "s0.e24vc", "s0.e24vh0",
               "s0.e24vac", "s0.e24v1", "s0.e24v2lr", "s0.e24v3nq")],
    "c2": [(f"toy.floorssl.s0.{t}.extL", TOY_KEY, HZ) for t in
           ("e23cW32", "e23cW64", "e23cW128", "e23cW256", "e23cW512", "e23cK0",
            "e23cK1", "e23cK3", "e23cK4", "e23cK6", "e23cW64h0", "e23cK6h0")],
    "toysw": [(f"toy.floorssl.{t}.extL", TOY_KEY, HZ) for t in
              ("s0.e24s0a", "s0.e24s0b", "s0.e24s0c", "s0.e24s0d", "s0.e24s1a",
               "s0.e24s1b", "s0.e24s1c", "s0.e24s0ac", "s0.e24s0bc", "s0.e24s0cc",
               "s0.e24s1ac", "s0.e24s1bc", "s0.e24s1cc", "s0.e24wz05", "s0.e24wz12",
               "s0.e24wz18", "s0.e24wz25", "s0.e24wt05", "s0.e24wt2", "s1.e24wrep")],
    "toypooled": [(f"toy.floorssl.s0.{t}.extL", TOY_KEY, HZ) for t in
                  ("e23jW32", "e23jW64", "e23jW128", "e23jW256", "e23jW512", "e23Llr",
                   "e23jK0", "e23jK1", "e23jK3", "e23jK4", "e23jK6", "e23jW64h0",
                   "e23jK2h0", "e23jK6h0")],
    "in100": [(f"in100.floorssl.s0.{t}.extL", "in100.pairs100.v1@audit_v1.o8", HZ)
              for t in ("e24va", "e24vb", "e24vc", "e24vh0", "e24vac", "e24vcc",
                        "d256vm4")],
    "in1k": [(f"in1k.floorssl.s0.{t}.extL", "in1k.pairs10.v1@audit_v1.o8", HZ)
             for t in ("d256vm", "d256vm3", "d256vm4")],
    "zoo": [(f"in1k.pub.{m}", "in1k.pairs10.v1@audit_v1.o8",
             ["pub.h.pool" if m == "siglipb16" else "pub.h.cls"])
            for m in ("dinob16", "clipb16", "siglipb16", "maeb16", "dinov2b14",
                      "dinov3b16")],
}


def load_views(store, run, key, base, V, l2=False):
    vs = []
    for k in range(V):
        v = np.asarray(store.get(run, key, f"{base}.view{k}"), dtype=np.float64)
        if l2:
            v /= np.linalg.norm(v, axis=1, keepdims=True) + 1e-12
        vs.append(v)
    return vs


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
                e = orbit_energies(vs)
                el2 = orbit_energies(load_views(store, run, key, base, V, l2=True))
                law = touch_law_stats(vs)
                rows.append({"group": group, "run": run, "space": base,
                             "W": e["W"], "B": e["B"], "omega": e["omega"],
                             "omega_l2": el2["omega"], "r_rms": e["r_rms"], **law})
                print(f"[law] {run} {base}: omega {e['omega']:.4f} M {law['M_med']:+.4f} "
                      f"T {law['T_frac']:.4f} loc {law['omega_local_med']:.4f} "
                      f"comp {law['n_comp']} giant {law['giant_share']:.3f}", flush=True)
        path = os.path.join(ROOT, f"results/diag/omega_lawcensus_{group}.csv")
        with open(path, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
        print(f"[write] {path} ({len(rows)} rows)", flush=True)


if __name__ == "__main__":
    main()
