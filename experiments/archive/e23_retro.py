"""E23 Deliverable 2 — the free retro-analysis: a/b/Λ/Ω over every existing o8/o32 orbit
store (family + e20f zoo + E17 arms), before anything trains. Per (store, base space):
raw + per-view-L2 orbit/center energies (W, B̂, B = B̂ − W/V exact debias, Ω, r_rms) and the
label-split center energies; per store: transmission a² = W_z/W_h, b² = B_z/B_h, Λ = b/a
from the declared h (+ student.h.cls) into every same-branch z tap — the depth-resolved
contraction profile through existing heads. o32 stores emit V∈{2,8,32} subset rows: after
the debias, B and Ω must be V-invariant (the estimator-verification table).
Definitions PROVISIONAL pending the E23 design discussion (Q2); numbers land raw.
Skips absent stores/spaces with a printed note (e2x_posneg convention)."""
import csv
import os
import re
import sys

import numpy as np

ROOT = "/nfs/scistore19/locatgrp/bdemirel/ssl_project"
sys.path.insert(0, ROOT)
from sslgap.extract.store import FeatureStore                    # noqa: E402
from sslgap.metrics.orbit_energy import orbit_energies, transmission  # noqa: E402

FEAT = os.path.join(ROOT, "features")
O8, O32 = "in100.pairs100.v1@audit_v1.o8", "in100.pairs100.v1@audit_v1.o32"
O8_IN1K = "in1k.pairs10.v1@audit_v1.o8"    # D-066 bump: 10/class × 1000 (label splits noisier, n_k=10)

# (declared_h, declared_z) of record: family = D-036 cls -> expander out; zoo = e20_score M
# / extract_orbits H/Z; E17 arms follow their method's zoo declaration.
LEJEPA = ("student.z.embed", "student.z.proj.out")
DECL = {}
for t in ["d256vm3zonly", "d256vm3", "d256vm3x2", "d256vm2", "d256vm4", "d256", "d128", "d64"]:
    DECL[f"in100.floorssl.s0.{t}.extL"] = ("student.h.cls", "student.z.proj.out")
for r in ["e17c.ext", "e20f.ext", "hpull_sigreg.ext", "hpull_sigreg_inv.ext",
          "hpull_sigreg3.ext", "hpull_sigreg_t.ext", "e12c1.ext", "e12f2.ext",
          "e12f7.ext", "e12f8.ext", "e20f.ep25.extL", "e20f.ep50.extL",
          "e20f.ep75.extL", "e20f.ep100.extL"]:
    DECL[f"in100.lejepa.s0.{r}"] = LEJEPA
for r in ["s0.ext", "s0.e17c.ext", "s0.e20f.ext", "s0.hpull_uniform.ext",
          "s0.hpull_uniform_align.ext"]:
    DECL[f"in100.simclr.{r}"] = ("student.h.cls", "student.z.proj.out")
for r in ["s0.ext", "s0.e17c.ext", "s0.e20f.ext", "s0.hpull_align.ext"]:
    DECL[f"in100.byol.{r}"] = ("student.h.cls", "student.z.pred.out")
for r in ["s0.e17c.ext", "s0.e20f.ext", "s0.hpull_varcov.c015.ext"]:
    DECL[f"in100.vicreg.{r}"] = ("student.h.cls", "student.z.proj.out")
for r in ["s0.e17c.ext", "s0.e20f.ext", "s0.hpull_protoce.ext"]:
    DECL[f"in100.dino.{r}"] = ("teacher.h.cls", "teacher.z.dino.bottleneck")
# the guillotine_zoo2 roster reads the .extL twins of the four e20f arms (L-taps)
DECL["in100.simclr.s0.e20f.extL"] = ("student.h.cls", "student.z.proj.out")
DECL["in100.byol.s0.e20f.extL"] = ("student.h.cls", "student.z.pred.out")
DECL["in100.vicreg.s0.e20f.extL"] = ("student.h.cls", "student.z.proj.out")
DECL["in100.dino.s0.e20f.extL"] = ("teacher.h.cls", "teacher.z.dino.bottleneck")
for r in ["s0.extL", "s0.e20f.extL"]:
    DECL[f"in100.mae.{r}"] = ("student.h.gap", "student.z.dec.tap8")
    DECL[f"in100.ijepa.{r}"] = ("teacher.h.gap", "student.z.pred.out")
O32_RUNS = ["in100.lejepa.s0.o32", "in100.lejepa.s0.e12f2.o32",
            "in100.lejepa.s0.hpull_sigreg_inv.o32"]
for r in O32_RUNS:
    DECL[r] = LEJEPA
for t in ["d256vm", "d256vm3", "d256vm4"]:
    DECL[f"in1k.floorssl.s0.{t}.extL"] = ("student.h.cls", "student.z.proj.out")


def load_views(store, run, key, base, V, l2=False):
    vs = []
    for k in range(V):
        v = np.asarray(store.get(run, key, f"{base}.view{k}"), dtype=np.float64)
        if l2:
            v /= np.linalg.norm(v, axis=1, keepdims=True) + 1e-12
        vs.append(v)
    return vs


def main():
    store = FeatureStore(FEAT)
    sp_rows, tr_rows = [], []
    for run, (dh, dz) in DECL.items():
        key = O32 if run in O32_RUNS else (O8_IN1K if run.startswith("in1k.") else O8)
        if not os.path.isdir(os.path.join(FEAT, run, key)):
            print(f"[skip] {run}/{key} absent", flush=True)
            continue
        meta = store.meta(run, key)
        V = meta["v"]
        # orbit stores name arrays per view ("<base>.view<k>", incl. in meta["spaces"]) —
        # bases are the deduped stems of what is actually on disk
        bases = sorted({re.sub(r"\.view\d+$", "", s) for s in store.spaces(run, key)})
        y = store.labels(run, key)
        v_grid = [2, 8, 32] if V == 32 else [V]
        energies = {}                                   # (base, V_used) -> raw dict
        for base in bases:
            for Vu in v_grid:
                raw = orbit_energies(load_views(store, run, key, base, Vu), labels=y)
                l2 = orbit_energies(load_views(store, run, key, base, Vu, l2=True))
                energies[(base, Vu)] = raw
                for framing, e in [("raw", raw), ("l2", l2)]:
                    sp_rows.append({"run": run, "V_used": Vu, "space": base,
                                    "framing": framing, **{k: e[k] for k in e}})
            print(f"[done] {run} {base}", flush=True)
        branch = dz.split(".")[0]
        froms = [f for f in dict.fromkeys([dh] + [b for b in bases if ".h." in b])
                 if f in bases]
        tos = [b for b in bases if b.startswith(f"{branch}.z.")]
        for Vu in v_grid:
            for f in froms:
                for t in tos:
                    if t == f:
                        continue
                    tr_rows.append({"run": run, "V_used": Vu, "from": f, "to": t,
                                    "declared": int(f == dh and t == dz),
                                    **transmission(energies[(f, Vu)], energies[(t, Vu)])})
    os.makedirs(os.path.join(ROOT, "results/diag"), exist_ok=True)
    for name, rows in [("e23_retro_spaces.csv", sp_rows), ("e23_retro_trans.csv", tr_rows)]:
        path = os.path.join(ROOT, "results/diag", name)
        with open(path, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
        print(f"[write] {path} ({len(rows)} rows)", flush=True)


if __name__ == "__main__":
    main()
