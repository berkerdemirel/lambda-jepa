"""M1 verify pass (dress-rehearsal follow-ups, Berker 2026-07-08): (1) pair_margin over every
pair store — positive- vs random-pair contrast per space, both aug stacks — the baseline without
which the matrix's alignment/invariance rows cannot be scored (h spaces are cone-compact, so raw
pair distances conflate invariance with global compactness); (2) scale check — per-dim std of
each z.final on the clean manifest vs the augmented pair store, testing whether VICReg's hinge
miss at z (0.31 @ gamma=1) is the floor being calibrated to augmented batches, not dead dims.

  python experiments/pair_margin_m1.py
"""
import glob
import os

import numpy as np
import pandas as pd

from sslgap.metrics.pairs import pair_margin
from sslgap.metrics.single import variance_floor

RES = "/nfs/scistore19/locatgrp/bdemirel/ssl_project/results"
FEAT = "/nfs/scistore19/locatgrp/bdemirel/ssl_project/features"
# mirrors experiments/report_m1.py (PROTOCOL §3 / D-003v2)
H_SPACE = {"simclr": "student.h.cls", "vicreg": "student.h.cls", "byol": "student.h.cls",  # D-036: projector input (was GAP)
           "dino": "teacher.h.cls", "mae": "student.h.gap", "ijepa": "teacher.h.gap",
           "lejepa": "student.z.embed"}
Z_FINAL = {"simclr": "student.z.proj.out", "vicreg": "student.z.proj.out",
           "byol": "student.z.pred.out", "dino": "teacher.z.dino.bottleneck",
           "mae": None, "ijepa": "student.z.pred.out", "lejepa": "student.z.proj.out"}
RUN_ID = {m: f"toy.{m}.s0.ext" for m in H_SPACE} | {"dino": "toy.dino.s0.probefix.ext"}
NULL_ID = {m: f"toy.{m}.s0.null.ext" for m in H_SPACE} | {"dino": "toy.dino.s0.probefix.null.ext"}


def _load(path):
    return np.load(path).astype(np.float64)


def margins_for(run_id):
    rows = []
    for pdir in sorted(glob.glob(os.path.join(FEAT, run_id, "imagenette.train.v1@*"))):
        stack = os.path.basename(pdir).split("@", 1)[1]
        for fa in sorted(glob.glob(os.path.join(pdir, "*.viewA.npy"))):
            space = os.path.basename(fa).removesuffix(".viewA.npy")
            A, B = _load(fa), _load(fa.replace(".viewA.npy", ".viewB.npy"))
            rows.append({"run_id": run_id, "stack": stack, "space": space,
                         "n": len(A), "d": A.shape[1]} | pair_margin(A, B))
    return rows


def scale_check(method):
    rows = []
    for space in filter(None, {H_SPACE[method], Z_FINAL[method]}):
        clean = os.path.join(FEAT, RUN_ID[method], "imagenette.train.v1", f"{space}.npy")
        if not os.path.exists(clean):
            continue
        Xc = _load(clean)
        row = {"method": method, "space": space, "mean_std_clean": float(Xc.std(0).mean()),
               "hinge_clean": variance_floor(Xc)["hinge"]}
        aug = glob.glob(os.path.join(FEAT, RUN_ID[method], f"imagenette.train.v1@own_{method}",
                                     f"{space}.view*.npy"))
        if aug:
            Xa = np.concatenate([_load(f) for f in sorted(aug)], 0)
            # per-branch (viewA only) = VICReg's exact training statistic shape: per-dim std
            # ACROSS IMAGES within one view-batch (Berker 2026-07-08 wording fix)
            Xb = _load(sorted(aug)[0])
            row |= {"mean_std_aug": float(Xa.std(0).mean()),
                    "hinge_aug": variance_floor(Xa)["hinge"],
                    "mean_std_branchA": float(Xb.std(0).mean()),
                    "hinge_branchA": variance_floor(Xb)["hinge"]}
            if len(aug) == 2:
                # WITHIN-image across-view spread (the quantity the invariance term drives to 0):
                # for 2 views, unbiased per-dim within-orbit var = (A-B)^2/2, averaged over images
                A, B = _load(sorted(aug)[0]), _load(sorted(aug)[1])
                row["mean_std_within"] = float(np.sqrt(((A - B) ** 2 / 2).mean(0)).mean())
        rows.append(row)
    return rows


def main():
    out_dir = os.path.join(RES, "M1")
    mrows, srows = [], []
    for m in H_SPACE:
        for rid in (RUN_ID[m], NULL_ID[m]):
            if os.path.isdir(os.path.join(FEAT, rid)):
                mrows += margins_for(rid)
        srows += scale_check(m)
    dfm, dfs = pd.DataFrame(mrows), pd.DataFrame(srows)
    dfm.to_csv(os.path.join(out_dir, "pair_margin.csv"), index=False)
    dfs.to_csv(os.path.join(out_dir, "scale_check.csv"), index=False)

    md = ["# M1 verify pass — pair margins + scale check (numbers only)\n",
          "> Emitted by experiments/pair_margin_m1.py. pair_margin = positive-pair vs random-pair",
          "> contrast in the SAME space (random pairs: cross-view, different images, identical",
          "> pipeline). cos_margin = pos_cos − rand_cos (higher = more view-invariance beyond",
          "> compactness); align_rel = align_pos / align_rand (lower = same, relative version).",
          "> Full grid incl. taps + null runs: results/M1/pair_margin.csv\n",
          "\n## Headline h vs z (audit_v1 stack, trained runs)\n"]
    rows = []
    for m in H_SPACE:
        # bracket access: "stack" is a pandas method name, dfm.stack shadows the column
        d = dfm[(dfm["run_id"] == RUN_ID[m]) & (dfm["stack"] == "audit_v1")]
        h, z = H_SPACE[m], Z_FINAL[m]
        rh = d[d["space"] == h].iloc[0] if len(d[d["space"] == h]) else None
        rz = d[d["space"] == z].iloc[0] if z and len(d[d["space"] == z]) else None
        rows.append({"method": m,
                     "h: pos/rand cos": f"{rh.pos_cos:.3f} / {rh.rand_cos:.3f}" if rh is not None else "—",
                     "h margin": round(rh.cos_margin, 3) if rh is not None else np.nan,
                     "z: pos/rand cos": f"{rz.pos_cos:.3f} / {rz.rand_cos:.3f}" if rz is not None else "—",
                     "z margin": round(rz.cos_margin, 3) if rz is not None else np.nan,
                     "h align_rel": round(rh.align_rel, 3) if rh is not None else np.nan,
                     "z align_rel": round(rz.align_rel, 3) if rz is not None else np.nan})
    md.append(pd.DataFrame(rows).to_markdown(index=False))
    md += ["\n\n## Scale check — per-dim std, clean manifest vs augmented pair store\n",
           "> variance_floor.hinge is computed at gamma=1 on whatever scale the space has;",
           "> VICReg trains its floor on augmented batches. hinge_clean vs hinge_aug per space:\n",
           dfs.to_markdown(index=False, floatfmt=".4f"), "\n"]
    with open(os.path.join(out_dir, "PAIR_MARGIN.md"), "w") as f:
        f.write("\n".join(md))
    print(f"[pair_margin_m1] wrote pair_margin.csv ({len(dfm)} rows), scale_check.csv "
          f"({len(dfs)} rows), PAIR_MARGIN.md")


if __name__ == "__main__":
    main()
