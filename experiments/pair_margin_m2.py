"""M2 pair-margin pass (D-013; E01-T8 made margin-scored pair glyphs FINAL): positive- vs
random-pair contrast per space over the IN-100 pair stores. Differences from the M1 script:
run_ids are in100.<m>.s0.ext; only the fixed audit stack exists at this rung (@own stacks were a
toy/dino-ctrl affordance); the null is the SHARED in100.randinit-s0.ext (h-side spaces only — no
per-method z-space nulls extracted at IN-100 yet). Scale check reads the audit-stack store and
is labeled as such (M1's used the method's own stack).

  python experiments/pair_margin_m2.py
"""
import glob
import os

import numpy as np
import pandas as pd

from sslgap.metrics.pairs import pair_margin
from sslgap.metrics.single import variance_floor

RES = "/nfs/scistore19/locatgrp/bdemirel/ssl_project/results"
FEAT = "/nfs/scistore19/locatgrp/bdemirel/ssl_project/features"
H_SPACE = {"simclr": "student.h.gap", "vicreg": "student.h.gap", "byol": "student.h.gap",
           "dino": "teacher.h.cls", "mae": "student.h.gap", "ijepa": "teacher.h.gap",
           "lejepa": "student.z.embed"}
Z_FINAL = {"simclr": "student.z.proj.out", "vicreg": "student.z.proj.out",
           "byol": "student.z.pred.out", "dino": "teacher.z.dino.bottleneck",
           "mae": None, "ijepa": "student.z.pred.out", "lejepa": "student.z.proj.out"}
RUN_ID = {m: f"in100.{m}.s0.ext" for m in H_SPACE}
NULL_RUN = "in100.randinit-s0.ext"


def _load(path):
    return np.load(path).astype(np.float64)


def margins_for(run_id):
    rows = []
    for pdir in sorted(glob.glob(os.path.join(FEAT, run_id, "in100.pairs100.v1@*"))):
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
        clean = os.path.join(FEAT, RUN_ID[method], "in100.train500.v1", f"{space}.npy")
        if not os.path.exists(clean):
            continue
        Xc = _load(clean)
        row = {"method": method, "space": space, "mean_std_clean": float(Xc.std(0).mean()),
               "hinge_clean": variance_floor(Xc)["hinge"]}
        aug = sorted(glob.glob(os.path.join(FEAT, RUN_ID[method], "in100.pairs100.v1@audit_v1",
                                            f"{space}.view*.npy")))
        if aug:
            Xa = np.concatenate([_load(f) for f in aug], 0)
            Xb = _load(aug[0])
            row |= {"mean_std_aug": float(Xa.std(0).mean()),
                    "hinge_aug": variance_floor(Xa)["hinge"],
                    "mean_std_branchA": float(Xb.std(0).mean()),
                    "hinge_branchA": variance_floor(Xb)["hinge"]}
            if len(aug) == 2:
                A, B = _load(aug[0]), _load(aug[1])
                row["mean_std_within"] = float(np.sqrt(((A - B) ** 2 / 2).mean(0)).mean())
        rows.append(row)
    return rows


def main():
    out_dir = os.path.join(RES, "M2")
    os.makedirs(out_dir, exist_ok=True)
    mrows, srows = [], []
    for m in H_SPACE:
        if os.path.isdir(os.path.join(FEAT, RUN_ID[m])):
            mrows += margins_for(RUN_ID[m])
            srows += scale_check(m)
    mrows += margins_for(NULL_RUN)
    dfm, dfs = pd.DataFrame(mrows), pd.DataFrame(srows)
    dfm.to_csv(os.path.join(out_dir, "pair_margin.csv"), index=False)
    dfs.to_csv(os.path.join(out_dir, "scale_check.csv"), index=False)

    md = ["# M2 pair margins + scale check — IN-100 seed 0 (numbers only)\n",
          "> Emitted by experiments/pair_margin_m2.py. Definitions as in results/M1/PAIR_MARGIN.md",
          "> (cos_margin = pos_cos − rand_cos; align_rel = align_pos/align_rand, lower = more",
          "> view-invariant relative to the space's own compactness). audit_v1 stack only at this",
          "> rung. Null = shared in100.randinit-s0.ext (h-side spaces only).\n",
          "\n## Headline h vs z (audit_v1 stack, trained runs)\n"]
    rows = []
    for m in H_SPACE:
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
    md += ["\n\n## Scale check — per-dim std, clean manifest vs audit-stack pair store\n",
           dfs.to_markdown(index=False, floatfmt=".4f"), "\n"]
    with open(os.path.join(out_dir, "PAIR_MARGIN.md"), "w") as f:
        f.write("\n".join(md))
    print(f"[pair_margin_m2] wrote pair_margin.csv ({len(dfm)} rows), scale_check.csv "
          f"({len(dfs)} rows), PAIR_MARGIN.md")


if __name__ == "__main__":
    main()
