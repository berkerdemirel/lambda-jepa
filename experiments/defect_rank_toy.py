"""k-hat over the toy feature store (D-019 battery-v2 first empirical pass, 2026-07-10).
Three estimators per cell (khat_eig axis-aligned count / khat_pp kurtosis-pursuit prefix /
khat_A Beta-spectrum fit) in the declared frame (top-64 standardized PCA coords), pursuit band
calibrated once per (n, m) and shared. Cells chosen where independent structure knowledge
exists (E01-T4/T9, E10-T3): numbers land raw.

  python experiments/defect_rank_toy.py   -> results/diag/defect_rank_toy.csv
"""
import os
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
from sslgap.metrics.defect_rank import khat_eig, khat_pursuit, khat_spectrum, pursuit_null_band

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FEAT = os.path.join(ROOT, "features")
MAN = "imagenette.train.v1"
CELLS = [  # (label, run_id, space, what we independently know)
    ("lejepa proj.out (K=16)", "toy.lejepa.s0.ext", "student.z.proj.out", "87-94% shape work (E01-T9)"),
    ("lejepa embed", "toy.lejepa.s0.ext", "student.z.embed", "rank 9.4/512, kurt 3.62 (E01-T4)"),
    ("lejepa h.cls", "toy.lejepa.s0.ext", "student.h.cls", "kurt 3.45"),
    ("A embed (free)", "toy.lejepa.s0.e10A.ext", "student.z.embed", "strong class clusters, NMI .57"),
    ("Dlr embed (constrained)", "toy.lejepa.s0.e10Dlr.ext", "student.z.embed", "partial clusters, NMI .25, ~15-17d"),
    ("D0 embed (post-storm)", "toy.lejepa.s0.e10D0.ext", "student.z.embed", "structure destroyed, NMI .01"),
    ("randinit embed", "toy.randinit-s0.ext", "student.z.embed", "10 modes, class-unaligned"),
    ("simclr z.proj", "toy.simclr.s0.ext", "student.z.proj.out", "kurt_worst 26.6 (sharpening)"),
    ("vicreg z.proj", "toy.vicreg.s0.ext", "student.z.proj.out", "kurt_worst 24.0"),
    ("dino z.bottleneck", "toy.dino.s0.probefix.ext", "teacher.z.dino.bottleneck", "kurt_worst 20.4"),
    ("byol z.pred", "toy.byol.s0.ext", "student.z.pred.out", "kurt_worst 0.76 (tame)"),
    # h-vs-z contrast rows (Berker 2026-07-10; branch conventions follow the E1 matrix)
    ("lejepa h.gap", "toy.lejepa.s0.ext", "student.h.gap", "—"),
    ("simclr h.cls", "toy.simclr.s0.ext", "student.h.cls", "—"),
    ("simclr h.gap", "toy.simclr.s0.ext", "student.h.gap", "—"),
    ("vicreg h.cls", "toy.vicreg.s0.ext", "student.h.cls", "—"),
    ("vicreg h.gap", "toy.vicreg.s0.ext", "student.h.gap", "—"),
    ("byol h.cls", "toy.byol.s0.ext", "student.h.cls", "—"),
    ("byol h.gap", "toy.byol.s0.ext", "student.h.gap", "—"),
    ("dino h.cls (teacher)", "toy.dino.s0.probefix.ext", "teacher.h.cls", "—"),
    ("dino h.gap (teacher)", "toy.dino.s0.probefix.ext", "teacher.h.gap", "—"),
    ("ijepa h.cls (teacher)", "toy.ijepa.s0.ext", "teacher.h.cls", "—"),
    ("ijepa h.gap (teacher)", "toy.ijepa.s0.ext", "teacher.h.gap", "—"),
    ("ijepa z.pred", "toy.ijepa.s0.ext", "student.z.pred.out", "matrix-canonical z"),
    ("mae h.cls", "toy.mae.s0.ext", "student.h.cls", "—"),
    ("mae h.gap", "toy.mae.s0.ext", "student.h.gap", "—"),
    ("randinit h.cls", "toy.randinit-s0.ext", "student.h.cls", "control"),
    ("randinit h.gap", "toy.randinit-s0.ext", "student.h.gap", "control"),
]


THETAS = [0.5, 1.0, 2.0, 5.0]   # declared magnitude grid: k-hat(theta) = #dirs with |kurt|>theta


def main():
    rows = []
    bands = {}
    for label, rid, space, known in CELLS:
        X = np.load(f"{FEAT}/{rid}/{MAN}/{space}.npy").astype(np.float64)
        n = X.shape[0]
        e = khat_eig(X, m=64)
        meff = len(e["eig_kurts"])
        nd = min(20, meff)
        key = (n, meff, nd)
        if key not in bands:
            print(f"[khat] calibrating pursuit band for n={n}, m={meff}, dirs={nd} ...")
            bands[key] = pursuit_null_band(n, meff, nd)
        b = khat_pursuit(X, m=64, n_dirs=nd, seed=3, band=bands[key])
        row = {"cell": label, "d": X.shape[1], "m": meff,
               "pp_band": round(b["pp_null_band"], 2),
               "top|k|_eig": round(float(np.abs(e["eig_kurts"]).max()), 2),
               "top|k|_pp": round(float(b["pp_profile"][0]), 2)}
        for th in THETAS:
            row[f"eig>{th}"] = int((np.abs(e["eig_kurts"]) > th).sum())
            row[f"pp>{th}"] = int((b["pp_profile"] > max(th, b["pp_null_band"])).sum())
        sp = khat_spectrum(X, m=64)
        row["khat_A"] = round(sp["khat_A"], 1) if sp["detect_A"] else float("nan")
        row["detA"] = sp["detect_A"]
        row["known"] = known
        rows.append(row)
        print(f"[khat] {label}: eig-profile {[row[f'eig>{t}'] for t in THETAS]} "
              f"pp-profile {[row[f'pp>{t}'] for t in THETAS]}")
    df = pd.DataFrame(rows)
    out = os.path.join(ROOT, "results", "diag", "defect_rank_toy.csv")
    df.to_csv(out, index=False)
    print(df.to_string(index=False))
    print(f"[khat] wrote {out}")


if __name__ == "__main__":
    main()
