"""M2 IN-100 E1-matrix assembler (seed 0): per-desideratum blocks (value_h, value_z, tau, nulls)
lined up against the LOCKED AUDIT_MATRIX predictions, headline-probe table (v2 converged family
per D-020, with cap-censor flags), and the E01-T2 locked-signs block laid beside the numbers
UNSCORED. NUMBERS ONLY by contract; AGREED-TAKEAWAY stays empty until discussion.

M2-specific frame notes: the null is the SHARED in100.randinit-s0.ext (h-side spaces only — no
per-method z-space nulls extracted at this rung; those columns are NaN by construction, not by
failure). Battery blocks appear for whichever audits have landed; rerun after vicreg/byol.

  python experiments/report_m2.py
"""
import os

import hydra
import pandas as pd
from omegaconf import DictConfig

from experiments.report_m1 import DESIDERATA, H_SPACE, PREDICTED, Z_FINAL, _battery_val, _pairs_val

HEADLINE_PROBES = ["linear_raw_v2", "knn_v1_k200"]           # D-020 headline pair
CONTINUITY_PROBES = ["linear_raw_v1", "linear_house_v2"]     # shown alongside, never headline

# E01-T2 (USER-APPROVED, toy): fourth-moment families — pre-registered SIGN predictions at IN-100
T2_PREDICTION = {"simclr": "cluster-sharpening: kurt z >> h", "vicreg": "cluster-sharpening: kurt z >> h",
                 "dino": "cluster-sharpening: kurt z >> h", "byol": "Gaussian-smoothing: kurt z < h",
                 "lejepa": "Gaussian-smoothing: kurt z < h", "ijepa": "flat (z trained to be h-space)",
                 "mae": "no z prediction (pixel loss)"}


@hydra.main(version_base=None, config_path="configs", config_name="report_m2")
def main(cfg: DictConfig):
    res = os.path.expanduser(cfg.results_root)
    out_dir = os.path.join(res, "M2")
    os.makedirs(out_dir, exist_ok=True)
    md = ["# M2 — IN-100 E1 matrix, seed 0 (numbers only, SKELETON)\n",
          "> Emitted by experiments/report_m2.py. NO cell is scored — glyph resolution happens in",
          "> discussion (CLAUDE.md contract). Predicted glyphs = AUDIT_MATRIX v1 (LOCKED",
          "> 2026-07-02); bold = the method's own desideratum. tau = value(h)/value(z).",
          "> Probes = D-020 v2 family (patience-converged; `capped` marks best_ep within",
          "> patience-reach of the 1000-ep cap). Null columns = shared in100.randinit-s0.ext",
          "> (h-side only at this rung; z-side nulls not extracted — NaN by construction).",
          "> Pair rows are margin-scored per D-013/E01-T8: results/M2/PAIR_MARGIN.md.",
          "> Seed-1 replication pending (G-M2 requirement) — nothing here is a headline cell.\n"]

    null_run = cfg.null_run
    dnull = {}
    for kind, suffix in (("bat", ".csv"), ("pairs", ".pairs.csv")):
        p = os.path.join(res, "battery", f"{null_run}{suffix}")
        if os.path.exists(p):
            dnull[kind] = pd.read_csv(p)

    def null_space(space):
        # the shared randinit stores student.* only; at random init teacher == student (EMA
        # branches are deepcopies, adapters.from_native), so teacher-space nulls read across
        return space.replace("teacher.", "student.", 1) if space else space

    data, missing_bat = {}, []
    for m in cfg.methods:
        rid = cfg.run_id_pattern.format(method=m)
        d = {}
        for kind, sub, suffix in (("bat", "battery", ".csv"), ("pairs", "battery", ".pairs.csv"),
                                  ("probes", "probes", ".csv")):
            p = os.path.join(res, sub, f"{rid}{suffix}")
            if os.path.exists(p):
                d[kind] = pd.read_csv(p)
        if "bat" not in d:
            missing_bat.append(m)
        data[m] = d
    if missing_bat:
        md.append(f"> **Batteries not yet landed for: {', '.join(missing_bat)}** (audit jobs"
                  " running) — their battery rows are absent below; rerun this script after.\n")

    for name, src, metric, variant in DESIDERATA:
        md.append(f"\n## {name} — `{metric}` ({variant})\n")
        rows = []
        for m, d in data.items():
            h, z = H_SPACE[m], Z_FINAL[m]
            if src == "battery":
                if "bat" not in d:
                    continue
                vh, _ = _battery_val(d["bat"], h, metric, variant)
                vz, _ = (_battery_val(d["bat"], z, metric, variant) if z else (float("nan"), None))
                nh, _ = (_battery_val(dnull["bat"], null_space(h), metric, variant)
                         if "bat" in dnull else (float("nan"), None))
                nz, _ = (_battery_val(dnull["bat"], null_space(z), metric, variant)
                         if ("bat" in dnull and z) else (float("nan"), None))
            else:
                if "pairs" not in d:
                    continue
                vh = _pairs_val(d["pairs"], h, metric)
                vz = _pairs_val(d["pairs"], z, metric) if z else float("nan")
                nh = (_pairs_val(dnull["pairs"], null_space(h), metric)
                      if "pairs" in dnull else float("nan"))
                nz = (_pairs_val(dnull["pairs"], null_space(z), metric)
                      if ("pairs" in dnull and z) else float("nan"))
            tau = vh / vz if (z and vz == vz and vz != 0) else float("nan")
            rows.append({"method": m, "h_space": h, "z_final": z or "—",
                         "value_h": vh, "value_z": vz if z else float("nan"), "tau": tau,
                         "null_h(randinit)": nh, "null_z(randinit)": nz,
                         "predicted h/z": PREDICTED[m].get(name.split(" (")[0], "")})
        if rows:
            md.append(pd.DataFrame(rows).to_markdown(index=False, floatfmt=".4g"))
        md.append("\n")

    md.append("\n## E01-T2 locked sign predictions vs numbers — UNSCORED (joint reading only)\n")
    rows = []
    for m, d in data.items():
        if "bat" not in d:
            rows.append({"method": m, "locked prediction": T2_PREDICTION[m],
                         "kurt_worst h": "battery pending", "kurt_worst z": "", "null_gauss z": ""})
            continue
        h, z = H_SPACE[m], Z_FINAL[m]
        vh, _ = _battery_val(d["bat"], h, "kurt_topeig.worst", "raw|full")
        vz, rz = (_battery_val(d["bat"], z, "kurt_topeig.worst", "raw|full")
                  if z else (float("nan"), None))
        ng = rz.null_gauss.iloc[0] if rz is not None and len(rz) else float("nan")
        rows.append({"method": m, "locked prediction": T2_PREDICTION[m],
                     "kurt_worst h": f"{vh:.3g}", "kurt_worst z": f"{vz:.3g}" if z else "—",
                     "null_gauss z": f"{ng:.3g}" if ng == ng else "—"})
    md.append(pd.DataFrame(rows).to_markdown(index=False))
    md.append("\n")

    md.append("\n## Headline probes (D-020 v2 pair) at h and z.final\n")
    md.append("> continuity columns (v1 raw @30ep — known censored at GAP spaces — and house_v2)"
              " shown for reference; headline = raw_v2 + knn.\n")
    rows = []
    for m, d in data.items():
        if "probes" not in d:
            continue
        pb = d["probes"]
        for space, label in ((H_SPACE[m], "h"), (Z_FINAL[m], "z.final")):
            if space is None:
                continue
            row = {"method": m, "at": label, "space": space}
            for probe in HEADLINE_PROBES + CONTINUITY_PROBES:
                r = pb[(pb.space == space) & (pb.probe == probe)]
                if not len(r):
                    row[probe] = float("nan")
                    continue
                v = r.val_acc.iloc[0]
                cap = ""
                if "best_ep" in r and "epochs_run" in r and r.epochs_run.notna().iloc[0]:
                    ep_run, best_ep = int(r.epochs_run.iloc[0]), int(r.best_ep.iloc[0])
                    if ep_run >= 1000 and best_ep >= ep_run - 120:
                        cap = " (capped)"
                row[probe] = f"{v:.4f}{cap}"
            rows.append(row)
    md.append(pd.DataFrame(rows).to_markdown(index=False))
    md.append("\n\n## AGREED TAKEAWAY\n\n*(empty — filled only after discussion; see CLAUDE.md)*\n")
    out = os.path.join(out_dir, "E1_IN100_MATRIX.md")
    with open(out, "w") as f:
        f.write("\n".join(md))
    print(f"[report_m2] wrote {out} (batteries present: "
          f"{len([m for m in data if 'bat' in data[m]])}/{len(cfg.methods)})")


if __name__ == "__main__":
    main()
