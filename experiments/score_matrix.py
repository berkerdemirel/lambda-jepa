"""D-024 mechanical glyph scoring of the E1 IN-100 matrix (seed 0) against AUDIT_MATRIX v1.

One-time delegated pass (Berker 2026-07-10: "i will trust your decisions on d24 only for this
one, dont be conclusive") — every glyph is the output of the fixed rules below, laid beside its
inputs, with EVERY CELL VETO-OPEN. This artifact is NOT agreed interpretation; agreed takeaways
live on the E01 card + DECISIONS only. Null wiring mirrors report_m2 (shared randinit, own-arch
nulls where landed, teacher->student read-across for the shared null); pair families are
margin-scored per D-013/E01-T8 (rule transcribed from the toy T8 resolution).

  python experiments/score_matrix.py
    -> results/M2/E1_IN100_SCORED.md + results/M2/e1_in100_scored.csv
"""
import os
import re

import numpy as np
import pandas as pd
import yaml

from report_m1 import H_SPACE, PREDICTED, Z_FINAL, _battery_val
from sslgap.metrics.pairs import pair_margin

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(ROOT, "results")
FEAT = os.path.join(ROOT, "features")

# AUDIT_MATRIX column absent from report_m1.PREDICTED (probe-based; no cell at this rung)
VIEWPRED = {"simclr": "?/—", "byol": "?/✓", "vicreg": "?/—", "dino": "?/—",
            "mae": "**?/—**", "ijepa": "**?/✓**", "lejepa": "**?/✓**"}

RULES = {
    "Alignment": "pairs (D-013/E01-T8): cos_margin + align_rel vs the shared-null same-space "
                 "baseline. ✓ margin ≥ .5 AND ≥ 3x null margin · ~ align_rel ≤ .7x null "
                 "align_rel OR margin ≥ 1.5x null margin (T8's 'relative-only') · ✗ else. "
                 "z has no null pair store at this rung: absolute ✓ ≥ .5 / ~ ≥ .25.",
    "Uniformity": "battery `uniformity` (raw|full; 0 = collapsed, more negative = more uniform): "
                  "✓ ≤ −3 · ~ ≤ −1.5 · ✗ else. Randinit null shown for reference.",
    "Variance floor (scale-free)": "`min_over_mean_std` ∈[0,1] (all-dims-alive): ✓ ≥ .5 · "
                                   "~ ≥ .1 · ✗ else (absolute; D-014 scale-free lead).",
    "Decorrelation": "`mean_abs_corr` (absolute): ✓ ≤ .05 · ~ ≤ .15 · ✗ else.",
    "Eff. rank": "`rankme` / d (fraction of ambient dim): ✓ ≥ .5 · ~ ≥ .2 · ✗ else. "
                 "SHAKIEST RULE — random features are high-rank, so the null is no reference "
                 "and the fraction threshold is a choice. Veto expected here first.",
    "Isotropy/Gauss.": "`kurt_topeig.worst` vs the MATCHED-GAUSSIAN null (T5 which-null): "
                       "q = |v|/max(|null_gauss|,.02): ✓ q ≤ 3 · ~ q ≤ 10 · ✗ else. "
                       "EP row stays unscored beside kurt (T5: never EP alone).",
    "Aug.-invariance": "same margin rule as Alignment (T8 scored the two jointly).",
    "View-predictability": "no cell at this rung (probe-based) — not measured.",
}


def _measured_glyph(fam, v, null, d=None, ngauss=None, mrow=None, nullm=None):
    if fam in ("Alignment", "Aug.-invariance"):
        mg, ar = mrow["cos_margin"], mrow["align_rel"]
        if nullm is None:
            return "✓" if mg >= 0.5 else ("~" if mg >= 0.25 else "✗")
        if mg >= 0.5 and mg >= 3 * nullm["cos_margin"]:
            return "✓"
        if ar <= 0.7 * nullm["align_rel"] or mg >= 1.5 * nullm["cos_margin"]:
            return "~"
        return "✗"
    if v != v:
        return "—"
    if fam == "Uniformity":
        return "✓" if v <= -3 else ("~" if v <= -1.5 else "✗")
    if fam == "Variance floor (scale-free)":
        return "✓" if v >= 0.5 else ("~" if v >= 0.1 else "✗")
    if fam == "Decorrelation":
        return "✓" if v <= 0.05 else ("~" if v <= 0.15 else "✗")
    if fam == "Eff. rank":
        f = v / d
        return "✓" if f >= 0.5 else ("~" if f >= 0.2 else "✗")
    if fam == "Isotropy/Gauss.":
        q = abs(v) / max(abs(ngauss) if ngauss == ngauss else 0.02, 0.02)
        return "✓" if q <= 3 else ("~" if q <= 10 else "✗")
    return "—"


def _verdict(pred, meas):
    pred = pred.strip("*")
    if pred == "—" or meas == "—":
        return "n/a"
    if pred == "?":
        return f"NEW:{meas}"
    if pred == meas:
        return "MATCH"
    if "~" in (pred, meas):
        return "soft"
    return "SURPRISE"


def _margins(run_id, space):
    pdir = os.path.join(FEAT, run_id, "in100.pairs100.v1@audit_v1")
    fa = os.path.join(pdir, f"{space}.viewA.npy")
    if not os.path.exists(fa):
        return None
    A = np.load(fa).astype(np.float64)
    B = np.load(fa.replace(".viewA.npy", ".viewB.npy")).astype(np.float64)
    return pair_margin(A, B)


def main():
    cfg = yaml.safe_load(open(os.path.join(ROOT, "experiments", "configs", "report_m2.yaml")))
    bat = {m: pd.read_csv(os.path.join(RES, "battery", f"in100.{m}.s0.ext.csv"))
           for m in cfg["methods"]}
    nullbat = {m: pd.read_csv(p) for m in cfg["methods"]
               if os.path.exists(p := os.path.join(
                   RES, "battery", f"{(cfg.get('method_null_runs') or {}).get(m, cfg['null_run'])}.csv"))}
    shared_null = pd.read_csv(os.path.join(RES, "battery", f"{cfg['null_run']}.csv"))

    fams = ["Alignment", "Uniformity", "Variance floor (scale-free)", "Decorrelation",
            "Eff. rank", "Isotropy/Gauss.", "Aug.-invariance", "View-predictability"]
    METRIC = {"Uniformity": ("uniformity", "raw|full"),
              "Variance floor (scale-free)": ("variance_floor.min_over_mean_std", "raw|full"),
              "Decorrelation": ("offdiag_redundancy.mean_abs_corr", "raw|full"),
              "Eff. rank": ("rankme", "raw|full"),
              "Isotropy/Gauss.": ("kurt_topeig.worst", "raw|full")}

    md = ["# E1 IN-100 SCORED (D-024 mechanical pass, seed 0) — EVERY CELL VETO-OPEN\n",
          "> Emitted by experiments/score_matrix.py. Glyphs are MECHANICAL-RULE outputs "
          "(delegated one-time, Berker 2026-07-10), laid beside their inputs; they are NOT "
          "agreed interpretation. Predictions = AUDIT_MATRIX v1 (LOCKED 2026-07-02); bold = "
          "own desideratum. Verdicts: MATCH · soft (a ~ on either side) · SURPRISE (✓↔✗) · "
          "NEW:<glyph> (prediction was ?, the study's new measurement — no match possible) · "
          "n/a (space absent / not measured). Nulls per D-014/T5: named per family below; "
          "own-arch nulls where landed, shared randinit otherwise (teacher→student "
          "read-across). Single seed per D-024.\n", "## Rules (the delegated choices)\n"]
    md += [f"- **{f}**: {RULES[f]}" for f in fams]

    flat = []
    for fam in fams:
        md.append(f"\n## {fam} — predicted vs measured\n")
        rows = []
        for m in cfg["methods"]:
            h, z = H_SPACE[m], Z_FINAL[m]
            pred = (PREDICTED[m].get(fam.split(" (")[0]) if fam != "View-predictability"
                    else VIEWPRED[m])
            if fam == "View-predictability":
                rows.append({"method": m, "predicted h/z": pred,
                             "measured h/z": "not measured at this rung", "verdict": "n/a"})
                flat.append({"family": fam, "method": m, "pred": pred, "gh": "", "gz": "",
                             "verdict_h": "n/a", "verdict_z": "n/a"})
                continue
            if fam in ("Alignment", "Aug.-invariance"):
                mh, mz = _margins(f"in100.{m}.s0.ext", h), (
                    _margins(f"in100.{m}.s0.ext", z) if z else None)
                nh = _margins(cfg["null_run"], h.replace("teacher.", "student.", 1))
                gh = _measured_glyph(fam, 0, 0, mrow=mh, nullm=nh) if mh else "—"
                gz = _measured_glyph(fam, 0, 0, mrow=mz, nullm=None) if mz else "—"
                nfmt = ((f"{nh['cos_margin']:.2f}", f"{nh['align_rel']:.2f}") if nh
                        else ("— no null store (lejepa-arch gap)",) * 2)
                iv_h = (f"mrg {mh['cos_margin']:.2f} (null {nfmt[0]}), "
                        f"arel {mh['align_rel']:.2f} (null {nfmt[1]})") if mh else "—"
                iv_z = f"mrg {mz['cos_margin']:.2f}, arel {mz['align_rel']:.2f}" if mz else "—"
            else:
                metric, variant = METRIC[fam]
                vh, rh = _battery_val(bat[m], h, metric, variant)
                vz, rz = _battery_val(bat[m], z, metric, variant) if z else (float("nan"), None)
                dn = nullbat.get(m, shared_null)
                ns = (lambda s: s) if m in (cfg.get("method_null_runs") or {}) else (
                    lambda s: s.replace("teacher.", "student.", 1))
                nh_v, _ = _battery_val(dn, ns(h), metric, variant)
                nz_v, _ = _battery_val(dn, ns(z), metric, variant) if z else (float("nan"), None)
                gh = _measured_glyph(fam, vh, nh_v, d=rh.d.iloc[0] if len(rh) else np.nan,
                                     ngauss=rh.null_gauss.iloc[0] if len(rh) else np.nan)
                gz = _measured_glyph(fam, vz, nz_v, d=rz.d.iloc[0] if rz is not None and len(rz)
                                     else np.nan, ngauss=rz.null_gauss.iloc[0]
                                     if rz is not None and len(rz) else np.nan) if z else "—"
                iv_h = f"{vh:.3g} (null {nh_v:.3g})"
                iv_z = f"{vz:.3g} (null {nz_v:.3g})" if z else "—"
            g = re.findall(r"[✓✗~?—]", pred or "")
            ph, pz = (g + ["—", "—"])[:2]
            vd_h, vd_z = _verdict(ph, gh), _verdict(pz, gz)
            rows.append({"method": m, "h inputs": iv_h, "glyph_h": gh, "z inputs": iv_z,
                         "glyph_z": gz, "predicted h/z": pred,
                         "verdict h/z": f"{vd_h} / {vd_z}"})
            flat.append({"family": fam, "method": m, "pred": pred, "gh": gh, "gz": gz,
                         "verdict_h": vd_h, "verdict_z": vd_z})
        md.append(pd.DataFrame(rows).to_markdown(index=False))
        md.append("")

    df = pd.DataFrame(flat)
    scored = df[~df.verdict_h.isin(["n/a"]) | ~df.verdict_z.isin(["n/a"])]
    md.append("\n## Summary (mechanical; no interpretation)\n")
    for side in ("h", "z"):
        c = df[f"verdict_{side}"].value_counts().to_dict()
        md.append(f"- {side}-side: " + ", ".join(f"{k} {v}" for k, v in sorted(c.items())))
    surp = df[(df.verdict_h == "SURPRISE") | (df.verdict_z == "SURPRISE")]
    md.append("- SURPRISE cells (flagged for discussion before any narrative, AUDIT_MATRIX "
              "rule): " + ("; ".join(f"{r.method}/{r.family}" for r in surp.itertuples())
                           if len(surp) else "none"))
    df.to_csv(os.path.join(RES, "M2", "e1_in100_scored.csv"), index=False)
    out = os.path.join(RES, "M2", "E1_IN100_SCORED.md")
    open(out, "w").write("\n".join(md) + "\n")
    print(f"[score] wrote {out} ({len(scored)} scored cells)")
    print(df.groupby("family")[["verdict_h", "verdict_z"]]
          .agg(lambda s: dict(s.value_counts())).to_string())


if __name__ == "__main__":
    main()
