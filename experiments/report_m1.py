"""M1 toy E1-matrix assembler: per-desideratum blocks (value_h, value_z, tau, nulls) for the
7-method grid, lined up against the LOCKED AUDIT_MATRIX predictions, plus the headline-probe
table. NUMBERS ONLY by contract (CLAUDE.md); AGREED-TAKEAWAY stays empty until discussion.

  python experiments/report_m1.py                       # all 7 methods, default run_ids
  python experiments/report_m1.py 'methods=[lejepa,vicreg]'
"""
import os

import hydra
import pandas as pd
from omegaconf import DictConfig

# PROTOCOL §3 / D-003v2 h and z.final per method — mirrors sslgap/ckpt/adapters._NATIVE_ASM
# (pinned there by experiments/adapter_selftest.py EXPECT_H).
H_SPACE = {"simclr": "student.h.gap", "vicreg": "student.h.gap", "byol": "student.h.gap",
           "dino": "teacher.h.cls", "mae": "student.h.gap", "ijepa": "teacher.h.gap",
           "lejepa": "student.z.embed"}
Z_FINAL = {"simclr": "student.z.proj.out", "vicreg": "student.z.proj.out",
           "byol": "student.z.pred.out", "dino": "teacher.z.dino.bottleneck",
           "mae": None,                       # PROTOCOL §3: pixel-loss space — "z = —" cells
           "ijepa": "student.z.pred.out",     # cross-branch h/z pair per §3
           "lejepa": "student.z.proj.out"}

# desideratum -> (csv, lead metric, variant) ; isotropy is NEVER EP alone (§6.4) — EP is emitted
# in the same block. View-predictability is not in the toy battery (enters M2).
DESIDERATA = [
    ("Alignment",       "pairs",   "alignment",                        "pairs"),
    ("Uniformity",      "battery", "uniformity",                       "raw|full"),
    ("Variance floor",  "battery", "variance_floor.hinge",             "raw|full"),
    ("Decorrelation",   "battery", "offdiag_redundancy.mean_abs_corr", "raw|full"),
    ("Eff. rank",       "battery", "rankme",                           "raw|full"),
    ("Isotropy/Gauss.", "battery", "kurt_topeig.worst",                "raw|full"),
    ("Isotropy (EP, paired)", "battery", "epps_pulley",                "raw|full"),
    ("Aug.-invariance", "pairs",   "cos_invariance",                   "pairs"),
]

# AUDIT_MATRIX v1 (LOCKED 2026-07-02) — h/z expectation glyphs, bold = own desideratum.
PREDICTED = {
    "simclr": {"Alignment": "**?/✓**", "Uniformity": "✗/✓", "Variance floor": "?/✓",
               "Decorrelation": "✗/~", "Eff. rank": "~/✗", "Isotropy/Gauss.": "✗/~",
               "Aug.-invariance": "**✗/✓**"},
    "byol":   {"Alignment": "**?/✓**", "Uniformity": "✗/~", "Variance floor": "?/~",
               "Decorrelation": "✗/~", "Eff. rank": "~/~", "Isotropy/Gauss.": "✗/✗",
               "Aug.-invariance": "✗/✓"},
    "dino":   {"Alignment": "?/✓", "Uniformity": "**✗/~**", "Variance floor": "?/?",
               "Decorrelation": "✗/?", "Eff. rank": "~/?", "Isotropy/Gauss.": "✗/✗",
               "Aug.-invariance": "✗/✓"},
    "vicreg": {"Alignment": "?/✓", "Uniformity": "✗/~", "Variance floor": "**~/✓**",
               "Decorrelation": "**✗/✓**", "Eff. rank": "~/✓", "Isotropy/Gauss.": "✗/~",
               "Aug.-invariance": "✗/✓"},
    "mae":    {"Alignment": "✗/—", "Uniformity": "✗/—", "Variance floor": "?/—",
               "Decorrelation": "✗/—", "Eff. rank": "**✗**/—", "Isotropy/Gauss.": "✗/—",
               "Aug.-invariance": "✗/—"},
    "ijepa":  {"Alignment": "?/✓", "Uniformity": "✗/✗", "Variance floor": "?/?",
               "Decorrelation": "✗/?", "Eff. rank": "?/?", "Isotropy/Gauss.": "✗/✗",
               "Aug.-invariance": "✗ (no augs)"},
    "lejepa": {"Alignment": "?/✓", "Uniformity": "?/✓", "Variance floor": "?/✓",
               "Decorrelation": "?/~", "Eff. rank": "?/✓", "Isotropy/Gauss.": "**?/✓**",
               "Aug.-invariance": "? (mild stack)"},
}

HEADLINE_PROBES = ["linear_raw_v1", "knn_v1_k200"]


def _battery_val(df, space, metric, variant):
    r = df[(df.space == space) & (df.metric == metric) & (df.variant == variant)]
    return (r.value.iloc[0], r) if len(r) else (float("nan"), r)


def _pairs_val(df, space, metric, stack="audit_v1"):
    r = df[(df.space == f"{space}") & (df.metric == metric) & (df.manifest.str.contains(stack))]
    return r.value.iloc[0] if len(r) else float("nan")


@hydra.main(version_base=None, config_path="configs", config_name="report_m1")
def main(cfg: DictConfig):
    res = os.path.expanduser(cfg.results_root)
    out_dir = os.path.join(res, "M1")
    os.makedirs(out_dir, exist_ok=True)
    md = ["# M1 — toy E1 matrix (numbers only, dress rehearsal)\n",
          "> Emitted by experiments/report_m1.py. Toy rung: recipe bring-up frame — protocol",
          "> shakedown, NOT locked interpretation (D-012 pending). Predicted glyphs = AUDIT_MATRIX",
          "> v1 (LOCKED); bold = the method's own desideratum. τ = value(h)/value(z) (E01).",
          "> Dim-sensitive metrics: raw|full shown; raw|pca64 in the appendix blocks (§6.2).\n"]

    data = {}
    overrides = cfg.get("run_id_overrides") or {}   # e.g. {dino: toy.dino.s0.probefix.ext}
    for m in cfg.methods:
        rid = overrides.get(m) or cfg.run_id_pattern.format(method=m)
        nid = (rid.removesuffix(".ext") + ".null.ext") if m in overrides \
            else cfg.null_pattern.format(method=m)
        d = {}
        for tag, r in (("", rid), ("null", nid)):
            base = os.path.join(res, "battery", f"{r}.csv")
            if os.path.exists(base):
                d["bat" + tag] = pd.read_csv(base)
            p = os.path.join(res, "battery", f"{r}.pairs.csv")
            if os.path.exists(p):
                d["pairs" + tag] = pd.read_csv(p)
            pr = os.path.join(res, "probes", f"{r}.csv")
            if os.path.exists(pr):
                d["probes" + tag] = pd.read_csv(pr)
        if "bat" not in d:
            print(f"[report_m1] {m}: no battery CSV yet ({rid}) — skipped")
            continue
        data[m] = d

    for name, src, metric, variant in DESIDERATA:
        md.append(f"\n## {name} — `{metric}` ({variant})\n")
        rows = []
        for m, d in data.items():
            h, z = H_SPACE[m], Z_FINAL[m]
            if src == "battery":
                vh, _ = _battery_val(d["bat"], h, metric, variant)
                vz, _ = (_battery_val(d["bat"], z, metric, variant) if z else (float("nan"), None))
                nh, _ = (_battery_val(d["batnull"], h, metric, variant)
                         if "batnull" in d else (float("nan"), None))
                nz, _ = (_battery_val(d["batnull"], z, metric, variant)
                         if ("batnull" in d and z) else (float("nan"), None))
            else:
                vh = _pairs_val(d["pairs"], h, metric) if "pairs" in d else float("nan")
                vz = _pairs_val(d["pairs"], z, metric) if ("pairs" in d and z) else float("nan")
                nh = _pairs_val(d["pairsnull"], h, metric) if "pairsnull" in d else float("nan")
                nz = (_pairs_val(d["pairsnull"], z, metric)
                      if ("pairsnull" in d and z) else float("nan"))
            tau = vh / vz if (z and vz == vz and vz != 0) else float("nan")
            rows.append({"method": m, "h_space": h, "z_final": z or "—",
                         "value_h": vh, "value_z": vz if z else float("nan"), "tau": tau,
                         "null_h": nh, "null_z": nz,
                         "predicted h/z": PREDICTED[m].get(name.split(" (")[0], "")})
        md.append(pd.DataFrame(rows).to_markdown(index=False, floatfmt=".4g"))
        md.append("\n")

    md.append("\n## Headline probes (D-006v2 pair) at h and z.final\n")
    rows = []
    for m, d in data.items():
        if "probes" not in d:
            continue
        pb = d["probes"]
        for space, label in ((H_SPACE[m], "h"), (Z_FINAL[m], "z.final")):
            if space is None:
                continue
            for probe in HEADLINE_PROBES:
                r = pb[(pb.space == space) & (pb.probe == probe)]
                n = (d["probesnull"][(d["probesnull"].space == space)
                                     & (d["probesnull"].probe == probe)]
                     if "probesnull" in d else None)
                rows.append({"method": m, "at": label, "space": space, "probe": probe,
                             "val_acc": r.val_acc.iloc[0] if len(r) else float("nan"),
                             "null": n.val_acc.iloc[0] if n is not None and len(n) else float("nan")})
    md.append(pd.DataFrame(rows).to_markdown(index=False, floatfmt=".4f"))
    md.append("\n")

    md.append("\n## Appendix — full per-space battery pointers + MAE decoder taps\n")
    for m, d in data.items():
        spaces = sorted(d["bat"].space.unique())
        md.append(f"- **{m}**: results/battery/{cfg.run_id_pattern.format(method=m)}.csv — "
                  f"{len(spaces)} spaces: {', '.join(spaces)}")
    md.append("\n\n## AGREED TAKEAWAY\n\n*(empty — filled only after discussion; see CLAUDE.md)*\n")
    out = os.path.join(out_dir, "E1_TOY_MATRIX.md")
    with open(out, "w") as f:
        f.write("\n".join(md))
    print(f"[report_m1] wrote {out} ({len(data)}/{len(cfg.methods)} methods present)")


if __name__ == "__main__":
    main()
