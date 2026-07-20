"""M0 report assembler: gathers battery/pairs/cross/probe CSVs for the given run_ids and emits
results/M0/FINDINGS.md — the mini audit matrix + transfer ratios + probe table + parity block.
NUMBERS ONLY by contract (CLAUDE.md): the AGREED-TAKEAWAY sections stay empty until discussion.

  python experiments/report_m0.py 'run_ids=[toy.lejepa-lamb002.ext,in100.dino-ctrl.ep100.ext]'
"""
import os

import hydra
import pandas as pd
from omegaconf import DictConfig

from sslgap.audit import matrix_markdown, transfer_ratios

KEY_METRICS = ["rankme", "effective_rank", "alpha", "epps_pulley", "kurt_topeig.worst",
               "uniformity", "offdiag_redundancy.mean_abs_corr", "variance_floor.hinge"]


@hydra.main(version_base=None, config_path="configs", config_name="report_m0")
def main(cfg: DictConfig):
    res = os.path.expanduser(cfg.results_root)
    out_dir = os.path.join(res, "M0")
    os.makedirs(out_dir, exist_ok=True)
    md = ["# M0 — mini two-space audit (numbers only)\n",
          "> Emitted by experiments/report_m0.py. Interpretation happens in discussion; "
          "AGREED TAKEAWAYS live in DECISIONS.md after user sign-off.\n"]

    for run_id in cfg.run_ids:
        bat = pd.read_csv(os.path.join(res, "battery", f"{run_id}.csv"))
        md.append(f"\n## {run_id}\n")
        spaces = sorted(bat.space.unique())
        frames = {f"{run_id} · {s}": bat[bat.space == s] for s in spaces}
        md.append("### Battery (variant raw|full)\n")
        md.append(matrix_markdown(frames, metrics=KEY_METRICS))
        md.append("\n")

        h_spaces = [s for s in spaces if ".h." in s]
        z_spaces = [s for s in spaces if ".z." in s]
        if h_spaces and z_spaces:
            md.append("\n### Transfer ratios τ = value(h) / value(z)\n")
            for h in h_spaces:
                for z in z_spaces:
                    if h.split(".")[0] != z.split(".")[0]:
                        continue                      # same branch only
                    tr = transfer_ratios(bat[bat.variant == "raw|full"], h, z)
                    tr = tr[tr.metric.isin(KEY_METRICS)]
                    if len(tr):
                        md.append(f"\n**{h} vs {z}**\n\n")
                        md.append(tr[["metric", "value_h", "value_z", "tau", "delta"]]
                                  .to_markdown(index=False, floatfmt=".4g"))
                        md.append("\n")

        pairs_csv = os.path.join(res, "battery", f"{run_id}.pairs.csv")
        if os.path.exists(pairs_csv):
            pr = pd.read_csv(pairs_csv)
            md.append("\n### Pair metrics (alignment ↓ = more view-invariant)\n\n")
            md.append(pr[["manifest", "space", "metric", "value", "ci_lo", "ci_hi"]]
                      .to_markdown(index=False, floatfmt=".4g"))
            md.append("\n")

        cross_csv = os.path.join(res, "battery", f"{run_id}.cross.csv")
        if os.path.exists(cross_csv):
            cr = pd.read_csv(cross_csv)
            md.append("\n### Cross-space similarity (CKA is contested — read as the triple)\n\n")
            md.append(cr[["space", "metric", "value"]].to_markdown(index=False, floatfmt=".4g"))
            md.append("\n")

        probes_csv = os.path.join(res, "probes", f"{run_id}.csv")
        if os.path.exists(probes_csv):
            pb = pd.read_csv(probes_csv)
            md.append("\n### Probes\n\n")
            md.append(pb.pivot_table(index="space", columns="probe", values="val_acc")
                      .to_markdown(floatfmt=".4f"))
            md.append("\n")

    if cfg.parity_note:
        md.append(f"\n## Parity block\n\n{cfg.parity_note}\n")
    md.append("\n## AGREED TAKEAWAY\n\n*(empty — filled only after discussion; see CLAUDE.md)*\n")
    out = os.path.join(out_dir, "FINDINGS.md")
    with open(out, "w") as f:
        f.write("\n".join(md))
    print(f"[report_m0] wrote {out}")


if __name__ == "__main__":
    main()
