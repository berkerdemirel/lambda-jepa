"""Audit-matrix assembly: join battery frames across spaces, derive transfer ratios
tau(metric, method) = value(h) / value(z) (report §5 E1), and render markdown tables.
Numbers only — interpretation lives in card discussions (CLAUDE.md contract)."""
import numpy as np
import pandas as pd


def transfer_ratios(df, h_space, z_space):
    """df = concatenated run_battery frames for one (run_id, manifest). Returns per
    (metric, variant): value_h, value_z, tau = h/z, delta = h - z."""
    cols = ["metric", "variant", "value", "ci_lo", "ci_hi"]
    h = df[df.space == h_space][cols].set_index(["metric", "variant"])
    z = df[df.space == z_space][cols].set_index(["metric", "variant"])
    j = h.join(z, lsuffix="_h", rsuffix="_z", how="inner").reset_index()
    with np.errstate(divide="ignore", invalid="ignore"):
        j["tau"] = j["value_h"] / j["value_z"]
    j["delta"] = j["value_h"] - j["value_z"]
    j.insert(0, "z_space", z_space)
    j.insert(0, "h_space", h_space)
    return j


def matrix_markdown(frames: dict, metrics=None, variant="raw|full", float_fmt="{:.4g}"):
    """frames: {row_label: battery DataFrame filtered to one space}. Renders metric-per-column
    markdown — the mini audit matrix for M0 and the per-space blocks of the full matrix later."""
    rows = []
    for label, df in frames.items():
        d = df[df.variant == variant] if "variant" in df else df
        r = {"model.space": label}
        for _, x in d.iterrows():
            if metrics and x["metric"] not in metrics:
                continue
            r[x["metric"]] = float_fmt.format(x["value"])
        rows.append(r)
    out = pd.DataFrame(rows).set_index("model.space")
    if metrics:
        out = out[[m for m in metrics if m in out.columns]]
    return out.to_markdown()
