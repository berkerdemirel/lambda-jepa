"""E24 canonical-o8 Ω_h(end)↔acc Spearman — the re-confirm of the card's in-training
sorting table on the landing channel (audit_v1 o8, raw framing; l2 secondary). Ω_h from
results/diag/e24_toy_spaces.csv (e24_grid_metrics.py over the 29 toy cells); acc = final
best test/acc across all wandb runs sharing the cell's display name (chain links resume
under one name). Groups mirror the card table (pooled/h-free/h.03 · vm-OAS · variants ·
combined). Numbers land RAW (stdout + results/diag/e24_omega_sort.csv); takeaways joint."""
import csv
import os

import numpy as np
import wandb

ROOT = "/nfs/scistore19/locatgrp/bdemirel/ssl_project"
SPACES_CSV = os.path.join(ROOT, "results/diag/e24_toy_spaces.csv")
OUT_CSV = os.path.join(ROOT, "results/diag/e24_omega_sort.csv")

POOLED_HFREE = ["s0.e24s0a", "s0.e24s0b", "s0.e24s0c", "s0.e24s0d", "s0.e24s0ac",
                "s0.e24s0bc", "s0.e24s0cc", "s0.e24wz05", "s0.e24wz12", "s0.e24wz18",
                "s0.e24wz25", "s0.e24wt05", "s0.e24wt2", "s1.e24wrep"]
POOLED_H03 = ["s0.e24s1a", "s0.e24s1b", "s0.e24s1c", "s0.e24s1ac", "s0.e24s1bc",
              "s0.e24s1cc"]
VM_OAS_H03 = ["s0.e24va", "s0.e24vb", "s0.e24vc", "s0.e24vac", "s0.e24v4oas"]
VM_OAS = VM_OAS_H03 + ["s0.e24vh0"]
VARIANTS = ["s0.e24v1", "s0.e24v2lr", "s0.e24v3nq"]
GROUPS = [
    ("pooled n20", POOLED_HFREE + POOLED_H03),
    ("pooled h-free n14", POOLED_HFREE),
    ("pooled h.03 n6", POOLED_H03),
    ("vm-OAS n6", VM_OAS),
    ("vm-OAS h.03 n5", VM_OAS_H03),
    ("view-mean all n9", VM_OAS + VARIANTS),
    ("combined n26", POOLED_HFREE + POOLED_H03 + VM_OAS),
    ("combined+variants n29", POOLED_HFREE + POOLED_H03 + VM_OAS + VARIANTS),
]


def rank_avg(x):
    x = np.asarray(x, dtype=np.float64)
    order = np.argsort(x, kind="stable")
    ranks = np.empty(len(x))
    i = 0
    while i < len(x):
        j = i
        while j + 1 < len(x) and x[order[j + 1]] == x[order[i]]:
            j += 1
        ranks[order[i:j + 1]] = (i + j) / 2.0
        i = j + 1
    return ranks


def spearman(a, b):
    ra, rb = rank_avg(a), rank_avg(b)
    ra, rb = ra - ra.mean(), rb - rb.mean()
    return float((ra * rb).sum() / np.sqrt((ra ** 2).sum() * (rb ** 2).sum()))


def main():
    omega = {}
    with open(SPACES_CSV) as fh:
        for row in csv.DictReader(fh):
            tag = row["run"].removeprefix("toy.floorssl.").removesuffix(".extL")
            if row["space"] == "student.h.cls":
                omega.setdefault(tag, {})[row["framing"]] = float(row["omega"])
            if row["space"] == "student.z.proj.out" and row["framing"] == "raw":
                omega.setdefault(tag, {})["z_raw"] = float(row["omega"])
    api = wandb.Api(timeout=120)
    acc = {}
    for tag in {t for _, cells in GROUPS for t in cells}:
        seed, name = tag.split(".", 1)
        runs = api.runs("causal-learning-ai-ista/sslgap",
                        filters={"display_name": f"toy.floorssl.{seed}.{name}"})
        best = float("nan")
        for r in runs:
            vals = [row["test/acc"] for row in
                    r.history(keys=["test/acc"], samples=2000, pandas=False)
                    if row.get("test/acc") is not None]
            if vals:
                best = np.nanmax([best] + vals)
        acc[tag] = float(best)
        print(f"[cell] {tag}: acc {best:.4f} omega_h(o8) raw "
              f"{omega.get(tag, {}).get('raw', float('nan')):.3f} l2 "
              f"{omega.get(tag, {}).get('l2', float('nan')):.3f}", flush=True)
    rows = []
    print("\ngroup, n, rho(omega_h raw), rho(omega_h l2)")
    for gname, cells in GROUPS:
        missing = [t for t in cells if t not in omega or np.isnan(acc.get(t, np.nan))]
        if missing:
            print(f"[{gname}] MISSING {missing} — skipped")
            continue
        a = [acc[t] for t in cells]
        for framing in ("raw", "l2"):
            o = [omega[t][framing] for t in cells]
            rows.append({"group": gname, "framing": framing, "n": len(cells),
                         "rho": spearman(o, a)})
        r_raw = [r for r in rows if r["group"] == gname and r["framing"] == "raw"][0]
        r_l2 = [r for r in rows if r["group"] == gname and r["framing"] == "l2"][0]
        print(f"{gname}: n={len(cells)} rho_raw {r_raw['rho']:+.3f} "
              f"rho_l2 {r_l2['rho']:+.3f}")
    with open(OUT_CSV, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["group", "framing", "n", "rho"])
        w.writeheader()
        w.writerows(rows)
    cell_csv = OUT_CSV.replace(".csv", "_cells.csv")
    with open(cell_csv, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["tag", "acc_best", "omega_h_o8_raw",
                                           "omega_h_o8_l2", "omega_z_o8_raw"])
        w.writeheader()
        for tag in sorted(acc):
            w.writerow({"tag": tag, "acc_best": acc[tag],
                        "omega_h_o8_raw": omega.get(tag, {}).get("raw"),
                        "omega_h_o8_l2": omega.get(tag, {}).get("l2"),
                        "omega_z_o8_raw": omega.get(tag, {}).get("z_raw")})
    print(f"[write] {OUT_CSV} + {cell_csv}", flush=True)


if __name__ == "__main__":
    main()
