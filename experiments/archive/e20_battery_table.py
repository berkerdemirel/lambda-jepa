"""E20 comparison pass (b): battery-vs-control per zoo lane at the lane's audited h space
(the space carrying the E20 headline probe numbers) plus the lane's loss-terminal z tap
(view lanes; aug-less pair h-only — boundary rows, no further spend). Pure CSV munging over
results/battery/*.csv; no models, no GPU. Rows -> results/compare/e20_battery_vs_ctrl.csv
(tidy: lane, space_role, space, metric, arm, ctrl, delta); focused table -> stdout. RAW."""
import csv
import os

ROOT = "/nfs/scistore19/locatgrp/bdemirel/ssl_project"
OUT = f"{ROOT}/results/compare/e20_battery_vs_ctrl.csv"

H_SPACE = {"simclr": "student.h.cls", "byol": "student.h.cls", "vicreg": "student.h.cls",
           "dino": "teacher.h.cls", "lejepa": "student.z.embed",
           "mae": "student.h.gap", "ijepa": "teacher.h.gap"}
Z_SPACE = {"simclr": "student.z.proj.out", "byol": "student.z.pred.out",
           "vicreg": "student.z.proj.out", "dino": "student.z.dino.bottleneck",
           "lejepa": "student.z.proj.out"}
METRICS = ["effective_rank|raw|full", "rankme|raw|full", "participation_ratio|raw|full",
           "alpha|raw|full", "kurt_topeig.worst|raw|full", "kurt_topeig.mean|raw|full",
           "kurt_slices_mean_abs|raw|full", "epps_pulley|raw|full",
           "gauss_kl_full.total|raw|full", "gauss_kl_full.location|raw|full",
           "gauss_kl_full.spectrum|raw|full", "uniformity|raw|full",
           "offdiag_redundancy.mean_abs_corr|raw|full",
           "variance_floor.min_over_mean_std|raw|full", "collapse_margin.min_std|raw|full"]
PAIR_METRICS = None  # discovered from files (all metric|variant present in pairs csvs)


def load(path, manifest=None):
    """manifest: substring filter on the manifest column — REQUIRED for pairs files, which
    carry one row-block per aug stack (audit_v1/blur/foveal/own_<m>); a keyed dict without the
    filter silently keeps whichever block is last in file order (the 2026-07-20 frame-mixing
    bug: ctrl pairs read own/foveal stacks while lejepa-e20f read audit_v1). Cross-lane pair
    comparisons are defined on the fixed audit_v1 stack (PROTOCOL §5)."""
    if not os.path.exists(path):
        return {}
    d = {}
    for r in csv.DictReader(open(path)):
        if manifest and manifest not in r["manifest"]:
            continue
        d[(r["space"], r["metric"] + "|" + r["variant"])] = float(r["value"])
    return d


def main():
    rows = []
    for lane in H_SPACE:
        arm = load(f"{ROOT}/results/battery/in100.{lane}.s0.e20f.ext.csv")
        ctl = load(f"{ROOT}/results/battery/in100.{lane}.s0.ext.csv")
        arm_p = load(f"{ROOT}/results/battery/in100.{lane}.s0.e20f.ext.pairs.csv", "@audit_v1")
        ctl_p = load(f"{ROOT}/results/battery/in100.{lane}.s0.ext.pairs.csv", "@audit_v1")
        pair_metrics = sorted({k[1] for k in arm_p} & {k[1] for k in ctl_p})
        for role, space in [("h", H_SPACE[lane])] + ([("z", Z_SPACE[lane])] if lane in Z_SPACE else []):
            for met in METRICS:
                a, c = arm.get((space, met)), ctl.get((space, met))
                if a is None and c is None:
                    continue
                rows.append({"lane": lane, "space_role": role, "space": space, "metric": met,
                             "arm": a, "ctrl": c,
                             "delta": None if None in (a, c) else a - c})
            if role == "h":
                for met in pair_metrics:
                    a, c = arm_p.get((space, met)), ctl_p.get((space, met))
                    if a is None and c is None:
                        continue
                    rows.append({"lane": lane, "space_role": "h.pairs", "space": space,
                                 "metric": met, "arm": a, "ctrl": c,
                                 "delta": None if None in (a, c) else a - c})
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print("wrote", OUT, f"({len(rows)} rows)")

    focus = ["effective_rank|raw|full", "rankme|raw|full", "kurt_topeig.worst|raw|full",
             "epps_pulley|raw|full", "gauss_kl_full.total|raw|full",
             "gauss_kl_full.location|raw|full", "uniformity|raw|full"]
    short = {m: m.split("|")[0] for m in focus}
    for role in ["h", "z"]:
        print(f"\n== {role}-space: arm / ctrl ==")
        hdr = f"{'lane':8s}" + "".join(f"{short[m]:>26s}" for m in focus)
        print(hdr)
        for lane in H_SPACE:
            cells = []
            for m in focus:
                r = next((x for x in rows if x["lane"] == lane and x["space_role"] == role
                          and x["metric"] == m), None)
                cells.append("-" if r is None or r["arm"] is None or r["ctrl"] is None
                             else f"{r['arm']:.3g}/{r['ctrl']:.3g}")
            print(f"{lane:8s}" + "".join(f"{c:>26s}" for c in cells))
    print("\n== h pair metrics: arm / ctrl ==")
    pm = sorted({r["metric"] for r in rows if r["space_role"] == "h.pairs"})
    print(f"{'lane':8s}" + "".join(f"{p.split('|')[0]:>26s}" for p in pm))
    for lane in H_SPACE:
        cells = []
        for m in pm:
            r = next((x for x in rows if x["lane"] == lane and x["space_role"] == "h.pairs"
                      and x["metric"] == m), None)
            cells.append("-" if r is None else f"{r['arm']:.3g}/{r['ctrl']:.3g}")
        print(f"{lane:8s}" + "".join(f"{c:>26s}" for c in cells))


if __name__ == "__main__":
    main()
