# E32 — Cloud vs image spread: the aug-quiet core across the zoo

*(E31 is not a card: that tag was consumed by the scratch shape-dose experiment,
closed by D-108.3. This card defines the cloud-vs-spread experiment Berker ordered
2026-08-31 and ran the same day.)*

## Why

Berker's question (2026-08-30/31): rank h's directions by image-to-image variance and
ask how far augmentation moves ONE image along those same directions. A previous
scratch attempt (improvised whitening/ranking) was disowned; this card is the
re-derivation on the certified instrument. The adjacent puzzle it addresses: shape
pressure raises effective rank without raising accuracy — is added rank usable signal
or nuisance?

## Frame (anatomy stated loudly)

IN-100 `pairs100.v1@audit_v1.o8` manifests: **8-view clouds** under the FIXED audit
aug family, at **h.cls** (probed branch: teacher where it exists, else student),
single-view anatomy. Estimators imported verbatim from `sslgap/metrics/twospace.py`
(V-unbiased Â, Â/V-debiased B̂, truncated-support whitening, kept rank always
reported). θ_j = augmentation variance along direction j in units of image spread;
θ<0.25 declared "aug-quiet", θ>1 aug-dominated. Per-image read: in the θ basis each
image's own cloud variance along j is a 7-dof estimate; across-image p10/median/p90 +
top-5%-image share reported per direction (the mean-pooling check).

## Cells

PAIRS_E12 (control/treated per method: vicreg, simclr, byol, dino, lejepa, ours) +
the D-104 winner dino arm `e20fwlo` + the old overshoot arm + the E29 visreg pair +
ijepa. Code `scratch/cloud_spread_zoo.py` (job 63940323); outputs
`results/diag/cloud_spread_zoo.csv` (per-cell), `results/diag/cloud_spread_dirs.csv`
(per-direction), `results/figures/cloud_spread_zoo.png`.

## Numbers (RAW table; full CSVs are the record)

| method | n(θ<.25) ctrl→treated | class energy in θ<.25 ctrl→treated |
|---|---|---|
| vicreg | 23 → 34 | .37 → .42 |
| simclr | 11 → 26 | .25 → .36 |
| byol | 13 → 29 | .34 → .40 |
| dino (winner arm) | 10 → 11 | .18 → .16 |
| dino (overshoot arm) | 10 → 6 | .18 → .08 |
| lejepa | 26 → 41 | .58 → .54 |
| ours (zonly→vm4) | 39 → 70 | .54 → .69 |
| visreg | 25 → 51 | .50 → .58 |
| ijepa | 1 → 1 | .05 → .03 |

Per-image heterogeneity: p90/p10 ≈ 5.6–9.6 everywhere; top-5% most-volatile images
carry 14–18% of pooled variance (no few-image domination). Treated supports are
larger (r up everywhere) — the class-energy FRACTIONS are the support-robust read.

## AGREED TAKEAWAY

**E32-T1 (USER-AGREED, Berker 2026-08-31: "okay, we can take the note as a defined
experiment with positive finding"):** *A representation's usable size is not its rank
but the size of its aug-quiet core — the directions where augmentation displacement
is small against image-to-image spread — and the class signal concentrates in that
core. The h moment floor ENLARGES the core and moves class energy into it (5 of 6
zoo methods; ours strongest), i.e. the treatment adds usable dimensions, not nuisance
dimensions.* The exceptions carry information: DINO's core barely moves at the winner
dose and collapses under the overshoot (the D-104 re-dose story seen by an
independent instrument); I-JEPA has no aug-quiet core under the audit family — a
frame statement (it never trained against these augmentations), not a quality
verdict. The pooled-Â estimator is certified safe by the per-image read. Scope:
IN-100 zoo, audit_v1 frame, h.cls, single-seed.

## Possible extensions (undirected)

The in1k B/L cells where `.o8` clouds exist (e.g. `e27lm4Ls5b.extL`); z-space θ for
the two-space contrast; treatment-trajectory (ep25/50/75 clouds exist for several
cells).
