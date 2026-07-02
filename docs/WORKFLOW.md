# WORKFLOW — how results become conclusions in this project

The standing rules of the study. Referenced by every experiment card; changes require a
[DECISIONS](DECISIONS.md) row.

## The conclusions contract

SSL methods are heavily hand-engineered (per-method grid searches, val-fitted probes), so weak
baselines may be tuning artifacts rather than properties of a method. Therefore:

1. **Pre-register, then run.** Every experiment card commits directional predictions *before* the
   numbers exist (the E1 predictions live in [experiments/AUDIT_MATRIX.md](experiments/AUDIT_MATRIX.md)).
2. **Numbers land raw.** Result sections and `results/*/FINDINGS.md` contain numbers, tables, and
   plots — no narrative.
3. **`AGREED TAKEAWAY` sections are filled only after discussion** between the collaborators, and the
   agreed wording is mirrored to [DECISIONS.md](DECISIONS.md). Disputed readings are recorded as
   disputed, not resolved silently.
4. **Protocol is versioned.** Any change to [PROTOCOL.md](PROTOCOL.md) (frames, probes, estimator
   settings, space definitions) requires a DECISIONS row; results computed under different protocol
   versions are never mixed in one table.
5. **Data-ladder gates.** Advancing a rung (toy → IN-100 → public IN-1k → IN-1k retrains) requires a
   gate row (G-*) in DECISIONS.md.
6. **Provenance is never mixed** within a comparison (controlled retrains vs public checkpoints).
7. Failures are reported as failures; parity misses as misses. A discrepancy is a finding, not a
   blemish to smooth over (see HISTORY.md 2026-07-02: the parity-protocol hunt).

## Operational constraints (hard-won; do not relearn)

- All compute via SLURM; even "quick" extraction is a GPU job. H100 usage capped at 2 concurrent
  jobs (singleton names `h100-slotA/B`); everything else on `gpu`/`defaultp`.
- Smoke-test with subset overrides before any sweep — in the predecessor project, setup was
  mistaken for results four times.
- Training hygiene for any new end-to-end loss: `grad_clip=1.0` from day one; cosine
  `eta_min ≤ lr/20`; warmup ≥ 1 epoch; per-audit checkpoints (never overwrite-only); per-step
  grad-norm + per-term loss logging from birth. A single-step grad-norm > 100× the running median
  is an incident: stop and read.
- **Isotropy/Gaussianity semantics:** the sliced Epps–Pulley statistic is a *rejection-style* test
  — a low value (failure to reject) does **not** certify isotropic Gaussianity, and decorrelation +
  CLT make random 1-D slices look Gaussian for almost any high-dimensional cloud. Isotropy claims
  lead with top-eigenvector / worst-direction kurtosis; EP is never reported alone. (See
  [METRICS.md](METRICS.md).)
- CKA is contested for cross-representation correspondence: only ever reported as the triple with
  neighbor-Jaccard and Procrustes.
- DataLoader hygiene: `persistent_workers=True`; never build throwaway iterators per checkpoint.
- Canonical IN-100 is `~/data/imagenet100` (CMC split) — the hyphenated sibling directory is a
  *different* 100-class subset (8% overlap) and is banned from this project (D-002).
- Feature-store budget per D-005: fp16, nothing >8192-d stored, patch tokens
  final-layer/final-epoch/probed-branch only, purge-after-table; hard cap enforced in code.
- Donor code (`third_party/`) is read-only reference: ports go through review, get a PORT_NOTES
  entry in the method dossier (donor commit, findings, deviations), and land in `sslgap/` in our
  style.

## Cross-partition scheduling (H100 budget + gpu partition)

SLURM's native OR (`--partition=gpu100,gpu`) is DISABLED on this cluster ("Multiple partition job
request not supported when a partition is set in the association" — verified 2026-07-02). Working
pattern instead: singleton lanes ONLY where a budget exists — `h100-slotA`/`h100-slotB` on gpu100
(the ≤2-H100 cap). The `gpu` partition is UNCAPPED by policy: submit independent per-method chains
(afterok within a chain, NO shared job name) on `--partition=gpu --constraint="A40|L40S"` and let
fair-share parallelize them (48 GB cards for training; 24 GB 3090Ti/A10 OOM at bs 256 ViT-S/8 and
are reserved for extraction/probes). Feature spellings per
`sinfo -o "%P %f"`. A submit-twice-cancel-loser race wrapper is possible for independent jobs if a
true first-available OR is ever needed; chains break its afterok wiring, so lanes are preferred.
