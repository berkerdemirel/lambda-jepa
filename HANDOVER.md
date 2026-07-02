# Handover — sslgap (updated 2026-07-02)

Thin entry point. Layers: `docs/ROADMAP.md` (phases/milestones), `PROTOCOL.md` (fixed frame),
`DECISIONS.md` (ledger), `MODELS.md` (checkpoint matrix), `docs/HISTORY.md` (append-only log),
`CLAUDE.md` (conventions + collaboration contract).

## Goal
Two-space audit of SSL desiderata: measure each method's stated property at the backbone `h` AND at
the head space `z` (+ per head layer), under a matched frame — the completed metrics × methods ×
spaces matrix with transfer ratios τ. Founding study design: `docs/report/ssl-projector-gap-report.html`.

## State (kickstart session, 2026-07-02)
- Repo scaffolded per approved kickstart plan; uv env resolves (torch 2.11.0+cu130); pushed to
  github.com/berkerdemirel/sslgap (SSH).
- Governance docs complete (PROTOCOL v1-draft; DECISIONS: 5 locked L-rows, 8 PROPOSED D-rows).
- Knowledge base complete: ROADMAP, E01–E11 cards + AUDIT_MATRIX pre-registration, 7 dossiers,
  THEORY_MAP (18 rows), OPEN_PROBLEMS (17), BIBLIOGRAPHY (102 refs) + refs.bib.
- Code complete for M0: adapters (lejepa_minimal, sslx_dino + random_init nulls), extractor +
  fp16 feature store, 18-metric battery (variants/bootstrap/Gaussian nulls), probes
  (linear_house_v1/linear_l2_v1/knn_v1), audit tables, Hydra drivers, sbatch templates.
- **M0 pipeline COMPLETE (2026-07-02)**: batteries + probes for all 6 ckpts (3 lejepa toy, 3 DINO
  IN-100 ep25/50/100) + 2 random-init nulls + 3 parity instances. Full numbers:
  `results/M0/FINDINGS.md` (+ per-run CSVs in results/battery, results/probes). Feature store 9.2 GB.
- **Parity: PASSED, exact** — 16/16 reference numbers within ±0.06 pt once the true protocol was
  identified (docs/HISTORY.md: diag_dino.py teacher/linspace-30k, NOT the "@4k" table label);
  RankMe/participation-ratio match to the printed decimal. kNN self-test green.
- Sanity signals: LeJEPA λ=0 arm shows textbook collapse (11.5% probe, RankMe 1.2 — collapse
  monitors work); random-init nulls near chance; layerwise guillotine curves monotone.

## Next action (BLOCKED ON USER — by design)
Discussion with Berker: (1) ratify D-001…D-008 + PROTOCOL v1 + lock AUDIT_MATRIX (E1
pre-registration); (2) walk through results/M0/FINDINGS.md headline cells — NO takeaways are
recorded yet (CLAUDE.md contract); (3) G-M0 gate row → start M1 (LeJEPA port first).

## Constraints reminder
≤2 concurrent H100 jobs (singleton names `h100-slotA/B`); everything via SLURM; canonical IN-100 =
`~/data/imagenet100` (CMC — the hyphenated dir is a different dataset!); h = trunk features (the
512-d lejepa "emb" is a head tap); no conclusions without a DECISIONS row.
