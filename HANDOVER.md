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
- M0 runs: extraction + probes done for all 6 ckpts (3 lejepa toy, 3 DINO IN-100) + 2 random-init
  nulls; audits + the corrected 30k parity chain in flight. First numbers sane (lejepa toy h.cls
  91.2% linear ≈ official ballpark; clean layerwise guillotine).
- **Parity lesson (docs/HISTORY.md)**: CAMPAIGN_LOG's "DINO @4k" label was wrong — actual reference
  protocol = diag_dino.py, teacher branch, 30k linspace probe-train, RankMe/effrank on val.

## Next action
When audits + parity30k land: `python experiments/report_m0.py 'run_ids=[…]'` → results/M0/FINDINGS.md;
check parity vs the HISTORY reference numbers; then user ratifies D-001…D-008 + signs off PROTOCOL v1
+ E1 predictions (AUDIT_MATRIX) → G-M0 gate row → start M1 (LeJEPA port first).

## Constraints reminder
≤2 concurrent H100 jobs (singleton names `h100-slotA/B`); everything via SLURM; canonical IN-100 =
`~/data/imagenet100` (CMC — the hyphenated dir is a different dataset!); h = trunk features (the
512-d lejepa "emb" is a head tap); no conclusions without a DECISIONS row.
