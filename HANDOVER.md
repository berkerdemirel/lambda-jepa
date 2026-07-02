# Handover — sslgap (updated 2026-07-02)

Thin entry point. Layers: `docs/ROADMAP.md` (phases/milestones), `PROTOCOL.md` (fixed frame),
`DECISIONS.md` (ledger), `MODELS.md` (checkpoint matrix), `docs/HISTORY.md` (append-only log),
`CLAUDE.md` (conventions + collaboration contract).

## Goal
Two-space audit of SSL desiderata: measure each method's stated property at the backbone `h` AND at
the head space `z` (+ per head layer), under a matched frame — the completed metrics × methods ×
spaces matrix with transfer ratios τ. Founding study design: `docs/report/ssl-projector-gap-report.html`.

## State (kickstart session, 2026-07-02)
- Repo scaffolded per approved kickstart plan; uv env resolves (torch 2.11.0+cu130).
- Governance docs written (PROTOCOL v1-draft, DECISIONS seeded: 5 locked L-rows, 8 PROPOSED D-rows).
- Knowledge base: in progress this session (ROADMAP, E-cards, dossiers, theory, bibliography).
- Code: in progress this session (ckpt adapters, feature store, battery, probes) — target: M0 smoke
  on existing lejepa + DINO-control checkpoints, zero training.
- solo-learn donor cloned at `third_party/solo-learn` @ 9187ea39 (2026-04-22).

## Next action
Finish M0 (see docs/ROADMAP.md → M0 exit criteria), then: user ratifies D-001…D-008 + signs off
PROTOCOL v1 + E1 predictions → G-M0 gate row → start M1 (LeJEPA port first).

## Constraints reminder
≤2 concurrent H100 jobs (singleton names `h100-slotA/B`); everything via SLURM; canonical IN-100 =
`~/data/imagenet100` (CMC — the hyphenated dir is a different dataset!); h = trunk features (the
512-d lejepa "emb" is a head tap); no conclusions without a DECISIONS row.
