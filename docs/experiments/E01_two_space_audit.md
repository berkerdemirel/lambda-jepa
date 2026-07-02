# E01 — Two-space desiderata audit

> Derived from docs/report/ssl-projector-gap-report.html (compiled 2026-07-01, quote-verified there)
> §5.1 E1. Tags: [E] established · [P] plausible · [O] open.

**Status:** draft (M0 mini-version on existing ckpts → full at M2)
**Pre-registered:** ❏ pending user sign-off (target: M0 exit) — predictions live in
[AUDIT_MATRIX.md](AUDIT_MATRIX.md) and must be locked *before* the first full-grid number is computed.
**Phase:** M0 (2 methods, existing ckpts) → M2 (core-7 + anchors, IN-100 grid) → M3 (public echo).

## Hypothesis (directional, from the report)

Each method satisfies its own stated desideratum in `z` and violates it at `h`; the magnitude of the
violation varies by family. Three competing patterns (report §3.2), all pre-registered as possible:

- **Pattern A — clean dissociation**: desiderata hold at z, fail at h, yet h probes better →
  confirms the buffer account; deliverable = first quantitative transfer law + per-method τ. (Most
  likely per RCDM/Guillotine. [P])
- **Pattern B — partial transfer tracks quality**: methods whose h inherits more of the desideratum
  probe better within-family → τ becomes a model-selection signal beyond RankMe. [O]
- **Pattern C — different desiderata matter at h**: h-space effective rank & information retention
  predict downstream while h-space alignment/uniformity don't → "the field regularizes the wrong
  space with the right words". Most publishable outcome. [P]

LeJEPA-specific sub-hypothesis: its z→h **isotropy** dissociation is smaller than the **invariance**
dissociation of view-based methods (a distributional constraint may transfer through the head better
than a pointwise one) — the reframed version of its "provable" claim.

## Protocol

- Features per model per space from the store (PROTOCOL §5 manifests: `m50k` battery, `val5k`
  probes, `pairs10k` alignment/invariance under BOTH aug stacks).
- Battery (PROTOCOL §6 discipline): alignment, uniformity, variance-floor profile, off-diag
  covariance/redundancy, RankMe, effective rank/participation ratio, spectral α, isotropy pair
  (Epps–Pulley + top-eig kurtosis), collapse margin, per-augmentation invariance,
  kNN-consistency(h,z), similarity triple (CKA + neighbor-overlap + Procrustes), view-predictability.
- Spaces: `h.cls`, `h.gap` (final layer; per-layer at E02), every `z.*` tap per PROTOCOL §3.
- Probes: `linear_l2_v1`, `knn_v1` (+ `linear_house_v1` parity at M0; `attentive_v1` on token spaces).
- Derived: **τ(m, M) = m(h) / m(z)** per metric×method, with bootstrap CIs; nulls (random-init,
  Gaussian-matched, label-shuffle) on every table.
- M0 scope: LeJEPA (3 toy ckpts: λ=0.02 / λ=0 / InfoNCE) + DINO (3 IN-100 control ckpts) — enough to
  prove the pipeline and stress dimension-sensitivity (z=16-d vs h=384-d).

## Data / models

M0: Track-A ckpts (MODELS.md). M2: core-7 × 2 seeds + supervised DeiT-lite + random-init on
`in100_vits16`. M3: Track-C public ckpts (patterns only; provenance-flagged, never mixed in-table).

## Expected failure to reveal

h fails the advertised properties across the board (RCDM/Guillotine prediction) while z satisfies
them; per-method τ heterogeneity is the new measurement. If instead τ ≈ 1 anywhere, that method's
desideratum genuinely reaches the backbone — flag immediately for discussion.

## Interpretation guide (report §5.1)

τ ≈ 0: pure buffer. τ ≈ 1 with worse accuracy at z: desiderata epiphenomenal. τ heterogeneous
across methods: desiderata-transfer is an unread design dial.

## Analysis plan

Directional tests per AUDIT_MATRIX cell; Holm–Bonferroni within metric families; report per-cell
(value_h, value_z, τ, CI, null-delta). No cross-provenance rows in one table.

## Kill criteria

Estimator instability across bootstrap resamples (CI spanning sign flips) in >20% of cells → stop,
fix estimator discipline before interpreting anything.

## Results

*(numbers only — links to results/E01/; no prose interpretation here)*

## AGREED TAKEAWAY

*(empty — filled only after discussion with Berker; mirrored to DECISIONS.md)*
