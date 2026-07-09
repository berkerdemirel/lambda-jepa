# E01 — Two-space desiderata audit

> Derived from docs/report/ssl-projector-gap-report.html (compiled 2026-07-01, quote-verified there)
> §5.1 E1. Tags: [E] established · [P] plausible · [O] open.

**Status:** PRE-REGISTERED (M0 mini-version done on existing ckpts → full at M2)
**Pre-registered:** ✅ 2026-07-02 (Berker) — predictions locked in [AUDIT_MATRIX.md](AUDIT_MATRIX.md)
before any controlled-grid number exists.
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

*(numbers only — no prose interpretation here)*

- **Toy (M1) matrix:** `results/M1/E1_TOY_MATRIX.md` (dino = probefix rerun; uniform seeded
  probes) · per-space batteries `results/battery/toy.*.ext.csv` (moment-matched Gaussian nulls in
  `null_gauss`) · probes `results/probes/toy.*.ext.csv` · figures `results/figures/` + wandb
  `m1-viz-gallery`.
- **Verify pass (2026-07-08):** `results/M1/PAIR_MARGIN.md` (+ `pair_margin.csv`,
  `scale_check.csv`) — within-space random-pair baselines for every pair row; clean-vs-augmented
  per-dim scale per z.

## AGREED TAKEAWAY

**Toy-rung (M1 dress rehearsal) — agreed 2026-07-08 (Berker + Claude). Toy numbers lock protocol
lessons and M2 predictions only, never final interpretation (D-012). IN-100 (M2) is the
evidential rung.**

- **T1 · Pair rows are unscoreable raw — FINAL GLYPHS (Berker sign-off 2026-07-08).** Every
  trained h is cone-compact, so raw alignment/cos_invariance read h as *more* view-invariant
  than z for all view-based methods (e.g. BYOL .104 vs .392) — including MAE (.175), which
  trained on no augmentations. Scored on `pair_margin` (D-013;
  results/figures/geometry/pair_margin.png): **view methods (simclr/byol/vicreg/dino/lejepa):
  ✓ at z, ~ at h (relative-only — align_rel halves, 0.79→0.44–0.54, while the absolute margin
  sits at/below the untrained baseline); MAE and I-JEPA: ✗ at h (align_rel unmoved from its
  untrained value; matches their locked predictions).** align_rel behaves as a detector: it
  fires exactly for the aug-trained methods.
- **T2 · Two z-families by fourth-moment behavior (pre-registered M2 prediction).**
  Cluster-sharpening losses (SimCLR/VICReg/DINO) drive z wildly non-Gaussian along top
  eigendirections (kurt-worst 20–27 vs h 0.5–2.5); prediction/Gaussian-pulling losses
  (BYOL/LeJEPA) end with z tamer than h (0.76/1.52 vs 3.16/3.62); I-JEPA flat by construction
  (its z is trained to *be* h-space). Prediction for M2: same signs replicate at IN-100.
- **T3 · Toy probe directions defy the folk rule — no conclusion until IN-100.** z ≥ h for
  SimCLR/VICReg under both headline probes; the headline pair *disagrees on which space is
  better* for BYOL/DINO; feature-type (CLS vs GAP) moves cells as much as h-vs-z (SimCLR h.cls
  .715/.650 vs h.gap .670/.535). Treated strictly as toy-scale observations (10 aug-invariant
  classes, 9.5k images, house optimizer).
- **T4 · LeJEPA: the constraint lives only where it is imposed.** proj.out (SIGReg's space) is
  ~full-rank (RankMe 15.86/16) and mean-slice-tame; there is NO upstream transfer through the
  head — at z.embed (one Linear from the trunk, upstream of the projector: CLS → Linear →
  z.embed → MLP → proj.out) rank is 9.4/512, kurt-worst 3.62, EP 203 — nowhere near isotropic
  Gaussian — yet it probes best-in-grid. Head-depth kurt profile is NON-monotone (embed 3.62 →
  tap1 16.08 → tap2 7.47 → proj.out 1.52): the statistic is satisfied only at the exact layer
  the loss touches. **Honesty-cell reporting format — FINAL (Berker 2026-07-08: "report all"):
  every SIGReg honesty cell reports all three references** (isotropic N(0,I) floor ·
  covariance-matched Gaussian · untrained net) **plus the decomposition** — data−covmatched =
  shape excess (no linear map fixes it), covmatched−floor = anisotropy part
  (`results/M1/SIGREG_DECOMP.md`: at proj.out, 87–94% of the deviation is shape). Consistency
  note: with SIGReg's declared target being the isotropic Gaussian (the ν=∞ corner), the
  moment-matched Gaussian reference is exactly the simple-null case TRD-π's own protocol
  retains (R8f / §5 Stage 4)
  (.804/.788). Observation to re-watch at IN-100: h effective rank ≈ class count (10);
  directional expectation only — rank at h grows with task complexity and stays ≪ width; NOT
  "rank = #classes" (class count is not the only driver of intrinsic dimension). Worst-direction
  honesty cells at proj.out (EP 7–12× matched-Gaussian null; kurt 1.5–3.5 vs ~0.14) recorded;
  their *interpretation* is PENDING the null-semantics discussion. 150ep-vs-800ep contrast
  (EP falls, kurt-worst rises) noted with its confound: recipe AND duration differ.
- **T5 · Which-null discipline.** Several glyph verdicts flip between the random-init-net null
  and the moment-matched Gaussian null (h kurt: tame vs init at 5.49, wild vs Gaussian at 0.065).
  Matrix header now names the null; every scored glyph states its null (D-014).
- **T6 · Guillotine reproduced at toy.** tap1 (first post-activation hidden layer of the head)
  is the best-probing space in 5/5 deep-head methods (LeJEPA proj.tap1 .828 = grid best);
  Bordes-style "peak one layer into the head" holds cross-method. Feeds E02's buffer-zone
  predictions as locked.
- **I-JEPA** is treated as well-behaved on the metric battery (no dissociation, tame kurt,
  healthy rank — structurally expected: its loss target is backbone-space); the open item is the
  metric↔probe discrepancy (healthy metrics, mid-pack probes .653/.548) — logged for E03.
- **T7 · VICReg variance floor (agreed 2026-07-08, Berker's formulation): "just because we
  optimize for it doesn't mean it is satisfied after training — and the scale issue is there at
  VICReg too."** The variance term ends training at an equilibrium, not at satisfaction:
  across-image per-dim std at z = 0.62 on view-batches (the objective's own statistic; hinge 0.38
  in-training [wandb train/var ≈ 0.78 = 2 branches × 0.39] and in-audit alike, despite weight 25).
  The miss is GLOBAL-SCALE only — shape is exactly met (min/mean std 0.973, zero dims below
  half-mean, mean|corr| .050): anti-collapse role ✓, literal γ=1 constraint ✗. Invariance is real
  but partial: within-orbit std 0.35 vs across-image 0.62 (orbit variance ≈ ⅓ of total;
  `results/M1/scale_check.csv`). Mechanism candidate (discussion-level): unnormalized inv (α²)
  and cov (α⁴) reward global shrinkage; the hinge resists linearly; house wd 5e-2 adds pressure —
  recipe-dependent equilibrium. M2 watchpoint: the var-term curve at IN-100. Generalizes the
  ex-post point (DESIDERATA_FRAMEWORK §0): end-of-training term values are equilibria, not
  satisfied constraints.
- **Deferred, flagged for discussion:** decorrelation Pattern-B candidate (VICReg h most
  decorrelated among student-GAP h's, .170 vs .218–.290 — partial transfer of its own
  desideratum; 1 seed, needs M2 seed-1).
