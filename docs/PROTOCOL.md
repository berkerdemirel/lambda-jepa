# PROTOCOL.md — the fixed experimental frame

**Version: v1-draft.5 (2026-07-14: D-036 — h redeclared as the projection input (pre-MLP); F1
methods SimCLR/BYOL/VICReg h moved trunk-GAP→trunk-CLS in §1/§3/F1; online monitor tap follows the
declared h; trunk-GAP demoted to intermediate tap. prior: v1-draft.4, 2026-07-10: D-025 k̂ defect-rank estimator spec, §6 item 9; prior:
v1-draft.3, 2026-07-09: D-020 convergence-guaranteed linear probes — v2 family is the
headline; v1-draft.2, 2026-07-02: D-003v2 space definitions, D-006v2 probe proposal, D-009
relrep). Pre-registration lock pending user sign-off.**
Any change to this file requires a DECISIONS row. Derived from the report's §5.0 fixed frame and §3
estimator discipline, adapted to this cluster and the binding data ladder (DECISIONS L-004).

## 1 · Spaces and notation (D-003v2 — Berker's rule)

- **`h` — "the representation": the input of the method's projection/head (pre-MLP), per D-036
  (2026-07-14).** For most methods this coincides with the feature the paper probes for its main
  linear-probe tables; for the ResNet-native F1 methods (SimCLR/BYOL/VICReg) the ViT port makes it
  the trunk-**CLS** (the projector's actual input), not trunk-GAP (superseded — see F1 below).
  Defined per method in §3 (projector inputs differ: CLS, embed, teacher branch, GAP-on-uncorrupted).
- **`z` — "the proj": the space the training loss is applied to**, `z.<role>.out`, plus every
  intermediate head-layer tap `z.<role>.tapK`. Role ∈ {embed, proj, pred, dec}.
- **Linear-only clause:** if a method's loss is applied after ONLY a linear projection of h, it
  counts as loss-on-representation ("we respect the method") and its audit row is flagged
  `same-space≈yes`. No core-7 method qualifies (nearest: MoCo v1's linear fc — later roster).
- All other measured spaces (trunk per-layer `*.h.cls.Lkk`/`*.h.gap.Lkk`, the lejepa
  `z.embed` Linear, head taps) are **guillotine points** (E2) — measured identically, but not "h".
- Branches: `student.*` (online) and `teacher.*` (EMA, where one exists); the h column in §3 names
  branch + feature exactly.
- τ(m, M) = metric m at h ÷ metric m at z (h and z per §3), per method M (report §5 E1).

## 2 · Frames (one per data-ladder rung)

| frame id | backbone | data (canonical root) | resolution | epochs | seeds | role |
|---|---|---|---|---|---|---|
| `toy_vits8` | `vit_small_patch8_224` trunk | Imagenette (HF `frgfm/imagenette` 160px, offline cache) | 128 | 150 | 1 (0) | recipe bring-up, dress rehearsal |
| `in100_vits16` | `vit_small_patch16_224` trunk | ImageNet-100 = `~/data/imagenet100` (CMC; D-002) | 224 | 100 | 2 (0,1) | **the controlled grid** |
| `rn18_in100` | ResNet-18 | same IN-100 | 224 | per solo-learn recipe | 1 | port validation ONLY (vs solo-learn published numbers); never enters the audit matrix |
| `in1k_public` | as released | ImageNet-1k `~/data/imagenet` (eval only) | as released | — | — | validation rung 3: "as papers report"; provenance flagged per ckpt |
| `in1k_vits16` | (rung 4, post-M4; scoped later) | — | — | — | — | — |

Frame owns (identical across methods within a frame): backbone topology; dataset+split; epoch
budget (epochs-matched, D-004; pixels/epoch recorded as covariate); checkpoint cadence
`ep{25,50,75,100}` + best + last; seed handling; online-probe monitor (LayerNorm+Linear on the
detached **declared h per §3**, monitor-only; both `_best` selection and the live curve track the
declared h — F1 methods moved GAP→CLS with the h redeclaration, code + fresh vicreg arms 2026-07-14,
D-036); logging schema (per-step grad_norm, per-term losses, collapse monitors);
bf16 autocast; grad_clip default 1.0 (recipe may override only with a DECISIONS row); resume/requeue;
wandb run-id persistence.

Recipe owns (faithful per method, documented in `docs/methods/<m>.md` + module docstring):
augmentations/views/masking; optimizer family, LR/wd/EMA/temperature schedules; batch size + grad
accumulation; head architectures and dims; loss hyperparameters; drop_path. Every deviation from the
paper/donor recipe is listed under "Deviations" in the dossier.

## 3 · Per-method space definitions (D-003v2 + D-008 + D-036)

| method | **h (projector input, D-036)** | z.final ("the proj") | intermediate z taps | notes |
|---|---|---|---|---|
| SimCLR | student trunk-**CLS** (projector input; paper avgpool→ViT, F1 re-resolved D-036) | proj.out (128-d) | proj.tap1 (2048) | trunk-GAP kept as intermediate tap |
| BYOL | student trunk-**CLS** (projector input; paper "we only keep the encoder"; F1 re-resolved D-036) | pred.out (256-d) | proj.tap1, proj.out (256), pred.tap1 | teacher proj.out = target space, stored; trunk-GAP kept as intermediate tap |
| VICReg | student trunk-**CLS** (projector input; F1 re-resolved D-036) | proj.out (8192-d expander) | proj.tap1, proj.tap2 (8192) | largest stored dim; trunk-GAP kept as intermediate tap |
| DINO | teacher **last-layer CLS** (F2 ruling: no concat readouts) | 256-d ℓ2-bottleneck (pre-prototypes) | proj.tap1, proj.tap2 (2048) | prototype logits (65k) never stored — recomputed `l2norm(bottleneck) @ W_proto^T` when needed |
| MAE | student trunk-GAP on uncorrupted images (paper adds a BN inside its linear probe — probe-side, flag F3) | dec.tapK = decoder tokens mean-pooled per block {2,5,8} (512-d) | — | pixel loss space handled metric-by-metric; "z = —" cells = `space missing` |
| I-JEPA | **teacher** last-layer avgpooled patches (F2 ruling: no concat/best-of) | pred.out tokens, pooled (384-d) | — | "we use the target-encoder for evaluation" |
| LeJEPA | **`z.embed` alias `h`** = trunk-CLS→Linear(384→512), the recipe's probed embedding (F4 ruling: linear maps add no capacity — CLS+linear is a valid h; no concat alternative) | proj.out (proj_dim; 16 in toy ckpts) | proj.tap1, proj.tap2 (2048) | both losses on proj.out; single branch |

**Consistency flags — RESOLVED (Berker, 2026-07-02):**
- **F1 (RE-RESOLVED, D-036 2026-07-14)** — ResNet-native methods (SimCLR/BYOL/VICReg): h = the
  **projector input** = trunk-**CLS** on the ViT frame (superseding the original trunk-GAP
  translation of the ResNet avgpool). Basis: in the source papers the probe space (avgpool) WAS the
  projector input; the ViT port broke that coincidence and the original F1 kept the pooling shape
  over the wiring. Empirically the projector input wins on all three (lin/knn: vicreg 64.3/55.1 vs
  59.0/39.4 GAP; simclr 60.6/49.4 vs 56.6/38.2; byol 58.4/45.9 vs 56.8/39.0). Trunk-GAP survives as
  a measured intermediate tap (guillotine overlay), not "h".
- **F2 (resolved)** — **No concatenated readouts, no best-of.** h uses the LAST layer's feature of
  the paper's feature type: DINO → last CLS; I-JEPA → last avgpooled patches (teacher). Paper-exact
  concat variants may appear only as E11 sensitivity arms. (h_layers stays [3,6,9,12] — guillotine
  grid only.)
- **F3 (resolved)** — Probe protocol is OURS and fixed; paper probe quirks (MAE's BN) are E11 arms.
- **F4 (resolved)** — Linear maps cannot add information/capacity, so a CLS→Linear "embedding"
  counts as h when that is what the method probes: LeJEPA h = `z.embed` (512-d). No dual/special-case
  reporting; single h per method.

## 4 · Probes (D-006v2 — USER-APPROVED 2026-07-02; D-020 convergence amendment 2026-07-09)

All probes run on frozen features from the store; fixed hyperparameters; identical across methods
and spaces. Feature-side preprocessing is part of the probe id.

**Convergence rule (D-020):** every trained (linear) probe must run to convergence, not to a fixed
budget — same transform/optimizer/selection as its v1, stopping at patience 120 on best-val (any
improvement resets), hard cap 1000 ep; `best_ep`/`epochs_run` recorded in every CSV (a best_ep
within patience-reach of the cap = censored, flag before quoting). Why: fixed 30 ep boundary-
censors plain Linear on unnormalized GAP features by 4.7–6.8 pts, differentially across methods
(results/diag/probe_conv.csv; patience calibrated there, max observed stale-gap 108 ep). v1 rows
are still computed for continuity.

**Headline pair** (every table, every space): linear separability + lightly-parity kNN.

| id | definition | role |
|---|---|---|
| `linear_raw_v2` | plain Linear on raw (unnormalized) features; AdamW lr 1e-3, wd 1e-7, patience-converged (D-020); best-val | **headline** linear separability |
| `knn_v1` | weighted-cosine kNN, k=200, t=0.1 (+ k=20 reported alongside); optimizer-free, unversioned by D-020 | **headline** ssl_explore/lightly-parity kNN |
| `linear_house_v2` / `linear_l2_v2` | patience-converged (D-020) LayerNorm→Linear / ℓ2→Linear | secondary / E11, converged counterparts |
| `linear_raw_v1` | as raw_v2 at fixed 30 ep (pre-D-020 headline) | continuity column only — known censored at GAP spaces, never headline |
| `linear_house_v1` | LayerNorm → Linear (ssl_explore `meters.offline_probe` exact port), 30 ep | continuity with prior in-house tables; M0 parity anchor |
| `linear_l2_v1` | ℓ2-normalize → Linear, same optimizer, 30 ep | E11 (probe-sensitivity study) |
| `attentive_v1` | 1 learned query, 1 CrossAttn block (6 heads) → Linear; fixed schedule | E11; **token spaces only** — vector spaces report `not-applicable`, never a silent fallback |
| `sololearn_linear` | solo-learn's own linear-eval recipe | `rn18_in100` port validation only |
| *(per-method paper probes)* | e.g. MAE's BN→Linear (flag F3) | E11 arms, added when their method enters |

## 5 · Feature manifests

Fixed, versioned, hashed image lists (`features/manifests/*.csv`; sha256 in every `meta.json`).

| manifest | contents | used by |
|---|---|---|
| `<ds>.m50k.v1` | 50k stratified from canonical train split (toy: full train) | battery, kNN gallery |
| `<ds>.val5k.v1` | 5k stratified val (toy: full val) | probe eval |
| `<ds>.pairs10k.v1` | 10k images × 2 views — extracted under BOTH the method's own aug stack (self-consistency) and the **fixed audit stack** (cross-method comparison; stack pinned in `configs/protocol/audit_v1.yaml`) | alignment, invariance |
| `<ds>.augparam10k.v1` | 10k images × parameterized augs with logged params (crop x/y/scale, jitter magnitudes, rotation) → `params.parquet` | E4 nuisance ledger |

## 6 · Estimator discipline (binding; from report §3 + ssl_explore lessons)

1. **N fixed per comparison** (50k train-manifest features unless stated); never compare
   rank/isotropy statistics across different N.
2. **Dimension sensitivity**: metrics flagged `dim_sensitive` (uniformity, isotropy tests, some
   spectra) are additionally computed on a PCA-k projection, k = min(64, d), fit per space; both
   raw-d and PCA-k rows are reported. Acute at M0: LeJEPA z is 16-d vs h 384-d.
3. **Normalization variants**: metrics are computed on raw and ℓ2-normalized features where the
   home method normalizes; the variant is part of the row key.
4. **Isotropy is never EP alone**: Epps–Pulley sliced statistic is always paired with top-eigvector
   excess kurtosis and worst-direction stats (sliced tests are foolable — WORKFLOW.md).
5. **CKA is contested** (refuted for cross-model correspondence in the report's verification): always
   reported as the triple (linear CKA, neighbor-overlap Jaccard@k, orthogonal-Procrustes distance).
6. **Nulls are mandatory**: every battery table carries (a) the random-init backbone run of the same
   frame, (b) a moment-matched Gaussian synthetic (`nulls.gaussian_match`) for every dim-sensitive
   metric, (c) label-shuffle for probes. Rows report (value, null, delta).
7. **Statistics**: bootstrap CIs over images (default 1000 resamples) for battery metrics; 2 seeds
   at IN-100 for retrain claims; within-experiment-family multiple-comparison correction
   (Holm–Bonferroni) for directional-hypothesis tests.
8. **Provenance separation**: no table mixes controlled-retrain rows with public-checkpoint rows
   (public DINOv2 additionally has no released heads → no z-space claims possible for it).
9. **k̂ defect rank (D-025, battery v2)** — presented as "structured-direction count", never "the
   rank". Declared frame: top-64 variance-ordered PCA directions, each standardized, eigenvalue
   floor λ/λ₁ ≥ 1e-4 (below it stored-fp16 quantization manufactures kurtosis). Report the profile
   k̂(θ), θ ∈ {0.5, 1, 2, 5}, count-above-max(θ, band) semantics (leading-run stays internal).
   Counts are lower bounds under rotation-mixing inside equal-variance blocks
   (`experiments/defect_rank_validate.py`, 10-cluster case). khat_eig (4σ sampling band) is the
   headline count; kurtosis-pursuit is a separate tail-direction-audit cell — raw κ, NO trimming
   or outlier removal (trim/top-share stats are forensic disclosure only), matched-null band at
   reps=20 (reps=6 had a false-positive floor of 3 on the Gaussian control); khat_A (slice-moment
   fit) is a flagged cross-check only: same-sign domain, ~2× bias, cap failures, and its
   detect/bootstrap treat correlated slices as independent — declared limitation at audit n.

## 7 · Standard evals per rung

- Toy/IN-100: in-distribution linear + kNN (per §4) at h and z; the full battery both spaces; guillotine
  curves over `z.*.tapK` + `h.*.L{03,06,09,12}`.
- Transfer/robustness/failure-probe suites (report §5.0: CIFAR, CUB, EuroSAT, IN-9, Waterbirds,
  Stylized-IN, …) enter with E3–E9 — specs live in the E-cards, added to this file when scheduled.

## 8 · Compute placement

Training: H100 2-slot singleton queue (`h100-slotA/B`); 2-view methods may fall back to A40/L40S
(~3× wall, proven by the DINO control). Extraction/battery/probes: `--partition=gpu` arrays.
Storage: D-005 policy; `FeatureStore` enforces the cap.

## 9 · Relative representations (D-009)

Cross-model comparison frame on shared data (Moschella et al. 2023-style), user-directed 2026-07-02.

- **Dataset anchors (primary):** for a space of dimension d, A = d anchor images, drawn as a fixed
  seeded id-list per manifest (`anchors(manifest, A, seed=0)`), SHARED across all models/spaces
  compared. relrep(x) = cosine(x, anchor_features) ∈ R^A. Because anchors are the same *images*,
  relreps of different models live in a common frame → cross-model metrics (CKA-triple,
  neighbor-Jaccard) and the battery can run on the shared space.
- **Random-orthonormal anchors (control):** Q ∈ R^{d×A} orthonormal (seeded); relrep = ℓ2(x)·Q. This
  is a rotation of the normalized feature space — it does NOT align different models; its role is a
  rotation-invariance check on metrics (metrics that claim rotation invariance must be unchanged).
- Reported per comparison: anchor mode, A, seed, manifest. relrep spaces are derived (computed from
  stored features; not separately stored unless reused).
- **Verified vs latentis** (Flegyas/latentis @ 800699f, the relrep authors' library,
  `transform/projection.py`): canonical cosine projection = ℓ2-normalize x and anchors, dot —
  **no centering by default** (`abs_transform=Identity`); Centering/STDScaling/StandardScaling are
  optional pre-transforms. We mirror this (`abs_transform ∈ {none, center, standardize}`, default
  none); non-default choices are part of the reported protocol. latentis also offers
  angular/euclidean/Lp/CoB projections — cosine is our v1.
