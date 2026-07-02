# PROTOCOL.md — the fixed experimental frame

**Version: v1-draft (pre-registration pending user sign-off at M0 exit).**
Any change to this file requires a DECISIONS row. Derived from the report's §5.0 fixed frame and §3
estimator discipline, adapted to this cluster and the binding data ladder (DECISIONS L-004).

## 1 · Spaces and notation

- `h` — backbone representation: trunk `forward_features` output (D-003). For ViT: `h.cls` (CLS
  token) and `h.gap` (patch-token mean), each per-layer (`L03/L06/L09/L12` for ViT-S depth 12);
  default = final layer. 384-d for ViT-S.
- `z` — head/loss space: the method's final head output `z.<role>.out` plus every intermediate head
  layer tap `z.<role>.tapK` (the guillotine axis). Role ∈ {embed, proj, pred, dec}.
- Branches: `student.*` (trained online branch) and `teacher.*` (EMA branch where one exists).
  `probed_branch` per method (§3) follows each paper's own eval convention.
- τ(m, M) = metric m at h ÷ metric m at z, per method M — the transfer ratio (report §5 E1).

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
`ep{25,50,75,100}` + best + last; seed handling; online-probe monitor (LayerNorm+Linear on detached
`h.gap`, monitor-only); logging schema (per-step grad_norm, per-term losses, collapse monitors);
bf16 autocast; grad_clip default 1.0 (recipe may override only with a DECISIONS row); resume/requeue;
wandb run-id persistence.

Recipe owns (faithful per method, documented in `docs/methods/<m>.md` + module docstring):
augmentations/views/masking; optimizer family, LR/wd/EMA/temperature schedules; batch size + grad
accumulation; head architectures and dims; loss hyperparameters; drop_path. Every deviation from the
paper/donor recipe is listed under "Deviations" in the dossier.

## 3 · Per-method space definitions (D-008) and probed branch

| method | z.final | intermediate z taps | probed branch | notes |
|---|---|---|---|---|
| SimCLR | proj.out (128-d) | proj.tap1 (2048) | student | |
| BYOL | pred.out (256-d) | proj.tap1, proj.out (256), pred.tap1 | student (paper: "we only keep the encoder") | teacher proj.out = target space, stored |
| VICReg | proj.out (8192-d expander) | proj.tap1, proj.tap2 (8192) | student | largest stored dim |
| DINO | proj.out = 256-d ℓ2-bottleneck (pre-prototypes) | proj.tap1, proj.tap2 (2048) | **teacher** (paper protocol) | prototype logits (65k) never stored — recomputed `l2norm(bottleneck) @ W_proto^T` from ckpt when a metric needs them |
| MAE | dec.tapK = decoder tokens mean-pooled per block {2,5,8} (512-d) | — | student (encoder on uncorrupted images) | pixel loss space handled metric-by-metric; report's "z = —" cells = `space missing` |
| I-JEPA | pred.out tokens, pooled (384-d) | — | **teacher** (paper: "we use the target-encoder for evaluation") | |
| LeJEPA | proj.out (proj_dim; 16 in toy ckpts) | proj.tap1, proj.tap2 (2048); z.embed = the Linear(384→512) | student (single branch) | official minimal recipe keeps both losses on proj.out |

## 4 · Probes (frozen; D-006)

All probes run on frozen features from the store; fixed hyperparameters; identical across methods
and spaces. Feature-side preprocessing is part of the probe id.

| id | definition | role |
|---|---|---|
| `linear_l2_v1` | ℓ2-normalize → Linear; AdamW lr 1e-3, wd 1e-7, 30 ep, cosine; best-val | **primary** linear probe |
| `linear_house_v1` | LayerNorm → Linear (ssl_explore `meters.offline_probe` exact port) | parity with prior CAMPAIGN_LOG tables only |
| `knn_v1` | weighted-cosine kNN, k=200, t=0.1 (+ k=20 reported alongside) | ssl_explore/lightly-parity kNN |
| `attentive_v1` | 1 learned query, 1 CrossAttn block (6 heads) → Linear; AdamW fixed schedule | **token spaces only**; vector spaces report `not-applicable`, never a silent fallback |
| `sololearn_linear` | solo-learn's own linear-eval recipe | `rn18_in100` port validation only |

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
   excess kurtosis and worst-direction stats (sliced tests are foolable — CLAUDE.md).
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

## 7 · Standard evals per rung

- Toy/IN-100: in-distribution linear + kNN (per §4) at h and z; the full battery both spaces; guillotine
  curves over `z.*.tapK` + `h.*.L{03,06,09,12}`.
- Transfer/robustness/failure-probe suites (report §5.0: CIFAR, CUB, EuroSAT, IN-9, Waterbirds,
  Stylized-IN, …) enter with E3–E9 — specs live in the E-cards, added to this file when scheduled.

## 8 · Compute placement

Training: H100 2-slot singleton queue (`h100-slotA/B`); 2-view methods may fall back to A40/L40S
(~3× wall, proven by the DINO control). Extraction/battery/probes: `--partition=gpu` arrays.
Storage: D-005 policy; `FeatureStore` enforces the cap.
