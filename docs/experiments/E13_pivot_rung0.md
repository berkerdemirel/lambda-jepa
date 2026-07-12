# E13 — PIVOT rung-0: does predictive-state distortion rank our zoo?

**Status: PRE-REGISTERED (drafted in-conversation 2026-07-12; Berker sign-off same day: "this is a
perfect construction! i say yes to the dino-ctrl eps, b is also good no veto, ridge lambda is also
good. lets go!"). Predictions locked BEFORE any number exists. Zero training — pure analysis over
stored `in100.pairs100.v1@audit_v1` features. Ledger row: D-029 (also records the D2 staged-merge
verdict on the PIVOT proposal, `docs/PIVOT_Standalone_Research_Proposal.pdf`).**

## Question

PIVOT's load-bearing empirical premise (proposal predictions 7/19; experiment E1 "frozen checkpoint
zoo") is that *held-out predictive distortion of a frozen context descriptor ranks representation
quality better than training losses and marginal/battery statistics do*. We test this at zero
training cost on a provenance-matched zoo — a harder-to-fake version of PIVOT's own E1 (their
public zoo confounds architecture/data; ours holds both fixed). The low bar is E01-T15: at h, NO
battery metric currently tracks probe ordering (all |ρ| ≤ .39, small-zoo). The refinement we add
and pre-register: PIVOT's "distortion" conflates an information meter and a deployed-geometry
meter, which our E12 instruments already separate — so we measure both.

## Construction (all offline; stored fp16 arrays only)

- **Event/context proxy:** event = the checkpoint's own viewA h; context descriptor = frozen
  `in100.mae.s0` `student.h.gap` on viewB. Declared: views are independent draws of the audit_v1
  augmentation stack, not declared foveal channels; overlap uncontrolled; T=MAE is a same-data
  reconstruction-trained proxy tokenizer, unaudited per PIVOT-E0. View draws were not pinned
  across extractions, so the target array is byte-identical for every zoo member (irreducible
  view noise cancels in ranking) and mae must sit out of the ranked set (its own viewA/viewB are
  a same-pass pair — intimacy advantage).
- **Sketch:** RFF φ(y) = √(2/D_m)·cos(Wy+b), W ~ N(0, σ⁻²I), b ~ U[0,2π), fixed per-cell seed
  (recorded), shared across the zoo. Descriptor standardized with train-fold stats before the
  sketch (PIVOT §5.2 step 2); sketch outputs left raw (they realize the kernel).
- **Two distortions per checkpoint** (both symmetrized: A→B and B→A averaged):
  - **D_read** = held-out 1−R² of ridge h(viewA)→φ(T(viewB)); per-dim standardized inputs; ridge
    α selected on a val fold per checkpoint from the declared grid (effective dof recorded;
    dof-matched variant at dof*=256 reported as control). ~Affine-invariant ⇒ *information* meter.
  - **D_kern** = 1 − Spearman between the off-diagonal cosine Gram of h(viewA) (test fold,
    centered by train-fold mean, rows L2-normalized) and the target sketch Gram Φ_te·Φ_teᵀ.
    Linear CKA reported alongside (secondary). Metrization-*sensitive* ⇒ *deployed-geometry*
    meter — the thing PIVOT actually ships.
- **Splits:** 6000/2000/2000 stratified by class, seed 0, identical image indices zoo-wide
  (runtime assert: labels arrays identical across runs).
- **Sweep (E0-style):** D_m ∈ {64, 256, 1024, 4096} × σ ∈ {¼, ½, 1, 2, 4}·σ_med (median pairwise
  distance of 2048 standardized train-fold descriptors, per tokenizer×direction, recorded), plus a
  raw-descriptor ridge arm (no RFF — PIVOT prediction 10). **Primary cell: D_m=1024, σ=1·σ_med,
  T=mae.** Everything else descriptive.
- **Tokenizer-independence control:** full repeat with T = `in100.randinit-s0` `student.h.gap`
  (same arch, same d).
- **Target audit before any ranking claim** (PIVOT-E0's reject rule, adapted): sketch spectrum /
  effective rank (2048-subsample SVD), kernel concentration (off-diag mean/sd/quantiles;
  near-constant and near-orthogonal gates), across-zoo dynamic range vs per-checkpoint bootstrap
  noise (primary cell), and class structure of the target itself — B/T every cell, kmeans-100 NMI
  on top-128 PCs at the primary + raw cells — quantifying the "tokenizer too semantic ⇒ this is
  distillation" bound. R4c-lens read of what the sketch responds to = analysis, not code.
- **Ridge details (declared):** α ∈ {1e-5,1e-4,1e-3,1e-2,1e-1,1,10}·n_train; selection by val
  1−R²; edge-of-grid selections flagged. dof(α) = Σ s²/(s²+α) on the standardized train design.
- **Baseline set, locked here (no post-hoc additions):** every (metric|variant) row of the
  standing battery CSVs at (train500, the run's declared h) present for ≥15 ranked members, plus
  `moment_kl_sliced` and `diag_kl` (floor-values estimator, D-026 settings: d′=128, ε=1e-4, 32
  draws, n=4096) computed at each run's declared h on train500. Each baseline's ρ is paired with
  the PIVOT-distortion ρ recomputed on the same member subset. Monitor-best (labeled online probe)
  is the ceiling reference for the lejepa sub-zoo — discussed, not a competitor (comparison class
  = label-free rankers). Cross-family training losses are non-comparable and declared out; the
  loss-functional-at-h columns cover the lejepa sub-zoo.
- **Ranking targets:** converged `linear_raw_v2` and `knn_v1_k200` at each run's declared h
  (D-020 probes, `results/probes/<run_id>.csv`). Reported on: full ranked zoo (n=20), lejepa
  sub-zoo (n=11: 10 e12 arms + lane), non-lejepa rest (n=9, descriptive — underpowered alone).

## Ranked zoo (n=20) + gate members

| group | run_ids | declared h |
|---|---|---|
| core-6 | `in100.{dino,simclr,byol,vicreg,ijepa,lejepa}.s0.ext` | dino `teacher.h.cls` · ijepa `teacher.h.gap` · simclr/byol/vicreg `student.h.gap` · lejepa `student.z.embed` |
| anchor | `in100.deitlite.s0.ext` | `student.h.cls` (D-022) |
| dino-ctrl | `in100.dino-ctrl.ep{25,50,100}.ext` | `teacher.h.cls` |
| e12 arms | `in100.lejepa.s0.e12{a1,a2,a3,c1,f1..f6}.ext` | `student.z.embed` |
| GATE (unranked) | `in100.randinit-s0.ext` (`student.h.cls` input; leakage gate) · `in100.mae.s0.ext` (tokenizer; D_read reported under T=randinit only) | — |

## Pre-registered predictions (locked 2026-07-12, before numbers)

- **E13-P1 (leakage/sanity gate).** Every trained checkpoint beats randinit-s0 on D_read by a
  margin large vs the trained spread. Randinit mid-pack ⇒ the target is low-level-leakage-
  dominated ⇒ stop at the audit, no ranking claims.
- **E13-P2 (PIVOT's claim, primary).** At the primary cell, Spearman(D_read, linear_raw_v2) over
  the ranked zoo ≤ −0.44 (permutation p < .05) **and** exceeds the best recomputed battery
  baseline in |ρ|.
- **E13-P3 (the dissociation quartet — sharpest falsifiable structure).** D_read is
  ~affine-invariant, so the whitened arms **A3/f6 read near-C1 on D_read despite their kNN
  wreckage** (E12-T1's calibration was invertible: effrank 275, no dead dims), and **f4 reads
  acceptable on D_read** (its linear information survives in low-variance directions that
  standardization rescales) — while **D_kern flags all three as damaged**. Zoo-level corollary:
  D_read ranks linear better than it ranks knn200; D_kern ranks knn200 better than D_read does.
  If this holds, "PIVOT distortion" splits into an information meter and a metrization meter — a
  refinement the proposal does not contain and exactly our two-space story at h.
- **E13-P4 (tokenizer content).** T=mae outranks T=randinit as a ranker (Δ|ρ| ≥ .15 at the
  primary cell). Parity ⇒ the "trained predictive state" carries no content at this rung —
  conditioning-artifact reading opens, PIVOT reading blocked.
- **E13-P5 (sweep shape).** Interior optimum in σ; D_m saturating by 1024; RFF ≥ raw-descriptor
  arm as a ranker (directional lean only — PIVOT prediction 10, recorded not gated).

## Kill criteria

- **E13-K1.** Target-audit failure at the primary cell (concentration gates; across-zoo dynamic
  range < 3× median bootstrap se; or P1 fail) ⇒ target-construction failure; no ranking language
  anywhere; report as such.
- **E13-K2 (rung-0 kill for PIVOT's premise).** NO cell in the whole T=mae sweep reaches
  |ρ| ≥ 0.44 (perm p < .05) on either probe column for either distortion ⇒ predictions 7/19 fail
  their cheapest matched-provenance test ⇒ first negative datum toward the proposal's central
  refutation (its prediction 17), scoped by the declared proxies (views ≠ foveal channels; T=MAE
  unaudited; overlap uncontrolled; n=1 seed).
- **E13-K3 (forking-paths guard).** Isolated significant cells with no contiguous (D_m, σ)
  neighborhood and a failed primary cell count as noise, not signal.

## Deviations at implementation (vs the signed in-conversation draft; bookkeeping only)

1. **parity30k excluded from the ranked zoo** — its probe CSVs carry only kNN rows (no converged
   `linear_raw_v2`), so it cannot serve both ranking targets. Ranked n = 20, not ~22.
2. **randinit-s0 is a gate member, not ranked** — keeps zoo membership identical under both
   tokenizers (P4's Δ|ρ| is then apples-to-apples). A secondary ρ with randinit included is
   reported for T=mae (it can only inflate ρ; the tokenizer-neutral n=20 is the honest primary).
3. Descriptor standardized before RFF / sketch left raw (precision of "targets standardized";
   proposal-faithful).
4. Monitor ceiling handled in discussion, not in the rank-corr CSV.
5. Ridge includes an intercept (targets centered by the train-fold mean, added back at prediction).
   Caught by the dry-run audits (job 62238021): uncentered RFF targets carry a large common
   mean-embedding component at σ_med (kernel off-diag mean ≈ .59), the zero-mean standardized
   design cannot represent it, and 1−R² read >1 zoo-wide; the raw-descriptor arm was immune
   (train-standardized ⇒ ~zero-mean), which localized the bug. Orderings were sane pre-fix
   (randinit worst on every meter); the fix removes a shared offset, it does not create signal.

## Artifacts & compute

`experiments/e13_pivot_rung0.py` (pure numpy/scipy/sklearn over stored arrays; SVD cached per
checkpoint×direction and reused across cells) → `slurm/e13.sbatch` (defaultp, CPU). Outputs:
`results/diag/e13_distortion.csv` (per run × tokenizer × direction × cell), `e13_target_audit.csv`
(per tokenizer × direction × cell), `e13_rank_corr.csv` (ranker × probe-target × zoo-subset × cell
with permutation p, 20k perms), `e13_gate.csv` (P1/K1 numbers incl. bootstrap se at the primary
cell). Figures land in `results/figures/e13/` after numbers (spearman_bars, scatter_read_lin,
scatter_kern_knn, cell_heatmap, target_audit). Order: CPU dry-run (2 checkpoints × primary+raw
cells, audits printed) → full job → numbers land raw below → discussion. Single-seed /
single-view-draw caveats stand. No PROTOCOL change (E13-local estimator; admitting it to the
standing battery would be its own DECISIONS row).

## Numbers land below this line as they arrive; AGREED TAKEAWAY only after joint discussion.

### Full sweep — 2026-07-12 (job 62238065, 15 min; dry-runs 62238021/62238054; figs 62241928)

**Sources:** `results/diag/e13_{distortion,target_audit,rank_corr,gate}.csv` ·
`results/figures/e13/e13_{spearman_bars,scatter,cell_heatmap,target_audit}.png`. All Spearman ρ
below: ranked zoo n=20, permutation p (20k), distortions direction-symmetrized. Estimator health:
0/40 α-edge selections at the primary cell; dof range 70–213.

**Gates (P1/K1) — PASSED.** randinit D_read .9785 vs trained mean .8638 ± .0181 (margin +6.35 sd;
0/20 trained runs worse); dynamic range = 6.3× median bootstrap se (bar: 3×). Primary-cell target
audit interior and healthy: kernel off-diag mean .60 ± .12, sketch effrank 31, target B/T .081,
target kmeans100-NMI .242 (tokenizer weakly class-aligned — far below distillation grade; dino h
reads .53 on the same instrument). Sweep gates behave as designed: σ/4 → near-orthogonal (off-diag
mean .003; those cells never clear), σ×4 → near-constant (mean 1.007 ± .013).

**Primary cell (T=mae, D_m=1024, σ=σ_med), full20:**

| ranker | ×linear_raw_v2 | ×knn_v1_k200 |
|---|---|---|
| D_read (val-selected α) | **−.538 (p=.017)** | −.322 (p=.166) |
| D_read dof-matched (declared control) | **−.635 (p=.003)** | −.444 (p=.051) |
| D_kern (Gram-Spearman) | −.081 (p=.73) | −.146 (p=.53) |
| −CKA | −.146 (p=.54) | −.215 (p=.36) |
| best battery baseline: kurt_topeig.mean\|raw\|full | −.553 (p=.013) | −.450 (p=.047) |
| raw-descriptor arm (no RFF), D_read / dof | −.475 / −.584 | −.277 / −.388 |
| full20+randinit (secondary), D_read | −.601 (p=.006) | — |

**Sweep (K2/K3):** 36 (ranker,cell,probe) rows clear |ρ|≥.44 & p<.05 for D_read/dof — a
contiguous region covering EVERY D_m ∈ {64…4096} at σ ∈ {½,1,2,4} plus the raw arm, on linear;
dof-matched also clears knn at several cells (best −.531 @64/σ½). Best single cells: D_read −.562
@ (64, σ_med); dof-matched −.666 (p=.0018) @ (64, σ_med). Ranking power is FLAT in D_m from 64 up
(nothing gained past 64); σ=¼ column dead (near-orthogonal target). K2 NOT fired; K3 satisfied.

**Family decomposition at the primary cell (the load-bearing raw fact):**

| ranker | lejepa11 lin / knn | nonlejepa9 lin / knn |
|---|---|---|
| D_read dof-matched | **−.700 (p=.018) / −.664 (p=.028)** | −.483 (p=.19) / −.167 (p=.67) |
| D_read | −.554 (p=.077) / −.409 | −.417 (p=.27) / −.100 |
| kurt_topeig.mean (battery best) | −.282 (p=.41) / **+.018 (p=.95)** | **−.717 (p=.037) / −.750 (p=.027)** |

The two top rankers draw their full-zoo ρ from DISJOINT parts of the zoo: kurt_topeig ranks the
heterogeneous method families and is DEAD within the lejepa/e12 family (where the floor arms
directly manipulated kurtosis); the PIVOT meter is strongest exactly there (within-family,
within-intervention) and weaker across families. Context: E01-T15's small-zoo null (no h battery
metric |ρ|>.39) does not replicate on this 20-member zoo — kurt_topeig now ranks at full20; note
55% of this zoo is one intervention family, so the full20 battery number is partly our own
intervention's signature.

**Per-run D_read / D_kern at the primary cell (sym; sorted by D_read; probes on the card of
origin):** c1 .847/.862 · ctrl-ep100 .849/.901 · f5 .850/.971 · byol .851/.859 · lane .852/.861 ·
simclr .853/.852 · ctrl-ep50 .853/.884 · f4 .854/.922 · f3 .855/**.998** · dino .855/.889 · f2
.855/.956 · vicreg .855/.869 · a1 .857/.903 · ctrl-ep25 .863/.867 · f1 .869/.973 · a2 .876/.964 ·
ijepa .886/.948 · f6 .890/.976 · a3 .890/.978 · deitlite **.917**/.948 · [gate] randinit
.979/.977. Raw observations: dino-ctrl epochs order monotonically with training on D_read
(ep25 .863 > ep50 .853 > ep100 .849) — the checkpoint-selection use case, within-run; deitlite
(supervised CE) is the WORST trained run on D_read while carrying high probes; D_kern orders by
floor dose within the e12 family (c1 .862 < f2 .956 < f1 .973 < a3 .978, f3 .998 worst-in-zoo)
and puts a3/f6 at randinit level.

**Mechanical outcomes vs the locked predictions:**
- **P1 HELD** (margin 6.35 sd, 0 inversions; dynamic range 6.3×).
- **P2 SPLIT — as locked, NOT MET.** Clause A held (D_read×lin −.538, p=.017 ≤ −.44); clause B
  missed by .015 (battery best −.553). The DECLARED dof-matched control clears both clauses
  (−.635, p=.003) — but it was registered as control, not primary; recorded accordingly.
- **P3 SPLIT.** D_kern flags the quartet exactly as predicted (a3/f6 at randinit level .976–.978;
  f4 .922 vs c1 .862) and f4 reads near-C1 on D_read (.854 vs .847) — standardization rescue
  confirmed. But a3/f6 do NOT read near-C1 on D_read (.890 = +2.4 trained-sd): the
  affine-forgiveness sub-prediction FAILED — D_read agrees with the converged linear probe that
  the whitened arms lost real information, not only metrization. Corollary half-held: D_read
  ranks linear ≫ knn (−.54 vs −.32) ✓; D_kern ranks nothing at zoo level (best −.19) ✗ — it is a
  degeneracy/re-metrization FLAG, not a ranker, at this construction.
- **P4 HELD.** T=randinit tokenizer: D_read×lin −.287 (p=.22) vs T=mae −.538 → Δ|ρ| = .25 ≥ .15.
  Trained-descriptor content is real; not a conditioning artifact.
- **P5 SPLIT.** σ has an interior optimum (¼ dead by orthogonality; ×4 degrades but survives via
  gate-flagged near-constant targets) ✓; "saturates by D_m=1024" technically true but the real
  shape is FLAT FROM 64 — small sketches already carry all ranking power ✗(as envisioned); RFF ≥
  raw held narrowly (−.562 best-RFF vs −.475 raw; dof-matched −.666 vs −.584) — prediction-10
  lean, recorded.
- **K1 not fired · K2 not fired · K3 satisfied** (contiguous clearing region incl. the primary).

### Amendment — supervised-anchor exclusion re-read (Berker-directed 2026-07-12, post-numbers;
### REGISTERED-LATE — decided after seeing deitlite as the against-trend outlier)

**Rationale (Berker):** deitlite's probe column is mechanism-linked — supervised CE trains exactly
the linear class separability the probe measures while discarding context-irrelevant information
(minimality w.r.t. labels), an advantage/handicap pair no SSL member has; on a dense/multi-task
probe suite it would invert hard while the SSL ordering barely moved. Linear probe acc is a fair
ranking target only across members with no supervision advantage. deitlite therefore demotes to an
unranked anchor (shown gray in figures). The pre-registered n=20 outcomes above remain the locked
record; this is the amended read. CSV: `results/diag/e13_rank_corr_n19.csv`; figures:
`results/figures/e13/e13_{spearman_bars,scatter,cell_heatmap}_n19.png` (job 62242290).

**full19 primary cell:**

| ranker | ×linear_raw_v2 | ×knn_v1_k200 |
|---|---|---|
| D_read | **−.616 (p=.007)** | **−.511 (p=.028)** |
| D_read dof-matched | **−.716 (p=.001)** | **−.653 (p=.004)** |
| best battery: kurt_topeig.mean | −.546 (p=.016) | −.426 (p=.068) |
| D_kern | −.114 (n.s.) | −.200 (n.s.) |

On the amended zoo **both D_read variants beat every battery baseline on BOTH probe columns**, and
D_read is now significant on kNN as well; the clearing region grows 36→62 rows. nonlejepa8
d_read×lin jumps −.417→−.619 (deitlite was the main cross-family violator; n=8, p=.12,
underpowered). P4 re-check holds under exclusion (T=randinit −.356 vs T=mae −.616; Δ|ρ|=.26).
D_kern conclusion unchanged (flag, not ranker).

## AGREED TAKEAWAY

*(Substance agreed in the 2026-07-12 discussion — Berker: "i think takeaways upon our discussion
is clear"; exact wording delegated, veto open. Full row texts in DECISIONS E13-T1…T3.)*

- **E13-T1 — the information half of PIVOT survives rung-0.** Held-out predictive-state
  distortion at h (ridge from viewA h onto the RFF sketch of the frozen MAE descriptor of viewB)
  RANKS the SSL zoo: on the supervised-anchor-free read (n=19) D_read −.62 lin / −.51 knn,
  dof-matched −.72 / −.65 — beating every h-side battery statistic on both probe columns — and it
  ranks WITHIN the e12 intervention family (−.70/−.66) where every marginal statistic is blind
  (kurt_topeig +.02). Gates clean; T=randinit control collapses the ranking (−.36) → the content
  is the trained descriptor. Supported form: "conditional-context information, linearly decodable
  at fixed budget, ranks representations." The locked n=20 P2 verdict (NOT MET by .015) stands as
  the pre-registered record; the amended read is operative per T3(a). Scope: aug-view proxy
  channels, one trained tokenizer, seed 0, IN-100 probes.
- **E13-T2 — the deployment half fails at rung-0.** D_kern (raw-Gram alignment to the predictive
  kernel — Corollary 4.2 / the E1 selection metric) has no ranking power anywhere in the sweep
  (best |ρ| .19) while ordering the e12 arms by regularization dose up to randinit level (f2, the
  best-probing h in the project, deep in the flagged tail at .956). Agreed reading: raw kernel
  distortion as an unlabeled selection metric on non-PIVOT models is near category error — Gram
  equality is rotation-strict, the target metric is reconstruction-flavored, and cosine is
  variance-weighted so re-metrization dominates it (the E12-T5 mechanism at the metric level:
  predictive information present up to a linear map but differently PLACED in the variance
  profile). This kills the advertised E1 use, not Theorem 1; D_kern is retained as a
  re-metrization flag and becomes meaningful as a training-time diagnostic only for PIVOT-trained
  models (the loss pins the rotation).
- **E13-T3 — instrument lessons.** (a) Linear-probe accuracy is a fair zoo-ranking target only
  among members with no supervision advantage: the supervised anchor's probe↔training-mechanism
  coupling (CE optimizes the probe's own quantity while discarding context information)
  contaminates the ranking from both sides — excluded, registered-late, Berker-directed.
  (b) Capacity-controlled readout (dof-matched) strictly dominates val-selected α (−.72 vs −.62):
  readout-capacity variation is noise; promoted to primary for any follow-up (vindicates
  PIVOT-E4's fixed-dof spec). (c) Ranking power is flat in D_m from 64 and the raw-descriptor arm
  is nearly as good — the kernel/distributional target machinery adds little at this rung
  (prediction 10 weak here); the target's predictable content is low-dimensional (effrank ~31).
