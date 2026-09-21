# E36 — Does the treated model's extra content live in the part the invariance term deletes?

## Why

E35 closed against the obstacle stated in `E35_open_question_brief.md` §6: the invariance term
*"has no preference that anything can violate — it is minimised by throwing everything away"*, so
every framing of the form "the objective does not want this direction but classification does"
collapsed into circularity. Five designs failed on it (brief §5 a–e).

The obstacle is an artefact of localising to **directions**. It dissolves under a decomposition
into **components**, because of an identity the objective itself supplies:

    (1/V^2) sum_{v,w} ||z_v - z_w||^2  =  (2/V) sum_v ||z_v - z_bar||^2

`inv` **is** the squared norm of the view residual. The term therefore does have a concrete,
violable target: it is driving `R_iv - R_bar_i` to zero. "Does that component carry class
information?" is a question the objective cannot answer vacuously — it is not indifferent to the
residual, it is spending weight 32.8 to destroy it. That is the question E35 could not ask.

The cross-model half comes from Berker (2026-09-10): in a **whitened shared-anchor frame** the null
map between two models is the *identity*, so the comparison needs no fitted `h_U -> h_T` map. That
is exactly what sank brief §5(e), where a least-squares head fit explained only ~55 % of z and the
failure was unattributable between "no map exists" and "the map is not learnable".

Scope stated up front: `inv` lives at **z**, not h. The objective prices only the share of h's view
residual that the projector transmits and is indifferent to the rest — so both are measured.

## Frame (anatomy stated loudly)

- **Cells.** `U` = `in100.floorssl.s0.d256vm4zonly` (`w_cond_h = 0`), `T` =
  `in100.floorssl.s0.d256vm4` (`w_cond_h = 1.894`); `_ep100.pt`, epoch 99. Identical otherwise:
  ViT-S/16 @224, IN-100 (CMC split, `~/data/imagenet100`), 100 epochs, **seed 0 both**, V = 4
  training views, batch 128. `h` = `student.h.cls` = trunk `forward_features` CLS = the projector's
  input (D-003/D-036), post-LayerNorm.
- **Clean store** (`extL`): `in100.train500.v1L` n = 50,000, `in100.val.v1L` n = 5,000,
  deterministic eval transform. **View store**: `in100.pairs1300.v1@audit_v1.o8`, the full train
  split n = 126,689 × **8 views**, stack `audit_v1` = *exactly the augmentation family both lanes
  trained under*. Both stores exist for both cells; labels present in both. fp16 (D-005).
- **The residual frame is not a knob.** Whitening at `alpha = 0`, full numerical rank, is the ONLY
  setting at which the affine null holds. Whitening is exactly equivariant under an invertible
  linear map there (`W_T' A = R W_U'` with `R` orthogonal — proof in `spectra.whiten_frame`'s
  docstring), and both shrinkage and rank truncation break it. Measured on synthetic data
  2026-09-10 before any cell was touched, sham `max|D|` on a cosine scale where the whole signal
  lives in [-1, 1]:

  | setting | sham `max\|D\|` |
  |---|---|
  | alpha = 0, full rank | **.00024** |
  | alpha = 1e-3 | **.85** |
  | alpha = 1e-2 | **.97** |
  | alpha = 0, rank 24/32 | **.45** |

  So `alpha` and rank are fixed at the one admissible corner, and every measured residual is read
  against the **sham arm at the same setting**, never against zero. Sweeping `alpha` remains correct
  for the *probe* (which needs no cross-model cancellation) and is forbidden for the *residual* —
  two instruments, two frames, stated separately wherever a number appears.
- **Anchors.** PROTOCOL §9: A = d = 384 anchor IMAGES, fixed seeded ids (`anchor_indices`, seed 0),
  drawn from the fit split, SHARED across cells. For the view arms an anchor's feature is that
  image's 8-view centroid (the anchor is then not itself an augmentation draw; under the affine null
  the centroid maps by the same `A`, so cancellation is unaffected).

## The decomposition (Berker 2026-09-10)

    R_iv = R_bar_i + (R_iv - R_bar_i),      R_bar_i = (1/V) sum_v R_iv
    D_inv = R_bar_T - R_bar_U               D_var = (R_T - R_bar_T) - (R_U - R_bar_U)

Both components inherit the same parameter-free null: under an invertible-affine recoding
`D_inv = D_var = 0` exactly.

**The intervention, and the correction it needs.** "Centroid versus centroid + view-dependent
component" is arithmetically "centroid versus a single view" — those differ by sqrt(V) of noise
averaging, so that contrast is dominated by denoising rather than content, in both cells. The arms
below instead hold the invariant part FIXED and toggle only the residual's availability:

**The residual's first moment is structurally unreadable — proven, then measured.** For each image
`sum_v (R_iv - R_bar_i) = 0` by construction. At `W = 0` the cross-entropy gradient on a residual
block is therefore `sum_v (p - y) x r_iv = (p - y) x 0 = 0` exactly, so those weights never leave
zero and a linear probe cannot use the residual at all. The 2026-09-10 smoke confirmed it to the
bit: `C_both` read bit-identical to `A_mean`, and residual-only read bit-identical (.0134) across
two *different* models — the class-prior solution. Reading that as "the residual carries no class
information" would have been wrong. The residual's content is **second order**, so the arm that
carries it is the anchor-indexed view-variance

    s_ij = (1/V) sum_v (R_ivj - R_bar_ij)^2

— how much image i's similarity to anchor j wobbles under augmentation. Per-image, anchor-indexed,
in the same shared frame, and identical across cells under the affine null because `R` is.

| arm | feature | rows |
|---|---|---|
| `A_mean` | `R_bar_i` | n images |
| `C_both` | `[R_bar_i , s_i]` | n images |
| `D_res2` | `s_i` alone | n images |
| `B_view` | `R_iv` (single view) | n × 8 |
| `D_res1` | `R_iv - R_bar_i` (first moment) | n × 8 — the PROVEN-null instrument check |

`C_both − A_mean` is the class content of the augmentation-sensitive structure GIVEN the invariant
part, with no denoising confound (the two arms share one row set and one invariant block; "centroid
vs centroid + residual" is arithmetically "centroid vs single view" and would have measured sqrt(V)
denoising instead). Splits are by **image**, never by (image, view) row.

**The increment needs a per-block ridge.** With one shared `lam` the concatenation does NOT nest its
own sub-model — measured, `C_both` .405 against `A_mean` .600, because the centroid block wants
`lam` 1e-4 and the second-moment block 1e-2. `lam` is therefore swept independently per block over a
grid that REACHES the block-off limit, which makes `C_both >= A_mean` a guarantee and the increment
a genuine lower bound (`linear_lbfgs_v1` takes a per-column `lam`).

## What is measured

1. **Reader (the 1.78 itself).** Conditioning-invariant probe on clean features: ridge multinomial
   logistic solved to the global optimum by L-BFGS (`linear_lbfgs_v1`), `alpha` × `lam` swept,
   selection on a 45k/5k inner split cut from train, reported on the untouched 5k val; plus
   shrunk LDA (exactly GL-invariant at shrink 0); plus kNN under the whitened metric; McNemar on
   paired predictions. Against the stored `linear_raw_v2` rows (U .7024 at best_ep **312**, T .7202
   at best_ep **7** — a 45× optimisation-cost gap on the same task).
2. **The residual, read structurally.** `rms(D)`, the 100×100 per-class-pair mean-change matrix
   (0 under the affine null), its diagonal against its off-diagonal, per-anchor rewrite magnitude
   with anchor images named, and `probe(D)` (a CONSTRAINED sub-probe: `D = [I, -I][R_U; R_T]`, so
   it is sufficient but not necessary evidence — a null `probe(D)` refutes nothing).
3. **The decomposition.** The four arms above per cell; `D_inv` and `D_var` and their class-pair
   matrices; the variance split within/between image reported **bias-corrected** (the 8-view
   centroid retains 1/V of the view variance, the residual is shrunk by (1 − 1/V)).
4. **The projector's share.** How much of h's view residual the frozen head transmits, per cell
   (head parity to the stored z: median cos .999992, E35 selftest 1), and whether the class
   information in the residual sits in the transmitted or the killed part.

## Arms, and what is NOT controlled

| arm | role |
|---|---|
| `U`, `T` | the comparison |
| `sham` (`G·h_U`, `G` random invertible, cond 1e4 and 1e1) | the null distribution of every statistic at each (alpha, rank) — NOT a pass/fail check |
| `noise` (`h_U` vs `h_U` + fp16-ulp-sized perturbation) | the store's own floor on the same scale |

**Not controlled, stated plainly:** there is no same-recipe different-seed run — every in100 cell in
the store is `s0`. Two networks from different inits are never affinely related, so **magnitude**
functionals (`rms(D)`, `probe(D)`, `probe(C) − probe(A)` compared ACROSS cells) carry an unknown
seed component and are reported as un-nulled. **Signed, class-aligned** functionals are centred on
zero under the seed null by symmetry (two same-recipe models have the same expected class margin),
so those are the primary reads. A seed-1 untreated retrain (~10 GPU-h) would calibrate the variance
of the signed statistics and make the magnitude ones live; it is not queued at time of writing.

Also uncontrolled by construction: `T` is not an independent draw from `U` — same seed, same data
order — so "treatment-private content" and "seed-private content" are not separable from one pair.

## Scope caveats

- Invariance here is invariance to **`audit_v1` at V = 8**, this lane's training family. It is not a
  unique notion of invariance (Berker 2026-09-10) and is not claimed to be.
- The stored V = 8 exceeds the trained V = 4, so the centroid is a *better* estimate of the
  invariant part than the training estimator's; the bias correction is reported, not assumed away.
- **The headline 1.78 is a CLEAN-feature number and the view store is augmented.** What the
  decomposition splits is the augmented-frame gap. Both are reported side by side and never
  conflated (the standing augmented-in/out-vs-clean caution).

## Pre-registered predictions (committed 2026-09-10, before any number exists)

- **P1 — the reader.** Under the conditioning-invariant reader the clean gap shrinks materially from
  1.78 but stays above 0; the kNN gap (8.42 at raw cosine) collapses much further than the linear
  one. Refuted if the linear gap is unmoved (≥ 1.6) or vanishes (≤ 0.2).
- **P2 — the reader's own selftest.** That reader gives the SAME accuracy on `h_U` and on `G·h_U`
  (cond 1e4). If it does not, it is not conditioning-invariant and nothing else here is readable.
- **P3 — the instrument clears its floor.** `rms(D_signal)` exceeds `max(rms(D_sham), rms(D_noise))`
  by ≥ 10× at the admissible frame. Refuted below 3× — in which case the fp16 store is the binding
  constraint and the cells are re-extracted in fp32 before anything is read.
- **P4 — structure.** The per-class-pair matrix of `D` is diagonal-dominant and positive: the
  treatment's relational rewrite pulls same-class images together relative to different-class ones,
  beyond any affine recoding.
- **P5 — where the gain lives (the card's question).** Writing `gap_mean = a_T(A) − a_U(A)` and
  `gap_full = a_T(C) − a_U(C)`: the majority of `gap_full` sits in the invariant part
  (`gap_mean / gap_full > 0.5`), but the augmentation-sensitive share is materially non-zero
  (> 10 %). Refuted if that share is ≤ 0 or ≥ 50 %. Genuinely two-sided: the treated cell already
  reads MORE view-dependent structure at h (view-spread ÷ image-spread .373 vs .255) while scoring
  1.78 higher, so "the extra content is in the augmentation-sensitive part" is live.
- **P6 — the residual is not label-noise.** `D_res2` (the residual's anchor-indexed second moment,
  alone) reads > 15 % top-1 in both cells against 1 % chance — the component the invariance term is
  driving to zero carries class information in both. `D_res1` must read chance by the proof above;
  if it does not, the instrument is wrong.
- **P7 — the program's thesis, sharp form.** The class information in h's view residual is
  concentrated in the share the projector does NOT transmit. Refuted if the transmitted share
  carries as much or more.

## The clean-residual decomposition (Berker 2026-09-10) — the direct form

The card's own numbers made the second-moment arm a dead end: the VIEW residual sums to zero over
views and is provably unreadable by a linear probe, so its content had to be reached second-order,
and the second moment then added nothing measurable beyond the centroid. Berker's replacement
removes the detour entirely, and is the next move precisely because it is smaller:

    m_i = (1/8) sum_v h_iv        the augmentation centroid
    c_i = h_i^clean - m_i         what the clean view adds on top of it

`c` does NOT sum to zero — it is one specific vector per image — so the FIRST moment is available
and a plain linear probe can read it. Three probes per cell, `m` · `c` · `[m, c]`, and one
comparison, **Acc([m,c]) − Acc(m)**. Since `span{m, c} = span{m, h_clean}`, that increment is
exactly *"after averaging over augmentations, does the clean view contain anything extra?"* Then the
increment is compared BETWEEN cells.

Frame: 50,000 images — the clean store's manifest is a complete subset of the view store's
(verified: 50,000 / 50,000 matched, 500 per class), split 60/20/20 by IMAGE, so the test set is
10,000 images and is NOT the protocol's 5k val (no view store exists for val), which is why these
accuracies are not directly comparable to .7024 / .7202. Two things from this card's own results are
carried in and nothing else: an alpha = 0 whitening frame per block at the block's numerical rank
(the stored AdamW reader moves **4.86 points** under a content-free recoding, so it cannot referee
an increment), and a per-block `lam` (with one shared `lam` the concatenation does not nest its own
sub-model — measured). `h_clean` rides along as a fourth probe for orientation.

**Pre-registered (committed 2026-09-10, before any number exists):**

- **Q1.** `Acc(m) > Acc(h_clean)` in both cells — averaging 8 augmented views beats the single clean
  view. Refuted if the clean view wins in either cell.
- **Q2.** `Acc(c)` alone is well above chance (> 25 % against 1 %) in both cells — the clean view's
  departure from the augmentation average is not noise.
- **Q3.** Both increments `Acc([m,c]) − Acc(m)` are small, under 1.0 point.
- **Q4.** Of Berker's three outcomes, the third: the centroid gap `Acc_T(m) − Acc_U(m)` is the
  LARGER term, and exceeds the difference of increments `inc_T − inc_U`. That is, the treatment
  improved the augmentation-invariant representation rather than living off retained view
  sensitivity. Refuted if `inc_T − inc_U` matches or exceeds the centroid gap.
- **Selftest.** The sham arm reproduces U's four accuracies at alpha = 0 to within a couple of
  images in 10,000; if not, the increment is not conditioning-free and nothing here is readable.

## Selftests (run in-job; no number is read until all pass)

1. **Affine null.** `D_sham = 0` to numerical precision at `alpha = 0`, full numerical rank.
2. **Reader invariance.** `linear_lbfgs_v1` accuracy on `h_U` equals that on `G·h_U` (P2).
3. **Anchor-frame anchor row.** An anchor image's own coordinate reads cos = 1 in both cells, so its
   residual is exactly 0 (384 of 50k rows; recorded, not silently dropped).
4. **Variance split closes.** Within + between = total, bias-corrected, per cell.
5. **Probe-row parity.** The clean `A_mean`-equivalent probe reproduces the stored `linear_raw_v2`
   .7024 / .7202 within probe noise when run at the stored protocol.

## Code

`sslgap/metrics/spectra.py` (`whiten_frame`, `apply_whiten`, `random_linear_map`) ·
`sslgap/metrics/relrep.py` (`relrep_against`) · `sslgap/metrics/pairs.py` (`class_margin`, reused
per D-054) · `sslgap/probes/linear.py` (`linear_lbfgs_v1`, `lda_shrunk_v1`) ·
`experiments/e36_relres.py` + `experiments/configs/e36_relres.yaml` · `slurm/e36_relres.sbatch`.
Output `results/diag/e36_*.csv` and `results/figures/e36/`.

## Numbers — RAW

### Instrument floors (pre-flight, job 65341343) — P3 PASSED

At the admissible frame (alpha = 0, rank 383), rms of the anchor-frame residual on the clean val:

| arm | rms | note |
|---|---|---|
| sham, cond 1e4 | .00003 | the affine null holds |
| sham, cond 1e1 | .00001 | |
| fp16 store noise | .00022 | the store's own floor |
| **signal (U vs T)** | **.05206** | **237× the store floor, 1700× the sham** |

So the fp16 store is NOT the binding constraint and the held fp32 re-extraction (D-122) is not
needed. The frame restriction is confirmed on real data as well as synthetically: at rank 256 the
sham rises to .0327 against a signal of .0585, and at alpha = 1 to .135 against .168. Numerical
rank is exactly 383 in both cells (untreated lam_384 = 5.2e-10, treated 2.7e-8; lam_383 = 7.3e-6
and 5.8e-3).

### kNN under a conditioning-free metric (job 65346294, Berker-requested 2026-09-10)

Cosine kNN on alpha = 0 whitened features is EXACTLY invariant to an invertible linear map of h.
Gallery = train500 50k, queries = val 5k, the stored protocol's readers. `raw` reproduces the
stored `knn_v1_k20` / `knn_v1_k200` rows exactly (.6126/.5784 and .6696/.6626).

| metric | k | untreated | treated | gap | McNemar p | sham deviation |
|---|---|---|---|---|---|---|
| raw cosine (stored) | 200 | .5784 | .6626 | **+8.42** | 1.1e-50 | **4.02** |
| centered | 200 | .6060 | .6628 | +5.68 | 6.2e-30 | 4.78 |
| **whitened, alpha = 0** | 200 | **.6654** | **.6804** | **+1.50** | .0010 | **.00** |
| raw cosine (stored) | 20 | .6126 | .6696 | +5.70 | 4.9e-25 | 5.20 |
| **whitened, alpha = 0** | 20 | **.6506** | **.6752** | **+2.46** | 1.1e-6 | **.02** |

Whitening lifts the UNTREATED cell by +8.70 points at k=200 (.5784 -> .6654) and the treated by
+1.78 (.6626 -> .6804) — a ~5× asymmetry. The sham column is the calibration that makes the
collapse readable: on the raw metric a CONTENT-FREE recoding of the untreated cell moves the reader
by 4.0–5.2 points, i.e. over half the size of the 8.42 gap it is being asked to explain. At
alpha = 0 the same recoding moves it by at most 1 query in 5,000. alpha = 0 is chosen by the
invariance argument and verified by the sham, never by performance, so no selection rides on it.

### Reader — the 1.78 under readers conditioning cannot fool (job 65344209)

Selection on a 45k/5k inner split cut from train; reported on the untouched 5k val.

| reader | untreated | treated | gap |
|---|---|---|---|
| stored `linear_raw_v2` (parity check) | **.7024** @ep312 | **.7202** @ep7 | +1.78 |
| the SAME stored reader on the SHAM recoding of U | **.6538** @ep341 | — | a content-free recoding costs it **4.86** |
| `linear_lbfgs_v1`, alpha = 0 (exactly GL-invariant) | .6932 | .7096 | **+1.64** |
| `linear_lbfgs_v1`, each cell's best alpha | .6994 (a=.1) | .7172 (a=1) | **+1.78** |
| `lda_shrunk_v1`, shrink .01 | .6194 | .6362 | +1.68 |

Matched-alpha gaps across the whole grid: +1.64 / +1.64 / +1.58 / +1.32 / +1.54 / +2.16. McNemar at
each cell's best: treated fixes 331, breaks 242, **p = 2.4e-4**. Selftest P2 PASSED: U against its
own sham recoding at alpha = 0 differs on **1 query in 5,000**. Selftest 5 (probe-row parity)
PASSED: both stored rows reproduced exactly, best_ep included. Rider on the LDA row — exact
GL-invariance holds only at shrink 0, and at shrink .01 the sham still deviates 2.4 points, so LDA
is a weaker instrument here than the alpha = 0 linear reader and is reported as such.

### Residual, clean features (job 65344209)

| arm | rms | class-pair diag | off-diag | SIGNED | probe(D) |
|---|---|---|---|---|---|
| signal (U vs T) | .05259 | **+.02339** | −.00034 | **+.02373** | **.5784** |
| sham | .00003 | −.0000000 | −.0000000 | +.0000000 | .0194 |

### The decomposition (job 65344209, n = 60,000; fit/sel/test = 35,998 / 11,998 / 12,004 by IMAGE)

| arm | untreated | treated | gap |
|---|---|---|---|
| `A_mean` (view-invariant centroid) | .7323 | .7447 | **+1.24** |
| `C_both` (centroid + residual 2nd moment) | .7312 | .7451 | **+1.39** |
| `D_res2` (residual 2nd moment ALONE) | **.4920** | **.5492** | **+5.72** |
| `B_view` (one augmented view) | .6448 | .6738 | +2.90 |
| `D_res1` (residual 1st moment) | .0110 | .0110 | 0 — the proof, bit-identical |

Within-cell increment from the second moment: **−0.11 (U) and +0.04 (T)**, both inside the ±0.4
point sampling band on 12,004 test images — the second moment adds nothing MEASURABLE beyond the
centroid in either cell, and the small negative shows the block-off limit is only approximately
reached numerically (a real nesting model cannot lose). Variance split in the conditioning-free
frame: within/between **.6498 (U) vs .4575 (T)** — the treated cell is MORE view-invariant here,
the OPPOSITE ordering to the raw-h numbers in the brief (.255 vs .373), so which cell is "more
augmentation-invariant" flips with the metric. Class margin: centroid +.1854 / +.2033; residual
rows −.0006 / −.0003 (as the proof requires); residual second moment **+.1227 / +.1610**.
`D_inv` class-pair SIGNED **+.02737** against a sham of −.00000.

### The clean-residual decomposition — RAW (job 65351670, n = 50,000, fit/sel/test = 30k/10k/10k by image)

| probe | untreated | treated | gap |
|---|---|---|---|
| `m` — the augmentation centroid | **.7333** | **.7506** | **+1.73** |
| `c` — clean minus centroid | .3794 | .4194 | +4.00 |
| `[m, c]` | .7347 | .7512 | +1.65 |
| `h_clean` (orientation) | .7127 | .7355 | +2.28 |

| | increment `Acc([m,c]) − Acc(m)` | McNemar |
|---|---|---|
| untreated | **+0.14** | fixes 167, breaks 153, **p = .47** |
| treated | **+0.06** | fixes 159, breaks 153, **p = .78** |
| sham (U recoded, cond 1e4) | +0.13 | 167 / 154, p = .50 |

**centroid gap `Acc_T(m) − Acc_U(m)` = +1.73 · increment gap `inc_T − inc_U` = −0.08**

Selftest |sham − U| out of 10,000: `m` **0**, `c` **1**, `mc` **1**, `h_clean` **0**. All four pass —
every probe here is invariant to an invertible linear recoding of h, so no number in this table is
attributable to conditioning.

SUPERSEDED RUN, kept as the reason the instrument is written the way it is: the first pass
(job 65350701) set each block's rank from a RELATIVE threshold re-derived per arm. The sham is
recoded at condition number 1e4, which spreads the spectrum by 1e8 in variance and pushed dozens of
directions under that cut, so the sham was probed in 324 dimensions against the untreated cell's 384
and its `c` selftest read 1.66 points — a measurement of that rank mismatch, not of the reader
(truncation is not equivariant). The U/T comparison was never affected (both at 383/384) and its
numbers are unchanged to 0.01 points; the rank is now fixed once on the untreated cell and reused
for every arm.

Rider on split discipline: on the SELECTION split the untreated increment read **+0.64** and on the
untouched test split **+0.14**. Selection was off the reported split by design; had it not been,
this card would have reported a four-fold larger increment.

### Theta decomposed in raw h (job 65357049) — Berker's correction, 2026-09-10

Berker: *"backbone invariance is also a function of image spread. it increases Theta defined as
spread across views over spread across images."* Correct, and it retires the claim made earlier in
this session that the h-term "never sees invariance": the term acts on the between-image
distribution of per-image view MEANS, which is Theta's DENOMINATOR, so it necessarily moves Theta.
Which term moved is the separate question, and it is measurable:

| cell | W (view) | B (image, debiased) | **Theta = W/B** | tr Cov(m) | effrank Cov(m) |
|---|---|---|---|---|---|
| untreated | .3599 | 1.6854 | **.2135** | 1.7304 | 50.5 |
| treated | 70.4263 | 222.1801 | **.3170** | 230.988 | 191.2 |
| **T/U** | **x195.7** | **x131.8** | **x1.485** | **x133.5** | **x3.8** |
| sham (U recoded, cond 1e4) | 197.96 | 927.99 | **.2133** | — | 26.5 |

Theta rose x1.485, and it rose through the NUMERATOR outrunning the denominator, not through image
spread falling — B rose x131.8. The dominant change is an order-of-magnitude inflation AND
isotropization of the view-mean distribution (trace 1.73 -> 231, effective rank 50.5 -> 191.2),
which is exactly the quantity `h_floor_batch = view_mean` optimizes toward N(0, I): at 384-d the
unit-variance target trace is ~384 and the untreated cell sits at 1.73.

Rider against this card's own earlier caution: Theta turned out to be essentially UNMOVED by a
content-free recoding (.2135 -> .2133, one draw of G at cond 1e4), so the warning that "Theta is a
raw-metric quantity and cannot referee anything" was not borne out here — W's and B's covariances
are near-proportional in the untreated cell. It is not invariant by construction, only empirically
under this draw.

| | outcome |
|---|---|
| P1 linear half | **REFUTED** — the linear gap does NOT shrink (1.78 -> 1.64 at alpha = 0, +1.78 at best alpha, p = 2.4e-4) |
| P1 kNN half | **CONFIRMED** — 8.42 -> 1.50 at k=200 |
| P2 reader selftest | **PASSED** — 1 query in 5,000 |
| P3 instrument floor | **PASSED** — 237× the fp16 store floor |
| P4 class-pair structure | **CONFIRMED** — diagonal-dominant, positive, sham at zero |
| P5 where the gain lives | **PARTIAL** — `gap_mean` carries 89 % of `gap_full` (predicted > 50 %), but the augmentation-sensitive share rests on within-cell increments of −0.11 / +0.04 that are inside noise, so the "> 10 % materially non-zero" clause is NOT supported |
| P6 residual is not label-noise | **CONFIRMED** — 49.2 % / 54.9 % against 1 % chance |
| P7 projector's transmitted share | **NOT MEASURED** — the stage was specified in §4 and not implemented; owed |
| Q1 `Acc(m) > Acc(h_clean)` | **CONFIRMED** — .7334 > .7127 and .7506 > .7354; averaging 8 augmented views beats the single clean view by ~2 points in both cells |
| Q2 `Acc(c)` above 25 % | **CONFIRMED** — .3794 / .4194 against 1 % chance, selftest-clean (1 image in 10,000) |
| Q3 both increments under 1.0 pt | **CONFIRMED** — +0.14 and +0.06, both inside noise (p = .47, .78) |
| Q4 the centroid is the larger term | **CONFIRMED** — centroid gap +1.73 against an increment gap of −0.08. Berker's THIRD outcome: the conditioner improved the augmentation-invariant representation; it does not live off retained view sensitivity |

## AGREED TAKEAWAY

*(empty by contract — filled only after the joint read with Berker)*
