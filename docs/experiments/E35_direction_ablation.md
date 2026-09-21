# E35 — Which directions of h does the objective price?

## Why

Berker 2026-09-09: *"we will compute pca of H on a heldout set. and let the eigenvectors be
ordered from high to low. then we remove the low variance directions. we will test (i) how
projected loss reacts dropping low eig directions, (ii) how linear probe performance would be
with the dropped directions. expected outcome: (i) ssl augmentation invariance do not change
that much (ii) downstream performance drops. … I would add one control: remove the same number
of random directions. That tells you whether the specifically low-variance tail contains
unusually useful information rather than this just being generic dimensionality reduction."*

The program's question is what a method's stated desideratum measures and what it misses. Every
card so far has compared *spaces* (h vs z) or *methods*. This one holds one model fixed and
asks the question inside h: cut the representation down one direction at a time and watch two
readers disagree — the training objective, and a linear classifier. If a set of directions can
be deleted with the loss barely noticing while top-1 falls, those directions carry information
the objective never charged for.

The cell is deliberately the UNTREATED arm — our loss with the conditioner at z only — because
that is the standard-SSL situation the paper argues about: everything the objective knows about
h reaches it through the projector.

## Frame (anatomy stated loudly)

- **Cell** `in100.floorssl.s0.d256vm4zonly`, checkpoint `_ep100.pt` (epoch 99), ViT-S/16 @224,
  IN-100 (CMC split, `~/data/imagenet100`). Method `spectral` with `w_cond_h = 0` (`h_lamb`);
  the loss as trained is `32.8·inv + 157.8·cond_z` (weights read from the checkpoint, never
  typed in). This is the grey **z-only** row of `docs/paper/figures/fig_treatment_appendix.png`.
- **h** = `student.h.cls` = trunk `forward_features` CLS = the projector's input (D-036). It is
  post-LayerNorm, so the all-ones direction carries no variance by construction — measured
  λ₃₈₄ = 5.2e-10 against λ₃₈₃ = 7.3e-6. The k = 383 point is therefore a built-in null: it must
  change nothing.
- **PCA frame** — mean and eigenvectors of Cov(h) on the CLEAN eval features of
  `in100.train500.v1L` (n = 50,000, deterministic eval transform). Measured on that set:
  effective rank 53.9, participation ratio 18.1; the top 128 directions hold 96.2 % of the
  variance, the top 54 hold 83.7 %, the top 8 hold 48.0 %. The basis is fit ONCE and every
  measurement below is cut along it (`pca_fit=val` is available as a robustness variant).
- **The intervention** — h ↦ μ + B Bᵀ(h − μ) with B the kept d×k orthonormal basis. **Ambient
  coordinates throughout**: the rank falls to k, the dimension stays 384, so the frozen head and
  the probe see the same input shape at every k and nothing is re-parameterized.
- **Loss side** — augmented views from `in100.pairs1300.v1@audit_v1.o8`: the full train split
  (126,689 images), 8 stored views, stack `audit_v1` = `orbit_stack` = *exactly the augmentation
  family this lane trained under* (`aug=lejepa` builds `ViewsDataset` with that default stack).
  V = 4 (views 0–3), the lane's training V. z = the frozen projector's `z.proj.out` (256-d),
  eval mode (BatchNorm running statistics), fp32.
- **Probe side** — clean features: train `in100.train500.v1L` (50k) → val `in100.val.v1L` (5k),
  the paper's two readers, `linear_raw_v2` (the "Linear" column) and `knn_v1_k200`.

## What is measured, per arm × k

Loss terms in the loss's own units, at the training estimator's shape:

- **`inv`** — the invariance term: all-pairs mean squared difference between the V = 4 z's of one
  image (the lane's V-generic form, which reduces to its pairwise MSE at V = 2).
- **`cond_z`** — the SpectralConditioner value at z: per-image view MEANS (`cond_z_batch =
  view_mean`, the lane's anatomy), n = 512 rows per batch — the lane's own
  `(queue_steps+1)·bs = 4·128`, so n/d′ = 4 exactly as in training — on a FIXED seeded 128-d
  orthonormal slice, exact scatter, ε = 1e-4, mean term included, averaged over the 247 disjoint
  batches of a fixed permutation (the store is class-grouped; consecutive rows would give a
  within-class covariance).
- **`loss`** = `w_inv·inv + w_cond_z·cond_z` with the checkpoint's weights.
- **`cond_h`** — DIAGNOSTIC, weight 0 in this lane: the same conditioner read on the ablated h
  (128-d slice of 384). It is the term the treated arm pays and this one does not, recorded so
  we can see whether the quantity the objective omits is the one that would have noticed.

Downstream, top-1 in percent:

- **`linear_raw_v2`** and **`knn_v1_k200`** on the KEPT subspace (probe train top-1 and best
  epoch ride along, so an underfit probe is visible rather than inferred).
- **`linear_raw_v2` on the REMOVED complement alone** (μ + (I − BBᵀ)(h − μ)) — how much class
  information sits in the directions we threw away, which is Berker's "unusually useful"
  question asked directly.

Bookkeeping per row: `frac_var_removed` (fraction of the PCA-fit covariance trace removed) next
to `n_removed`, because the two arms are matched on the second and wildly unmatched on the first.

## Arms and grid

| arm | kept basis | role |
|---|---|---|
| `remove_low` | the top-k eigenvectors | the experiment |
| `remove_random` | a Haar-random k-frame, seeds 0/1/2 | the count-matched control |
| `remove_high` | the bottom-k eigenvectors | the mirror anchor (the instrument must be able to move) |

k ∈ {384, 368, 352, 320, 288, 256, 224, 192, 160, 128, 112, 96, 80, 64, 48, 32, 24, 16, 8, 4}.

**The control is count-matched, not energy-matched** — at k = 128 `remove_low` drops 3.8 % of the
variance while `remove_random` drops ≈ 66.7 % of it in expectation. That asymmetry is the point
of the control (it answers "is this just dimensionality reduction?"), and both numbers ride in
every row so no reading can quietly confuse them.

## Pre-registered predictions (committed 2026-09-09, before any number exists)

- **P1 — Berker's (i).** On `remove_low` the invariance term `inv` stays within a small factor of
  its k = 384 value across the range where top-1 has already fallen.
- **P2 — Berker's (ii).** On `remove_low` linear top-1 falls monotonically and materially while
  the total loss is still near flat.
- **P3 — the decomposition (mine).** If the total loss moves on `remove_low` it moves through
  `cond_z`, not `inv`, and it starts moving near k ≈ 256 = dim z, where z's covariance must lose
  rank. `inv` may *decrease*: fewer directions is less to disagree about, and the invariance term
  rewards discarding.
- **P4 — the control (mine).** At matched k, `remove_random` costs more top-1 than `remove_low`
  AND moves the loss more. The tail is "unusually useful" only if the two arms separate in the
  loss-versus-accuracy plane — accuracy lost per unit of loss moved, not accuracy alone.
- **P5 — the mirror.** `remove_high` collapses both readers early; refuted (as an instrument
  check, not as a claim) if the loss fails to move when the dominant directions are deleted.
- **P6 — the diagnostic.** `cond_h` rises on `remove_low` from the first removed directions
  onward, monotonically in the number removed.
- **P7 — the tail alone.** At k = 128 a linear probe on the removed 256 directions ALONE (3.8 %
  of the variance) reaches ≥ 20 % top-1 against 70.2 % full and 1 % chance. Refuted below 5 %.

## Selftests (run in-job; the numbers are not read until all four pass)

1. **Head parity** — `head(stored h.cls)` against the stored `z.proj.out`, `pairs100…o8` view0:
   median cosine ≥ .999. Extraction ran under bf16 autocast and this script runs fp32, so the
   residual is precision and nothing else.
2. **Conditioner parity** — `metrics.isotropy.moment_kl_slice` against the live
   `SpectralConditioner` on the same seeded slice: relative difference ≤ 1e-4.
3. **k = 384 anchor** — the ablation is the identity, so `linear_raw_v2` must reproduce the
   stored .7024 (`results/probes/in100.floorssl.s0.d256vm4zonly.extL.csv`) to probe noise.
4. **k = 383 null** — deleting only the LayerNorm null direction moves nothing measurably.

## Code

`sslgap/metrics/spectra.py` (`pca_frame`, `random_basis`, `keep_directions`) ·
`sslgap/metrics/isotropy.py` (`fixed_slice`, `moment_kl_slice` — the training-form conditioner
read on a held slice, promoted out of `experiments/e27_meanwash.py`'s inline copy per D-054) ·
`experiments/e35_directions.py` + `experiments/configs/e35_directions.yaml` ·
`slurm/e35_directions.sbatch`. Output `results/diag/e35_directions.<cell>.csv` and
`results/figures/e35/e35_directions.png`.

## Selftests — PASSED (smoke job 65274690, 2026-09-09, gpu150)

| check | result |
|---|---|
| frozen head reproduces the stored z | median cos **.999992**, min .999929 (fp32 here vs bf16 autocast at extraction) |
| readout conditioner = the training term | live .31238186 vs readout .31238185 |
| thickness = `metrics.orbit_energy`'s | .123475 both |
| k = 384 anchor vs the stored probe row | linear **.7024** and kNN **.5784**, both exact |
| k = 383 null (the LayerNorm direction) | every column identical to k = 384; a probe on that direction alone reads **.0100** = chance |

Smoke rows (loss columns on a 4,096-image subset; probe columns already at full protocol):

| removed | % var removed | loss | inv | cond_z | cond_h | Θ_z | Θ_h | linear | kNN | linear on the removed set alone |
|---|---|---|---|---|---|---|---|---|---|---|
| 0 | 0 | 244.27 | .1481 | 1.5172 | 3.045 | .132 | .304 | .7024 | .5784 | — |
| 1 | .00 | 244.27 | .1481 | 1.5172 | 3.045 | .132 | .304 | .7024 | .5784 | .0100 |
| 256 | 3.81 | 241.99 | .1443 | 1.5035 | 3.187 | .136 | .253 | .6936 | .5796 | .4218 |
| 368 | 40.52 | 287.66 | .0547 | 1.8115 | 3.878 | .197 | .140 | .5136 | .4520 | .6898 |

RIDER on `cond_h` (found while reading the smoke, applies to every h-side conditioner number here):
this arm never paid that term, so h sits at its own small scale — the top eigenvalue is .35 and the
tail is ~1e-5, while the term's ridge is ε = 1e-4. The ridge therefore sits ABOVE the tail
eigenvalues, so `cond_h` cannot resolve a 1e-5 direction from a deleted one. The number is faithful
to the term as configured (it is what the term would have contributed had it been switched on at
ep100) and is NOT a scale-free measure of how anisotropic h is.

## Numbers — RAW

*(full sweep 65275652: 20 truncation levels x 3 arms, random at 3 seeds; loss on all 126,689 train
images. Filled on landing.)*

## AGREED TAKEAWAY

*(empty by contract — filled only after the joint read with Berker)*
