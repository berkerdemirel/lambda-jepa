# METRICS — what the battery measures, and how to read it

Reference for every metric the audit computes (D-010). Each entry: definition (as implemented in
`sslgap/metrics/`), range & direction, which SSL desideratum it operationalizes / which method
optimizes it and where, and caveats. Estimator discipline (N-matching, PCA-k variants, nulls,
pairings) is binding per [PROTOCOL.md §6](PROTOCOL.md).

**General reading rules**

1. A metric value is only meaningful **relative to its nulls**: the random-init backbone (same
   architecture, no training) and, where flagged, the moment-matched Gaussian
   (`nulls.gaussian_match` — same mean/covariance, so spectral metrics are preserved by
   construction and deviations isolate non-Gaussian structure). Battery rows carry `null_gauss`.
2. Several metrics are **N- and d-sensitive** (rank measures, uniformity, isotropy statistics):
   compare only at matched N, and consult the PCA-64 variant when spaces differ in dimension
   (`variant` column: `raw|full`, `l2|full`, `raw|pca64`).
3. **No single metric is a verdict.** The battery is read as a profile; the report's thesis is
   precisely that single-space, single-number readings mislead.

---

## Geometry of view pairs

### `alignment` (pairs)
Wang–Isola alignment: E‖u−u′‖² over positive pairs, on ℓ2-normalized features. Range [0, 4];
**lower = more view-invariant**. Directly optimized (at z) by every view-based method's attraction
term (SimCLR numerator, BYOL/LeJEPA prediction, VICReg invariance term).
**Caveat (coupling):** alignment is only interpretable jointly with `uniformity` — a cone-collapsed
space (all features near one direction) has trivially small alignment with no invariance achievement.
M0 example: InfoNCE-toy h has alignment 0.016 *because* its uniformity is −0.10 (a tight cone), not
because h is invariant. Read (alignment, uniformity) as a pair, per the original paper.
Pairs come in two stacks (PROTOCOL §5): the method's **own** stack (self-consistency: did training
achieve its own objective's invariance) and the **fixed audit stack** (cross-method comparison).

### `cos_invariance` (pairs)
Mean cosine similarity of raw (unnormalized-then-cosine) positive pairs. Range [−1, 1]; **higher =
more view-invariant**. Same coupling caveat as alignment; kept because it is scale-free and matches
the RCDM-style invariance readouts in the literature.

### `pair_margin` (pairs) — the coupling caveat, made a number (D-013)
Positive-pair vs random-pair contrast within the same space: `cos_margin` = mean pos-pair cos −
mean random-pair cos (random pairs = cross-view, different images, identical pipeline);
`align_rel` = pos-pair E‖a−b‖² / random-pair E‖a−b‖². **cos_margin higher / align_rel lower =
view-invariance beyond global compactness.** Added 2026-07-08 (M1 dress rehearsal): every trained
h is cone-compact (uniformity −0.1…−0.45), so raw alignment/cos_invariance read h as *more*
invariant than z across all view methods — including MAE, which trained on no augmentations.

**Headline convention (Berker ruling 2026-07-14, D-037; resolves the E12-T9(ii) question):** the
two normalized forms couple to conditioning — a cone-collapsed space shrinks positive AND random
distances together, flattering ratio-form `align_rel` (E12 G-wave: the vicreg control reads
*better* on align_rel at h while its rand-pair cosine is .84). **Convention: where a space's
random-pair cosine reads ≈0 (decorrelated negatives), `cos_margin` is the headline invariance
readout; where it does not, the two readouts carry EQUAL weight and are reported jointly** — the
rand-cos panel is always shown as the referee either way.
Alignment/invariance glyphs are scored on the margin, never on the raw pair value alone.

## Spread / anti-collapse

### `uniformity`
Wang–Isola uniformity: log E exp(−2‖u−u′‖²) over pairs of ℓ2-normalized features (4096-point
subsample). Range (−4, 0] in practice on the sphere; **lower = more spread**; 0 ⇒ complete
collapse to a point. Implicitly optimized at z by contrastive repulsion (InfoNCE denominator);
explicitly by KoLeo (DINOv2, near-h). Dimension-sensitive (higher d → more room to spread): use
PCA-64 variant + Gaussian null when comparing across dims.

### `collapse_margin.{trace_cov, min_std, nn_dist_p5}`
Sanity tier: total variance, smallest per-dim std, 5th percentile of nearest-neighbor distances
(2048-subsample). No fixed range (scale-dependent — compare within a space across checkpoints/
epochs, not across spaces). **Near-zero values = complete collapse.** M0 example: LeJEPA λ=0
(SIGReg off) → probe 11.5%, RankMe 1.2 — the textbook failure these rows exist to catch early.

### `variance_floor.{hinge, frac_below_half_mean, frac_below_tenth_mean, min_over_mean_std}`
VICReg's variance criterion read as a meter. `hinge` = mean ReLU(1 − std_j) over dims — **exactly
the VICReg variance term**; VICReg training drives it to ~0 *at z* (its 8192-d expander), which is
the audit's reference point. Range [0, 1] (for centered/scaled data ~[0,1]); **lower = every
dimension alive at the γ=1 scale**. `hinge` is scale-dependent (a globally small-scale space has
large hinge without being degenerate) — the `frac_below_*_mean` and `min_over_mean_std` companions
are scale-relative and compare across spaces. Direction: fractions **lower = better-conditioned
variance profile**; `min_over_mean_std` **higher = flatter profile** (1 = perfectly flat).

## Redundancy / decorrelation

### `offdiag_redundancy.{offdiag_msq, mean_abs_corr}`
Barlow-Twins/VICReg decorrelation read as a meter, on standardized features: mean squared
off-diagonal of the correlation matrix, and mean |corr|. Range [0, 1]; **lower = more decorrelated**.
Optimized exactly at z by BT (cross-correlation → identity) and VICReg (covariance term). Their
papers never measured it at h — the audit's decorrelation-transfer cells are new measurements.
Caveat: correlation ≠ dependence — 0 here does not imply independent dimensions (BT's own motivation
for wide heads; nonlinear redundancy is invisible to this metric).

## Spectrum / rank

### `rankme`
Exp-entropy of the **uncentered** singular-value distribution (Garrido et al. 2023), computed via
the d×d gram. Range [1, min(N, d)]; **higher = more directions carry mass**; label-free
downstream proxy (validated on z in the paper; the h↔z monotonicity is an empirical regularity we
test, not a theorem). Needs N ≳ 10k for stability; never compare across different N. Not centered —
a large mean vector eats one direction.

### `effective_rank` / `participation_ratio`
On the **centered** covariance spectrum: exp-entropy of normalized eigenvalues / (Σλ)²/Σλ². Ranges
[1, d]; **higher = higher-dimensional structure**. `participation_ratio` is the prior project's
`eff_rank` (M0 parity anchor). Complements RankMe (centered vs not, eigenvalues vs singular values).
Dimensional collapse (Jing et al.) reads as low values *relative to d* — report alongside d.

### `alpha`
Power-law exponent of the covariance eigenspectrum (α-ReQ, Agrawal et al.): −slope of log λ vs
log rank over the top-200. **~0 = flat/isotropic spectrum; larger = steeper decay.** α-ReQ found a
readout-quality band at h for their settings (neither too flat nor too steep); we treat that band as
literature context, not a target. Sensitive to the fit range (top-200 fixed by protocol).

## Isotropy / Gaussianity — always a PAIR

### `epps_pulley`
Sliced Epps–Pulley statistic: distance of random 1-D projections to N(0,1) (256 seeded slices,
standardized dims by default; the LeJEPA SIGReg statistic run as a meter). Range [0, ∞);
**lower = closer to isotropic Gaussian under this test**. Optimized *at z* by LeJEPA (SIGReg).
**Binding caveat (user, 2026-07-02): EP is a rejection-style test — a LOW value (failure to
reject) does NOT certify isotropic Gaussianity.** Two failure modes: (i) Diaconis–Freedman: random
1-D projections of almost any high-dimensional cloud look Gaussian (decorrelation + CLT), so slices
wash out structured non-Gaussianity; (ii) at small N the test is underpowered. Large EP *is*
informative (the space is measurably non-Gaussian); small EP is only "not rejected". This is why the
battery never reports EP alone:

### `kurt_topeig.{mean, worst}` (the partner)
Excess kurtosis along the top-10 covariance eigenvectors — the privileged directions random slices
wash out. 0 = Gaussian; **|larger| = heavier/lighter tails along principal directions**; `worst` =
max |excess| (lead isotropy claims with this, per house rule). M0 example: DINO's bottleneck has
EP-competitive slices but `worst` ≈ 45 — wildly non-Gaussian where it matters.

### `kurt_slices_mean_abs` (context)
Mean |excess kurtosis| over random slices — kept to *demonstrate* the Diaconis–Freedman washout
(expect ≈ 0 even when kurt_topeig explodes), not as evidence of Gaussianity.

**Class-count caveat (E01-T11, agreed 2026-07-10).** All fourth-moment cells lose sensitivity to
CLASS-driven structure as class count grows: each direction's 1-d marginal mixes many class
means and Gaussianizes (slice-CLT), even while class geometry strengthens in second order
(IN-100 h: between-class/total variance .18–.51 and per-direction η² .23–.55, yet mean |kurt|
only .37–.69 — results/diag/e1_class_variance_h.csv). Kurtosis magnitudes are therefore NOT
comparable across class counts (toy 10-class vs IN-100), and small |excess| at many classes
does not mean "no class structure". For class-driven structure use label-aware cells
(between/total, per-direction η², the E10-T3 cluster-alignment battery-v2 candidate); also
kurtosis is a shape-deviation flag, never a mode counter (E10-T3 control: real feature spaces
are ~10-modal from random init on).

## Cross-space / cross-model structure

### `neighbor_jaccard` (+ `knn_consistency`)
Mean Jaccard overlap of k-NN sets (k=10, 5k subsample) computed in two spaces over the SAME images.
Range [0, 1]; **higher = the two spaces agree on neighborhoods**. The direct "does the head reorder
semantic geometry" readout; low overlap with higher h-probe accuracy = the head actively rewrites
neighborhoods.

### `knn_label_agreement`
Fraction of images whose k-NN majority *label* matches between two spaces. Range [0,1]; higher =
functionally similar neighborhoods even if identities differ. Coarser than Jaccard (label-level).

### `cka_linear` — CONTESTED
Biased linear CKA on centered features. Range [0, 1]; higher = more similar (up to rotation/scale).
**Refuted for cross-representation correspondence in the report's adversarial verification** — only
ever reported as the triple with `neighbor_jaccard` and `procrustes_distance`; a CKA number alone
is not evidence in this project.

### `procrustes_distance`
Orthogonal-Procrustes residual between PCA-64 projections, unit-scaled. Range [0, √2]; **lower =
similar up to rotation**. The triple's third leg; dimension-matched by construction.

### relative representations (`relrep_*`; D-009)
Represent each sample by cosine similarities to **anchors**. Two modes with different semantics:
dataset anchors (A = feature dim, fixed seeded image ids per manifest, SHARED across models) put
different models in one frame — the primary cross-model tool (`experiments/compare.py`); the
random-orthonormal control is a *rotation* of the normalized space (checks rotation-invariance
claims of metrics; does NOT align models). Validated property: relrep of a rotated copy of a space
matches the original exactly (CKA 1.0). **Semantics verified against latentis (relrep authors'
library, @800699f): canonical cosine projection = l2-normalize both sides + dot, NO centering by
default; Centering/StandardScaling exist only as optional `abs_transform` pre-transforms — ours
mirrors this (`abs_transform` param, default `none`; non-default choices reported with results).** Battery metrics can be computed on the relrep space; report
anchor mode/A/seed with every number.

## Probes (utility readouts; PROTOCOL §4, D-006v2)

`linear_raw_v1` (headline linear separability: plain Linear, raw features), `knn_v1` (headline
lightly-parity weighted kNN k=200/t=0.1), `linear_house_v1` (LN+Linear; in-house continuity/parity),
`linear_l2_v1` + `attentive_v1` (+ per-paper variants like MAE's BN probe) in E11 where probe
sensitivity is the object. Probes measure *usefulness under a fixed readout class* — the report's
E11 point is that they are protocol-dependent by 5–17 pts across families, hence fixed and disclosed.

## Metric ↔ downstream relations

What we know from the literature (context, not our evidence): RankMe correlates with downstream
accuracy on z across hyperparameters (Garrido et al.); α at h sits in a quality band (α-ReQ);
alignment+uniformity at z correlate with linear probes in Wang–Isola's settings. **E3 measures these
relations in OUR two-space setting** (per space × per family, under shift). M0's
`results/M0/metric_vs_probe.csv` gives exploratory rank correlations across the M0 spaces —
**non-evidential** (few models, one seed, layer-spaces are not independent); it exists to build
intuition for E3's design, nothing more (D-010).
