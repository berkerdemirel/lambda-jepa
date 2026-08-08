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

**Decomposition convention (Berker 2026-07-14, extends D-037):** `cos_margin` is NEVER reported
alone — its two components are shown *with* it: `pos_cos` (same-image view alignment = the actual
invariance) and `rand_cos` (random-pair cosine = the cone/anisotropy). The margin is a *difference*,
so reading it alone conflates the two. Decomposing the h-side controls (2026-07-14): the backbone's
positive-pair alignment stays HIGH (pos_cos ≈0.92 vicreg / 0.94 lejepa at h, ≥ its z value) while
the low h-margin is the high cone (rand_cos 0.55–0.70); the h→z margin jump is the cone collapsing,
NOT alignment appearing. So **"the backbone is not invariant" is a misreading of the margin** — h is
aligned, just anisotropic. Report the (pos_cos, rand_cos, cos_margin) triple wherever invariance is
shown (H-wave figs `results/figures/e12h/` do this).

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

## ISO-ladder v2 additions (D-040, 2026-07-17; Berker-approved "simple implementations")

- **`gauss_kl_full`** (`isotropy.gauss_kl_full`; subkeys `total/location/spectrum`): the exact
  MOMENT component of KL(P‖N(0,I)) via the Pythagorean split KL(P‖N(0,I)) = KL(P‖P_G) +
  KL(P_G‖N(0,I)) — a two-Gaussian log-ratio is quadratic, so only (μ,Σ) enter; machine-checked
  cluster-blind (its gauss-null row equals the data row identically). `location` = ‖μ‖²/2d (the
  cone as a first-class number); `spectrum` = Stein/Burg divergence Σ(λ−1−log λ)/2d, eigenvalues
  shrunk by 1e-3·mean (logdet barrier alive). Deliberately scale-sensitive (calibration
  instrument). CAVEAT: a per-dim-calibrated rank collapse (λ ≈ 0.1–0.2, unit diagonal — E17
  sigreg_inv) reads only mildly elevated — this COMPLEMENTS effrank, it does not replace it.
- **`radial_gauss`** (`isotropy.radial_gauss`; subkeys `var_ratio/mean_ratio`): cross-fit radial
  law — whitening moments from one half, r² = ‖W(x−μ)‖² on the held-out half; var_ratio =
  Var(r²)/2d, mean_ratio = mean(r²)/d. Rotation-invariant and CLT-immune (reads shells vs
  clumped mixtures where sliced tests get fooled at high rank). Finite-sample whitening bias is
  systematic ⇒ **read BOTH subkeys against the battery's gauss_null row** (EP precedent), never
  against the analytic χ²_d (a KS-vs-χ² variant saturated and was dropped at validation).
  Validation snapshot (data var_ratio / null): simclr ctrl 17.3/0.85 · uniform 16.4/0.90 ·
  f2 5.7/0.71 · lejepa ctrl 0.80/0.13 · sigreg_inv 0.29/0.08 (most shell-compressed measured).
Both registered in `DEFAULT_BATTERY` (l2_variant off, gauss_null on, spectral boot); they enter
every future audit; historical stores gain them on their next battery pass.

## Head linearity — how much of z is linearly reachable from h (D-015; built 2026-08-07)

`sslgap.metrics.cross.linear_map_fit`, driven by `experiments/head_linearity.py` →
`results/diag/head_linearity.csv`. OLS h → z fitted on a train split (both sides centred by
TRAIN means), scored on a held-out split. Three variants over the same V=8 cloud stores:

- **`centers`** — `mean_v h[i,v] → mean_v z[i,v]`: are the cloud CENTRES linearly related,
  with augmentation averaged out?
- **`views`** — `h[i,v] → z[i,v]` over all (image, view) rows: the same for the full map,
  augmentation directions included.
- **`eval`** — deterministic single-view features, train-manifest fit → val-manifest score
  (continuity with the retired `e2x_zpred` numbers).

**Split discipline:** `centers`/`views` split 80/20 **by image**, never by row — a row-wise
split puts other views of the same image on both sides and makes `views` trivially easy.

**`r2_total` is NOT reportable alone (D-060).** Berker cancelled it as a standalone metric in
July: a contractive many-to-little head reads HIGH R² while doing heavy nonlinear work, because
R² measures linear *reachability of the output*, not head magnitude. Every row therefore carries
the fitted map's spectrum, which reads contraction directly and is what D-060 said survives:

| column | reads |
|---|---|
| `sigma_min`, `cond` | whether the linear part annihilates directions of h |
| `effrank_map` | how many directions the linear part actually uses (entropy of W's singular spectrum; scale-free) |
| `resid_effrank` | effective rank of what the linear map misses — the nonlinear work |

**Instrument validation:** the `K0` cells (`head_layers=0`, i.e. `Linear(384→2048) →
Linear(2048→256)` with no nonlinearity = one affine map) return **r2_total = 1.000 exactly**.
The `W32` cells (`expander_hidden=32`, a 32-wide bottleneck) return r2 ≈ .98 through a map of
**effective rank 2.5–3.6 with cond ~1e6** — D-060's failure mode made concrete: near-perfect
linear reachability *because* the head has amputated almost everything.

## The cloud calculus (D-068, 2026-07-29; usage rules amended 2026-08-03)

The program's primary lens since E23. Embed one image's V augmentations → a **cloud** of V
points; its **center** is their mean. Everything here is **label-free** by design and computed
from V-view stores (`o8` is sufficient — see the debias note). Implemented in
`sslgap/metrics/orbit_energy.py` and `sslgap/metrics/census.py`; terminology per
[GLOSSARY.md](GLOSSARY.md) (**the code still spells "orbit" — the rename waits for D-083's
Wave B, see the glossary's decoder line**).

### `W`, `B`, `omega` (`orbit_energies`) — thickness

- **W** = within-cloud energy: `E_i E_{u<v} ‖s_iu − s_iv‖²`. Pairwise and mean-free, so it is
  **V-unbiased by construction**; `r_rms = √(W/2)` is the cloud radius.
- **B** = between-center energy: `E_{i≠j} ‖m_i − m_j‖²` **debiased** as `B = B̂ − W/V`. The
  sample view-mean inflates center energy by the view noise (`E‖m̂_i − m̂_j‖² = ‖µ_i − µ_j‖² +
  2σ̄²/V`), which is exactly the E17 o8-vs-o32 ~5% gap. Verified: `B̂` drifts ~30% over
  V ∈ {2, 8, 32} while debiased `B` is V-invariant to three digits — **which is why o8 stores
  suffice and o32 was never needed**.
- **Ω = W/B** — **thickness**, cloud size relative to image spacing. One number per space per
  checkpoint. Range (0, ∞), no intrinsic good direction: it is read against the touching
  threshold below, not maximized or minimized.
- With labels, `orbit_energies` additionally splits center energy within/between class
  (`B_within_cls`, `B_between_cls`, `BW_cls`). **These are evaluation-side only** — per D-068's
  addendum the primary lens stays label-free, and class-conditioned columns were dropped from
  new guillotines (Berker's bias concern).

### `a`, `b`, `lam` (`transmission`) — what the head does to a cloud

From two `orbit_energies` dicts on the same store: `a² = W_z/W_h` (how within-cloud energy is
scaled through the head), `b² = B_z/B_h` (how center separation is scaled), **`Λ = b/a`**, with
the identity `Ω_h = Ω_z·Λ²`.

**Binding usage rule (D-068 addendum, Berker 2026-08-03):** cross-space and cross-model reads
ride on **Ω and Λ only** — they are dimension-free. `a` and `b` are within-cell decompositions:
their levels carry a `√(D_z/D_h)` dimension mass and a scale factor that is unidentifiable
under BN, so they do not compare across cells of different width. If a cross-D level is
unavoidable, use the per-direction `â = a·√(D_h/D_z)`. Unbounded-ratio panels get per-row
y-limits.

### `transmission_spectrum` — the `{a_k}` selectivity spectrum

Least-squares `J` minimizing `E‖δz − Jδh‖²` over view residuals, solved in the `δh` PC basis;
`a_k = ‖J u_k‖` is the head's gain on the k-th h-residual PC. **Selectivity is the spread of
`{a_k}`** (`cv_ak`), not its level. The scalar `a²` decomposes linearly as
`a2_lin = Σ a_k² s_k / Σ s_k` with residual `1 − r2_lin`. Replaced the rejected α-grouped-
projector control (which amputated head–trunk co-training; its flat spectrum is the null here).

**Caveat carried from E23-T5: a small `a` alone is not a health readout.** Two different
regimes produce it — forced amputation (a too-narrow head) and learned selection (a healthy
head discarding augmentation directions). Read `(a, cv_ak)` together with the h-state.

### `touch_census` — who overlaps whom

Every instance as anchor over all candidates; candidate *j* **touches** anchor *i* when their
radii sum exceeds their center distance. **ω(i,j) = (r_i + r_j − d_ij)/(r_i + r_j)** is the
signed fractional overlap depth (negative = a gap, in radius units). Reported as pool-size-free
probabilities `p_pos` (touch-same-class) and `p_neg` (touch-different), their ratio
`enrich = p_pos/p_neg`, and `purity` against the base rate; ω **levels** are reported
class-conditioned (per-anchor median ω to same vs foreign clouds). Per-class rows use
ratio-of-means (per-anchor ratios blow up at `deg_neg = 0`); the run summary is the median over
classes (E17 convention). An α = .75 column rides along as an operating-point robustness check.

**Caveat:** cloud overlap does not guarantee same-class connectivity — the census measures
touching, and whether touching is *selective* is exactly what `enrich`/`purity` report.

### `touch_law_stats` — the calibration that makes Ω readable

Thickness alone is an arbitrary ratio. Label-free aggregates over all unordered pairs: **M** =
median signed ω, **T** = touching fraction. If a training family shares its constellation's
*shape* and differs only in scale, then

    M = 1 − c/√Ω

with **c** the family's **shape constant**. Consequences: a **touching threshold Ω\* = c²** per
frame, and T a fixed increasing function of Ω. Measured: **c = .82 ± .02** across 51 toy + 7
IN-100 + 3 IN-1k runs *and* the 6-model public zoo (the zoo's own fit R² = .996 over four
training families) — one constant, both spaces. **Collapse departs the line** (implied c
1.6–2.0), which makes the law a health boundary as well as a calibration.

Also returned: **`omega_local(k)`** — per-anchor cloud-local thickness, own-cloud pair energy
over the mean debiased center energy to its k nearest centers (default k = 20), with the exact
pair debias `(W_i+W_j)/(2V)`. It rescues the accuracy sort exactly where global Ω saturates
(stage C′: −.175 → −.720). Plus the α = 1 touch-graph component census.

### `touch_graph_profile` — the connectivity fingerprint

Clouds are linked when `α·(r_i + r_j) > d_ij`, with α swept (default .5 → 1.5): α = 1 is the
physical touch relation, α < 1 demands overlap depth, α > 1 admits near-misses. Per α: touching
fraction, component count, giant-component share, singleton count. The **transition point** is
the informative number — at dense frames the α = 1 graph saturates to one component and says
nothing, while the α-profile still discriminates. Trajectories (toy K3 cadence ep38→150, the
landed IN-100 cadence) show the fragmentation front still moving at budget end.

**Scope:** these columns are E23-scoped instruments computed by the landing scripts, **not**
registered in `DEFAULT_BATTERY`. Battery-wide promotion needs its own D-row (the D-054 path).
