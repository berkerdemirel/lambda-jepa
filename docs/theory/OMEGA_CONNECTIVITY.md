# Two augmentation-cloud spaces: thickness, transport, and a backbone regularization that works

**Status: PROPOSAL v2** (restructured 2026-08-06 to the jointly agreed spine — Berker's
six-point map, in-conversation review of v1; drafted by Claude). Formal results in
§3–§6 are stated with **proof sketches**; full derivations are the next work item and
happen in-conversation before they land here. Nothing in this document is a final
program conclusion; those are joint and live in `docs/DECISIONS.md`. Predictions in
§9 were registered before their tests. Terminology note: this document says **cloud**
(the set of embeddings of one image's augmentations); code artifacts retain the
historical "orbit" naming (`sslgap/metrics/orbit_energy.py`, the `.o8` stores).

---

## 1. Positioning: the two-space problem and what we deliver

Joint-embedding SSL trains an encoder by making augmented views of an image agree,
under a regularizer that prevents collapse. Two spaces exist in every such method:

- **h** — the backbone output, the representation actually kept and used;
- **z** — the output of a small MLP head on h, where the loss is applied.

Of the eleven methods surveyed in this program's founding report, **zero** impose
their loss on the representation they tell you to probe. The discrepancy between the
two spaces is therefore not an implementation detail — it is the central unexplained
object. **This document's position: explain the relationship between the two
augmentation-cloud spaces, and derive from that explanation a backbone-space
regularization that provably does something — which is the program's moment floor at
h, elevated from empirical ingredient to theoretical consequence.**

## 2. The tool: augmentation-cloud thickness, and its measurement theory

Embed one image's V augmentations: a **cloud** of V points. Its **center** is their
mean; the image's semantic content lives (to first order) at the center, and the
augmentation-induced variation is the cloud's spread around it. Over N images:

- **W** = mean squared distance between two views of the same image (averaged over
  view pairs, then images) — *within-cloud energy*, how big clouds are;
- **B** = mean squared distance between two different images' centers (debiased by
  W/V for the sampling noise of the estimated center) — *between-center energy*, how
  spaced the images are;
- **thickness Ω = W/B** — cloud size relative to spacing, one number per space per
  checkpoint. All label-free.

Thickness alone would be an arbitrary ratio without a calibration. The **touch law**
supplies it. Say two clouds *touch* when their radii sum exceeds their center
distance; let M be the median signed overlap over all pairs and T the touching
fraction. If runs of a training family share the constellation's *shape* and differ
only in scale (an empirical hypothesis, tested below), then

    M = 1 − c/√Ω,

with c a single **shape constant** of the family. Consequences: a measurable
**touching threshold** Ω\* = c² per frame (median clouds overlap above it), and T a
fixed increasing function of Ω. Measured (9 view-mean toy runs, both spaces):
c = 0.849 (h) / 0.836 (z) at R² = 0.99 — one constant, two spaces, runs differing in
dose and estimator (`results/figures/theory/omega_touchlaw.png`). Every top-accuracy
run sits at **49–68% of the threshold**; accuracy degrades monotonically toward and
past it, and the three estimator-damaged runs lie on the same geometric line —
damage routed through thickness (`omega_band.png`, §8). The connectivity reading
(touch graph, giant component, separated groups) extends the tool from pairs to the
whole space and carries the performance proxies (§9, R3).

## 3. Result 1 (impossibility): projected objectives cannot condition the backbone

**Claim.** Any training objective evaluated on z = g(h) — invariance terms,
moment/variance regularizers, contrastive losses alike — leaves the backbone's
conditioning unconstrained whenever the head g is expressive: for every
reparameterization φ of h-space there is a compensating head g∘φ⁻¹ giving the
identical loss with arbitrarily ill-conditioned h.

*Proof sketch.* (f, g) → (φ∘f, g∘φ⁻¹) is loss-invariant for any diffeomorphism φ;
choose φ to collapse or explode chosen directions. The only obstructions are the
architecture classes of f and g (capacity, smoothness) — which is exactly where §5's
assumptions enter. ∎ *(half a page to formalize; pending.)*

This is the theorem-form of two program observations: the leakage frame (within-cloud
energy at h hiding in directions the head's Jacobian kills — E23) and the 0-of-11
survey fact. Its role is to *license* §5: without an h-side term, nothing about h is
guaranteed; the question becomes what the cheapest sufficient h-side term is.

## 4. Result 2 (ideal case): whitened alignment selects one eigenspace, under either whitening

**Claim.** In a fixed shared feature class with nondegenerate center covariance,
minimizing within-cloud thickness under a whitening constraint selects the
generalized eigenspace of (within-cloud covariance, between-center covariance) with
the smallest eigenvalues — the directions where clouds are thinnest relative to
center spread — and **center-whitening and total-whitening select the same
eigenspace**: total covariance = between + within, so the two generalized
eigenproblems share eigenvectors, with eigenvalues related by the monotone map
λ ↦ λ/(1+λ). Ordering, hence selection, is identical.

*Proof sketch.* Simultaneous diagonalization of the two quadratic forms; monotone
eigenvalue transform preserves the bottom-k choice. ∎ *(one page; pending.)*

**Program remark (the unification).** Center- vs total-whitening is *exactly* the
view-mean vs pooled anatomy: view-mean floors whiten per-image center moments,
pooled floors whiten the total. Result 2 says the two objectives select the same
representation in the linear, exact-estimation regime — so everything that separated
them in practice (the ring-staleness noise, T4; the scatter competition, T5;
estimator sample-complexity at n/d′) was *estimation*, not objective. Two of the
program's costliest empirical episodes become corollaries.

## 5. Result 3 (transport, and the floor as the h-side answer)

Three parts, increasing realism:

**(a) Linear heads transport directional thickness exactly.** For an invertible
linear head, within-cloud and between-center covariances transform congruently, so
the *spectrum* of directional thickness (the generalized eigenvalues of the pair) is
exactly invariant — thickness direction-by-direction passes through unchanged, even
though the total Ω does not. The transmission-spectrum instrument ({a_k},
`e24_toy_spectrum*.csv`) is this statement's measured form; the identity
Ω_h = Λ²·Ω_z with Λ = (center gain)/(cloud gain) is its trace-level shadow.

**(b) Nonlinear heads that are bi-Lipschitz give two-sided trace bounds.** If the
head neither collapses nor explodes distances by more than a factor L, transported
thickness is pinned within [L⁻², L²] of the source — Λ becomes a bounded quantity
rather than a free one. This is the honest nonlinear version of (a), and the
assumption is what a finite, trained MLP plausibly satisfies on the data manifold.
*(short; pending.)*

**(c) The moment floor pins the backbone's shape up to rotation.** Matching h's
center moments to a standard Gaussian by a KL budget ε gives, from
KL = ½Σᵢ(λᵢ − log λᵢ − 1) + ½‖mean‖², **two-sided bounds on every eigenvalue** of
the center covariance as a function of ε alone — no eigenvalue can silently die
(active rank is protected, quantitatively) and none can blow up. And it resolves
Result 1's slack in plain terms: after Result 1, many differently-shaped backbones
achieve the same loss — the free distortions are unlimited. Once the floor holds,
**the only transformations of h that cost nothing are rotations** — and rotations
change no distance, no thickness, no probe. The floor shrinks the freedom from "any
distortion" to "rotation only," which is why a few percent of the gradient budget
suffices: it is not pulling the representation anywhere, it is removing the slack
that Result 1 exposes. *(KL-to-eigenvalue bounds are standard; the
rotation-residual statement is short; pending.)*

## 6. Result 4 (transfer): what the thickness buys downstream

**Claim.** For a *declared* family of future tasks, the linear-probe error of a
representation decomposes as: (approximation term — how well the task family's
optimal predictors are realized as linear maps on cloud centers) + (a decoded
cloud-thickness term — within-cloud spread acting as label-independent noise on the
probe). Transfer claims are conditional on the semantic compatibility of the task
family with the centers; the thickness term is the part our training controls.
*(one lemma with explicit constants; pending.)*

The E25 transfer battery is this result's field test, and P-E25-3 (per-dataset
thickness computed on the transfer datasets' own images, predicting per-dataset
accuracy) is its registered, not-yet-fired instrument.

## 7. Operational consequence: a frame-invariant placement target

Raw Ω does not transport across frames (established: P-E24-5). The touch law gives
every frame its own measured threshold c², and R6 (registered) predicts the good
band is frame-invariant **in threshold units**: best runs at ~0.5–0.7 of their own
frame's threshold. If R6 holds, dose-placement stops being a per-frame search:
**dose toward the geometric target Ω_h ≈ 0.6·c²(frame)**, measurable label-free from
the run itself. This is the free guide for the multi-ViT in1k program (E25-T1:
ViT-S/B/L with placement on target), where the current in1k state reads as
over-removal (vm4 at Ω(cls) = .211, below the strong-public-model stratum .32–.39).

## 8. Experimental support (landed; unchanged from v1 except terminology)

Primary population: the 9 completed view-mean toy runs (6 dose-map cells + 3
estimator variants; Imagenette, ViT-S/8, 150 ep, V=4 training views; measurement =
V=8 fresh views, full train split, fp16 stores). Evidence: the touch-law fit
(§2 numbers; deprecated pooled-anatomy runs shown grey as a stability check only);
the label-free chain ρ(acc, Ω) = ρ(acc, T) = ρ(acc, M) = −0.90 (one signal, n=9);
the band table (best runs .49–.68 of threshold; estimator damage routed through
geometry); per-stage Λ factors on view-mean runs of all three frames (every stage's
factor > 1 in healthy runs; the early trunk's share grows with scale 1.26 → 1.52 →
1.99, toy → IN-100 → IN-1k; within-frame sd .02–.09 log units, up to .13 on the n=3
IN-1k column). The zoo reference band (E26; six public backbones, verified anchors)
adds: the augmentation-invariance family sits at Ω(cls) .32–.39, language-supervised
at ≈1.0, reconstruction at 2.65, with ρ(probe, Ω) = +.83 across the zoo — recorded
raw, scoring joint.

## 9. Registered predictions (unchanged from v1)

R1 cloud-local thickness Ω_local (k=20) re-sorts the saturated band (|ρ| ≥ .8 where
global Ω gave −.66); R2 Ω_local obeys the same law with its own stable constant;
R3 the accuracy cliff coincides with the touch graph's giant-component jump, and
separated-group count tracks accuracy below threshold; R4 stage-factor health on the
C′ grid (craters localize to stages with factor < 1); R5 the constant c transports
to IN-100/IN-1k within a few percent; R6 the band transports in threshold units;
R7 the zoo obeys the direction and threshold position with no engineered moment
matching. All computable from landed stores except R4's grid read (landed, held).

## 10. Threats to validity

As v1 (n=9 primary; construction coupling — the law's content is the one-constant
functional form and the accuracy links, not the correlation; one signal not three;
designed not sampled runs; reconstructed label-free aggregates pending native
recomputation; probe = online linear at toy scale; c possibly family-dependent until
R5) **plus: §3–§6 are proof sketches** — statements are jointly agreed, derivations
pending; any sketch that fails to formalize is reported as a failure, not patched.

## Appendix A — settings (unchanged from v1)

Toy frame Imagenette 10-class N=9,469 @128, ViT-S/8, h = CLS 384-d, head 2×2048 BN
MLP → 256-d z, V=4 train views; measurement V=8, fp16, stations trunk-3/6/9/CLS +
head taps; pairwise W, debiased B, raw features (l2 rider); census = all pairs.

## Appendix B — reproduction

`experiments/e24_grid_metrics.py` → spaces/census CSVs; `experiments/e24_omega_sort.py`
→ accuracy join; `experiments/omega_theory_figs.py` → both figures + printed fit
numbers; `experiments/e26_stations.py`/`e26_guillotine.py` → zoo band;
`sslgap/metrics/orbit_energy.py` (cloud energies; historical module name),
`sslgap/metrics/census.py` (touch census). Ω_local + touch-graph summaries land
there upon R1–R3 execution.
