# Two augmentation-cloud spaces: thickness, transport, and a backbone regularization that works

**Status: PROPOSAL v2 + derivations REVIEW-PENDING** (structure jointly settled
2026-08-06; derivations presented in-conversation the same day and carried here on
Berker's instruction — "carry your theory derivations on a proposal doc and i will
review from there". Each §3–§6 result now has a **Derivation** block; none is settled
until his pass. One correction to the v2 sketch is flagged where it occurs: §5(b)'s
bound needs an additive cloud-radius term.) Nothing in this document is a final
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

*Derivation (review-pending).* Let L(f, g) be any functional that reads the encoder f
only through z = g∘f — every invariance, moment, variance, or contrastive term
qualifies. Let Φ be a set of invertible reparameterizations φ of h-space such that the
encoder class is closed under postcomposition (φ∘f) and the head class under
precomposition (g∘φ⁻¹). Then for every φ ∈ Φ,

    L(φ∘f, g∘φ⁻¹) = L(f, g),

because (g∘φ⁻¹)∘(φ∘f) = g∘f pointwise — every sample's z is byte-identical, so any
functional of z is unchanged. Realization: whenever f ends with a linear layer and g
begins with one (every surveyed method), Φ contains all invertible affine maps h ↦ Ah —
absorb A into f's last weight and A⁻¹ into g's first. The center covariance's orbit is
then {AΣAᵀ}: pick A diagonal to drive any chosen eigenvalue toward 0 or ∞ at exactly
equal loss. Hence no loss level implies any conditioning bound at h. The only
obstructions are (i) architecture classes not closed under Φ — BN pinning after every
hidden linear shrinks the available group, which is the E19-T1 conduit/firewall fact,
measured — and (ii) regularizers on the two absorbed layers (weight decay taxes ‖A‖ —
the E23 BN-gauge caveat). These obstructions are exactly where §5's assumptions enter. ∎

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

*Derivation (review-pending).* Write S_W and S_B for the within-cloud and
between-center covariances (S_B ≻ 0), S_T = S_B + S_W. Selecting a k-dimensional
linear readout U that minimizes within-cloud energy tr(UᵀS_W U):

(a) under center-whitening UᵀS_B U = I: substitute U = S_B^{−1/2}V; the constraint
becomes VᵀV = I and the objective tr(VᵀCV) with C = S_B^{−1/2} S_W S_B^{−1/2}, which
orthonormal V minimizes at the bottom-k eigenvectors of C (Ky Fan) — equivalently the
bottom-k generalized eigenvectors of the pencil (S_W, S_B), eigenvalues λ.

(b) under total-whitening UᵀS_T U = I: same substitution with S_T gives the pencil
(S_W, S_T), eigenvalues μ. But S_W v = μ(S_B + S_W)v ⇔ (1−μ)S_W v = μS_B v ⇔
S_W v = [μ/(1−μ)]S_B v: the SAME eigenvectors, with μ = λ/(1+λ) strictly increasing —
so the bottom-k selection is identical.

The two solutions differ only by an invertible linear map inside the selected subspace
(an S_B-orthonormal vs an S_T-orthonormal frame): invisible to linear probes; a
metric readout (kNN) sees an eigenvalue-wise reweighting 1/√(1+λ_k), bounded and tight
when the selected λ's are small (thin clouds). ∎

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
linear head z = Ah, within-cloud and between-center covariances transform congruently
(S ↦ ASAᵀ), and det(S_W^z − λS_B^z) = det(A)²·det(S_W^h − λS_B^h): the generalized
eigenvalues — the *spectrum* of directional thickness — are exactly invariant, even
though the total Ω is not. The transmission-spectrum instrument ({a_k},
`e24_toy_spectrum*.csv`) is this statement's measured form; the identity
Ω_h = Λ²·Ω_z with Λ = (center gain)/(cloud gain) is its trace-level shadow. ∎

**(b) Nonlinear heads that are bi-Lipschitz give two-sided bounds — with a
cloud-radius correction.** *(Derivation, review-pending; CORRECTS the v2 sketch,
whose clean [L⁻², L²] claim was too strong.)* Assume L⁻¹‖x−y‖ ≤ ‖g(x)−g(y)‖ ≤ L‖x−y‖
on the data manifold. View pairs map directly: W_z ∈ [L⁻²W_h, L²W_h]. Centers do
NOT, because g of a cloud's center is not the center of the mapped cloud; the error is
bounded by the cloud's own radius: ‖m_i^z − g(m_i^h)‖ ≤ E_v‖g(h_iv) − g(m_i^h)‖ ≤
L·r̄_i (Jensen, then Lipschitz). Per center pair this gives
√B_z ∈ [L⁻¹√B_h − 2L·r̄_h, L√B_h + 2L·r̄_h], i.e. multiplicative bounds up to a
relative correction δ ≈ 2L²·r̄_h/√B_h ≲ L²√(2Ω_h). Combining:
Λ ∈ [L⁻²(1−δ), L²(1+δ)] and Ω_z within L⁴(1±δ)² of Ω_h. Below threshold
(Ω_h < c² < 1) δ is small and the head can move thickness by at most ≈ L⁴ — Λ is a
bounded quantity, not a free one; the correction is the price of nonlinearity and
vanishes as clouds thin. ∎

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
that Result 1 exposes.

*Derivation (review-pending).* KL(N(m,Σ)‖N(0,I_p)) = ½‖m‖² + ½Σᵢψ(λᵢ) with
ψ(λ) = λ − log λ − 1 ≥ 0, convex, zero only at λ = 1. A floor budget ε therefore
forces ψ(λᵢ) ≤ 2ε for EVERY eigenvalue of the center covariance separately, and
‖m‖² ≤ 2ε. The two roots of ψ(λ) = 2ε bound each eigenvalue two-sidedly:
λᵢ ∈ [−W₀(−e^{−1−2ε}), −W₋₁(−e^{−1−2ε})] (the two Lambert-W branches)
= [1 − 2√ε + O(ε), 1 + 2√ε + O(ε)] — no direction silently dies, none explodes,
quantitatively in ε alone. Rotation residual: a reparameterization that is loss-free
(Result 1's affine orbit) AND keeps the floor at budget must satisfy AΣAᵀ = Σ with
Σ = I + O(√ε), which forces AAᵀ = I + O(√ε): A lies within O(√ε) of the orthogonal
group. Rotations move no distance, no thickness, no probe — so the floor's whole
effect is to shrink Result 1's slack group from all invertible distortions to
(a √ε-neighborhood of) the rotations. Scope: this pins second moments; non-affine
freedom is constrained only through (b)'s bi-Lipschitz constant. ∎

## 6. Result 4 (transfer): what the thickness buys downstream

**Claim.** For a *declared* family of future tasks, the linear-probe error of a
representation decomposes as: (approximation term — how well the task family's
optimal predictors are realized as linear maps on cloud centers) + (a decoded
cloud-thickness term — within-cloud spread acting as label-independent noise on the
probe). Transfer claims are conditional on the semantic compatibility of the task
family with the centers; the thickness term is the part our training controls.

*Derivation (review-pending).* Evaluate on views drawn from the declared augmentation
family. Write a test embedding as h = m(image) + δ with E[δ | image] = 0 and
Cov(δ) = S_W. For ANY linear probe W, however it was fit:

    risk(W) = E‖y − Wᵀm‖² + tr(WᵀS_W W),

exactly — the cross term vanishes because δ is mean-zero given the image. The first
term is the task family's approximation error on cloud CENTERS (the part no
label-free procedure can certify — the formal content of "conditional on semantic
compatibility"). The second is a label-independent noise price ≤ ‖W‖²_F·λ_max(S_W).
With the floor active, S_B ≈ I (Result 3c), so S_W expressed in center-whitened
coordinates is precisely the directional-thickness operator of Results 2/3a — the
probe's noise price is thickness, in the same eigenbasis our training controls.
Margin/kNN rider: for neighbor-based readouts the same split appears as same-class vs
different-class cloud overlap — the touch census's enrichment is its empirical form,
and P-E25-3 is its registered field test. ∎

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
R5) **plus: §3–§6 derivations are drafted (2026-08-06) and REVIEW-PENDING** — one
sketch failed to formalize as stated and is reported, not patched: §5(b)'s clean
[L⁻², L²] transport needs the additive cloud-radius correction now in the text. The
native label-free census + per-frame c fits ran 2026-08-06
(`results/diag/omega_lawcensus_*.csv`); their numbers enter §8 only after the joint
read.

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
