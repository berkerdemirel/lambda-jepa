# What we want from a representation — the same-page note

**Status: agreed discussion basis (Berker + Claude, 2026-07-02) — the framework conversation's
written form, saved at Berker's request. NOT results, NOT locked pre-registration. Claims about
prior papers below are stated from memory at discussion confidence; a quote-verification pass
(same standard as the founding report) is the agreed next step before anything here is cited.**

Companions: THEORY_MAP.md (which space each theorem governs) · OPEN_PROBLEMS.md ·
LDM_HEAD_COMPOSITION.md (OP-17) · the E1/E2/E4/E10/E11 cards this note retro-grounds.

## 0 · Berker's position (the input to this note)

Much of SSL (BYOL, DINO especially) is benchmark-tuned engineering: given the methods'
descriptions ex ante, there is no principled way to predict their ordering — they do not follow
from an interpretable account of what a good representation *is*. **And the claim holds ex post
(Berker, 2026-07-03): even reading the validation losses of the very quantities each method
optimizes, you cannot predict the downstream ordering.** Two independent reasons: (i) *wrong
functional* — the achieved value of a constraint says nothing about usable structure (Tschannen:
tighter MI bounds can yield worse representations; collapse: BYOL-minus-predictor satisfies its
loss at 0.3% accuracy; our SIGReg cells: trained-low loss, non-Gaussian space; the label-free
selection literature — RankMe/α-ReQ — exists precisely because loss values don't rank);
(ii) *wrong space* — the loss is measured a head away from what gets probed. Caveat kept honest:
within one method's hyperparameter sweep (functional and space fixed) loss↔downstream correlation
can hold — LeJEPA claims exactly this — the failure is cross-method. E3 is the constructive
response: which statistic, measured in which space, recovers the ordering the loss values cannot. VICReg/LeJEPA at least argue
from shape/regularization — but impose it after an MLP. MI maximization is not sufficient
(zip/identity counterexample: the image is its own informationally-complete representation with
zero utility). What we want: **sufficiency + invariances that relate to minimality**, delivered as
**hierarchical abstraction** — the police-sketch analogy: a shared generative prior (the drawer
knows faces; the model knows natural images), and the witness communicates the most discriminating
deviations first (dark hair, glasses, age), fine detail as memory/rate permits. **Factor
groupings** should be linearly separable (day/night, crowded, cars present); **derived quantities**
(the *number* of cars) need not be linearly decodable — they are computations over represented
factors ("see the cars linearly, then count"). **Anti-collapse is the elephant in the room**: an
ungrounded bias about embedding geometry; ungrounded regularization forfeits the best
representation. LeJEPA is interesting but (1) proves optimality for a space it discards
(projector), and (2) enforces isotropy by a statistical test that can read zero without the
distribution being isotropic Gaussian — failing to reject ≠ accepting.

## 1 · The desiderata, formalized

Setup: world factors `s ~ p(s)` (structured), nuisance `n`, image `x = g(s, n)`; views defined by
an augmentation/orbit structure; task tiers below.

- **D1 — Sufficiency for a task *family***: not one label; every task expressible over `s`
  (operationally: every task invariant under the view group).
- **D2 — Invariance → minimality**: discard `n`; invariance is the *mechanism* of minimality.
  (Achille–Soatto: among sufficient representations, minimal ⇔ invariant to nuisance.)
- **D3 — Hierarchy**: the representation is the *residual code* of `x` under a shared prior, rate
  allocated to the most discriminating deviations first; fine detail last.
- **D4 — Tiered decodability**: factor groupings **linearly** readable; derived quantities
  (counts, relations) computable from the representation at bounded depth but not necessarily
  linearly. The sharpest and least-covered requirement.
- **D5 — Grounded, honestly-enforced marginal**: the embedding marginal is a *derived commitment*
  (an explicit latent prior, or a minimax argument), not an aesthetic; and it is enforced by a
  discrepancy that is actually characteristic, not a finite test battery.

## 2 · The SIGReg honesty question (D5's enforcement half), stated precisely

A distribution *is* multivariate Gaussian iff **every** 1-d projection is Gaussian — so the
*population* sliced objective (expectation over fresh random directions) is characteristic; in the
infinite-slice/sample limit, zero does imply isotropic Gaussian. The failure is finite-budget, and
it has a name: **Diaconis–Freedman** — typical low-dimensional projections of high-dimensional
data are near-Gaussian *regardless of structure*. Consequences: the power of random-slice
normality statistics against structured alternatives collapses as d grows; the loss landscape
around non-Gaussian structure is nearly flat; gradient pressure goes only where the statistic
looks. Plus Goodhart: minimizing a statistic ≠ passing an independent test. (This is also the
predecessor project's hard-won lesson — the battery never reports Epps–Pulley alone, D-010.)

**Measured (M0 battery, toy, N=9469 — numbers, not takeaways):** on Berker's own converged λ=0.02
checkpoint, at the 16-d projector output — the exact space SIGReg trains on — Epps–Pulley is
24.9 vs 1.98 for the moment-matched Gaussian null (same N, same d); worst-direction excess
kurtosis 3.54 vs 0.13. The trained space fails Gaussianity against its own null. Caveats: our
estimator settings ≠ the training loss's slice budget/normalization, so this does not contradict
their training curves — it *instantiates* the finite-battery gap. The E1 grid re-measures this
per method with CIs.

Fix-shape for D5: characteristic discrepancies (MMD with characteristic kernels, energy distance,
sliced-Wasserstein with fresh slices as an unbiased estimator of the full integral) — and
reporting isotropy with adversarial/worst-direction statistics rather than average random slices.

## 3 · Prior frameworks × Berker's angles

| framework | covers | stops short |
|---|---|---|
| IB / minimal sufficiency (Achille–Soatto '18; Federici et al. multi-view IB '20) | D1+D2 exactly; multi-view IB makes it label-free (sufficiency = shared-across-views info) | information-only: blind to geometry, hierarchy, usability — the zip objection applies to naive readings |
| Tschannen et al. '20, *On MI Maximization* (Berker's citation) | formalizes the zip point: MI is bijection-invariant, so information content cannot explain representation quality; estimator/architecture biases do the work | diagnosis, no constructive replacement |
| Dubois et al. '21 *Lossy Compression for Lossless Prediction*; Dubois et al. '22 *Idealized Representations* | cleanest existing "what we want": minimize rate s.t. lossless prediction of **every task invariant under the view group** (D1+D2 as one variational problem); '22 adds **linear** predictability of invariant tasks + dimension requirements (half of D4) | one readout tier; no hierarchy; loss-layer theory — the head/two-space question absent |
| **V-information** (Xu et al. '20, usable information under computational constraints) | the formalism for D4: I_V(Z→t) = information usable by function class V. Data-processing **fails by design** — computation creates usable information. "Represented" (I_linear high) vs "derivable" (I_V₂ high, I_linear low) = the cars example, formalized | a measurement language, not a learning objective |
| Nonlinear ICA / identifiability (Zimmermann et al. '21; iVAE; LDM '26) | the formal home of "factors of variation"; Zimmermann: InfoNCE inverts the generative process **conditional on an assumed latent marginal (uniform on sphere)** → anti-collapse terms = the assumed latent prior of an implicit generative model, which identifiability results condition on. LDM: every SSL family = alignment + an entropy estimator; identifiability up to affine | priors chosen for tractability, not truth; heads absent everywhere (LDM verified silent); hierarchy absent |
| LeJEPA optimality ('25) | the one principled *derivation* of a marginal: isotropy as the minimax choice (best worst-case downstream bias with no task model) | Berker's two flaws: enforced at the projector; enforced by a finite test battery (§2) |
| MDL / Kolmogorov structure function, sophistication | philosophical anchor for D3 + the zip case: the image is all information, zero *structure*; the witness transmits posterior-minus-prior bits, highest-value-first = rate allocation in a two-part code | uncomputable; practical shadows (hierarchical VAEs, ordered/nested latents) underdeveloped in SSL |

Also: Saunshi et al. '22 (loss-level analysis is vacuous without function-class assumptions)
is the theorem-form of "no way to guess the method ordering ex ante".

## 4 · The composed program (the sketch analogy, assembled)

> A representation is the **residual code of x under a shared world-model prior**, with **rate
> minimized subject to V-tiered lossless prediction over the invariant task family**, and a
> **declared marginal enforced by a characteristic discrepancy**.

Components: Dubois's task-family distortion (D1+D2) + Xu's usability tiers (D4) + MDL's
prior/residual structure (D3) + an explicit D5. Each component is anchored in prior work; **the
composition appears to be open** — and every component's existing theory lives at the loss layer,
so the two-space audit is this program's empirical instrument.

Formally, first cut (to attack in discussion):
minimize R(Z) (rate: I(X;Z) or H(Z)) subject to
(i) tier-1: I_lin(Z→t) ≥ I(X;t) − ε for factor tasks t;
(ii) tier-2: I_V₂(Z→t) ≥ I(X;t) − ε for derived tasks (V₂ = bounded-depth readouts);
(iii) D(p_Z, π) ≤ δ for a *declared* prior π and characteristic D.
Every SSL method is then a degenerate corner: contrastive = KDE-entropy surrogate, no tiering;
VICReg = Gaussian-moment surrogate for (iii); LeJEPA = isotropic π with a test-battery D at z;
MAE = pixel-sufficiency, no invariance tier. The audit measures which corner properties survive at h.

## 5 · Where the angles are NOT covered (candidate contributions)

1. **The two-space/head axis** — absent from every framework above; the project's core (E1/E2/E10;
   OP-17's composition note is the theory-side sketch).
2. **D3 hierarchy** — no mature SSL-theory home; E2's guillotine axis is arguably its empirical
   shadow (rate allocation across depth).
3. **D4's two-tier decodability** — appears nowhere as a *design principle*; E11's MLP-vs-linear
   gap per space is a finite-V estimate of the usability gap (I_V − I_lin).
4. **The D5 prior fork (the disagreement worth having):** Berker's sketch implies a
   **sparse/hierarchical** deviation code (the witness names FEW factors; most factors are
   prior-typical per image), while LeJEPA's minimax argument yields **isotropy** precisely by
   refusing that structure. Sparse-factorial vs isotropic marginal is a testable modeling fork
   settled by neither side's theorem.

## 6 · Agreed next steps

1. Quote-verification pass on the anchors (Dubois '21/'22; Xu '20; Achille–Soatto '18;
   Zimmermann '21; Tschannen '20; + LDM/LeJEPA re-reads) — then this note's §3 table gets
   citation-grade.
2. Discussion over this note: attack §4's formalization; have the §5.4 prior-fork argument.
3. Only after that: decide whether the composed program becomes the project's framework paper
   angle (the E-cards already generate its empirical section).
