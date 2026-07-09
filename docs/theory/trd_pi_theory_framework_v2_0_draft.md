# TRD-π — Tiered Rate–Distortion with a Declared Prior

**A normative framework and audit protocol for self-supervised representation learning**

**PROPOSED v2.0 DRAFT (Claude, 2026-07-09) — for discussion with Berker; NOT agreed, NOT locked.
`trd_pi_theory_framework_v1_3.md` remains the document of record until this draft is reviewed.
Nothing in this file creates or modifies an AGREED TAKEAWAY; measured numbers cited here are
either USER-APPROVED takeaways (cited by ID) or raw numbers (flagged raw).** Self-sufficient by
design: desiderata, results, proofs (Appendix A), constants (Appendix B), protocol, predictions,
positioning, limitations, and open problems are all in this file. Review lineage lives in
`TRD_PI_PROVENANCE.md`; the delta against v1.3 is itemized in Appendix D.

Claim classes used throughout: **[T]** theorem/proposition, proof in hand (Appendix A) · **[L]**
lemma, proof in hand · **[PS]** proposition-sketch (strategy clear, bookkeeping unwritten; outside
the theorem core) · **[F]** fact imported from cited work ([F-verified] = quote-verified against
the source in `verification/`) · **[M]** measured in this project's audit (provenance named
per use; USER-APPROVED takeaways cited by ID) · **[D]** design principle · **[P]** falsifiable
prediction · **[C]** conjecture, filed as an open problem.

---

## 0 · Problem and thesis

SSL objectives are tuned to benchmarks, and their loss values do not order methods by downstream
quality across methods — only within one method's own hyperparameter sweep. Two mechanisms explain
this: the **wrong functional** (losses admit useless minimizers — BYOL's objective admits collapsed
solutions, and its minus-target-network / minus-predictor ablations land at 0.3% / 0.2% top-1
against 0.1% chance [F-verified: Grill et al. '20 §3.2, Table 5(b) rows 6–7; the paper reports no
loss value at the collapsed cells, so the claim is "the objective admits collapse and the ablations
collapse", not "the loss was measured low"]; Saunshi et al. '22 give the theorem form — in the
(near-)disjoint-augmentation regime, any downstream bound monotone in the loss value is vacuous at
every loss value, Cor. 4.1 [F-verified, regime-scoped]) and the **wrong space** (losses live at the
projector output z while probing happens at the backbone h).

Two empirical hooks make this concrete, one per mechanism, both from this project's audit:

- **The honesty gap [M].** On a converged checkpoint whose training statistic reports isotropy —
  the 16-d projector cell, where low width makes an all-direction search effectively exhaustive
  (scoped against the method's best 64-d configuration in R6) — the search finds Epps–Pulley 24.9
  against a moment-matched null of 1.98 and worst-direction excess kurtosis 3.54 against 0.13
  (N = 9469): the trained test reads clean while the marginal is not (M0 battery;
  E01-T4/T9, USER-APPROVED).
- **One statistic, two jobs [M — E10-T1, USER-APPROVED].** The *same* deployed statistic changes
  its job with the dimension of the space it is applied to: at K = 16, 84% of the trained
  deviation is genuine distribution-shape work; at K = 384–512 it is a pure first-two-moment /
  degeneracy meter — every trained 512-d embedding reads exactly like its covariance-matched
  Gaussian twin (shape residue ≈ 0), and stability of training is set by *placement* (the
  projector is a measured gradient sink; unbuffered placements carry a finite-time instability).
  R4 now derives both halves (the audit-level and the loss-level scaling laws, §3-R4), and §3-R6f
  carries the placement dynamics.

**Thesis.** TRD-π poses representation learning as **tiered rate–distortion against a declared
prior**: minimize description length measured against a declared codebook family, subject to
linear decodability of a declared battery of invariant factors, with honesty enforced by a
worst-direction audit held outside the loss, in the space where downstream probing actually
happens.

The stance in one sentence: **we impose the demand and instrument the discovery.** The prior
demands that a calibrated marginal and (in the finite-ν case) a preferred basis *exist*; whether
the data supply them is measured, never assumed — by the pilot-read selection stage, the seed
null, and an audit the loss cannot see (§4, §5).

What the framework buys is a contribution triple: (i) a normative program in which all six
desiderata (D0–D5) are jointly statable and jointly operational — we know of no alternative frame
with this property (R1, §3-R7; positioning in §8); (ii) impossibility, blind-spot, and scaling
results that explain current practice quantitatively (R2, R4, R6); (iii) a falsifiable
audit-and-estimation protocol with pre-registered predictions (§5–§7).

Four layers, one job each: **normative** (this program — what representations should be),
**descriptive** (latent distribution matching — what existing methods are: corners of the same
Lagrangian, §2), **conditions** (identifiability — when minimizers invert nature, R3),
**measurement** (the two-space audit, R4–R6, §5). The submission spine is the measurement layer
(R1, R4–R6) with the prior fork (R8) as a conditional extension — adopted from review round 2 and
retained here; R8's strengthened status (§3-R8) does not change the ordering, it derisks the
extension.

---

## 1 · Desiderata and their carriers

**Standing qualifier** (single home; everything below inherits it): sufficiency and invariance are
relative to a declared nuisance mechanism G and task family T, never absolute. G is a group action
only in the clean setting; in practical SSL it is a stochastic, generally lossy and non-invertible
transformation family (crop, blur, masking, jitter), formalized through the content/style
decomposition of von Kügelgen et al. '21. "Maximal invariant" is used only in the group/equivalence
case; "invariant content variable U" otherwise; every statement below survives the substitution.
G is the framework's one load-bearing ungrounded input (what augmentations do and don't determine
is its own literature). The framework converts "choose a loss" into "declare **(G, battery F,
π-family, probe tiers V, budget chain B, resolution σ)**" — the factor battery F is part of the
declaration (R7a), not a derived object — of which four are estimated or audited downstream, and
G and F are inherited from the deployment.

**Label discipline** (single home): TRD-π is self-supervised in its training objective and
factor-aware in its governance — F-labels enter at selection (§5, Stage 1) and evaluation/audit
(R7c, E5), never in L. Mainstream SSL already consumes the same labels at model selection,
silently; TRD-π pre-registers that use and prices it.

Each desideratum below states the want, then its formal carrier in §3.

**D0 · Prior-relative residuality.** Spend rate on the case-specific information a sample
transmits, not on re-encoding what is already typical of the domain. *Carrier* (R1, [T]):
E_x KL(q(z|x) ‖ π) = I(X;Z) + KL(q_Z ‖ π). I(X;Z) is instance-specific information measured
against the encoder's own aggregate marginal q_Z — mutual information does not know the external
prior — while KL(q_Z ‖ π) is the calibration gap between that aggregate and the declared codebook;
the decomposition forces calibration into the open rather than leaving it implicit. π thereby has
two pulls, **descriptive** (a model of the embedding shadow natural data induces) and **normative**
(a geometric commitment for probing and identifiability, R8), held apart by the declare → select →
train → estimate → audit pipeline (§5). The restriction to invariant content is D1's job, not D0's.

**D1 · Sufficiency.** Every task in the declared G-invariant family remains solvable from the
representation. *Carrier*: the reduction of G-invariant tasks to the invariant content U and its
declared battery F (R7a; deterministic labelings factor exactly through the maximal invariant,
stochastic tasks factor at the level of conditional laws — the two cases are kept distinct per the
verified Lemma 5 / Lemma 6 split of Dubois et al. '21); in training, the alignment surrogate
(R1, level ii); at selection and audit, tier-1 decodability of F (R1 level iii; R7c).

**D2 · Minimality.** Among sufficient representations, prefer minimal ones — invariance to
nuisance is equivalent to information-minimality among sufficient representations (Achille &
Soatto '18, Prop. 3.1: "invariant (maximally insensitive)" ⇔ minimal, with the worst-case-nuisance
construction requiring y discrete and a residual ε ≤ H(y|x) [F-verified]). A second, rate-free
grounding is imported at R7: invariance is also the condition that tames the ERM argmin set in
Dubois et al. '22's sample-optimality theorem — worst-case generalization of arbitrary probes,
with no information-theoretic input [F-verified]. Two independent groundings, one desideratum.
*Carrier*: the rate term, measured against the declared π (R1), which is what makes minimality
operational rather than bijection-blind.

**D3′ · Hierarchy (conditional).** If deployment declares a budget chain B = (m₁ < m₂ < …), the
representation should admit a flag — nested subspaces V_{m₁} ⊂ V_{m₂} ⊂ … — each solving the
tier-1 program at its budget. Absent a declared B, D3′ imposes nothing; with a single budget the
meaningful object is a subspace (Grassmannian) and any within-subspace permutation is gauge.
"Importance" is not a primitive: it is the shadow price of a factor under a task prior; ties
collapse the flag into blocks. The unconditional residue — "admits a near-optimal flag" — is a
rotation-invariant property of the information spectrum. *Carrier*: R7d (prefix-nested program;
exact PCA equivalence in the linear-Gaussian case).

**D4 · Tiered accessibility.** Factor tasks should be linearly decodable (tier 1, probe class V₁);
derived quantities need only bounded-depth decodability (tier 2, V₂ ⊇ Φ∘Lin). Sufficiency does not
imply linear accessibility; the gap Δ = I_{V₂} − I_{V₁} (Xu's V-information) is measured, never
trained, and is well-defined as a nonnegative quantity only under a marginal-predictor condition
now stated at R7c [L]. The forward direction — tier-1 factors ⇒ tier-2 derived tasks — is now a
proposition with an explicit modulus (R7b, [T]); the converse fails by construction (V-information
is designed to violate data processing). *Carriers*: R7b–c; R6b (the two-space prediction for Δ);
the affordability floor R7e ([T]: declaring a finite battery is what makes tier-1 compatible with
compression at all).

**D5 · Grounded marginal — two numbers, never one.** The embedding marginal is governed by a
declared prior family (here ∏ₖ t_ν, unit variance, ν ∈ (4, ∞]; the isotropic Gaussian is the
light-tailed ν → ∞ corner). Scale convention: unit coordinate variance is pinned to the dithered
variable h̃ at the declared σ, after the declared normalization map; audit-side per-coordinate
scale estimates are nuisance standardization (§5, Stage 3). Two distinct members appear and are
never conflated: **ν_train**, the member inside the loss, frozen before final training by the
pre-registered selection stage (§5, Stage 1); and **ν̂**, the member the audit tests against,
estimated after training on a held-out split from loss-independent statistics. The gap between
them (reported on the 1/ν scale) is itself a diagnostic: did training land where it aimed. Fit is
audited by a characteristic, worst-direction-reported statistic computed outside the loss, against
a parametric-bootstrap null drawn from the fitted declared prior (§5, Stage 4). The sparse
(finite-ν) variant is gated to the complete regime (R3). Rate role and audit role are never played
by the same estimator (R1). *Carriers*: R1 (identity), R4 (why in-loss enforcement is blind — now
including the loss-level scaling law), R5 (the audit that isn't), R8 (the fork), §5 (the pipeline).

### The police-sketch analogy — the framework's conceptual heart

Every element of the story is an existing formal object; none required new machinery.

| Sketch element | Formal object |
|---|---|
| Drawer's general knowledge of faces | π — declared embedding-shadow of the background prior |
| Witness's transmitted details | E_x KL(q(z\|x) ‖ π) |
| Genuinely case-specific content | I(X; Z) |
| Drawer/witness mis-calibration | KL(q_Z ‖ π) — audited, never assumed zero |
| Pose, lighting, viewpoint | the declared nuisance mechanism G |
| The criminal's identity | invariant content U (maximal invariant in the group case) |
| Readable at a glance vs on study | tier 1 (V₁) vs tier 2 (V₂); gap Δ |
| Coarse-to-fine sketching order | D3′'s flag, iff a budget chain is declared |
| "Is the sketch actually right?" | the audit, outside the loss (§5) |

### The spine — desiderata → results → existing practice

| D | The want | Formal carrier | Where existing methods stand (§2) | What TRD-π adds |
|---|---|---|---|---|
| D0 | rate only for case-specific content | R1 identity | neural compression learns π jointly with the code (descriptive only); SSL leaves calibration implicit | declared family, pre-registered member, audited calibration gap |
| D1 | invariant tasks stay solvable | R7a reduction; alignment surrogate (R1) | every alignment loss is the same surrogate, unaudited | battery F declared; surrogate slack priced at selection and audit |
| D2 | no rate beyond sufficiency | R1 rate term at declared σ; second grounding via W_n (R7) | per-method entropy surrogates, mutually incomparable | closed-form cross-entropy rate against the declared π |
| D3′ | budgeted nesting on demand | R7d | Matryoshka as an engineering trick | conditional desideratum + linear-Gaussian equivalence [T] + OP-b |
| D4 | factors readable at a glance | R7b [T] + R7c Δ protocol + R7e floor [T] | probing literature supplies budget matching; '22 fixes one probe family per guarantee | Δ measured, never trained; composition modulus proven; affordability floor proven |
| D5 | the marginal you claim is the marginal you have | R4 (incl. loss-level scaling), R5, R8, §5 | mean-slice / moment surrogates inside the loss (blind: R4) | worst-direction audit outside the loss; ν_train / ν̂ discipline |

### Desiderata × results matrix (✓ = the result carries or protects the desideratum)

|  | R1 | R2 | R3 | R4 | R5 | R6 | R7 | R8 |
|---|---|---|---|---|---|---|---|---|
| D0 | ✓ |  | ✓ |  |  |  |  | ✓ |
| D1 |  |  |  |  |  | ✓ | ✓ |  |
| D2 | ✓ |  |  |  |  | ✓ | ✓ |  |
| D3′ |  | ✓ |  |  |  |  | ✓ |  |
| D4 |  | ✓ |  |  |  | ✓ | ✓ |  |
| D5 | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |  | ✓ |

---

## 2 · Setup, notation, and the map of existing methods

x ∼ p(x); G the declared nuisance mechanism; U the invariant content (maximal invariant when G is
a group/equivalence). Backbone h = f(x) ∈ ℝ^d; head z = g(h) ∈ ℝ^{d′}; dithered backbone
h̃ = h + σε, ε ∼ N(0, I) (σ = declared resolution; every rate/marginal number carries its σ; only
equal-σ comparisons are meaningful). π ∈ P the declared prior family; ν_train vs ν̂ per D5. F the
declared factor battery. V₁ = linear probes, V₂ = bounded-depth probes; I_V = predictive
V-information (Xu et al. '20). q_Z = the embedding marginal. C⁽⁴⁾ denotes the fourth-cumulant
tensor. K is defined once: the audited ambient dimension of whichever space a statistic lives in —
K = d at h, K = d′ at z. N denotes the sample size a statistic is computed from (batch size in a
loss, manifest size in the audit) — the R4 scaling laws depend on (K, N) jointly and the two
deployments differ by orders of magnitude in N. s★ denotes generative source latents (R3 only;
never a learned quantity). Density-level hygiene: all density-level statements (rates, KLs,
entropies) for deterministic encoders are understood after the declared dither σ; the un-dithered
pushforwards themselves may be singular (R6d).

**Existing methods as corners of the program** (descriptive layer). Attribution, scoped per the
verification pass [F-verified]: the latent-distribution-matching reading (Mikulasch & Zenke '26)
itself derives VICReg, SimCLR, CPC, BYOL/SimSiam, and JEPA (alignment to an assumed latent model +
an entropy estimator; DINO-family derived only up to a "related" simplex model, with the paper's
own caveat that real DINO "might not directly be related to any entropy estimator"). The LeJEPA,
Barlow Twins, DINO/SwAV-as-Sinkhorn, and MAE rows below are **our extensions of that reading**,
not derivations found in the paper. Reading rule: the table is an idealized, population-level
reading of these objectives under common normalizations, not a claim that the implemented losses
literally optimize these exact divergences.

| Method | Alignment | Rate/entropy surrogate | Implied π | Space | Residual symmetry at optimum |
|---|---|---|---|---|---|
| LeJEPA | prediction | mean-random-slice Epps–Pulley | isotropic Gaussian | z (proj. dim 64 ablated best) | full O(K) |
| VICReg | MSE | variance hinge + covariance penalty | 2nd-moment factorial | z (8192 expander → R6d) | signed permutations (setwise); per-solution gauge in equal-variance blocks |
| Barlow Twins | cross-corr. | correlation → I | whitened | z | full O(K) at exact optimum |
| InfoNCE / uniformity | contrastive | KDE-entropy / uniformity | uniform on sphere | z | full O(K) |
| BYOL / SimSiam | prediction | implicit (EMA/predictor — implicit spectral whitening, Tian et al. '21; conditional-entropy plugin in the LDM reading) | implicit / empirical | z | full O(K) (alignment part) |
| DINO / SwAV | cross-view prototype assignment | Sinkhorn equipartition / centering + sharpening | (near-)uniform over prototypes — a declared prior over assignments, arguably the cleanest pre-existing instance of the declared-prior reading | z (prototype head) | prototype permutations |
| MAE | pixel reconstruction | — | pixel-space generative; no invariance tier — listed for contrast, not as a corner of the invariant program | input | — |

Every corner is a projection of the program: fix π implicitly, drop the audit, collapse the tiers,
regulate z instead of h. TRD-π is what remains when nothing is dropped, and it differs from every
corner on four labeled axes: **(1)** worst-direction pressure where they average (R4–R5); **(2)** a
declared family with pre-registered ν_train and separately audited ν̂ where they imply a fixed
target (R8, §5); **(3)** placement at h, where probing happens, where they regulate z (R6) — now
carrying explicit stability premises (R6f, §4); **(4)** an audit held outside the loss where they
trust the training statistic (§5).

---

## 3 · Results

### R1 · The rate–prior identity, the canonical objective, and the compression reading — serves D0, D2, D5

**[T] (Rate–prior identity.)** For any encoder q(z|x) and prior π (densities defined):
E_x KL(q(z|x) ‖ π) = I(X; Z) + KL(q_Z ‖ π). *Proof* A.1.1 (add and subtract log q_Z(z); Hoffman &
Johnson '16; the ≥-form is the variational information-bottleneck bound, Alemi et al. '17).

**[T] (Companion identity.)** E[−log π(h̃)] = h(q_h̃) + KL(q_h̃ ‖ π) — the cross-entropy rate term
already contains a compression part (the differential entropy of the dithered marginal) and a
calibration part. *Proof* A.1.2. This is what licenses reading §4's marginal term as re-weighting
calibration pressure rather than introducing a new quantity.

**[T] (Closed-form dithered rate.)** With the dithered encoder q(h̃|x) = N(f(x), σ²I):
Rate(σ) = const(σ) + E[−log π(h̃)] — the rate term is the cross-entropy of dithered embeddings
under the declared prior. *Proof* A.1.3. Instances: π = N(0, I) gives the embedding-norm penalty
E‖h̃‖²/2; π = ∏ t_ν (unit variance) gives ((ν+1)/2)·Σₖ log(1 + h̃ₖ²/(ν−2)) — up to additive
constants, unit-variance coordinates assumed — a log-growing robust penalty whose score/influence
function ψ(u) = (ν+1)u/((ν−2)+u²) redescends. The prior's tails and the penalty's influence
function are the same object: the declared codebook made tangible. (Dithered-quantization
classics: Ziv '85; Zamir & Feder.)

**Compression provenance.** The dithered rate term is precisely the learned-entropy-model
objective of neural compression (Theis et al. '17; Ballé et al. '17, '18; Minnen et al. '18), with
additive noise as the standard quantization proxy. The contrast is the point of D0's "two pulls":
compression learns π jointly with the encoder — a purely descriptive prior that chases the code —
whereas TRD-π **declares the family, pre-registers the trained member, and audits the fit**,
preserving the normative pull. TRD-π's rate machinery is therefore not exotic; its governance of π
is what is new.

**Canonical objective decomposition.** In the ideal program, anti-collapse is not a primitive: the
tier-1 distortion constraint forbids collapse and rate-minimization supplies the collapse
pressure. Three levels, never conflated. **(i)** In the ideal constrained program, tier-1
sufficiency is enforced directly on the declared battery F. **(ii)** In the self-supervised
implementation, F labels are unavailable during training, so that constraint is replaced by view
alignment — a label-free surrogate (Federici et al. '20) whose validity is itself conditional:
alignment stands in for sufficiency only jointly with the marginal term (alone it is satisfiable
by collapse, §4) and only insofar as view-shared content covers U (the G declaration again).
**(iii)** The battery enters through selection and audit (§5 Stage 1; R7c; E5) — never through the
unsupervised loss, unless factor labels are in fact available, in which case a direct tier-1 term
is admissible. Hence: **L = view alignment at z (surrogate for the tier-1 constraint) +
cross-entropy rate at h̃ + an entropy-side marginal surrogate (compensating the
alignment–constraint slack) — with the audit entirely outside L.** This derives the
latent-distribution-matching reading of existing methods (alignment + entropy estimator) instead
of assuming it — with the LDM attribution scoped as in §2; the equilibrium among the three forces
is made explicit in §4 ("who fights whom").

**[D] Role split**: any consistent surrogate may shape the optimum (rate role); the claim
"marginal ≈ π" is made only by a characteristic, worst-direction, held-out-settings audit (audit
role). The split does not prevent gaming; it makes gaming visible by separating the training
objective from the held-out audit — a tripwire, not a guarantee.

### R2 · What objectives cannot select (symmetry inventory) — serves D3′, D4, D5

**[T]** The orthogonal-symmetry inventory of standard objectives at the minimizer level
(*proofs* A.2):

1. **Full O(K)**: pairwise-distance alignment + any rotation-equivariant discrepancy to a
   rotation-invariant π (InfoNCE/uniformity with the sphere-uniform target, BYOL-style alignment,
   LeJEPA's sliced discrepancy to an isotropic Gaussian). Minimizer sets are closed under global
   rotations.
2. **Full O(K) for Barlow Twins at its exact optimum**: correlation = I is whitening, and
   whitening is rotation-invariant — the "decorrelation method" is, at optimum, exactly as
   basis-blind as the isotropic methods.
3. **Signed permutations — and only signed permutations — for VICReg** at the level of the
   minimizer family: they are the sole orthogonal maps conjugating every admissible diagonal
   covariance to a diagonal one. A given minimizer with spectrum diag(λ) admits additional gauge —
   rotations inside its equal-λ blocks, the stabilizer ∏ O(m_j) of that solution — but this
   freedom is solution-dependent, not a symmetry of the loss. Remark (counterexample that killed
   the naive claim): VICReg's hinge only enforces per-coordinate std ≥ γ; take a minimizer with
   Cov = diag(1, 4) — a 45° rotation produces off-diagonal covariance and re-activates the
   penalty, so the minimizer set is not O(K)-closed, and it is not {Cov = I} either.

**Takeaway** (scoped to population minimizers). At the level of population minimizer symmetries,
**no standard objective by itself identifies sparsity, importance ordering, or basis structure
beyond at most second-order decorrelation with degeneracy gauge** — anything more in a trained
model is architecture/optimizer implicit bias, not the loss; whether implicit bias already
delivers axes in practice is an empirical question decided by the seed null (§6.3). Tier-1 linear
decodability is itself rotation-invariant, so isotropy is safe for D4; D3′ (when active) and
monosemantic-D5 require an explicit symmetry break: a non-rotation-invariant π (R3) or prefix
nesting (R7d). Scope: this is an identifiability-from-the-objective statement about the loss
landscape, not about SGD dynamics.

### R3 · When minimizers invert nature: affine → signed permutation — serves D0, D3′-frame, D5

**[T]** Assume **(A1)** generative source latents s★ ∼ ∏ₖ pₖ mutually independent, at most one
Gaussian, x = g★(s★) injective; **(A2)** the method achieves affine identifiability
f(x) = A s★ + b, A invertible — the conclusion of the LDM predictive-SSL theorem under its
assumptions (theorems stated over the predictive factorization ∏ₜ P(z_t|z_{:t}); temporal in all
of that paper's statements and experiments [F-verified]), of Roeder et al. '21-style linear
identifiability of discriminative families, of the nonlinear-ICA line TCL/PCL/GCL (Hyvärinen &
Morioka '16, '17; Hyvärinen et al. '19) — the actual ancestry of "affine identifiability from
self-supervision" — or of Zimmermann-/iVAE-class results under theirs (Zimmermann's theorems
condition on an assumed latent marginal — uniform on the sphere or a convex body — and on the
loss's conditional form matching the generative conditional [F-verified]); **(A3)** the objective
drives the learned marginal to be factorial and non-Gaussian. Then **A = ΠΛ**: recovery up to
permutation and per-coordinate sign/scale. Engine: Comon '94 / Darmois–Skitovich. *Proof* A.3.1.

**[L] (Pairwise sufficiency in this regime.)** If y = As★ as above (A invertible; sources
independent, nondegenerate, with finite variances; at most one Gaussian) and the coordinates of
y are pairwise independent, then A = ΠΛ. *Proof* A.3.2 (Darmois–Skitovich applied per pair:
every non-Gaussian source can load on at most one output row; pairwise independence implies
uncorrelatedness, which forces the (≤1) Gaussian column to a single row as well; invertibility
does the rest). The finite-variance and nondegeneracy hypotheses are load-bearing (review):
Cauchy sources disable the covariance step, and a degenerate source admits shear
counterexamples; both are satisfied trivially by the declared t_ν family (ν > 4). Hence pairwise-HSIC
penalties are principled surrogates at the optimum, inside the complete linear-ICA regime.
Outside that regime, pairwise HSIC is diagnostic, not sufficient. Quantitative robustness
(ε-independence ⟹ δ(ε)-close to a signed permutation) is open [C → OP-d].

**Takeaways.** (1) **The inversion**: the isotropic Gaussian is the unique marginal that forfeits
basis identifiability (the classical ICA obstruction) — so minimax probe-neutrality and
identifiability pull in provably opposite directions. The fork's two arms carry different
epistemic weight — this theorem versus R8(a); the statuses are stated once, at R8. (2) **Regime
gate**: the theorem lives in the complete regime (embedding width ≥ factors active at this
budget). Undercomplete, the rate-optimal code superposes (Elhage et al. '22) and a factorial push
degenerates into sparse-coding pressure (Olshausen & Field '96 territory) — the sparse machinery
is deployed only after §6.1's regime check. (3) For dependent factors, the primary fallback is
sparsity-based identifiability (Lachapelle et al. '22), not independence.

### R4 · Random-slice blindness, made quantitative at both deployments — serves D5

The blindness of averaged sliced statistics is now derived twice, once per deployment: at **audit
N** (linear-in-cumulant statistics, the Beta attenuation) and at **training N** (the quadratic
CF-distance channel theorem — new in v2.0, formalizing what was the [PS] "signal reading"). The
two laws share the attenuation engine and differ in the exponent; together they predict, and the
measured cells confirm [M], that the same statistic does different jobs at K = 16 and K = 512.

**(a) [L] (Rank-k attenuation — audit level.)** If all non-Gaussian structure of Z (E Z = 0,
Cov Z = I) lies in a k-dimensional subspace with independent Gaussian complement, then for θ
uniform on S^{K−1}: κ_r(θᵀZ) = α^r κ_r(v_θ) with α² ∼ Beta(k/2, (K−k)/2), so
E[α⁴] = k(k+2)/(K(K+2)) and generally E[α^{2m}] = ∏_{j<m}(k+2j)/(K+2j). *Proof* A.4.1 (cumulant
additivity/homogeneity + Beta moments).

**(b) [T] (Tensor-general form; no subspace assumption.)** E_θ[κ₄(θᵀZ)] =
3·Σ_{ij} C⁽⁴⁾_{iijj} / (K(K+2)), a trace-type contraction of the fourth-cumulant tensor C⁽⁴⁾,
while the worst slice sup_θ |κ₄(θᵀZ)| is the tensor's injective norm (absolute values — for a
platykurtic defect the unsigned sup can sit near 0); the mean/max ratio reaches
K(K+2)/3 for a rank-1 leptokurtic defect. *Proof* A.4.2 (uniform fourth-moment identity
E[θᵢθⱼθₖθ_l] = (δδ-symmetrization)/(K(K+2)) contracted against the symmetric tensor).

**(c) [T] (Channel decomposition of weighted sliced-CF statistics — loss level; NEW.)** Let
T_w(s; N) = N ∫ |ψ̂_N(t) − e^{−t²/2}|² w(t) dt be the weighted CF distance of N iid draws of a
real variable s against the standard normal (ψ̂_N the empirical CF; w ≥ 0 integrable — the
deployed SIGReg is the member with w(t) = quadrature × e^{−t²/2} on [0, 3], 17 knots, and the
per-slice loss is the S^{K−1}-average of T_w(θᵀZ; N) **without per-slice standardization**, so
slice means and variances are constrained, not normalized away). Then (*proofs* A.4.3; constants
Appendix B):

1. **Exact mean.** E[T] = N ∫ |ψ(t) − φ(t)|² w(t) dt + ∫ (1 − |ψ(t)|²) w(t) dt, where ψ is the
   population CF of s and φ(t) = e^{−t²/2}. The first term is the population signal (∝ N); the
   second is the finite-sample noise floor (N-free).
2. **Floor.** s ∼ N(0,1) ⇒ E[T] = ∫(1 − e^{−t²}) w(t) dt =: c₀(w), independent of N and K.
   Shipped statistic: c₀ = 1.053.
3. **Collapse value.** s degenerate at 0 ⇒ T = N·∫(1 − φ(t))² w(t) dt =: N·c_∞(w). Shipped:
   c_∞ = 0.402; at training N = 256, N·c_∞ ≈ 103. Not a supremum (review correction): among
   degenerate laws, mass at 0 is the *smallest* reading — a point mass at offset c rises toward
   N·∫(1 + φ²)w ≈ 9.8·N·c_∞ as |c| grows. Trained collapse cells nevertheless pin near N·c_∞
   because the mean channel's 1/K attenuation (item 5) drives typical slices of a bounded
   degenerate cloud to the mean-zero corner — the constant is the collapse *reading*, not a
   ceiling theorem.
4. **Channel constants — signal and floor shift.** For a slice with small cumulant deviations
   from N(0,1) — mean m, variance 1 + ε, skewness γ, excess kurtosis κ — the expected statistic
   splits into an N-scaled second-order signal and an N-free FIRST-order noise-floor shift
   (review correction: the floor shift was dropped in this theorem's first draft):
   E[T] − c₀ ≈ N·[C₂ m² + C₄ ε² + C₆ γ² + C₈ κ² + same-parity cross terms] + F(ε, κ),
   F(ε, κ) = C₂·ε − 0.0402·κ (odd cumulants leave |ψ|, hence the floor, unchanged at first
   order). C_{2r}(w) = (1/(r!)²) ∫ t^{2r} e^{−t²} w(t) dt; shipped values: C₂ = 0.482,
   C₄ = 0.121, C₆ = 0.0223, C₈ = 0.00325 — the signal term is ~150× more sensitive to a unit
   mean defect than to a unit kurtosis defect *before* any slicing attenuation. Because the
   floor shift is first-order in the *attenuated* κ̃₄(θ) (∝ 1/K) while the signal is
   second-order (∝ 1/K²), the floor shift dominates the shape response beyond
   K* ≈ 9NC₈κ/C₄ (≈ 186 at N = 256, κ = 3) — and its sign for a leptokurtic defect is
   *negative*: at large K the per-step statistic reads a pure shape defect slightly BELOW the
   Gaussian floor.
5. **Slicing attenuation per channel.** For θ uniform on S^{K−1} and Z with mean μ, covariance Σ
   (eigenvalues λᵢ, mean λ̄, spread s² = K⁻¹Σ(λᵢ − λ̄)²), and shape defects as in (a):
   E_θ[m_θ²] = ‖μ‖²/K (mean channel: 1/K);
   E_θ[ε_θ²] = (λ̄ − 1)² + 2s²/(K+2), exactly (variance channel: **global scale unattenuated**,
   anisotropy 1/K);
   E_θ[κ_θ²] = κ²·E[α⁸] ~ κ²(k/K)⁴ for a rank-k kurtosis defect, and
   E_θ[κ_θ²] ≈ 9κ²/(K+2)² for a factorial (all-coordinate) defect (shape channel: 1/K² to
   1/K⁴).
6. **Anti-CLT regime.** If Z is concentrated near an r-point set or a low-dimensional set with
   r ≪ N (clustered/degenerate clouds), typical slices are r-summand mixtures: no
   Gaussianization occurs, and T = Θ(N) in most directions, K-independent. Two mechanisms, kept
   distinct: well-separated clusters read Θ(N) through the r-atom mixture's CF; a degenerate
   (collapsed) cloud reads ≈ N·c_∞ through its near-zero slice variances (item 3).

**Corollary (one statistic, two jobs) [T + M].** As K grows at fixed defect and fixed N, the
shape channels vanish at rates K^{−2}–K^{−4} while the scale/degeneracy channel is K-independent:
the statistic degrades continuously from a (weak) shape test into a pure first-two-moment /
degeneracy meter. Per-step detectability, shipped constants (exact-CF values, review-corrected):
a factorial κ = 3 kurtosis defect moves the per-step statistic by +0.160 at (K = 16, N = 256)
(signal +0.179, floor −0.019) against an H0 sd of ≈ 0.19 — under one sd per step even at K = 16
— and by **−4.5·10⁻⁴** at (K = 512, N = 256): three orders below the H0 sd 0.08 and *negative*
(the floor-shift regime of item 4). At large K the per-step statistic is not merely blind to
shape — its marginal response to a leptokurtic defect points the wrong way. The same defect at
audit N = 50k is ~200× louder in the signal term while the floor terms are N-free. Measured on trained networks
[M — E10-T1, USER-APPROVED; results/diag/sigreg_ref.csv]: at K = 16, 84% of the trained
deviation is shape work (independently matching the audit-N decomposition's 87–94%, E01-T9);
at K = 512, every trained embedding's statistic equals its covariance-matched Gaussian twin's
(shape residue ≈ 0 ± MC noise) — the entire trained value is covariance information. MC
verification of every row: Appendix B.

**(d) Consequences.** Mean-random-slice surrogates — the deployed SIGReg (which documentedly
replaces its theorem's max over directions with an average to avoid sparse gradients), and the
whole EP ≈ sliced-MMD family (KerJEPA; fixed-kernel power decay, Ramdas et al. '15) — are
**structurally under-pressured on low-rank shape defects, at any λ**: a global loss weight
multiplies all channels equally and cannot repair the shape:moment sensitivity ratio, which falls
like K^{−2}–K^{−4} (this is the theorem behind the measured λ-rescue failures, §3-R6f). What a
finite-budget audit can actually detect is calibrated, not assumed: §6.0 and OP-h. M0 arithmetic:
worst-direction κ₄ = 3.54 at K = 16 predicts mean-slice κ₄ ≈ 0.037 if the defect were rank-1; the
aggregate EP = 24.9 is consistent with rank > 1 — the defect-rank estimator k̂ (fit the slice-κ₄
spectrum to the Beta prediction of (a)) decides [P]. Interventions that DO restore shape pressure
at large K are enumerated, each a different declared loss: per-slice standardization before the
CF (kills the moment channel, isolates shape), larger per-step N, targeted/adversarial slices
(R5's bank), or imposing the constraint on a whitened space [D].

### R5 · The audit that isn't blind — serves D5

**[T]** The population max-sliced discrepancy sup_θ d₁(θ♯q, θ♯π) is characteristic (Cramér–Wold)
and, for d₁ = W_p, a metric (Deshpande et al. '19).

**[D]** Train the mean-slice term (bounded, stable gradients) plus a soft-max over a persistent
adversarial direction bank — formally a low-capacity instance of a learned deep-kernel two-sample
statistic (Liu et al. '20) — with fresh random slices for exploration. The bank raises the cost of
hiding a rank-k defect from O(K^r) to O(bank lag); it does not eliminate hiding (minimax chasing
is possible [C → OP-g]; mitigations: bank momentum, random restarts, and the held-out audit as
referee).

**Principle (evidence, not certificates — hard reporting rule).** The population supremum is
characteristic; finite banks and finite searches are capacity-limited audits whose failures are
evidence, not certificates of equality. The audit therefore reports "the worst witness found under
a declared search budget," together with a matched-search null generated from the fitted declared
prior (§5, Stage 4) — and never reports "passed." Small discrepancies are evidence of fit under
that budget. (Required for self-consistency: the framework's own critique of "failing to reject ≠
accepting" applies to its audit too.)

### R6 · Two spaces: what crosses the head — serves D1, D2, D4, D5

Let z = g(h). **(a) [F]** I(X; z) ≤ I(X; h): sufficiency established at z holds at h for free —
the only desideratum that crosses the head automatically, in the right direction. **(b) [F]**
Tier-1 at z is only tier-(lin∘g) at h: training accessibility at z predicts a nonzero Δ at h.
**(c) [T]** Let g be bi-Lipschitz with constants (ℓ, L) onto its image, and let π be supported on
Im(g) (automatic when dim z = dim h and g is a bi-Lipschitz bijection; for dim z > dim h no
full-support π qualifies — see (d)). Then W_p(q_h, (g^{-1})♯π) ≤ ℓ^{-1} W_p(q_z, π). *Proof*
A.5.1. Remark (Kirszbraun): dropping the support hypothesis, g⁻¹ extends from Im(g) to a
(1/ℓ)-Lipschitz map G on all of ℝ^{d′}, giving W_p(q_h, G♯π) ≤ ℓ^{-1}W_p(q_z, π) for **some**
extension-dependent pullback G♯π — proximity transfers, the target's identity does not. With an
unconstrained head, D(q_z, π) = 0 constrains q_h essentially not at all. **(d) [T] (Expander
singularity.)** If dim(z) > dim(h) with a deterministic **locally Lipschitz** head g (any MLP
qualifies), q_z is supported on a set of Hausdorff dimension ≤ dim(h) < dim(z), hence
Lebesgue-null: KL(q_z ‖ N(0, I)) = +∞ and "z ∼ isotropic Gaussian" is unsatisfiable in principle —
only moment/slice surrogates can be driven down. Dither repairs well-posedness. (The Lipschitz
hypothesis is load-bearing: merely measurable maps can have space-filling images.) *Proof* A.5.2.
**(e″) [T, both cases] (i — dimension-preserving entropy transfer.)** Let q_h be absolutely
continuous with finite entropy, dim z = dim h = d, g L-Lipschitz (hence a.e. differentiable,
Rademacher), injective on supp(q_h) up to null sets, det Dg ≠ 0 a.e., H(z) finite. Then
H(h) = H(z) − E log|det Dg(h)| ≥ H(z) − d·log L. *Proof* A.5.3. **(ii — dimension-reducing
impossibility; upgraded from [PS] to a negative theorem.)** For dim z < dim h, **no** function of
(H(z), L, dims) lower-bounds H(h): there is a family with fixed z-marginal and H(z) finite, g a
1-Lipschitz coordinate projection, and H(h) → −∞ (h = (u, v) with H(v) → −∞, z = u; the same
family shows a dimension-reducing head can *increase* differential entropy). What survives is
exactly the decomposition H(h) = H(z) + H(fibers | z) with the fiber term unconstrained by
anything measured at z: entropy at z certifies spread of the pushforward only; it gives no lower
bound along the fibers g^{-1}(z), the unread degrees of freedom of h. *Proof* A.5.4.

**Takeaways.** **Placement rule: alignment at z; rate + marginal audit + tier-1 probing at h;
spectrally normalize the head.** Projector-space regularity is not backbone-space regularity
unless the head is dimension-preserving and suitably invertible; a dimension-reducing head can be
healthy at z while the backbone retains collapsed or pathological unread degrees of freedom — and
the best-performing LeJEPA projector output is 64-dimensional against a much wider backbone (M0's
16-d cell was picked for exhaustive searchability, but even the best configuration is exactly case
(ii)).

**Head-conditioning exposure, handled structurally.** Spectral normalization controls the head's
upper constant L; nothing in L constrains σ_min(g), so the head can contract directions, and
content living in contracted fibers receives no alignment pressure. Three structural answers:
nothing load-bearing rides the z→h transfer — every certified quantity is measured at h; the
transfer claim (c) assumes bi-Lipschitz explicitly; and head conditioning (σ_min(g), effective
rank of g) is part of the reported battery (§5), so the failure mode is watched rather than
assumed away. An h-space alignment term was considered and rejected: it would re-import the
augmentation-variant content the head exists to absorb.

**Candidate mechanism [P]** for the cross-method failure: z-space losses sit behind
method-specific warps g, so cross-method loss comparisons compare warps (within one method the
warp is stable, which is why within-sweep correlations survive); prediction E3 follows. Where
implicit entropy estimators (BYOL's EMA/predictor) attach in the placement rule is the concrete
open sub-problem of the head-composition program [→ OP-i].

**(f) Placement dynamics — stability is set by placement [M, toy-scoped; P].** New in v2.0; the
measured half of E10-T1 (USER-APPROVED) plus raw E10 numbers, imported because any program that
imposes a marginal *at h* — this one included — inherits them as engineering premises (§4).
Measured, toy frame (ViT-S/8, LeJEPA family, K = 512 embed / 384 CLS, λ as noted;
E10 card + results/diag/e10_grad_share.csv + results/figures/e10/):

1. **The projector is a gradient sink.** In the buffered configuration (statistic at 16-d
   proj.out), the trunk's share of the constraint gradient falls 0.90 → 0.32 as the projector
   absorbs the constraint; in unbuffered placements (embed/CLS) the share reaches ≥ 0.999 at
   failure — the trunk carries the whole correction.
2. **Unbuffered placements carry a finite-time instability whose fuse scales with lr.** Ignition
   epochs 12/19/32 (embed/CLS/depth-0, lr 1e-3) → ~38 (lr 3e-4); the buffered arm ran 145 epochs
   with max grad-norm 6.2 and zero spikes. The instability ignites *from* the jointly-best loss
   state (both terms low) — the good state is reachable but not stable without the buffer,
   at those settings.
3. **The moment channel is the mechanistic suspect [P].** R4c says the scale channel is the one
   unattenuated pressure at large K; at init the constrained space sat ~10× under-scaled (a
   framework-library default: trunc-normal 0.02 head init, not fan-in-scaled), so the
   statistic's first job is a large rescale — buffered in A (the sink absorbs it), trunk-borne
   otherwise. Data-dependent init calibration (fold first-batch μ, σ into the head Linear)
   removes the opening yank and measurably delays, but does not prevent, ignition; λ-rebalancing
   (×14 range) does not prevent it either — consistent with R4c's "λ cannot repair the ratio".
4. **The no-buffer configuration is trainable when moved as a package [M, raw — joint analysis
   pending].** Both losses at the representation + init calibration + lr 3e-4 + λ = 0.02
   (arm e10Dlr) trained 150 epochs monotone, quietest gradients in the family, moment part of
   the statistic descending monotonically 9.2 → 5.3; its shape residue (~1.3–1.7) is the
   largest and most consistent positive residue measured at K = 512 (all other 512-d cells sit
   in −0.5…+0.95, ≈ 0 within MC noise) — a low-dimensional/cluster-structure signature whose
   semantic status is an open battery question. Raw numbers; UNSCORED.

Design premises extracted for §4 [D]: (p1) spectrally normalize the alignment head; (p2)
calibrate any constrained space to unit scale at init (or equivalently, burn in the
standardization map before enabling the marginal term); (p3) set λ by equal-pull at init
(λ·‖∇_trunk marginal‖ = (1−λ)·‖∇_trunk align‖) rather than by sweep; (p4) treat peak lr as a
stability parameter of the placement, monitored per-step (grad-norm median × trailing-median
kill rule). Theory of the fuse-length law is open [C → OP-j].

### R7 · Tiers and hierarchy: train the tier-1 surrogate, measure the rest — serves D1, D3′, D4

**(a) [F-verified + declaration]** "All G-invariant tasks" reduces to the invariant content U in
two precise senses (Dubois et al. '21, App. B.1): deterministic invariant labelings factor
*exactly* through a maximal invariant (Lemma 5: f = h∘M), while stochastic tasks factor at the
level of conditional laws (Lemma 6: Y ⫫ X | M(X); the random variable Y is not itself a function
of M(X)). Conditions: standard Borel spaces, measurable projection, finite risks; equivalence
relations generalize group orbits. The next step is a declaration, not a theorem: when U admits a
finite factor coordinate system whose span covers the declared task family T, sufficiency is
operationalized as tier-1 decodability of that finite battery F. Finiteness enters through the
declared tuple (§1) — the program is finite because F is — and tasks outside span(F) are
explicitly invisible to D1 (§9).

**(b) [T] (Composition with an explicit modulus; upgraded from [PS].)** Let t₁, …, t_m be factor
tasks and suppose linear readouts ŵᵢ achieve E[(ŵᵢᵀz − tᵢ(x))²] ≤ εᵢ (tier-1 with slack). Let a
derived task be t = φ(t₁, …, t_m) with φ L_φ-Lipschitz (ℓ₂). Then the tier-2 probe φ(Wz) ∈ Φ∘Lin
achieves E[(φ(Wz) − t(x))²] ≤ L_φ² Σᵢ εᵢ. If φ is only uniformly continuous with concave modulus
ω, then E|φ(Wz) − t| ≤ ω(√(Σᵢ εᵢ)). *Proof* A.6.1. **The converse fails by construction** — and
not merely by V-information's decryption metaphor: take z = (t₁ ⊕ t₂, W) with t₁, t₂ iid fair
bits and W ⊥ (t₁, t₂); the derived task t₁ ⊕ t₂ is tier-1 readable at z while I(z; t₁) = 0 —
the factors are not merely linearly buried, they are absent. Hence: optimize only the tier-1 surrogate (a direct
battery term enters L only when factor labels exist); measure the rest.

**(c) [D + P + L] (The Δ protocol, with its well-definedness condition.)** Measure
Δ(t, space) = I_{V₂} − I_{V₁} with budget-matched probes (equal sample/compute for both tiers —
the probing literature's answer to probe-capacity confounds: Voita & Titov '20, Hewitt & Liang
'19). **[L] (Sign condition, new.)** V₁ ⊆ V₂ alone does not make Δ ≥ 0 (both H_V(t) and H_V(t|Z)
are non-increasing in V); Δ ≥ 0 holds when the tiers share their marginal predictors — e.g. both
families realize all constant distributions as ignorance predictions, so H_{V₁}(t) = H_{V₂}(t) =
H(t) — which our probe families satisfy by construction and the protocol now states. *Proof*
A.6.2. Pass/fail semantics: Δ ≈ 0 required on the factor battery, unconstrained on derived tasks;
inflated Δ at h for models trained at z is predicted via (b) + R6(b). Theorem-grade rationale for
budget matching, imported [F-verified]: Dubois et al. '22's sample-optimality is defined by a
worst case over the **full ERM argmin set** (their W_n := sup_t E_{D_t} sup_{f̂ ∈ ERM set}
R_t(φ, f̂)) — unmatched probe capacity measures the spread of that argmin set, not the encoder;
budget-matched, seeded probes are the finite-V shadow of controlling it (E11 additionally reports
worst-of-probes alongside mean-of-probes for exactly this reason).

**(d) [T for the linear-Gaussian case]** D3′'s estimator: the prefix-nested program equals PCA
ordering under linear-Gaussian assumptions (Eckart–Young; recovered exactly by nested dropout,
Rippel et al. '14); the general practical estimator is Matryoshka-style prefix replication
(Kusupati et al. '22); general prefix-optimality conditions are open [C → OP-b].

**(e) [T] (The battery floor — declaring F is what makes tier-1 affordable; NEW.)** Work in
Dubois '22's clean setting (finite X, exact orbits, deterministic labelings). For the **full**
invariant task family, sample optimality with linear probes forces dim ≥ |X/∼| − 1 [F-verified:
their Theorem 1] — exponential in the number of independent binary factors (|X/∼| = 2^m) and
unaffordable at any real width. For a **declared battery** F of m binary factor tasks whose
combinations all occur, tier-1 readability of F requires and is achieved at **d = m**: (≥) m
affine readouts realize at most Σ_{i≤d} C(m, i) sign patterns on any point set (hyperplane-
arrangement bound), and all 2^m factor combinations must be realized, forcing d ≥ m; (≤) the
hypercube code z = (2t₁−1, …, 2t_m−1) reads each factor by a coordinate. *Proof* A.6.3. For
cᵢ-ary factors, the direct-sum simplex code gives d = Σᵢ(cᵢ − 1) sufficiency; the matching
general lower bound is [PS]. Consequence, stated once: **the exponential→linear drop from
declaring F is exactly what makes compression (D2) and accessibility (D4) jointly satisfiable**
— under the ∀-labelings quantifier they provably cannot coexist (the dimension floor forbids
compression), and a rate term then needs a codebook to be measured against, which is where the
declared π stops being optional (§8, positioning).

**Scope declaration (the program's largest known boundary).** Everything above is stated for flat
vector representations, and the invariant-content reduction assumes tasks are functions of one
global factor vector. Object-centric and relational tasks (counting, binding) need slot/set
representations and a tier calculus over permutation-invariant readouts — outside this theory's
scope and filed [C → OP-e].

### R8 · The prior fork, priced, estimated, directional — serves D0, D5

**(a) [F-verified, scoped — upgraded from [PS pending appendix]].** The reference
isotropy-optimality analysis (Balestriero & LeCun '25) splits cleanly across probe tiers, per the
2026-07-08 verification of arXiv 2511.08544v3 (quotes in TRD_PI_REVIEW_NOTES): the linear-probe
results (their Lemmas 1–2; proofs B.1–B.2) are functions of the covariance spectrum alone — ridge
bias −λ(XᵀX+λI)⁻¹β and OLS variance σ²tr((XᵀX)⁻¹) — so **any unit-covariance embedding law,
factorial heavy-tailed members included, earns the identical guarantee under that fixed-design,
second-moment analysis**; the paper's own §3.1→3.2 transition concedes the split ("the
distribution of features must be isotropic. We now move to nonlinear probing where the standard
Gaussian will emerge as the unique optimum"). Three permanent scope notes: "minimax-equivalent"
means indistinguishable under the paper's fixed-design, second-moment analysis (spectrum
comparisons, not a minimax theorem); Gaussian uniqueness at tier 2 is exact for the leading
p-dependent kNN-ISB term under an isotropic task prior but only an upper bound in the kernel
case; probe class = {radius-kNN, Nadaraya–Watson} — no MLP result exists.

**(a′) [L leading order; remainder PS] (Random-design extension at tier 1 — the open flank,
quantified; NEW.)** The
fixed-design scoping left open whether random-design linear probing re-admits heavy tails through
fourth moments of (ZᵀZ)⁻¹. It does, at second order only: for factorial, unit-variance,
whitened features with **absolutely continuous** coordinate laws (every member of the declared
family qualifies; the hypothesis is necessary — discrete laws can make E[tr(ZᵀZ)⁻¹] = +∞ via
P(singular) > 0, adversarial-review counterexample: Rademacher coordinates) and mean
per-coordinate excess kurtosis κ̄ (finite: ν > 4; remainder control: ν > 8), the OLS excess
prediction risk over n samples is (fixed d, n → ∞)
(σ²d/n)·(1 + (d + 1 + κ̄)/n + o(1/n)).
The leading term σ²d/n is law-free; the law enters at relative order κ̄/n. Quantitative caution
at heavy tails (review): at ν = 5 (κ̄ = 6) the lemma's remainder gate fails (t₅ has no 6th
moment), and MC shows the realized κ̄-shift approaching the formula from below (64% / 75% / 85%
of predicted at n = 100/400/1600, d = 5) — an order-of-magnitude estimate there, not a proven
expansion; at probe scale n = 10⁴–10⁵ the predicted relative penalty κ̄/n is 6·10⁻⁵–6·10⁻⁴.
Gaussian benchmark: exact inverse-Wishart value d/(n − d − 1) recovers the formula at κ̄ = 0;
MC confirms the κ̄ term at κ̄ ∈ {−1.2, +3} (uniform/Laplace coordinates). *Proof* A.7.1
(leading order; remainder sketched, not proven — the one [PS] grain in this result). Filed
consequence: the tier-1 arm of the fork survives random design to leading order, with the
correction now priced rather than open.

**(b) [F]** Gaussian uniqueness enters only through nonlinear local-smoother bias, proportional
to J(p) = ∫‖∇ log p‖² p — the **Fisher-information functional** (Cramér–Rao equality iff
Gaussian; naming per the verified Lemma 6/Theorem 9, B.4 of the reference) — minimized by the
isotropic Gaussian under a scalar covariance constraint.

**(c) [L]** Unit-variance Student-t: I(ν) = ν(ν+1)/((ν+3)(ν−2)), excess kurtosis 6/(ν−4); at
ν = 5, I = 1.25 and κ₄ = 6. Calibrated statement: in the matched Fisher-information calculation,
the t₅ prior carries a 25% larger constant than the Gaussian; whether this transfers
quantitatively to the reference paper's stated ISB constant depends on matching their exact
functional, and adaptive-bandwidth probe pipelines plausibly sit below it.

**(d) The fork, made directional.** At tier 1 — the tier this framework trains — all
unit-covariance members of the family are equivalent under the verified fixed-design analysis
(a), and equivalent to leading order under random design with the correction priced (a′). Moving
from ν = ∞ to finite ν therefore costs nothing at tier 1 up to O(κ̄/n) finite-sample terms while
purchasing basis identifiability in the complete regime (R3). The price is confined to those
finite-sample constants ((c): ≈25% at ν = 5 in the matched Fisher calculation) and tier-2
local-smoother bias ((b), for the verified probe classes). The default direction therefore
stands: when sparse/leptokurtic symptoms are present in the complete regime (§5 Stage 1 / §6.2),
**finite ν is the preferred design point and the Gaussian corner the fallback** — with the
residual conditionality now confined to tier-2 probe classes beyond {kNN, kernel} (no MLP
theorem exists on either side). Either way the family nests the Gaussian at the boundary: in the
natural estimation parameter 1/ν ∈ [0, 1/ν_min], the Gaussian is the boundary point 0; testing
Gaussianity within the family is a boundary-hypothesis problem — one-sided tests,
chi-bar-squared LRT asymptotics, confidence intervals reported on 1/ν.

**(e)** Coupled design choice: heavy-tailed targets are safe to train against because the
grounding distances are bounded characteristic (CF-type) statistics. Sparse-task rate separation
remains a conjecture [C → OP-a].

**(f) What the audit means at each corner.** With a declared fixed ν — including the Gaussian
corner — Stage 4 tests a **simple null**: training hit the declared codebook (the strong
normative claim). With an estimated ν̂ it tests a **composite null**: the marginal lies in the
declared family (an adequacy claim), kept honest by the bootstrap's per-draw re-estimation (§5).
D5's content shifts across the fork, and every report states which claim is being made.

---

## 4 · The objective

With spectrally normalized head g and dithered backbone h̃ (dither applied in the rate/marginal
terms only):

**L =** align_z(views) — *distortion surrogate, at z*
**+** λ_rate · E[−log π_{ν_train}(h̃)] — *rate (closed form, R1), at h̃*
**+** λ_marg · [ mean-slice CF(h̃ → π_{ν_train}) + η · bank-max ] — *marginal surrogate +
pressure (R5)*
**+** gated: β · pairwise-HSIC(h̃) — *complete regime only (R3), after §6.1*
**+** Matryoshka prefix replication — *iff a budget chain B is declared (D3′, R7d)*

All in-loss marginal terms target the single frozen member π_{ν_train} (§5, Stage 1); at
ν_train = ∞ the objective degrades gracefully to the isotropic-Gaussian corner. The audit (§5) is
never a term in L. Neither is the battery F: during unsupervised training, sufficiency acts only
through the alignment surrogate; F itself enters at selection (§5 Stage 1) and at evaluation/audit
(R7c, E5) — R1's three-level separation.

**Stability premises (new in v2.0 — R6f's measured lessons, stated as preconditions rather than
discovered as incidents).** Placing rate + marginal pressure at h̃ is exactly the placement that
R6f measured as carrying a finite-time instability when naive. The objective above is therefore
declared **jointly with**: (p1) spectral normalization of g (already in the placement rule); (p2)
init-scale calibration of h̃ to unit scale under the declared normalization map, so the marginal
term's opening gradient is not a large rescale borne by the trunk (R4c: the scale channel is the
one unattenuated, hence loudest, channel at large K); (p3) λ_marg initialized by equal-pull at
init on shared parameters, then fixed; (p4) peak lr treated as a placement-dependent stability
parameter with a pre-registered per-step grad-norm kill rule. These are engineering premises
[D] with toy-scale measured backing [M], not theorems; the fuse-length law behind (p4) is OP-j.

**Who fights whom (the objective's equilibrium).** Alignment is invariance pressure and is
satisfiable by collapse. The rate term is compression pressure and rewards collapse — after
dithering, a point mass at the origin minimizes cross-entropy up to the σ floor. The marginal
term is therefore **the only anti-collapse force in L, and simultaneously the calibration force**
(companion identity, R1): it pushes q_h̃ toward the declared full-rank member. R4c sharpens this:
at large K the marginal surrogate's *effective* content is precisely the moment/degeneracy
channel — anti-collapse pressure survives dimension for free, while its shape content does not
(that is delegated to the bank and the audit). Anti-collapse in TRD-π is not a primitive but a
consequence of demanding that the aggregate look like π — which is why auditing the marginal,
rather than trusting the loss, is load-bearing rather than decorative.

**Who picks the basis (the demand/discovery split).** The finite-ν prior imposes that a preferred
basis *exist* — the explicit symmetry break R2 shows no standard objective supplies — not *which*
basis; which one is the data's job. Inside the affine-identifiability regime (R3's A2), the gated
independence term's floor is attainable only at source-aligned axes up to signed permutation
(Darmois–Skitovich, R3's lemma): when a factorial non-Gaussian representation exists, the axes
are discovered; when none exists, the floor is unreachable and Stage 4 reports the misfit — the
prior cannot manufacture sparsity that survives its own audit. Outside A2, a sufficiently
flexible encoder can transport nearly any law onto ∏ t_ν coordinates, and the framework claims no
factor semantics for the axes. Discovery is instrumented three ways: Stage-1 symptoms are read on
the ν = ∞ pilot — a model whose rotation-invariant loss could not have imposed them — in the
Varimax basis, a canonical rotation rather than the native one; E6 demands an identifiability
gain at matched tier-1 accuracy; and the seed null (§6.3) separates loss-induced from
optimizer-induced axes. **We impose the demand and instrument the discovery.**

---

## 5 · Audit & estimation protocol v3.1 — declare → select → train → estimate → audit

**Rule 0 (the pipeline).** One symbol never does two jobs: ν_train (in-loss) and ν̂ (audit-side)
are distinct by construction (D5), and the five stages are time-ordered so each depends only on
stages upstream of it.

**Stage 0 · Declare** (before any training): the family ∏ t_ν, ν ∈ (4, ∞]; a finite selection
grid N_grid ⊂ (4, ∞]; battery F; tiers V (with shared marginal predictors, R7c); chain B;
resolution σ; the init-calibration and λ/lr stability premises (§4); search budgets and estimator
settings for every later stage.

**Stage 1 · Select ν_train** (pre-registered, pilot-based): train the isotropic corner ν = ∞ as
the pilot — it needs no estimate (∞ is a declared constant) and this run doubles as the baseline.
Run the fork experiment (§6.2) on the pilot's estimation split. Symptoms absent → ν_train = ∞.
Present → ν_train = the grid point nearest the kurtosis-matched value ν = 4 + 6/κ̂₄ read in the
Varimax basis. ν_train is now frozen; it is a function of the pilot only, never of the model it
will train.

**Stage 2 · Train** the final model with π_{ν_train} on the training split. ν does not move
during training.

**Stage 3 · Estimate ν̂** on the estimation split of the final representation, from
loss-independent statistics (probe-anchored bases, worst-direction searches — never the
mean-slice statistics the loss optimized). Per-coordinate scales are nuisance standardization,
not parameters of the null: fit a diagonal standardization map on the estimation split, freeze
it, apply it before every audited statistic, and report the fitted scales as calibration
diagnostics — their deviation from 1 is itself a finding. The declared family stays
unit-variance; auditing a scaled family ∏ₖ sₖ·t_ν would be a different, weaker declaration.
Report ν̂ vs ν_train on the 1/ν scale: the did-training-land-where-it-aimed diagnostic.

**Stage 4 · Audit** on the third split, disjoint from both. Null: a parametric bootstrap that
replicates the whole estimate → audit pipeline on synthetic data — draw estimation-split- and
audit-split-sized samples from π_ν̂; re-estimate ν and the standardization map on the synthetic
estimation part; run the identical adversarial search, same compute and estimator settings, on
the synthetic audit part against the re-fitted member; the distribution of the resulting maxima
is the null. The per-draw re-estimation is the Lilliefors-type correction that keeps a composite
null honest — without it the audit is systematically lenient. Because θ♯∏ t_ν has no closed form
(the family is not convolution-closed), per-direction reference laws are themselves Monte Carlo
from the fitted prior under the declared budget. The moment-matched Gaussian null survives only
(i) as the exact simple null at the ν = ∞ corner and (ii) as a separate Gaussianity diagnostic.

**Rules 1–8.** 1. Double split for directions: worst directions found on sub-split A, statistics
reported on sub-split B, both inside the audit split. 2. Matched search on the null: identical
compute and estimator settings. 3. Budget-matched tiers: V₁ and V₂ probes trained under equal
budgets before Δ is read; probes run to convergence, not to a fixed budget (fixed budgets censor
differentially across feature geometries — measured: results/diag/probe_conv.csv, D-020);
worst-of-probes reported alongside mean-of-probes (R7c/W_n rationale). 4. Regime check (step 0 of
any sparse deployment): effective active-factor count vs width (participation ratio, k̂, spectral
monitors). 5. Seed null (§6.3): axis-kurtosis + Varimax-sparsity across ≥ 5 seeds under the
isotropic objective. 6. σ disclosure: every rate/marginal number carries its dither σ; equal-σ
comparisons only. 7. Both spaces: every audited statistic is reported at h and at z. 8. Sliced
statistics are reported with their three-reference decomposition (new; the R4c practice):
isotropic floor c₀ · covariance-matched twin (moment part) · shape/degeneracy residue — plus the
untrained-net reference where one exists; raw sliced values are never quoted alone, and
values across spaces of different K are never compared as numbers (R4c: not the same
instrument).

**Reported battery**: worst-direction EP and κ₄ with bootstrap CIs · defect rank k̂ ·
minimum-detectable amplitude at the declared budget (from §6.0) · the moment-part/shape-residue
decomposition per sliced cell (Rule 8) · spectral monitors (RankMe, α-ReQ, LIDAR) · the tier-gap
profile Δ(t, space) with best_ep/convergence flags · head conditioning: σ_min(g) and effective
rank of g (R6 exposure monitor) · per-step grad-norm shares on shared parameters for any at-h
term (R6f monitor) · fitted standardization scales (Stage 3 diagnostic) · ν_train, ν̂, and a CI
on 1/ν · which null (simple / composite) the audit ran.

---

## 6 · Decision experiments

**6.0 · Audit calibration by defect injection** (run once per (K, N, budget) regime). Sample from
the declared prior; inject rank-k non-Gaussian defects at controlled amplitude (kurtosis bumps /
sparse mixtures along k random directions); sweep (k, K, N, search budget); report ROC and
minimum-detectable-amplitude curves for worst-direction vs mean-slice statistics, with the §5
bootstrap as the null. Output: the audit's power curve — what the instrument can and cannot see
at the declared budget. R4c supplies the analytic prediction the calibration should reproduce
(channel constants × attenuation × N); §6.0 measures the search's realized power against it.
This converts R4 from an asymptotic argument into a calibrated instrument and supplies Stage 4's
error bars. Theory companion: OP-h (Nadjahi et al. '20 sliced-divergence rates; Meckes waiting
times). §6.0 precedes any audited claim.

**6.1 · Regime check.** Estimate active-factor count vs embedding width (protocol rule 4). Gates
everything sparse.

**6.2 · The fork experiment — the selection stage** (§5, Stage 1). Frozen pilot h → linear probes
on the battery F → Varimax rotation of the probe-weight matrix → test (i) sparsity gain vs a
random-rotation null and (ii) coordinatewise excess kurtosis in the Varimax basis. Theory hook:
Varimax performs valid inference precisely under leptokurtic factors (Rohe & Zeng), so (i) and
(ii) are two symptoms of the same sparse world — exactly the quantities R3 needs and the Gaussian
world nulls. Both null ⟹ ν_train = ∞ for this data regime; either present ⟹ finite ν_train from
the declared grid per §5, with identifiability per R3.

**6.3 · Seed null** (protocol rule 5): decides implicit-bias vs explicit symmetry breaking.

**6.4 · Spectral-norm rerun** (most informative per unit cost): retrain any baseline with a
spectrally normalized head and test whether z↔h statistic correlation rises (R6's placement
claims made cheap to falsify). Companion measured datum already in hand [M, toy]: the E10
placement family (buffered vs unbuffered SIGReg) is the *removal* version of this experiment and
produced R6f; the spectral-norm arm is the *softening* version.

---

## 7 · Falsifiable predictions (status as of 2026-07-09)

*(Numbering note: E1–E7 here are TRD-π-internal prediction labels, as in v1.3 — distinct from
the repo's experiment cards E01–E11; the status lines name the repo cards that carry each.)*

**E1.** z-statistics correlate with downstream within-method and decorrelate across methods;
shared-estimator h-statistics partially restore cross-method ordering; spectrally normalized
heads raise z↔h statistic correlation. Add k̂ per cell; check M0's mean-slice κ₄ against the
Beta-moment prediction (R4a). *Status: toy matrix complete (E01-T1…T9, USER-APPROVED,
protocol-lesson-scoped); IN-100 seed-0 matrix emitted UNSCORED; seed-1 pending.*

**E2.** Layer-wise "guillotine" curves read as depth-wise rate-allocation profiles; a
Matryoshka-trained variant flattens the drop. *Status: pre-registered for IN-100 with per-probe
scoring qualifier; toy showed tap-1 peak in 5/5 deep-head methods (E01-T6).*

**E3.** Recomputing each method's two decomposition terms (alignment; entropy proxy) with one
shared estimator at h, same backbone architecture, partially recovers the cross-method ordering
that z-space losses cannot provide.

**E4.** Marginal statistics at z behave as head-warp artifacts under unconstrained heads;
expander-head cells show the R6(d) singularity signature (effective support dim ≤ dim h).

**E5.** Δ ≈ 0 on the factor battery, unconstrained on derived tasks; models with tier constraints
trained at z show inflated Δ at h.

**E6 (the selection stage is itself falsifiable).** In regimes where §6.2 fires on the pilot,
ν̂ < ∞ replicates across seeds and the finite-ν_train model shows a positive identifiability gain
(factor alignment in the Varimax basis) over the isotropic corner at matched tier-1 accuracy; in
regimes where §6.2 nulls, no such gain appears and finite-ν training buys nothing.

**E7 (new; from R4c + R6f).** (i) A per-slice-standardized variant of the sliced-CF term (a
different declared loss) restores shape pressure at large K: its trained space shows reduced
worst-direction κ₄ at matched moment health, where the unstandardized statistic leaves
worst-direction κ₄ at the untrained level. (ii) The moment part of the three-reference
decomposition — not the raw statistic — is the quantity that tracks training health at large K.
(iii) Fuse-length ordering: for unbuffered placements, ignition time is increasing in 1/lr and
under equal-pull λ; buffered placements show no ignition at matched settings. (i) is untested;
(ii)–(iii) have toy-scale support (E10-T1 + raw Dlr trajectory) and are stated as IN-100
predictions.

---

## 8 · Positioning — what exists, what is imported, what is new

*(Absorbs THEORY_MAP §6.3 and the DUBOIS22_VS_TRD_PI synthesis; comparative readings marked
[our reading] are proposals for discussion, not agreed takeaways.)*

**The two Dubois papers are the two halves of this program, never composed [our reading,
verification-backed].** '21 (*Lossy Compression for Lossless Prediction*) is D1+D2 at a single
unconstrained readout tier — a rate theory of what must be **retained**, deliberately silent on
arrangement ("our theory discusses Bayes risk, which is independent of specific predictors"
[F-verified]). '22 (*Idealized Representations*) is D1+D4 at a declared tier — an accessibility
theory with **no rate term at all** (dimension is only bounded below; heads "as large as
possible" [F-verified]). Under '22's ∀-labelings quantifier the two halves provably cannot
coexist: the dimension floor (R7e's |X/∼| − 1) forbids compression. Cutting the task quantifier
to a declared finite battery F collapses the floor from exponential to linear (R7e, [T]) — at
which point a rate term becomes affordable and needs a codebook to be measured against: the
declared π stops being optional. TRD-π is, in this precise sense, the composition of '21 and '22
that their own quantifiers forbid — bought by the declaration, priced by the audit.

**Dubois '22 is not head-silent — the novelty is the audit, not the theorem [F-verified].** '22
proves the *need* for projection heads and derives asymmetric heads ("we are the first to relate
the architecture of the probing family and the projection head"; SimCLR's symmetric projection
"not even population optimal for linear F"). What no paper in the surveyed line does is the
**two-space audit**: measure each desideratum at both φ(X) and the head output under a matched
frame, with statistical machinery for the statistics being gamed, and a cross-tier gap object
(Δ) — '22 fixes one probe family per guarantee and never defines one. Decisive datum that the
prescriptive arm does not close the question: the same group's ICML '23 paper (169 models) could
not confirm the asymmetric-head gains and writes "we still do not completely understand the
impact of non-linear projections" (Dubois, Hashimoto & Liang '23, §5.3.3 [F-verified — the
quote lives in the 2023 paper, not in '21/'22]). The characterization exists; its prescriptions
don't visibly bind practice; the field lacks the instrument that says why. DISSL/CISSL enter the
roster as positive controls: encoders *designed* to be idealized for linear probes are a
calibration standard for the audit (prediction: small Δ at h where SimCLR shows inflated Δ).

**LeJEPA** supplies the normative anchor for D5's Gaussian corner and the fork's tier-1 base
(R8a) — and instantiates the two-space gap this project measures: both its losses are applied at
the projector output while evaluation probes a backbone-side embedding, and the paper contains no
sentence acknowledging that the theorem's space differs from the probed space [F-verified].
SIGReg's deployed statistic is the R4c member with w = quadrature × Gaussian window; E10-T1 [M]
is its measured two-jobs behavior. KerJEPA generalizes the sliced family's kernels; the
fixed-kernel power decay (Ramdas et al. '15) keeps it inside R4's scope.

**Latent Distribution Matching** (Mikulasch & Zenke '26) is the descriptive layer's engine
(alignment + entropy estimator), imported with verified scope (§2): five families derived;
real-DINO explicitly excepted; Barlow Twins/SwAV/MAE absent; LeJEPA classified by LDM as
"single-variable LDM" without identifiability guarantees. Its identifiability theorem (affine
recovery, predictive/temporal factorization, Gaussian or vMF predictive residuals) is R3's A2
supplier — and the paper formalizes z = f(x) directly, folding real methods' projectors into f:
the head-composition question is not merely undiscussed, the formalization erases it
[F-verified]. Composing LDM identifiability with a head is OP-i/OP-17 (problem note:
LDM_HEAD_COMPOSITION.md).

**Saunshi et al. '22** is the wrong-functional theorem: in the (near-)disjoint-augmentation
regime, every downstream bound monotone in the loss value is vacuous — at exact minimizers, at
near-minimizers, and at every loss value (Lemma 4.1/Cor. 4.1; Table 1's equal-loss pair
4.939/4.939 → 100% vs 50% [F-verified, regime-scoped: "provably … in some settings"; their
augmentation-identification experiment argues real image augmentations sit near that regime]).
"The head is a function-class device" remains our inference — the paper folds the projector into
f and probes at projector output; unpublished source archaeology shows the authors considered
the head/pre-head split and cut it [F-verified, not citable as published content].

**Prior desiderata frameworks.** Achille–Soatto '18 grounds D2 (with the paper's own
bijection-degeneracy acknowledgment and TC-disentanglement answer — an information functional,
hence still blind to *our* D3/D5 geometry); Tschannen et al. '20 is the wrong-functional
diagnosis for MI (bijection invariance; tighter bounds, worse representations; their §5 calls
for exactly the geometry-aware notion of information D5 supplies); Xu et al. '20 provides D4's
measurement language (I_V; DPI violated by design), with the Δ sign condition now stated at
R7c; Wang & Isola '20 supply alignment/uniformity — loss-layer objects whose h-space readings
the audit showed to be coupling-confounded (pair_margin discipline, D-013/E01-T1).
Zimmermann et al. '21: identifiability conditional on an assumed latent marginal
(theorem-level), with the paper's own robustness experiments and its prior-free bijectivity
appendix kept in view [F-verified]. Wen & Li '22, U-MAE (Zhang et al. '22), Kong et al. '23:
the gap-crossing exceptions (encoder-under-head theorems, pixel-to-encoder reductions, content
claims) — the shapes a two-space theory should aim at.

**Novelty claim, scoped (for the eventual paper; to attack in discussion).** Not the first
two-space theorem ('22 has one); not the first tier-aware design ('22 relates probes to heads).
The claimed contributions: (i) the audit program itself — both spaces, matched frame,
worst-direction honesty machinery, pre-registered; (ii) the R4 blindness suite including the
loss-level channel theorem and its measured confirmation (one statistic, two jobs); (iii) the
composition R7e makes affordable (rate + tiers + audited marginal, with the declaration priced);
(iv) the fork with its estimation pipeline (ν_train/ν̂, boundary inference); (v) the placement
dynamics (R6f) as measured engineering premises for any at-h program.

---

## 9 · Limitations — an index, each stated once at its home

- **Scope**: flat vectors; tasks as functions of one global factor vector (R7 scope declaration;
  object-centric extension → OP-e).
- **Grounding**: G is declared, not grounded; F is declared, not derived — tasks outside span(F)
  are invisible to D1 (§1, R7a).
- **Audit epistemics**: tripwire, not guarantee (R1 role split); finite banks give evidence, not
  certificates (R5); the bank minimax game can cycle (OP-g).
- **Pipeline cost and risk**: one pilot run; grid selection can misfire when the pilot's regime
  differs from the final model's — mitigated by the grid itself and the ν̂-vs-ν_train gap
  diagnostic (§5).
- **Claim semantics**: the composite-null audit (estimated ν̂) makes a weaker claim than the
  simple-null audit (declared ν); every report states which (R8f). Bootstrap validity at the
  Gaussian boundary rests on the one-sided / chi-bar-squared asymptotics of 1/ν = 0 (R8d).
- **Claim status**: R8(a) is now [F-verified, scoped] (fixed-design, second-moment; kNN/kernel
  tier-2 classes; no MLP result on either side), with the random-design flank priced at tier 1
  (R8a′); the remaining [PS] items are R7e's general-c lower bound and R4's cross-channel
  bookkeeping beyond second order — both outside the theorem core.
- **Placement dynamics evidence is toy-scale**: R6f's fuse/sink facts are measured on one
  method family at toy scale (E10-T1 approved; Dlr raw); §4's premises (p1)–(p4) are engineering
  defaults until the IN-100 arms run (E7 predictions).
- **Head**: spectral normalization bounds only the upper constant L; σ_min(g) is monitored, not
  constrained (R6).
- **Labels**: selection and audit consume F-labels; the label-free claim is confined to the
  training objective (§1).
- **Regime estimation** (rule 4) is crude; the superposition boundary is itself open (OP-f).

---

## 10 · Open problems

**OP-a** sparse-task rate–distortion separation as a theorem. **OP-b** prefix-optimality beyond
linear-Gaussian. **OP-c** bank search-time guarantees for rank-k defects (Meckes waiting-time as
the tool; R4c now supplies the target signal sizes). **OP-d** quantitative Darmois–Skitovich
(robust R3). **OP-e** object-centric tier calculus. **OP-f** the budget/sparsity boundary where
monosemantic frames stop being optimal. **OP-g** bank minimax dynamics (convergence vs cycling).
**OP-h** finite-budget detection theory: minimum detectable amplitude for rank-k defects under a
declared search budget (companion to §6.0; Nadjahi et al. rates + Meckes waiting times as tools;
the fixed-statistic side is now solved by R4c — what remains is the searched-statistic side).
**OP-i** placement of implicit entropy estimators (EMA/predictor) in the two-space rule
(formerly OP-17′). **OP-j (new)** the fuse-length law: derive the ignition time of the
unbuffered-placement instability as a function of (lr, λ, K, init scale) — the measured ordering
is ep12/19/32 @1e-3 → ~38 @3e-4 with no ignition when buffered; candidate frame: two-timescale
analysis where the unattenuated moment channel couples the constrained space's scale error to
trunk curvature. **OP-k (new)** statistical theory of the defect-rank estimator k̂ (consistency
and CI coverage when fitting the slice-κ₄ spectrum to the Beta attenuation of R4a). **OP-l
(new)** general-c battery floor: matching lower bound for R7e beyond binary factors.

---

## Appendix A · Proofs

### A.1 R1 — identities

**A.1.1 (Rate–prior).** E_x KL(q(z|x)‖π) = E_{x,z}[log q(z|x) − log π(z)]
= E_{x,z}[log q(z|x) − log q_Z(z)] + E_z[log q_Z(z) − log π(z)] = I(X;Z) + KL(q_Z‖π). ∎

**A.1.2 (Companion).** E[−log π(h̃)] = E[−log q_h̃(h̃)] + E[log q_h̃(h̃) − log π(h̃)]
= h(q_h̃) + KL(q_h̃‖π). ∎

**A.1.3 (Closed-form dithered rate).** With q(h̃|x) = N(f(x), σ²I), I(X; h̃) + KL(q_h̃‖π)
= E_x KL(q(h̃|x)‖π) (A.1.1) = E_x E[−log π(h̃)] − h(N(0, σ²I)), and h(N(0, σ²I)) = const(σ). ∎

### A.2 R2 — symmetry inventory

**(1) Full O(K).** Let R ∈ O(K). Pairwise-distance alignment terms depend on features only
through ‖zᵢ − zⱼ‖ (or cos after ℓ2, likewise invariant). A discrepancy D(q, π) is
rotation-equivariant when D(R♯q, R♯π) = D(q, π) (true for sliced/CF/MMD-with-radial-kernel
constructions); if π is rotation-invariant (N(0,I), sphere-uniform), R♯π = π, so
D(R♯q, π) = D(q, π). Every term is invariant under z ↦ Rz, hence the minimizer set is O(K)-closed.
∎

**(2) Barlow Twins at optimum.** The optimum set is {cross-correlation = I}. If C(z) = I then
C(Rz) = R C(z) Rᵀ = R Rᵀ = I. The optimum set is O(K)-closed (statements about non-optimal level
sets are not claimed: the loss itself is basis-dependent). ∎

**(3) VICReg.** Admissible minimizers have diagonal covariance with per-coordinate std ≥ γ
(variance hinge inactive) and zero off-diagonal covariance (covariance penalty zero); the
invariance term is rotation-invariant. An orthogonal U maps every admissible diagonal Σ to
UΣUᵀ, again diagonal for **all** admissible Σ iff U conjugates all diagonal matrices to diagonal
matrices iff U is a signed permutation (take Σ = diag with distinct entries: UΣUᵀ diagonal forces
U's columns to be ±standard basis vectors; this step uses that some minimizer of the full loss
realizes a distinct-entry admissible Σ — population-level with a flexible encoder, the standing
assumption of the minimizer-symmetry frame). Setwise symmetry = signed permutations. A *given*
minimizer with spectrum multiplicities m_j admits the stabilizer ∏ O(m_j) (rotations within
equal-variance blocks preserve that Σ) — solution-dependent gauge, not a loss symmetry.
Counterexample retained: Cov = diag(1, 4) rotated by 45° has off-diagonal 1.5 ≠ 0 — re-activates
the penalty; hence the minimizer set is neither {Cov = I} nor O(K)-closed. ∎

### A.3 R3 — inversion

**A.3.1.** Under (A2), y := f(x) = As★ + b with A invertible; under (A3) the coordinates of
y − b are mutually independent (factorial marginal), in particular pairwise independent. The
sources are independent with at most one Gaussian (A1), so A.3.2 applies directly: A = ΠΛ. (This
is Comon '94's identifiability theorem, recovered through its Darmois–Skitovich engine; A.3.2 is
the sharper statement that pairwise independence already suffices.) ∎

**A.3.2 (Pairwise lemma).** Let y = As★, coordinates pairwise independent; sources independent,
nondegenerate, finite variances, at most one Gaussian. Fix i ≠ j and a source
index k with a_ik ≠ 0 and a_jk ≠ 0. Darmois–Skitovich: if Σ_k a_ik s_k and Σ_k a_jk s_k are
independent, every s_k appearing in both with nonzero coefficients is Gaussian. Hence each
non-Gaussian source loads on at most one row. At most one source (say s_g) is Gaussian. Pairwise
independence ⇒ pairwise uncorrelatedness (finite variances used exactly here — Cauchy-type
sources would disable this step); if s_g loaded on two rows i ≠ j, then
Cov(y_i, y_j) = a_ig a_jg Var(s_g) ≠ 0 (Var(s_g) > 0 by nondegeneracy — a degenerate "source"
admits shear counterexamples; all other contributions vanish by independence and single-row
loading) — contradiction. So every source loads on at most one row; invertibility of A
forces exactly one per row and column: A is monomial, A = ΠΛ. ∎

### A.4 R4 — blindness suite

**A.4.1 (Beta attenuation).** Decompose Z = W + G with W supported in the k-dim subspace S
carrying all non-Gaussian structure and G an independent Gaussian on S^⊥ (hypothesis). For θ
uniform: θᵀZ = α·(vᵀW) + θ_⊥ᵀG, α = ‖P_S θ‖, v = P_S θ/α. Cumulants of order r ≥ 3: Gaussian
part contributes 0; homogeneity gives κ_r(θᵀZ) = α^r κ_r(vᵀW). α² = ‖P_Sθ‖² for uniform θ is
Beta(k/2, (K−k)/2) (project a uniform sphere vector; standard). Moments:
E[(α²)^m] = B(k/2+m, (K−k)/2)/B(k/2, (K−k)/2) = ∏_{j=0}^{m−1} (k+2j)/(K+2j). ∎

**A.4.2 (Tensor contraction).** κ₄(θᵀZ) = Σ_{ijkl} θᵢθⱼθₖθ_l C⁽⁴⁾_{ijkl}. For uniform θ,
E[θᵢθⱼθₖθ_l] = (δ_{ij}δ_{kl} + δ_{ik}δ_{jl} + δ_{il}δ_{jk})/(K(K+2)). Contracting against the
fully symmetric C⁽⁴⁾ gives E_θ[κ₄] = 3 Σ_{ij} C⁽⁴⁾_{iijj}/(K(K+2)). The worst slice in absolute
value is sup_{‖θ‖=1} |⟨C⁽⁴⁾, θ^{⊗4}⟩| = the injective norm (the unsigned sup can be ≈ 0 for
platykurtic defects — review precision). Rank-1 defect C⁽⁴⁾ = κ·v^{⊗4}, κ > 0:
mean = 3κ/(K(K+2)), max = κ, ratio = K(K+2)/3. ∎

**A.4.3 (Channel theorem).** Write ψ̂_N(t) = N^{−1}Σ_a e^{it s_a}. (1) *Exact mean:*
E|ψ̂_N(t) − φ(t)|² = |ψ(t) − φ(t)|² + Var[ψ̂_N(t)] and
Var[ψ̂_N(t)] = N^{−1}(1 − |ψ(t)|²) (independence; |e^{its}| = 1). Multiply by N and integrate
against w. (2) *Floor:* ψ = φ kills the first term; 1 − |φ|² = 1 − e^{−t²}. Numerically with the
shipped w: c₀ = 1.053 (Appendix B). (3) *Collapse value:* s ≡ 0 ⇒ ψ̂ ≡ 1 deterministically ⇒
T = N∫(1 − φ)²w; c_∞ = 0.402. Not a supremum over degenerate laws: for a point mass at c,
T/N = ∫(1 − 2φ(t)cos(tc) + φ(t)²)w(t)dt, minimized at c = 0 and rising to ∫(1 + φ²)w = 3.95 as
|c| → ∞ (computed: c = 1 → 1.19, c = 2 → 2.64, c = 5 → 3.94). (4) *Channels:* for cumulant
deviations
(κ̃₁, κ̃₂, κ̃₃, κ̃₄) = (m, σ²−1, γ, κ) small, the CF expands as
ψ(t) = φ(t)·exp(Σ_r κ̃_r (it)^r/r!) = φ(t)(1 + Σ_r κ̃_r (it)^r/r! + O(‖κ̃‖²)), so
|ψ − φ|² = φ(t)²·[(κ̃₂t²/2 − κ̃₄t⁴/24 − …)² + (κ̃₁t − κ̃₃t³/6 + …)²] + O(‖κ̃‖³): a psd quadratic
form; pure channels give N·C_{2r}κ̃_r² with C_{2r} = (1/(r!)²)∫t^{2r}φ(t)²w(t)dt (φ² from the CF
deviation; w carries its own window). Cross terms couple only same-parity channels (mean↔skew,
variance↔kurtosis) and are bounded by Cauchy–Schwarz between neighboring constants. The noise
floor moves too, at FIRST order in the even deviations (review correction): |ψ(t)|² =
e^{−t²}·exp(−κ̃₂t² + κ̃₄t⁴/12) = e^{−t²}(1 − κ̃₂t² + κ̃₄t⁴/12 + O(‖κ̃‖²)), so
∫(1 − |ψ|²)w = c₀ + κ̃₂∫t²e^{−t²}w − (κ̃₄/12)∫t⁴e^{−t²}w + O(‖κ̃‖²) = c₀ + C₂κ̃₂ − 0.0402·κ̃₄ +
O(‖κ̃‖²) (for the shipped w, ∫t²e^{−t²}w = ∫t⁴e^{−t²}w = 0.482 — the a = 3/2 Gaussian-moment
ratio makes these equal exactly); odd cumulants leave |ψ| unchanged. Hence
E[T] − c₀ = N·Q(κ̃) + F(κ̃₂, κ̃₄) + h.o.t. as stated in-text. Sliced, E_θ[F] carries the LINEAR
attenuation of κ̃₄(θ) (∝ 1/K) against the signal's quadratic one (∝ 1/K²); dominance crossover
at K* = 9NC₈κ/C₄ in the factorial case. Exact-CF check (factorial Laplace κ = 3, N = 256):
K = 16 → E[T] − c₀ = +0.160 (signal +0.179, floor −0.019), matching the deployed-module MC
(1.21 ± 0.24 = c₀ + 0.16); K = 512 → −4.48·10⁻⁴ (signal +2.54·10⁻⁴, floor −7.02·10⁻⁴). (5) *Sliced
attenuation:* mean channel m_θ = θᵀμ ⇒ E[m_θ²] = μᵀE[θθᵀ]μ = ‖μ‖²/K. Variance channel
ε_θ = θᵀΣθ − 1: E[θᵀΣθ] = trΣ/K = λ̄; E[(θᵀΣθ)²] = ((trΣ)² + 2‖Σ‖_F²)/(K(K+2)) (contract the
uniform fourth-moment identity against Σ⊗Σ), and substituting trΣ = Kλ̄, ‖Σ‖_F² = K(λ̄² + s²)
gives the exact identity E[ε_θ²] = E[(θᵀΣθ)²] − 2λ̄ + 1 = (λ̄−1)² + 2s²/(K+2). Shape channels: κ_r(θ)² = α^{2r}κ_r(v)²; independence of α and v plus
A.4.1's moments give E[κ_r(θ)²] = ∏_{j<r}(k+2j)/(K+2j) · E_v[κ_r(vᵀW)²] ~ (k/K)^r; factorial
equal-κ case: κ₄(θ) = κΣᵢθᵢ⁴, E[Σθᵢ⁴] = 3/(K+2), and Var[Σθᵢ⁴] = O(1/K³) so
E[κ₄(θ)²] = 9κ²/(K+2)² + O(1/K³). (6) *Anti-CLT:* if Z concentrates on r points {z_1..z_r} with
r ≪ N, θᵀZ concentrates on r atoms and no CLT-mixing occurs. Two K-independent Θ(N) mechanisms,
kept distinct (review precision): well-separated clusters read Θ(N) because an r-atom mixture's
CF stays far from φ on the weight's support; a *degenerate* (collapsed) cloud reads ≈ N·c_∞
because its slice variances are near zero (item 3). Either way T = Θ(N) over most θ,
independent of K. Detectability numbers and MC confirmation:
Appendix B. ∎

### A.5 R6 — two-space proofs

**A.5.1 (c).** q_h = (g⁻¹)♯q_z on Im(g); g⁻¹ is (1/ℓ)-Lipschitz on Im(g) (bi-Lipschitz lower
bound). For any coupling γ of (q_z, π) supported on Im(g)², the pushforward (g⁻¹×g⁻¹)♯γ couples
(q_h, (g⁻¹)♯π) with cost E‖g⁻¹(z) − g⁻¹(z′)‖^p ≤ ℓ^{−p}E‖z − z′‖^p. Take infima. Kirszbraun
remark: extend g⁻¹ to G: ℝ^{d′} → ℝ^d with the same constant (Kirszbraun's theorem); the same
coupling argument gives the extension-dependent statement. ∎

**A.5.2 (d).** g locally Lipschitz ⇒ g(A) has Hausdorff dimension ≤ dim A for every A (Lipschitz
maps do not raise Hausdorff dimension; countable exhaustion by Lipschitz pieces). q_z(g(ℝ^d)) = 1
and g(ℝ^d) has H-dim ≤ d < d′ ⇒ q_z gives full mass to a Lebesgue-null set (the image need not
be closed, so we argue on the image itself, not on supp — review precision) ⇒ q_z ⊥ Lebesgue,
in particular q_z has no
density w.r.t. N(0, I_{d′}): KL(q_z‖N) = +∞ by definition (q_z ⋠ N). Dither: q_z ∗ N(0, σ²I) is
absolutely continuous — well-posedness restored. ∎

**A.5.3 (e″ case i).** Under the hypotheses, the change-of-variables formula holds a.e.:
p_z(g(h))·|det Dg(h)| = p_h(h). Then H(z) = −E[log p_z(z)] = −E[log p_h(h)] + E[log|det Dg(h)|]
= H(h) + E log|det Dg(h)|. Each singular value of Dg is ≤ L (Lipschitz), so
log|det Dg| ≤ d log L. Rearranged: H(h) = H(z) − E log|det Dg(h)| ≥ H(z) − d log L. ∎

**A.5.4 (e″ case ii, impossibility).** Family: h_ε = (u, v_ε) ∈ ℝ² with u ∼ N(0,1) ⊥
v_ε ∼ N(0, ε²); g(u, v) = u (1-Lipschitz projection; L = 1). H(z) = H(u) fixed and finite;
H(h_ε) = H(u) + H(v_ε) = H(u) + log(ε√(2πe)) → −∞ as ε → 0. Any putative bound
H(h) ≥ B(H(z), L, d, d′) fails for ε small. The exact decomposition H(h) = H(z) + H(h|z) (here
H(v|u) = H(v)) identifies the uncontrolled term as the fiber entropy. Both signs occur: the
ε → 0 end has H(z) > H(h) (the dimension-reducing head *increases* differential entropy — the
original v1.3 counterexample), the ε → ∞ end has H(z) < H(h); no inequality in either direction
is forced. ∎

### A.6 R7 — tier proofs

**A.6.1 (b).** ‖(Wz)ᵢ − tᵢ‖_{L²} ≤ √εᵢ by hypothesis (rows of W = ŵᵢ). Then
E[(φ(Wz) − φ(t₁..t_m))²] ≤ L_φ² E‖Wz − t‖² = L_φ² Σᵢ E[(ŵᵢᵀz − tᵢ)²] ≤ L_φ² Σᵢ εᵢ. Modulus
version: E|φ(Wz) − φ(t)| ≤ E[ω(‖Wz − t‖)] ≤ ω(E‖Wz − t‖) ≤ ω(√(Σεᵢ)) for concave ω (Jensen).
The composed probe has depth = depth(φ) + 1 ∈ Φ∘Lin. Converse counterexample in-text. ∎

**A.6.2 (c — Δ sign).**
Δ = I_{V₂} − I_{V₁} = [H_{V₂}(t) − H_{V₂}(t|Z)] − [H_{V₁}(t) − H_{V₁}(t|Z)]. V₁ ⊆ V₂ gives
H_{V₂}(t|Z) ≤ H_{V₁}(t|Z) and *also* H_{V₂}(t) ≤ H_{V₁}(t) — the two differences fight. The
condition that restores the sign is H_{V₁}(t) = H_{V₂}(t) (shared ∅-predictor class — the
minimal condition; review precision); a convenient sufficient case is both families realizing
every constant marginal prediction at ∅ (optional-ignorance built into Xu's predictive families
plus richness of the ∅-slots), which gives the common value H(t) (cross-entropy against the
true marginal is the common infimum). Then Δ = H_{V₁}(t|Z) − H_{V₂}(t|Z) ≥ 0. ∎

**A.6.3 (e — battery floor).** Lower bound: suppose linear readouts (wᵢ, bᵢ) read the m binary
factors: sign(wᵢᵀz(x) + bᵢ) = tᵢ(x) for all x. The m hyperplanes partition ℝ^d into at most
Σ_{i=0}^{d} C(m, i) cells (Schläfli/Zaslavsky), each cell carrying one sign pattern. All 2^m
factor combinations occur by hypothesis, and distinct combinations force distinct sign patterns,
so Σ_{i≤d} C(m, i) ≥ 2^m = Σ_{i≤m} C(m, i), which forces d ≥ m. Upper bound: z(x) =
(2t₁(x) − 1, …, 2t_m(x) − 1) ∈ {±1}^m reads factor i by coordinate i with margin 1. General cᵢ:
concatenate per-factor simplex codes (cᵢ vertices in ℝ^{cᵢ−1}), d = Σ(cᵢ − 1); argmax-linear
readout per factor. ∎

### A.7 R8 — random-design lemma

**A.7.1 (a′).** Setup: n iid rows z_a ∈ ℝ^d, factorial law with absolutely continuous
coordinates (so S = ZᵀZ/n is a.s. invertible for n ≥ d, and E[tr S⁻¹] < ∞ for n large under the
stated moment conditions — absolute continuity is necessary: for Rademacher coordinates, d = 2,
n = 10, P(S singular) = 2·2^{−(n−1)} > 0 and E[tr S⁻¹] = +∞ while the formula would read 2.20;
MC-confirmed in review), E z = 0, Cov = I, per-coordinate
excess kurtosis κᵢ, κ̄ = d^{−1}Σκᵢ; y = zᵀβ + ε, ε ⊥ z, Var ε = σ². OLS excess prediction risk
= σ²·E[tr((ZᵀZ)⁻¹)] = (σ²/n)·E[tr(S⁻¹)], S = ZᵀZ/n. Write S = I + E, E = n^{−1}Σ_a(z_a z_aᵀ − I),
E[E] = 0. Second-order von Neumann expansion: S⁻¹ = I − E + E² − E³(I + E)⁻¹, so
E[tr S⁻¹] = d + E[tr E²] + R. Compute E[tr E²] = Σ_{ij} E[E_{ij}E_{ji}]
= n^{−1} Σ_{ij} (E[zᵢ²zⱼ²] − δ_{ij}) [iid rows, mean-zero summands]
= n^{−1} [Σ_{i≠j} 1 + Σᵢ (3 + κᵢ) − d] = n^{−1}[d(d − 1) + 3d + Σκᵢ − d] = (d/n)(d + 1 + κ̄).
Hence E[tr S⁻¹] = d[1 + (d + 1 + κ̄)/n] + R, and the risk is (σ²d/n)(1 + (d + 1 + κ̄)/n) + σ²R/n.
Remainder [PS]: ‖E‖ = O_p(√(d/n)), but the von Neumann series is invalid on the
positive-probability event {‖E‖ ≥ 1}, so controlling E[tr S⁻¹]'s tail requires a
truncation/bad-event argument with 6th/8th-moment inputs (ν > 8 for t_ν) — sketched, not
written. For 4 < ν ≤ 8 the displayed correction term is still finite and the statement is
conjectured in the trimmed/truncated-feature sense, with MC support (the κ̄-shift reaches 85% of
predicted by n = 1600 at ν = 5). Gaussian check: κ̄ = 0 and exact
E[tr(ZᵀZ)⁻¹] = d/(n − d − 1) = (d/n)(1 + (d + 1)/n + O(n⁻²)); MC confirms the κ̄ term for
uniform (κ̄ = −1.2) and Laplace (κ̄ = +3) coordinates at n = 50–100, d = 5. ∎

---

## Appendix B · Constants and measured confirmation of the R4c channel theorem

Shipped statistic (verbatim from the deployed implementation, `sslgap/methods/lejepa.py`):
M = 256 fresh directions/step, 17 knots on [0, 3], quadrature weights (2Δt interior, Δt
endpoints) × Gaussian window e^{−t²/2}; per-slice statistic multiplied by the per-view sample
count N; slices NOT standardized. Loss = λ·(mean over views and slices) + (1 − λ)·inv, λ = 0.02,
training N = 256.

**Constants (analytic value | 17-knot quadrature).** Verified 2026-07-09 (scratchpad
`check_constants.py`; analytic = truncated-at-3 integrals):

| constant | meaning | analytic | quadrature |
|---|---|---|---|
| c₀ | H0 floor (N-free, K-free) | 1.0527 | 1.0525 |
| c_∞ | mean-zero collapse value / N | 0.4022 | 0.4020 |
| C₂ | mean-channel constant | 0.48240 | 0.48240 |
| C₄ | variance-channel constant | 0.12059 | 0.12059 |
| C₆ | skewness-channel constant | 0.02233 | 0.02232 |
| C₈ | kurtosis-channel constant | 0.00325 | 0.00325 |

Collapse value at N = 256: 102.9 — the mean-zero-collapse reading, not a supremum (A.4.3(3));
the measured "stuck at ~70 ≈ ⅔ of 103" and "pinned at 103" readings of the failed/collapsed
arms are this constant live. Per-step shape response (κ = 3 factorial; exact CF = signal +
first-order floor shift): K = 16 → +0.160 vs H0 sd ≈ 0.19; K = 512 → **−4.5·10⁻⁴** vs H0 sd
≈ 0.08 (negative — the floor-shift regime).

**MC verification (exact deployed module, N = 256, 20 reps, mean ± sd)** — from
SIGREG_DIM_SCALING §4 (2026-07-09):

| synthetic input | K=16 | K=512 | prediction |
|---|---|---|---|
| N(0,I) | 1.02 ± 0.15 | 1.07 ± 0.08 | floor 1.053, K-indep ✓ |
| var ×1.2 all dims | 2.21 ± 0.23 | 2.17 ± 0.09 | exact 2.19, K-indep ✓ (2nd-order-signal-only: 2.29) |
| var ×2 all dims | 17.5 ± 0.9 | 17.3 ± 0.5 | exact 17.1, K-indep ✓ (beyond-perturbative) |
| rank-1 kurt κ=3 | 1.03 ± 0.19 | 1.08 ± 0.05 | exact +2.9·10⁻³ / −1.4·10⁻⁶ — invisible ✓ |
| factorial kurt κ=3 | 1.21 ± 0.24 | 1.06 ± 0.07 | exact +0.160 / **−4.5·10⁻⁴** ✓ |
| 10-cluster + 0.1σ noise | 25.6 ± 4.3 | 27.5 ± 1.7 | anti-CLT: K-indep, large ✓ |

Prediction column review-corrected to exact-CF values (signal + first-order floor shift,
A.4.3(4)); the first draft quoted second-order-signal-only numbers, and the rank-1 K = 512
entry additionally inherited a transcription error (+2·10⁻⁵) from SIGREG_DIM_SCALING §4 — the
formula's signal value is +1.1·10⁻⁸ and the exact response −1.4·10⁻⁶ (floor-dominated,
negative); the source note should carry the same fix. H0 per-step sd at K = 16 is 0.19 under a
200-rep MC (the 20-rep table row reads 0.15). Note the MC means match the exact-CF predictions
within error in every row — including the two rows where they contradict the first draft's
prediction column.

**Measured on trained networks** (results/diag/sigreg_ref.csv, jobs 62176231/62178570; training
conditions replayed; T_actual = c₀ + moment part + shape residue):

| cell | K | T_actual | moment part | shape residue |
|---|---|---|---|---|
| A @proj (buffered) | 16 | 3.80 | 0.44 | **2.32 (84% of deviation)** |
| Blr_best @embed | 512 | 7.07 | 6.23 | −0.22 ≈ 0 |
| D0 end @embed | 512 | 16.58 | 16.01 | −0.48 ≈ 0 |
| Dr end @embed | 512 | 71.69 | 69.69 | 0.95 ≈ 0 |
| Dlr ep38 → ep150 @embed | 512 | 11.56 → 7.93 | 9.22 → 5.31 | 1.29 → 1.57 (raw, UNSCORED) |

Cross-validation: the K = 16 shape fraction (84%) independently matches the audit-N
decomposition at proj.out (87–94% shape; E01-T9, different N — both can hold because signal ∝ N
while the floor is N-free). Audit-side EP uses plain-Δt quadrature and audit N — training-loss
and audit EP values are **not** directly comparable numbers (METRICS note).

---

## Appendix C · Verification record (what is quote-verified, where)

All [F-verified] tags in this document resolve to `docs/theory/verification/`: `anchors_ib_mi_byol.md`
(Achille–Soatto Prop. 3.1 conditions; Tschannen §1/§3; BYOL Table 5(a)/(b)/19 cell-exact numbers
and the "admits collapsed solutions" wording), `anchors_dubois_xu.md` ('21 Thm 2/Lemmas 5–6 with
conditions; '22 Thm 1, W_n Def. 3, head theorems; the 2302.03068 re-attribution of the
nonlinear-projection quote; Xu Defs. 1–3, DPI-violation-by-design; the Δ-sign caveat),
`anchors_identifiability.md` (Zimmermann theorem hypotheses incl. uniform-marginal conditioning
and the prior-free bijectivity appendix; LDM decomposition/table/theorem scoping; LDM projector
absence across v1–v3; Mikulasch & Zenke authorship), `anchors_saunshi.md` (vacuity regime
scoping; minimizers-vs-values; head-silence + source archaeology), and
`TRD_PI_REVIEW_NOTES.md` §risk-1 (LeJEPA 2511.08544v3: covariance-only linear lemmas; Fisher
functional naming; 64-d ablation; both-losses-at-projector + backbone probing; no
two-space-acknowledging sentence).

---

## Appendix D · Changes from v1.3 (itemized; no agreed takeaway altered)

**Discharged pre-submission checklist items (provenance §4):**
1. R8(a) re-derivation → incorporated as [F-verified, scoped] with the three amendments (Fisher
   functional; fixed-design/second-moment scoping; kNN/kernel-only tier-2 classes); NEW R8(a′)
   random-design lemma prices the flank the verification left open. (Item 1: discharged.)
2. BYOL cell → corrected wording and exact cells in §0. (Item 2: discharged.)
3. R7(b) proven with explicit modulus [T]; R4 signal reading proven as the channel theorem R4c
   [T] with constants (Appendix B). (Item 3: discharged.)
4. §6.0-before-audited-claims retained; R4c gives it analytic targets. (Item 4: unchanged, ours
   to run.)
5. Submission spine (R1/R4–R6 headline, R8 conditional extension) now stated in §0. (Item 5.)
6. Audit thresholds: E-cards carry them; E7 added with pre-registerable statements. (Item 6:
   partially — thresholds still to be locked per-card.)

**New results:** R4c channel theorem + corollaries (formalizes SIGREG_DIM_SCALING; E10-T1 is its
measured confirmation) · R6(e″) case ii upgraded to an impossibility theorem · R6(c) support
hypothesis + Kirszbraun remark; R6(d) locally-Lipschitz hypothesis made explicit · R6(f)
placement dynamics [M/D/P] · R7(b) [PS→T] · R7(c) Δ-sign lemma + W_n budget-matching rationale +
worst-of-probes rule · R7(e) battery floor [T] · R8(a′) random-design lemma [L] · A.3.2 pairwise
D–S written out.

**Citation/attribution fixes (from the verification pass):** Mikulasch **& Zenke** (two authors);
corners-table attribution scoped (which rows are LDM-derived vs our extension); Dubois–Hashimoto–
Liang ICML 2023 (2302.03068) added and the "nonlinear projections" quote re-homed; Saunshi
vacuity regime-scoped; BYOL "satisfying their objective" → "whose objective admits collapsed
solutions" with exact cells; R7(a) restated per Lemma 5/Lemma 6; Zimmermann convex-body variant
and robustness noted at R3(A2).

**Protocol delta:** v3.0 → v3.1 — Rule 3 gains convergence + worst-of-probes clauses (D-020;
W_n); Rule 8 (three-reference decomposition; no cross-K numeric comparison) added; reported
battery gains moment/shape decomposition, grad-share monitor, convergence flags; Stage 0 gains
the stability premises.

**New sections:** §8 positioning (absorbs THEORY_MAP §6.3 + DUBOIS22_VS_TRD_PI synthesis, marked
[our reading]); §7 E7; OP-j/k/l.

**v2.0 adversarial proof-review round (2026-07-09, same day as the draft; corrections kept
visible per the provenance culture).** An independent reviewer pass MC/analytically confirmed
14/16 proof units and every Appendix-B constant to the quoted digit, and found two genuine
errors plus three gaps, all folded in above: (1) the channel theorem's first draft dropped the
first-order noise-floor shift F(κ̃₂, κ̃₄) — at K = 512 the shift flips the sign of the per-step
shape response (R4c(4), corollary, A.4.3(4)); the two-jobs conclusion *strengthens*: beyond
K* ≈ 9NC₈κ/C₄ the marginal response to a leptokurtic defect is negative. (2) "Collapse
ceiling" was a misnomer — mass at 0 is the minimum among degenerate laws (a point mass at large
offset reads ~9.8× higher); the constant is the collapse *reading*, its observed pinning
explained by mean-channel attenuation (R4c(3), A.4.3(3)). (3) A.7.1 required an
absolute-continuity hypothesis (Rademacher counterexample: E[tr S⁻¹] = +∞ under the old
hypotheses) and its remainder is [PS]; the ν = 5 application sits outside the remainder gate
(MC: 64→85% of the predicted shift at n = 100→1600) and the κ̄/n range was corrected to
6·10⁻⁵–6·10⁻⁴. (4) Minor hypotheses added: finite variances + nondegenerate sources in
R3-[L]/A.3.2; realizability in A.2(3); |·| in the injective-norm reading of R4(b); independence
of the padding in R7(b)'s counterexample; mass-1 wording in A.5.2; minimal Δ-sign condition in
A.6.2. (5) Appendix-B prediction column corrected to exact-CF values; the rank-1 K = 512 entry
had inherited a transcription error (+2·10⁻⁵) from SIGREG_DIM_SCALING §4 — that source note
needs the same fix (not applied there; existing docs untouched by this draft).

**Unchanged:** D0–D5 and the sketch analogy; R1/R2/R3/R5 statements (proofs now written out);
§4's objective form; §5 pipeline core; §6 experiments; every standing redline (provenance §5) —
spot-check: no "passed", no "dissolves", "consistent with rank > 1 — k̂ decides", "worst witness
found under the declared search budget", "+25% in the matched Fisher-information functional".

---

## References (anchor list)

Achille & Soatto, JMLR 19(50), 2018 (1706.01350) · Alemi et al., ICLR 2017 (1612.00410); ICML
2018 (Fixing a Broken ELBO) · Babu & Rao, Sankhyā 2004 (goodness-of-fit with estimated
parameters) · Balestriero & LeCun, 2025 (2511.08544, LeJEPA) · Ballé, Laparra & Simoncelli, ICLR
2017; Ballé et al., ICLR 2018 (hyperprior); Minnen et al., NeurIPS 2018; Theis et al., ICLR 2017
(neural compression / learned entropy models) · Bordes et al., TMLR 2023 (guillotine) · Caron et
al., NeurIPS 2020 (SwAV); Caron et al., ICCV 2021 (DINO) · Chen & He, 2021 (SimSiam) · Comon,
Signal Processing 36, 1994 · Deshpande et al., CVPR 2019 (max-sliced W) · Diaconis & Freedman,
Ann. Statist. 12(3), 1984; Dümbgen & Zerial, 2011; Meckes, 2009/2010 (quantitative, waiting
time) · Dubois et al., NeurIPS 2021 (2106.10800); NeurIPS 2022 (2209.06235) · **Dubois,
Hashimoto & Liang, ICML 2023 (2302.03068, risk decomposition — added v2.0)** · Elhage et al.,
2022 (toy models of superposition) · Epps & Pulley, Biometrika 1983 · Federici et al., ICLR 2020
(2002.07017) · Garrido et al., ICML 2023 (RankMe); Agrawal et al., NeurIPS 2022 (α-ReQ); Thilak
et al., 2023 (LIDAR) · Gorham & Mackey, 2017; Liu, Lee & Jordan, 2016; Chwialkowski et al., 2016
(KSD family; complementary audit) · Grill et al., 2020 (2006.07733, BYOL) · Grünwald, 2007 (MDL)
· Hewitt & Liang, EMNLP 2019 · Hoffman & Johnson, 2016 (ELBO surgery) · Hyvärinen & Morioka,
NeurIPS 2016 (TCL); AISTATS 2017 (PCL); Hyvärinen, Sasaki & Turner, AISTATS 2019 (GCL) ·
Khemakhem et al., AISTATS 2020 (iVAE) · Kirszbraun, Fund. Math. 22, 1934 (Lipschitz extension) ·
Kusupati et al., NeurIPS 2022 (Matryoshka) · Lachapelle et al., 2022 (sparse identifiability) ·
Li et al., 2021 (SSL-HSIC) · Lilliefors, JASA 1967 (tests with estimated parameters) · Liu et
al., ICML 2020 (deep-kernel two-sample) · **Mikulasch & Zenke, ICML 2026 Spotlight (2605.03517,
latent distribution matching — authorship corrected v2.0)** · KerJEPA, 2025 (2512.19605) ·
Nadjahi et al., NeurIPS 2020 (statistical properties of sliced divergences) · Olshausen & Field,
Nature 1996 · Ramdas et al., AAAI 2015 · Rippel et al., ICML 2014 (nested dropout) · Roeder,
Metz & Kingma, ICML 2021 (linear identifiability of learned representations) · Rohe & Zeng,
JRSS-B (Varimax inference) · Saunshi et al., ICML 2022 (2202.14037) · Schläfli/Zaslavsky
(hyperplane-arrangement cell counts; Zaslavsky, Mem. AMS 1975) · Tian, Chen & Ganguli, ICML 2021
(predictor/stop-grad dynamics as implicit whitening) · Tschannen et al., ICLR 2020 (1907.13625)
· Voita & Titov, EMNLP 2020 · von Kügelgen et al., NeurIPS 2021 · Wang & Isola, ICML 2020 · Wen
& Li, NeurIPS 2022 (prediction-head mechanism) · Xu et al., ICLR 2020 (2002.10689,
V-information) · Zamir & Feder (dithered quantization); Ziv, 1985 · Zbontar et al., ICML 2021
(Barlow Twins) · Zhang et al., NeurIPS 2022 (U-MAE) · Zimmermann et al., ICML 2021 (2102.08850).
