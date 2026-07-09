# TRD-π — Tiered Rate–Distortion with a Declared Prior

**A normative framework and audit protocol for self-supervised representation learning**

Consolidated v1.3 · 2026-07-05 · supersedes v1.2 · review lineage, corrections record, and declined-changes register now live in the companion document `TRD_PI_PROVENANCE.md`. (Filename and internal label are aligned from this version onward; the earlier file named v1_3 carried internal label v1.2.)

Claim classes used throughout: **[T]** theorem/proposition, proof in hand · **[L]** lemma, proof in hand · **[PS]** proposition-sketch (strategy clear, bookkeeping unwritten; outside the theorem core) · **[F]** fact imported from cited work · **[D]** design principle · **[P]** falsifiable prediction · **[C]** conjecture, filed as an open problem.

---

## 0 · Problem and thesis

SSL objectives are tuned to benchmarks, and their loss values do not order methods by downstream quality across methods — only within one method's own hyperparameter sweep. Two mechanisms explain this: the **wrong functional** (losses are satisfiable by useless representations — BYOL-family ablations collapse to near-chance while satisfying their objective) and the **wrong space** (losses live at the projector output z while probing happens at the backbone h). An empirical hook makes the honesty gap concrete: on a converged checkpoint whose training statistic reports isotropy — the 16-d projector ablation cell, where low width makes an all-direction search effectively exhaustive (scoped against the method's best 64-d configuration in R6) — the search finds Epps–Pulley 24.9 against a moment-matched null of 1.98 and worst-direction excess kurtosis 3.54 against 0.13 (N = 9469): the trained test reads clean while the marginal is not (measurement M0).

**Thesis.** TRD-π poses representation learning as **tiered rate–distortion against a declared prior**: minimize description length measured against a declared codebook family, subject to linear decodability of a declared battery of invariant factors, with honesty enforced by a worst-direction audit held outside the loss, in the space where downstream probing actually happens.

The stance in one sentence: **we impose the demand and instrument the discovery.** The prior demands that a calibrated marginal and (in the finite-ν case) a preferred basis *exist*; whether the data supply them is measured, never assumed — by the pilot-read selection stage, the seed null, and an audit the loss cannot see (§4, §5).

What the framework buys is a contribution triple: (i) a normative program in which all six desiderata (D0–D5) are jointly statable and jointly operational — we know of no alternative frame with this property (R1, §4); (ii) impossibility and blind-spot results that explain current practice (R2, R4, R6); (iii) a falsifiable audit-and-estimation protocol with pre-registered predictions (§5–§7).

Four layers, one job each: **normative** (this program — what representations should be), **descriptive** (latent distribution matching — what existing methods are: corners of the same Lagrangian, §2), **conditions** (identifiability — when minimizers invert nature, R3), **measurement** (the two-space audit, R4–R6, §5).

---

## 1 · Desiderata and their carriers

**Standing qualifier** (single home; everything below inherits it): sufficiency and invariance are relative to a declared nuisance mechanism G and task family T, never absolute. G is a group action only in the clean setting; in practical SSL it is a stochastic, generally lossy and non-invertible transformation family (crop, blur, masking, jitter), formalized through the content/style decomposition of von Kügelgen et al. '21. "Maximal invariant" is used only in the group case; "invariant content variable U" otherwise; every statement below survives the substitution. G is the framework's one load-bearing ungrounded input (what augmentations do and don't determine is its own literature). The framework converts "choose a loss" into "declare **(G, battery F, π-family, probe tiers V, budget chain B, resolution σ)**" — the factor battery F is part of the declaration (R7a), not a derived object — of which four are estimated or audited downstream, and G and F are inherited from the deployment.

**Label discipline** (single home): TRD-π is self-supervised in its training objective and factor-aware in its governance — F-labels enter at selection (§5, Stage 1) and evaluation/audit (R7c, E5), never in L. Mainstream SSL already consumes the same labels at model selection, silently; TRD-π pre-registers that use and prices it.

Each desideratum below states the want, then its formal carrier in §3.

**D0 · Prior-relative residuality.** Spend rate on the case-specific information a sample transmits, not on re-encoding what is already typical of the domain. *Carrier* (R1, [T]): E_x KL(q(z|x) ‖ π) = I(X;Z) + KL(q_Z ‖ π). I(X;Z) is instance-specific information measured against the encoder's own aggregate marginal q_Z — mutual information does not know the external prior — while KL(q_Z ‖ π) is the calibration gap between that aggregate and the declared codebook; the decomposition forces calibration into the open rather than leaving it implicit. π thereby has two pulls, **descriptive** (a model of the embedding shadow natural data induces) and **normative** (a geometric commitment for probing and identifiability, R8), held apart by the declare → select → train → estimate → audit pipeline (§5). The restriction to invariant content is D1's job, not D0's.

**D1 · Sufficiency.** Every task in the declared G-invariant family remains solvable from the representation. *Carrier*: the reduction of G-invariant tasks to the invariant content U and its declared battery F (R7a); in training, the alignment surrogate (R1, level ii); at selection and audit, tier-1 decodability of F (R1 level iii; R7c).

**D2 · Minimality.** Among sufficient representations, prefer minimal ones — invariance to nuisance is equivalent to information-minimality among sufficient representations (Achille & Soatto '18). *Carrier*: the rate term, measured against the declared π (R1), which is what makes minimality operational rather than bijection-blind.

**D3′ · Hierarchy (conditional).** If deployment declares a budget chain B = (m₁ < m₂ < …), the representation should admit a flag — nested subspaces V_{m₁} ⊂ V_{m₂} ⊂ … — each solving the tier-1 program at its budget. Absent a declared B, D3′ imposes nothing; with a single budget the meaningful object is a subspace (Grassmannian) and any within-subspace permutation is gauge. "Importance" is not a primitive: it is the shadow price of a factor under a task prior; ties collapse the flag into blocks. The unconditional residue — "admits a near-optimal flag" — is a rotation-invariant property of the information spectrum. *Carrier*: R7d (prefix-nested program; exact PCA equivalence in the linear-Gaussian case).

**D4 · Tiered accessibility.** Factor tasks should be linearly decodable (tier 1, probe class V₁); derived quantities need only bounded-depth decodability (tier 2, V₂ ⊇ Φ∘Lin). Sufficiency does not imply linear accessibility; the gap Δ = I_{V₂} − I_{V₁} (Xu's V-information) is measured, never trained. *Carrier*: R7c (the Δ protocol); R6b (the two-space prediction for Δ).

**D5 · Grounded marginal — two numbers, never one.** The embedding marginal is governed by a declared prior family (here ∏ₖ t_ν, unit variance, ν ∈ (4, ∞]; the isotropic Gaussian is the light-tailed ν → ∞ corner). Scale convention: unit coordinate variance is pinned to the dithered variable h̃ at the declared σ, after the declared normalization map; audit-side per-coordinate scale estimates are nuisance standardization (§5, Stage 3). Two distinct members appear and are never conflated: **ν_train**, the member inside the loss, frozen before final training by the pre-registered selection stage (§5, Stage 1); and **ν̂**, the member the audit tests against, estimated after training on a held-out split from loss-independent statistics. The gap between them (reported on the 1/ν scale) is itself a diagnostic: did training land where it aimed. Fit is audited by a characteristic, worst-direction-reported statistic computed outside the loss, against a parametric-bootstrap null drawn from the fitted declared prior (§5, Stage 4). The sparse (finite-ν) variant is gated to the complete regime (R3). Rate role and audit role are never played by the same estimator (R1). *Carriers*: R1 (identity), R4 (why in-loss enforcement is blind), R5 (the audit that isn't), R8 (the fork), §5 (the pipeline).

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
| D2 | no rate beyond sufficiency | R1 rate term at declared σ | per-method entropy surrogates, mutually incomparable | closed-form cross-entropy rate against the declared π |
| D3′ | budgeted nesting on demand | R7d | Matryoshka as an engineering trick | conditional desideratum + linear-Gaussian equivalence [T] + OP-b |
| D4 | factors readable at a glance | R7c Δ protocol | probing literature supplies budget matching | Δ measured, never trained; pass/fail semantics per task class |
| D5 | the marginal you claim is the marginal you have | R4, R5, R8, §5 | mean-slice / moment surrogates inside the loss (blind: R4) | worst-direction audit outside the loss; ν_train / ν̂ discipline |

### Desiderata × results matrix (✓ = the result carries or protects the desideratum)

|  | R1 | R2 | R3 | R4 | R5 | R6 | R7 | R8 |
|---|---|---|---|---|---|---|---|---|
| D0 | ✓ |  | ✓ |  |  |  |  | ✓ |
| D1 |  |  |  |  |  | ✓ | ✓ |  |
| D2 | ✓ |  |  |  |  | ✓ |  |  |
| D3′ |  | ✓ |  |  |  |  | ✓ |  |
| D4 |  | ✓ |  |  |  | ✓ | ✓ |  |
| D5 | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |  | ✓ |

---

## 2 · Setup, notation, and the map of existing methods

x ∼ p(x); G the declared nuisance mechanism; U the invariant content (maximal invariant when G is a group). Backbone h = f(x) ∈ ℝ^d; head z = g(h) ∈ ℝ^{d′}; dithered backbone h̃ = h + σε, ε ∼ N(0, I) (σ = declared resolution; every rate/marginal number carries its σ; only equal-σ comparisons are meaningful). π ∈ P the declared prior family; ν_train vs ν̂ per D5. F the declared factor battery. V₁ = linear probes, V₂ = bounded-depth probes; I_V = predictive V-information (Xu et al. '20). q_Z = the embedding marginal. C⁽⁴⁾ denotes the fourth-cumulant tensor. K is defined once: the audited ambient dimension of whichever space a statistic lives in — K = d at h, K = d′ at z. s★ denotes generative source latents (R3 only; never a learned quantity). Density-level hygiene: all density-level statements (rates, KLs, entropies) for deterministic encoders are understood after the declared dither σ; the un-dithered pushforwards themselves may be singular (R6d).

**Existing methods as corners of the program** (descriptive layer; per-method derivations from the latent-distribution-matching reading, Mikulasch et al. '26). Reading rule: the table is an idealized, population-level reading of these objectives under common normalizations, not a claim that the implemented losses literally optimize these exact divergences.

| Method | Alignment | Rate/entropy surrogate | Implied π | Space | Residual symmetry at optimum |
|---|---|---|---|---|---|
| LeJEPA | prediction | mean-random-slice Epps–Pulley | isotropic Gaussian | z (proj. dim 64 ablated best) | full O(K) |
| VICReg | MSE | variance hinge + covariance penalty | 2nd-moment factorial | z (8192 expander → R6d) | signed permutations (setwise); per-solution gauge in equal-variance blocks |
| Barlow Twins | cross-corr. | correlation → I | whitened | z | full O(K) at exact optimum |
| InfoNCE / uniformity | contrastive | KDE-entropy / uniformity | uniform on sphere | z | full O(K) |
| BYOL / SimSiam | prediction | implicit (EMA/predictor — implicit spectral whitening, Tian et al. '21) | implicit | z | full O(K) (alignment part) |
| DINO / SwAV | cross-view prototype assignment | Sinkhorn equipartition / centering + sharpening | (near-)uniform over prototypes — a declared prior over assignments, arguably the cleanest pre-existing instance of the declared-prior reading | z (prototype head) | prototype permutations |
| MAE | pixel reconstruction | — | pixel-space generative; no invariance tier — listed for contrast, not as a corner of the invariant program | input | — |

Every corner is a projection of the program: fix π implicitly, drop the audit, collapse the tiers, regulate z instead of h. TRD-π is what remains when nothing is dropped, and it differs from every corner on four labeled axes: **(1)** worst-direction pressure where they average (R4–R5); **(2)** a declared family with pre-registered ν_train and separately audited ν̂ where they imply a fixed target (R8, §5); **(3)** placement at h, where probing happens, where they regulate z (R6); **(4)** an audit held outside the loss where they trust the training statistic (§5).

---

## 3 · Results

### R1 · The rate–prior identity, the canonical objective, and the compression reading — serves D0, D2, D5

**[T]** For any encoder q(z|x) and prior π (densities defined): E_x KL(q(z|x) ‖ π) = I(X; Z) + KL(q_Z ‖ π). Proof: add and subtract log q(z). (Hoffman & Johnson '16; the ≥-form is the variational information-bottleneck bound, Alemi et al. '17.)

**[F]** Companion identity, same one-line proof: E[−log π(h̃)] = h(q_h̃) + KL(q_h̃ ‖ π) — the cross-entropy rate term already contains a compression part (the differential entropy of the dithered marginal) and a calibration part. This is what licenses reading §4's marginal term as re-weighting calibration pressure rather than introducing a new quantity.

**[T]** With the dithered encoder q(h̃|x) = N(f(x), σ²I): Rate(σ) = const(σ) + E[−log π(h̃)] — the rate term is the cross-entropy of dithered embeddings under the declared prior. Instances: π = N(0, I) gives the embedding-norm penalty E‖h̃‖²/2; π = ∏ t_ν (unit variance) gives ((ν+1)/2)·Ʃₖ log(1 + h̃ₖ²/(ν−2)) — up to additive constants, unit-variance coordinates assumed — a log-growing robust penalty whose score/influence function ψ(z) = (ν+1)z/((ν−2)+z²) redescends. The prior's tails and the penalty's influence function are the same object: the declared codebook made tangible. (Dithered-quantization classics: Ziv '85; Zamir & Feder.)

**Compression provenance.** The dithered rate term is precisely the learned-entropy-model objective of neural compression (Theis et al. '17; Ballé et al. '17, '18; Minnen et al. '18), with additive noise as the standard quantization proxy. The contrast is the point of D0's "two pulls": compression learns π jointly with the encoder — a purely descriptive prior that chases the code — whereas TRD-π **declares the family, pre-registers the trained member, and audits the fit**, preserving the normative pull. TRD-π's rate machinery is therefore not exotic; its governance of π is what is new.

**Canonical objective decomposition.** In the ideal program, anti-collapse is not a primitive: the tier-1 distortion constraint forbids collapse and rate-minimization supplies the collapse pressure. Three levels, never conflated. **(i)** In the ideal constrained program, tier-1 sufficiency is enforced directly on the declared battery F. **(ii)** In the self-supervised implementation, F labels are unavailable during training, so that constraint is replaced by view alignment — a label-free surrogate (Federici et al. '20) whose validity is itself conditional: alignment stands in for sufficiency only jointly with the marginal term (alone it is satisfiable by collapse, §4) and only insofar as view-shared content covers U (the G declaration again). **(iii)** The battery enters through selection and audit (§5 Stage 1; R7c; E5) — never through the unsupervised loss, unless factor labels are in fact available, in which case a direct tier-1 term is admissible. Hence: **L = view alignment at z (surrogate for the tier-1 constraint) + cross-entropy rate at h̃ + an entropy-side marginal surrogate (compensating the alignment–constraint slack) — with the audit entirely outside L.** This derives the latent-distribution-matching reading of existing methods (alignment + entropy estimator) instead of assuming it; the equilibrium among the three forces is made explicit in §4 ("who fights whom").

**[D] Role split**: any consistent surrogate may shape the optimum (rate role); the claim "marginal ≈ π" is made only by a characteristic, worst-direction, held-out-settings audit (audit role). The split does not prevent gaming; it makes gaming visible by separating the training objective from the held-out audit — a tripwire, not a guarantee.

### R2 · What objectives cannot select (symmetry inventory) — serves D3′, D4, D5

**[T]** The orthogonal-symmetry inventory of standard objectives at the minimizer level:

1. **Full O(K)**: pairwise-distance alignment + any rotation-equivariant discrepancy to a rotation-invariant π (InfoNCE/uniformity with the sphere-uniform target, BYOL-style alignment, LeJEPA's sliced discrepancy to an isotropic Gaussian). Minimizer sets are closed under global rotations.
2. **Full O(K) for Barlow Twins at its exact optimum**: correlation = I is whitening, and whitening is rotation-invariant — the "decorrelation method" is, at optimum, exactly as basis-blind as the isotropic methods.
3. **Signed permutations — and only signed permutations — for VICReg** at the level of the minimizer family: they are the sole orthogonal maps conjugating every admissible diagonal covariance to a diagonal one. A given minimizer with spectrum diag(λ) admits additional gauge — rotations inside its equal-λ blocks, the stabilizer ∏ O(m_j) of that solution — but this freedom is solution-dependent, not a symmetry of the loss. Remark (counterexample that killed the naive claim): VICReg's hinge only enforces per-coordinate std ≥ γ; take a minimizer with Cov = diag(1, 4) — a 45° rotation produces off-diagonal covariance and re-activates the penalty, so the minimizer set is not O(K)-closed, and it is not {Cov = I} either.

**Takeaway** (scoped to population minimizers). At the level of population minimizer symmetries, **no standard objective by itself identifies sparsity, importance ordering, or basis structure beyond at most second-order decorrelation with degeneracy gauge** — anything more in a trained model is architecture/optimizer implicit bias, not the loss; whether implicit bias already delivers axes in practice is an empirical question decided by the seed null (§6.3). Tier-1 linear decodability is itself rotation-invariant, so isotropy is safe for D4; D3′ (when active) and monosemantic-D5 require an explicit symmetry break: a non-rotation-invariant π (R3) or prefix nesting (R7d). Scope: this is an identifiability-from-the-objective statement about the loss landscape, not about SGD dynamics.

### R3 · When minimizers invert nature: affine → signed permutation — serves D0, D3′-frame, D5

**[T]** Assume **(A1)** generative source latents s★ ∼ ∏ₖ pₖ mutually independent, at most one Gaussian, x = g★(s★) injective; **(A2)** the method achieves affine identifiability f(x) = A s★ + b, A invertible — the conclusion of the LDM predictive-SSL theorem under its assumptions (scoped to predictive SSL on temporal data), of Roeder et al. '21-style linear identifiability of discriminative families, of the nonlinear-ICA line TCL/PCL/GCL (Hyvärinen & Morioka '16, '17; Hyvärinen et al. '19) — the actual ancestry of "affine identifiability from self-supervision" — or of Zimmermann-/iVAE-class results under theirs; **(A3)** the objective drives the learned marginal to be factorial and non-Gaussian. Then **A = ΠΛ**: recovery up to permutation and per-coordinate sign/scale. Engine: Comon '94 / Darmois–Skitovich.

**[L] (Pairwise sufficiency in this regime.)** If y = As★ as above and the coordinates of y are pairwise independent, then A = ΠΛ. Proof: Darmois–Skitovich is a two-form statement; apply it per pair — every non-Gaussian source can load on at most one output row; pairwise independence implies uncorrelatedness, which forces the (≤1) Gaussian column to a single row as well; invertibility does the rest. Hence pairwise-HSIC penalties are principled surrogates at the optimum, inside the complete linear-ICA regime. Outside that regime, pairwise HSIC is diagnostic, not sufficient. Quantitative robustness (ε-independence ⟹ δ(ε)-close to a signed permutation) is open [C → OP-d].

**Takeaways.** (1) **The inversion**: the isotropic Gaussian is the unique marginal that forfeits basis identifiability (the classical ICA obstruction) — so minimax probe-neutrality and identifiability pull in provably opposite directions. The fork's two arms carry different epistemic weight — this theorem versus R8(a); the statuses are stated once, at R8. (2) **Regime gate**: the theorem lives in the complete regime (embedding width ≥ factors active at this budget). Undercomplete, the rate-optimal code superposes (Elhage et al. '22) and a factorial push degenerates into sparse-coding pressure (Olshausen & Field '96 territory) — the sparse machinery is deployed only after §6.1's regime check. (3) For dependent factors, the primary fallback is sparsity-based identifiability (Lachapelle et al. '22), not independence.

### R4 · Random-slice blindness, made quantitative — serves D5

**[L] (Rank-k attenuation.)** If all non-Gaussian structure of Z (E Z = 0, Cov Z = I) lies in a k-dimensional subspace with independent Gaussian complement, then for θ uniform on S^{K−1}: κ_r(θᵀZ) = α^r κ_r(v_θ) with α² ∼ Beta(k/2, (K−k)/2), so E[α⁴] = k(k+2)/(K(K+2)) and generally E[α^{2m}] = ∏_{j<m}(k+2j)/(K+2j). Proof: cumulant additivity/homogeneity + Beta moments.

**[T] (Tensor-general form; no subspace assumption.)** E_θ[κ₄(θᵀZ)] = 3·Ʃ_{ij} C⁽⁴⁾_{iijj} / (K(K+2)), a trace-type contraction of the fourth-cumulant tensor C⁽⁴⁾, while the worst slice is the tensor's injective norm — the mean/max ratio can reach order K². Proof: uniform fourth moments E[θᵢθⱼθₖθ_l] = (δδ-symmetrization)/(K(K+2)) contracted against the symmetric tensor.

**[PS] (Signal reading, quadratic regime.)** Weighted-CF statistics expand quadratically in cumulant deviations near Gaussian, so a mean-over-slices loss carries order-(k/K)^r pressure against an order-r rank-k defect while worst-direction pressure is O(1); resampling directions each step restores unbiasedness, not signal-to-noise (Meckes's waiting-time bounds are the citation-grade form).

**Consequences.** Mean-random-slice surrogates — the deployed SIGReg (which documentedly replaces its theorem's max over directions with an average to avoid sparse gradients), and the whole EP ≈ sliced-MMD family (KerJEPA; fixed-kernel power decay, Ramdas et al. '15) — are **structurally under-pressured on low-rank defects**. M0 arithmetic: worst-direction κ₄ = 3.54 at K = 16 predicts mean-slice κ₄ ≈ 0.037 if the defect were rank-1; the aggregate EP = 24.9 is consistent with rank > 1 — the defect-rank estimator k̂ (fit the slice-κ₄ spectrum to the Beta prediction) decides [P]. What a finite-budget audit can actually detect is calibrated, not assumed: §6.0 and OP-h.

### R5 · The audit that isn't blind — serves D5

**[T]** The population max-sliced discrepancy sup_θ d₁(θ♯q, θ♯π) is characteristic (Cramér–Wold) and, for d₁ = W_p, a metric (Deshpande et al. '19).

**[D]** Train the mean-slice term (bounded, stable gradients) plus a soft-max over a persistent adversarial direction bank — formally a low-capacity instance of a learned deep-kernel two-sample statistic (Liu et al. '20) — with fresh random slices for exploration. The bank raises the cost of hiding a rank-k defect from O(K^r) to O(bank lag); it does not eliminate hiding (minimax chasing is possible [C → OP-g]; mitigations: bank momentum, random restarts, and the held-out audit as referee).

**Principle (evidence, not certificates — hard reporting rule).** The population supremum is characteristic; finite banks and finite searches are capacity-limited audits whose failures are evidence, not certificates of equality. The audit therefore reports "the worst witness found under a declared search budget," together with a matched-search null generated from the fitted declared prior (§5, Stage 4) — and never reports "passed." Small discrepancies are evidence of fit under that budget. (Required for self-consistency: the framework's own critique of "failing to reject ≠ accepting" applies to its audit too.)

### R6 · Two spaces: what crosses the head — serves D1, D2, D4, D5

Let z = g(h). **(a) [F]** I(X; z) ≤ I(X; h): sufficiency established at z holds at h for free — the only desideratum that crosses the head automatically, in the right direction. **(b) [F]** Tier-1 at z is only tier-(lin∘g) at h: training accessibility at z predicts a nonzero Δ at h. **(c) [T]** For bi-Lipschitz g with constants (ℓ, L): W_p(q_h, (g^{-1})♯π) ≤ ℓ^{-1} W_p(q_z, π); with an unconstrained head, D(q_z, π) = 0 constrains q_h essentially not at all. **(d) [T] (Expander singularity.)** If dim(z) > dim(h) with a deterministic encoder (VICReg's 8192-d expander, DINO prototypes), q_z is supported on a ≤ dim(h)-dimensional set: KL(q_z ‖ N(0, I)) = +∞ and "z ∼ isotropic Gaussian" is unsatisfiable in principle — only moment/slice surrogates can be driven down. Dither repairs well-posedness. **(e″) [T case i / PS case ii]** Let q_h be absolutely continuous with finite entropy, g L-Lipschitz (hence a.e. differentiable, Rademacher), injective on supp(q_h) up to null sets, det Dg ≠ 0 a.e., H(z) finite. Then, dimension-preserving: H(h) ≥ H(z) − d·log L. Remark (counterexample that killed the naive claim): with h = (u, v), H(v) → −∞ and z = u, a dimension-reducing head *increases* differential entropy — so when dim(z) < dim(h), entropy at z certifies spread only of the pushforward; it gives no lower bound along the fibers g^{-1}(z), the unread degrees of freedom of h.

**Takeaways.** **Placement rule: alignment at z; rate + marginal audit + tier-1 probing at h; spectrally normalize the head.** Projector-space regularity is not backbone-space regularity unless the head is dimension-preserving and suitably invertible; a dimension-reducing head can be healthy at z while the backbone retains collapsed or pathological unread degrees of freedom — and the best-performing LeJEPA projector output is 64-dimensional against a much wider backbone (M0's 16-d cell was picked for exhaustive searchability, but even the best configuration is exactly case (ii)).

**Head-conditioning exposure, handled structurally.** Spectral normalization controls the head's upper constant L; nothing in L constrains σ_min(g), so the head can contract directions, and content living in contracted fibers receives no alignment pressure. Three structural answers: nothing load-bearing rides the z→h transfer — every certified quantity is measured at h; the transfer claim (c) assumes bi-Lipschitz explicitly; and head conditioning (σ_min(g), effective rank of g) is part of the reported battery (§5), so the failure mode is watched rather than assumed away. An h-space alignment term was considered and rejected: it would re-import the augmentation-variant content the head exists to absorb.

**Candidate mechanism [P]** for the cross-method failure: z-space losses sit behind method-specific warps g, so cross-method loss comparisons compare warps (within one method the warp is stable, which is why within-sweep correlations survive); prediction E3 follows. Where implicit entropy estimators (BYOL's EMA/predictor) attach in the placement rule is the concrete open sub-problem of the head-composition program [→ OP-i].

### R7 · Tiers and hierarchy: train the tier-1 surrogate, measure the rest — serves D1, D3′, D4

**(a) [F + declaration]** "All G-invariant tasks" reduces to measurable functions of the invariant content U (Dubois et al. '21). The next step is a declaration, not a theorem: when U admits a finite factor coordinate system whose span covers the declared task family T, sufficiency is operationalized as tier-1 decodability of that finite battery F. Finiteness enters through the declared tuple (§1) — the program is finite because F is — and tasks outside span(F) are explicitly invisible to D1 (§8). **(b) [PS — outside the theorem core]** If derived tasks compose factor tasks through bounded-complexity φ, tier-1 on factors yields tier-2 on derived tasks for V₂ ⊇ Φ∘Lin, with error controlled by φ's modulus. **(c) [D + P]** The converse fails (V-information is designed to violate data processing: nonlinear objectives can bury factors), so: optimize only the tier-1 surrogate (a direct battery term enters L only when factor labels exist); measure Δ(t, space) = I_{V₂} − I_{V₁} with budget-matched probes (equal sample/compute for both tiers — the probing literature's answer to probe-capacity confounds: Voita & Titov '20, Hewitt & Liang '19). Pass/fail semantics: Δ ≈ 0 required on the factor battery, unconstrained on derived tasks; inflated Δ at h for models trained at z is predicted via (b) + R6(b). **(d) [T for the linear-Gaussian case]** D3′'s estimator: the prefix-nested program equals PCA ordering under linear-Gaussian assumptions (Eckart–Young; recovered exactly by nested dropout, Rippel et al. '14); the general practical estimator is Matryoshka-style prefix replication (Kusupati et al. '22); general prefix-optimality conditions are open [C → OP-b].

**Scope declaration (the program's largest known boundary).** Everything above is stated for flat vector representations, and the invariant-content reduction assumes tasks are functions of one global factor vector. Object-centric and relational tasks (counting, binding) need slot/set representations and a tier calculus over permutation-invariant readouts — outside this theory's scope and filed [C → OP-e].

### R8 · The prior fork, priced, estimated, directional — serves D0, D5

**(a) [PS, pending appendix]** Under the reference isotropy-optimality analysis (Balestriero & LeCun '25, §3), our reading of the lemma structure is that the linear-probe results depend on the embedding law only through its covariance — hence every unit-covariance marginal, including factorial heavy-tailed ones, would be minimax-equivalent for linear probing. The exact lemma forms are to be quoted and the constant re-derived in an appendix before submission; nothing in the theorem core depends on (a). This is the single home of (a)'s status; every use below inherits it silently.

**(b) [F]** Gaussian uniqueness enters only through nonlinear local-smoother bias, proportional to a roughness functional J(p) minimized by the isotropic Gaussian under a scalar covariance constraint.

**(c) [L]** Unit-variance Student-t: I(ν) = ν(ν+1)/((ν+3)(ν−2)), excess kurtosis 6/(ν−4); at ν = 5, I = 1.25 and κ₄ = 6. Calibrated statement: in the matched Fisher-information calculation, the t₅ prior carries a 25% larger constant than the Gaussian; whether this transfers quantitatively to the reference paper's stated ISB constant depends on matching their exact functional (pending re-derivation), and adaptive-bandwidth probe pipelines plausibly sit below it.

**(d) The fork, made directional (conditional on (a)).** At tier 1 — the tier this framework trains — all unit-covariance members of the family are minimax-equivalent under (a). Moving from ν = ∞ to finite ν therefore costs nothing in minimax linear-probe terms while purchasing basis identifiability in the complete regime (R3). The price is confined to finite-sample constants ((c): ≈25% at ν = 5 in the matched Fisher calculation) and tier-2 local-smoother bias ((b)). The default direction therefore reverses: when sparse/leptokurtic symptoms are present in the complete regime (§5 Stage 1 / §6.2), **finite ν is the preferred design point and the Gaussian corner the fallback** — not vice versa; absent (a), the fork remains an empirically motivated design choice priced by (b)–(c). Either way the family nests the Gaussian at the boundary: in the natural estimation parameter 1/ν ∈ [0, 1/ν_min], the Gaussian is the boundary point 0; testing Gaussianity within the family is a boundary-hypothesis problem — one-sided tests, chi-bar-squared LRT asymptotics, confidence intervals reported on 1/ν.

**(e)** Coupled design choice: heavy-tailed targets are safe to train against because the grounding distances are bounded characteristic (CF-type) statistics. Sparse-task rate separation remains a conjecture [C → OP-a].

**(f) What the audit means at each corner.** With a declared fixed ν — including the Gaussian corner — Stage 4 tests a **simple null**: training hit the declared codebook (the strong normative claim). With an estimated ν̂ it tests a **composite null**: the marginal lies in the declared family (an adequacy claim), kept honest by the bootstrap's per-draw re-estimation (§5). D5's content shifts across the fork, and every report states which claim is being made.

---

## 4 · The objective

With spectrally normalized head g and dithered backbone h̃ (dither applied in the rate/marginal terms only):

**L =** align_z(views) — *distortion surrogate, at z*
**+** λ_rate · E[−log π_{ν_train}(h̃)] — *rate (closed form, R1), at h̃*
**+** λ_marg · [ mean-slice CF(h̃ → π_{ν_train}) + η · bank-max ] — *marginal surrogate + pressure (R5)*
**+** gated: β · pairwise-HSIC(h̃) — *complete regime only (R3), after §6.1*
**+** Matryoshka prefix replication — *iff a budget chain B is declared (D3′, R7d)*

All in-loss marginal terms target the single frozen member π_{ν_train} (§5, Stage 1); at ν_train = ∞ the objective degrades gracefully to the isotropic-Gaussian corner. The audit (§5) is never a term in L. Neither is the battery F: during unsupervised training, sufficiency acts only through the alignment surrogate; F itself enters at selection (§5, Stage 1) and at evaluation/audit (R7c, E5) — R1's three-level separation.

**Who fights whom (the objective's equilibrium).** Alignment is invariance pressure and is satisfiable by collapse. The rate term is compression pressure and rewards collapse — after dithering, a point mass at the origin minimizes cross-entropy up to the σ floor. The marginal term is therefore **the only anti-collapse force in L, and simultaneously the calibration force** (companion identity, R1): it pushes q_h̃ toward the declared full-rank member. Anti-collapse in TRD-π is not a primitive but a consequence of demanding that the aggregate look like π — which is why auditing the marginal, rather than trusting the loss, is load-bearing rather than decorative.

**Who picks the basis (the demand/discovery split).** The finite-ν prior imposes that a preferred basis *exist* — the explicit symmetry break R2 shows no standard objective supplies — not *which* basis; which one is the data's job. Inside the affine-identifiability regime (R3's A2), the gated independence term's floor is attainable only at source-aligned axes up to signed permutation (Darmois–Skitovich, R3's lemma): when a factorial non-Gaussian representation exists, the axes are discovered; when none exists, the floor is unreachable and Stage 4 reports the misfit — the prior cannot manufacture sparsity that survives its own audit. Outside A2, a sufficiently flexible encoder can transport nearly any law onto ∏ t_ν coordinates, and the framework claims no factor semantics for the axes. Discovery is instrumented three ways: Stage-1 symptoms are read on the ν = ∞ pilot — a model whose rotation-invariant loss could not have imposed them — in the Varimax basis, a canonical rotation rather than the native one; E6 demands an identifiability gain at matched tier-1 accuracy; and the seed null (§6.3) separates loss-induced from optimizer-induced axes. **We impose the demand and instrument the discovery.**

---

## 5 · Audit & estimation protocol v3.0 — declare → select → train → estimate → audit

**Rule 0 (the pipeline).** One symbol never does two jobs: ν_train (in-loss) and ν̂ (audit-side) are distinct by construction (D5), and the five stages are time-ordered so each depends only on stages upstream of it.

**Stage 0 · Declare** (before any training): the family ∏ t_ν, ν ∈ (4, ∞]; a finite selection grid N_grid ⊂ (4, ∞]; battery F; tiers V; chain B; resolution σ; search budgets and estimator settings for every later stage.

**Stage 1 · Select ν_train** (pre-registered, pilot-based): train the isotropic corner ν = ∞ as the pilot — it needs no estimate (∞ is a declared constant) and this run doubles as the baseline. Run the fork experiment (§6.2) on the pilot's estimation split. Symptoms absent → ν_train = ∞. Present → ν_train = the grid point nearest the kurtosis-matched value ν = 4 + 6/κ̂₄ read in the Varimax basis. ν_train is now frozen; it is a function of the pilot only, never of the model it will train.

**Stage 2 · Train** the final model with π_{ν_train} on the training split. ν does not move during training.

**Stage 3 · Estimate ν̂** on the estimation split of the final representation, from loss-independent statistics (probe-anchored bases, worst-direction searches — never the mean-slice statistics the loss optimized). Per-coordinate scales are nuisance standardization, not parameters of the null: fit a diagonal standardization map on the estimation split, freeze it, apply it before every audited statistic, and report the fitted scales as calibration diagnostics — their deviation from 1 is itself a finding. The declared family stays unit-variance; auditing a scaled family ∏ₖ sₖ·t_ν would be a different, weaker declaration. Report ν̂ vs ν_train on the 1/ν scale: the did-training-land-where-it-aimed diagnostic.

**Stage 4 · Audit** on the third split, disjoint from both. Null: a parametric bootstrap that replicates the whole estimate → audit pipeline on synthetic data — draw estimation-split- and audit-split-sized samples from π_ν̂; re-estimate ν and the standardization map on the synthetic estimation part; run the identical adversarial search, same compute and estimator settings, on the synthetic audit part against the re-fitted member; the distribution of the resulting maxima is the null. The per-draw re-estimation is the Lilliefors-type correction that keeps a composite null honest — without it the audit is systematically lenient. Because θ♯∏ t_ν has no closed form (the family is not convolution-closed), per-direction reference laws are themselves Monte Carlo from the fitted prior under the declared budget. The moment-matched Gaussian null survives only (i) as the exact simple null at the ν = ∞ corner and (ii) as a separate Gaussianity diagnostic.

**Rules 1–7.** 1. Double split for directions: worst directions found on sub-split A, statistics reported on sub-split B, both inside the audit split. 2. Matched search on the null: identical compute and estimator settings. 3. Budget-matched tiers: V₁ and V₂ probes trained under equal budgets before Δ is read. 4. Regime check (step 0 of any sparse deployment): effective active-factor count vs width (participation ratio, k̂, spectral monitors). 5. Seed null (§6.3): axis-kurtosis + Varimax-sparsity across ≥ 5 seeds under the isotropic objective. 6. σ disclosure: every rate/marginal number carries its dither σ; equal-σ comparisons only. 7. Both spaces: every audited statistic is reported at h and at z.

**Reported battery**: worst-direction EP and κ₄ with bootstrap CIs · defect rank k̂ · minimum-detectable amplitude at the declared budget (from §6.0) · spectral monitors (RankMe, α-ReQ, LIDAR) · the tier-gap profile Δ(t, space) · head conditioning: σ_min(g) and effective rank of g (R6 exposure monitor) · fitted standardization scales (Stage 3 diagnostic) · ν_train, ν̂, and a CI on 1/ν · which null (simple / composite) the audit ran.

---

## 6 · Decision experiments

**6.0 · Audit calibration by defect injection** (run once per (K, N, budget) regime). Sample from the declared prior; inject rank-k non-Gaussian defects at controlled amplitude (kurtosis bumps / sparse mixtures along k random directions); sweep (k, K, N, search budget); report ROC and minimum-detectable-amplitude curves for worst-direction vs mean-slice statistics, with the §5 bootstrap as the null. Output: the audit's power curve — what the instrument can and cannot see at the declared budget. This converts R4 from an asymptotic argument into a calibrated instrument and supplies Stage 4's error bars. Theory companion: OP-h (Nadjahi et al. '20 sliced-divergence rates; Meckes waiting times). §6.0 precedes any audited claim.

**6.1 · Regime check.** Estimate active-factor count vs embedding width (protocol rule 4). Gates everything sparse.

**6.2 · The fork experiment — the selection stage** (§5, Stage 1). Frozen pilot h → linear probes on the battery F → Varimax rotation of the probe-weight matrix → test (i) sparsity gain vs a random-rotation null and (ii) coordinatewise excess kurtosis in the Varimax basis. Theory hook: Varimax performs valid inference precisely under leptokurtic factors (Rohe & Zeng), so (i) and (ii) are two symptoms of the same sparse world — exactly the quantities R3 needs and the Gaussian world nulls. Both null ⟹ ν_train = ∞ for this data regime; either present ⟹ finite ν_train from the declared grid per §5, with identifiability per R3.

**6.3 · Seed null** (protocol rule 5): decides implicit-bias vs explicit symmetry breaking.

**6.4 · Spectral-norm rerun** (most informative per unit cost): retrain any baseline with a spectrally normalized head and test whether z↔h statistic correlation rises (R6's placement claims made cheap to falsify).

---

## 7 · Falsifiable predictions (pre-registered E-cards)

**E1.** z-statistics correlate with downstream within-method and decorrelate across methods; shared-estimator h-statistics partially restore cross-method ordering; spectrally normalized heads raise z↔h statistic correlation. Add k̂ per cell; check M0's mean-slice κ₄ against the Beta-moment prediction.

**E2.** Layer-wise "guillotine" curves read as depth-wise rate-allocation profiles; a Matryoshka-trained variant flattens the drop.

**E3.** Recomputing each method's two decomposition terms (alignment; entropy proxy) with one shared estimator at h, same backbone architecture, partially recovers the cross-method ordering that z-space losses cannot provide.

**E4.** Marginal statistics at z behave as head-warp artifacts under unconstrained heads; expander-head cells show the R6(d) singularity signature (effective support dim ≤ dim h).

**E5.** Δ ≈ 0 on the factor battery, unconstrained on derived tasks; models with tier constraints trained at z show inflated Δ at h.

**E6 (the selection stage is itself falsifiable).** In regimes where §6.2 fires on the pilot, ν̂ < ∞ replicates across seeds and the finite-ν_train model shows a positive identifiability gain (factor alignment in the Varimax basis) over the isotropic corner at matched tier-1 accuracy; in regimes where §6.2 nulls, no such gain appears and finite-ν training buys nothing.

---

## 8 · Limitations — an index, each stated once at its home

- **Scope**: flat vectors; tasks as functions of one global factor vector (R7 scope declaration; object-centric extension → OP-e).
- **Grounding**: G is declared, not grounded; F is declared, not derived — tasks outside span(F) are invisible to D1 (§1, R7a).
- **Audit epistemics**: tripwire, not guarantee (R1 role split); finite banks give evidence, not certificates (R5); the bank minimax game can cycle (OP-g).
- **Pipeline cost and risk**: one pilot run; grid selection can misfire when the pilot's regime differs from the final model's — mitigated by the grid itself and the ν̂-vs-ν_train gap diagnostic (§5).
- **Claim semantics**: the composite-null audit (estimated ν̂) makes a weaker claim than the simple-null audit (declared ν); every report states which (R8f). Bootstrap validity at the Gaussian boundary rests on the one-sided / chi-bar-squared asymptotics of 1/ν = 0 (R8d).
- **Claim status**: R8(a) — and hence R8(d)'s direction — is [PS] pending the appendix re-derivation (R8a); the other [PS] items (R4 signal reading, R7b, R6e″ case ii) are outside the theorem core.
- **Head**: spectral normalization bounds only the upper constant L; σ_min(g) is monitored, not constrained (R6).
- **Labels**: selection and audit consume F-labels; the label-free claim is confined to the training objective (§1).
- **Regime estimation** (rule 4) is crude; the superposition boundary is itself open (OP-f).

---

## 9 · Open problems

**OP-a** sparse-task rate–distortion separation as a theorem. **OP-b** prefix-optimality beyond linear-Gaussian. **OP-c** bank search-time guarantees for rank-k defects (Meckes waiting-time as the tool). **OP-d** quantitative Darmois–Skitovich (robust R3). **OP-e** object-centric tier calculus. **OP-f** the budget/sparsity boundary where monosemantic frames stop being optimal. **OP-g** bank minimax dynamics (convergence vs cycling). **OP-h** finite-budget detection theory: minimum detectable amplitude for rank-k defects under a declared search budget (companion to §6.0; Nadjahi et al. rates + Meckes waiting times as tools). **OP-i** placement of implicit entropy estimators (EMA/predictor) in the two-space rule (formerly OP-17′).

---

## References (anchor list)

Achille & Soatto, JMLR 19(50), 2018 (1706.01350) · Alemi et al., ICLR 2017 (1612.00410); ICML 2018 (Fixing a Broken ELBO) · Babu & Rao, Sankhyā 2004 (goodness-of-fit with estimated parameters) · Balestriero & LeCun, 2025 (2511.08544, LeJEPA) · Ballé, Laparra & Simoncelli, ICLR 2017; Ballé et al., ICLR 2018 (hyperprior); Minnen et al., NeurIPS 2018; Theis et al., ICLR 2017 (neural compression / learned entropy models) · Bordes et al., TMLR 2023 (guillotine) · Caron et al., NeurIPS 2020 (SwAV); Caron et al., ICCV 2021 (DINO) · Chen & He, 2021 (SimSiam) · Comon, Signal Processing 36, 1994 · Deshpande et al., CVPR 2019 (max-sliced W) · Diaconis & Freedman, Ann. Statist. 12(3), 1984; Dümbgen & Zerial, 2011; Meckes, 2009/2010 (quantitative, waiting time) · Dubois et al., NeurIPS 2021 (2106.10800); NeurIPS 2022 (2209.06235) · Elhage et al., 2022 (toy models of superposition) · Epps & Pulley, Biometrika 1983 · Federici et al., ICLR 2020 (2002.07017) · Garrido et al., ICML 2023 (RankMe); Agrawal et al., NeurIPS 2022 (α-ReQ); Thilak et al., 2023 (LIDAR) · Gorham & Mackey, 2017; Liu, Lee & Jordan, 2016; Chwialkowski et al., 2016 (KSD family; complementary audit) · Grill et al., 2020 (2006.07733, BYOL) · Grünwald, 2007 (MDL) · Hewitt & Liang, EMNLP 2019 · Hoffman & Johnson, 2016 (ELBO surgery) · Hyvärinen & Morioka, NeurIPS 2016 (TCL); AISTATS 2017 (PCL); Hyvärinen, Sasaki & Turner, AISTATS 2019 (GCL) · Khemakhem et al., AISTATS 2020 (iVAE) · Kusupati et al., NeurIPS 2022 (Matryoshka) · Lachapelle et al., 2022 (sparse identifiability) · Li et al., 2021 (SSL-HSIC) · Lilliefors, JASA 1967 (tests with estimated parameters) · Liu et al., ICML 2020 (deep-kernel two-sample) · Mikulasch et al., 2026 (2605.03517, latent distribution matching) · KerJEPA, 2025 (2512.19605) · Nadjahi et al., NeurIPS 2020 (statistical properties of sliced divergences) · Olshausen & Field, Nature 1996 · Ramdas et al., AAAI 2015 · Rippel et al., ICML 2014 (nested dropout) · Roeder, Metz & Kingma, ICML 2021 (linear identifiability of learned representations) · Rohe & Zeng, JRSS-B (Varimax inference) · Saunshi et al., ICML 2022 (2202.14037) · Tian, Chen & Ganguli, ICML 2021 (predictor/stop-grad dynamics as implicit whitening) · Tschannen et al., ICLR 2020 (1907.13625) · Voita & Titov, EMNLP 2020 · von Kügelgen et al., NeurIPS 2021 · Wang & Isola, ICML 2020 · Xu et al., ICLR 2020 (2002.10689, V-information) · Zamir & Feder (dithered quantization); Ziv, 1985 · Zbontar et al., ICML 2021 (Barlow Twins) · Zimmermann et al., ICML 2021 (2102.08850).
