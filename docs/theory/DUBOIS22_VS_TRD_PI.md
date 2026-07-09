# Dubois '22 (Idealized Representations) × TRD-π — the scope conditions, side by side

**DRAFT for discussion (Claude, 2026-07-08, written at Berker's request) — NOT agreed; nothing
here is a takeaway.** Companion to `verification/anchors_dubois_xu.md` (all quotes marked
[verified] live there); quotes new to this note were re-fetched from ar5iv 2209.06235 on
2026-07-08 and are marked ⟨fetched⟩. Claim tags follow the framework convention ([F] imported
fact · [PS] proposition-sketch · [our reading]/[our synthesis] = ours, unagreed).

Prompting question (Berker, on Claim 1(c) of the verification doc): the '21 guarantee is stated
as an unconstrained Bayes-risk infimum. If we instead allow MLPs as the function class, the
*backward* direction leaks — at finite compute many MLP solutions fit the risk even when the
representation is broken; an expressive probe severs the input–output link. Is the scheme
trusting labeling-agnosticism, or ignoring objectives? Paper 2 ('22) is exactly the paper about
this seam; this note incorporates it into the framework angle by angle.

---

## 1 · Three distinct leaks in "low probe risk ⇒ good representation"

The backward direction fails for three separable reasons; keeping them apart shows which
quantifier closes which.

**L1 · Bijection-blindness (population level, unconstrained or universal probes).** The
unconstrained infimum is invariant under any injective re-coding of the representation: a
bit-scrambled φ has identical Bayes risk on every task. '21 lives here *deliberately* — it is a
rate–distortion theory of what must be **retained**, not how it must be **arranged** ("our
theory discusses Bayes risk, which is independent of specific predictors and generalization"
[verified]). "Multiple solutions achieve the infimum" is not a leak *for that claim*: the claim
never speaks about which predictor is found, only that the information is there. The leak
appears the moment the number is read as representation *quality* — which is a D4 reading '21
never licenses.

**L2 · ERM non-identifiability (finite samples/compute, expressive probes).** Fix V = MLPs. The
population infimum over V still collapses to (near-)L1 by universal approximation, and now a
second, statistical failure stacks on top: the *empirical* argmin set F̂(D,φ) is large, its
members agree on the sample and diverge off it, and "the trained probe" is an unctrolled
selection from that set. A guarantee about inf_{f∈V} says nothing about sup_{f̂∈F̂}.

**L3 · Single-task coincidence.** For one fixed labeling, even a linear-probe success is weak
evidence about φ: the encoder may have placed that one decision boundary conveniently (or leaked
it) while being arbitrary for everything else. Probe risk on one task is a property of the pair
(φ, t), not of φ.

Devices on the market: **shrink V** (Xu '20 — I_V is not bijection-invariant, closing L1, and is
PAC-estimable for low-complexity V, closing L2 on the estimation side); **quantify over the task
family** (closes L3); **quantify over the ERM argmin set** ('22's move — closes L2 *without*
shrinking V). '22 deploys the last two simultaneously and pays for them with its scope
conditions (§4 below).

## 2 · '22's two load-bearing quantifiers (the direct answer to the question)

**Labeling-agnosticism — yes, and it is the maximal version.** Definition 1 ⟨fetched⟩: T is
*all* input–label distributions whose labeling is deterministic and constant on ∼-equivalence
classes — every partition of the orbit set, unrestricted class count. All content of the linear
characterization flows from this ∀: to serve *every* orbit labeling with a *fixed weak* probe
family, φ must place the |X/∼| orbit codes in affine general position with dim ≥ |X/∼| − 1
(Theorem 1 [verified]). Drop the ∀ (one task) and L3 reopens; weaken it and the geometry demand
weakens in step.

**Worst case over the ERM argmin set — the part the question predicted.** Definition 3 ⟨fetched⟩:

> "An encoder φ\* is sample optimal for T, F iff it is population optimal and minimizes the
> worst-case expected risk of ERMs for arbitrary sample sizes, i.e., for all n ≥ 1:
> φ\* ∈ argmin_{φ∈Φ_pop} W_n(φ,F,T)", with
> W_n(φ,F,T) := sup_{t∈T} E_{D_t∼p_tⁿ} [ sup_{f̂ ∈ F̂(D_t,φ)} R_t(φ, f̂) ].

The inner sup ranges over **the full set of empirical risk minimizers**. Their motivating prose
⟨fetched⟩: "When n is small a fitted probe (ERM) f̂ ∈ F̂(D_t,φ) could be a terrible population
predictor even when the underlying encoder is population optimal. Ideally, representations would
thus also guarantee that *any* ERM performs as well as possible for *any* desired task and
dataset size n." So "multiple MLP solutions fit their risk" is not an unconsidered leak in '22 —
it is *inside the optimality concept*, and the theorem characterizes the φ for which even the
worst interpolant is as good as possible.

**The mechanism that tames the argmin set is the invariance condition.** [our reading of §3.1]
Exact orbit collapse (x ∼ x⁺ ⟹ φ(x) = φ(x⁺)) makes every fitted probe — however wild inside V —
agree on *all* points of every orbit the sample touched: the probe's freedom off-sample is
exactly its freedom on unseen *orbits*, which is what the E_{D_t} over n prices. Invariance is
the sample-optimality ingredient, predictability + dimension the population one ('22 §3.1 prose:
population-optimal encoders are "essentially" the d ≥ |X/∼|−1 linearly-M-predictable ones
⟨fetched⟩; the clean split is our reading — the appendix-level statement has not been
re-verified). Note what this does to D2: invariance here is grounded by **worst-case
generalization of arbitrary probes**, with no information-theoretic input — a second, rate-free
grounding, independent of the Achille–Soatto route (invariance ⇔ minimality among sufficient).

**"Or are we not considering the objectives?" — both papers are representation-level;
objectives enter only ex post.** '21 is objective-free by construction. '22 reverse-engineers
objectives (CISSL/DISSL) whose *global minimizers* satisfy the characterization for the declared
probe family — the probe class is pushed into the loss so the loss's optimum set is pruned to
V-good encoders. What stays open even there: the **encoder-side** argmin set (which minimizer
SGD selects, and whether near-minimizers inherit anything). '22 closes probe-side multiplicity
by quantifier and leaves encoder-side multiplicity to experiments — exactly the slot where the
framework's seed null (§6.3) and the outside-the-loss audit live.

## 3 · What survives at each probe capacity — the dial

[our reading; composed from Theorem 1 + '22 §5's stated differences ("encoders should ensure
predictability of M(X) for the desired F⁺"; "the dimensionality requirement decreases with
complexity of F⁺, e.g., for universal F⁺ it is d ≥ 1" [verified]); the formal F⁺ theorem is in
their appendix and has not been re-verified.]

In the finite-X world, sample optimality for probe family V pins φ to: **φ = ι ∘ M** — collapse
every orbit (invariance), then encode orbits by an injection ι whose image V can shatter.

- **Universal V (MLPs):** any injective ι works; d ≥ 1. The certified content of "MLP-probe
  optimal" is exactly *an injective re-coding of the maximal invariant* — no geometry
  whatsoever. Berker's "MLP removes the input–output link" is thus the theorem's own content
  read backwards: with expressive probes there is nothing about arrangement left to certify, and
  the theory is honest about it (the residue is invariance + orbit-injectivity, which is not
  nothing: it is what W_n's inner sup needs).
- **Linear V:** ι must place |X/∼| codes in affine general position; d ≥ |X/∼| − 1. Geometry
  maximal — essentially a simplex code of the orbit space.

**Guarantee strength ≈ task-family breadth ÷ probe capacity.** '22 pins both knobs at an extreme
(all labelings; declared V). TRD-π declares both knobs and *measures the residual*: battery F
in place of ∀-labelings (R7a), tiers V₁ ⊂ V₂ carried jointly, and the cross-tier gap
Δ = I_{V₂} − I_{V₁} as the reported object — the object '22's one-family-per-guarantee frame
never defines (verified, Claim 2c). Same algebra, different governance: they buy theorems with
maximal quantifiers; we buy measurability with declared ones.

This also grounds the audit's tier discipline from the probe side: an MLP-probe accuracy,
*alone*, certifies at most orbit-injectivity (L1), so tier-2 numbers appear only inside Δ
(budget-matched, Rule 3) and never as standalone quality claims; tier-1 (linear) is the
certifying measurement. R7c's "converse fails" now has a theorem-shaped reason, not just a
V-information slogan.

## 4 · Object-by-object exchange rate

| Dubois '22 | TRD-π | divergence that matters |
|---|---|---|
| equivalence ∼ on **finite** X; exact orbits | declared nuisance mechanism G — stochastic, lossy, non-invertible (content/style U) | '22 is the clean-room limit of the standing qualifier; real augmentations do not induce an equivalence relation, so exact invariance is not even well-posed for us — surrogate + audit replace it |
| maximal invariant M | invariant content U (M only in the group/equivalence case) | same object where both exist (R7a) |
| T = **all** deterministic ∼-invariant labelings | declared finite battery F; tasks outside span(F) invisible by declaration | the quantifier knob. ∀-labelings ⇒ dim ≥ \|X/∼\|−1, unaffordable at any real width (superposition regime, R3 takeaway 2 / §6.1); declaring F drops the floor to O(\|F\|) [PS, our synthesis] — the declaration is what makes tier-1 *affordable* |
| probe family F or F⁺, **one per guarantee** | tiers V₁ ⊂ V₂, carried simultaneously | no cross-tier object in '22; Δ(t, space) is ours (verified, Claim 2c) |
| population optimality: inf_{f∈F} R_t(φ,f) = 0 ∀t | H_{V}(t\|φ(X)) = 0 ∀t ∈ F — I_V saturation [our gloss; their risk is 0-1, Xu's is log-loss] | '22 ≈ "which φ saturate I_V for all invariant t at once", plus the W_n refinement Xu's frame lacks |
| sample optimality: sup over the **ERM argmin set**, all n | Rule 3 budget-matched probes; E11 probe-sensitivity axis | W_n is the theorem-grade rationale for budget-matching: unmatched probe capacity measures the argmin set's spread, not φ |
| invariance condition (necessary for sample optimality) | D2 | second grounding of D2: worst-case ERM control, rate-free — independent of Achille–Soatto |
| dimension condition — a **lower** bound; heads/dim "as large as possible" | rate term against declared π pushes width *down*; regime check arbitrates | '22 has **no rate/minimality term at all** — see §5(b) |
| collapse impossible at optimum (predictability of M excludes it) | §4 equilibrium: the marginal term is the only anti-collapse force in L | convergent support for "anti-collapse is never a primitive": both normative frames *derive* it — '22 from the sufficiency side, TRD-π from the calibration side |
| CISSL/DISSL; asymmetric projection heads derived ("SimCLR … not even population optimal for linear F" [verified]) | placement rule (R6): alignment at z, probing/rate/audit at h | same two-space instinct, opposite instrument: they fix it in the architecture ex ante, we measure it ex post. Direct prior for E1/E10 (= verification doc fix #3). Caveat: Dubois–Hashimoto–Liang '23 could not confirm the asymmetric-head gains over 169 models [verified] — the prescriptive arm is empirically shaky, which is an argument *for* the audit arm |

## 5 · Where the frames part ways (beyond the table)

**(a) Checkability.** In '22's finite world the conditions are decidable by enumeration —
invariance and general position are inspectable properties, so no audit is needed *in-world*.
Under a lossy stochastic G, none of the three conditions is even stateable exactly; the audit
(worst-direction, outside the loss, evidence-not-certificates) is what replaces enumeration.
Their theorems need no epistemics; our protocol is mostly epistemics. That asymmetry is the
cost of leaving the clean room, and it is the project's reason to exist.

**(b) The two Dubois papers are the two halves of the program, never composed.** '21 = D1+D2 at
a single unconstrained readout tier (rate story, accessibility-blind — Claim 1c). '22 = D1+D4
at a declared tier (accessibility story, **rate-free** — dimension is only bounded below, heads
"as large as possible"). Neither paper holds compression and accessibility at once; under
'22's ∀-labelings quantifier they *cannot* coexist (the dimension floor forbids compression).
The composition becomes affordable exactly when the task quantifier is cut to a declared finite
F — after which a rate term needs a codebook to be measured against, which is where the declared
π stops being optional and becomes load-bearing. [our synthesis — the cleanest one-paragraph
justification of TRD-π's shape relative to this line; to attack in discussion.]

**(c) Ex-ante vs ex-post.** '22 designs objectives whose minimizers are idealized for the chosen
V. We audit where *existing* methods' minimizers landed, per space. Complementary, not
competing: their scope conditions (which V, which task family, exact invariance, orbit
granularity) are precisely the knobs our audit turns into measured quantities (probe tiers,
declared battery, invariance level at h vs z, Δ).

**(d) No distributional desideratum.** Nothing in '22 constrains the *marginal* of φ(X) beyond
general position — no π, no isotropy, no calibration, no D0/D3′/D5 counterpart. The entire
marginal/prior axis of TRD-π (and the LeJEPA fork) is orthogonal to this line.

## 6 · Import candidates and seams (for discussion — nothing filed)

1. **Anchor import:** sample-optimality/W_n as the formal home of probe non-identifiability —
   attach to R7c (tier discipline) and protocol Rule 3 (budget-matching) in the next framework
   version bump; '22 joins Xu as D4's second anchor (characterization side, vs measurement side).
2. **D2 second grounding** (invariance = worst-case-ERM control) — one sentence at D2, next bump.
3. **E11 reading (proposal only, card untouched):** the spread of probe outcomes across
   seeds/architectures at fixed φ is a finite-V shadow of sup_{f̂∈F̂}; E11 could report
   worst-of-probes alongside mean-of-probes to estimate the population↔sample gap empirically.
4. **Open seam:** an approximate/agnostic '22 — ε-invariance + δ-predictability under a
   stochastic lossy G, with graceful W_n degradation — does any version exist or is it open?
   (Relates to the R7 scope declaration and OP-e; candidate OP row if we decide to care.)
5. **Practice caveat, one line:** SGD-trained MLP probes have implicit bias that narrows F̂ in
   practice — real but unlicensed by any of the above; it is why budget-matched, seeded probe
   protocols (Rule 3, E11) are the only way tier-2 numbers mean anything.

---

## 7 · Strategy read after the deep-dive (Claude's recommendation, 2026-07-08 — for discussion, NOT agreed)

Berker's question: are we still novel enough, theoretically and empirically — or is this a
solved problem we shouldn't start?

**Recommendation: proceed.** The problem '22 solved is not the problem this project starts.

- **What is solved:** the clean-room corner — finite X, exact orbits, deterministic labelings,
  0-1 loss: exact characterization + objectives whose global minimizers satisfy it. The bold
  name "idealized representations" names that corner; the paper's own dimension bound
  (d ≥ |X/∼|−1 for the full task family) makes the ideal unreachable at any real width, and no
  approximation theory under a lossy stochastic G exists.
- **Decisive datum that it is not solved:** the same group's ICML '23 paper (169 models) could
  not confirm '22's asymmetric-head prescription and wrote "we still do not completely
  understand the impact of non-linear projections" [verified, anchors doc Claim 2b]. The
  characterization exists; its prescriptions don't visibly bind practice; the field lacks the
  instrument that says why. The instrument is this project.
- **Untouched by '22:** the rate/minimality axis; the marginal/prior/audit axis (D0/D3′/D5;
  R2/R4/R5/R8; the LeJEPA fork; M0's honesty cell); the cross-tier gap object Δ; the ex-post
  cross-method ordering problem (E3); the two-space audit and transfer ratios τ (OP-1: all
  existing transfer evidence qualitative, per the report's survey). Of six desiderata it
  touches D1, D4, and D2's mechanism.
- **Concessions:** (done) "first two-space theorem" dropped in the verification pass; (to do)
  never claim first tier-aware design — their §6 claims the probe↔head-architecture relation;
  D4's positioning is "two tiers carried *jointly* + the measured gap object Δ", and
  DESIDERATA §5 item 3 ("appears nowhere as a design principle") needs that re-scope.
- **Gains:** §4.1 as an external directional prior for E1/E10; W_n as the theorem-grade
  rationale for Rule 3 + the E11 worst-of-probes reading; and **DISSL/CISSL as a
  positive-control roster candidate** — an encoder *designed* to be idealized for linear probes
  is a calibration standard for the audit (prediction: small Δ at h where SimCLR shows inflated
  Δ), preempting the "your instrument might measure noise" review.
- **Draft positioning paragraph** (for the eventual paper, to attack): Dubois '22 tells you
  what the endpoint should look like and how to train toward it in a clean room; Dubois '23
  shows the clean-room prescriptions don't clearly survive contact; nothing in the surveyed
  literature measures the desiderata methods claim, in both spaces, under a matched frame, with
  honesty machinery for the statistics being gamed. Theory-side, the composition — rate against
  a declared prior + jointly-carried tiers + audited marginal — remains open; '21 and '22 are
  its two halves, never composed, and under '22's quantifier they provably cannot be.
- **Action queue (each needs Berker's sign-off):** (1) re-scope DESIDERATA §5 item 3; (2) '22
  §4.1 prior into E10 — pre-registration discipline: arms A–D0 have already run, so it can only
  enter as a REGISTERED-LATE annotation or attach to not-yet-run rescue arms; (3) DISSL onto
  the M5+ roster candidate list (donor port via the PORT_NOTES process); (4) proper read of
  2302.03068 — the nearest empirical neighbor, so far mined for one quote only; (5) optional
  belt-and-braces: citation sweep of 2209.06235's descendants before locking a novelty
  paragraph (risk rated low — the report's survey is a week old).
