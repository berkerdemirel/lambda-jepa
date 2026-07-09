# Quote verification: Achille–Soatto '18 · Tschannen '20 · BYOL '20

**AGENT-GENERATED, UNREVIEWED (2026-07-08).** Produced by a Claude verification agent for the
quote-verification pass agreed in `DESIDERATA_FRAMEWORK.md` §6.1. Not a joint takeaway; verdicts
below are the agent's and await Berker's review before the §3 table is stamped citation-grade.

**Method.** Full HTML of each paper downloaded from ar5iv (`ar5iv.labs.arxiv.org/html/<id>`,
LaTeXML render of the **latest arXiv version**) and quotes extracted by direct text search of the
downloaded file — not paraphrased through a summarizing fetcher. Math notation in quotes is
transcribed to ASCII from the MathML (`I(z;n)` etc.); English wording is verbatim. Versions
served/checked: 1706.01350v3 (2018-06-28; = JMLR 19(50) version), 1907.13625v2 (2020-01-23;
ICLR 2020), 2006.07733v3 (2020-09-10; NeurIPS 2020).

**Claims under test** (verbatim from our own docs):

- `DESIDERATA_FRAMEWORK.md` §1/D2: "(Achille–Soatto: among sufficient representations, minimal ⇔ invariant to nuisance.)"
- §3 row 1: "IB / minimal sufficiency (Achille–Soatto '18; Federici …) | D1+D2 exactly … | information-only: blind to geometry, hierarchy, usability — the zip objection applies to naive readings"
- §0 / §3 row 2 (Tschannen): "tighter MI bounds can yield worse representations"; "MI is bijection-invariant, so information content cannot explain representation quality; estimator/architecture biases do the work | diagnosis, no constructive replacement"
- §0: "collapse: BYOL-minus-predictor satisfies its loss at 0.3% accuracy"
- `trd_pi_theory_framework_v1_3.md` §0: "losses are satisfiable by useless representations — BYOL-family ablations collapse to near-chance while satisfying their objective"

---

## Paper 1 — Achille & Soatto, *Emergence of Invariance and Disentanglement in Deep Representations* (arXiv 1706.01350v3, JMLR 19(50), 2018)

### 1(a) "among sufficient representations, minimal ⇔ invariant to nuisance" — **VERIFIED**

The equivalence is stated three times, at increasing precision.

Abstract:

> "we show that invariance to nuisance factors in a deep neural network is equivalent to
> information minimality of the learned representation"

Section 1 (contributions, item (a)):

> "(a) a sufficient representation of the data is invariant if and only if it is minimal, i.e., it
> contains the smallest amount of information, although may not have small dimension"

The formal result is **Proposition 3.1 (Invariance and minimality)**, Section 3, restated with
proof as Proposition C.2, Appendix C.2:

> "**Proposition 3.1 (Invariance and minimality, Appendix C.2)** Let n be a nuisance for the task y
> and let z be a sufficient representation of the input x. Suppose that z depends on n only through
> x (i.e., n → x → z). Then,
>
> I(z;n) ≤ I(z;x) − I(x;y).
>
> Moreover, there is a nuisance n such that equality holds up to a (generally small) residual ε
>
> I(z;n) = I(z;x) − I(x;y) − ε,
>
> where ε := I(z;y|n) − I(x;y). In particular 0 ≤ ε ≤ H(y|x), and ε = 0 whenever y is a
> deterministic function of x. Under these conditions, a sufficient statistic z is invariant
> (maximally insensitive) to nuisances if and only if it is minimal."

Precise assumptions and definitions carried by the proposition (Sections 2.1–2.2):

1. **Representation**: "z is a representation of x if z is a stochastic function of x … In
   particular we have the Markov chain y → x → z."
2. **Sufficiency** (supervised, single task): "a representation z of x is sufficient for y if
   y ⊥ x | z, or equivalently if I(z;y) = I(x;y)"; **minimality**: "it is minimal when I(x;z) is
   smallest among sufficient representations." (So "among sufficient representations" is built into
   the definition — matching our phrasing.)
3. **Nuisance**: "A nuisance is any random variable that affects the observed data x, but is not
   informative to the task we are trying to solve. More formally, a random variable n is a nuisance
   for the task y if y ⊥ n, or equivalently I(y;n) = 0." Explicitly more general than group
   nuisances: "not restricted to deterministic functions, nor to group nuisances."
4. **Invariance vs maximal insensitivity**: "z is invariant to the nuisance n if z ⊥ n, or
   I(z;n) = 0. When z is not strictly invariant but it minimizes I(z;n) among all sufficient
   representations, we say that the representation z is maximally insensitive to n."
5. **Markov condition**: z depends on n only through x (n → x → z).
6. The equality-attaining ("worst-case") nuisance is constructed via **Proposition 2.1 /
   Lemma C.1 (Task-nuisance decomposition)**: "Given a joint distribution p(x,y), where y is a
   **discrete** random variable, we can always find a random variable n independent of y such that
   x = f(y,n), for some deterministic function f." So the attainment side assumes y discrete, and
   holds up to the residual ε (ε = 0 when y is a deterministic function of x; Remark 3.2: "usually
   H(y|x) = 0 or at least H(y|x) ≪ I(x;z), we can generally ignore the extra term").

Fine print to keep when citing: the "⇔" is exactly the paper's sentence, but "invariant" there is
glossed "(maximally insensitive)" — i.e., minimality is equivalent to minimizing the (worst-case)
nuisance information I(z;n) among sufficient representations, with I(z;n) = 0 not attainable in
general; direction "minimal ⇒ insensitive" holds for **all** nuisances via the upper bound, and the
converse uses the constructed worst-case nuisance (y discrete, residual ε). Our D2 sentence is a
faithful compression of this.

### 1(b) "covers D1+D2 … information-only: blind to geometry/hierarchy/usability" — **VERIFIED-WITH-AMENDMENT**

**Information-only: verified.** Every desideratum in the paper is a mutual-information (or
Total-Correlation) functional; sufficiency is I(z;y) = I(x;y) with no function-class or readout
constraint (a text search finds no notion of linear decodability, readout capacity, or probe
anywhere). The paper *itself* states the bijection degeneracy our "zip objection" points at
(Section 3):

> "Such a representation, if it exists, would not be unique, since any bijective mapping preserves
> all these properties. We can use this to our advantage and further aim to make the representation
> (d) maximally disentangled, i.e., choose the one(s) for which TC(z) is minimal."

**Amendments (three):**

1. **Not fully "blind" — it names the degeneracy and answers with disentanglement.** The chosen
   tie-breaker, minimal Total Correlation, is still an information functional (coordinate-wise
   independence; coordinate-sensitive but carrying no metric/linear structure), so the substance of
   "blind to geometry and usability" stands; but a citation-grade sentence should say the paper
   *acknowledges* bijection-invariance of (a)–(c) and responds with TC, rather than that it is
   blind to the issue.
2. **"Geometry" does occur — but of the loss landscape, not the representation.** Abstract: "sheds
   light on the relation between the geometry of the loss function, invariance properties of the
   learned representation, and generalization error" (flat minima, Prop. 4.3). No
   representation-space geometry. Similarly a layer result exists — Proposition 3.5 ("Stacking
   increases invariance") — but it is invariance accumulating across layers, not D3-style
   rate-ordered hierarchy *within* a code. So "no hierarchy/geometry *in our D3/D5 sense*" is
   accurate; unqualified "blind to geometry, hierarchy" invites a referee correction.
3. **D1 is covered in single-task form only.** Sufficiency here is for one supervised task y, not a
   task family; the label-free/task-family upgrade is Federici et al. '20 (cited jointly in our §3
   row, so the row is fine as a bundle — but A&S alone should not be cited for D1-as-family).
   (Federici not re-verified here; out of this pass's scope.)

---

## Paper 2 — Tschannen, Djolonga, Rubenstein, Gelly, Lucic, *On Mutual Information Maximization for Representation Learning* (arXiv 1907.13625v2, ICLR 2020)

### 2(a) "tighter MI bounds can yield worse representations" — **VERIFIED**

Section 1 (contribution statement):

> "In fact, we show that maximizing tighter bounds on MI can result in worse representations."

The experiment: **Section 3.2, "Higher capacity critics can lead to worse downstream performance"**
(Figure 3; MNIST-style two-halves setup, I_NCE / I_NWJ estimators, bilinear vs separable vs MLP
critics):

> "While the testing I_NCE value is close to the theoretically achievable maximum value for all
> critics, the testing I_NWJ value is higher for the MLP critic than for the separable and bilinear
> critics, resulting in a tighter bound on the MI. However, despite achieving the smallest I_NWJ
> testing value, the simple bilinear critic leads to a better downstream performance than the
> higher-capacity separable and MLP critics."

Figure 3 caption: "Bilinear and separable critics lead to higher downstream accuracy than MLP
critics, while reaching lower I_NWJ." Section 1 bullet: "For I_NCE and I_NWJ, higher-capacity
critics admit tighter bounds on MI. We demonstrate that simple critics yielding loose bounds can
lead to better representations than high-capacity critics." Conclusion (§5): "—perhaps
surprisingly—looser bounds can lead to better representations."

Scope note for citation: demonstrated for critic-capacity-induced tightness (test-set I_NWJ) in
deliberately small-scale experiments; the paper's phrasing is "can", not "always". A companion
result strengthens our "wrong functional" use of it — **Section 3.3** trains two encoder
architectures to the *same* bound value (loss L_t = |I_EST − t|): "Despite matching lower bounds,
ConvNet encoders lead to clearly superior classification accuracy" — i.e., the achieved value of
the optimized functional does not determine representation quality.

### 2(b) MI bijection-invariant ⇒ information content alone can't explain quality — **VERIFIED**

Section 1:

> "Firstly, MI is invariant under reparametrization of the variables — namely, if X′ = f₁(X) and
> Y′ = f₂(Y) are homeomorphisms (i.e. smooth invertible maps), then I(X;Y) = I(X′;Y′)."

Abstract: "using it as an objective for representation learning may lead to highly entangled
representations due to its invariance under arbitrary invertible transformations." The argument is
made empirical in **Section 3.1 ("Large MI is not predictive of downstream performance")** with
bijective (RealNVP) encoders, for which every parameter setting maximizes true MI:

> "there exist invertible encoders for which the representation quality is worse than using raw
> pixels, despite also maximizing MI." (Section 1 bullet; Section 3.1: "Next we demonstrate that
> for the same invertible encoder architecture there are model parameters for which linear
> classification performance is significantly worse than using raw pixels, despite also being
> globally optimal MI maximizers.")

Figure 1 caption: "This demonstrates the existence of encoders that provably maximize MI yet have
bad downstream performance." Conclusion (§5): "While MI has appealing theoretical properties, it is
clearly not sufficient for this task—it is hard to estimate, invariant to bijections and can result
in suboptimal representations which do not correlate with downstream performance."

### 2(c) "estimator/architecture inductive biases do the work" — **VERIFIED**

Abstract:

> "In this paper we argue, and provide empirical evidence, that the success of these methods cannot
> be attributed to the properties of MI alone, and that they strongly depend on the inductive bias
> in both the choice of feature extractor architectures and the parametrization of the employed MI
> estimators."

Conclusion (§5): "We have revealed that the commonly used estimators have strong inductive biases
and—perhaps surprisingly—looser bounds can lead to better representations." Section 3 intro: "the
performance of these methods depends strongly on the bias that is encoded not only in the encoders,
but also on the actual form of the used estimators."

Nuance for our §3 "stops short: diagnosis, no constructive replacement" column: accurate in that no
new objective is proposed, but the paper does offer a reinterpretation — the deep-metric-learning /
triplet view (§4) — and explicitly calls for what our D5/geometry program wants (§5, "Alternative
measures of information"): "a new notion of information should account for both the amount of
information stored in a representation and the geometry of the induced space necessary for good
performance on downstream tasks." Worth citing in our favor rather than only as a gap.

---

## Paper 3 — Grill et al., *Bootstrap Your Own Latent* (arXiv 2006.07733v3, NeurIPS 2020)

### Claim: "BYOL-minus-predictor satisfies its objective while collapsing to ~0.3% accuracy" — **VERIFIED-WITH-AMENDMENT (two corrections needed)**

Ablation protocol (Section 5): ResNet-50, 300 epochs, ImageNet linear evaluation top-1, 3 seeds,
lr 0.3, batch 4096, wd 1e-6, τ_base = 0.99. Chance on 1000 classes = 0.1%.

**Correction 1 — the 0.3% cell is minus-TARGET-NETWORK (predictor kept), not minus-predictor.
Minus-predictor is 0.2%.** Two distinct ablations produce 0.3%:

**Table 5(a)** ("Results for different target modes", Section 5 "Bootstrapping"): row
"Stop gradient of online†, τ_base = 0 → **0.3**" (footnote: "†In the stop gradient of online,
τ = τ_base = 0 is kept constant throughout training"). Predictor present; the EMA is removed
(target instantaneously follows the online network). Text: "Instantaneously updating the target
network (τ = 0) destabilizes training, yielding very poor performance". Other rows: constant random
network (τ=1) 18.8±0.7; EMA 0.999/0.99/0.9 → 69.8/72.5/68.4.

**Table 5(b)** ("Intermediate variants between BYOL and SimCLR"; columns Method | Predictor |
Target network | β, where β=1 adds SimCLR's negative-pair term and β=0 removes it), exact rows:

| Method | Predictor | Target network | β | Top-1 |
|---|---|---|---|---|
| BYOL | ✓ | ✓ | 0 | **72.5** |
| − | ✓ | ✓ | 1 | 70.9 |
| − | ✗ | ✓ | 1 | 70.7 |
| SimCLR | ✗ | ✗ | 1 | 69.4 |
| − | ✓ | ✗ | 1 | 69.1 |
| − | ✓ | ✗ | 0 | **0.3** |
| − | ✗ | ✓ | 0 | **0.2** |
| − | ✗ | ✗ | 0 | 0.1 |

So: predictor-without-target-network at β=0 → **0.3%**; **BYOL minus predictor (target network
kept, β=0) → 0.2%**, and the paper's own words attach "removing the predictor" to the 0.2% row
(Section 5, "Relationship with Mean Teacher"):

> "Removing the predictor in BYOL results in an unsupervised version of MT with no classification
> loss … This variant of BYOL collapses (Row 7 of Table 5) which suggests that the additional
> predictor is critical to prevent collapse in an unsupervised scenario."

and (Section 5, "Importance of a near-optimal predictor"): "Table 5(b) already shows the importance
of combining a predictor and a target network: the representation does collapse when either is
removed."

Appendix Table 19 (Section F.4) further splits "no target network" into target parameters θ
(gradient flows) vs sg(θ) (stop-gradient), all with β=0: (✓ pred, θ) → 0.3; (✓ pred, sg(θ)) →
**5.5**; (✗ pred, ξ EMA) → 0.2; (✗ pred, sg(θ)) → 0.1; (✗ pred, θ) → 0.1. Note the paper as
printed thus has predictor+stop-gradient at 5.5% (Table 19) next to Table 5(a)'s
"stop gradient of online, τ=0" at 0.3% — nearly the same configuration with very different
outcomes (the τ=0 target is a one-step-stale copy; Table 19's sg(θ) is the current weights).
Citation-safe course: quote a specific table cell, not "the" stop-gradient number.

**Correction 2 — "satisfies its objective" is paper-implied, not paper-measured.** The paper
nowhere reports the loss value achieved in any collapsed ablation cell, and for τ=0 it attributes
failure to destabilized training. What it does state is that collapsed solutions satisfy the
objective *in principle*:

> Abstract/§1: "While this objective admits collapsed solutions, e.g., outputting the same vector
> for all images, we empirically show that BYOL does not converge to such solutions."
>
> §3.2: "As BYOL does not use an explicit term to prevent collapse (such as negative examples [10])
> while minimizing L^BYOL_{θ,ξ} with respect to θ, it may seem that BYOL should converge to a
> minimum of this loss with respect to (θ,ξ) (e.g., a collapsed constant representation)."

(§3.2 also hypothesizes the collapsed constant equilibria are unstable *for BYOL with predictor +
EMA* — "hence our hypothesis on these collapsed constant equilibria being unstable".)

**Verdict detail.** TRD-π §0's plural form — "BYOL-family ablations collapse to near-chance while
satisfying their objective" — survives with one soft amendment: 0.1–0.3% vs 0.1% chance is
near-chance ✓; "while satisfying their objective" should be sourced as "the objective admits
collapsed solutions (Grill et al. §3.2) and these ablations collapse (Table 5)", since no loss
value at collapse is reported. DESIDERATA_FRAMEWORK §0's specific sentence needs its attribution
fixed: either "BYOL-minus-**target-network** collapses to 0.3% (minus-predictor: 0.2%)" or
"BYOL-family ablations collapse to 0.1–0.3%".

---

## Verdict summary

| # | Claim (our docs) | Verdict |
|---|---|---|
| 1a | A&S: among sufficient representations, minimal ⇔ invariant to nuisance | **VERIFIED** (Prop 3.1/C.2; "invariant (maximally insensitive)"; assumptions: I(y;n)=0, sufficiency I(z;y)=I(x;y), Markov n→x→z; attainment via worst-case nuisance, y discrete, residual ε ≤ H(y|x)) |
| 1b | A&S covers D1+D2; information-only, blind to geometry/hierarchy/usability | **VERIFIED-WITH-AMENDMENT** (paper itself flags bijection-degeneracy and answers with TC-disentanglement; "geometry" present only as loss-landscape; D1 single-task in A&S alone) |
| 2a | Tschannen: tighter MI bounds can yield worse representations | **VERIFIED** (§1 verbatim; §3.2/Fig 3; plus §3.3 matched-bound experiment) |
| 2b | MI bijection-invariant ⇒ information content can't explain quality | **VERIFIED** (§1 reparametrization invariance; §3.1 invertible encoders worse than raw pixels while provably maximizing MI) |
| 2c | Estimator/architecture inductive biases do the work | **VERIFIED** (abstract + §5 conclusion verbatim; "no constructive replacement" fair, but note their §4 metric-learning reading and §5 call for geometry-aware information) |
| 3 | BYOL-minus-predictor satisfies its loss at 0.3% | **VERIFIED-WITH-AMENDMENT** (0.3% = minus-target-network w/ predictor, Table 5(b) row 6, and τ=0 stop-grad, Table 5(a); minus-predictor = 0.2%, Table 5(b) row 7; "satisfies its loss" is implied by §3.2's "admits collapsed solutions" + observed collapse, never measured/reported at the collapsed cells) |

Suggested minimal edits to our docs (for Berker's approval, not applied):
- `DESIDERATA_FRAMEWORK.md` §0: "BYOL-minus-predictor satisfies its loss at 0.3% accuracy" →
  "BYOL's objective admits collapsed solutions, and its minus-target-network / τ=0 ablations
  collapse to 0.3% top-1 (minus-predictor: 0.2%) [Grill et al. '20, §3.2, Table 5]".
- `trd_pi_theory_framework_v1_3.md` §0: keep "collapse to near-chance", change "while satisfying
  their objective" → "whose objective such collapsed solutions satisfy" (or cite §3.2 explicitly).
- §3 row 1 "blind to geometry, hierarchy, usability" → add "(A&S acknowledge the bijection
  degeneracy and answer with TC-disentanglement — still an information functional)".
