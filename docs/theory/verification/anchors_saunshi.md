# Quote verification: Saunshi et al. '22 — *Understanding Contrastive Learning Requires Incorporating Inductive Biases*

**AGENT-GENERATED, UNREVIEWED — 2026-07-08.** Adversarial quote-verification pass for the Saunshi
rows in `docs/theory/THEORY_MAP.md` (row "Vacuity of loss-only analyses", and failure-mode item 2),
`docs/theory/DESIDERATA_FRAMEWORK.md` §3 ("Also: Saunshi et al. '22 …"), and
`docs/literature/BIBLIOGRAPHY.md`. Nothing here is an AGREED TAKEAWAY.

**Sources (full text, fetched 2026-07-08).** arXiv **2202.14037v1** (only version on arXiv;
submitted 2022-02-28): PDF → `pdftotext` full text, **plus the arXiv e-print LaTeX source tarball**
(dated 2022-03-01) for cross-checking and source archaeology. Also the ICML 2022 camera-ready
(PMLR v162, `saunshi22a.pdf`) fetched and grepped for the head question and numbering.
Theorem numbering is identical in arXiv v1 and the PMLR camera-ready (checked: Lemma 4.1/4.2,
Corollary 4.1/4.2, Example 1). ar5iv is broken for this ID (serves a stub titled "Active Learning
for Cost-Sensitive Classification" — wrong paper); arxiv.org/html 404s for all version suffixes.
Prose quotes below are verbatim from the pdftotext extraction; math is transcribed to ASCII
(marked ⟨math transcribed⟩ where nontrivial).

Paper: Saunshi, Ash, Goel, Misra, Zhang, Arora, Kakade, Krishnamurthy, ICML 2022.

---

## 0 · The claims under audit (as written in our docs)

1. `DESIDERATA_FRAMEWORK.md` §3: "Saunshi et al. '22 (**loss-level analysis is vacuous without
   function-class assumptions**) is the theorem-form of 'no way to guess the method ordering ex
   ante'." (Similarly `BIBLIOGRAPHY.md`: "loss-value analyses can be vacuous".)
2. `THEORY_MAP.md` row: "The enabling result: **same loss, same augmentations, different function
   class ⇒ different downstream. The head is precisely such a function-class device.** [P]" — the
   second sentence is our inference; audit question is whether the paper itself says anything about
   projection heads.
3. Not a doc claim, a requested recording: does the vacuity apply to **exact minimizers,
   near-minimizers, or both**?

---

## 1 · Claim 1 — "Loss-level analysis is vacuous without function-class assumptions" — **VERIFIED-WITH-AMENDMENT**

The result exists and is theorem-form, but the *provable* vacuity is scoped to the
**disjoint-augmentations regime** (extended to near-disjoint via an O(τ²) suboptimality condition);
outside that regime the claim is empirical. The paper's own abstract carries the scoping:

> "We demonstrate that such analyses, that ignore *inductive biases* of the function class and
> training algorithm, cannot adequately explain the success of contrastive learning, even
> *provably* leading to vacuous guarantees **in some settings**."
> — Abstract (emphasis theirs except the bold).

### 1a · What "loss-level analysis" formally means in the paper (§2, Eq. 5 vs Eq. 6)

The paper abstracts all prior analyses (Arora et al. '19, Tosh et al. '21, HaoChen et al. '21) as a
transfer function that sees only problem quantities, the loss value, and the dimension —
⟨math transcribed⟩:

> "L_clf(f; ȳ*) ≤ T(Γ, L_cont(f), d), where Γ = (D_X̄, A, ȳ*, L_cont)   (5)"
> "These guarantees only depend on (1) problem dependent quantities like input marginals D_X̄,
> properties of augmentations A, downstream label ȳ*, form of contrastive loss L_cont, (2)
> contrastive loss L_cont(f) of the representation f and (3) its dimensionality d."
> "These bounds place a premium on the value of contrastive loss of f , but are agnostic to any
> other properties of f , like the representation function class F it belongs to or how it was
> trained."

and the desired repair is Eq. (6): "L_clf(f; ȳ*) ≤ T(Γ, L_cont(f), **F**)". So "loss-level
analysis without function-class assumptions" = Eq. (5)-form bounds; exactly our shorthand's target.

### 1b · The vacuity theorem (Section 4.1: Lemma 4.1, Lemma 4.2, Corollary 4.1; joint proof = Theorem C.1)

> "**Lemma 4.1.** Let |X̄| = N and d = O(N/log₂(N)). Suppose the labeling function is balanced,
> i.e. Σᵢ yᵢ* = 0, and let D_X be uniform over X̄. If the augmentation distribution is disjoint,
> then for any f*: X → R^d there exists a f̂: X → R^d such that:
> L_cont(f̂) ≤ L_cont(f*), & L_clf(f̂) ≥ 1/2 − O(√(d·log(N)/N))." ⟨math transcribed⟩

Lemma 4.2 extends this to constrained (e.g. normalized) representations under a shared-randomness
augmentation protocol: "In the setup of Lemma 4.1, suppose further that representations are
constrained (to any given set) … Then the conclusion of Lemma 4.1 holds." Both are proven jointly
as **Theorem C.1** (Appendix C).

> "**Corollary 4.1.** In the setup of Lemma 4.1 or Lemma 4.2, consider a transfer function T
> bounding the downstream performance as L_clf(f; ȳ*) ≤ T(Γ, L_cont(f), d) as in Equation (5) …
> Suppose T is monotonic in its second argument, then **for all** f: X → R^d:
> T(Γ, L_cont(f), d) ≥ 1/2 − Õ(√(d/|X̄|))" ⟨math transcribed⟩

Framing sentences (§4.1):

> "any representation f can be transformed — **by shuffling identities of examples** — to a new
> representation f̃ that has lower (or equal) contrastive loss but near-trivial downstream
> performance. An immediate consequence is that any function class agnostic analysis (including
> all previous analyses) will necessarily leads to vacuous downstream guarantees." [sic: "leads"]

> "**Takeaways.** The above lower bounds suggest that previous analyses for contrastive learning
> are vacuous in the disjoint augmentation setting, due to existence of bad minimizers of the
> contrastive loss."

### 1c · The construction (Appendix C proof of Theorem C.1, three steps, verbatim)

> "1. Show that every instance x̄ ∈ X̄ has an embedding v_x̄, such that if we embed x̄ and all of
> its augmentations to v_x̄ then we obtain a new embedding function f̂ for which L_cont(f̂) is no
> worse than L_cont(f*).
> 2. Let V := {v_x̄ : x̄ ∈ X̄}. Show that for any bijection π: V → V, we have
> L_cont(π ∘ f̂) = L_cont(f̂). In other words, if we apply a permutation to the embeddings of f̂ we
> do not change the contrastive loss.
> 3. Show that there exists some permutation π such that π ∘ f̂ has very high downstream error rate."

Step 1 is Jensen/convexity (mean-embedding each instance's augmentations weakly lowers the loss —
disjointness makes this feasible without collision); step 2 is exact loss invariance under
permutation of instance identities; step 3 is a combinatorial/probabilistic argument that some
permutation scrambles labels to error ≥ 1/2 − O(√(d·log N / N)). Works for both L_SimCLR and
L_spec ("results hold for both L_SimCLR and L_spec and we abbreviate these by L_cont").

### 1d · Concrete instantiation (Example 1 hypercube; Table 1; Corollary 4.2)

Example 1 (§3): inputs X̄ = {±1}^D, label linear in first k ≪ D coordinates, augmentations scale
the last coordinates by τ ~ U((0,1]) — disjoint because "the original input can be recovered from
an augmentation by simply performing x̄ = sign(x)". Table 1 (verbatim numbers):

> "∃f (perfect): contrastive loss 4.939, accuracy 100 · ∃g (spurious): contrastive loss **4.939**,
> accuracy **50** · MLP + Adam: 5.039 ± 0.001, 74.1 ± 4.3 · MLP + Adam + wd: 5.040 ± 0.002,
> 89.5 ± 4.9 · Linear: 5.134 ± 0.002, 99.5 ± 0.1"

Figure 2 caption: "There exist global minimizers of L_cont with perfect (top right) and worst
possible (bottom right) downstream classification error L_clf." — **identical loss value (4.939,
both global minimizers), same augmentations, downstream 100% vs 50% (= random guessing for the
balanced binary task)**. The theorem-form of the whole package on this example:

> "**Corollary 4.2.** Consider the setting from Example 1. … (a) All function class-agnostic
> transfer guarantees are vacuous. (b) For any f ∈ F_φ, we have
> L_clf(f; ȳ*) ≤ 32k · (L_spec(f) − inf_{f*∈F_φ} L_spec(f*))." ⟨math transcribed; F_φ = linear class⟩

### 1e · Amendments to our shorthand

- **Regime scoping.** Provable vacuity requires disjoint augmentations (Definition 4.1), plus
  technical conditions (d = O(N/log₂N), balanced labels, uniform D_X̄ — "technical in nature and
  can be potentially relaxed"). Near-disjoint extension: Lemma 4.4 shows HaoChen-style bounds are
  "non-vacuous only when L_cont(f) ≤ inf_{f*} L_cont(f*) + O(τ²), which is a stringent condition to
  satisfy"; and empirically "we evaluate this augmentation classification metric on standard
  augmentations on images, and find that the accuracy achievable is almost 100%, suggesting that we
  might be closer to the disjoint augmentation setting than we think". For genuinely overlapping
  augmentations the paper does NOT prove vacuity — there the evidence is empirical (§5,
  label-orthogonal training; see §2b below). Our unqualified "is vacuous" should be read/written as
  "provably vacuous in the (near-)disjoint regime; empirically unreliable in general".
- **"Arbitrarily different"** (task phrasing, not in our docs verbatim): the proven spread is
  perfect ↔ 1/2 − Õ(√(d/|X̄|)), i.e. up to random guessing on balanced binary — the maximal
  possible spread for the task, but bounded by it; the paper's words are "worst possible" /
  "near-trivial" / "close to random guessing", never "arbitrarily".
- Attribution detail: the paper notes Robinson et al. '21 (Prop. 1) has a related disjoint-regime
  lower bound ("shortcut solutions" / feature suppression), theirs being "shown in much more
  general settings".

---

## 2 · Claim 2 — THEORY_MAP row — first half **VERIFIED**, second half **correctly labeled OUR inference (paper is head-silent analytically)**

### 2a · "same loss, same augmentations, different function class ⇒ different downstream" — **VERIFIED**

Near-verbatim in the abstract:

> "Extensive experiments on image and text domains highlight the ubiquity of this problem —
> **different function classes and algorithms behave very differently on downstream tasks, despite
> having the same augmentations and contrastive losses.**"

Intro, first key phenomenon: "**Function class sensitivity.** Downstream performance of a
representation depends not just on its contrastive loss, but it is also sensitive to the function
class (architecture) and training procedure used to learn it." Theorem-form = Corollary 4.2(a)
vs (b) (§1d above): on the same example with the same augmentations, function-class-agnostic
guarantees are vacuous while the linear class provably transfers. Formal-abstraction form =
Eq. (6) vs Eq. (5): "the downstream performance at a particular value of contrastive loss depends
also on the representation function class, which we also find to be true in many experiments."

**Adversarial precision on "same loss":** in Table 1 the *exactly equal* loss pair (4.939/4.939,
100% vs 50%) is two representations in the *unrestricted* class — the function-class contrast
there is "unconstrained vs linear/MLP", and across the named architectures the losses are NOT
equal (MLP 5.039 beats linear 5.134 on loss yet transfers worse, 74.1 vs 99.5 — a loss/downstream
*reversal*, stronger than same-loss-different-downstream). The "same loss, different class,
different downstream" reading at matched loss values is carried by (i) Corollary 4.1 (at *every*
loss value the agnostic bound is vacuous), (ii) the Figure 3 trajectory plots through
(L_cont, accuracy) space, and (iii) hash-pixels (§5.1): "ViT and MLP-Mixer representations make
the contrastive loss much smaller than ResNet, but have close to random guessing downstream
performance. ResNet … does well on the downstream task, despite being far from minimizing the
contrastive loss." The row's arrow-statement is a fair compression; if we ever need the
single-quote version, use the abstract sentence.

### 2b · Beyond disjoint augmentations (supports "ubiquity", empirically)

§5.1, label-orthogonal training — with the FULL standard SimCLR augmentations (overlapping):

> "However for the same augmentations, we can find pathological representations that have small
> contrastive loss but poor downstream performance, by introducing an adversarial modification to
> the training algorithm and minor tweak to ResNet architecture … This suggests that guarantees
> depending only on the contrastive loss, but not the function class or algorithm, cannot explain
> the effectiveness of contrastive learning with standard architectures and augmentations."

### 2c · "The head is precisely such a function-class device" — OUR inference; paper mentions the projector only as setup, never analyzes it

- The phrase "projection head" appears **nowhere** in arXiv v1 or the ICML camera-ready (grepped
  both full texts, case-insensitive; the only "head" hits are ViT/Transformer attention heads).
- The projector appears **once**, in Appendix D.2 (CIFAR-10 experimental setup), as part of the
  model definition:

  > "In each model, the representation for contrastive learning is computed by adding an extra
  > MLP (**projection layer**) on top of the base model, as proposed in Chen et al. [2020]. The
  > projection layer has 1 hidden layer with 2048 dimensions, followed by a batch norm layer and
  > ReLU non-linearity, and output dimensionality of 1024."

- **The head is folded into f.** Downstream evaluation is "linear classification accuracy of the
  learned representation f" and the hyperparameter table lists "Representation dimension 1024" =
  the projector's output dimensionality — i.e. in their vision experiments the probed
  representation IS the projector output (trunk+head = one function f; no trunk/head distinction
  exists anywhere in their theory: the function class F is the whole encoder).
- **Source archaeology (unpublished — do not cite as the paper):** the arXiv v1 source tarball
  contains `table_fig/transfer_fig.tex`, a figure caption for an experiment that is `\input`
  nowhere and absent from both compiled versions: "we perform downstream classification using
  either the representation obtained from the projection head (**which is closer to what our
  theory describes**), or the representation obtained before the projection head (**which is what
  is typically done in practice**)" (standard vs permuted-pixel ResNet at matched NCE loss). So
  the authors demonstrably considered the head/pre-head split and its relation to their theory,
  and cut it. Corroborates that "theory space = head output" was the authors' own reading, but it
  is not in any published version.

**Verdict on the row as written:** first half VERIFIED; second half is our inference and the row
already presents it as such (separate sentence, [P] tag) — the paper neither states nor contradicts
it. It is *consistent* with the paper: the head is part of f, so head choice is literally part of
the function class F in their Eq. (6); what the paper never does is single out the head, or any
architecture component, as the locus of the inductive bias. Keep the [P].

---

## 3 · Claim 3 — minimizers vs loss values — **BOTH, verified; recorded**

The vacuity is stated at three strengths, all in the paper:

1. **Exact/global minimizers.** "Both lemmas show that when the representation dimension is small
   relative to the size of the input space (as is typical) and the augmentations are disjoint,
   **there exists a global minimizer of the contrastive loss with vacuous transfer to
   downstream**." (§4.1) Also Figure 2: "There exist global minimizers of L_cont with perfect …
   and worst possible … downstream classification error"; and on HaoChen's spectral loss (§4.2):
   "the global minimizer of L_spec is not unique, and some of those could be terrible on
   downstream, as in our proof for Lemma 4.1." Intro bullet ("Brittleness of transfer"):
   "**Minimizing the contrastive loss to optimality** can sometimes have a non-monotonic,
   deleterious effect on downstream performance, despite the augmentations being effective for
   some function classes."
2. **Every loss value (hence all near-minimizers).** Lemma 4.1 quantifies over ANY f* — "for any
   f*: X → R^d there exists a f̂ … L_cont(f̂) ≤ L_cont(f*)" — i.e. at every achievable loss level
   there is a bad representation at-or-below it (and by proof step 2, one with *exactly* the same
   loss as the identity-shuffled f̂). Corollary 4.1 is stated "**for all** f: X → R^d", killing any
   bound monotonic in the loss value at every value, which covers suboptimality-gap
   (near-minimizer) bounds like HaoChen's "L_clf(f) ≤ α(L_cont(f) − min_{f*} L_cont(f*)) + β" (§2).
3. **Near-disjoint quantitative version.** Lemma 4.4: for 1−τ-disjoint augmentations, the
   HaoChen guarantee is "non-vacuous only when L_cont(f) ≤ inf_{f*} L_cont(f*) + O(τ²), which is a
   stringent condition to satisfy" — near-minimizer bounds survive only inside an O(τ²) ball, and
   empirically τ ≈ 0 for standard image augmentations (input-identification accuracy "almost 100%").

The mirror-image positive result also spans both: Corollary 4.2(b) discussion — "finding the
minimizer (**or an approximate minimizer**) of the contrastive loss, **within the class F_φ**, is
sufficient to guarantee good downstream performance". So the paper's picture: with the function
class incorporated, value-based bounds work (approximately-minimizing linear f transfers); without
it, they fail at exact minimizers, at near-minimizers, and at every loss value in the
(near-)disjoint regime.

---

## 4 · Per-claim verdict table

| # | Doc claim | Verdict | Anchor |
|---|---|---|---|
| 1 | "Loss-level analysis is vacuous without function-class assumptions" | **VERIFIED-WITH-AMENDMENT** (add regime scoping: provable in disjoint/near-disjoint augmentations — Def. 4.1, Lemma 4.4; empirical elsewhere; paper says "in some settings") | Lemmas 4.1–4.2 (= Thm C.1), Corollary 4.1, Eq. (5) vs (6), Table 1 |
| 2a | "same loss, same augmentations, different function class ⇒ different downstream" | **VERIFIED** (abstract near-verbatim; theorem-form = Cor. 4.2(a)/(b); note Table 1's exact-equal-loss pair is within the unconstrained class, and across architectures the pattern is even a loss/accuracy reversal) | Abstract; Cor. 4.2; Table 1; Fig. 3; §5.1 hash-pixels |
| 2b | "The head is precisely such a function-class device" | **OUR INFERENCE, correctly flagged [P]** — paper never says "projection head"; sole projector mention is Appendix D.2 setup ("extra MLP (projection layer) … as proposed in Chen et al."); head is inside f, probed at projector output; unpublished source caption shows authors equated theory space with head output but cut the figure | App. D.2; grep of both compiled versions; `table_fig/transfer_fig.tex` (unpublished) |
| 3 | Vacuity: minimizers vs values | **BOTH** — exact global minimizers ("there exists a global minimizer … with vacuous transfer") AND all loss values/near-minimizers (Lemma 4.1 "for any f*"; Cor. 4.1 "for all f"; Lemma 4.4's O(τ²) condition) | §4.1 Takeaways; Lemma 4.1; Cor. 4.1; Lemma 4.4; Cor. 4.2(b) |

Proposed doc touch-ups (await discussion, not applied): (i) DESIDERATA_FRAMEWORK §3 line could
gain "(provably so under (near-)disjoint augmentations; empirically beyond)"; (ii) THEORY_MAP row
is accurate as compressed — optionally cite "Cor. 4.1 + Cor. 4.2" in the row for the arrow
statement.
