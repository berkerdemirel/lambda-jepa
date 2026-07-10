# Box-query SSL — a baby idea (saved discussion, not a proposal)

**Status: BABY IDEA — saved 2026-07-09 at Berker's request. NOT concluded, NOT agreed, NOT an
E-card, no DECISIONS row attached; nothing here is scheduled or pre-registered.** The mechanism
is Berker's (his pre-project core idea); the analysis is Claude's take from one conversation
(2026-07-09), unreviewed. If this ever advances, it goes through the normal path: discussion →
proposal → pre-registration. Companions: `trd_pi_theory_framework_v2_0_draft.md` (Appendix E is
the method this is read against), `SIGREG_DIM_SCALING.md` (the (K, N) scaling used below),
PROTOCOL §9 / D-009 (relative representations).

## 1 · The mechanism (Berker, pre-project)

Given the nuisance mechanism G (the augmentation/view stack), do **not** align global image
features directly. Instead:

1. **Query = arbitrary rectangular box**; box features via ROI-Align on the feature map.
2. **Box alignment:** features of *corresponding* boxes (same region mapped through the
   geometric transform relating the views) must match across views.
3. **Within-image relational term:** among multiple sampled boxes of one image, the pairwise
   cosine relations (the box-feature Gram) must match across views.
4. **Batchwise negative structure** (form open) as the anti-collapse/spread device.

Berker's specific question: shape priors (VICReg/LeJEPA-style marginal constraints) vs data as
negatives (SimCLR-like) — which belongs in slot 4, and how does the whole thing relate to the
TRD-π mechanism.

## 2 · Claude's take (one pass, unagreed)

**Nearest neighbors, for honest novelty accounting:** SCRL (random boxes in the crop
intersection, BYOL-style ROI-align matching), SoCo (selective-search boxes + ROI-Align,
detection-oriented), DetCon (mask-pooled regions), VICRegL (VICReg on geometry-matched local
features). Box alignment alone is occupied territory. The **distinctive piece is the
within-image relational term across views**: relational consistency in the literature is mostly
inter-image (ReSSL; RKD in distillation); "the internal Gram of sampled regions is a
view-invariant object" as a first-class loss term is not a crowded slot. And that term is
literally **D-009 promoted from audit tool to training signal**: a box's cosine row against the
other boxes of its image is a relative representation with in-image anchors.

**Framework reading, term by term:**

- Box matching = the alignment surrogate applied to a *query-indexed* feature f(x, box) — a
  change of **declaration**, not just estimator: the implied task family now includes localized
  predicates, so the sufficiency target (D1) is "content + layout" instead of "content
  marginalized over layout." Directly responsive to the report's E05–E09 failure axis; boxes in
  the view intersection manufacture *correct* correspondence by construction (vs OP-14's
  object-absent local views, where multi-crop manufactures wrong supervision).
- The relational term = **gauge-relaxed alignment**: Grams are invariant to a per-view global
  rotation of feature space, so it transmits layout structure while tolerating what the direct
  term punishes. Redundant if the direct term is perfectly satisfied; its value is as the
  smoother signal and as robustness when the direct match has an irreducible floor (§ risks).
- **The design has a collapse mode image-level SSL does not: query-collapse.** A model that
  ignores the box and returns its global image feature satisfies terms 2 AND 3 (corresponding
  boxes trivially agree; Gram = all-ones in both views). Standard batch negatives do NOT fix
  this — per-image-constant box features still separate images, so an InfoNCE over
  boxes-across-images is satisfied. The anti-collapse slot has two jobs here (inter-image
  spread; within-image query-sensitivity) and no single standard device covers both.

**Shape priors vs data negatives (the asked question):**

- Same slot in the LDM/TRD-π reading (entropy/marginal estimators), unequal instruments.
  E10-T1 + the channel theorem: a sliced/moment prior at K = 384–512 polices first two moments
  and degeneracy; shape sensitivity is dead (1/K²–1/K⁴). Data negatives are a **data-adaptive
  adversary** — every batch sample is a witness placed where mass actually concentrates —
  the one standard mechanism with per-sample, high-capacity in-loss marginal pressure.
- The region setting flips two things: (i) **N-multiplier** — ~64 boxes/image at bs 256 ⇒
  ~16k region features/step; shape signal ∝ N at fixed K (R4c), so box-level SSL is the regime
  where sliced priors become measurably non-vacuous per step (correlated boxes shrink effective
  N; the multiplier is still large). (ii) **Region-level false negatives** — two skies, two
  grass patches, two car hoods across images are near-identical content; region negatives
  collide far more than image negatives, and repelling them fights exactly the dense
  sufficiency being bought. Naive batchwise box negatives < naive image negatives.
- **Recommended shape (Claude's, unagreed): role split, not a winner.** Negatives at the
  image/global level only (collision-manageable, adversarial pressure strongest); box-level
  terms negative-free (alignment + relational); and a **within-image variance floor over the
  box-feature population** as the query-collapse guard — "features must respond to the query"
  is exactly a variance constraint across boxes of one image, which a VICReg-style term
  expresses and negatives do not. Optional: covariance/whitening on the pooled region
  population. Each slot separately auditable.

**Relation to the program:** an admissible Stage-2 alignment block for the assembled method
(Appendix E fixes placement/governance, not the surrogate). It attacks the two-space problem
from the opposite direction from E10 — instead of moving the loss to h, it widens what
alignment binds so the token field must carry localized content (the head can no longer absorb
layout). Testable prediction: box-trained models show higher h-side desiderata transfer and a
smaller attentive-vs-linear gap on token spaces (E11 axis). **Cheap pre-test available without
training:** final-layer patch tokens for probed branches are already in the store — "do
existing M2 models have view-stable within-image region Grams?" is computable post-hoc and
would measure the relational term's headroom on the current grid. Longer-term: the "arbitrary
query" framing makes the representation an operator (query → descriptor) — a concrete,
geometry-indexed first step toward the slot/set world the framework currently scopes out
(OP-e).

**Risks / hidden declarations:**

- **Context leakage:** ViT box features attend outside the box; corresponding boxes under
  different crops see different context ⇒ irreducible alignment floor, and simultaneously an
  implicit *context-invariance* pressure on local descriptors (E06 fork: plausibly helps group
  robustness, may hurt scene-level tasks — a real trade, not a bug).
- **The box-sampling distribution is a hidden declaration:** scale/aspect/location of queries
  is a prior over which granularities matter (small-box-heavy = texture-bias pressure) and
  belongs in the declared tuple next to G.
- Position shortcut is mostly self-controlled: the geometric transform between views makes
  pure-position solutions non-invariant.

## 3 · If it ever advances (none of this is scheduled)

Discussion first. Natural first artifacts, in order of cheapness: (1) the post-hoc region-Gram
stability cell on the existing M2 store (no training); (2) a toy-frame pilot of the hybrid
(alignment + relational + within-image variance floor + global negatives) with a
**query-collapse monitor pre-registered from day one** (within-image box-feature variance);
(3) only then any comparison against the corners. Each step needs its own proposal per the
collaboration contract.
