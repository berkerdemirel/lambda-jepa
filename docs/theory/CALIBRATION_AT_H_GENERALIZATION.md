# Hypothesis — the slight-desideratum-at-h principle (generalizing the moment floor)

**Status: HYPOTHESIS (Berker 2026-07-14), to reconsider. NOT a plan, NOT a takeaway.** Recorded per
his instruction ("note this down — it would lead something more general and bigger").

## The question
Is the calibration-toward-h benefit **specific to our moment-KL floor**, or is it an instance of a
**general principle**:

> For a broad class of SSL methods, applying a *small-strength* version of the method's **own
> desideratum term** — its regularization term and/or its invariance term — at the backbone
> representation **h** (on top of the full-strength version at z) improves h.

The moment-KL floor (E12) and the view-invariance assist (H-wave) are then just the **easy-to-implement
instances**, not the whole story. If the principle holds across terms and methods, the contribution is
**bigger and more general** than "a moment floor helps."

## Implementation gradient (Berker's framing)
- **Easy:** invariance pull and moment-KL at h — simple functional forms, already built
  (`h_inv` in lejepa/vicreg; `MomentFloor` in `_common.py`). Directly droppable at any tap.
- **Hard / the real test:** **DINO's loss at h** — its centering+sharpening / prototype-assignment
  (clustering) objective has no obvious slight-at-h form without instantiating the prototype head at
  h. This is the small challenge that decides whether the principle is *general* (every method's own
  term transfers) or *term-specific* (only the simple regularizers transfer). SwAV/assignment losses
  share this difficulty; contrastive/variance/covariance/decorrelation terms are closer to the easy end.

## Why it matters
- If general: the two-space audit's calibration-toward-h finding becomes a **method-design principle**
  ("mirror your z-desideratum weakly at h"), not a single trick — a larger claim and a new experiment
  line (candidate E17+: the desideratum-transfer matrix — each method's own term, slight, at its
  declared h).
- The DINO case is the falsifier: if clustering-at-h can't be made to help (or can't be implemented
  without smuggling in the head), the principle is bounded to differentiable pointwise regularizers.

## Anchors
- Instances so far: E12 (moment-KL floor at h), H-wave (inv assist at h). See `docs/experiments/E12_moment_floor.md`.
- Nearest prior work: Kalapos 2408.07519 (hard ZCA layer at h, method-agnostic) —
  `docs/literature/related_work/kalapos_whitening_improves_ssl.md`. Note: their result is a *premise*
  for the easy end (whitening/decorrelation at h helps broadly); it does NOT test the desideratum-transfer
  generalization (each method's OWN term), which is the novel part here.
- Reconsider after the H-wave lands (gv2/gvcls give the clean CLS floor + inv-assist result at h).

## Sharpened framing (Berker 2026-07-14) — burden off the MLP, and a test of the method's own claim
The intervention is not "does mirroring help" but **"take some burden off the projector/MLP."**
Methods enforce their desideratum at z (post-projector); the MLP carries the claim. **VICReg and
LeJEPA claim distributional properties *for the representation* (var/cov; isotropy) yet only enforce
them at the expander/projector** — h never has to satisfy the claim. Pulling the term to h tests
whether the representation can carry the property the method asserts for it, and by how much the MLP
relaxes.
- **Two readings.** vicreg/lejepa = a test of the method's *own representational claim* (does h carry
  the distributional property). simclr/byol/dino = a test of *mechanism transferability* (their term
  is a contrastive/predictive/clustering burden, not a distributional claim about h).
- **Adjudicates the guillotine.** Bordes: the head is a *protective buffer* (desideratum belongs off
  h). The methods' claims say the opposite. Burden-shift is the referee, per method.
- **Measurable mechanism = head burden** (primary readout): projector invariance-jump, operator
  head-Lipschitz (σ_max/layer), diag-KL ladder. f7 already showed the inv-jump shrink (0.37→0.16).
- **Dose caveat — "some", not full.** E12-T1: full second-moment enforcement at h scrubs semantics
  (guillotine damage is real at high dose). E12-T2: small dose = net-positive conditioner with an
  interior optimum. Target the small-dose regime: head-burden drops AND h-probe holds.
- **Scope decisions:** ijepa is OUT (loss already targets backbone patch reps — no z to mirror); dino
  uses a small *linear* prototype head at h (Berker: fair). byol has only the align step (no
  non-collapse loss); open design Q = predictor-at-h vs EMA-only.
- Runnable plan: `GOAL_PROMPT_hpull.md` (self-explanatory run names `in100.<method>.s0.hpull_<term>`;
  head-burden primary; small-dose discipline; discussion-first for byol/dino).

## VERDICT — E17 executed; hypothesis CONFIRMED IN BOUNDED FORM and re-grounded (2026-07-16; D-039; details: E17 card + `docs/report/e17_desiderata_at_h_findings.md`)

The principle survives with its content changed. What transfers to h beneficially is NOT "the
method's own desideratum" in general and NOT calibration/Gaussianity — it is the **affine/spread
component**: mean-removal (the h-cone is a mean offset, and removing it is what the projector's
"invariance jump" actually was) plus a non-absorbable spectral floor (rank/tail-decrowding),
under **cluster-blindness** as the safety condition. The moment-KL floor helped BECAUSE it is
exactly this and nothing else — "safety (cluster-blind) × activity (rotation-swept,
non-absorbable floor)"; f5/f2/A3/sigreg quadrangulate the two ingredients and the dose bound.

Boundaries established:
- **Shape enforcement at h taxes** (lejepa's own SIGReg: mean class-pair d′ −17%, trunk-deep) —
  the cluster-SEEING member of the family is the harmful one. Gaussian shape per se was never the
  lever; under f2 the higher moments stay data-driven (Varimax coords leptokurtic).
- **The invariance half is a connectivity hazard, not a burden to offload**: alignment is born in
  the trunk; explicit inv at h shrinks augmentation clouds and disconnects the class manifold
  (touch% vs kNN, monotone within every ±inv family; fig `e17_touch_vs_knn.png`). Shape-dominant
  mixing (f8, 4.5:1) is the favorable corner.
- **The DINO falsifier**: assignment losses at h train and offload but RESHAPE (margin fattens,
  lin↑ / knn↓, fragmentation) — the principle stays bounded to pointwise/pairwise spread
  regularizers, as pre-registered.
- **vicreg**: mean-blind var+cov removes the cone only indirectly (variance-hinge saturation) and
  dose-hungrily — at 6×/ep100 it joins the winners; the floor does the same at λ=.02. Instrument
  dose-efficiency, not possibility, separates the terms.

Open fork (approved direction, next session): the declared-prior arm — SIGReg-to-t_ν (finite-ν
per the P5 Varimax datum) at the same tap/dose isolates "shape enforcement per se harms" vs
"wrong shape target harms".
