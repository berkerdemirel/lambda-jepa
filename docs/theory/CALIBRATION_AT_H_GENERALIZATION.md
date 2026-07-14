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
