# E05 — Object-absent & background-only crops   `STUB — detailed when scheduled (see docs/ROADMAP.md)`

> Derived from docs/report/ssl-projector-gap-report.html (compiled 2026-07-01, quote-verified there). Tags: [E] established · [P] plausible interpretation · [O] open question.

**Status:** stub · phase M5+ (per ROADMAP index) · **Pre-registered:** no

## Hypothesis (from report, to be refined before pre-registration)

View-based models place background-only crops close (in h) to the class centroid of the absent
foreground — they learned crop-cooccurrence, not objects. MAE/I-JEPA (no cross-crop positives)
show weaker binding; DINO's multi-crop shows the **strongest** (its local views trained exactly
this mapping). [P]

## Protocol sketch

Build **COCO-crops** from COCO segmentation masks:
(a) object-present crops; (b) object-absent crops from images containing the object elsewhere;
(c) pure-background crops from object-free regions; (d) RRC-simulating crop pairs with logged
IoU-with-object.
Measure: cos(h(crop), class centroid); retrieval rank of source image; k-NN class of crop;
linear-probe accuracy conditional on object-in-crop IoU bins. Repeat in z.

## Datasets / models

COCO (masks), IN-9 (factorized), VOC. All §5.0 models, both spaces.

## Expected failure to reveal

High class-recoverability from object-absent crops for multi-crop distillation models;
accuracy-vs-IoU curves **flat** (context-driven) rather than rising (object-driven).

## Interpretation guide

A graded, per-family measurement of the Purushwalkam–Gupta/Mishra crop tension — resolving it from
a dataset dispute into a mechanism measurement; connects to déjà vu (Meehan) without the privacy
framing. Also the direct probe of the report's [O] "object-absent local views remain unstudied"
(OPEN_PROBLEMS OP-14) and the déjà-vu-adjacent memorization angle (OP-13).

## Dependencies (features/datasets/models needed)

- **COCO-crops bank** — purpose-built from COCO segmentation masks (to build; §5.0 dataset
  additions), with logged IoU metadata.
- COCO, IN-9, VOC locally staged.
- Frozen features for the crop bank in both spaces (new extraction pass over the bank).
- Class centroids + retrieval index from the E1 feature store.

## Results

*(empty)*

## AGREED TAKEAWAY

*(empty — never filled unilaterally; see WORKFLOW.md)*
