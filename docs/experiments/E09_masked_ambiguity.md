# E09 — Masked-region ambiguity (content, not location)   `STUB — detailed when scheduled (see docs/ROADMAP.md)`

> Derived from docs/report/ssl-projector-gap-report.html (compiled 2026-07-01, quote-verified there). Tags: [E] established · [P] plausible interpretation · [O] open question.

**Status:** stub · phase M5+ (per ROADMAP index) · **Pre-registered:** no

## Hypothesis (from report, to be refined before pre-registration)

For I-JEPA: target-block predictability from context varies enormously; on high-ambiguity blocks
the predictor regresses to a mode-average whose norm/specificity is measurably lower; models
trained with more ambiguous masking encode less local semantics (DMT-JEPA's complaint —
"insufficient understanding of local semantics" — made quantitative). [P]

## Protocol sketch

Define ambiguity operationally:
(a) ensemble disagreement of k independent predictors (retrain the predictor k× on the frozen
encoder); (b) conditional-entropy proxy via a small generative in-filler; (c) human-free structural
proxies (distance from context, texture homogeneity).
Correlate ambiguity with: prediction error, target-representation norm, downstream per-region
segmentation quality. Compare MAE (pixel loss) vs I-JEPA (latent loss) vs data2vec on **identical
masks**.

## Datasets / models

I-JEPA, MAE, and data2vec 2.0 (the MAE↔I-JEPA midpoint, a §5.0 roster addition) under identical
masks; segmentation-annotated data for per-region downstream quality (§5.0 dense suite:
ADE20K/VOC frozen-feature protocol).

## Expected failure to reveal

Latent-space averaging under ambiguity produces "semantic blur" — regions whose representations
are systematically less class-informative; pixel-space MAE degrades more gracefully (blur is
visible, not semantic). [P]

## Interpretation guide

The first content-multimodality measurement for masked prediction (named absence; OPEN_PROBLEMS
OP-16 — StoP handles *location* ambiguity only). Directly informs masking-policy design and the
stochastic-JEPA line. Background theory: masked prediction is ill-posed ("we can guess that there
is a tail, but we cannot determine its exact location", Bar et al.); the encoder "cannot adaptively
modulate the type of predicted... features based on the feasibility of the masked prediction task"
(Littwin et al.) — when context underdetermines the target, the objective still demands a point
estimate.

## Dependencies (features/datasets/models needed)

- k× predictor retrains on a frozen I-JEPA encoder (extra compute; small — predictor only).
- A small generative in-filler for the conditional-entropy proxy.
- data2vec 2.0 checkpoint/port (roster addition — not in core-7; needs a MODELS.md row).
- Mask-identical evaluation harness across MAE / I-JEPA / data2vec; per-region segmentation eval
  from the §5.0 dense suite.

## Results

*(empty)*

## AGREED TAKEAWAY

*(empty — never filled unilaterally; see CLAUDE.md)*
