# E08 — Small objects & multi-object scenes   `STUB — detailed when scheduled (see docs/ROADMAP.md)`

> Derived from docs/report/ssl-projector-gap-report.html (compiled 2026-07-01, quote-verified there). Tags: [E] established · [P] plausible interpretation · [O] open question.

**Status:** stub · phase M5+ (per ROADMAP index) · **Pre-registered:** no

## Hypothesis (from report, to be refined before pre-registration)

Feature suppression (Chen et al.: "the dominant object would suppress the learning of smaller
objects") manifests as: multi-label probe recall falling with object-size rank; global-vector
methods (SimCLR/BYOL/DINO-CLS) worst, patch-token methods (iBOT/DINOv2 patch pooling, MAE) best;
**z strictly worse than h everywhere** (the head compresses toward the dominant object). [P]

## Protocol sketch

Frozen-feature multi-label linear probes on COCO/VOC; per-object-size-decile recall; retrieval
tests: does the image embedding retrieve the second/third object's class neighbors? Compare
[CLS]/global vs patch-pooled h.

## Datasets / models

COCO, VOC multi-label. All §5.0 models; patch-token pooling variants where the architecture
provides them (iBOT/DINOv2 in the M5+ roster; MAE/DINO in core-7).

## Expected failure to reveal

Steep size-rank decay for contrastive global vectors; **DINO-CLS ≈ SimCLR despite dense-task
fame** ([CLS] vs patch dissociation).

## Interpretation guide

Connects instance-discrimination's single-vector bottleneck to dense-recognition needs; motivates
patch-level desiderata (the iBOT/DINOv2 direction) as the fix the field already stumbled into. [P]
Ties the §4.1 feature-suppression literature (Chen; Robinson; Xue: SGD simplicity bias) to a
measurable per-size-decile curve, informing method choice for detection pipelines.

## Dependencies (features/datasets/models needed)

- COCO + VOC multi-label ground truth with per-object size metadata (size-decile binning).
- Patch tokens stored for the probed models (feature-store policy D-005: final-layer/final-epoch/
  probed-branch only — E08 must fit that budget or request a DECISIONS row).
- Multi-label probe + retrieval harness on frozen features, both spaces.

## Results

*(empty)*

## AGREED TAKEAWAY

*(empty — never filled unilaterally; see WORKFLOW.md)*
