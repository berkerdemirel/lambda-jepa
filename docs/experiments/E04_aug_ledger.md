# E04 — Augmentation-information ledger   `STUB — detailed when scheduled (see docs/ROADMAP.md)`

> Derived from docs/report/ssl-projector-gap-report.html (compiled 2026-07-01, quote-verified there). Tags: [E] established · [P] plausible interpretation · [O] open question.

**Status:** stub · phase M3 (per ROADMAP index) · **Pre-registered:** no

## Hypothesis (from report, to be refined before pre-registration)

Nuisance decodability at h orders **MAE > I-JEPA > DINO ≈ BYOL ≈ SimCLR ≫ each method's z**;
LeJEPA (also projector-based) sits among the joint-embedding methods, and its z retains less
nuisance than its h — the buffer is present in LeJEPA too. [P] Report Table 5: "Nuisance
decodability at h: MAE > JEPA > contrastive ≫ their z; LeJEPA's h retains more than its z
(projector present, as in all)."

## Protocol sketch

Frozen-feature probes predicting: crop position & scale, color-jitter parameters, rotation
(applied synthetically), background category (Places-labeled), object position/scale (bbox
datasets). Both spaces, all models. This is SimCLR Table 3 and the CASSLE probe, run for the first
time across the modern zoo. (Probe = the SimCLR-Tab.3 protocol, generalized.)

## Datasets / models

All §5.0 models (both tracks at M3: controlled retrains + public checkpoints, provenance never
mixed in-table). Synthetic augmentation-parameter sets; Places-labeled backgrounds; bbox datasets
for object position/scale.

## Expected failure to reveal

Invariance-trained models retain far more nuisance at h than their papers imply (RCDM
prediction) — the "invariant representation" claim survives only in z.

## Interpretation guide

The information ledger quantifies the buffer directly; LeJEPA's cell tests whether a
heuristics-free, distributionally-regularized projector buffers differently from an
invariance-trained one. (Report reading: expect MAE ≫ contrastive at h for nuisances; the
interesting cell is DINO at h.) Direct test of the buffer theory (§2.5 family 1); locates LeJEPA
among projector methods.

## Dependencies (features/datasets/models needed)

- E1 feature store (both spaces, all models, head taps) — features largely cached from Phase 1.
- Augmentation-parameter logging in the extraction path (crop box, jitter magnitudes, rotation
  angle applied synthetically at extraction time).
- Places-labeled background data; a bbox dataset (COCO/VOC) for object position/scale targets.
- Frozen-feature regression/classification probe infrastructure (PROTOCOL probes).

## Results

*(empty)*

## AGREED TAKEAWAY

*(empty — never filled unilaterally; see WORKFLOW.md)*
