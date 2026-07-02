# E07 — Texture/shape cue-conflict for MIM & JEPA   `STUB — detailed when scheduled (see docs/ROADMAP.md)`

> Derived from docs/report/ssl-projector-gap-report.html (compiled 2026-07-01, quote-verified there). Tags: [E] established · [P] plausible interpretation · [O] open question.

**Status:** stub · phase M5+ (per ROADMAP index) · **Pre-registered:** no

## Hypothesis (from report, to be refined before pre-registration)

At matched ViT-B: shape-bias ordering **DINO ≥ contrastive > I-JEPA > MAE** (Park et al.'s
frequency analysis predicts MIM texture reliance); shape bias measured from z **exceeds** that
from h for view-based methods (color/texture nuisances filtered by the head). [P]

## Protocol sketch

Geirhos cue-conflict stimuli through frozen features + linear probe (trained on IN);
shape-vs-texture decision fractions; error-consistency vs humans and vs supervised ViT;
Stylized-IN accuracy. Both spaces.

## Datasets / models

Cue-conflict Stylized-IN stimuli (Geirhos shape-bias psychophysics); matched-architecture ViT-B
checkpoints across families; supervised ViT (DeiT recipe) baseline per §5.0 anchors.

## Expected failure to reveal

MIM/JEPA texture-biased beyond supervised ViT baselines; completes the missing rows of the
psychophysics literature (named absence, §4.5 — MIM/JEPA models are absent from the 85k-trial
error-consistency literature; note ViT architecture, not SSL objective, moves shape bias in the
contrastive era, Naseer et al.).

## Interpretation guide

Family-level inductive-bias fingerprints at h; input to "which pretraining for which deployment"
guidance. First two-space bias measurement (h vs z shape bias). Fills OPEN_PROBLEMS OP-11.

## Dependencies (features/datasets/models needed)

- Cue-conflict stimulus set + Stylized-IN staged locally.
- Matched ViT-B public checkpoints across families (Track C, head-inventory verified) — the
  hypothesis is stated at ViT-B; core-7 ViT-S gives the toy echo.
- IN-trained linear probes per model per space (reuse E1 probes); error-consistency tooling
  (human response data from the Geirhos releases).

## Results

*(empty)*

## AGREED TAKEAWAY

*(empty — never filled unilaterally; see CLAUDE.md)*
