# E10 — Causal head interventions: move the loss, keep everything else

> Derived from docs/report/ssl-projector-gap-report.html §5.3 E10 — "the decisive experiment".

**Status:** draft · **Phase:** M4 (IN-100 retrain grid) · **Gate:** requires M2/M3 conclusions
USER-APPROVED (G-M2)
**Pre-registered:** ❏ pending (frontier predictions to be locked before the grid launches)

## Hypothesis

The h↔z dissociation is **caused** by head placement, and it trades off: enforcing desiderata at h
increases their satisfaction there while reducing nuisance retention and (for misaligned pretexts)
transfer breadth — a measurable frontier, not a cliff.

## Grid (~15–18 configs × 2 seeds, IN-100 ViT-S/16)

1. **VICReg**: var/cov terms at z (standard) | at h (expander kept for invariance only) | at both.
2. **SimCLR**: projector depth 0/1/2/3 (0 = DirectCLR-style loss on backbone sub-vector).
3. **BYOL**: trained predictor → DirectPred closed-form spectral predictor (stretch arm).
4. **LeJEPA**: standard (SIGReg on 3-layer projector, as shipped) | SIGReg moved onto the backbone
   (projector = identity) | projector depth 0/1/2/3 — *the* intervention that tests "impose the
   desideratum at h".
5. **DINO**: + KoLeo regularizer at h (the DINOv2 hint, isolated).

Full E01 battery + E04 ledger + (available) transfer suite on every run.

## Pre-registered expectations (report)

Desiderata-at-h variants: higher metric satisfaction at h, lower augmentation-info retention,
equal-or-better aligned-task accuracy, worse misaligned-task transfer — the Guillotine alignment
story, made causal. **LeJEPA decision cell**: if standard (projector) LeJEPA beats SIGReg-on-backbone,
the buffer is real even for a distributional constraint; if close, isotropy is gentle enough to
impose at h and the projector is dispensable for it. *Either answer is a finding.* [O]

## Compute

≈15–18 H100-days per seed pass at the 2-slot cap (ROADMAP M4).

## Results

*(numbers only)*

## AGREED TAKEAWAY

*(empty)*
