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

## LeJEPA SIGReg-on-h arm — detailed design (drafted 2026-07-02, PROPOSED)

**The linear-absorption confound, and why "on h" is three arms, not one.** F4 rules that a linear
map adds no *probe* capacity (hence LeJEPA h = z.embed). But SIGReg is not linear-invariant: a
trainable Linear can whiten any full-rank cloud, so a distributional constraint imposed after a
Linear can be absorbed BY that Linear, leaving the trunk anisotropic. "Impose isotropy at h"
therefore splits into inequivalent placements:

| arm | SIGReg applied to | inv term | trainable map after trunk | question it isolates |
|---|---|---|---|---|
| A (standard) | proj.out (16-d) | proj.out | Linear(384→512) + MLP(2048,2048,16) | baseline, as shipped |
| B | z.embed (512-d) | proj.out (projector kept) | Linear only, under the constraint | is ONE linear map enough buffer for a distributional constraint? |
| C | trunk CLS (384-d) | proj.out (projector kept) | none under the constraint | pure "desideratum at h", invariance still buffered |
| D (depth sweep) | output of proj depth d ∈ {0,1,2,3} | same point | d MLP layers | both losses move together — the report's head-free axis (OP-4/OP-5) |

B/C isolate SIGReg placement (single-factor); D moves the whole loss (comparable to the SimCLR
depth sweep). Arm-C h stays z.embed (the Linear remains, trained by inv-gradient only) — the
probed space is unchanged across A/B/C, which is what makes their battery rows comparable.

**Battery cells that decide it** (all E01 rows, both spaces + taps): `kurt_topeig`/worst-direction
isotropy at {CLS, z.embed, proj.out}; E04 aug-info retention at h; headline probes.
Pre-registered readings (PROPOSED):
- A vs C at h-isotropy: if C ≫ A at CLS-isotropy with equal probes, the projector was never needed
  for the *constraint* — only for the *invariance* term (sharpens the report's decision cell).
- B's signature: z.embed isotropic while CLS stays anisotropic = the Linear absorbed the
  constraint → the "buffer" for distributional pressure is as shallow as one linear map; then the
  F4 ruling (probe-side) and loss-side buffering formally dissociate — worth a THEORY_MAP note.
- If C degrades probes markedly where B does not, the trunk pays for isotropy in retained
  information (E04 ledger quantifies what was spent) — the frontier point the hypothesis predicts.

**Recipe discipline for the moved arms:** these are no longer the official recipe, so D-011's
exemptions lapse — house hygiene reapplies (grad_clip 1.0, eta_min ≤ lr/20, warmup ≥ 1 ep; needs
its own DECISIONS row before launch). λ stays fixed at 0.02 in the first pass — placement is the
only moving factor; SIGReg's slice statistics are dim-sensitive (16 vs 384/512-d, PROTOCOL §6.2),
so a λ mini-sweep is pre-authorized ONLY as separately-labeled arms if a fixed-λ arm collapses
(kill-triggers: probe < chance+5pts @ ep25, or kurt_topeig diverging).

**Toy pilot (PROPOSED, cheap):** arms A/B/C × 1 seed on the toy frame (~3 × 1 h H100) once M1
wraps, to de-risk λ-transfer and hygiene before committing H100-days at IN-100. Pilot numbers
calibrate the run plan only; frontier claims stay M4-gated.

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
