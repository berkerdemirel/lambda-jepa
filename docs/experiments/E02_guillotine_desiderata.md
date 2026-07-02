# E02 — Guillotine curves for desiderata (not accuracy)

> Derived from docs/report/ssl-projector-gap-report.html §5.1 E2.

**Status:** draft · **Phase:** M2 (needs stored head-layer taps — retrained ckpts only)
**Pre-registered:** ❏ pending (locked together with E01 at G-M0; predictions below)

## Hypothesis

Desiderata are not binary between spaces but **decay along the head**; the per-metric, per-layer
decay profile predicts the accuracy-vs-layer curve of Bordes et al. (Guillotine Regularization)
*without labels*. Pre-registered directional predictions: invariance/uniformity rise monotonically
toward `z` while probe accuracy falls after layer 0–1; the crossover layer shifts toward `z` for
pretext-aligned tasks (reproduce the EuroSAT/CLEVR inversion with metric covariates, when transfer
sets enter at M3+).

## Protocol

For every model: the full battery at `h.gap`/`h.cls` per backbone layer (L03/06/09/12) and at every
head tap (`z.*.tapK`, `z.*.out`; PROTOCOL §3), overlaid with `linear_l2_v1` + `knn_v1` accuracy per
layer (the classic guillotine curve). Fit accuracy-vs-layer from metrics-vs-layer,
leave-one-method-out (report: "a label-free layer-selection rule falls out if the fit is tight" —
§2.6 open problem #2).

## Data / models

M2 grid (core-7 × 2 seeds + anchors); MAE additionally layer-resolved inside the backbone
(MIM-Refiner: best MIM features sit mid-backbone — the head boundary is fuzzy [E]).

## Interpretation guide

Tight fit → label-free readout-layer selection (extends RankMe/α-ReQ from whole-model to per-layer).
Loose fit → desiderata decay and usefulness decay are different axes; report per-family.

## Results

*(numbers only)*

## AGREED TAKEAWAY

*(empty — see WORKFLOW.md)*
