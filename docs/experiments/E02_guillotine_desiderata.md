# E02 — Guillotine curves for desiderata (not accuracy)

> Derived from docs/report/ssl-projector-gap-report.html §5.1 E2.

**Status:** locked for IN-100 · **Phase:** M2 (needs stored head-layer taps — retrained ckpts only)
**Pre-registered:** ✅ 2026-07-08 (Berker, dress-rehearsal discussion) — with the per-probe scoring
qualifier below, added after the M1 toy grid showed the two headline probes moving in OPPOSITE
directions along MAE's decoder (linear_raw .669→.693 up, knn_v1 .528→.450 down): an unqualified
"accuracy" monotonicity prediction is unscoreable. Toy M1 numbers (including every tap probe)
were visible when this lock was signed; the lock binds the IN-100 rung.

## Hypothesis

Desiderata are not binary between spaces but **decay along the head**; the per-metric, per-layer
decay profile predicts the accuracy-vs-layer curve of Bordes et al. (Guillotine Regularization)
*without labels* (OP-1 quantitative transfer law; OP-2 readout-layer selection).

## The depth axis

Per model, ordered probe points `ℓ = 0..L`:

```
h.{cls,gap}.L03 → L06 → L09 → L12(=h final) → z taps in forward order (per PROTOCOL §3):
  simclr/vicreg:  proj.tap1 [→ proj.tap2] → proj.out
  byol:           proj.tap1 → proj.out → pred.tap1 → pred.out          (student)
  dino:           dino.tap1 → dino.tap2 → dino.bottleneck [→ proto logits, recomputed]
  mae:            dec.tap2 → dec.tap5 → dec.tap8                        (decoder depth axis)
  ijepa:          pred.out                                              (one z point only)
  lejepa:         z.embed(=h, F4) → proj.tap1 → proj.tap2 → proj.out
```

The feature type at trunk layers follows the method's h feature type (F1/F2: gap for
simclr/vicreg/byol/mae/ijepa, cls for dino/lejepa); the other type is kept as a sensitivity
overlay, not the headline curve.

## Pre-registered curve-shape predictions (PROPOSED)

Monotonicity is scored as Spearman ρ(metric, depth index) over the ordered points; "monotone"
pre-declared as |ρ| ≥ 0.8 with the stated sign. Each method's OWN desideratum (AUDIT_MATRIX bold
cell) is the headline curve; families are Holm–Bonferroni-corrected per method (PROTOCOL §6.7).

**Per-probe scoring rule (qualifier added at lock, 2026-07-08):** every accuracy-curve prediction
is scored SEPARATELY for `knn_v1` and `linear_raw_v1` — the toy grid showed the pair can disagree
on direction (MAE decoder). Where the table below says "peak"/"↓ monotone" without naming a probe,
the prediction binds `knn_v1`; `linear_raw_v1` carries the same sign as a soft prediction EXCEPT
MAE's decoder segment, where linear_raw is left as an open measurement (toy showed it rising).
Pair metrics along the depth axis (alignment/invariance) are read jointly with uniformity and the
per-space `pair_margin` baseline (METRICS.md coupling caveat) — raw pair distances alone are not
scored.

| method | own-desideratum curve (prediction) | accuracy curve (knn_v1 + linear_raw_v1) |
|---|---|---|
| SimCLR | alignment & uniformity ↑ monotone toward proj.out | peak at L12 or tap1, strictly below peak at proj.out |
| VICReg | variance-floor satisfaction & decorrelation ↑ monotone toward proj.out | same shape as SimCLR |
| BYOL | teacher-alignment ↑ monotone toward pred.out; uniformity FLAT/weak (no repulsion term — divergence from the contrastive family is itself the prediction) | peak at L12 or proj.tap1 |
| DINO | per-sample prototype-entropy ↓ / bottleneck uniformity ~ (centering only, no KoLeo); sharpest change at the bottleneck (ℓ2 + dim 256 squeeze) | peak at L12 (teacher cls), falls through head |
| MAE | probe accuracy ↓ monotone through dec.tap2→8 (pixel specialization); eff-rank at dec taps ? (no prediction — new measurement) | peak INSIDE the backbone (L06–L12, MIM-Refiner precedent) — the one method whose backbone segment is predicted non-monotone |
| I-JEPA | with one z point, no curve claim at z; backbone segment: teacher-gap accuracy ↑ to L12, pred.out below L12 | peak at L12 (teacher) |
| LeJEPA | isotropy (kurt_topeig-led per §6.4) ↑ monotone across the FULL axis incl. z.embed→proj.out; where isotropy first "arrives" is the study's central unknown — no prediction for the h endpoint (AUDIT_MATRIX `?/✓`) | peak at z.embed (=h) by construction of the recipe's own probes |

Cross-method prediction: the desideratum-satisfaction crossover (first depth index where the
normalized own-desideratum curve exceeds 0.9 of its z-endpoint value) sits DEEPER than the
accuracy peak for every method — the region between them is the "buffer zone" whose width is the
per-method transfer-law summary (OP-1). Predicted widest for deep-head methods (BYOL: proj+pred;
DINO: 3-layer+bottleneck), narrowest for LeJEPA (distributional constraint, report §6 bet).

## The OP-2 fit (label-free layer selection)

Per (model, layer): x = battery metrics standardized within model across layers (dim-sensitive
metrics on PCA-64 per §6.2, raw-d kept as sensitivity); y = knn_v1 accuracy at that layer.
Leave-one-method-out linear fit (7 folds). Pre-declared readout (PROPOSED): LOMO R² ≥ 0.5 counts
as "tight" (→ a label-free layer-selection rule exists); R² < 0.2 counts as "the two decays are
different axes"; between = report per-family fits. Also report the rank version: does argmax of
the fitted curve select the true best layer within 1 position, per held-out method?

## Estimator discipline notes

- N fixed (m50k manifest) at every layer; layers differ only in d → §6.2 PCA-64 rows are the
  comparable ones for dim-sensitive metrics (taps span 16–8192 dims).
- DINO prototype logits recomputed from the bottleneck (D-005) enter as an optional final depth
  point, flagged (65k-d, PCA-64 only).
- Pairs metrics (alignment/invariance) use the fixed audit stack at every layer (self-consistency
  stack as overlay), so the depth axis is the only moving part.
- Nulls: the random-init trunk's own layer curve is subtracted as the zero-reference for every
  monotonicity claim (an untrained ViT already has depth trends — the M0 null showed nonflat
  spectra; predictions above are about the TRAINED−NULL residual curve).

## Data / models

M2 grid (core-7 × 2 seeds + supervised DeiT-lite + random-init anchors). Toy-rung dress rehearsal
runs the identical pipeline on the M1 grid (h_layers=[3,6,9,12] already in the extraction) — toy
numbers inform protocol bugs only, never lock interpretation (D-012 pending).

## Interpretation guide

Tight fit → label-free readout-layer selection (extends RankMe/α-ReQ from whole-model to
per-layer). Loose fit → desiderata decay and usefulness decay are different axes; report
per-family. Either resolution feeds OP-1/OP-2; the buffer-zone widths feed E10's frontier
predictions.

## Results

*(numbers only)*

## AGREED TAKEAWAY

*(empty — see WORKFLOW.md)*
