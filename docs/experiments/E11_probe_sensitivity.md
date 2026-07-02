# E11 — Probe-protocol sensitivity map

> Derived from docs/report/ssl-projector-gap-report.html §5.3 E11 — "cheapest of all eleven
> experiments (features precomputed in E1)".

**Status:** draft · **Phase:** M2 (nearly free once E01 features are cached)
**Pre-registered:** ❏ pending

## Hypothesis

Reported family rankings are probe artifacts within predictable bounds: attentive probes compress
the MAE/JEPA deficit (V-JEPA's 16–17-point precedent [E]); kNN favors uniformity-trained models;
per-space probes disagree most for methods with deep heads.

## Protocol

All models × {`linear_l2_v1`, `knn_v1`, `attentive_v1` (token spaces), 2-layer-MLP probe} ×
{h.cls, h.gap, best-layer, z.final} on IN-100 (+ transfer sets at M3). Deliverables: the full
ranking-stability tensor and a per-method "protocol sensitivity" score; a minimal probe set that
stabilizes rankings — the empirical basis for the four-class reporting standard (report §7.4).

## Results

*(numbers only)*

## AGREED TAKEAWAY

*(empty)*
