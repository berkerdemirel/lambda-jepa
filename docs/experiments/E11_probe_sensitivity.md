# E11 — Probe-protocol sensitivity map

> Derived from docs/report/ssl-projector-gap-report.html §5.3 E11 — "cheapest of all eleven
> experiments (features precomputed in E1)".

**Status:** locked for IN-100 · **Phase:** M2 (nearly free once E01 features are cached)
**Pre-registered:** ✅ 2026-07-08 (Berker, dress-rehearsal discussion). **Honesty note:** the M1
toy grid was fully visible at lock time, and it already leans on some of these predictions at toy
scale — #2 (kNN favors uniformity-trained z: simclr/vicreg z beat their h by +18.5/+15 kNN pts),
#3 partially (BYOL/DINO are the methods where the headline probe pair disagrees on h-vs-z
direction), #4 untested, #1/#5 untouched (no token/attentive/BN probes at toy). The lock therefore
binds the IN-100 rung as a *replication* test for #2/#3 and a fresh test for the rest.

## Hypothesis

Reported family rankings are probe artifacts within predictable bounds: attentive probes compress
the MAE/JEPA deficit (V-JEPA's 16–17-point precedent [E]); kNN favors uniformity-trained models;
per-space probes disagree most for methods with deep heads.

## Probe grid

{`linear_raw_v1`, `linear_house_v1`, `linear_l2_v1`, `knn_v1` (k200 headline; k20 overlay),
`attentive_v1` (token spaces only, `not-applicable` elsewhere — never a silent fallback),
`mlp2_v1` (2-layer MLP, to be added to sslgap/probes with fixed schedule at M2 start),
per-method paper probes (MAE BN→Linear, F3)} × spaces {h.cls, h.gap, best-backbone-layer
(argmax of knn_v1 over L03–L12, chosen per model), z.final} × core-7 (+anchors), IN-100 rung.
Token spaces (patch tokens, final layer, probed branch — D-005 budget) enter for attentive only.

## Deliverables (definitions fixed now, PROPOSED)

1. **Ranking-stability tensor:** Kendall τ_b between the method ranking under every (probe, space)
   cell and the headline cell (`linear_raw_v1` + `knn_v1` at the method's own h, D-006v2).
2. **Per-method protocol-sensitivity score:** max absolute rank displacement of that method across
   all cells (reported with the cell pair achieving it).
3. **Minimal stabilizing probe set:** smallest probe subset whose mean ranking (rank-average)
   reaches τ_b ≥ 0.9 against the full-grid mean ranking — the empirical basis for the four-class
   reporting standard (report §7.4).

## Pre-registered directional predictions (PROPOSED; signs only, magnitudes are new measurements)

1. **Attentive compression:** (MAE, I-JEPA) gain more from `attentive_v1` vs `linear_raw_v1` on
   token spaces than every view-based method does (per-method gain difference > 0; Holm-corrected
   sign tests within the E11 family). Direction from the V-JEPA precedent; the toy/IN-100
   magnitude is a new number — no quantitative bound pre-registered.
2. **kNN-vs-linear ordering:** uniformity-trained methods (SimCLR, DINO, LeJEPA) rank strictly
   higher under `knn_v1` than under `linear_raw_v1` relative to (BYOL, MAE, I-JEPA); test as the
   signed rank-shift of the two groups.
3. **Deep-head disagreement:** the h-vs-z ranking disagreement (Kendall distance between the
   probe-averaged h ranking and z ranking) is largest for BYOL and DINO (head depth ≥ 3
   trainable layers + bottleneck/predictor asymmetry), smallest for LeJEPA (h = z.embed sits one
   Linear from the trunk).
4. **Normalization sensitivity:** `linear_l2_v1` vs `linear_raw_v1` moves methods whose home
   loss ℓ2-normalizes (SimCLR, BYOL, DINO bottleneck) less than it moves the unnormalized-loss
   methods (VICReg, MAE, I-JEPA, LeJEPA) — feature-scale information is load-bearing only where
   the loss never removed it.
5. **MAE's paper probe (BN→Linear) closes a nonzero fraction of the MAE linear deficit** (sign
   only; the F3 ruling keeps it out of headline tables regardless).

## Estimator discipline

Probe hyperparameters are fixed by id (PROTOCOL §4) — no per-cell tuning; label-shuffle null per
cell (§6.6c); bootstrap CIs over val images; rankings compared only within identical
(N, manifest, space-dim treatment) frames. Provenance never mixed (§6.8).

## Results

*(numbers only)*

## AGREED TAKEAWAY

*(empty)*
