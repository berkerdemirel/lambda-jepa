# E03 — Which space's metrics predict downstream?

> Derived from docs/report/ssl-projector-gap-report.html §5.1 E3.

**Status:** draft · **Phase:** M3 (needs transfer suite + both tracks)
**Pre-registered:** ❏ pending

## Hypothesis

h-space effective rank + nuisance-retention predict transfer-suite performance better than z-space
alignment/uniformity. The RankMe h↔z monotonicity holds **within** but not **across** method
families. z-space metrics look excellent on the training distribution and decouple under shift
(IN-R/A, ObjectNet targets — added when the robustness suite enters).

## Protocol

Regress each downstream score (transfer classification, low-shot, later dense/robustness) on each
metric×space, per family and pooled; rank correlations with CIs; test RankMe(z) vs RankMe(h) as
label-free selectors on the retrain grid's hyperparameter variation (the RankMe use case, now
two-space). Downstream sets at M3 minimum: IN-100 val + CIFAR-10/100 + EuroSAT (deliberately —
Guillotine shows its optimal layer differs) + IN-1% low-shot.

## Interpretation guide

Directly answers "which desiderata matter, where" — the actionable output for method designers;
feeds the four-class evaluation standard (report §7.4).

## Results

*(numbers only)*

## AGREED TAKEAWAY

*(empty)*
