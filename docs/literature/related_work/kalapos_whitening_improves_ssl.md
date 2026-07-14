# Close-read — Kalapos & Gyires-Tóth, "Whitening Consistently Improves Self-Supervised Learning"

**arXiv:2408.07519** (cs.CV, Aug 2024) · repo `kaland313/SSL-Whitening` · tag `[P]`
**Verdict (Berker 2026-07-14): a good, honest nearest-neighbor — NOT a blocker.** It is the closest
prior work to our *calibration-toward-h* premise, but it does not touch our contribution (the
two-space audit + the trunk-relocation mechanism + the soft-diagonal-floor characterization +
placement). Cite it as the precedent for "calibrating/whitening h helps, method-agnostically"; drop
any "first to show" language about that premise.

> Read provenance: abstract + ar5iv full text + repo README (small-model-summarized). The method
> shape is unambiguous; **do a human PDF pass on the exact numbers before camera-ready.**

## What they do
A **differentiable ZCA-whitening layer** (IterNorm — Newton-iteration, per-minibatch; repo depends on
`huangleiBuaa/IterNorm`) appended as the **final encoder layer**, active **during pretraining on both
branches** and retained at evaluation. It is part of the encoder forward pass — **not** post-hoc, and
**not** a loss term (the abstract's "pretrained encoders" is loose wording; IterNorm is intrinsically
in-training). Applied to the *encoder output* h (they explicitly compute their metrics "directly on
encoder features … more informative … directly measures the encoders' properties").

- **Claim:** whitening the encoder output improves probing **method-agnostically, +1–5% linear/kNN**.
- **Coverage:** BYOL, SimCLR, VICReg, SwAV, Barlow Twins (± DINO); ResNet-18 + ConvNeXtV2-Pico;
  CIFAR-10 / STL-10 / TinyImageNet. Small scale (no ImageNet, no ViT-S/16@224).
- **Proposed metrics:** mean |off-diagonal correlation|; mean feature std; anisotropy σ₁²/Σσᵢ².
- **Honest caveats they report (useful to us):** NOT universal — VICReg/ConvNeXt −2.96%, supervised
  slightly negative, DINO/STL10 mixed — but it *rescues a collapsed SwAV run* (+63.9%).

## How it differs from our moment-KL floor at h
| | Kalapos (2408.07519) | sslgap calibration-toward-h |
|---|---|---|
| **Intervention** | *hard* whitening **layer** in the forward pass (train + eval) | *soft* **loss term** (moment-KL); trained encoder read **as-is**, no added layer |
| **Constrains** | **full covariance** (off-diagonals decorrelated) | **diagonal moments** only — `diag_read = ½(⟨μ²⟩+⟨var−1−log var⟩)`; decorrelation is a *byproduct* (E12-T8), not the target |
| **Question** | "does whitening the encoder improve probing?" (+ 3 metrics) | **two-space audit**: *where* is each desideratum satisfied (h vs z), and what does calibrating h *do* to both spaces |
| **Placement** | fixed at "encoder output" | studied explicitly (GAP / CLS / projector-input — D-036 + the H-wave placement matrix) |
| **Frame** | CIFAR/STL/TinyIN, RN-18/ConvNeXt-Pico | IN-100, ViT-S/16@224, matched controlled retrains |

Their three metrics are a **strict subset** of our battery. Our WORKFLOW.md lesson is precisely that
anisotropy/sliced stats *alone* are foolable (decorrelation + CLT), which is why we lead with
`kurt_topeig` / worst-direction alongside Epps–Pulley. So their diagnostics don't threaten ours;
ours contain them and are hardened against the failure mode their anisotropy number is exposed to.

## Contribution framing (agreed)
- **Preempted (do not claim):** the bare result "decorrelating/whitening the representation h improves
  SSL probing, method-agnostically" — theirs, in a *stronger* hard-ZCA form.
- **Ours (unaffected):** (1) the two-space audit; (2) the mechanism that h-calibration *relocates
  conditioning/invariance into the trunk and is invisible at z* (E12-T8) — they never look at z;
  (3) the soft **diagonal** moment-floor as a *conditioner with an interior optimum, not a satisfied
  constraint* (E12-T2/T7) — a different object than a hard whitening layer; (4) the placement study;
  (5) the view-invariance assist at h (H-wave).
- **Their mixed results are a selling point for us**, not a threat: our interior-optimum /
  tax-in-some-columns finding is exactly the account of *when and why* whitening-at-h helps vs hurts
  that they lack (VICReg −2.96%, supervised −, DINO mixed).

## Follow-ups if we lean on this
- Confirm exact ZCA regularization (IterNorm ε / iteration count) from their code before contrasting
  our floor's soft equilibrium against their hard transform.
- Their W-MSE lineage (Ermolov 2007.06346, already in BIBLIOGRAPHY) is the *loss-side* whitening
  precedent; Kalapos is the *architectural-layer* precedent — both sit upstream of our "soft floor at
  the declared h" and neither runs the two-space audit.
