# VICReg — dossier

> Derived from docs/report/ssl-projector-gap-report.html (compiled 2026-07-01, quote-verified there). Tags: [E] established · [P] plausible interpretation · [O] open question.

## Identity

- **Paper:** *VICReg: Variance-Invariance-Covariance Regularization* — Bardes, Ponce, LeCun.
  ICLR 2022 · arXiv:2105.04906. [E]
- **Family:** redundancy reduction / covariance regularization (with Barlow Twins): exact
  satisfaction at z, directional transfer to h.

## Stated desiderata (paper's own language)

Maintain per-dimension variance ≥ γ; decorrelate dimensions; align views — explicitly "avoid the
collapse problem." [E]

## Where the loss lives

- **Objective:** variance + invariance + covariance terms (batch statistics).
- **Loss applied to:** z = 8192-d expander output — "The loss is computed at the embedding level
  on z and z′." [E]
- **Head between h and loss:** 3-layer expander (8192).
- **Asymmetry machinery:** none (branches may differ).
- **Same space as eval?** No.

## Eval space & paper protocol

h = avgpool 2048-d — "the expander is discarded." [E] Frozen linear probe.

## Head ablations & known head facts

- The width curve: 55.9% at 256-d expander → 68.8% at 16 384-d — **while the probed backbone is
  fixed at 2048-d throughout**. (Barlow Twins analogue: "keeps improving with all output
  dimensionality tested," to 16 384, no saturation.) [E]
- The transfer statement — App. D.8: the std target is met **exactly** on the embeddings and
  "translates in an increased standard deviation at the representation level" — the literature's
  most explicit partial-transfer claim. [E]
- Expander role, in the authors' words: eliminate view-difference information, and expand dimension
  "so that decorrelating the embedding variables will reduce the dependencies (not just the
  correlations) between the variables of the representation." [E]
- **No ablation regularizes h directly** (verified absence). [E]
- Shwartz-Ziv et al. (NeurIPS 2023) information bounds certify entropy/decorrelation of the
  regularized embedding — properties h is never forced to have (§6 map). [E]
- Reading [P→O]: the redundancy-reduction family argues its desideratum transfers h-ward but
  demonstrates it only qualitatively — how much decorrelation/variance survives at h, as a function
  of expander width/depth, is open and directly measurable. [O]

## Documented weaknesses / failure modes

Table-4 row (shared with Barlow Twins; ° = prediction [P]):

| Expected strengths | Documented weaknesses | Predicted weak spots (§5 probes) |
|---|---|---|
| no asymmetry machinery; wide-head decorrelation; stable | inherit contrastive failure modes (contrastive/non-contrastive duality); decorrelation unverified at h; uniform prior (VICReg named by Assran et al.) | decorrelation audit at h°; imbalanced pretraining°; texture-shape° |

Detail: the hidden uniform cluster prior (Assran et al., ICLR 2023) names VICReg among methods that
"learn features that enable uniform clustering" — excellent on balanced ImageNet, "can hamper
performance when pretraining on class-imbalanced data" [E]; a z-space equipartition-like constraint
acting as a dataset-level assumption. Decorrelation at h was never measured in the original
papers [O].

## Audit expectations (this project)

Report §3.1 predicted-matrix row (h / z; predictions [P]):

| Alignment | Uniformity | Var. floor | Decorrelation | Eff. rank high | Isotropy/Gauss. | Aug.-invariance | View-predictability |
|---|---|---|---|---|---|---|---|
| **?/✓** | ✗/~ | **~/✓ (D.8: directional)** | **✗/✓** | ~/✓ | ✗/~ | ✗/✓ | ?/— |

**Own-desideratum cells (bold):** variance floor (the one explicit partial-transfer claim in the
literature), decorrelation (never measured at h — prediction: substantial residual correlation),
and alignment (the invariance term).

## Recipe & port plan (OURS — scaffold)

- **Donor:** `third_party/solo-learn` @ 9187ea39 + the paper (arXiv:2105.04906).
- **Known recipe risks (kickstart plan):** none singled out — general training-hygiene rules
  (WORKFLOW.md) apply.

### PORT_NOTES

*(empty — filled at port review time: donor commit, review findings, deviations)*
