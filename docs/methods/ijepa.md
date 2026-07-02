# I-JEPA — dossier

> Derived from docs/report/ssl-projector-gap-report.html (compiled 2026-07-01, quote-verified there). Tags: [E] established · [P] plausible interpretation · [O] open question.

## Identity

- **Paper:** *Self-Supervised Learning from Images with a Joint-Embedding Predictive Architecture*
  — Assran et al. CVPR 2023 · arXiv:2301.08243. [E]
- **Family:** joint-embedding predictive (latent-target masked prediction); "latent prediction,
  teacher-side evaluation."

## Stated desiderata (paper's own language)

"Highly semantic image representations without relying on hand-crafted data-augmentations";
predict target-block representations from a context block. [E]

## Where the loss lives

- **Objective:** average L2 in latent space, masked targets.
- **Loss applied to:** predictor ViT output vs EMA-target-encoder patch representations.
- **Head between h and loss:** narrow ViT predictor (384-d, 6–12 blocks), conditioned on target
  positions.
- **Asymmetry machinery:** EMA target encoder (de-facto stop-grad); multi-block masking.
- **Same space as eval?** No — and eval uses the teacher branch.

## Eval space & paper protocol

h of the **EMA teacher**: "We use the target-encoder for evaluation and average pool its output."
Probe protocol: best of {last-layer avgpool, concat last-4 avgpool} — linear, not attentive
(attentive probing is V-JEPA's protocol, motivated there by "no a priori reason for the encoder to
yield a linearly separable subspace"). [E]
Architectural note (report §1): I-JEPA evaluates the EMA teacher, not the trained student — "the
backbone" is itself a temporal ensemble; any h-space audit must specify which branch (our D-008:
probed branch = teacher; both branches stored on retrains).

## Head ablations & known head facts

- Latent targets vs pixel targets: 66.9 vs 40.7 on IN-1% — the loss space's *type* (latent vs
  pixel) is worth 26 points. [E]
- Predictor depth 6→12: 64.0→66.9. [E]
- Masking design is fragile by the paper's own account: targets need "sufficiently large scale
  (semantic)" and context must be "sufficiently informative." [E]

## Documented weaknesses / failure modes

Table-4 row (° = prediction [P]):

| Expected strengths | Documented weaknesses | Predicted weak spots (§5 probes) |
|---|---|---|
| no hand-crafted augmentations; semantic latents; compute-efficient | fragile EMA anti-collapse; context-insufficiency without relief; high-influence-feature bias; weak dense/local semantics; probe-protocol sensitivity; low-shot deficit vs iBOT/DINO | masked-region ambiguity°; small objects°; multi-object scenes°; linear-vs-attentive probe deltas° |

Detail (§4.4, [E] unless noted):
- Fragile anti-collapse: C-JEPA (NeurIPS 2024) identifies "the inefficacy of EMA from I-JEPA in
  preventing entire collapse" and patches it with VICReg terms; LeJEPA's abstract reads as the
  internal verdict — JEPA practice was "ad-hoc R&D" on a heuristic stack.
- Context-insufficiency has no relief valve: the encoder "cannot adaptively modulate the type of
  predicted... features based on the feasibility of the masked prediction task" (Littwin et al.,
  SSLTP 2024) — when context underdetermines the target, the objective still demands a point
  estimate in latent space. [P] This is MAE's ambiguity problem relocated to latent space, where
  averaging over modes produces blurred *semantics* instead of blurred pixels — plausibly worse and
  definitely less visible.
- Implicit bias toward high-influence features (Littwin et al., NeurIPS 2024, deep linear models);
  the phenotype: V-JEPA 2.1 reports dense features "do not emerge reliably when the prediction loss
  is applied only to masked regions"; DMT-JEPA: "insufficient understanding of local semantics."
- Probe dependence: attentive-vs-linear gap of 16–17 points (V-JEPA); I-JEPA takes best-of-two
  pooling schemes and evaluates the EMA teacher.
- Low-shot deficit: I-JEPA ViT-H 73.3 vs iBOT ViT-L 81.0 on IN-1% — augmentation-based methods
  still dominate low-shot by ~7–8 points.
- [O] No independent autopsy exists: every published I-JEPA critique is a method paper justifying
  its fix, several from affiliated labs. A neutral layer-wise, bias-profiled, two-space audit has
  not been done — a concrete opening for this study.

## Audit expectations (this project)

Report §3.1 predicted-matrix row (h / z; predictions [P]):

| Alignment | Uniformity | Var. floor | Decorrelation | Eff. rank high | Isotropy/Gauss. | Aug.-invariance | View-predictability |
|---|---|---|---|---|---|---|---|
| ?/✓ (latent L2) | ✗/✗ | ?/? | ✗/? | ?/? (C-JEPA: fragile) | ✗/✗ | ✗ (by design: no augs) | **?/✓** |

**Own-desideratum cell (bold):** view-/latent-predictability (the L2 prediction objective;
"semantic latents"). Aug.-invariance is ✗ by design — I-JEPA trains no augmentations, so that
column is a *negative* control for it. Effective rank at h is a genuine unknown (C-JEPA's fragility
result).

## Recipe & port plan (OURS — scaffold)

- **Donor:** official `facebookresearch/ijepa` + `../ssl_explore/sslx/ijepa.py` modules.
- **Known recipe risks (kickstart plan):** fragile EMA anti-collapse → target-variance monitor
  from step 0 (do not wait for probe numbers to detect collapse).

### PORT_NOTES

*(empty — filled at port review time: donor commit, review findings, deviations)*
