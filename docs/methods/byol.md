# BYOL — dossier

> Derived from docs/report/ssl-projector-gap-report.html (compiled 2026-07-01, quote-verified there). Tags: [E] established · [P] plausible interpretation · [O] open question.

## Identity

- **Paper:** *Bootstrap Your Own Latent* — Grill et al. NeurIPS 2020 · arXiv:2006.07733. [E]
- **Family:** negative-free self-distillation (predictor + EMA teacher). Collapse-avoidance as head
  machinery.

## Stated desiderata (paper's own language)

View-invariant prediction without negatives; non-collapse: "BYOL does not converge to such
[collapsed] solutions" — attributed to predictor + EMA "encouraging encoding more and more
information within the online projection." [E]

## Where the loss lives

- **Objective:** MSE on normalized vectors.
- **Loss applied to:** predictor output q(g(h)) vs EMA-teacher projection g′(h′), 256-d. [E]
- **Head between h and loss:** projector (4096→256) + predictor (same architecture).
- **Asymmetry machinery:** stop-gradient on target + EMA teacher (τ 0.996→1).
- **Same space as eval?** No.

## Eval space & paper protocol

h = avgpool 2048-d — "At the end of training, we only keep the encoder f." [E] Frozen linear probe.

## Head ablations & known head facts

- No predictor + no negatives: **0.3%** (total collapse). Random-constant target: 18.8%.
  τ_base=0.99: 72.5 (vs SimCLR 69.4). "The additional predictor is critical to prevent
  collapse." [E]
- Predictor necessity is BYOL-specific, not universal: MoCo v3 without predictor loses only 1 pt
  (75.5 vs 76.5) — "MoCo as a contrastive method does not need the predictor MLP to work." [E]
- DirectPred: the linear predictor's eigenspace provably aligns with its input correlation;
  replacing the trained predictor with a closed-form spectral function matches BYOL (72.4 vs 72.5);
  removing stop-gradient provably collapses (Thm 2). (Tian, Chen, Ganguli, ICML 2021.) [E]
- BYOL works without batch statistics (73.9 vs 74.3), killing the BN-as-negatives folk theory
  (Richemond et al.); [P] the field's understanding of its own anti-collapse machinery has been
  empirically wrong at least twice (also Zhang et al. refuting parts of SimSiam's self-explanation).
- Optimal linear predictor ≈ orthogonal projection; EMA + stop-grad act as "an efficient
  orthonormalization mechanism" (Richemond et al., ICML 2023). [E]
- Rank Differential Mechanism: every asymmetric device (predictor, EMA, …) is a spectral filter
  creating a consistent rank difference between branch outputs, provably raising effective
  dimensionality — explicitly *not* connected to backbone rank (named open edge). (Zhuo et al.,
  ICLR 2023.) [E]
- Gap-aware exception: Wen & Li (NeurIPS 2022) prove a theorem about the encoder *beneath* the
  head — substitution + acceleration effects make the encoder "learn all the features rather than
  focus only on learning the stronger features"; the head is a training device. [E]
- Reading [P]: BYOL's non-collapse desideratum is not a property its loss enforces on any space —
  it is an emergent property of head-side optimization dynamics. Whether the backbone's spectrum
  benefits or merely survives is exactly the h-vs-z rank question (§3 battery).

## Documented weaknesses / failure modes

Table-4 row (° = prediction [P]):

| Expected strengths | Documented weaknesses | Predicted weak spots (§5 probes) |
|---|---|---|
| as SimCLR + robust without negatives; less false-negative noise | anti-collapse is fragile machinery (predictor-dependent); same augmentation priors as SimCLR | same as SimCLR°; predictor-ablated variants collapse (E10) |

Inherits the §4.1 view-based failure surface (crop tension, feature suppression, background
reliance, invariance mis-specification, texture bias) — see simclr.md for the shared detail.

## Audit expectations (this project)

Report §3.1 predicted-matrix row (h / z; predictions [P]):

| Alignment | Uniformity | Var. floor | Decorrelation | Eff. rank high | Isotropy/Gauss. | Aug.-invariance | View-predictability |
|---|---|---|---|---|---|---|---|
| **?/✓** | ✗/~ | ?/~ | ✗/~ | **~/~ (rank differential)** | ✗/✗ | ✗/✓ | **?/✓** |

**Own-desideratum cells (bold):** alignment + view-predictability (predict the other view) and
non-collapse (eff.-rank / collapse-margin cells — a head-dynamics fact, not an enforced h-property).

## Recipe & port plan (OURS — scaffold)

- **Donor:** `third_party/solo-learn` @ 9187ea39 + the paper (arXiv:2006.07733).
- **Known recipe risks (kickstart plan):** EMA base rate must scale with steps/epoch (τ schedule is
  defined against the training-step count); collapse canary = the 0.3% failure mode — collapse
  monitors on from step 0.

### PORT_NOTES (2026-07-02, faithfulness review vs donors)

Verified vs solo-learn @9187ea3 (`solo/losses/byol.py`, `solo/methods/byol.py`, IN-100 yaml):
- Loss: 2−2·cos per direction, symmetrized sum — IDENTICAL (their simplified path).
- Proj/pred: Linear→BN→ReLU→Linear both — IDENTICAL structure. **Fixed during review:** pred hidden
  8192 per their IN-100 config (ours had shared 4096); proj 4096→256 confirmed.
- Teacher BN in train mode (batch stats), matching donor/paper. Stop-grad via no_grad ✓.
- Deliberate deviations: EMA base 0.99 vs paper 0.996 (37 steps/ep — house incident ledger,
  documented in the yaml); house AdamW (donor: LARS). Collapse canary: per-step teacher-proj std.
