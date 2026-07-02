# MAE — dossier

> Derived from docs/report/ssl-projector-gap-report.html (compiled 2026-07-01, quote-verified there). Tags: [E] established · [P] plausible interpretation · [O] open question.

## Identity

- **Paper:** *Masked Autoencoders Are Scalable Vision Learners* — He, Chen, Xie, Li, Dollár,
  Girshick. CVPR 2022 · arXiv:2111.06377. [E]
- **Family:** masked image modeling (pixel reconstruction). "The decoder is the projector, pixels
  are the loss space."

## Stated desiderata (paper's own language)

Predictive sufficiency via masked reconstruction: "mask random patches... and reconstruct the
missing pixels"; scalability. [E]

## Where the loss lives

- **Objective:** MSE on per-patch-normalized pixels, masked patches only.
- **Loss applied to:** pixel space, through an 8-block, 512-d decoder.
- **Head between h and loss:** the full ViT decoder — no teacher, no negatives, no projector; the
  decoder *is* the head.
- **Asymmetry machinery:** none; 75% masking; encoder sees visible patches only.
- **Same space as eval?** No.

## Eval space & paper protocol

h = encoder on the **uncorrupted** image (train/test input mismatch by design); avgpool or [CLS];
the linear probe adds a non-affine BN before the classifier. [E]

## Head ablations & known head facts

- Decoder depth is a linear-probe dial: 65.5% (1 block) → 73.5% (8 blocks) while fine-tuning stays
  flat at ~84.9 — "a sufficiently deep decoder is important for linear probing... can yield up to
  8% improvement." [E]
- Linear and fine-tune results are "largely uncorrelated"; MAE features are "less linearly
  separable, but stronger non-linear features." [E]
- MIM-Refiner (ICLR 2025): "strong representations within MIM models generally reside in
  intermediate layers" — a within-backbone guillotine effect: the last blocks specialize toward
  reconstruction, i.e. the head boundary is fuzzy and partially *inside* the encoder. [E]
  → [P] any h-space audit of MAE must be layer-resolved, not last-layer-only.
- U-MAE (Zhang et al., NeurIPS 2022): MAE features **dimensionally collapse** without an added
  uniformity term — the anti-collapse desideratum, absent from MAE's design, turns out to bind
  at h. Their theory pulls the pixel loss back to encoder-level alignment of mask-induced positive
  pairs (the reverse-direction bridge; see THEORY_MAP). [E]
- Reading [P]: MAE inverts the joint-embedding story — instead of the head absorbing invariance,
  the decoder absorbs low-level pixel statistics so the encoder can stay abstract; same mechanism,
  opposite direction. How much pixel-level nuisance still leaks into h is measurable (E4, E9).

## Documented weaknesses / failure modes

Table-4 row (° = prediction [P]):

| Expected strengths | Documented weaknesses | Predicted weak spots (§5 probes) |
|---|---|---|
| fine-tuning king; dense/geometric tasks; locality; no augmentation priors | weak linear separability; pixel-variance shortcut; texture/high-freq reliance; late-layer degradation; collapse without uniformity (U-MAE) | background-only° (should pass object-absence better°); masked-ambiguity probes°; low-shot semantics° |

Detail (§4.3, [E] unless noted): linear-separability deficit by the paper's own account (ViT-B
68.0 / L 75.8 / H 76.6 trail MoCo v3; one fine-tuned block: 73.5→81.0). Pixel-variance shortcut,
formalized: reconstruction "allocates a model's capacity towards a subspace of the data explaining
the observed variance — a subspace with uninformative features"; on TinyImageNet the top-variance
subspace (90% of pixel variance) supports only 45% accuracy vs 55% for the bottom 20%; perception
features are "learned last," explaining the long schedules (Balestriero & LeCun, ICML 2024).
Frequency/attention signature: MIM uses high-frequency (texture-adjacent) signals vs contrastive
low-frequency/global shape; MIM keeps locality and head diversity (Park et al.; Xie et al.) — the
families' h-spaces should look measurably different. Masked-region ambiguity is ill-posed — "we can
guess that there is a tail, but we cannot determine its exact location" (Bar et al.); StoP fixes
location ambiguity only; content multimodality unquantified [O → E9].

## Audit expectations (this project)

Report §3.1 predicted-matrix row (h / z; — = space doesn't exist for MAE in that form: its z is
pixel logits; predictions [P]):

| Alignment | Uniformity | Var. floor | Decorrelation | Eff. rank high | Isotropy/Gauss. | Aug.-invariance | View-predictability |
|---|---|---|---|---|---|---|---|
| ✗/— (pixel loss) | ✗/— | ?/— | ✗/— | ✗ (U-MAE: collapse-prone)/— | ✗/— | ✗ (none trained)/— | ?/— |

**Own desideratum:** predictive sufficiency ("reconstruct everything") is not a geometry column —
it is operationalized by the information-retention ledger (E4), where MAE is the predicted
max-retention anchor (nuisance decodability at h: MAE ≫ contrastive), and by E9. z taps per D-008
are decoder-block taps; guillotine curves run through the decoder.

## Recipe & port plan (OURS — scaffold)

- **Donor:** solo-learn / MMSelfSup donor — **no paper ViT-S recipe exists**; recipe must be
  assembled from the donors and flagged as such in MODELS.md provenance.
- **Known recipe risks (kickstart plan):** weak linear probe is EXPECTED (not a failed-port
  signal) — validate the port via kNN and fine-tune-style signals instead.

### PORT_NOTES

*(empty — filled at port review time: donor commit, review findings, deviations)*
