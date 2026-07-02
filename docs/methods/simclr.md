# SimCLR — dossier

> Derived from docs/report/ssl-projector-gap-report.html (compiled 2026-07-01, quote-verified there). Tags: [E] established · [P] plausible interpretation · [O] open question.

## Identity

- **Paper:** *A Simple Framework for Contrastive Learning of Visual Representations* — Chen,
  Kornblith, Norouzi, Hinton. ICML 2020 · arXiv:2002.05709. [E]
- **Family:** contrastive (view-based, in-batch negatives). The canonical projector-ablation paper.

## Stated desiderata (paper's own language)

View-invariant representations via contrastive agreement. The projector is motivated in the
abstract itself: "a learnable nonlinear transformation between the representation and the
contrastive loss substantially improves the quality of the learned representations." [E]

## Where the loss lives

- **Objective:** NT-Xent (InfoNCE).
- **Loss applied to:** z = g(h), 128-d — "the space where contrastive loss is applied";
  "beneficial to define the contrastive loss on z's rather than h's." [E]
- **Head between h and loss:** 2-layer MLP projector.
- **Asymmetry machinery:** none (in-batch negatives).
- **Same space as eval?** No.

## Eval space & paper protocol

h = ResNet avgpool 2048-d — "we throw away the projection head g(·)"; frozen linear probe. [E]
Protocol-drift note (report §1): each method's "backbone features" definition differs; audits must
hold the readout fixed (our PROTOCOL.md probes).

## Head ablations & known head facts

- Projector ablation: nonlinear > linear (+3%) >> none (>10%); **h beats z by >10 points** of
  linear-probe accuracy — the paper's own ablation (§4.2). [E]
- Table 3 (probe recognizing the applied transformation): rotation predictable from h at **67.6%**
  vs from z at **25.6%**; original-vs-corrupted 99.5 vs 59.6. Paper's own conjecture: "z is trained
  to be invariant to data transformation. Thus, g can remove information that may be useful for the
  downstream task, such as the color or orientation of objects." [E]
- "Without a projector, SimCLR suffers from dimensional collapse in the representation space";
  a linear projector "only needs to be low-rank" (Prop. 2). DirectCLR (loss on a sub-vector of h,
  no trainable projector) beats linear-projector SimCLR (62.7 vs 61.1) but not the nonlinear head
  (66.5). (Jing et al., ICLR 2022.) [E]
- The projector is "a uniform projector, regardless of whether the uniformity part appears
  explicitly in the objective function"; backbone entropy consistently below projector entropy
  (SimCLR 1.75 vs 1.85, CIFAR-10/ResNet-18) — the encoder offloads the uniformity burden onto the
  head (Ma et al., IJCAI 2024). [E]
- RCDM: SSL backbone representations are *not* invariant to their training augmentations —
  post-projector embeddings are (Bordes et al., TMLR 2022). [E]
- Reading [P]: SimCLR contains, in 2020, the full thesis of the report in miniature — stated
  invariance desideratum, enforcement at z, evaluation at h, and quantitative proof the desideratum
  does not hold at h. What it did not do: measure alignment/uniformity/rank at h, or ask whether
  that is a bug or the mechanism.

## Documented weaknesses / failure modes

Table-4 row (° = prediction [P], not a published measurement):

| Expected strengths | Documented weaknesses | Predicted weak spots (§5 probes) |
|---|---|---|
| strong linear separability; global/low-frequency structure; calibration | texture-biased errors ≈ supervised; feature suppression; background reliance; color-information loss; adversarial fragility (uniformity-linked); crop tension on multi-object data | object-absent crops°; background-only probes°; texture-shape conflict°; small objects° |

Detail (§4.1, [E] unless noted): crop tension / object-centric dependence (Purushwalkam & Gupta;
Mishra: object-proposal crops +8.8 mAP over scene-level on OpenImages; contested by Van Gansbeke —
resolution [P]: real for small-object/multi-object regimes and co-occurrence shortcuts, not a
blanket scene-data failure). Feature suppression: "a few bits of easy-to-learn shared features can
suppress, and even fully prevent, the learning of other sets of competing features" (Chen et al.);
InfoNCE "does not always sufficiently guide which features are extracted" (Robinson et al.);
shortcuts come in multiples, "mitigating one amplifies others" (Whac-A-Mole). Background reliance
up to déjà-vu memorization (Meehan et al.: foreground recoverable from background-only training
crops, undetectable by conventional representation evals). Invariance mis-specification
(LooC/InfoMin/Ericsson: "different downstream tasks benefit from polar opposite (in)variances";
SSL "fail[s] to preserve colour information as well as supervised alternatives"). Texture bias not
fixed by SSL — error patterns "surprisingly similar" to supervised (Geirhos). Adversarial fragility
with a z-space cause: "intrinsically higher sensitivity to perturbations" attributed to hypersphere
uniformity + false negatives (Gupta et al., AAAI 2023); whether it lives in h or only z-adjacent
geometry is untested [P/O — audit cell].

## Audit expectations (this project)

Report §3.1 predicted-matrix row (cell = expectation at h / at z; ~ = partial, — = space absent in
that form; predictions [P]):

| Alignment | Uniformity | Var. floor | Decorrelation | Eff. rank high | Isotropy/Gauss. | Aug.-invariance | View-predictability |
|---|---|---|---|---|---|---|---|
| ?/✓ | **✗/✓** | ?/✓ | ✗/~ | ~/✗ (dim. collapse at z) | ✗/~ | **✗/✓ (RCDM)** | ?/— |

**Own-desideratum cells (bold):** aug.-invariance (stated) and uniformity (implicit); alignment is
the view-agreement half of the objective. The matrix predicts both hold only at z — the diagonal
the audit tests (E1).

## Recipe & port plan (OURS — scaffold)

- **Donor:** `third_party/solo-learn` @ 9187ea39 + the paper (arXiv:2002.05709).
- **Known recipe risks (kickstart plan):** SimCLR-on-ViT stability — if unstable, consider the
  MoCo-v3 frozen patch-embed trick.

### PORT_NOTES (2026-07-02, faithfulness review vs donors)

Verified vs solo-learn @9187ea3 (`solo/losses/simclr.py`, `solo/methods/simclr.py`,
`scripts/pretrain/imagenet-100/simclr.yaml`) and paper:
- Loss: their exp/pos-mask/neg-mask formulation == our CE-with-masked-diagonal for V=2 (self
  excluded from both; denominator = pos+neg) — IDENTICAL math. temp 0.2 = their IN-100 config.
- Projector: Linear→ReLU→Linear, NO BatchNorm == donor exactly. **Fixed during review:** dims
  4096→512 per their IN-100 config (ours had been 2048→256). Note: the original TF SimCLR applies
  BN inside the head; we follow our declared donor (solo-learn), recorded here.
- Augs: paper stack (RRC .08-1, flip, jitter(.8,.8,.8,.2)@.8, gray .2, blur .5) — verbatim.
- Toy deviations (D-012): house AdamW (donor: LARS lr .4 sqrt-scaled); ViT-S trunk CLS feature
  (donor RN18 avgpool — F1).
- Donor-reasoning check (2026-07-02): solo-learn's IN-100 configs are comment-free — no stated
  rationale for 4096/512 (paper default would be hidden=repr-dim, out 128). Their JMLR paper
  describes a per-method IN-100 tuning campaign without per-choice justification. Read: an
  empirically tuned IN-100/RN18 value, not a principled constant.
