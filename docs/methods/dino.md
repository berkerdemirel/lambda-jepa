# DINO — dossier

> Derived from docs/report/ssl-projector-gap-report.html (compiled 2026-07-01, quote-verified there). Tags: [E] established · [P] plausible interpretation · [O] open question.

## Identity

- **Paper:** *Emerging Properties in Self-Supervised Vision Transformers* — Caron et al.
  ICCV 2021 · arXiv:2104.14294. [E]
- **Family:** self-distillation with a categorical prototype loss space (clustering-adjacent);
  introduces the multi-crop + centering/sharpening stack this project must audit. Siblings iBOT /
  DINOv2 (methods/later/) extend the same head design.

## Stated desiderata (paper's own language)

Self-distillation with no labels; collapse controlled by centering — which "prevents one dimension
to dominate but encourages collapse to the uniform distribution" — balanced against sharpening. [E]

## Where the loss lives

- **Objective:** cross-entropy between teacher and student distributions.
- **Loss applied to:** K = 65 536-way softmax on the head output — a categorical loss space, "the
  furthest of any family from the probed geometry." [E]
- **Head between h and loss:** 3-layer MLP (2048) → ℓ2 bottleneck 256 → weight-normed K prototypes.
- **Asymmetry machinery:** stop-grad + EMA teacher; centering + sharpening; multi-crop
  (2 global + V local 96px crops).
- **Same space as eval?** No.

## Eval space & paper protocol

h = [CLS] — "features used in downstream tasks are the backbone f output." Repo probes
concat-of-last-4-[CLS] for ViT-S, [CLS]+avgpooled patches for ViT-B — even "the backbone
representation" is a protocol choice, not a single object. [E] Strong k-NN at h (78.3) — emergent,
not enforced (Table 2). [E]

## Head ablations & known head facts

- K matters little above 4096: 67.8 k-NN @ K=1k → 69.7 @ K=65k. [E]
- The ℓ2 bottleneck is what prevents deep-head collapse — without it: **0.1**. [E]
- (Family contrast, DINOv2:) KoLeo is applied to the [CLS] batch — one step closer to h than a
  prototype loss — and visibly works (+0.5 IN-1k, +8 retrieval): the surveyed field's closest
  existing move toward "impose the desideratum nearer h." Even it is not applied at h directly. [E]
- Reading [P]: this family's desiderata (uniform prototype usage, centering) are constraints on a
  categorical distribution over discarded prototypes — the loosest possible coupling to backbone
  geometry, yet it produces the best frozen features of the pre-2025 era.

## Documented weaknesses / failure modes

Table-4 row (° = prediction [P]):

| Expected strengths | Documented weaknesses | Predicted weak spots (§5 probes) |
|---|---|---|
| excellent k-NN/frozen features; emergent segmentation-ish attention | same cluster prior as SwAV; centering/sharpening delicacy; local-to-global forces context inference | context-swap°; foreground-background conflict°; crop-semantics mismatch° |

Detail (§4.2): hidden uniform cluster prior (Assran et al., ICLR 2023) — features that "enable
uniform clustering"; "can hamper performance when pretraining on class-imbalanced data" [E].
Multi-crop as a failure surface: local 96px crops (<50% area) train "local-to-global
correspondences" — semantically, the model is required to agree about an image it mostly cannot
see; on scene data local views may contain a different object than the global view [E]. First
dedicated task-conflict analysis only appeared Feb 2026 (De Plaen et al.: per-view-type predictors
+3.8–4% — the shared head is forced to solve heterogeneous alignment tasks) [E]. Object-absent
local views remain unstudied [O]. Prototype pathologies (dead/duplicated prototypes, long-tail
prototype-class mismatch, what prototype layers encode at scale): no dedicated failure study
found [O].

## Audit expectations (this project)

Report §3.1 predicted-matrix row (h / z; predictions [P]):

| Alignment | Uniformity | Var. floor | Decorrelation | Eff. rank high | Isotropy/Gauss. | Aug.-invariance | View-predictability |
|---|---|---|---|---|---|---|---|
| **?/✓** | **✗/~ (centering)** | ?/? | ✗/? | ~/? | ✗/✗ | ✗/✓ | ?/— |

**Own-desideratum cells (bold):** teacher–student agreement (alignment, enforced in prototype/code
space) and non-collapse via centering↔sharpening (uniformity + collapse-margin cells). Note the
many `?` cells — DINO's z is the least-characterized of the core-7 because it is categorical
(our z taps per D-008: 256-d bottleneck pre-prototypes as z.final; prototype logits recomputed).

## Recipe & port plan (OURS — scaffold)

- **Donor:** `../ssl_explore/experiments/train_dinov2.py` — restructure into the sslgap trainer
  (in-house code, already validated on the IN-100 control checkpoints).
- **Known recipe risks (kickstart plan):** teacher-temperature ramp + EMA-rate incidents known
  in-house — watch both from the first smoke run.

### PORT_NOTES (2026-07-02, faithfulness review vs OFFICIAL repo)

Verified vs facebookresearch/dino @7c446df (`main_dino.py:363-417`, `utils.py:144-149`):
- Loss: teacher softmax((t−center)/temp) × student log-softmax over all teacher-global×student-view
  pairs EXCLUDING same-view, divided by n_terms — our pair set and mean are IDENTICAL (14 terms for
  2g+6l).
- Center update: center·m + batch_mean(teacher logits over both globals)·(1−m), applied after the
  loss uses the OLD center — identical formula and sequencing (ours in post_step).
- Constant teacher temp 0.04 is the OFFICIAL DEFAULT (main_dino arg), not a deviation.
- Prototype freeze: official zeroes last_layer grads for epoch<1; ours toggles requires_grad —
  equivalent. norm_last_layer=True g-freeze == official ViT-S setting.
- Teachers kept in EVAL mode (drop_path off in targets) — **fixed during review** (the frame loop
  had blanket .train(); sslx control also kept teachers eval).
- Multi-crop views: global (0.4,1)@128 blur 1.0/0.1 solarize 0/0.2, local (0.05,0.4)@64 blur 0.5 —
  Lightly/DINO-faithful (port of the control's `_dino_view`).
- Toy deviations (D-012 + sslx incident ledger): K=4096, EMA base 0.99, house AdamW (official: lr
  scaled, wd cosine 0.04→0.4, no-decay-on-norm/bias — enters at M2 with the control recipe).
