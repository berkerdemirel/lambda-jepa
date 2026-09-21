# LeVJEPA — dossier (video side project; donor registered 2026-09-02)

## Identity

LeVJEPA (Kuhn, Maes, Serra, Le Lidec, LeCun, Balestriero, Buettner; arXiv 2608.27395; code
github.com/MLO-lab/LeVJEPA, formerly galilai-group/levjepa; checkpoint
huggingface.co/galilai-group/LeVJEPA-VideoMix-Large). LeJEPA on video: one encoder, one loss —
invariance between the global view and V local views of a clip at the projector output, plus
SIGReg on each view's embeddings — no predictor, no target encoder, no stop-gradient. ViT with
per-frame tokenization (tubelet 1), 95 percent of patch tokens dropped in training, block-causal
attention over 16 frames (bidirectional within a frame, causal across frames; CLS reads all).

## Where the loss lives

Z = projector output (2048 → 256, BatchNorm) of the CLS token. H = the CLS token at the encoder
output (their "features"; the released checkpoint is used as `last_hidden_state`). Both terms of
their loss act at Z; nothing acts at H — the same placement as image LeJEPA (`docs/methods/lejepa.md`).

## Eval space & paper protocol

Frozen encoder, attentive probe (V-JEPA protocol, unmodified): one cross-attention block with a
learnable query over all output tokens + a linear classifier. ImageNet-1k (images repeated along
time), Something-Something-v2, Kinetics-400 (linear probe in the FLOP-matched table). Their numbers
for the paper's grid: ViT-S/B/L, 240 epochs on a class-balanced 20 percent subsample of K710
(union of the K400/K600/K700 training sets with validation overlap removed, following V-JEPA);
FLOP-matched ViT-B 61.0 IN-1k / 40.4 SSv2 / 44.6 K400. **Neither the subsample list nor the
attentive-probe code is in the public release** (checked at commit 3ea0dda: the repository ships
the Walking Tours download + Lance encoding pipeline, the trainer, and the model definition).

## Recipe (their released config, ViT-B on Walking Tours; "follows the ViT-L video-mixture runs")

16 frames at stride 2 from a 15 fps store (7.5 fps, ~2.1 s), 1 global view @224 (scale .8–1) +
10 local views @96 (scale .02–.4), hflip, color jitter .8 (hue .1), grayscale .2, no blur; token
drop .95; projector 2048/256 BN; SIGReg weight .02 (17 knots, 1024 projections); AdamW lr 4e-4,
wd .04, betas (.9, .95), bias/norm excluded from wd; linear warmup 1,200 steps then FLAT; effective
batch 3,072 (8 GPUs × 2 nodes × 96 × accumulation 2); bf16-mixed; EMA of the encoder (.9999 every
32 steps) saved for evaluation only — it plays no part in the loss; no gradient clipping.

## PORT_NOTES (2026-09-02; the fork is `video/levjepa/`, the pristine clone `third_party/levjepa/`)

Direction of the port is the reverse of the image donors: OUR recipe goes INTO THEIR codebase
(Berker: "keeping things faithful to the actual codebase. we update theirs"). What was ported,
in math verbatim from `sslgap/methods/_common.py::SpectralConditioner`,
`sslgap/methods/floorssl.py::training_step` (the v6 tensor-views path), `experiments/train_ddp.py`
(`_all_gather`, the share logger) and `sslgap/metrics/orbit_energy.py`:

- **The house v6 recipe** (the ImageNet-1k cells e27v6s/v6b/v6L): V = 6 global views of the clip
  through the lejepa view stack (`sslgap/data.py::orbit_stack`, FIXED AUDIT STACK v1, incl. the
  wrapped-solarize semantics), each view's parameters drawn once and applied to every frame; inv =
  view-to-mean MSE at Z on the all-pairs scale (× 2V/(V−1)); the two-sided spectral conditioner on
  the PER-CLIP VIEW CENTERS at Z (slice 128, ring q3) and at H (slice D/3: S 128 q3, B 256 q7,
  L 384 q11); doses = the ImageNet-1k chain values of the matching arch AS THEY ARE (S 31.07 /
  225.50 / 1.671; B 22.81 / 222.26 / 4.054; L 14.15 / 411.9 / 10.20), no re-dosing (Berker
  2026-09-02). No swa twin: LeVJEPA's loss has none ("if lejepa doesnt have ema twin keep it
  without"). No OAS. No multicrop.
- **Untouched:** token dropping, block-causal attention, projector, optimizer, schedule, EMA
  checkpoint weights, Lance loader and clip sampling, the multicrop path (still the default).
- **Files:** `video/levjepa/sslgap_reg.py` (conditioner, ring, autograd-aware gather, pull
  instrument, cloud energies, `ShareLogger` callback); `data/loader.py` + `LejepaViewsTransform`
  and `build_loader(views=)`; `main.py` + `sslgap_terms`/`sslgap_forward`, module wiring and the
  share logger behind `loss.type`; `conf/config.yaml` (`loss.type`, `loss.sslgap.*`,
  `augmentation.views`, `grad_clip`); `conf/sslgap_vitb.yaml` (the ViT-B recipe);
  `scripts/sslgap_smoke.py` (synthetic end-to-end smoke through spt's manual optimization —
  PASSED 2026-09-02 on CPU, vit_tiny: rings fill, clipping runs, the `[share]` line prints).
- **Declared deviations, all forced by the donor's harness:**
  1. DDP: per-clip centers are all-gathered with the autograd-aware gather (house convention);
     the random slice is drawn from a step-seeded generator shared by all ranks (SIGReg draws its
     projections per rank, harmless for an ECF average, wrong for a covariance).
  2. Gradient clipping (house hygiene, norm 1.0) is applied through `trainer.gradient_clip_val_`
     because Lightning rejects `Trainer(gradient_clip_val)` under manual optimization; the donor
     recipe clips nothing — one config key (`grad_clip`), Berker may veto.
  3. The share logger's fixed batch is drawn from the training set with a seeded no-worker
     loader at train start (house: a seeded 128-batch); the cloud energies (omega_h, omega_z,
     lam) are over the 6 views of that batch, single-process, no gather.
  4. The environment: `stable-pretraining` is a git dependency; GitHub over HTTPS fails from the
     cluster, the env was built with the SSH rewrite (memory `ista-cluster-facts`).
  5. **Loader read call (2026-09-03, measured):** the donor's `_fetch` issues one Lance `take(rows)` per
     batch. On our multi-fragment Kinetics stores (K700: 13 files × ~1 M rows) a cold `take` of one
     clip's 16 rows measured 24–210 s (0.2 rows/s), against 0.2 s for `to_table(offset, limit)` over
     the same span — the first two launches never finished their first step (30-min DDP timeout).
     `_fetch` now reads each clip's contiguous span as a range (gaps ≤ 64 rows merged) and discards
     the few extra rows; same rows, same sampling. Not visible on Walking Tours (one small store).
- **Compute note (fact, not a matching claim):** six dropped-token global views ≈ 940 tokens per
  clip vs ≈ 450 for 1 global + 10 locals; the recipe is 6 views by decision ("we dont care about
  matched compute our recipe is 6 views", Berker 2026-09-02).
- **Data (BeeGFS `/mnt/beegfs/locatgrp/shared/datasets/`):** Kinetics tarballs from the CVDF
  mirror (K400/K600/K700-2020 train + K400 val), SSv2 from Qualcomm's direct links, Walking Tours
  via yt-dlp; scripts under `scratch/video/`. K710 = union of the three train sets, duplicates
  removed by YouTube id, validation overlap removed, 20 percent per class, seed 0 (Berker
  2026-09-02). Encoding follows their Walking Tours builder (15 fps, short edge 384, JPEG q90).
- **Evaluator (written 2026-09-02 evening, unit-checked, not yet run on a real checkpoint):**
  `video/levjepa/scripts/attentive_probe.py` + `slurm/attentive_probe_in1k.slurm` — the V-JEPA frozen
  attentive probe on ImageNet-1k ported from facebookresearch/jepa (AttentiveClassifier depth 1,
  frame-repeat pre-hook, timm 'original' AutoAugment + random erasing, AdamW 1e-3 cosine, wd 1e-3 ->
  1e-6, 20 epochs, bf16). Deviation to carry with every number: global batch 128 (8 x 16) against
  V-JEPA's 1,024 at the same lr. Until it has run on a checkpoint, only within-pipeline comparisons
  are clean. K400 linear probe and the SSv2 attentive probe: still owed.
  **Port fix 2026-09-06:** the ported `CrossAttention.forward` had dropped the donor's output projection (`self.proj`, applied
  after the attention in facebookresearch/jepa; the init's 1/√2 rescale of its weight was there, the call was not). Found
  when DDP's reducer hung on the unused parameters at the first training step; fixed, verified on CPU (every parameter
  receives a gradient). The 09-04 peek (5.57 EMA / 8.91 student, reduced protocol) ran without the projection.
  Multi-node launcher `slurm/attentive_probe_in1k_8x1.slurm` (one process per node, env:// rendezvous) added the same day.
  **Audit 2026-09-06 (against the donor's `eval.py` + `vitl16_in1k.yaml`; E34 card §fairness audit):** classifier, schedules,
  augmentations and the frame-repeat input match the donor; three undeclared deviations found — (1) the donor clips the classifier's
  gradient to norm 1.0 at every step, the port does not; (2) the donor's final `Linear` keeps PyTorch's default init (outside the pooler's
  `_init_weights`), the port trunc-normal-initializes it; (3) the donor runs float16 autocast + GradScaler, the port bf16 without a
  scaler. The batch deviation stands (64 on the 4 × 1 shape, 128 on 8 × 1, vs 1,024 = 16× / 8× the head updates). The donor reads its
  EMA target encoder (`checkpoint_key: target_encoder`). Berker 2026-09-06: the probe's batch size is not a confound → D-118 withdrawn; the port stays as it is, the deviations declared here.
- **Video probes (2026-09-06, D-119 PROPOSED):** `scripts/video_probe.py` = V-JEPA's `evals/video_classification_frozen` (commit 51c59d5)
  on our clip-file stores — SSv2 attentive probe (16 × 2 × 3) and the K400 linear probe on mean-pooled tokens (their App. C); the sampling
  index logic, the RandAugment / RRC / cube-erasing train transform, the 3-crop eval transform, the segment concatenation, the schedules and
  the clip are verbatim; declared readings: frame_step in stored frames (SSv2 native 12 fps → 4; K400 15-fps store → 2), CLS kept in each
  segment's tokens, bf16, batch per GPU = what fits. Stores by `scripts/build_clipfiles.py` (SSv2 native 12 fps / 240p; K400 from the CVDF
  tarballs at 15 fps / short edge 256). Their released checkpoint (ViT-L VideoMix: 69.5 IN-1k / 55.0 SSv2 published) is the port-validation
  reference; no S / B checkpoint exists.

## Reference numbers (read 2026-09-04 from arXiv 2608.27395 v1, Figure 2, by pixel measurement of the markers; calibrated against the text's ViT-B values 50.7 IN-1k / 4.8 EFLOP and V-JEPA 2's 36.4 EFLOP, all reproduced to ±0.2)

Frozen attentive probe on IN-1k, 240 epochs on the 20 % K710 subsample, their protocol: **LeVJEPA ViT-S 39.4** (≈ 1.3
EFLOP total pretraining), ViT-B 50.7 (≈ 4.6–4.8 EFLOP), ViT-L 57.5 (≈ 15 EFLOP); V-JEPA 2 ViT-S 38.7, ViT-B 51.6,
ViT-L 55.6; VideoMAEv2 ViT-B 47.1. The ViT-S value is the like-for-like target of E34's landing (same arch, same
data protocol, same epochs, same probe); read-off uncertainty ±0.3. Caveats carried by our side: our own 20 % draw,
our probe port, global batch 128 vs their 1,024 at the same lr.
