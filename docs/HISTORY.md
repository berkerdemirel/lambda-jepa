# HISTORY — append-only findings & lessons log
(append entries; never rewrite. Full result tables live in results/*/FINDINGS.md)

## 2026-07-02 — project kickstart
- Repo scaffolded from the approved kickstart plan (docs/report + plan in .claude/plans).
- Verified: ~/data/imagenet100 (CMC) vs ~/data/imagenet-100 share only 8/100 classes — different datasets. Canonical: imagenet100 (D-002).
- Verified: lejepa/sslx "emb" = trunk(384) + Linear(384→512) (timm num_classes=512) — h redefined as trunk forward_features; the Linear is head tap z.embed (D-003).
- solo-learn zoo checkpoint downloads discontinued as of 2026 → controlled retrains are the only route to head-preserving apples-to-apples checkpoints.
- Donor pinned: third_party/solo-learn @ 9187ea39c2c3f43c455ab06664d2b019a9802954 (2026-04-22).

## 2026-07-02 — M0 parity hunt: the reference protocol was not what its label said
- First parity attempt vs CAMPAIGN_LOG "DINO ep100 (4k)" failed by 8–13 pts (our linspace-4k probe).
- Root cause (verified in ssl_explore code + diag-dino job logs): the DINO rows came from
  `diag_dino.py`, which probes the **teacher** with **max_imgs=30000** evenly-spaced probe-train
  images (the "@4k" in the table refers to the LOCAL branch's n_train, not the GLOBAL probe);
  RankMe/eff-rank there are computed on **val** features (N=5000).
- Reference numbers (teacher): ep100 GAP lin 0.6442 / kNN 0.4662 / RankMe(val) 217.0 / effrank(val)
  39.5; CLS lin 0.6856 / kNN 0.6136 / RankMe 236.9 / effrank 72.2. ep50 GAP 0.6014/0.4188, CLS
  0.6550/0.5696. ep25 GAP 0.4912/0.3532, CLS 0.5234/0.4386.
- Lesson (E11's thesis, live): a probe protocol is only defined by its code, not its table label.
  Every sslgap probe row carries manifest name + N; parity rows must name the exact selection rule.

## 2026-07-02 — M0 parity PASSED (exact) once the true reference protocol was used
teacher branch, linspace-30k probe-train, full 5k val (`in100.dino-ctrl.ep100.parity30k`):
GAP linear 0.6442 vs ref 0.6442 (Δ 0.0000) · GAP kNN 0.4658 vs 0.4662 · CLS linear 0.6854 vs
0.6856 · CLS kNN 0.6142 vs 0.6136 — all within ±0.06 pt (criterion ±0.5). Validates extraction
transforms, branch handling, h.cls/h.gap definitions, fp16 storage, and the probe/kNN ports
end-to-end against the prior stack.
Full-curve confirmation (teacher, linspace-30k): ep25 GAP lin 0.4912/ref 0.4912, GAP kNN
0.3532/0.3532, CLS lin 0.5232/0.5234, CLS kNN 0.4380/0.4386; ep50 GAP lin 0.6010/0.6014, GAP kNN
0.4190/0.4188, CLS lin 0.6548/0.6550, CLS kNN 0.5692/0.5696 — 12/12 within ±0.05 pt.
Spectral parity (val features, N=5000): RankMe 217.0/236.9 and participation-ratio 39.5/72.2
(GAP/CLS) — exact to the reference decimals. M0 parity criterion fully met (16/16 numbers).

## 2026-07-02 — decision ratifications + space-definition revision (Berker)
- USER-APPROVED: D-001 (amended: clean repo — session files untracked, governance under docs/),
  D-002, D-004, D-005, D-007, D-008. D-003 → **D-003v2**: h = the paper-probed representation;
  z = the loss space; loss-after-only-a-linear counts as loss-on-representation (no core-7 case).
  Consistency flags F1–F4 recorded in PROTOCOL §3 (ViT analog for ResNet-native h; DINO cat4-CLS →
  h_layers [3,6,9,10,11,12] from M1; MAE probe-BN as E11 arm; LeJEPA minimal-recipe h = emb-512).
- D-006v2 probe proposal staged (headline = linear_raw_v1 + knn_v1) — awaiting OK.
- D-009 relative representations (user-directed) + D-010 METRICS.md reference added.

## 2026-07-02 — F1–F4 resolved, D-006v2 approved (Berker)
No concat/best-of readouts anywhere: h = last-layer feature of the paper's type (DINO last CLS,
I-JEPA teacher last GAP); probe protocol fixed as ours (paper quirks → E11); linear maps add no
capacity → LeJEPA h = emb-512. Headline probes locked: linear_raw_v1 + knn_v1 (house = secondary,
l2/attentive → E11). M0 probe CSVs predate linear_raw_v1 → re-running probe jobs to add it.

## 2026-07-02 — G-M0 PASSED; AUDIT_MATRIX v1 LOCKED (Berker sign-off). M1 begins: LeJEPA port.

## 2026-07-02 — M1 full-roster implementation (parallel to LeJEPA portval, per Berker)
SimCLR, VICReg, BYOL, DINO (control port), MAE, I-JEPA implemented on the frame; CPU dry-run:
step+backward+post_step+arch-rebuild pass for all six, init losses at theory values (ln3, 4.0,
ln4096, ...). Toy recipe policy = D-012. Queueing smoke→150ep chains on h100-slotB.

## 2026-07-02 — faithfulness review vs donors/officials (Berker directive; full-150 runs held)
Cloned facebookresearch/{dino@7c446df, ijepa@52c1ae9, mae@efb2a80} as donors; line-cited review of
all six methods. VERIFIED IDENTICAL: SimCLR loss math, VICReg all terms/weights, BYOL loss+MLPs,
DINO loss/center/freeze (const t_temp 0.04 = official default), I-JEPA loss/LN-order/masks, MAE
masking/loss/patchify. FIXED: SimCLR proj 4096->512 (donor IN-100), BYOL pred_hidden 8192, teachers
now EVAL for DINO/I-JEPA (drop_path in targets), MAE decoder rewritten canonical (512x8x16, fixed
sincos + cls through decoder), I-JEPA predictor pos -> fixed sincos + min_keep=10. Remaining
documented deviations in per-dossier PORT_NOTES (house optimizer per D-012; I-JEPA cls-in-context;
predictor depth 6). All six re-validated on CPU after fixes.

## 2026-07-02 — incident: stale pre-review checkpoints crashed post-review resumes
simclr full-150 died loading a _last.pt written by the cancelled pre-review attempt (old 2048→256
projector vs new 4096→512). Fixed: stale tag-less grid ckpts purged for all six methods (vicreg had
one waiting too); trainer now refuses resume when the saved arch block differs from the current
method arch, with an actionable message. Lesson: architecture changes invalidate tag-less run_ids —
purge or retag.

## 2026-07-02 — incident: DINO online probe trained on misaligned labels (view-major features)

toy.dino.s0 finished with online probe ~chance all run (best=0.2046, final 0.162) while every
objective monitor was healthy (teacher entropy_sample 0.87→0.185, proto_used 74→257, loss falling).
Offline probes on the extracted ep150 features then came back HEALTHY: student.h.cls linear_raw
0.785 / kNN@200 0.718, teacher.h.cls 0.778 — squarely in the family range. Root cause: the trainer
aligns probe labels image-major (`y.repeat_interleave(k)`, train.py); DINO's training_step alone
emitted probe features VIEW-major (`g.transpose(0,1).flatten` for the 2 globals), so the online
probe trained on wrong labels for ~all pairs; the model itself was unaffected (probe features are
detached; loss path untouched). All other methods flatten image-major (verified: simclr/byol/vicreg
`views.flatten(0,1)`, lejepa likewise, mae/ijepa k=1). Fixed in dino.py (reorder to image-major) +
probe contract clarified in base.py. Residual effects on the finished run: (1) `_best` selection
keyed on a meaningless monitor — use cadence/`_last` ckpts for toy.dino.s0, never `_best`;
(2) probe grads on garbage labels shared the global grad_clip(3.0) budget with method grads all
run — features look fine offline, but the run is not bit-identical to a clean one; rerun decision
deferred to discussion. Lessons: online-probe-at-chance + healthy-objective ⇒ check the MONITOR
before the model (offline probe on stored features is the arbiter); label-alignment bugs are
smoke-invisible (probe at chance at ep2 is normal, criterion can't catch it).

## 2026-07-03 — LeJEPA-800 portval PASSES: the trainer stack is certified

`toy.lejepa.s0.portval` (D-011 port-exact recipe, 800 ep, H100) vs Berker's official-minimal run
(`lejepa-reproduce/z2zqw1bs`, final 0.90217): final ep800 online probe **0.9037 (Δ +0.15 pt)**,
best 0.9113; sigreg loss-curve shape correlation 0.9997, inv 0.987; per-epoch acc mean|Δ| 1.4 pt
with the last 30 epochs at +0.2 pt. Within the pre-registered ±1 pt + shape criterion → the frame
loop, recipe hooks, and sslgap/ckpt/v1 format reproduce the official implementation. Every other
M1 grid instance rides this certificate (they share the loop; only recipes differ).

## 2026-07-03 — DINO probe incident CLOSED: rerun confirms monitor-only damage

`toy.dino.s0.probefix` (identical recipe, probe-label fix only): online best **0.7758** — the
monitor now reads what the audit reads (old run's offline student.h.cls was 0.785). Offline
old-vs-new comparison over all 130 matched (space, probe) cells: mean Δ **+0.0005 ± 0.0071**, no
systematic direction — the misaligned probe's shared grad-clip budget had no measurable effect on
the learned representation. Matrix/viz/orbits point at the probefix run (cleaner provenance);
the incident run's features remain in the store for the record.
