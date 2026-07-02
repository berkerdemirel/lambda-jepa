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
