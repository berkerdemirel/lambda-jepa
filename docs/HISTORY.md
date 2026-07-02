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
