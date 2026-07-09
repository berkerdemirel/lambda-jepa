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

## 2026-07-09 — probe-convergence incident: linear_raw@30ep was boundary-censored (D-020)

Trigger: Berker questioned the M2 seed-0 offline probes sitting below the online monitors. First
finding: the comparison itself was mis-drawn — the monitor head is LN+Linear (= house twin) and
probes the audited-h space per method (dino teacher-CLS, lejepa embed-512), so the honest deltas
were small and both-signed (offline house ≥ online for 6/7; dino +2.0 = probe-train-size/aug
edge, consistent with the certified dino-ctrl 30k→50k slope +1.6). Second finding (Berker's
convergence hypothesis, confirmed by diagnostic job 62168225 → results/diag/probe_conv{,_curves}
.csv): `linear_raw_v1`'s fixed 30-ep budget boundary-censors plain Linear on unnormalized GAP
features — best_ep=29 in every GAP cell, 4.7–6.8 pts below converged, differentially across
methods (vicreg hit hardest; converged ordering differs from raw@30 ordering). Affine
standardization (capacity-neutral) at 30 ep recovers ~all of it ⇒ optimizer conditioning, not
feature geometry; per-sample LN residual ≈ nil at convergence (raw/std/house within ~1 pt at
300 ep). CLS spaces converge by ep~20 — the artifact was invisible at CLS, biting only at
GAP/embed. Resolution per Berker ("we shouldnt change our way of evaluation but we should make
sure it converges"): v2 probe family, patience 120/cap 1000, best_ep+epochs_run in every CSV
(D-020; PROTOCOL v1-draft.3); v1 columns kept for continuity; all stored-feature run_ids
re-probed. Lessons: (1) a fixed probe budget is itself a protocol choice — convergence must be
measured, not assumed (E11 gains a convergence axis; this diag is its first datum); (2) monitor
numbers are only comparable to offline probes at the same space with the same head — HANDOVER
now names each method's monitored space.

## 2026-07-09 — E10 rescue campaign: 8 runs → a mechanism, a scaling law, and E10-T1

Berker's rescue mandate (3×H100 granted, D-021) ran the full arc in one day. Grad-share
measurement (results/diag/e10_grad_share.csv) revised the pre-registered λ rule mid-flight
(A-matched → equal-pull; amendment dated pre-launch): at init B's balance was already A-like —
the pilots died DYNAMICALLY. Rescue arms: Br (λ=.0035) stormed ep12; Bi (calibrated init,
λ=.02) stormed ep12 — the timm head init (trunc_normal .02, not fan-in-scaled) leaves the embed
~10× under-scaled, and fixing it delayed nothing; Dr (λ=.0015) never stormed but
variance-collapsed (inv→1e-5, sigreg pinned at the derived σ²→0 ceiling 103 — theory value
visible in a live curve); Blr (lr 3e-4 + calib) reached the family's healthiest state (sigreg
5.2, inv 2e-4, probe .28) then stormed ~ep38. Berker's curve-reading found the mechanism class
(post-warmup lr-triggered instability) and his forensics instruction is now standing practice
(memory: curve-forensics-default); D0's "late decay" was re-diagnosed as the same storm
(ignition ep32 — the grad-share replay on fixed ckpts was structurally blind to on-trajectory
storms). Fuse-length law: ignition ep12/19/32 @1e-3 → ~38 @3e-4; buffered A: no ignition in
145 ep. SIGReg value calibration (results/diag/sigreg_ref.csv): isotropic floor 1.053; at K=512
every trained embed reads ≈ its covariance-matched Gaussian twin (shape residue ≈0) vs K=16
where 84% of A's deviation is shape → E10-T1 agreed (dimensionality sets the statistic's job;
placement sets stability — Berker's correction of the initial phrasing). Final arm e10Dlr
(both losses at embed, λ=.02, calib, lr 3e-4) pre-registered and running: broke every family
record (.42 @ep55, quiet grads); analysis when it finishes. Also: deitlite supervised anchor
built+launched (D-022; trainer y pass-through, 8-method CPU revalidation); Dr battery chain
landed (toy.lejepa.s0.e10Dr.ext).

## 2026-07-09 — D-020 re-probe fleet complete (33/33); l2-probe censoring surfaced by design

All stored-feature run_ids re-probed with the v2 family. Every M2 HEADLINE cell (linear_raw_v2
at each method's h and z.final) converged uncensored. The recorded best_ep/epochs_run columns
immediately surfaced a follow-on protocol fact: `linear_l2_v2` hits the 1000-ep cap in dozens
of cells across toy+IN-100 (unit-norm inputs → tiny initial logit scale → slow convergence —
the same conditioning mechanism as the raw@30 incident, one layer deeper). l2 is an E11 arm,
not headline; deferred to the E11 M2 pass (options there: longer cap, per-probe lr, or carry
capped flags — any change needs its own DECISIONS row). Superseded dino incident-run CSVs
(toy.dino.s0.ext*) intentionally left v1-only.

## 2026-07-08 — theory day (evening): anchor verification landed; Dubois '22 deep-read + positioning

Four agent quote-verification reports landed in `docs/theory/verification/` (ib_mi_byol,
dubois_xu, identifiability, saunshi); in-place corrections applied to DESIDERATA §3 with
[verified] tags. Headline finding: the OP-4 quote ("we still do not completely understand the
impact of non-linear projections") exists in NEITHER anchor paper — it is Dubois–Hashimoto–Liang,
ICML 2023 (arXiv 2302.03068) §5.3.3; OP-4 re-cited, founding report needs the same fix. Also:
Dubois '22 is NOT head-silent (proves the need for heads; symmetric projection breaks
representation-layer linear optimality) — DESIDERATA §5.1 novelty claim re-scoped to the audit.
Berker's backward-direction question on '21's unconstrained-Bayes-risk guarantee (expressive
probes: multiple ERM fits sever the probe-risk ↔ representation link) → Dubois '22 deep-read:
their Def. 3 W_n takes sup over the FULL ERM argmin set — the worry is inside their optimality
concept, with exact invariance as the argmin-taming condition. Comparison note
`docs/theory/DUBOIS22_VS_TRD_PI.md` written (DRAFT, not agreed): quantifier algebra, exchange-rate
table, import candidates, and (§7) the novelty/strategy read — recommendation "proceed" (the
'22-solved corner is not our problem; their own '23 paper couldn't confirm the '22 head
prescription), two novelty re-scopes, DISSL-as-positive-control candidate. ALL pending
discussion; nothing agreed. Lesson (for the ledger): quote-verify anchors BEFORE building on
them — one misattributed quote had propagated into OP-4 and the founding report.
