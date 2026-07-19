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

## 2026-07-11 — INCIDENT (protocol, not stability): house config likely not near-optimal for lejepa at IN-100

Surfaced by the E12 C1 control (Berker: "c1 performs significantly better than our previous
lejepa run … take a note that we might not be running the best universal config").
`in100.lejepa.s0.e12c1` = IDENTICAL placement/method/frame/seed as the M2 lane
`in100.lejepa.s0`; the FULL config diff is three package items (D-026): **lr 1e-3 → 3e-4 ·
embed_calib off → on · projector spec_norm off → on** (all else byte-equal: bs 128, V=4,
proj_dim 16, wd 5e-2, warmup 10, eta_min 1e-5, grad_clip 1.0, 100 ep, seed 0; same trainer,
same online-probe schedule). **Gain (online monitor, epoch-matched, raw):** ep25 .343→.423
(+8.0) · ep40 .426→.501 (+7.5) · ep52 .475→.556 (+8.1); C1 running max .625 @ ~ep60 already
exceeds the lane's best-over-100-ep .594 (final C1 number lands on the E12 card). Both runs
QUIET (grad-norm p99 2.2 vs 2.5, no storms; same sigreg/inv equilibria) — a quality gap, not a
stability rescue. Attribution 3-way confounded BY DESIGN (C1 is A2/A3's matched control, not an
ablation); prime suspect lr (1e-3 AdamW is hot for ViT-S/16 @ bs 128 × 989 steps/ep; house value
was toy-validated at 37 steps/ep, D-018), calib/spec-norm secondary; single-factor decomposition
arms defined but PARKED (Berker's pull). Consequences: (1) M2 matrix rows carry D-012
"healthy-not-best" — now QUANTIFIED for one method (~8 pts epoch-matched); read cross-method
orderings (E01-T13/T15) with this bar. (2) E12-internal comparisons unaffected (all four arms
share the package). (3) Any future "universal config" claim must cite this entry.

## 2026-07-11/12 — the E12 saga (combined entry, owed): thesis → lazy-projector catch → K4 → f2 → G-wave

E12 ("the moment floor: what shape regularization at h is for", D-026) ran its full arc.
Pre-registered thesis: the useful part of marginal regularization at the probed space is
first-two-moment calibration; higher-order shape left free. **Berker's pre-launch catch** (on the
2-ep smokes): the as-registered moved placement (A2/A3) admits alignment-free minima — the
projector can zero `inv` by shrinkage once the z-side scale pin is gone; measured (smoke inv
15–50× below the shipped lane's END value at 20% peak lr; proj.out across-image std
near-degenerate with within>across ×1.9) → arms amended to ADDITIVE (z-side byte-identical to
control), the theory-note hole (v2.0 Appendix-E) logged. **Main arms:** P1 held (all quiet), P2
FALSIFIED (C1 > A1 > A2 ≫ A3), K2+K4 FIRED — full calibration at h achieves its own aim (effrank
275/512, calibrated) while scrubbing class structure to the randinit null; strong-form thesis
dead (E12-T1). **F-wave (pre-registered before those numbers):** dose has an INTERIOR OPTIMUM —
f2 (λ_h=.02) beats control on every converged probe (+1.2 lin/+7.5 kNN200; best lejepa-family h
measured) with ~93% of achievable constraint satisfaction (saturating dose curve) → marginal
terms at h are conditioners, not destinations (E12-T2). f6 ≈ A3 (equilibrium damage, T3); f4 =
rank-collapse attractor under pure shape pressure (effrank 2.8 < null, linear survives at .61;
framework-E7(i) falsified at scale, T4); f5's own floor UNSATISFIED at the audit frame (diag_kl
.91, kurt_worst 55.8) despite descending in training — the train-mode vs audit-frame lesson; the
A3−f5 whitening isolate stays "consistent with" (T3). Instrument dissociations (B/T vs NMI;
effrank vs linear; sliced floors under-enforce native spectra) = E12-T5. P5 Varimax numbers
landed raw (fork-gate discussion pending). **G-wave:** cross-method f2 test launched (D-028;
`e12gv` vicreg trunk-GAP, `e12gd` dino student global-CLS, λ_h=.02 additive) + **matched
floor-off controls** (D-030, Berker-directed after his recipe-confound concern): `e12gvc`/`e12gdc`
= the g-commands verbatim minus the floor, same working tree — the single-factor primary
comparators; launch-line verification found NO lr delta between g-arms and lanes (the real,
documented lr delta is the lejepa lane vs the E12 package — see the 2026-07-11 INCIDENT entry
above); what the controls close is pre-refactor lane code, hardware pool, and run epoch.

## 2026-07-12 — PIVOT day: staged merge (D-029), E13 rung-0 built+run+concluded same-session, T1–T3

Berker's D2 verdict: the external PIVOT proposal (Predictive Foveal Isometry,
`docs/PIVOT_Standalone_Research_Proposal.pdf`) enters via staged checks inside our machinery.
**E13** pre-registered IN-CONVERSATION (predictions P1–P5, kills K1–K3, primary cell and baseline
set locked pre-numbers), then run same-session over stored `pairs100` features: 20-checkpoint
provenance-matched zoo, D_read (ridge → RFF sketch of frozen MAE viewB descriptor) + D_kern
(raw-Gram alignment), full (D_m, σ) sweep, T=randinit tokenizer control, PIVOT-E0-style target
audit. The CPU dry-run caught a real estimator bug before launch (missing ridge intercept — a
shared offset from the sketch's mean-embedding component; card deviation 5). Locked n=20 outcome:
P2 NOT MET by .015 (D_read −.538 vs battery-best kurt_topeig −.553) with the declared dof-matched
control clearing everything (−.635). Berker's deitlite catch (supervised probe↔mechanism
coupling) → REGISTERED-LATE exclusion re-read: **both D_read variants beat every battery baseline
on both probe columns** (−.62/−.51; dof-matched −.72/−.65), and the family decomposition shows
the PIVOT meter and the battery's best draw their ρ from DISJOINT zoo halves (within-family vs
cross-family). D_kern: flat everywhere as a ranker, dose-ordered as a re-metrization flag.
Takeaways **E13-T1/T2/T3** (information half survives; deployment half fails; instrument lessons
incl. dof-matched-primary) — substance agreed in discussion, wording delegated veto-open.
Direction set by Berker: FOLLOW PIVOT next (rung-1 / MVI-training pre-registration discussion)
while the G-wave lands. Process note: three pre-registered sub-predictions failed informatively
(P2 photo-finish, P3 affine-forgiveness, P5 D_m-saturation) — the pre-registration discipline is
what makes those failures data.

## 2026-07-13 — PIVOT training rung: E15 (marginal MVI) and E16 (dense rebuild) KILLED; the distillation-ceiling lesson (D-032…D-034)

E15 — the first PIVOT-native training (two H100 arms, pre-registered) — hit Berker's "loss too
small, not enough tension" flag at ~ep71; forensics: `pred_over_varz` below the constant-h floor
since ep10 (h genuinely image-predictive), but the 2-draw target-noise estimate gave
E Var(z|image)=.000434 ⇒ perfect-image residual bound .300 — arm A had mined ~80% of the
extractable target with the online probe flat at ~.35. KILLED ep74: the pooled-h.gap /
position-marginal instantiation had collapsed the position-indexed conditional-law family to its
scene-average (~32 effective target dims, 30% draw noise). Salvage: head-less direct regression
is STABLE without EMA/projector; the target-budget toolkit became the D-033 METHOD RULE. E16 —
the position-conditioned dense rebuild (§6.1; e16a local, e16b +global anchor) — passed its gate
(C_pos−C_gist=+.068), smoked at 26× the tension floor, mined the target far deeper (residual
.26–.31 vs E15's .44) — and plateaued at the SAME ~.35. KILLED ep66/60 (Berker: "this turns into
a distillation setting where we try to recover mae semantics which is not even good"). Lessons:
(i) probe-invariance under deeper target extraction is the signature of a CONTENT ceiling, not
an optimization problem; (ii) the budget gate verified that position-information EXISTS but
never checked the ceiling against the zoo (C_pos .510 < vicreg's own h .58) — "beat the zoo's
best h or be self-referential" is the missing inequality (D-033 amendment PROPOSED 07-14).
En-route audits worth keeping: frozen target = scale anchor; the var floor is load-bearing
(~90% of steps active); bf16 contributes 0.5% of residual at ratio .5.

## 2026-07-13/14 — G-wave concluded (T7/T8); instrument corrections; the H-wave (5 arms); h REDECLARED (D-036)

G-wave scored against the matched controls: the moment-KL floor is a method-general
kNN-favoring conditioner (T7 — λ selected ONCE, on lejepa, transplanted untuned to vicreg/dino
at partial enforcement) and the two-space headline (T8): calibrated h wins essentially every
h-side stress dimension (control cones removed: negative-pair cosine .84→.00 vicreg) while the
floor is INVISIBLE at z — arm≈control across battery, V=8 invariance, and converged probes;
depth ladders show conditioning relocated into the TRUNK with the head's trajectory untouched.
T9 (instrument rows: monitor bias, margin-vs-ratio coupling) left OPEN by Berker. The vicreg
monitor sign flip's aug-input mechanism was tested and REFUTED (offline converged probes on the
monitor's own view distribution stay negative). New standing instruments: V=8 orbit stores at
the E02 ladder taps + per-layer v1L eval stores (six ckpts); head-Lipschitz (per-segment
view-pair stretch + weight spectral norms) — arm heads tamer exactly at the segment touching
the floored tap, deeper segments numerically IDENTICAL; dino's bottleneck pocket now on three
instruments. Berker's wiring catch: vicreg's projector consumes CLS while the audited h was GAP
— gv had calibrated a tap the loss never eats, the vicreg Lipschitz "first segment" was not a
computed path (corrected: cls→tap1, 1.5× taming, trunk-propagated), and CLS out-probes GAP on
ALL three F1 methods (+1.6…+5.3 lin, +7…+15.7 knn) ⇒ **h REDECLARED = the input of the
projection (D-036; execution owed next session)**. E13 extension: every zoo method as the
D_read target — ranking power tracks target quality (dino −.91; T=mae −.62 reproduces E13-T1
exactly) and the affinity knockout is clean (lejepa −.43 full-zoo → −.98 family-excluded).
H-wave launched (D-035): gd2 (dino floor CLS+GAP, gap dose pull-matched .0401) · f7 (f2 +
h_inv=.8478 at embed; its floor RECOVERED under the pull, 2.1→1.4 by ep57) · gv2 (floor at CLS
— "the fixed gv") · gvcls (floor+inv at CLS, h_inv=.9790) · gvi (floor+inv at GAP; HELD after
smoke: half-pace probe, floor above the control-free level). Doses via the generalized per-term
pull convention (e12h_pull.py; assists at 10% of the shipped invariance pull) — the
coefficient-vs-pull lesson again: "very small" landed at coefficients .85–2.9 because the
assist terms' raw gradients are 3–9× weaker than the reference. Process lessons: smoke WHAT YOU
LAUNCH (dose-dependent risk forced second smokes at measured λ); a rising floor in a 2-ep smoke
is tension, not collapse (f7's chain recovered it); the floor's relative pull at equal nominal
λ spans 0.7%/2.2%/8.8% across methods — nominal λ is not a cross-method dose.

## 2026-07-14 — D-036 EXECUTED (h = projector input, F1 GAP→CLS); monitor-tap fix + fresh vicreg restarts; clean CLS migration; PIVOT ceiling ruled (D-038)

**D-036 fully executed** (the redeclaration owed at the prior close). Docs: PROTOCOL §1/§3/F1
rows + monitor line + v1-draft.5; AUDIT_MATRIX v1.1 locking row (prediction letters UNCHANGED —
only which stored tap is read as "h" moved); E02 lock feature-type amendment landed BEFORE any
E02 score (integrity preserved). Code: the declared-h tap lives in ~13 hand-kept mirror dicts +
the `adapters.py` extraction authority + two inverse edits (`e1_matrix_figs.ALT`→GAP, randinit
null→CLS) — all flipped simclr/vicreg/byol → `student.h.cls`, guardrail-grepped (every surviving
`student.h.gap` is legitimately mae/ijepa/toy/pivot/floor-values or an intentional GAP diagnostic).
Battery + probes are tap-AGNOSTIC (store every space) ⇒ **no re-extraction/battery/probe** — the
CLS numbers already existed; regeneration is pure re-selection + re-plot. Regenerated at CLS and
spot-verified (M2 table + class-align now read `student.h.cls`): report_m2 (E1_IN100_MATRIX.md),
e1_matrix (5 figs), pair_margin_m2, e12g h-vs-z (4 figs) + ladder, class-align, e13 chain (7
scatters + 4 figs).

**The monitor-tap fix (Berker's catch made precise).** Berker asked "is the linear probe already
at the right position for dino/lejepa?" — it is, verified in code on BOTH monitor sides: dino
trains+evals on teacher CLS (`dino.py:121/147`), lejepa on `emb`=z.embed (`lejepa.py:261/265`).
vicreg/simclr/byol sat at **GAP on both sides** (`probe_feats` train + `eval_features` eval) —
and fixing only one would train the probe on CLS but score it on GAP, a silently-broken monitor.
Moved both sides → CLS = declared h; `_best.pt` now selects on the declared tap (was GAP). gvcls/gv2
restarted **fresh** on the CLS monitor (Berker: "they are fresh enough"; verified CLS probe_acc
climbs *above* the old GAP curve as D-036 predicts). simclr/byol fixed too (future-runs; not
running). **Lesson: the "probe position" is a train+eval PAIR — a half-fix is worse than none.**

**Clean CLS migration** (Berker: "clean audit, clean viz, migrate fully to cls whenever
necessary"). Re-stamped 76 IN-100 F1 `.ext` meta.json `h_space` GAP→CLS (filtered by method so
mae/ijepa untouched) — this is the ONLY consumer that reads stored provenance (`audit.py:68`),
everything else reads the mirror dicts. Clean audit re-run for gv/gvc (CLS `.cross.csv`); clean
viz (toy rung — its config is toy-only). GAP-specific diagnostics (`probe_conv_check` =
GAP-feature probe-censoring evidence for D-020; `e12_aug_probes` = the parked T9(i) monitor-bias
instrument) left at GAP BY DESIGN — they *study* the GAP tap, they don't measure h; Berker
confirmed "gap specific measurements we dont care right now." Toy meta + simclr/byol audit +
IN-100 viz config deferred ("whenever necessary").

**T7/T8 held (Berker ruling), not amended.** gv/gvc CALIBRATED the floor at GAP; reading them at
the new declared h (CLS) is a placement mismatch — "you cannot reclaim anything from that
experiment." At CLS the gv-vs-gvc deltas are a mild +1.0 knn / +0.5 lin (the GAP floor barely
conditions the projector-input tap); the dramatic +6.1 knn / −1.6 lin survives as the GAP
*intermediate-tap* reading. The clean CLS-calibrated-and-CLS-measured result comes from gv2 (floor
at CLS) + gvcls (floor+inv at CLS) — the H-wave arms. So the regenerated h-vs-z figures showing
gv≈gvc at CLS must be read as "GAP-floor's weak effect at CLS," NOT "calibrated vicreg."

**Chain crash/resume + landings.** All four H-wave chains took a cluster wall/preemption mid-run;
the two mature runs resumed cleanly from checkpoints (gd2 ep48, f7 ep57) — mature-run ckpt hygiene
held — while gvcls/gv2 were too young to have a ckpt and restarted (then were re-restarted fresh
anyway for the monitor fix). gd2's later 8h wall-rollover resumed correctly via the singleton
follow-on (a `DUE TO TIME LIMIT` signal the incident monitor correctly surfaced — a normal boundary,
not a failure). f7 landed best .6972, gd2 best .6944; both extracted (adapter=native, h_layers
[3,6,9], orbit V=8, do_pairs=false) + probe/battery chained. H100 pool returned to the 2-slot cap
on its own as slotA/B freed. **Lesson: a fig script that TRAINS probes needs a GPU** —
`e12g_ladder_fig` (fresh L-tap probes via `sslgap.probes`) failed on the CPU partition; the
"CPU-only npy reader" assumption held for the read-only fig scripts but not this one.

**PIVOT ceiling ruled — D-038.** (1) D-034 distillation-ceiling diagnosis **CONFIRMED** (Berker:
"keep consistent to distillation view") — evidence = extraction-depth invariance (E15→E16 mined
deeper, ratio .26–.31 vs .44, at the SAME ~.35 plateau) + the plateau below mae's own raw h-probe
(.434). (2) D-033 target-budget gate **AMENDED — the beat-the-zoo condition**: an external-teacher
target must clear the best zoo h-probe + margin, or be self-referential (teacher-free); Berker's
emphasis: teacher-free, **self-distillation the leading redesign direction**. Then a full close-read
of the PIVOT proposal (Predictive Foveal Isometry) surfaced that **self-distillation competes with
the PIVOT theory**: converting the frozen external tokenizer to an EMA teacher escapes the content
ceiling but forfeits Theorem 1's fixed-target identification (§4.1), the target-content audit
(§5.2/§10 — the very D-034 instrument), and the "declared-not-emergent" high ground (§1.2/§7.4) —
PIVOT drifts toward the I-JEPA row it defines itself against; it also forces the variance floor to
become load-bearing (gate 14's "PIVOT incomplete"), which MERGES the redesign with the
calibration-at-h/E12 program, and weakens the RFF justification (gate 10 — a strong EMA teacher
smooths away the conditional multimodality that RFF exists to capture). The surviving distinctive
is the declared foveal channel Ψ (gate 8). Berker is brainstorming the fork; redesign PARKED,
budget gate standing.

**Two artifacts + a hypothesis noted.** (a) `docs/theory/CALIBRATION_AT_H_GENERALIZATION.md` —
Berker's hypothesis that calibration-toward-h may be general (any method's OWN term, slight, at h),
with inv/moment-KL the easy instances and DINO's clustering-at-h the falsifier; "would lead
something more general and bigger." (b) `docs/literature/related_work/kalapos_whitening_improves_ssl.md`
— close-read of arXiv:2408.07519 (whitening as a differentiable IterNorm layer at the encoder
output h, method-agnostic, +1–5%): the nearest neighbor to our premise, **NOT a blocker** (Berker),
our two-space audit / trunk-relocation mechanism / soft-diagonal-floor / placement all untouched;
recorded as a post-report BIBLIOGRAPHY addition. **git push d662fa3** (all the above; slides/tarball/
session-scaffolding deliberately left uncommitted).

## 2026-07-16 — E17 landed: the desideratum-transfer matrix scored, interpreted, APPROVED (D-039); H-wave completed; the connectivity finding

**The session that turned E17's raw numbers into the project's strongest result set.** Opened with
a skeptical re-verification of the prior session's (different-model) work: every scoring number
reproduced exactly; implementations faithful to the signed-off designs (one 2%-magnitude deviation
logged: simclr uniformity pools both views). Then four instrument-building passes over the landed
stores, each triggered by a Berker question, each becoming a card block: (a) **jump decomposition**
— the head's margin jump is entirely rand-side, alignment contribution NEGATIVE in all non-dino
controls; (b) **cone = mean** — rand_cos ≈ ‖μ‖²/E‖x‖² to 3 decimals everywhere, centered rand ≈ 0
(⇒ vicreg's var+cov is mean-blind vs a mean-carried cone — derivation); (c) **centered probes** —
our knn_v1 is weighted-cosine ⇒ mean-sensitive; eval-time centering recovers only +0.5–3.4 of the
gains, every winner beats its centered control, f2 centering-insensitive (T7-continuity held);
(d) **class-pair d′ + h~z gentleness + wandb final losses** — sigreg drops mean d′ −17% (the shape
channel visible in class-contrast directions), f2 lifts the worst tail (p10 +15%); head-gentleness
rises with total desideratum satisfaction but occupies all four {gentle,wild}×{good,bad} cells;
final z-terms equal-or-worse everywhere except byol. **The connectivity finding (Berker: "one of
the best findings we have so far"): augmentation-cloud touch% tracks kNN monotonically within
every ±inv family** (C1→f2→f8→f7 = 87.6/99.7/95.3/91.2 touch vs 53.1/60.5/60.2/57.6 knn);
threshold-like; figure `results/figures/e17/e17_touch_vs_knn.png`. **The f2 mechanism explanation
(Berker: "one of the biggest wins"): safety (cluster-blind) × activity (rotation-swept,
non-absorbable graded floor — per-eigen slope ½(1−1/λ) ⇒ ~65× weakest-first asymmetry at ε-dose;
whitener at destination dose)**; f5/f2/A3/sigreg quadrangulate; higher moments stay data-driven
(Varimax kurt .44). **T1–T7 APPROVED** (card §AGREED TAKEAWAY; D-039; theory doc verdict appended;
reviewer report `docs/report/e17_desiderata_at_h_findings.md`).

**H-wave scoring COMPLETED (owed since 07-14):** f8 = best-of-both (lin 67.0 — best lejepa linear
measured — at f2's kNN; shape-dominant 4.5:1 is the favorable mixing corner); **gv2 settled the gv
stake: GAP gains were tap-local** (gap knn −0.7 vs gv's +6.1; gv reinterpretation owed discussion);
floor-only gv2 zeroes vicreg's head jump (+.474→+.005) — the mean-seeing/mean-blind contrast
cross-validated; gvcls adds ≈ nothing over gv2 (P3 MISS recorded); gd2-P4 landed after
val-store re-extraction (`e12gd{,c}.ext2`): both-tap flooring removes the mid-trunk deficit
(gd2 −0.9/+1.6/−1.9 vs gd −3.0/−5.1/−3.6).

**vicreg closed — WITH A CORRECTION OWNED IN-SESSION:** the takeaway bullet was first drafted from
ep15 extrapolations ("can only dilute; needs 15–20×") and the converged c015 numbers refuted it
minutes later — at 6×/ep100 the cone breaks (rand .035; var-hinge saturation tr→362≈d + residual
mean halving) and vicreg JOINS the T7 winners (+4.4 knn). Bullet corrected on the card same-day,
D-039 row carries the amendment flag, Berker re-confirmation requested. Lesson repeated: never
extrapolate mid-training reads to convergence claims.

**Session lesson (meta):** the strongest results came from instruments built to answer Berker's
questions within the hour (cone decomposition, centered-kNN, d′, touch%) — cheap store-analyses,
each < 30 min, each changing the interpretation. The expensive part of E17 was already paid; the
understanding was in the cheap passes.

## 2026-07-17 — E18 (declared-prior t_ν + sigreg@3%) and E19 (floorssl 2×2) launched; the reach instrument iterated to null-calibrated v3 under Berker's live design pressure; μ-drift attributed-candidate; aug-family caveat recorded

**Two experiment families pre-registered and launched, four chains running overnight.** E18
(D-041): `hpull_sigreg_t` = E17's sigreg-at-h with only the CF target swapped to unit-variance
t_8.2 (ν refit on the actual lane: e17c Varimax kurt 1.428; e12c1 reproduces the recorded 1.063;
slice-kurt datum 0.534 recorded as the sign-flip interpretive key, ν_slice≈15 queued) at the same
10%-pull rule (fresh pull byte-reproduced the E17 record) + rider `hpull_sigreg3` (3% dose,
retires T3's over-dose flag). E19 (D-042): `floorssl` = the minimal method (vicreg frame, var+cov
→ floor at z, w_floor 19.10 by destination-dose equal-pull; menu Z1≡Z3 collapsed — vicreg's
alignment is already plain MSE) + `floorssl_hz` (gv2's h-floor verbatim) completing a 2×2 whose
other cells (control, gv2) existed. Berker's session pivot: "focus on our f2 winner — you have a
theory of why moment works at h, move it to z; big priority."

**The reachability instrument went through three design rounds in-session, each driven by a
Berker catch:** v1 (ball-proxy within-class percolation + MST span cost) → his "cannot compare
with negatives; if everything is collapsed your version would look good" → v2 adds matched
negative-null rows (the D-040 data-vs-null convention) + his relay/interleaving axis → the nulls
exposed winner saturation (f2 margin +0.00 at every fixed radius: UNIVERSAL touching) → his
"margins are not tight enough" → v3 sweeps shrunken balls α·r_mean and reports margin_max/α*
(self-calibrating operating point; f2 de-saturates at α=.75 with margins +.33/+.28). Full sweep
landed (`results/diag/e17_reach.csv`, 16 runs × 100 classes; relay axis degenerate — full-graph
percolation saturates everywhere, the graded interleaving readout is perc_borrow). The
f2-paradox (best kNN, zero raw margin) resolved by the distance-anatomy figure: kNN reads the
d_intra-vs-d_inter ORDERING, percolation reads the touching THRESHOLD — f2 orders cleanest while
touching universally (`e17_reach_anatomy.png`). Berker's traversal probe (Prim walk on
min-view-pair support distances) built and piloted: his anisotropy hypothesis CONFIRMED
(top-eig share .31–.58 vs 8-point null .17; sigreg_inv most elongated, f2 roundest), alignment
is class-global not local-filament, and the walk is expensive+impure everywhere at 8 views —
the undersampling bound motivated o32 re-extraction (Berker-approved; ctrl landed, f2/sigreg_inv
OOM-killed at 96G and resubmitted at 160G; fresh read NEXT session).

**μ-drift (T5 loose end) landed raw:** c015 cadence shows dilution done by ep25 (trΣ 334/362),
the mean INFLATING first (48.7→82) then decaying ×3.9 alone while ROTATING fully away from the
ctrl cone (cos .24→−.07). Candidate named for discussion: wd erosion of the loss-orphaned mean.
Deprioritized by Berker; wd-ablation is his trigger.

**Corrections and calibrations owned:** ν=8.2 is fit on a CONVERGED control — circular as a
method recipe, diagnostic-only (card caveat); "MomentFloor consumes no RNG" sharpened to
at-construction-only after Berker's catch; the aug families are per-method BY DESIGN (D-004) —
lejepa 4×symmetric-strong vs vicreg BYOL-pair vs ijepa crop+mask-only vs VISReg DINO-multicrop —
recorded consequence: floorssl is aug-unconfounded, f2's win carries an aug-family scope caveat
(findings report §6 + E19 card). Dose-matching declared init-only (Berker): trajectory scoring
(per-epoch per-term g_enc from cadence ckpts, per-knot CF residuals, geometry trajectories) now
mandatory on both cards. Cross-lane raw loss values shown non-commensurable (floorssl-vs-f2
term comparison untangled at matched steps); real watch item: floorssl_hz's h-floor residual
RISES (1.25→1.39) where f2's same-λ h-floor falls — the two floors may interact through the
expander.

**Novelty landscape shifted:** VISReg (sliced-Wasserstein, shape-KEEPING — our published foil),
Weak-SIGReg (sketched cov-only), KerJEPA found; "first sliced moment regularizer at h" is the
claim Berker set; both papers queued as pre-paper must-reads.

**Session lesson (meta):** presentation discipline re-escalated twice (Berker: "tired of plain
texts … present the metrics, what you see, then takeaways — you threw 30 observations at me"):
metric-definition-first is mandatory, few observations per message, figures over tables, and
mechanical detail stays on the cards. The instrument iterations (v1→v3 in hours) worked exactly
because each of his design catches was implemented against the same stores within the hour.

## 2026-07-17b (same-day refresh) — the commensurability session: both parked questions closed at instrument level; chains mid-flight

**The parked h_kl question (q2) answered with a new instrument, not an opinion.** Built
`e19_floor_anatomy.py`: the floor's logged value decomposes EXACTLY into three dimensionless,
cross-tap-commensurable parts (cone ½‖μ_Q‖²/d′ · scale ½(m−1−ln m) · aniso = logdet Jensen
gap), computed at cadence ckpts on fixed seed-0 batches (n=512 at every tap in all lanes);
recomputed totals reproduce every live curve (f2 .189/.19, gv2 1.00/1.02, hz 1.37/1.4,
z-floor .43/.43 — triple cross-validation incl. the pull rows' loss values). The read:
floorssl_hz's rising h_kl RIDES gv2's own rise→crest→slow-fall hump (lane/tap property, not a
z-floor interaction); the CLS value = frozen scale (per-dim var never leaves .15–.19 at λ=.02)
+ an aniso hump that carries all the net decline; f2's falling curve is the same functional
with scale held ≈ 0 the whole run by the calibrated affine embed — its converged .19 is ~98%
aniso residual. Bonus structural finding (raw, discussion pending): the floor's pressure
TRANSMITS through lejepa's affine emb (f2's floor@embed cleans CLS one layer upstream: rand
.76→.04, var ×70) but NOT through vicreg's BN expander (floorssl's floor@z leaves CLS cone/var
byte-near the control's) — BN as moment firewall vs affine as conduit.

**Dose-matching-is-init-only, quantified.** `e12h_pull.py` gained the card-mandated `+ckpt=`
extension (gotcha: `frame=in100_vits16` must be explicit — default frame is toy; first
submission wave failed the arch assert, plus f2's lane needs `+method.spec_norm=true`).
Realized floor shares drift ×9 UP (f2; lejepa's z-side g collapses 455→1.7 after init) vs
×6 DOWN (gv2; vicreg's z-side g grows — the late grad_norm rise) vs destination→parity
(floorssl 83%→48% by ep18): drift direction is a LANE property. Same-name floors at
mid-training span ~2000× in realized share. Memory + E19 card updated; figure
e19_pull_traj.png.

**Reach v3 read (q1) + T4 transport.** 3-axis figure: run-level margin_max@α* couples to
orbit tightness (pos_c) and anti-tracks kNN (f2 lowest margin, sigreg_inv top); the α-sweep
de-saturates every spread-orbit space the fixed radius lost; within f2→f8→f7 margin_max is
the exact inverse of T4's touch% — T4 transports sign-flipped and null-calibrated. Q3
(standing readout) queued for Berker. o32 traversal landed; instrument bug caught before the
aniso read (null hardcoded V=8,D=512 + 8-view-chunked shares) and fixed same-day: full-cloud
shares vs matched nulls STRENGTHEN the anisotropy confirmation (f2 .180 / ctrl .357 /
sigreg_inv .451 vs V=32 null .049) and keep f2 least-anisotropic; ctrl's walk-adjacent axis
alignment now exceeds random (.35 vs .23) — the "class-global not filament-local" o8 reading
softens for ctrl.

**Chains (mid-flight, no extrapolation):** sigreg_t ep23 (.273) · sigreg3 ep29 (.297) ·
floorssl ep21 (.351) · floorssl_hz ep24 (.389) — all healthy, grad_norm on-family, 3×8h
chains sufficient (slotA tightest at ~19.5h to ep100). Both E19 arms track/exceed the
controls' early probe pace; P3 kill criteria never approached.

## 2026-07-17b (second half) — Berker's jobs: the conduit arms (D-043/D-044) + the intersection question driven to resolution

**D-043 conduit arms launched** (Berker: "add an affine transform to vicreg's trunk and rerun
floorssl"): floorssl_emb + floorssl_hz_emb — bare on-path Linear(384→384), the lejepa topology
transplanted (created after the expander: init streams byte-matched; running-chain
byte-invariance CPU-verified before every vicreg.py edit). Launch gotcha banked: num_classes=100
must be explicit (toy default 10 → probe nll assert killed the first smoke pair). **The hz_emb
h_kl rise Berker flagged was DIAGNOSED same-day** with the anatomy instrument: kl 2.00@ep10 =
aniso 1.96 + scale .0008 + cone .04 (per-dim var .954) — the ε-floor WINS the affine-absorbable
components and transmits them (cls cone halved vs the unfloored twin), but the unconstrained
Linear is itself an anisotropy source at ε-dose; NOT init mis-scale. **D-044** on Berker's
explicit correction ("i was expecting a well behaving f2 like h_moment_kl … create your run
accordingly" — the uncalibrated cell was Claude's stricter unilateral choice): embed_calib
transplanted to vicreg (one-shot fold, once-only + resume-safe validated), arm
floorssl_hz_emb_cal launched after a smoke whose h_kl starts .854 FALLING — on f2's early
trajectory. Stale-NFS lesson: a pull job read vicreg.py seconds after the edit and ran OLD code
(rows byte-duplicated the uncalibrated record — caught by value comparison; invalid row
superseded, e19embcal2). P-cal-A (stays low = init-transient story) vs P-cal-B (climbs =
lane-intrinsic aniso; f2's difference = inv-through-embed shared duty) pre-registered;
Claude's pick B.

**The correlation puzzle resolved** (Berker: "everything I think should correlate positively
anti-correlates with kNN"): radius-thresholded statistics measure where r_mean sits in the
distance distribution (= spread, the pos_c axis); kNN reads the ordering; the winning terms
move both. Census mechanics: p_pos ceilings at 1 while p_neg has 100× headroom ⇒ enrichment
decays to 1 in well-connected spaces and is large exactly when balls are tiny. Instrument
lineage e17_intersect v1→v4 (two real nulls: no view-point interpenetration anywhere;
ball-touch ⇏ containment) → **Berker's touch census** (all anchors × all candidates,
p_pos/p_neg/enrichment/purity): E>1 in all 16 runs (weak hypothesis universal), E spans
19.1(lejepa ctrl)→1.1(f2, touch-saturated at chance purity); within-family kNN winners touch
MORE and select LESS (byol excepted). What survives: ordinal readouts (walk purity — f2 78%
best) and spectral ones; prescription recorded. Figures: e17_touch_census.png,
e17_intersect(_explainer).png; full numbers on the E17 addendum.

**CORRECTION (same evening, Berker's catch):** the "calibration verified — h_kl on f2's early
trajectory" line above was wrong (read 40 steps only). Full curve: cal bottoms .69 (~s33–140)
then re-inflates to 1.62 by end of ep2 (≈ uncalibrated init level; twin plateaus ~2.0) — f2
never re-inflates (.84→.19 monotone). Calibration re-based the start only; something
structural separates lejepa's embed from vicreg's emb-out under the identical ε-floor. Question
carried to next session with ZERO direction (SESSION_OPENER.md); P-cal-A/B predate this read.
Also on record: both emb arms log s1=1.63 — autocast's weight-cast cache makes step 1 use
pre-fold weights (cosmetic, from s2 the fold applies). Second in-session lesson about checking
curves fully before claiming behavior (first was the E17 c015 incident): mid-run claims cite
the WHOLE logged trajectory, not its head.

## 2026-07-17c → 07-19 — the dose-law session: /goal resolved, E18+E19 scored, E20 zoo complete

**The /goal (Berker: "identify the issue between f2 line and this one … lr match first …
divergence points").** Six single-knob cells on the arm-5 calibrated base + a two-point dose
rung, all pre-registered (E19 card §ladder). Resolution, certified: the issue is the
NOMINAL-λ transplant — same λ=.02 buys f2 6.2% realized share at its tap (its z-side
adversary dies ×98 post-init) but 0.13% in the vicreg-emb lane (adversary persists ~25×
stronger); the floor wins flat channels (cone/scale) at any dose and loses the contested
aniso channel in proportion to share (crest 2.02@ep12 → 1.24@ep2 → none as share .13% → .7%
→ 6%). Fix: λ* = λ_ref·share_ref/share_lane (h_lamb≈1.0 in that lane) — the lam50x cell held
ON f2's curve to ep12.3, anatomy-verified (0,I)+falling-aniso, probe-neutral. Eliminated by
direct cells: lr (clock rescale ≈ lr-ratio), head-out dim (d128 fails FASTER), spec_norm and
emb-wd0 (trace-identical to the failure), init (arm 5), topology (D-043), the Linear's own
drift (identical unfloored pathology in both lanes). Bonus: at correct dose the f2 reach-back
conduit reproduces in vicreg (D-043's question was dose-gated). Berker killed arm 5 en route.

**E19 ep100 scored (4 arms):** P1 MISSED — floorssl (z-floor as sole anti-collapse) reads
BELOW the shipped var+cov control at declared h (−2.1 lin/−2.9 knn); hz recovers knn to
parity only; P2 held (cone@z dead, rankme z 875–905 vs 562–567); arm ordering matches the
ladder's dose logic. hz_emb's standing watch answered: crest 2.02@ep12 → 1.05@ep94.

**E18 ep100 scored (2 arms) + knot instrument built** (`e18_knot_residual.py`): P-A branch —
the shape tax PERSISTS under the lane-fit t_8.2 prior (knn −4.8 ≈ gauss −4.3) while the
instrument proves the arm OBEYED it (slice-κ .34→+1.00 toward 1.43 — tail-pressure sign
REVERSED; fits its own target best at tail knots; only kurt_worst moves, 8.53 vs 2.86).
Rider: tax SURVIVES 3% dose — T3's over-dose flag dead. Extraction-spec deviation (defaults
vs card's h_layers/o8 spec) caught against the card and redone before any dependent row.

**E20 (Berker: calibrated per-method floor at h, zoo-wide, z untouched):** h_reg=moment
added to simclr/byol/mae/ijepa (absent-key byte-identical, CPU dry-runs vs stored ckpts);
dose rule λ_m = .0638·A_m/ĝ_h with T=6% and ĝ_h=.30 (held-state denominator — AMENDED
pre-launch when the control-state convention failed the lejepa anchor .0007-vs-.02);
per-lane control-matching overrides; 7 smokes → 7 chains → all landed + scored in ~30 h.
FULL TABLE (Δlin raw · Δknn vs CENTERED ctrl, Berker's convention): lejepa +6.6/+10.0
(plain lane BEATS f2 +1.2/+2.3; lr-difference flag) · byol +6.1/+9.7 · simclr +1.1/+5.0 ·
vicreg −0.2/+3.0 (≈3× gv2; raw knn = c015) · dino −2.5/−0.4 · ijepa −5.5/+2.1 · mae
−8.4/−0.1. All floors held (no divergences); arms centering-invariant. TAKEAWAYS APPROVED:
E20-T1 (dino's miss = DOSE — its three-point ladder .02/.06/.258 declines monotonically;
T=6% not a universal optimum) · E20-T2 (the four winning view arms unconverged at 100 ep —
slopes +1.2…+2.0/20ep; controls also climb; extension = separate decision, parked).

**Instrument/process lessons:** e12h_pull g_enc is trunk-module-only (cross-lane h-term
share comparisons biased — memory + card); torch spectral_norm DOES draw RNG at construction
(measured); non-chained train.sbatch jobs AUTO-REQUEUE on timeout (cancel explicitly —
diag cells looped once); mae/ijepa yaml patch=8 defaults ≠ M2's 16 (arch assert caught);
wandb sampled histories starve sparse per-epoch keys (pull test rows with keys=). New
instruments: e19_diag_curves · e18_knot_residual · e20_curves · e20_centered · anatomy argv
mode · traverse explainer (priority (ii) delivered). Session ran ~fully autonomously on
watcher-chained mechanical batches; all raw, takeaways only where Berker stated them.
