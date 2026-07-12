# E14 — PIVOT rung-1: does D_read still rank on the DECLARED foveal channel?

**Status:** pre-registered 2026-07-12 in-conversation (Berker sign-offs: "e14 first makes more
sense to me too" · "rest looks good and you can use h100 for this job" · previews approved:
"ok we stick with only f96. do T=dino control as well."). DECISIONS D-031. Zero-training rung:
extraction + CPU scoring only.

## Question

E13-T1 established that held-out predictive-state distortion ranks the zoo — on augmentation-view
proxy channels with overlap uncontrolled (its biggest declared caveat). E14 rebuilds the channel
as PIVOT §6.1 declares it (sharp fovea + degraded surround; contexts by known crop maps, overlap
controlled BY CONSTRUCTION) and asks whether the ranking survives. Last zero-training
falsification before any PIVOT-native training (D-029 staged merge); the foveal sampler built
here is the MVI input pipeline if E15 happens.

## Construction (locked; previews approved before launch)

- **Base scene:** house eval geometry — Resize(224) + CenterCrop(224), deterministic.
- **Event (stack `foveal_v1`):** fovea = sharp 96×96 window (6×6 tokens, 16-px aligned; 81 slots)
  pasted into the base scene downsampled ×4 (224→56→224, bilinear both ways). Two events per
  image (foveas at c_A, c_B). Composite is scene-coherent; normalization after compositing.
- **Context (stack `foveal_v1_ctx`):** the SHARP 96² window at the partner's fovea, bilinear-
  resized to 224 for the tokenizer. Slot convention as E13: D_read A→B pairs the member's slot-0
  (event_A) with the tokenizer's slot-1 (ctx_B); symmetrized with B→A.
- **Determinism:** stratum keyed by manifest ref only; positions keyed by (ref, stack); md5-seeded
  rngs (`sslgap/data.py: foveal_boxes`). Every extraction pass yields byte-identical views ⇒ the
  target array is shared zoo-wide AND no member has a same-pass draw advantage (removes E13's mae
  intimacy asymmetry; mae still sits out of the ranked set — self-target objection stands).
- **Overlap strata** (assigned over the 10k pairs100 refs): **far** IoU=0, 50% (A resampled until
  a disjoint partner exists — centered 96² foveas admit none on 224; A-marginal is edge-biased in
  this stratum, declared); **near** IoU ∈ (0,.5], 30%; **copy** identical box, 20% (re-encoding
  contrast cell). Achieved IoU distribution lands in §Numbers.
- **Declared off-distribution:** no zoo member trained on degraded surrounds; the shift is
  member-symmetric and is PIVOT's deployment condition. Raw D_read LEVELS vs E13 levels reported
  per member so shift cost is visible; ranks carry the claim.
- **Known channel property (seen in previews, declared):** the surround legitimately carries the
  low frequencies of the target region; texture-dominated scenes make targets partially
  recoverable from surround statistics. The randinit gates + copy-vs-far contrast separate cheap
  low-frequency recovery from ranking signal.
- Previews: `results/figures/e14/preview_foveal_v1.png` (approved; `_f64` variant viewed and
  REJECTED for scope — f96 is the only channel, no size secondary).

## Estimator (E13 machinery + T3 promotions; per-stratum)

- **Per-stratum ridge:** splits 60/20/20 stratified by class WITHIN stratum, seed 0, identical
  indices zoo-wide (runtime assert on labels). Far ≈ 5000 → 3000/1000/1000; near ≈ 3000; copy
  ≈ 2000. Ridge from per-dim standardized event-h onto the RFF sketch of the standardized frozen
  tokenizer descriptor; symmetrized; **dof-matched dof*=256 is PRIMARY (E13-T3b), val-selected α
  from the E13 grid reported as secondary**; edge-of-grid flagged.
- **Sketch:** RFF as E13; σ_med per tokenizer from 2048 pooled train-fold ctx descriptors
  (stratum-agnostic — the ctx marginal is stratum-independent), recorded. Scope clarifications
  (pre-numbers): per-dim standardization (both sides) uses stratum-train stats; RFF W/b per-cell
  seed shared across strata and tokenizers. Achieved strata (deterministic, verified pre-launch):
  far 5034 / near 2985 / copy 1981; near-IoU mean .19, max .50.
- **Sweep (far stratum only):** D_m ∈ {256, 1024} × σ ∈ {½, 1, 2}·σ_med + raw-descriptor arm.
  near/copy computed at the primary cell only. **Primary cell: far, D_m=1024, σ=1·σ_med, T=mae,
  dof-matched.**
- **Tokenizers:** T = `in100.mae.s0` h.gap (THE target, PIVOT §5.2 reconstruction-based; E15
  would freeze this). T = `in100.randinit-s0` (leakage/content control, full primary-cell
  repeat). **T = `in100.dino.s0` teacher h.cls (specificity control vs the deflationary
  "distillation from any strong teacher" reading; primary cell only, descriptive).**
- **Target audit before any ranking read** (E13 list): sketch spectrum/effrank, kernel
  concentration, across-zoo dynamic range vs bootstrap se, kmeans-100 NMI on top-128 PCs —
  per tokenizer.
- **Baselines & targets:** E13's locked battery baseline table reused verbatim (per-run
  properties, unchanged); ranking targets converged `linear_raw_v2` / `knn_v1_k200` at declared h.
- **Ranked zoo:** E13's n=20 table with **E13-T3(a) operative: deitlite = unranked anchor →
  primary reads on n=19**; with-anchor Spearman reported descriptively. Gates: randinit
  (member), mae (T=randinit read only). D_kern: primary-cell descriptive only (E13-T2 kill
  stands; retained as re-metrization flag).

## Pre-registered predictions (locked 2026-07-12, before numbers)

- **E14-P1 (gates).** Randinit-member separation ≫ trained spread (E13: +6.35 sd, 0 inversions);
  dynamic range ≥ 3× median bootstrap se; T=mae target NMI below distillation-grade.
- **E14-P2 (primary claim).** Far-stratum dof-matched D_read ranks n=19: |ρ| ≥ .44 (perm p<.05)
  on lin AND beats the best battery baseline in |ρ| on BOTH probe columns — the E13-T1 bar
  restated on the declared channel.
- **E14-P3 (channel sanity, per-model).** D_read(copy) < D_read(near) < D_read(far) for every
  trained member.
- **E14-P4 (the caveat-killer).** ρ_far(lin, dof-matched) ≤ −.5 — comparable to E13's aug-view
  primary (−.72), i.e. E13-T1 was not an overlap artifact. Downgrade trigger: far fails P2 while
  copy ranks at |ρ| ≥ .44 ⇒ E13-T1 was re-encoding-driven, premise reading downgrades.
- **E14-P5 (e12 family).** Far-stratum D_read ranks within the e12 arms (|ρ| ≥ .5) where every
  marginal statistic is blind.
- **E14-P6 (tokenizer specificity, descriptive).** T=dino target NMI ≫ T=mae's (semantic flag
  fires); directional lean: T=dino ranking DOES NOT BEAT T=mae — if it matches/beats it, the
  conditional-context reading weakens toward "any strong teacher" distillation.
- **E14-P7 (machinery, recorded not gated).** With overlap controlled, RFF sketch > raw-descriptor
  arm by more than E13's slim margin (PIVOT prediction 10's fair test).

## Kill criteria

- **E14-K1.** Target-audit failure at the primary cell ⇒ construction failure, no ranking
  language; ONE declared re-roll at surround factor ×2 (flagged REGISTERED-FALLBACK), else stop.
- **E14-K2 (rung-1 kill).** No far-stratum cell in the reduced sweep reaches |ρ| ≥ .44
  (perm p<.05) on either probe column ⇒ PIVOT's premise does not transfer to its own declared
  channel ⇒ E15's case reopens for discussion before any training spend.
- **E14-K3.** T=randinit ranks comparably to T=mae ⇒ leakage/low-level content, not the trained
  descriptor; conditioning-artifact reading opens.

## Extraction plan (H100 granted; G-wave chains untouched)

22 event jobs (20 zoo-table members + randinit + mae) with `foveal=event`; mae/randinit/dino get
`foveal=both` (ctx in the same job). `do_eval=false do_pairs=false`; ckpt path + adapter +
random_init/seed read from each run's stored `ckpt_provenance`. Pre-launch checks (done): loader
is shuffle=False (rows = manifest order, E13's alignment assert applies) and foveal INPUT tensors
hash byte-identical across separate processes (20 refs × both views × both modes) — feature bytes
across jobs are NOT asserted (GPU matmul nondeterminism; not required — the shared-target property
lives at the input level). Dry-run: pairs1 manifest (100 images), one member event + one tokenizer
both, isolated `in100.e14dry.*` run_ids (purged after), then fleet via singleton slots
`h100-slotA/B`. Store: ~30 MB/member fp16 — negligible (D-005 fine).

## Addendum — no-fovea null (REGISTERED-LATE, Berker-required 2026-07-13; locked pre-numbers)

Berker's challenge on the far stratum: "far becomes almost predict-sharp-from-blurred — I don't
understand the contribution of keeping a non-intersecting fovea; predicting the sharp mae from
fully blurred would be as hard." Control: **`blur_v1`** events — the same deterministic base
scene fully ×4 down/up, NO sharp fovea (A = B by construction, location-free); ctx targets,
strata, and folds byte-identical reuse of the foveal run. Far primary cell only, T=mae + T=dino,
dof-matched primary, same n19/n18 zoos. Fleet 62248967+21 (blur events for all 22 members);
scoring `e14_foveal_rung1.py --blur` → `results/diag/e14_{distortion,rank_corr}_blur.csv`.

Decision rules (locked): **B1 (levels)** — if median trained far D_read_dof(blur) − (foveal)
< 2× median bootstrap se (.0055), the fovea contributes nothing measurable to far prediction for
frozen non-PIVOT checkpoints (Berker's null); my registered lean: it rises by more than that.
**B2 (ranking)** — if T=dino far ρ_lin under blur stays within .05 of −.794, the far ranking
signal is low-frequency scene gist, event construction irrelevant at this rung (deflationary);
if |ρ| drops ≥ .1, the foveal event is load-bearing for the ranking. T=mae far ρ (−.253, already
dead) read descriptively. Frozen-model caveat declared: zoo members never trained to integrate
sharp+degraded inputs, so a null here does NOT establish that the fovea is useless for
PIVOT-TRAINED models — it bounds what rung-1 can claim about the event construction.

## Numbers land below this line as they arrive; AGREED TAKEAWAY only after joint discussion.

### Full pass — 2026-07-13 (scoring 62246451; dry 62246318; fleet 62246002–023, 22/22 done)

CSVs: `results/diag/e14_{distortion,rank_corr,target_audit,gate,levels,strata}.csv`.

**Gates (P1, far/mae primary):** randinit member +3.39 sd above trained mean, 0 inversions
(E13: +6.35); dynamic range 7.8× median bootstrap se (bar 3×); T=mae target NMI .372/.377 (A2B/
B2A; E13 aug-view was .24). Gates pass; margins thinner than rung-0.

**Ranking, dof-matched d_read (full19, lin/knn):**

| cell | rho lin | p | rho knn | p |
|---|---|---|---|---|
| T=mae far (PRIMARY) | −.253 | .30 | −.193 | .43 |
| T=mae near | −.404 | .09 | −.335 | .16 |
| T=mae copy | **−.526** | .022 | **−.546** | .017 |
| T=randinit far (control) | +.154 | .52 | +.109 | .65 |
| T=dino far (control, n=18) | **−.794** | .0005 | **−.664** | .0033 |
| battery best (kurt_topeig) | −.546 | .016 | −.426 | .068 |

T=mae far sweep is FLAT and dead everywhere (best |ρ| .288 across all 6 RFF cells + raw arm; raw
= −.268/−.210 ≈ RFF). Paired n=18 subset: T=mae −.226 vs T=dino −.794 on identical members.
Sub-zoos at far/mae: lejepa11 −.654 lin (p=.033) / −.291 knn; nonlejepa8 +.333/+.667 (wrong-signed,
n=8). Target audit: sketch effrank mae ~32.5, dino ~102, randinit ~5.3; NMI mae .37 / dino .47 /
randinit .31.

**Levels (`e14_levels.csv`):** per-member far < near for 18/20 and copy < near for 17/20 (levels,
not ranking power); ALL members read LOWER distortion on foveal-far than on E13's aug-view primary
(mean −.19; the deterministic, photometrically clean channel is more predictable than heavy-aug
views). Instrument note (mechanical): cross-stratum LEVEL comparisons carry unequal per-stratum
n_tr (2982/1754/1147) — pre-registration did not equalize; within-stratum rankings unaffected.

**Pre-registered verdict mechanics (interpretation = joint discussion, not here):**
- **P1 PASS** (all three gate conditions; margins noted above).
- **P2 NOT MET** — primary cell −.253 (n.s.), loses to battery best on both columns.
- **P3 NOT MET** — full monotone 2/20 (copy<near 17/20 holds; near<far 2/20 fails; see instrument
  note).
- **P4 NOT MET and its DOWNGRADE TRIGGER FIRES** — far fails P2 while copy ranks at −.53/−.55
  (≥.44, p<.03 both): per the locked wording, "E13-T1 was re-encoding-driven, premise reading
  downgrades" — scoped by the T=dino control below.
- **P5 PARTIAL** — e12-family far/mae ranks on linear (−.654, p=.033), not knn (−.291).
- **P6 LEAN FALSIFIED (inverted)** — T=dino does not merely match T=mae, it dominates
  (−.794/−.664 vs −.253/−.193; paired n=18 −.79 vs −.23) and beats E13's aug-view primary
  (−.72/−.65) and every battery statistic — the strongest zoo ranker measured in this project,
  on the far channel, while T=randinit is dead (+.15). Semantic flag fires mechanically
  (NMI .47 vs .37; effrank 102 vs 32).
- **P7 NOT MET** — RFF ≈ raw again (−.253 vs −.268).
- **K1 not fired** (gates pass). **K3 not fired** (randinit control dead → content is the trained
  descriptor). **K2: fires for the DECLARED construction** — no T=mae far cell (the pre-registered
  reduced sweep) reaches |ρ| ≥ .44 on either column ⇒ the premise as PIVOT declares it
  (reconstruction-based target) does not transfer to the declared channel; the T=dino far control
  clears the bar (−.79) but sits outside the declared sweep. Per K2: E15's case reopens for
  discussion before any training spend.

## AGREED TAKEAWAY

*(Substance agreed in the 2026-07-13 discussion — Berker: "i think we already discussed and i am
satisfied with your explanations (apart from the null experiment i required), you can fill the
takeaways as we already discussed"; wording delegated, veto open. Scoped PRE-addendum: the
registered no-fovea null (B1/B2 above) can amend T1/T2 when it lands. Full row texts in DECISIONS
E14-T1…T3.)*

- **E14-T1 — the cross-family information claim does not transfer to the declared channel with
  the declared target class; E13-T1's cross-family support was substantially overlap-carried.**
  On the overlap-controlled foveal channel, D_read with the reconstruction tokenizer has no
  cross-family ranking power in any far cell (best |ρ| .29, n.s.; loses to kurt_topeig
  −.55/−.43), and ranking power is overlap-graded — copy −.53/−.55 (p<.03) > near −.40 > far
  −.25 — firing the pre-registered downgrade trigger: E13-T1's "beats every battery statistic"
  clause is now scoped as aug-view/overlap-assisted and does not extend to the declared channel.
  WITHIN family (e12 dose arms + lane, method/affinity constant) far/mae still ranks converged
  linear (−.65, p=.03; knn n.s.) — the family-internal dose-response signal survives its third
  instrument. Gates clean (randinit member +3.4 sd / 0 inversions; range 7.8×; T=randinit
  tokenizer dead ⇒ content = trained descriptor). K2 fired for the declared construction ⇒ per
  D-029 staging, E15 requires an explicit new decision. Scope: n=19, IN-100 probes, seed 0, one
  frozen mae, f96/×4, pre-blur-control.
- **E14-T2 — cross-family, reconstruction-target proximity reads as method affinity, not
  quality; a semantic target restores ranking but is mechanism-capped (agreed reading).** ijepa
  (masked-prediction training) is the BEST mae-far predictor (R² .384) while second-worst on
  probe; byol/vicreg/simclr shift the same way; under T=dino ijepa returns to mid-pack and the
  ranking straightens to −.79/−.66 (n=18) — the strongest label-free zoo ranker measured in this
  project — with the dino target PREDICTED WORSE (median R² .24 vs .33, smaller spread):
  predictability level and ranking power dissociate. Agreed position (Berker): the dino cell is
  expected and neither rehabilitates E1-style selection nor indicts the mechanism — proximity to
  a strong teacher correlates with quality but is capped as a training signal; PIVOT's mechanism
  bet (reconstruction-flavored target + augmentation/sufficiency pressure) is not tested by
  frozen-model ranking. What the contrast establishes: at zero overlap, the cross-family rank
  signal lives in semantic context content, not reconstruction-flavored content.
- **E14-T3 — instrument lessons.** (a) The deterministic declared channel is the
  better-conditioned instrument: E13's aug-view target carried an irreducible independent-draw
  noise floor (trained held-out R² 8–14%, level sd .016) vs E14 far R² 22–38%, sd .043 (2.6×),
  dynamic range 7.8× vs 6.3× — aug-view D_read was noise-dominated, which also compressed its
  scatter axis. (b) Per-member cross-stratum levels are not a channel-difficulty meter: P3
  monotonicity 2/20 (near carries the smallest-n_tr confound, declared post-hoc; randinit shows
  the opposite gradient); within-stratum rankings unaffected. (c) "far" = IoU 0 INCLUDES
  edge-adjacent boxes — zero-overlap ≠ distant; center-distance recomputable as covariate.
  (d) dof-matched readout again ≈ val-selected (median R² .325 vs .327): capacity not binding;
  E13-T3(b) promotion held.
