# E12 — The moment floor: what shape regularization at h is for

**Status: CLOSED — takeaways E12-T1…T9 in §AGREED TAKEAWAY (T9(i) OPEN per D-037); main arms +
F/G/H waves all scored. Folded to current truth 2026-07-20 (D-053): superseded layers are
lineage notes below; the pre-fold text is in git history. Originally PRE-REGISTERED (Berker
sign-off 2026-07-11: "rest got my pass you can start!", with the compute amendment "2 h100 is
capped but gpu partition will likely to be free"); predictions locked in this file BEFORE any
arm produced a number. First arm of OUR METHOD (direction A of the 2026-07-11 design
discussion); vehicle = M4's parked one-branch arm (LeJEPA lane). Ledger row: D-026 (archive).**

## Question

What marginal/shape regularization at the probed space is useful, and why. Invariance is the
declared, largely-settled axis (G is given; the head absorbing part of it is a measured fact,
E01-T8/T12). The open axis is the marginal term: DINO works consistently for unexplained reasons
behind a prototype buffer; LeJEPA claims isotropy that our audit undermined at the probed space
(E01-T4; E10-T4, toy-scoped). **Thesis under test: the useful part of shape regularization at the
probed space is first-two-moment calibration (anti-degeneracy + scale/anisotropy control); higher-
order shape should be left free for semantics and audited, never trained.**

The decomposition the arms realize: R4c [T] says the sliced-CF term at K≈512 is mostly a moment
meter, *except* the anti-CLT channel (R4c-6), which reads cluster structure K-independently — the
plausible mechanism of E10-T3's measured class-structure suppression. So "Dlr at IN-100" replicates
the toy datum but does not instantiate the thesis; the thesis-carrying arm needs a floor **provably
blind to clusters**: an explicit second-moment calibration term (Gaussian-moment KL form, below).

## Frame (all arms)

IN-100 canonical CMC split (`~/data/imagenet100`, D-002) · ViT-S/16@224 (`frame=in100_vits16`,
100 ep, D-004 epochs-matched) · M2 conventions: bs 128, num_workers 10, num_classes 100 · house
optimizer (D-018): AdamW wd 5e-2, warmup 10 ep, cosine eta_min 1e-5 · house hygiene (D-016):
grad_clip 1.0, per-step grad_norm + per-term logging, INCIDENT trigger (>100× running median) ·
seed 0 only (D-024 spirit; seed replicates deferred) · quarter-cadence ckpts + best + last ·
wandb online · LeJEPA lane: V=4, emb_dim 512, proj_dim 16, drop_path 0.1 (matched to
`in100.lejepa.s0`). h = the 512-d embedding (D-003v2: LeJEPA h = z.embed).

**Stability package** (framework §4 premises p1–p4, from E10-T1b/T2): (p1) spectral normalization
of the projector's Linear layers where a projector exists (torch parametrization; BN layers remain
unnormalized — declared caveat: the package bounds the Linears' spectral norms, not the full head
Lipschitz constant; σ_min(g)/eff-rank(g) stay monitored via D-015); (p2) one-shot data-dependent
init calibration of the embedding (`embed_calib`, the E10 mechanism); (p3) λ set by equal-pull at
init on shared (encoder) parameters for the arms whose regularizer moves to h — measured
pre-launch on the real first batch at seed 0, **launched with the measured value whatever it is**
(no mid-flight discretion; value recorded here + `results/diag/e12_equal_pull.csv`); (p4) peak lr
3e-4 = the declared package lr (the healthy-datum value), with the kill rule live.

## Arms (4 runs; run_ids `in100.lejepa.s0.e12{a1,a2,a3,c1}`; table = the FINAL, amended arms)

| arm | loss (final, post-amendment) | projector | λ_h | lr | init calib |
|---|---|---|---|---|---|
| **A1** package-Dlr (replication) | both terms at h (depth-0), sliced-CF | none (Identity) | 0.02 fixed (Dlr-faithful) | 3e-4 | yes |
| **A2** C1 + SIGReg@embed | 0.98·inv + 0.02·SIGReg@proj + λ_h·SIGReg@embed | depth-3, spec-normed | 0.0257 (equal-pull) | 3e-4 | yes |
| **A3** C1 + MomentFloor@embed (thesis cell) | 0.98·inv + 0.02·SIGReg@proj + λ_h·MomentFloor@embed | depth-3, spec-normed | 0.4775 (equal-pull) | 3e-4 | yes |
| **C1** buffered control (matched package) | all losses at proj.out (as shipped) | depth-3, spec-normed | — (z λ 0.02 shipped) | 3e-4 | yes |

Existing reference rows (no new compute): `in100.lejepa.s0` (shipped-style M2 lane, lr 1e-3, no
package — C1 vs it reads recipe sensitivity for free), `in100.randinit`, `in100.deitlite.s0`, rest
of the M2 matrix. **Declared confound, accepted at sign-off:** A2/A3 differ from C1 in λ rule
(equal-pull vs fixed 0.02) as well as placement; the measured λ is recorded and its distance from
0.02 bounds the confound. C1 doubles as the framework's §6.4 spectral-norm-rerun datum. A1's
alignment sits at h (depth-0) — replication fidelity, not method identity.

**The moment floor (A3's term), declared.** Per step, over the n = bs·V = 512 view-embeddings:
draw a fresh random d′-dim orthonormal basis Q (d′ = 128; n = 512 samples make the full 512-d
empirical covariance rank-deficient, so the KL is computed on a random subspace per step —
coverage across steps, same fresh-slice logic as SIGReg), compute
KL(N(Qᵀμ̂, QᵀΣ̂Q) ‖ N(0, I_{d′})) / d′ = (tr + ‖mean‖² − d′ − logdet)/2d′, with eigenvalue floor
ε = 1e-4 inside the logdet, in fp32 outside autocast. This is R1's rate+calibration term under a
Gaussian read of q_h — a pure function of batch mean and covariance, hence blind to clusters and
all higher-order shape by construction; the logdet is the anti-degeneracy barrier. Estimator
settings (d′, ε, n, per-dim normalization) are declared here and fixed.

## Lineage — the lazy-projector amendment (2026-07-11, pre-full-launch; folded 2026-07-20)

The arms table above is the AMENDED design (approved "yes please do that"); the original A2/A3
MOVED the regularizer to h, with proj.out carrying only `inv`. **Berker's catch, before any full
arm ran:** that loss admits alignment-free minima — spectral norm pins σ_max but leaves BN-γ
shrinkage open, and `inv` is scale-dependent, so with the z-side scale pin removed the projector
zeroes `inv` by shrinkage without aligning anything (alignment is a sufficiency surrogate only
JOINTLY with a marginal term in the same space; the same hole existed in the v2.0 draft's
Appendix-E method — theory-note owed). **Measured demonstration** (2-ep smokes; wandb
`e12a{2,3}smoke` kept as the failure record): smoke `inv` 0.098→0.0011 (A2) / 0.119→0.0038 (A3)
in 2 warmup epochs — 15–50× below the shipped lane's end-of-100-ep value (0.053) at ~20% of peak
lr — with proj.out across-image std 0.016/0.032 and within-image (view) std ~1.9× LARGER than
across: the arms were training floor-only encoders. The amendment makes the h-term the ONLY
factor (z-side byte-identical across A2/A3/C1); λ_h values carry over unchanged (same forward
graph at init). Rejected: moments-only guard at proj (re-opens a z-side confound vs C1);
scale-free/cosine alignment (deviates from the shipped functional); dropping the projector
(collapses A2 into A1). Predictions/kills survived with wording changes only ("moved to h" reads
"added at h"; the pure moved-placement cell remains A1).

## Pre-registered predictions (locked 2026-07-11, before numbers; anchors in brackets)

- **P1 · Stability.** All four arms complete 100 ep with zero kill-rule triggers; A1–A3 grad-norm
  profiles comparable to C1. [E10-T1b/T2; framework-E7(iii). Known risk, stated: toy arm-B
  topology (align at proj + SIGReg at embed) ignited even at lr 3e-4 without the full package;
  A2 is its first full-package (incl. spec-norm) test — if A2 ignites, that is the finding.]
- **P2 · Suppression ordering (load-bearing).** Class-aligned structure at h (η², between/total,
  label-NMI on eigendirections — the E10-T3 instrument set; kurt cells supporting only per
  E01-T11) orders: **C1 (free h) ≈ A3 (moment floor) > A2 ≥ A1 (CF-constrained h)**. A3 holding
  C1-level class alignment while carrying calibrated moments = the thesis stands; the CF arms
  replicate the toy tax via the anti-CLT channel. [E10-T3/T4 toy; R4c-6.]
- **P3 · Probes.** Converged linear_raw_v2 + knn_v1 (D-020) at h: A3 within noise of C1; A1/A2 at
  most a small tax (≤2 pts predicted); everything far above randinit. A null (no tax anywhere at
  IN-100) scopes E10-T4's suppression as toy-scale — reported as such.
- **P4 · Audit health.** A1–A3 h reads calibrated first two moments (Stage-3-style standardization
  scales ≈1; no dead dims); sliced-cell shape residue (Rule-8 three-reference decomposition)
  small-positive for A1/A2, largest-and-class-aligned for A3 and C1's free h; C1 replicates the
  E01-T4 pattern (constraint lives only where imposed: proj.out tame, h uncalibrated). k̂ (D-025)
  reported per cell as structured-direction count.
- **P5 · ν=∞ pilot double-duty.** Varimax sparsity + coordinatewise-kurtosis symptoms (framework
  §6.2) are read on A3/C1 h checkpoints and recorded — no action this pass; feeds direction B.

## Kill criteria

- **K1** any ignition/INCIDENT → arm stops; per-step wandb curve forensics at onset before any
  rerun (house default).
- **K2** method-arm (A2/A3) probe tax > 3 pts vs C1 at matched budget.
- **K3** A3's floor fails to floor: standardization scales far from 1, or rank/logdet degeneracy
  at h.
- **K4 (thesis-kill)** A3 suppresses class alignment as much as A1/A2 → the tax is not
  cluster-channel-borne; thesis reworked before any further arm.

## Deliberately out of this pass

The ν-fork (direction B waits on P5), the adversarial bank, audit "pass" language, seed
replicates, IN-1k anything. §6.0 defect-injection calibration gates external quoting of honesty
cells, not training.

## Compute plan (per sign-off)

A1/A2 on the H100 singleton slots (`h100-slotA/B`, cap 2 respected); A3/C1 on the `gpu` partition
(L40S|A40|A100). ~1 H100-day-equivalent per arm; 8h walltime chunks with requeue+resume. Order of
operations: CPU dry-run (all arms: step + backward + arch-rebuild round-trip) + equal-pull
measurement job → 2-ep GPU smoke of the two new code paths (A2 spec-norm, A3 floor) → full launch.

## Equal-pull measurement record (filled at launch, mechanical — not interpretation)

- λ_A2 (equal-pull, encoder-shared grads, post-calib first batch, seed 0): **0.0257**
  (g_inv 4.04, g_sigreg 153.0; ×1.29 from fixed 0.02 — the A2↔C1 λ-confound is bounded there)
- λ_A3 (same protocol): **0.4775** (g_inv 4.04, g_moment_kl 4.42 — the moment-KL gradient is
  naturally alignment-scale, so equal-pull lands near ½; launched verbatim per the locked rule)
- Job 62211218 (A40), 2026-07-11; CSV: `results/diag/e12_equal_pull.csv`. Dry-run job 62211217:
  all 4 arms PASS (step+backward, strict arch round-trip incl. spec-norm keys, calib, moment_kl).

## F-WAVE — mechanism-attribution follow-up arms (D-027; Berker 2026-07-11: "prepare the next arm
## codes and queue them … launch many … theory relevant … bookkeep which arm tests what")

**Honesty flag:** these arms are registered AFTER observing the A3-vs-C1 online-monitor gap
(~7–8 pts, roughly constant, ep20–52) but BEFORE any converged probe / battery / P2-geometry
number exists — REGISTERED-LATE w.r.t. the monitor, pre-registered w.r.t. every scoring
instrument. **Purpose: exploratory-diagnostic (mechanism attribution for the A3 tax + audit-power
tests), NOT beat-LeJEPA arms** — the deliverable is the price list for marginal regularization at
h; f3 doubles as the refined-thesis method cell if it lands tax-free.

**Candidate mechanisms for A3 < C1** (from the theory): **M1** over-weighted floor (equal-pull at
init ≠ equilibrium pull; generic constraint-budget tax) · **M2** anisotropy-IS-semantics (class
structure lives in the covariance — between-class scatter = top-eigenvalue anisotropy, E10-T3 /
E01-T11; Σ→I equalizes signal and noise directions; note Stage-3 of our own protocol treats scale
as *nuisance standardization*, so the loss-side equality was stronger than our own audit demands)
· **M3** transient optimization interference (gradient competition on the trunk) · **M5**
sliced-logdet estimator noise. Zero-compute discriminator already available: under M2-as-whitening
the tax should load on kNN ≫ converged-linear (kNN is not affine-invariant; population linear
separability is) — read from A3's probe pair, no new arm needed.

| arm (tag) | h-term @ λ_h | tests | prediction if that mechanism is real |
|---|---|---|---|
| f1 `e12f1` | moment-KL @ 0.1 | M1 dose axis (with A3 @.478, f2 @.02) | tax decays ~monotonically in λ; calibration health persists via barrier steepness. If tax instead tracks *achieved* calibration level (end `h_moment_kl`), points to M2 |
| f2 `e12f2` | moment-KL @ 0.02 | M1 dose axis, low end | as f1; if tax ≈ 0 here with calibration retained, method survives with a corrected weight rule (p3 revision: measure at burn-in, not init) |
| f3 `e12f3` | **SpectralFloor** @ 0.4775 (matched to A3) | **M2 — the refined thesis**: one-sided relative eigenvalue barrier (τ=0.01, d′=128) + scale/mean pin; anisotropy above floor untouched | probes ≈ C1 within noise + an auditable non-degeneracy certificate ⇒ method deliverable. If even this taxes ⇒ *any* spectral demand at h costs — strong negative for D5-at-h in any form |
| f4 `e12f4` | **per-slice-standardized SIGReg** @ equal-pull (fallback 0.1) | isolates the cluster/anti-CLT channel (moment channel killed by standardization — R4c(d), framework-E7(i), untested anywhere) | if E10-T3 suppression is cluster-borne: class-alignment (NMI/mode metrics) suppressed more per unit pull than A3, with less anisotropy/probe-SNR damage. If the term never binds (CLT + rank-k attenuation at K=512): empirical falsification of the E7(i) rescue at scale — a §6.0/OP-h power datum |
| f5 `e12f5` | **DiagMomentFloor** @ equal-pull | the affine-absorbable calibration cell: encoder can comply via diagonal rescale+bias of the emb Linear (information-free) ⇒ any tax ≈ pure optimization-interference floor (M1/M3); **A3 − f5 isolates the whitening/decorrelation component** (M2). Doubles as the estimator-noise control (no subspace, no logdet) vs M5 | tax(f5) ≪ tax(A3) ⇒ M2; tax(f5) ≈ tax(A3) ⇒ interference/weight |
| f6 `e12f6` | moment-KL @ 0.4775, **enabled at ep10** | M3 transient-vs-equilibrium (§4-p2's own "burn in the standardization map before enabling") | tax(f6) ≪ tax(A3) ⇒ damage is opening dynamics; ≈ ⇒ equilibrium competition |

The content 2×2 this wave completes: moment-pressure × cluster-pressure at h = C1 (0,0) · A3
(1,0) · f4 (0,1) · A2 (1,1) — whatever the outcomes, the decomposition of "which component of a
marginal constraint taxes what" is the result.

**Degenerate-case audit (the lazy-projector lesson, per new term).** All f-arms retain the
shipped SIGReg@proj (no lazy-projector hole by construction). f3: scale-inflation gaming closed
by the trace pin; "junk-dim" compliance allowed by design (certificate is spectral, not
semantic); eigvalsh on jittered PSD. f4: dead-slice division guarded (std clamp; gradient
vanishes at exact death — the term is not the anti-collapse device, C1's z-term and f-context
are); may be near-satisfied at init by CLT (binds only as clusters form — intended). f5:
compliance via diagonal affine is the *point* (see table); per-dim log barrier keeps per-dim
anti-collapse. f6: no new loss. λ rules pre-declared: f1/f2/f6 fixed; f4/f5 equal-pull measured
at init (launched verbatim); f3 expected DEGENERATE at init (barrier unbound) → declared fallback
0.4775; degeneracy flag = λ>0.9 or g_reg<0.01·g_inv (printed by equal_pull.py).

**Frame:** identical to the main arms (IN-100, package, seed 0, 100 ep, gpu partition —
uncapped per Berker; 1 primary + 3 afterany chain links per arm). Scoring: same instruments as
P2–P4 + the probe-pair (kNN vs linear) M2 signature; discussion-first, no takeaways here.

## F-wave equal-pull record (mechanical fill at launch)

- λ_f4 (sigreg_std): **0.0706** (g_reg 53.2 — standardization leaves real init gradient; launched
  verbatim) · λ_f5 (moment_diag): **0.552** (g_reg 3.28, near A3's 0.478 as the same-geometry
  argument predicted) · λ_f3 (spec_floor): measured 0.590 NON-degenerate (the barrier binds
  mildly at init — random-init correlation spectra do dip below τ), but per the pre-declared
  rule f3 LAUNCHES AT THE MATCHED 0.4775 (content isolation vs A3 at identical weight);
  measurement recorded here. Job 62217085; CSV results/diag/e12_equal_pull.csv. Dry-run
  62217084: all 8 configs PASS incl. f6 gate-hold.

## Numbers land below this line as they arrive; AGREED TAKEAWAY only after joint discussion.

### First scoring pass — MAIN ARMS at ep100 (2026-07-11; raw numbers + MECHANICAL rule outcomes
### only; interpretation pending discussion)

**Sources:** monitor finals (train logs) · converged probes `results/probes/in100.lejepa.s0.e12*.ext.csv`
· class-alignment `results/diag/e12_class_align.csv` (committed E10-T3-style instrument,
`experiments/e12_class_align.py`) · battery CSVs landed (not yet folded in).

**Probes at h (= z.embed), converged (D-020):**

| arm | linear_raw_v2 | knn_v1_k200 | knn_v1_k20 | monitor best |
|---|---|---|---|---|
| A1 package-Dlr | .4732 | .3754 | .4006 | .4296 |
| A2 C1+CF@h | .4310 | .3364 | .3254 | .3932 |
| A3 C1+momentKL@h | .5070 | .3250 | **.2500** | .5652 |
| C1 control | **.6456** | **.5306** | **.5652** | .6404 |
| M2 lane (ref) | .6022 | .5240 | .5530 | .5944 |

**Class alignment at h (train500):**

| arm | between/total | mean η² top16 | kmeans100 NMI | effrank |
|---|---|---|---|---|
| A1 | .4209 | .4415 | .4267 | 22.1 |
| A2 | .1187 | .1658 | .3047 | 76.3 |
| A3 | **.0426** | .0587 | **.0529** | **274.8** |
| C1 | .4821 | .4943 | .5271 | 22.0 |
| M2 lane | .5123 | .5007 | .5213 | 17.9 |
| randinit null | .1093 | .0595 | .0911 | 3.7 |

**Mechanical outcomes vs the locked predictions:**
- **P1 HELD** — all four arms trained 100 ep, zero INCIDENT (fuse window passed without event).
- **P2 FALSIFIED** — predicted C1 ≈ A3 > A2 ≥ A1; measured **C1 > A1 > A2 ≫ A3**: the thesis
  cell is WORST, its class alignment at/below the randinit null.
- **P3 / K2 FIRED** — A3 linear tax −13.9, A2 −21.5 (≫ 3-pt kill bar); A1 −17.2.
- **K4 (THESIS-KILL) FIRED** — A3 suppresses class structure more than A1/A2, not equal-or-less.
  Per the locked rule: thesis reworked before any further arm — the F-wave (pre-registered
  BEFORE these numbers) is that rework's probe set; no arms beyond it until discussion.
- **P4 split** — the floor achieved its own declared aim (A3 h: effrank 274.8/512, calibrated,
  no degeneracy): the constraint worked; the AIM is what failed.
- **Pre-registered whitening discriminator (F-wave block) matched by A3's probe pair:** kNN tax
  (−20.6 @k200, −31.5 @k20) ≫ linear tax (−13.9), and A3 is the only arm whose kNN worsens as k
  shrinks — local-neighborhood damage, the M2/whitening signature. A2's damage is uniform across
  probe types (different profile).
- **E10-T4 travel check (A1 = the like-for-like cell):** toy suppression (NMI .246 vs buffered
  .568, −57% rel.) is DILUTED at IN-100 (A1 .427 vs C1 .527, −19% rel.) — direction survives,
  magnitude does not; consistent with the slice-CLT caveat (E01-T11: anti-CLT/cluster channel
  weakens as class count grows). Raw observation for the E10-T4 travel discussion.
- **P5 pending** (Varimax symptom read on A3/C1 not yet run). Battery cells not yet folded in.

*(The proposed reading that stood here was resolved into E12-T1/T5 — see AGREED TAKEAWAY.)*

### F-wave scoring — ALL ARMS at ep100 (2026-07-12; raw + mechanical outcomes; discussion owed)

All at h = z.embed; converged probes (D-020); class-align = `results/diag/e12_class_align.csv`
(12 rows); sorted by converged linear:

| arm | h-term @ λ_h | lin_v2 | knn200 | knn20 | B/T | kmeans100-NMI | effrank | monitor |
|---|---|---|---|---|---|---|---|---|
| **f2** | moment-KL @ .02 | **.6578** | **.6056** | **.6156** | .157 | **.533** | 217.2 | .6758 |
| C1 | — (control) | .6456 | .5306 | .5652 | .482 | .527 | 22.0 | .6404 |
| f1 | moment-KL @ .10 | .6338 | .5718 | .5736 | .096 | .440 | 252.3 | .6560 |
| f5 | diag-moment @ .552 | .6322 | .5074 | .5396 | .357 | .504 | 6.8 | .6062 |
| f3 | spec-floor @ .478 | .6312 | .4928 | .5254 | .441 | .480 | 9.7 | .6376 |
| f4 | std-CF @ .071 | .6136 | **.2724** | .3722 | .340 | .161 | **2.2** | .5838 |
| M2 lane | ref (lr 1e-3) | .6022 | .5240 | .5530 | .512 | .521 | 17.9 | .5944 |
| A3 | moment-KL @ .478 | .5070 | .3250 | .2500 | .043 | .053 | 274.8 | .5652 |
| f6 | moment-KL @ .478, ep10 | .5056 | .3518 | .2564 | .040 | .072 | 308.0 | .5604 |
| A1 | Dlr (both at h) | .4732 | .3754 | .4006 | .421 | .427 | 22.1 | .4296 |
| A2 | CF @ .026 | .4310 | .3364 | .3254 | .119 | .305 | 76.3 | .3932 |
| null | randinit | — | — | — | .109 | .091 | 3.7 | — |

**Mechanical outcomes vs the F-wave predictions:**
- **f1/f2 (dose):** tax monotone in λ as M1 predicted — WITH AN INTERIOR OPTIMUM: **f2 beats C1
  on every probe (+1.2 lin, +7.5 knn200, +5.0 knn20) and beats the M2 lane by +5.6/+8.2** — the
  low-dose floor is net-POSITIVE, strongest on neighborhoods. B/T is scrubbed even at .02
  (.157) while NMI holds (.533): anisotropy removal and cluster destruction DISSOCIATE with dose.
- **f5 (affine-absorbable):** CONFIRMED — tax −1.3 lin vs A3's −13.9 at comparable λ:
  **A3 − f5 ≈ −12.5 lin / −18 kNN = the whitening/off-diagonal component is the damage.**
- **f6 (delay):** CONFIRMED equilibrium branch — f6 ≈ A3 on everything; the transient/opening
  hypothesis is dead.
- **f3 (rank floor):** PARTIAL — small tax (−1.4 lin, −3.8 knn200), class structure largely kept
  (NMI .480), but the certificate did NOT deliver: effrank FELL to 9.7 (< C1's 22) — the sliced
  one-sided floor under-enforces the native spectrum (rotation-mixing makes sliced spectra read
  flatter than native — the D-025 k̂ lower-bound lesson, now on the loss side). Battery fold-in
  needed before any certificate language.
- **f4 (cluster isolate):** NEITHER predicted branch — the term found a third path: near
  rank-collapse (effrank 2.2, BELOW the randinit null) with NMI .161, kNN −25.8, while converged
  linear survives at .6136 (linear info persists in low-variance directions). Proposed mechanism
  (NOT agreed): per-slice-standardized shape pressure is satisfiable by concentrating variance —
  degenerate directions make every random slice read Gaussian — so the shape term creates a
  RANK-COLLAPSE INCENTIVE the moment channel normally opposes; the degenerate-case audit's
  "the term sees deadness but is not its fixer" was the loophole, realized as an attractor.
  Framework-E7(i) at IN-100 scale: per-slice standardization does not benignly restore shape
  pressure. Also an instrument lesson: effrank (variance-weighted) and linear separability
  dissociate completely here.

*(All items owed here landed: battery fold-in + P5 Varimax read below; takeaways in AGREED
TAKEAWAY; the f2-as-candidate-method note became the E17→E19→E21 program. Dose-response
0 → .02 → .1 → .478 = .6456 → .6578 → .6338 → .5070 lin. Seeds remain n=1, deferred per D-024.)*

### Floor-satisfaction table (Berker's panel request, 2026-07-12) + T3 confound datum

`results/diag/e12_floor_values.csv` (audit-side moment-KL at h, 32 draws) +
`results/figures/e12/e12_dose_vs_loss.png`: C1 free h reads 1.65; λ=.02 → **0.123** (~93% of the
achievable reduction); λ=.1 → 0.087; λ=.478 → 0.070 — **satisfaction saturates at low dose**
(feeds E12-T2's condition, verified). Two raw observations for discussion: (a) **T3's
loss-strength confound is REAL (Berker's caveat, now measured):** f5's own diagonal floor reads
UNSATISFIED at the audit frame (diag_kl 0.91; per-dim eval variance ≈ 0.13) despite its training
term descending — train-mode/augmented-batch satisfaction does not transfer to the eval/audit
frame (drop_path + aug-distribution gap; a loss-vs-audit-frame lesson in its own right). So
A3−f5 as the "whitening isolate" is confounded by enforcement level; T3 stays UNWRITTEN.
(b) A1's h is NOT moment-calibrated at eval (2.74) — its SIGReg@h enforced little of the moment
target at this K/λ.

## G-WAVE — cross-method f2 test (D-028; launched 2026-07-12, pre-registered before numbers)

Arms: `e12gv` = vicreg M2-lane config + moment-KL@.02 at trunk-GAP h · `e12gd` = dino M2-lane
config + moment-KL@.02 at the student global-crop CLS (teacher h follows by EMA). Controls =
existing `in100.{vicreg,dino}.s0` (byte-identical configs minus the floor). Question: is E12-T2's
conditioner effect lejepa-specific or method-general — dino sharp case (best h in grid; its own
z-side prototype pressure untouched). Prediction (directional): same-sign as f2 — kNN-favoring
improvement or at worst a small tax; a dino improvement would be the strongest possible datum for
"the floor is a general conditioner." Declared limitations: λ nominally matched (not equal-pull
re-measured), gradient scales differ per method; single seed; n(floor samples)/step = 512 (vicreg)
/ 256 (dino) vs lejepa's 512. Scoring = same chain (probes + class-align + floor-values extended).

**Amendment 2026-07-12 (D-030, Berker-directed): matched floor-off controls launched** —
`e12gvc` (62241903 + 3 links) / `e12gdc` (62241908 + 3 links), the g-arm commands verbatim minus
the floor overrides. These are the PRIMARY controls (single-factor pairs g-arm vs g-control from
the same working tree + hardware pool); `in100.{vicreg,dino}.s0` demote to reference rows.
Launch-line verification recorded in D-030: no lr delta found between g-arms and lanes (all yaml
lr 1e-3); actual residuals closed = pre-refactor lane code + H100→A40 (dino) + run epoch.
Scoring tables gain the two control rows; predictions unchanged (they were g-arm vs matched
control by intent).

### Battery fold-in (D3, 2026-07-12) — h = z.embed, train500, raw

| arm | rankme | eff_rank | EP | kurt_worst | mean_abs_corr | var min/mean |
|---|---|---|---|---|---|---|
| C1 | 87.1 | 34.0 | 52.7 | 2.68 | .128 | .485 |
| f2 | 305.2 | 253.3 | 16.4 | 0.51 | .042 | .614 |
| f1 | 360.3 | 308.2 | 7.2 | 0.34 | .036 | .566 |
| f3 | 175.2 | 29.7 | 56.4 | 3.85 | .191 | .529 |
| f5 | 52.8 | 15.8 | 64.9 | **55.8** | .450 | .646 |
| f4 | 39.4 | **2.8** | 16.3 | 0.60 | .260 | .287 |
| A3 | 366.0 | 320.4 | 7.3 | 0.73 | .033 | .600 |
| f6 | 371.7 | 340.5 | 6.7 | 0.73 | .029 | .634 |
| A1 | 40.2 | 24.2 | 52.7 | 1.86 | .167 | .594 |
| A2 | 121.9 | 88.6 | 13.2 | 0.42 | .087 | .539 |
| lane | 35.0 | 24.5 | 57.5 | 1.50 | .150 | .386 |
| null | 79.5 | 6.9 | 213.6 | 2.11 | .371 | .375 |

Raw notes: f5's kurt_worst 55.8 = with per-dim variance pinned, structure fled into extremely
heavy-tailed correlated directions (feeds T3's confound story); f4's EP 16 with eff_rank 2.8 =
the slices read tame exactly because of the collapse (the gaming, visible); f2 sits at the
well-conditioned corner on every column. P5 Varimax read: `results/diag/e12_varimax_p5.csv`
(A3/C1/f2; §6.2 adaptation — class-probe weight matrix as the F-proxy at IN-100). **Numbers
(raw; fork-gate reading = discussion):** sparsity gain over the random-rotation null ≈ 10–12×
in ALL three arms (symptom i PRESENT everywhere); coordinatewise excess kurtosis in the Varimax
basis: C1 mean 1.06 (81% positive, p90 2.1) and f2 mean 0.44 (87% positive) = leptokurtic
(symptom ii PRESENT on free/low-dose h), while A3 reads ≈ 0 — the full-dose floor Gaussianized
the coordinates, consistent with E12-T1. Mechanical §6.2 reading: on the ν=∞-pilot analog (C1),
BOTH symptoms fire ⟹ the framework's own rule would select a finite-ν (heavy-tailed) declared
prior for this data regime. Whether the fork is still the right vehicle post-PIVOT is exactly
the next-session discussion.

## AGREED TAKEAWAY

- **E12-T1** (USER-APPROVED 2026-07-12) — full second-moment calibration at h scrubs semantics
  while succeeding at its own aim; strong-form thesis falsified (P2+K4).
- **E12-T2** (USER-APPROVED 2026-07-12, loss-condition verified) — at λ=.02 the floor is a
  net-positive conditioner with ~93% constraint satisfaction (saturating dose curve); marginal
  terms at h are conditioners with an interior optimum, not destinations.
- **E12-T3** (USER-APPROVED 2026-07-12) — damage is equilibrium (f6≈A3) with the re-metrization
  signature (kNN-heavy, k-worsening); whitening-component attribution held at "consistent with"
  (A3−f5 enforcement-confounded: f5's floor unsatisfied at the eval frame — the train-mode vs
  audit-frame lesson).
- **E12-T4** (USER-APPROVED 2026-07-12) — pure shape pressure at h admits a rank-collapse
  attractor (variance concentration makes every slice read Gaussian); framework-E7(i) falsified
  at this scale.
- **E12-T5** (USER-APPROVED 2026-07-12) — instrument lessons: B/T vs NMI dissociate; effrank vs
  linear separability dissociate; sliced floors under-enforce native spectra (k̂ lower-bound
  caveat, loss-side).
- T6 (E10-T4 travel scope) deferred — more exploration first. Full row texts in DECISIONS.
- **E12-T7** (USER-APPROVED 2026-07-13, wording delegated, veto open) — **the moment-KL floor at
  h is a method-general kNN-favoring conditioner — and λ was selected ONCE, on lejepa only.** In
  the one per-model-tuned case (lejepa, dose curve measured) the chosen dose improved BOTH probes
  substantially (f2: knn200 +7.5, converged linear +1.2, no tax in any column); transplanted with
  NO per-model tuning (λ=.02 verbatim, enforcement only partial: moment-KL 2.46→0.89 vicreg /
  0.93→0.33 dino) it still delivers knn200 +6.1 (vicreg h.gap) / +3.6 (dino teacher CLS) over
  matched single-factor controls with kmeans-NMI rising (.30→.34, .51→.55); D-028's
  pre-registered direction MATCHED on both methods. The converged-linear cost on the untuned
  methods is small and LOCAL to the constrained tap (−1.6 / −0.7 there; ≈0..+0.8 one tap away
  and across the head; single seed, no seed CIs — magnitude unresolved vs seed noise); the tuned
  case shows the tax is not intrinsic. Cross-method effect-size ordering UNCLAIMED (equal-pull
  unmeasured, D-028 limitation); λ*/seed sweep on vicreg/dino = declared headroom, not claimed.
- **E12-T8** (USER-APPROVED 2026-07-13, wording delegated, veto open) — **two-space result:
  calibrated h wins on essentially every h-side stress dimension measured, and the floor is
  INVISIBLE at z.** At the declared h, arm vs matched control (lejepa/vicreg/dino order):
  effective rank 253/149/198 vs 34/37/93; participation ratio 217/113/153 vs 22/15/59; α-ReQ
  decay .35/.99/.67 vs 2.54/1.69/1.45; Epps–Pulley 43/66/55 vs 454/386/157; worst top-eig |kurt|
  .51/.32/.48 vs 2.68/1.19/.90; mean |off-diag corr| .042/.057/.048 vs .128/.128/.085;
  uniformity −3.9/−3.8/−3.7 vs −1.6/−0.6/−3.3; variance-floor hinge .13/.69/.42 vs .45/.90/.63;
  **negative-pair (different-image) cosine collapses to ≈0** (.003/.006/.051 vs .548/.840/.143 —
  the control cone is gone), and the invariance margin rises where the cone was worst (vicreg
  .46 vs .10, lejepa .54 vs .39; dino ≈flat, .52 vs .54 — its one h-side non-win). Meanwhile at
  each method's loss space z, arm ≈ control across the full battery, V=8 view-invariance, and
  converged probes (results/figures/e12g/): the floor's effect is confined to h and the trunk
  below it — invariance margin lifted through the TRUNK segment only, arm/control depth curves
  merging from the first head layer in all three methods, the vicreg control's margin notch at h
  (L09 .20 → h .10 → tap1 .63) erased by the floor (.31→.46 smooth). Reading: each objective
  re-establishes its own z geometry regardless of h conditioning; the floor relocates
  conditioning/invariance into the backbone without touching the head's trajectory. Open pocket
  (parked): dino deep-head tap2 probes −3.2..−3.7 both networks + arm bottleneck diag-KL .5→1.1
  — see floor-placement question below.
- **E12-T9** (PARTIALLY RESOLVED 2026-07-14, D-037 — part (i) monitor bias: OPEN, deprioritized
  by Berker ("we can look at it later… not a priority till we fix the other issues"; his prior:
  online should ≈ offline in general, which is why we monitor online at all — the ~+3pt arm bias
  stays an unexplained anomaly for later); part (ii) RESOLVED as a conditional convention: cos
  margin is the headline WHERE rand-cos ≈ 0 (decorrelated negatives), otherwise the two readouts
  carry equal weight, reported jointly — METRICS.md updated) — **instrument rows.**
  (i) The online monitor is not arm-neutral: ~+3 pts toward floor arms vs converged clean
  offline probes when its head trains on aggressive augmented student views (lejepa +2.98,
  vicreg +2.84; ≈neutral on dino's EMA-teacher mild crops, +0.44); vicreg's monitor-vs-offline
  sign flip (+1.3 vs −1.6) is that bias crossing zero. The augmented-input-distribution
  mechanism was tested and REFUTED (converged offline probes trained on the monitor's own view
  distribution stay negative: vicreg −1.8; results/diag/e12_aug_probes.csv) — the bias lives in
  the online/co-training setup; converged offline v2 probes are the arbiter. (ii) The two
  normalized invariance readouts couple to conditioning: a cone-collapsed space flatters
  ratio-form align_rel (positive AND random distances shrink together) while the difference-form
  cos margin stays interpretable when read with the rand-cos panel as referee. **cos margin =
  the project's headline invariance readout** (Berker 2026-07-13: the calibrated space's
  rand-cos decorrelates almost completely, making the margin clean; align_rel stays recorded in
  CSVs, not headlined). Also recorded per Berker: B/T is not read as class separability.

### G-wave scoring — landed 2026-07-13 (chained pipeline; controls = PRIMARY per D-030)

Audited converged probes, arm vs matched control: **vicreg** h.gap lin .5808 vs .5966 (−1.6),
knn200 **.4614 vs .4000 (+6.1)**; **dino** teacher.h.cls lin .6852 vs .6922 (−0.7), knn200
**.6342 vs .5978 (+3.6)** (student.h.cls mirrors: −0.7/+3.4). D-028's directional prediction
(same-sign as f2: kNN-favoring improvement, at worst a small linear tax) — **pattern matches on
both methods**. Controls ≈ original lanes on every statistic (e12gvc≈vicreg lane, e12gdc≈dino
lane) — the D-030 control construction closed confounds that turn out negligible. Mechanism
corroboration (results/diag/e12_{floor_values,class_align}_g.csv): floor genuinely enforced at h
(moment_kl 2.46→0.89 vicreg, 0.93→0.33 dino; dino floor-space ≈ audited-h, EMA follows); effrank
15→113 (vicreg), 59→153 (dino); B/T falls while kmeans-NMI RISES (.30→.34, .51→.55) — the E12-T5
variance/recoverability dissociation replicated cross-method. Takeaways T7–T9 FILLED
(AGREED TAKEAWAY above; Berker 2026-07-13 "given our discussion you can fill the takeaways").

### G-wave close-out artifacts + parked follow-ups (2026-07-13)

Artifacts: h-vs-z figures `results/figures/e12g/e12g_hz_{battery,invariance,probes}.png` +
`e12g_depth_invariance.png` (depth-ladder preview) · raw CSVs
`results/diag/{e12g_tap_deltas,e12g_orbit_invariance,e12_aug_probes}.csv` · V=8 orbit stores
`in100.pairs100.v1@<stack>.o8` + per-layer clean stores `in100.{train500,val}.v1L` under the six
`.ext` run_ids (h_layers 3/6/9; HEAD_OVERLAP_LIPSCHITZ.md extraction record).

**Floor placement — UNPARKED same day (H-wave below). Original note (Berker 2026-07-13: "are we
sure we employed calibration at the correct part? maybe we should add it to gap + cls
overall").** Angle (Claude, endorsed as worth
an arm): in BOTH g-methods the un-floored sibling trunk tap gets a linear BENEFIT while the
floored tap pays the tax (dino: gap +0.7/+0.8 lin vs floored cls −0.7; vicreg: cls +0.5 vs
floored gap −1.6) — pinning one readout while its sibling floats may be exactly what localizes
the tax. Candidate arms: (a) joint {CLS, GAP} floor; (b) token-level floor (pooled patch-token
stats) so every readout inherits calibration. Either would also probe whether dino's deep-head
pocket (T8) is a placement artifact, and feeds H3's where-does-the-constraint-bind question.
New training runs → user gate before launch.

## H-WAVE — floor-placement + inv-assist arms (Berker-directed 2026-07-13; PRE-REGISTERED before numbers; D-035)

Arms (2-ep smokes precede full launches; H100 slots per Berker "use 2 h100s"):

- **e12gd2** = the e12gd command verbatim + `+method.h_taps=clsgap` — moment-KL floor at BOTH
  student trunk readouts (CLS and patch-GAP), per-tap dose λ=.02 each (total pressure 2× gd;
  per-tap dose matched to gd for tap-level comparability — declared choice). Comparisons:
  e12gdc (no floor), e12gd (CLS-only) = the placement contrast.
- **e12f7** = the f2 command verbatim + `+method.h_inv=0.01` — tiny ADDITIVE view-invariance
  pull at the embedding (mean-squared deviation of each view's embed from the per-image
  view-mean; same functional form as the shipped proj-space inv term), alongside f2's moment
  floor (h_lamb=.02). h_inv=.01 is a DECLARED first dose, not equal-pull measured; per-term
  logging watches its actual share. Comparisons: f2 (floor only), c1 (neither).

Pre-registered directional predictions:

- **gd2-P1:** kNN gains over gdc persist at BOTH trunk readouts.
- **gd2-P2 (placement question):** pinning both taps removes the floored-tap-tax /
  floating-sibling-gain asymmetry — GAP's linear gain (+0.7/+0.8 in gd) shrinks or flips once
  GAP is pinned; if instead both taps keep kNN gains at ≈no linear tax, joint flooring is
  strictly better placement.
- **gd2-P3:** the deep-head pocket (tap2/bottleneck damage; arm bottleneck diag-KL 1.5 vs .45)
  PERSISTS — it keys on flooring the head's input (CLS), which gd2 still does. If it shrinks,
  the pocket was the CLS/GAP imbalance.
- **gd2-P4 (open, no confident direction):** the mid-trunk linear deficit (gd L03/06/09
  −2.9/−5.2/−3.5) — record both outcomes; mechanism unknown.
- **f7-P1:** invariance margin at embed rises vs f2.
- **f7-P2 (H3):** the invariance JUMP across the projector (margin at proj.out − margin at
  embed) SHRINKS vs f2 — the head does less invariance work.
- **f7-P3:** head empirical Lipschitz (instrument: results/diag/e12g_head_lipschitz.csv) reads
  ≤ f2 on the embed→out segments.
- **f7-P4:** probes pay no tax at ε dose (knn200/linear within noise of f2 or better). Declared
  risk: inv-at-h is collapse-flavored pressure; the floor's variance barrier opposes it —
  grad_norm kill-trigger standing (100× running median).

Frame/discipline: everything else = parent commands verbatim (M2 frame, seed 0, cadence ckpts,
wandb online). Code: `h_taps` in `sslgap/methods/dino.py`, `h_inv` in `sslgap/methods/lejepa.py`.

### H-wave λ measurement (2026-07-13; jobs 62288090/91; results/diag/e12h_pull.csv; rules declared in-conversation BEFORE the numbers)

Per-term UNWEIGHTED encoder-grad norms, real first batch, seed 0, training autocast (p3
convention generalized). lejepa/f7 config: g_sigreg 454.7 · g_inv 4.039 · g_h_moment 4.362 ·
g_h_inv 0.4669. Context datum: f2's fixed λ=.02 ⇒ the floor ran at a weighted pull of 2.2% of
the shipped inv pull. dino/gd2 config: g_dino 1.552 · g_cls_floor 6.793 · g_gap_floor 3.389
(cls/gap pull ratio 2.00 at equal λ — outside the declared 1.5× parity band).

**Launch doses (measured, no further discretion):**
- **e12f7: `+method.h_inv=0.8478`** — rule: weighted h_inv pull = 10% of the shipped weighted
  inv pull (0.098·g_inv/g_h_inv). The .01 first guess would have been 0.12% — negligible.
- **e12gd2: `+method.h_lamb=0.02` (cls, unchanged from gd) + `+method.h_lamb_gap=0.0401`** —
  rule: gap pull matched to the cls floor's pull (.02·g_cls/g_gap); per-tap weighted pulls
  0.136 each = 8.8% of the dino-loss pull each.

Fresh 2-ep smokes AT the launch doses precede the chains (collapse risk is dose-dependent;
the first smokes at .01/.02-gap validated the code paths only). Watch-items: per-term pull
drift (h_inv is scale-dependent; the floor pins embed scale), grad_norm kill-trigger standing.

**H-wave dose-smokes PASS (2026-07-13, jobs 62288110/111; 2 ep each).** gd2@gap-λ.0401: both
floor terms descending (gap .39 vs .50 at λ.02 — stronger dose enforces harder), probe pace =
code-path smoke, grad_norm quiet. f7@h_inv.8478: h_inv .011 by ep2 (hard enforcement); the
embed floor RISES to ~2.1 and plateaus (vs .84 at negligible h_inv) — the invariance pull and
the calibration floor visibly fight to an equilibrium at h; z-side terms and probe unchanged
(inv .163/.161, sigreg 3.92/3.97, probe .1400/.1400); no incident. WATCH-ITEM (declared): if
f7's h_moment_kl climbs past the CONTROL's free-h level (~1.65) and keeps rising by ep25, the
dose is buying invariance by de-calibrating h — a landing-discussion datum, not a mid-flight
change. Chains launched: 3×8h links per slot.

### H-wave third arm — e12gvi (Berker 2026-07-14: "use another h100, to run vicreg with added
### h_inv term, use a similar strategy when setting the weight"; PRE-REGISTERED before numbers)

**e12gvi** = the e12gv command verbatim (vicreg, floor at trunk-GAP λ=.02, bs=256) +
`+method.h_inv=<pull-measured>` — the vicreg mirror of f7: floor + tiny view-invariance pull at
the SAME tap (GAP), same functional form, same 10%-of-shipped-inv-pull rule (here: 10% of
w_inv·g_inv, w_inv=25). Third H100 slot `h100-slotC` = a USER-AUTHORIZED exception to the
standing 2-slot cap. Reading of "vicreg with added h_inv": floor RETAINED (f7-mirror; the
floor's barrier is what makes an inv pull at h collapse-safe) — flagged to Berker, correctable.

Declared structural difference from f7: vicreg's projector reads the CLS token; the floor and
the assist sit at GAP, so the assist is NOT on the head's direct input path (lejepa's embed
feeds its projector directly). gvi therefore also probes whether the assist must sit on the
head-input path to move the head's burden.

Predictions (pre-registered):
- **gvi-P1:** invariance margin at h.gap rises vs gv (gv arm: .46).
- **gvi-P2 (H3):** the head's invariance jump (margin at proj.out − margin at gap; gv: .23)
  shrinks vs gv.
- **gvi-P3:** head empirical Lipschitz on gap→out ≤ gv's.
- **gvi-P4:** no probe tax at the 10%-pull dose (knn200/linear within noise of gv or better).
- Watch-item (open direction, from f7's live datum): the h-floor equilibrium under the pull —
  f7's floor RECOVERED calibration (2.1→1.4 by ep57); record vicreg's direction.
Discipline: pull measurement → dose recorded here → 2-ep dose smoke on slotC → 3×8h chain.

**e12gvi λ measurement (2026-07-14, job 62298401; e12h_pull.csv).** Unweighted encoder pulls:
g_inv .4155 (×w_inv 25 = 10.39 weighted) · g_var .6581 (×25 = 16.45) · g_cov 35.66 (×1) ·
g_h_moment 3.664 (×.02 = .0733 — the gv floor runs at 0.7% of the weighted inv pull; lejepa
2.2%, dino 8.8%: the T7 equal-pull caveat fully quantified) · g_h_inv .3572.
**Launch dose: `+method.h_inv=2.9087`** (10%-of-shipped-inv-pull rule: .10×25×.4155/.3572).
Raw observation: cov dominates the encoder pull at init (35.7 of ~63 total weighted). Single
2-ep smoke at the launch dose covers code path + dose (first vicreg h_inv smoke).

### H-wave fourth arm — e12gvcls + a Lipschitz-instrument CORRECTION (Berker 2026-07-14: "no
### then we should do both the inv and kl on cls if thats the feature space mate … lambda 0.02
### on cls"; PRE-REGISTERED before numbers)

**Correction first (Berker catch):** vicreg's projector reads the CLS token; the trunk-GAP is
the audited h but NOT on the head's input path. Consequences: (a) gv "calibrated vicreg" floored
a tap the loss machinery never consumes — its audited-gap gains are trunk-side conditioning, and
the interpretation of gv is now explicitly under review against the new arms; (b) the
head-Lipschitz vicreg chain used gap→tap1, which is NOT a computed path — those rows were
cross-tap displacement ratios; the instrument is corrected to cls→tap1→tap2→out and re-run
(figure + CSV regenerate in place; the lejepa/dino chains were real paths and stand).

**e12gvcls** = vicreg M2 config (bs 256) + moment-KL floor at **CLS** (λ=.02, per Berker) +
h_inv at **CLS** (10%-of-shipped-inv-pull rule, measured for the CLS placement). Monitor stays
at the audited GAP (PROTOCOL §3; comparability across arms unchanged). Fourth H100 slot
`h100-slotD` = user-authorized. **e12gvi (gap placement) CONTINUES on slotC** — the pair is the
on-path/off-path contrast: same terms, same rules, only the tap differs.

Predictions (pre-registered):
- **gvcls-P1 (path matters):** head-burden effects STRONGER than gvi's — margin at CLS up; the
  head's invariance jump (proj.out − CLS) shrinks more than gvi's (proj.out − gap).
- **gvcls-P2 (tap-local signature moves):** the T7 pattern relocates to CLS — kNN gain
  concentrates at CLS, any small linear tax sits at CLS; the audited GAP becomes the "floating
  sibling" (gv gave it +0.5 lin/+1.0 knn when un-floored — direction now open, recorded).
- **gvcls-P3:** head empirical Lipschitz on the REAL path (cls→tap1) ≤ gv's corrected value.
- **gvcls-P4:** z-side invisibility persists (T8 pattern).
- **Interpretation stake (Berker):** if gvcls reproduces gv's audited-gap improvements through
  the shared trunk, "calibrated vicreg" is trunk conditioning, placement-robust; if the gap
  gains vanish, gv's story was tap-local — either way the gv interpretation gets revised here.
Discipline: pull measurement (CLS config) → dose recorded → 2-ep dose smoke on slotD → 3×8h
chain. Floor equilibrium + var/cov interaction = watch-items; kill-trigger standing.

### H-wave fifth arm — e12gv2, the FIXED gv (Berker 2026-07-14: "launch a fixed gv as well. it
### is important."; PRE-REGISTERED before numbers)

**e12gv2** = vicreg M2 config (bs 256) + moment-KL floor at **CLS only** (λ=.02, the gv dose at
the corrected placement; NO h_inv). The missing single-factor cell: gv2−gvc isolates the floor
at the head's real input; gv2−gv isolates placement alone; gvcls−gv2 isolates the added inv
term. Monitor stays at audited GAP. Slot `h100-slotE` (fifth concurrent H100, user-directed).
Dino/lejepa checked for the same mistake: NOT present (dino's floor sat on student CLS = its
head input; lejepa's on embed = its projector input) — vicreg-only issue (audited h GAP ≠ head
input CLS, F1 port flag).

Predictions (pre-registered):
- **gv2-P1:** the T7 tap-local signature appears at CLS (kNN gain concentrated at CLS; any
  small linear tax local to CLS).
- **gv2-P2 (the gv-interpretation stake):** direction of the audited-GAP readouts vs gvc is the
  decisive datum — if GAP improves (trunk propagation), gv's story generalizes
  placement-robustly; if not, gv was tap-local conditioning of the audit space.
- **gv2-P3:** z-side invisibility persists (T8).
- **gv2-P4:** head empirical Lipschitz on the real path (cls→tap1) tamer than gvc's.
Discipline: fixed λ (known term, f2/gv lineage — no pull measurement needed); 2-ep smoke on
slotE (first training of the h_tap=cls code path alone) → 3×8h chain.

**e12gvcls λ measurement + corrected-Lipschitz datum (2026-07-14, jobs 62298737/62298738).**
CLS-placement pulls: g_h_moment 6.537 (1.8× the gap placement's 3.664 — echoes dino's cls/gap
= 2.0) · g_h_inv 1.0605 (3× gap's .357) · inv/var/cov reproduce the gvi batch (internal
consistency ✓). **Launch dose: `+method.h_inv=0.9790`** (10%-pull rule at CLS); floor λ=.02
fixed (Berker). Corrected head-Lipschitz (real path cls→tap1): gv arm .547 vs gvc .838 — the
1.5× taming SURVIVES the correction at reduced magnitude (was 3.7× on the invalid gap ratio),
and it occurred with the floor off-path at GAP: trunk-propagated. Deeper segments identical
(1.438/1.436, 5.833/5.825). Figure/CSV regenerated in place.

### H-wave sixth arm — e12f8, the SHAPE-DOMINANT lejepa (Berker 2026-07-14: "h invariance pull for lejepa was dominating the kl moment shape regularization part … redo that experiment where shape is more important than h_inv pull"; PRE-REGISTERED before numbers)

**Observation that motivates it (from the e12h h-vs-z figure, RAW):** in f7 the h_inv pull dominated
the moment-KL floor at the embed — weighted inv pull 0.042·... = 0.8478·g_h_inv(0.467) = **0.396**
vs the floor pull 0.02·g_h_moment(4.362) = **0.087** (inv : floor = **4.5:1**), and the embed
DE-CALIBRATED: moment-KL 0.583(c1)→0.043(f2)→**0.976**(f7), Epps-Pulley 454→42→165, excess-kurt
2.68→0.51→**12.1**, while embed invariance rose 0.391→0.540→**0.754**. So f7 bought h-invariance by
spending h-gaussianity.

**e12f8** = the f7 command verbatim + **h_inv reduced 0.8478 → 0.042** (nothing else changes; floor
`h_lamb=0.02` kept). This flips the dominance to **floor : inv = 4.5:1** — the exact mirror of f7's
inv-dominant 4.5:1 (same magnitude, reversed). Doses grounded in e12h_pull.csv (per-unit grad norms
are init-determined, so f7's g_h_moment=4.362 / g_h_inv=0.467 apply to f8): inv pull 0.042·0.467 =
0.0196 = floor pull 0.087 / 4.5. One H100 (slot `h100-slotF`, user-authorized "use one h100").

Pre-registered directional predictions:
- **f8-P1 (calibration holds):** embed moment-KL stays LOW — near f2's 0.043, NOT f7's 0.976;
  Epps-Pulley near f2's ~42 not 165; worst excess-kurt near f2's ~0.5 not 12. The dominant floor
  holds the embed's gaussianity.
- **f8-P2 (partial invariance):** embed cos_margin rises above f2 (0.540) but stays below f7 (0.754)
  — the small inv pull buys *some* invariance that the dominant floor permits.
- **f8-P3 (best-of-both probe test):** embed linear/knn ≥ f2 (f2: lin 65.8 / knn 60.6). If the
  small inv assist adds probe value WITHOUT the de-calibration cost, shape-dominant is the better
  balance; if probes ≈ f2, the inv at this dose is inert.
- **f8-P4 (head burden):** projector inv-jump (proj.out − embed cos_margin) and head operator
  Lipschitz (σ_max) both between f2 and f7.
Discipline: known f7 code path, inv 20× smaller (collapse risk strictly lower than f7); 2-ep smoke →
3×8h chain; grad_norm kill-trigger standing; watch-item — does the floor hold the embed moment-KL
low through training (the f7 failure was a late rise)?

### H-WAVE SCORING — COMPLETE except gd2-P4 (2026-07-16; raw + mechanical vs pre-reg; `experiments/e12h_score.py`; NO takeaway)

First-pass gd2/f7 numbers unchanged (gd2-P1 ✓ knn +2.9 cls / +12.2 gap; gd2-P2 both-pinned removes
the gd asymmetry at a −2.0/−1.9 lin cost; gd2-P3 pocket PERSISTS, teacher bottleneck diag-KL
.478→1.78; f7-P1 ✓ +21.3; f7-P2 ✓ jump .373→.163; f7-P3 MIXED (embed→tap1 1.443 > f2 .341);
f7-P4 knn −2.9 / lin +2.2). New this pass:

**f8 (shape-dominant, floor:inv 4.5:1):**
- P1 PARTIAL: embed diag-KL .143 (~f2 .043, NOT f7 .976 ✓); EP 58.7 (f2 42.5, f7 164.7 ✓);
  **kurt_worst 7.11 — the worst direction is NOT held** (f2 0.51, f7 12.08): bulk calibration holds,
  worst-direction doesn't.
- P2 ✓ exactly: margin@embed .693 ∈ (f2 .540, f7 .754).
- P3: **lin 67.0 = +1.2 vs f2, +2.4 vs c1 — the best lejepa linear measured**; knn200 60.2 = −0.4 vs
  f2 (+7.1 vs c1) → the shape-dominant mix keeps f2's kNN while adding f7's lin gain.
- P4 ✓ direction: inv-jump .223 ∈ (f7 .163, f2 .373); seg-stretch embed→tap1 .549 ∈ (f2 .341,
  f7 1.443); head σmax (raw-orig scale — E12-lane heads are spec_norm-parametrized, effective
  σmax=1 by construction; reader fixed 07-16): f8 [8.44, 20.19, 2.2] between f2 [10.58, 20.54,
  2.18] and f7 [6.37, 16.6, 1.75] per-layer ✓ — monotone in inv dose.

**gv2 (floor@CLS only) vs gvc — the CLS placement matrix:**
- P1 ✓ T7 tap-local at CLS: knn +2.4, lin +0.1.
- **P2 DECISIVE (the gv-interpretation stake): audited-GAP readouts do NOT propagate** — gap lin
  −1.5 / knn −0.7 vs gv's gap +6.1 knn ⇒ per the pre-registered stake, gv's audited-gap gains were
  TAP-LOCAL conditioning of the measured space, not trunk improvement. gv reinterpretation
  triggered (discussion owed).
- P3 ✓ z-side invisibility (proj.out Δ +0.1/+0.1). P4 ✓ cls→tap1 .478 < gvc .838.
- **NOTABLE (mechanical): floor-ONLY gv2 drives the head inv-jump +.474→+.005** with margin@cls
  .215→.684 — no inv term involved; the mean-seeing floor removes the CLS cone by itself
  (cross-validates E17's mean-carried-cone datum; the own var+cov could not — E17 T5 lane).

**gvcls (floor+inv@CLS) vs gvc/gv2:**
- P1: inv-jump +.0105 ≈ gv2's +.0048 — no extra shrink (the floor had already zeroed it).
- P2: cls lin +0.5 / knn +2.3; gap lin +1.4 / knn +1.7 — the added inv contributes ≈ nothing over
  gv2 (knn +2.3 vs +2.4), consistent with the E17 +inv pattern.
- P3 ✗ MISS (recorded): cls→tap1 .807 NOT ≤ gv .547. P4 ✓ (+0.6/+0.4).
- Placement contrast gv2−gv: cls knn +1.4, gap knn −6.8 (each tap gains when floored, loses when
  the floor moves away — tap-local both ways).

**gd2-P4 LANDED (2026-07-16):** gd/gdc val-side stores lacked L-taps (probe.py probes train∩val
spaces) → re-extracted as `in100.dino.s0.e12gd{,c}.ext2` (jobs 62379959–62; sanity: gdc.ext2
teacher-CLS lin 69.1/knn 59.8 reproduces). Teacher mid-trunk linear Δ vs gdc: **gd2 −0.9 / +1.6 /
−1.9 at L03/06/09 vs gd −3.0 / −5.1 / −3.6** (matches the remembered −2.9/−5.2/−3.5) — the
CLS-only floor's mid-trunk deficit largely VANISHES when both trunk readouts are floored.
(P4 was pre-registered open, "record both outcomes".)
