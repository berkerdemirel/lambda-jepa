# E12 — The moment floor: what shape regularization at h is for

**Status: PRE-REGISTERED (Berker sign-off 2026-07-11: "rest got my pass you can start!", with the
compute amendment "2 h100 is capped but gpu partition will likely to be free"). Predictions locked
in this file BEFORE any arm has produced a number. First arm of OUR METHOD (direction A of the
2026-07-11 design discussion); vehicle = M4's parked one-branch arm (LeJEPA lane), per HANDOVER
note. Ledger row: D-026.**

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

## Arms (4 runs; run_ids `in100.lejepa.s0.e12{a1,a2,a3,c1}`)

> **A2/A3 rows below are SUPERSEDED by the 2026-07-11 pre-full-launch amendment (§Amendment):
> the h-side term is ADDITIVE on top of the shipped SIGReg@proj, not a replacement.**

| arm | placement | regularizer | projector | λ | lr | init calib |
|---|---|---|---|---|---|---|
| **A1** package-Dlr (replication) | both terms at h (depth-0) | sliced-CF (SIGReg) | none (Identity) | 0.02 fixed (Dlr-faithful) | 3e-4 | yes |
| **A2** sliced floor at h (method cell, CF form) | align at proj.out; SIGReg at h | sliced-CF | depth-3, spec-normed | equal-pull (measured) | 3e-4 | yes |
| **A3** moment floor (thesis cell) | align at proj.out; floor at h | Gaussian-moment KL | depth-3, spec-normed | equal-pull (measured) | 3e-4 | yes |
| **C1** buffered control (matched package) | all losses at proj.out (as shipped) | sliced-CF | depth-3, spec-normed | 0.02 (shipped) | 3e-4 | yes |

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

## AMENDMENT — 2026-07-11, pre-full-launch (Berker ⊕ Claude; approved "yes please do that")

**Berker's catch, before any full arm ran:** in A2/A3 as pre-registered (regularizer MOVED to h,
proj.out carrying only `inv`), the loss admits alignment-free minima — the projector can zero
`inv` unilaterally. Spectral norm pins σ_max of each Linear at 1 but leaves BN-γ shrinkage and
variation-subspace annihilation open; `inv` is scale-dependent, and with the z-side scale pin
removed, shrinkage minimizes it without aligning anything. This is the framework's own fine print
realized (alignment is a sufficiency surrogate only JOINTLY with a marginal term **in the same
space**; R6 head-conditioning exposure σ_min(g)→0 binding, not just watched) — BYOL's
"objective admits collapse" logic one level up, at the head. The same hole exists in the v2.0
draft's Appendix-E assembled method → theory-note owed to the framework discussion.

**Measured demonstration (2-ep smokes, jobs 62211226/62211227; forensic job 62211346,
`outputs/e12-lazy_62211346.out`; wandb `e12a{2,3}smoke` kept as the failure record):** smoke `inv`
0.098→0.0011 (A2) / 0.119→0.0038 (A3) in 2 warmup epochs — 15–50× below the shipped lane's
END-of-100-ep value (0.053) at ~20% of peak lr; at ep2 proj.out across-image std 0.016/0.032
(near-degenerate scale) with within-image (view) std ~1.9× LARGER than across-image — shrinkage,
no semantic alignment, while the embedding grew to healthy scale under the floor: the arms were
training floor-only encoders.

**Amended arms (A1/C1 untouched; z-side now byte-identical across A2/A3/C1 — the h-term becomes
the ONLY factor, a strictly cleaner design than the original):**

- **A2 := C1 + SIGReg@embed** — L = 0.98·inv + 0.02·SIGReg@proj + λ_h·SIGReg@embed, λ_h = 0.0257.
- **A3 := C1 + MomentFloor@embed** — L = 0.98·inv + 0.02·SIGReg@proj + λ_h·MomentFloor@embed,
  λ_h = 0.4775.
- λ_h values carry over from the equal-pull measurement unchanged (g_inv and g_reg_h come from
  the same forward graph; the added proj term alters neither at init) — no re-measurement.
- Rejected alternatives: moments-only guard at proj (re-opens a z-side confound vs C1);
  scale-free/cosine alignment (deviates from the shipped alignment functional); dropping the
  projector in method arms (collapses A2 into A1, loses the buffered-alignment axis).

**Predictions and kills survive with wording changes only:** every "moved to h" reads "added at
h"; the pure moved-placement cell remains A1 (safe: alignment and floor share the space — the Dlr
equilibrium). P1's framework-E7(iii) reading now applies to the added trunk-direct h-term.
Re-smoke criterion before full launch: A2/A3 `inv` tracks the shipped lane's early trajectory
(epoch-1 scale ~0.1–0.2, no 100× collapse) and proj.out across-image std stays O(shipped).

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

*(Proposed reading, NOT agreed: class structure at h lives overwhelmingly in the second moments —
between-class anisotropy — so full second-moment calibration at the probed space scrubs semantics
by construction; effrank ordering null 3.7 < C1 22 ≈ A1 22 < A2 76 < A3 275 tracks the
suppression monotonically. The F-wave separates the candidate repairs: f5 = is it specifically
decorrelation; f3 = is a one-sided rank floor free; f1/f2 = dose; f4 = pure cluster channel.)*

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

**Still owed:** battery fold-in for all arms (sliced three-reference decompositions, k̂, rank
monitors — f3/f4 spectra especially); P5 Varimax read; seeds (everything here is n=1); joint
discussion before ANY takeaway. Candidate-method note for the discussion (proposed, not agreed):
**f2's configuration — C1 + moment-KL floor at h at λ≈.02 — is the best lejepa-family h we have
ever measured on every probe**, and its term is R1's closed-form rate/calibration expression at
gentle weight; dose-response (0 → .02 → .1 → .478 = .6456 → .6578 → .6338 → .5070 lin) suggests
a λ* sweep + seeds as the obvious next compute.

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
