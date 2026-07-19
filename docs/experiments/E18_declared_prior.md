# E18 — the declared-prior fork: SIGReg-to-t_ν at h (+ rider: sigreg@3%)

**Status: PRE-REGISTERED 2026-07-17 (design + predictions locked before numbers; launch pending
pull measurement + 2-ep smoke).** Successor to E17-T6's open arm (D-039; HANDOVER agenda iv,
Berker 2026-07-16 "test next session"). D-row: D-041 (PROPOSED at launch).

## Question

E17 found that the one shape-seeing own-term, SIGReg at lejepa's h, taxes trunk-deep (T3: mean
class-pair d′ −17%, lin −1.4 / knn −4.3) while the shape-blind moment floor with the same sliced
machinery wins (+7.5 knn). T6 attributed the harm to the shape channel and called Gaussian shape
"incidental" — but E17 cannot distinguish two readings:

- **shape enforcement per se harms at h** (any fixed unimodal slice-shape target spends class
  structure: a separated class pair IS a bimodal projection, and unimodality is shared by every
  t_ν target too), vs
- **the WRONG declared prior harms** (Gaussian demands lighter tails than h wants — the P5
  Varimax coordinates are leptokurtic; a correctly-declared heavy-tailed prior would not tax).

E18 swaps ONLY the declared prior: the Gaussian CF target at SIGReg's 17 knots becomes the
unit-variance spherical t_ν CF (slice-coherent: t_ν with identity shape is rotation-invariant, so
every random slice shares one 1-d target — the property that makes the fork well-posed, HANDOVER
iii). Everything else — tap (embed), lane (shipped), dose rule (10% of shipped z-pull), frame,
seed, machinery — is byte-matched to E17's `hpull_sigreg`.

## Frame

Identical to E17 lejepa arms: `in100_vits16` (ViT-S/16 @224, IN-100 CMC), bs=128, seed 0, 100 ep,
shipped lane (no embed_calib). Control = `in100.lejepa.s0.e17c.ext` (reused; no retrain).
Comparison triple: **ctrl vs hpull_sigreg (Gaussian target, E17) vs hpull_sigreg_t (t_ν target)**
at matched nominal pull.

## Arm

`in100.lejepa.s0.hpull_sigreg_t` — overrides:
`+method.h_reg=sigreg_t +method.sigreg_nu=8.2 +method.h_lamb=<from pull, below>`.

Implementation (`sslgap/methods/lejepa.py::t_nu_cf`, one-constant change): `SIGReg(nu=8.2)`
replaces the `phi` buffer with the unit-variance t_ν CF at the knots,
φ(t) = K_{ν/2}(u)·u^{ν/2} / (Γ(ν/2)·2^{ν/2−1}), u = √(ν−2)·t (unit-variance scaling ⇒ the
moment-1/2 targets stay exactly the Gaussian arm's — mean 0, variance 1; only the tail/shape
declaration moves). The quadrature weights KEEP the Gaussian window exp(−t²/2): the integration
measure over t is byte-identical across arms; only the target swaps. Constants are float64 scipy
(Bessel K) at init; no autograd through them.

Validation (2026-07-17, recorded): Bessel formula vs direct quadrature of the t density at
ν=9.7 agrees to 2.6e-8; ν→∞ recovers the Gaussian CF (O(1/ν): 0.011 @ν=50, 0.0027 @ν=200);
`SIGReg()` default path byte-identical (phi ≡ window, weights shared); forward/backward clean;
direction sanity: Gaussian batch scores worse under the t-target, heavy-tailed batch scores
better (t-target 3.41 vs Gaussian-target 5.05).

## ν fit record (`experiments/e18_nu_fit.py` → `results/diag/e18_nu_fit.csv`, job 62388673)

Recipe (Berker, agenda iv): ν from the P5 Varimax kurtosis datum, re-fit properly at design time.
Moment inversion for a t_ν marginal: excess kurtosis κ = 6/(ν−4) ⇒ ν = 4 + 6/κ. The P5 statistic
(probe-subspace Varimax coordinate kurtosis, `e12_varimax_p5.py` procedure) recomputed on the
ACTUAL E18 lane (e17c, student.z.embed, train500):

| arm | kurt Varimax mean/med | kurt slice mean/med | kurt coord mean/med | ν(Varimax mean) | ν(slice mean) |
|---|---|---|---|---|---|
| **e17c** (E18 lane) | **1.428** / 0.724 | 0.534 / 0.357 | 1.008 / 0.621 | **8.2** | 15.24 |
| e12c1 (repro check) | 1.063 / 0.673 | 0.371 / 0.204 | 0.626 / 0.447 | 9.65 (≈ rough 9.7 ✓) | 20.16 |

**Declared: ν = 8.2** (the recipe's estimator on the right lane — fit on the CONVERGED shipped
control `in100.lejepa.s0.e17c.ext`, ep100, embed tap, train500 store; ν_slice 15.24 = the same
checkpoint's random-slice kurtosis, moment-inverted).

**Scope caveat (Berker 2026-07-17):** fitting ν from a converged control checkpoint is CIRCULAR
as a method recipe (you need the trained control before you can declare the prior). E18 uses it
DIAGNOSTICALLY — the question is whether a tail-matched prior removes the tax, not how to pick ν
a priori. If the declared-prior fork survives E18, the a-priori-ν question (cheap early-run
statistics? architecture prior?) becomes its own design problem.

**Interpretive key, recorded before numbers (the slice datum):** the term enforces on SLICES, and
e17c's slices are only half as leptokurtic as its Varimax coordinates (0.534 vs 1.428 — partial
CLT washing ⇒ h is not exactly elliptical; no spherical prior fits both bases at once). The t_8.2
target asks slices for excess kurtosis 6/(8.2−4) = 1.43 where the data has 0.53: at init the
tail pressure REVERSES SIGN (Gaussian squashed tails; t_8.2 fattens them) rather than vanishing.
The zero-initial-tail-gradient member of the family is ν_slice ≈ 15 — kept as the queued
discriminator arm if the ν=8.2 read is ambiguous (see P-C).

## Pre-registered directional predictions (locked 2026-07-17, before any E18 number)

Outcome space agreed with Berker (HANDOVER iv) + the attribution plan:

- **P-A — tax persists** (knn ≈ −4 vs ctrl, mean class-pair d′ down ≈ 15%, lin ≤ ctrl):
  **shape enforcement per se harms at h** — T6's strongest form. Unimodality/anti-bimodality is
  shared by every member of the family; declaring better tails doesn't return the class structure
  the shape channel spends. Attribution split (battery, pre-declared): d′ drops while tail
  metrics (kurt_topeig/kurt_worst, radial_gauss vs its gauss-null) move heavier-or-flat ⇒ the
  damage ran through the anti-bimodality channel, not tail direction.
- **P-B — tax shrinks toward f2** (knn → positive, d′ flat-or-up, cone still removed, effrank up):
  **wrong-prior story** — the Gaussian target's tail-squash was the harm; direct support for the
  framework's declared-prior fork (declare, then enforce, the prior the data's diagnostics ask
  for).
- **P-C — tax worsens or mutates** (knn ≤ sigreg's −4.3 AND kurt_worst/radial tails spike above
  ctrl, norm-outlier structure grows): **overshoot** — the coordinate-fit ν over-declares slice
  tails (the slice datum above); the harm is again wrong-prior but in the opposite direction.
  Next discriminator = ν_slice ≈ 15 (zero initial tail gradient ⇒ pure shape-anchoring test).

**Primary directional pick (Claude, from T3's measured mechanism — the d′-collapse gradient is
target-agnostic): P-A, with the tail-side metrics partially relieved** (kurt penalties fall, d′/
knn tax remains). Berker's fork deliberately left the direction open; this line is the card's
committed prediction, not a joint conclusion.

Common to all branches: cone removal expected (the t target still sees the mean — rand@h → ~0);
head jump collapses as in every mean-seeing arm (T1); z-loss equal-or-worse (T2).

## Rider R1 — sigreg@3% (LOW priority; retires E17's over-dose flag)

`in100.lejepa.s0.hpull_sigreg3`: the EXACT E17 `hpull_sigreg` code path at 3% pull instead of
10% — h_lamb = 0.0011289 (= 0.3 × 0.003763; same measured g_enc 170.74, job 62341124; no new
pull needed). Scope: E17-T3 carries the flag "sigreg over-dose (POS .615 mid-run); lower-dose tax
unmapped." Pre-registered directions: (i) tax ∝ dose (knn tax shrinks ≈ proportionally, POS@h
recovers toward control) ⇒ the E17 sigreg number overstates the shape tax, T3's magnitude gets a
dose asterisk but the SIGN conclusion stands; (ii) tax survives ≈ undiminished at 3% ⇒ the shape
channel taxes at any tested dose — strengthens T6 exactly where the flag weakened it. Read
jointly with the E18 main arm (a 2×2: {Gaussian, t_8.2} × {10%, 3%} minus one cell).

## Kill criteria (carried from E17)

- **K1** ignition/INCIDENT (grad-norm > 100× running median) → stop; per-step forensics at onset.
- **K2** converged-linear tax > 3 pts vs control at tap AND h → dose too high; re-read the dose
  curve before any second arm.

## Dose measurement record (mechanical fill at launch — not interpretation)

Rule: h-term weighted pull = 10% of the shipped z-term's weighted pull (weight × g_enc) on the
real first batch, seed 0 (`e12h_pull.py`), launched verbatim. Pull job: 62388918 (tag e18pull).

| arm | shipped z-term (weight × g_enc) | h-term unit g_enc | launch dose |
|---|---|---|---|
| hpull_sigreg_t | SIGReg@proj 0.02×321.247 = 6.42494 (job 62388918; reproduces E17's 321.25) | h_sigreg_t 165.656 (loss 33.25 vs Gaussian arm's 39.0 — the raw embed starts closer to the t-target) | **h_lamb = 0.003879** (10%) |
| hpull_sigreg3 (rider) | SIGReg@proj 0.02×321.25 = 6.425 (E17 record) | h_sigreg 170.74 (E17 record) | **h_lamb = 0.0011289** (3%) |

## Trajectory scoring requirement (Berker 2026-07-17 — dose matching is init-only)

The 10%-pull match holds at the measured first batch only; Gaussian and t_ν CF errors have
different curvature/saturation, so realized pressure can diverge over training. The scoring
pass therefore records, at cadence ckpts ep25/50/75/100 (both arms + control), NOT just
converged probes: (1) per-term encoder-grad norms — pull measurement re-run with checkpoint
loading (small `e12h_pull.py` extension: init from a saved ckpt); (2) the realized CF residual
BY KNOT (the tail knots carry the shape channel — per-knot err(t_k) on stored embed features
against both targets); (3) the geometry trajectory: pos/rand view triple, effrank, class-pair
d′, connectivity margins. Cadence extractions of both arms at scoring time (c015-style).

## Discipline (unchanged from E17)

Pre-registration before numbers (this file) · dose by measured pull, recorded before launch ·
2-ep smoke at launch dose → 3×8h chain · per-lane grad_clip (lejepa: none, port-exact) ·
grad-norm kill-trigger standing · self-explanatory run names · ≤2 H100 singleton slots
(chains: slotA = sigreg_t, slotB = rider) · numbers land raw; NO takeaway without Berker ·
extraction declaration-agnostic (`adapter=native h_layers=[3,6,9] orbit_v=8 do_eval=true
do_pairs=false bs=128`, run_id `<run>.ext`) → probe → battery (gauss_kl_full + radial_gauss enter
automatically, D-040) → `e17_score.py` re-pointed (arm vs e17c) + centered probe + class-pair d′.

## Execution log (in-flight state; not results)

- 2026-07-17: ν fit landed (62388673, table above). `t_nu_cf` + `sigreg_t` routing landed in
  `sslgap/methods/lejepa.py`, CPU-validated (record above). Pull job 62388918 landed (dose table
  filled; z-pull reproduces the E17 record byte-for-byte).
- 2026-07-17: 2-ep smokes PASSED — 62388936 sigreg_t (.0598/.0950), 62388937 sigreg3
  (.0596/.0994); E17 sigreg smoke was .0574/.0852 — in family; no INCIDENT; live wandb check:
  `h_sigreg_t` logged and optimizing (33.25 init → ~6 mid-ep2 at 10% pull), `h_sigreg`@3% at
  ~14.8 (weaker pull, less optimized — expected ordering).
- 2026-07-17: chains LAUNCHED — slotA `hpull_sigreg_t` 62389089/90/91, slotB `hpull_sigreg3`
  62389092/93/94 (3×8h singleton links each, gpu100/H100). Both runs resume-safe from `_last.pt`;
  add a 4th link if ep100 isn't reached.
- Next after ep100: extraction (declaration-agnostic, run_id `<run>.ext`) → probe → battery →
  `e17_score.py` re-pointed + centered probe + class-pair d′ → RAW numbers to Berker.

## Numbers land below this line as they arrive; AGREED TAKEAWAY only after joint discussion.

### 2026-07-18 — ep100 landed; per-knot CF residual instrument BUILT + first read (RAW)

Both chains landed ep100 on link 2 (sigreg_t online best .5056 · sigreg3 .5080; no
incidents). Extraction correction on record: the first extraction pass (jobs 62409675/76)
used store defaults, NOT the card-declared spec — caught against the card, stores deleted,
re-extracted verbatim (`h_layers=[3,6,9] orbit_v=8 do_eval=true do_pairs=false bs=128`, jobs
62409739/40); the four probe/audit jobs submitted against the wrong stores were cancelled
before producing rows.

**Per-knot CF residual instrument** (card requirement (2)): `experiments/e18_knot_residual.py`
→ `results/diag/e18_knot_residual.csv` + `results/figures/e18/e18_knot_residual.png` (job
62409693). Sliced empirical CF at embed vs BOTH targets at the 17 knots (the exact SIGReg
integrand pre-window) + slice excess kurtosis; 4×1024 fresh slices, seed-0 batches. RAW @ep100:

| run | slice-κ (excess) | err@t=3 vs gauss | vs t_8.2 |
|---|---|---|---|
| ctrl | +0.339 | 4.99e-1 | 4.71e-1 |
| hpull_sigreg (gauss 10%) | +0.244 | 4.48e-3 | 4.66e-3 |
| **hpull_sigreg_t (t_8.2 10%)** | **+0.996** | 9.15e-3 | **6.34e-3** |
| hpull_sigreg3 (gauss 3%) | +0.298 | 5.36e-3 | 5.38e-3 |

Mechanical reads (no takeaway): (i) the D-041 interpretive-key prediction REALIZED — the
t-target REVERSES the tail-pressure sign in the trained state: gauss arm pushes slice-κ DOWN
(.34→.24), the t arm pushes it UP (.34→+1.00, toward the declared slice marginal's
6/(ν−4)=1.43); (ii) each arm fits its OWN declared target best at the tail knots (sigreg_t:
6.3e-3 vs t-target < 9.2e-3 vs gauss; gauss arms the reverse); (iii) all arms sit ~100× below
ctrl's residual at t=3. P-A/P-B/P-C resolution awaits the offline probes (tax-persists vs
tax-shrinks) after the corrected extraction.

### 2026-07-18 — first scoring pass (RAW + mechanical direction check; v2 probes converged,
### best_ep 579/700 · 402/523; NO takeaway)

Headline pair at DECLARED h (student.z.embed), linear_raw_v2 / knn_v1_k200:

| run | lin_v2 | knn200 | Δlin | Δknn |
|---|---|---|---|---|
| e17c ctrl | .6036 | .5238 | — | — |
| hpull_sigreg (gauss @10%, E17) | .5894 | .4812 | −1.4 | −4.3 |
| **hpull_sigreg_t (t_8.2 @10%)** | .5714 | .4760 | **−3.2** | **−4.8** |
| **hpull_sigreg3 (gauss @3%)** | .5596 | .4766 | **−4.4** | **−4.7** |

Direction check vs the locked predictions: **P-A branch realized (tax PERSISTS)** — swapping
the declared prior to the lane-fit t_8.2 does NOT relieve the shape tax (knn −4.8 ≈ gauss
−4.3; lin worsens to −3.2), while the knot instrument (above) confirms the arm DID move its
slices toward the declared prior — the tax is therefore not wrong-prior mis-fit at this
resolution (P-B dead: no shrink toward f2's +7.5; P-C's overshoot signature not matched —
tails moved TOWARD the target, no spike past it, no kill). **Rider R1: the tax-survives
branch** — at 3% dose the knn tax is UNCHANGED (−4.7 vs −4.3 at 10%): the T3 over-dose flag
does NOT explain the sigreg tax (T6-strengthening direction). Single-seed oddity flagged
raw: sigreg3's LIN tax (−4.4) exceeds the 10% arm's (−1.4) — dose-inverted on the linear
column only; both probes converged; no interpretation attached.

**Deep-scoring pass (e17_score re-pointed + centered probe, job 62410174; RAW).** The
pre-declared d′-vs-tail-metrics attribution split for P-A resolves at the tail-metrics side:
sigreg_t ≈ sigreg on EVERY head-burden/cone/moment column — inv-jump +.129 vs +.145 (ctrl
+.539; both erase it), cone@h rand .047/.048 (mu_share .037/.043), diagKL@h .087/.068 (ctrl
.956), effrank 28.6/31.4, head σmax [31.0,24.7,3.1] vs [31.4,26.5,3.1] — while the ONE
diverging column is **kurt_worst: 8.53 (sigreg_t) vs 2.86 (sigreg) vs 2.76 (sigreg3) vs ctrl**
— the declared heavy-tail prior's imprint at full-D (pairs with slice-κ +1.00): the arm
OBEYED its target and the tax did not move. Centered-probe rows appended
(results/diag/e17_centered.csv: sigreg_t knn_c 47.6 / sigreg3 47.7 — the tax is not
mean-carried). Cadence pulls landed (e18tp.* ×24 rows): the t-arm's h-term share
self-amplifies f2-style as the z-side g collapses (g_sigreg 321→1.60 by ep100; h share
~10%-of-z-pull at init → ~32% of lane total at ep100; same pattern in the 3% rider). All
E18 card-mandated instruments/records now filled; P-A/P-B/P-C + T3/T6 consequences = joint
discussion.

## AGREED TAKEAWAY

**E18-T1 (USER-APPROVED 2026-07-19; full row in DECISIONS):** failed rescue — the declared-
prior swap does not relieve the h-shape tax though the arm obeys its prior (knot instrument);
3% rider kills the over-dose account; T6 strengthened (shape-per-se). Candidate mechanism on
record: h-CF-shape terms impede inv optimization (same-batch inv@ep100: sigreg_t .0742 /
sigreg3 .0760 vs e20f .0461 / f2 .0500) — insufficient alone (inv-matched epochs still favor
e20f). Mechanism OPEN; deeper analysis owed.

## Deeper analysis (the OWED mechanism; opened 2026-07-19, priority-list item iv)

**Instrument 1 — within-lane opposition grid** (`experiments/e18_opposition.py`, job
62421061; e20_opposition mold): at matched states — the four arm lineages (sigreg_t ·
sigreg3 · e20f · f2) + the control, ep25 (formation) and ep100 (converged) — the cosine
between each candidate h-term's trunk-gradient and the lane's inv trunk-gradient, for all
THREE functionals through the method's own cfg.h_reg paths (moment · sigreg · sigreg_t,
ν=8.2), same seed-0 batch, unweighted norms (e12h_pull convention). Separates
term-functional from state: at the SAME state, is the CF-shape gradient more inv-opposed
than the floor's?

Pre-registered directions (Claude's pick: P-opp-A at ep25, fading by ep100):
- **P-opp-A** cos(CF, inv) < cos(floor, inv) at matched states — first-order gradient
  opposition carries the impede-inv account.
- **P-opp-B** cosines ≈ equal — the impediment is not first-order opposition (curvature /
  conditioning / slice-noise channel next).
- **P-opp-C** floor MORE opposed yet wins on inv — opposition refuted as the mechanism
  (the E20 opposition-read lesson repeating: cosine ≠ outcome).

Numbers land below as they arrive; AGREED TAKEAWAY only after joint discussion.

**Instrument 2 — inv-vs-val phase plot LANDED** (`experiments/e18_inv_phase.py` →
`results/figures/e18/e18_inv_phase.png`; online-probe y-axis, E12-T9 caveat rides):
val interpolated at matched inv — inv=.10: e20f 47.0 / f2 47.1 vs sigreg_t 35.9 / sigreg3
33.7 / sigreg 37.3 / control 36.8 · inv=.08: 55.4/56.4 vs 46.0/46.1/47.7/45.6 · the CF
cells NEVER reach inv ≤ .06 (stall above it) while the floors pass .05 (e20f 67.9). Two raw
facts now separated on one figure: (1) CF-shape terms stall inv above ~.07 (the impediment
half of E18-T1's candidate mechanism); (2) at MATCHED inv the CF cells sit ON the control's
val-per-inv line while the floor cells sit ~+10 ABOVE both (≈+7 net of the E12-T9 ~+3
monitor bias) — the floor's gain is a val-per-inv lift the CF terms never had; the E18 tax
vs f2 reads as impeded-inv, not as a lost non-inv channel. Joint-discussion material only.

**Instrument 1 LANDED** (e18_opposition.csv, job 62421061; 30 rows = 5 states × ep25/100 ×
3 functionals). Mechanical score vs the pre-registered directions: **P-opp-A REFUTED** — at
matched states cos(CF, inv) is never meaningfully more negative than cos(floor, inv); all
|cos| ≤ .20 (no first-order antagonism channel at all), and the small effects run the OTHER
way: at ep100 states the FLOOR is the more inv-opposed functional (moment −.12/−.12/−.20/−.08
across the four arm states vs CF −.01..−.08) yet the floors hold the lane's best inv — the
E20 opposition-read lesson (cosine ≠ outcome) reproduced within-lane ⇒ the outcome mixes
P-opp-B (no first-order opposition) with P-opp-C's direction. Secondary raw facts: sigreg vs
sigreg_t gradient directions nearly identical at every state (Δcos ≤ .03 — the declared-
prior swap barely rotates the CF gradient; the E18-T1 null in gradient form); unweighted
|g_CF| = 7–49 vs |g_floor| = .17–.44 (the λ-normalization the dose law exists for); control
states mildly inv-aligned for all three (+.02..+.07). Joint with Instrument 2: the
impede-inv mechanism is NOT gradient opposition — remaining candidates for discussion:
slice-noise in the CF estimator (fresh random projections per step ⇒ noisy effective
gradient at matched |step|), and curvature/conditioning of the CF objective near the
constraint surface. NO takeaway without Berker.
