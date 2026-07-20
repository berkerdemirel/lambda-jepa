# HISTORY — append-only findings & lessons log
(append entries; never rewrite. Sections older than the active program are periodically moved
VERBATIM to HISTORY_ARCHIVE.md per DECISIONS D-053 — the index below lists them. Full result
tables live in results/*/FINDINGS.md)

## Archived sections (2026-07-02 … 2026-07-14, verbatim in HISTORY_ARCHIVE.md)

- 07-02 — project kickstart · M0 parity hunt · M0 parity PASSED · decision ratifications +
  space-definition revision · F1–F4 resolved, D-006v2 · G-M0 PASSED, AUDIT_MATRIX v1 LOCKED ·
  M1 full-roster implementation · faithfulness review vs donors · incident: stale pre-review
  checkpoints · incident: DINO probe on misaligned labels
- 07-03 — LeJEPA-800 portval PASSES (trainer stack certified) · DINO probe incident CLOSED
- 07-08 — theory day: anchor verification; Dubois '22 deep-read + positioning
- 07-09 — probe-convergence incident (D-020) · E10 rescue campaign (mechanism, scaling law,
  E10-T1) · D-020 re-probe fleet complete (33/33)
- 07-11 — INCIDENT (protocol): house config likely not near-optimal for lejepa at IN-100
- 07-11/12 — the E12 saga: thesis → lazy-projector catch → K4 → f2 → G-wave
- 07-12 — PIVOT day: staged merge (D-029), E13 rung-0 built+run+concluded, T1–T3
- 07-13 — PIVOT training rung: E15 + E16 KILLED; the distillation-ceiling lesson (D-032…D-034)
- 07-13/14 — G-wave concluded (T7/T8); instrument corrections; H-wave; h REDECLARED (D-036)
- 07-14 — D-036 EXECUTED (F1 GAP→CLS, monitor-tap fix, CLS migration); PIVOT ceiling (D-038)

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

## 2026-07-19b — the floorssl z-collapse day (E20 comparison pass + E18 mechanism + D-047/D-048)

**Delivered:** (1) E20 comparison pass complete (card §Comparison pass): anatomy cadence
over 32 ckpts (instrument grew dino multi-crop handling) — cone+scale won zoo-wide, aniso
the contested channel, R ×2–3 (floor spends h-invariance), z untouched except simclr raw-z
scale ×20; battery-vs-ctrl table + e20_score deep pass (inv-jump shrinks in the 4 winning
lanes, INVERTS in dino — h-margin .537→.329) + class-pair d′ (moves OPPOSITE to probe
deltas in both directions) + centered fig. (2) E18 owed mechanism: opposition grid (P-opp-A
refuted — no gradient antagonism, |cos|≤.2; ν-swap rotates the CF gradient ≤.03) + inv-val
phase plot (CF cells stall inv ON the control's val-per-inv line; floors +10 at matched
inv). (3) Chain-handoff hardening: smoke→chain via SLURM-native afterok+singleton (the
in-session watchers died with the previous session — ping-only, no losses).

**The collapse:** both lejepa-V4 floorssl cells (bn + nobn heads) hit the pre-registered
kill at ep3→4 via a SMOOTH info-starved equilibrium (inv sacrificed; no incident) while v2
(same doses, byol pair) was healthy. Frame-bridge pull (hold v2_best, swap ONLY augs):
g_inv −24% under lejepa V=4, floors frame-invariant; equal-pull re-dose (D-047: 32.8/43.8/
.617) restored v2's 35% inv share and STILL collapsed one epoch later (laug2 .0944@3→
.0728@4; nobn2 fell at 3) ⇒ **dose axis refuted under strong views**. D-048 fork response:
nobn2/laug2 killed at their forks (no ep10 waits — Berker's no-wasted-compute directive);
laug_eps launched (z-floor demoted to the certified 6% share; inv owns z 91%) = the
deferred Z5 cell, running at close. Commensurability read behind Berker's inv concern:
raw inv values are not cross-lane comparable — R at the loss space is; even HEALTHY v2
runs R@z .61 vs f2's aligned .15, and healthy z spectra are anisotropic (vicreg 22% effrank,
kurt 245) EXCEPT lejepa's proj (99.3% — fully isotropized at 2% share and still healthy).

**Instrument/process lessons:** init pulls CANNOT see aug-family differences (measured:
near-identical term values on a random trunk; the frame difference lives at formation
states — bridge at a healthy formation state with augs swapped, the e21_pull pattern);
MomentFloor's own E12 docstring warning ("destination weights scrub class structure") was
the collapse's h-space precedent; the 2×2 synthesis (demand-type × share: every healthy
cell is barrier×owner or shape×minority, every floorssl failure shape×owner) = the fix
session's map, SESSION_OPENER.md. New instruments: e20_battery_table · e20_score ·
e20_pair_dprime · e20_centered_fig · e21_curves · e21_pull (frame bridge) ·
e18_opposition · e18_inv_phase.
