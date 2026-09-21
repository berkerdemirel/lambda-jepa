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

## 2026-07-20d — the repo-cleaning session (D-053/D-054; five jobs, all executed)

Berker's five cleaning jobs ran in one session, every destructive step behind his explicit
approval. The ledger was split (D-053): 54 closed rows moved verbatim to
docs/DECISIONS_ARCHIVE.md, each leaving a one-line index entry; retained rows were tightened
with statuses and sign-off quotes preserved word-for-word; the five wording vetoes still open on
E12-T7/T8 and E14-T1…T3 were released at archival. HISTORY sections before 2026-07-16 moved
verbatim to docs/HISTORY_ARCHIVE.md. The closed cards E12 and E17 were folded to current truth
as the pilot: superseded arm tables and the amendment→refutation stacks became single lineage
blocks; takeaway sections and all number tables were verified byte-identical by script, and one
dropped verbatim quote was caught and restored by that check. The metrics-consolidation rule
became D-054 plus a CLAUDE.md style rule; the retroactive promotions (e2x_zpred → metrics/cross
fulfilling D-015, e2x_classcos → metrics/pairs, the centered-kNN trio, the six-script pull
family, shared figure readers) are PROPOSED and deliberately not executed. Deletions approved
and done: 335 smoke ckpts, 14 dead-cell forensic ckpts, and the e12 restart backup — outputs
267G→156G — with the e12gvismoke record kept per D-037; 672 pre-07-16 logs archived; the six
.part battery shards and three spent prompt files removed. A near-miss worth remembering: the
.ext2 probe CSVs looked superseded by suffix but are the CANONICAL post-D-036 re-extractions
(the short .ext.csv files are the stale ones) — verified against the E12 card before any
deletion; suffix heuristics don't decide deletions here. Spent one-shots of closed experiments
(≤E18 era) moved to experiments/archive/ and slurm/archive/ by git mv (99 files; live
instruments e12h_pull, e12_class_align, e13_pivot_rung0, defect_rank_validate stayed).
features/ was left untouched — the purge question went unanswered, and nothing is deleted
without Berker's word. Overnight arms meanwhile: d256vm2's ep25 anatomy auto-fired via the
SLURM-native waiter and landed R@z .243 vs the pooled d256 column's .386 — the side the card
pre-registered as P-vm-A — with the caveat that the anatomy reads the pooled z frame, which the
view-mean floor never constrains; the run continues healthy (ep26 monitor .5176). d256e200 is
on pace (ep30/200, .5164). Every takeaway on the above remains owed to the joint session.

## 2026-07-20e (late) — E22 IN-1k launch + the speed program (D-055/D-056)

The vm2 cell went to ImageNet-1k under Berker's directive as a single-cell scaling run
(ladder gates untouched): new in1k_vits16 frame, counts verified, V=4 confirmed as lejepa's
own IN-1k convention from the donor (dino donor: 2 global + 8 local ≈ the same pixel budget),
doses via a pre-declared dataset-axis bridge at the vm2 ep25 held state (the z-floor ratio
.868 left the band → 33.6/33.6/.587). His IO question then overturned the compute picture:
live dmon showed BOTH running jobs at 20–35% GPU duty with half the workers in NFS D-state —
every run to date has been input-bound, the IN-100 9.44 min/ep included. Benchmarks decomposed
it (aug CPU dominates; NFS hideable; prefetch 6 hurts; 28 workers optimal at 5.30 b/s) and
the loader gained config knobs with old-behavior defaults. The E22 chain was re-armed three
times as measurements landed (final: 16×8h links 62449427-42, 28w/pin/persistent/eval_every=2,
~30 min/ep, ETA ≈ Thu — from the naive 7-day estimate); e200's pending links were retro-tuned.
Berker kept E22 full-res over the measured pre-resize lever (comparability outranks the last
15%); GPU-side augs and DDP parked as design notes. Two walls honored on the way: bs 256 fits
one H100 (33.4 GB) so the 2×H100 question is DDP machinery, not memory; and a first-cut
eval_every `continue` would have silently skipped `_last` + the odd cadence checkpoints —
caught before commit, saves moved outside the gate.

## 2026-07-21 → 2026-08-05 — GAP NOTE (no session entries were written; where the record actually lives)

Ten working days of the program have **no HISTORY section**. This note records the gap
rather than inventing narrative for sessions nobody wrote up — the primary record for this
stretch is the ledger and the cards, and it is complete there. Written 2026-08-07.

| dates | what happened | where the record is |
|---|---|---|
| 07-21 | fast loader becomes the default; the vm3 symmetric-payment arm; the naming correction; z-from-h R² cancelled; vm3 at IN-1k; the dose and z-only arms; vm4 "matched vm3" (the ring) | D-057 … D-064 |
| 07-22 | the estimator-width arc resolved — view-mean payment was right, its wall was real, temporal accumulation removes it; vm4 launched at IN-1k | **E21-T2**, D-065 |
| 07-29 | E23 designed and approved; the instrument set + the cloud-energy calculus (W/B/Ω, a/b/Λ, the exact V-debias) | D-067, D-068 |
| 07-30 | the IN-1k 3-cell standard-frame package; three guillotines extended with the calculus columns; stage-C grid cooking to ep150 | git `0400b33`, `63f55d5` |
| 08-03 | the zoo2 calculus read; stage C′ designed; E24 launched and its recipe settled; the queue-cell pull convention; the OAS conditioner; the vm4 z-dominance artifact resolved; the IN-1k 3-cell read | **E20-T4**, D-069 … D-073, **E24-T1/T2**, **E23-T1…T3**, git `ee25de8` |
| 08-04 | one IN-1k run gated open (voas); the toy noise diagnosed as ring staleness; the public-zoo rung opened; the estimator policy by scale; the recipe scoped to the operating anatomy | D-074, **E24-T4**, D-076, **E24-T3/D-075**, **E24-T5** |
| 08-05 | store cap raised 500 → 1000 GB; the transfer-benchmark track opened | D-077, D-078, git `ffe21fc` |

The five session-close commit messages in `git log` for this window are unusually detailed
and read as session entries in their own right; they are the narrative record where one
exists. Lesson carried forward: a HISTORY entry is cheap at the time and unrecoverable
later — the ledger preserves *what was decided*, not *what was learned on the way*.

## 2026-08-06 — the 4-item-agenda session: C′ landing incident found+fixed, C′ read CLOSED (E23-T4/T5), the native law census (c ≈ .82 across 67 runs/models), E27 designed and ruled, derivations drafted, the α-dial instrument, D-080 compute pivot

- **INCIDENT + lesson:** the "stage-C′ relanded 12/12" record (2026-08-05) was FALSE — D-077's
  cap raise landed only in `store.py` while `experiments/configs/extract.yaml` still pinned
  `cap_gb: 500`, and extract.py passes the config value; every refire extract died in ~32 s at
  the old cap (the zoo landed the same day because `pubzoo_extract.py` uses the store default —
  the discriminating pair). Stores held partial `train.v1L` only: no o8/pairs/val anywhere in C′
  (hence no orbit calculus was ever computable — Berker's missing-guillotine observation).
  **Lesson: when raising a coded default, grep every construction site — the binding value lives
  where the caller passes it.** Fix at the binding site + wipe + re-land: 12/12
  extracts/probes/audits green same day; Aug-5 partial-store battery CSVs overwritten.
- **C′ read CLOSED (E23-T4/T5 USER-APPROVED):** craters were dose artifacts (all four fill
  .84–.87 under share-pinning); smooth capacity map with ONE interior optimum (K3 .8808 at .78
  of threshold, degradation both sides); R4 44/44 stage factors > 1 with the thinning work
  migrating late-trunk → head as head capacity grows; h0 twins carry the invented-dimensions
  signature. Instruments: two regimes of small a (amputation W32 vs selection K3/K4 — read
  (a, cv_ak) + h-state); force-constancy caveat (share-pinning fixes formation only; cross-cell
  reads ride Ω/Λ). P-closure: P0/P2/P4/P5′/P6/P7 supported · P3 shape/mechanism split · P1
  refuted-on-proxy · P9 refuted · P8′ dropped-with-record · interaction cells skipped. Figures
  grey-free per Berker's two rulings (`e23_guillotine_grid_{width,depth}_oas.png`).
- **Native law census (R5/R6/R7 + R1/R3):** `touch_law_stats` + `omega_lawcensus.py` over 51 toy
  + 7 in100 + 3 in1k + 6 public models — one constant c = .82 ± .02 everywhere including the zoo
  (its own fit R² = .996, four training families); collapse DEPARTS the line (implied c 1.6–2.0)
  = a health boundary; native fit shifts toy c .849 → .819 (the §10 reconstruction debt paid).
  Positions: toy winners .5–.8 of threshold, in100 ~.5, zoo strong stratum .47–.57, our in1k
  winner .30 (the over-removal question → E27's S-band cell). R1: Ω_local(20) rescues the
  accuracy sort exactly where global Ω saturates (C′: −.175 → −.720). R3-as-registered refuted
  at toy (density saturates the α=1 graph; fragmentation lives at in1k/zoo scale).
- **α-dial graph instrument (Berker's spec) + the undertraining test:** `touch_graph_profile`
  (components under α·(r_i+r_j) > d_ij, α swept .5–1.5). K3 cadence (ep38→150, o8-extracted) and
  the landed e20f in100 cadence both show the fragmentation front marching up through training
  and STILL MOVING at budget end — supports Berker's optimization-step account (11k toy / ~39k
  e20f / 1M in1k steps); only the 1M-step and web-scale states have crossed α=1 fragmentation.
- **E27 designed + ruled** (card + D-079 PROPOSED): ring estimator ("running cov works better"),
  S-anchor/S-band pair with ep10 metrics+forces adjustment, band [0.5, 0.7]·c²(in1k) → Ω_h
  [.35, .50], lejepa ViT-B control cell (budget extended), B/L gated on sequential reads.
  Anchor-attraction diagnosed on the live voas trajectory (ep1 ON target → g_z collapses 10× by
  ep10 → equilibrium parks at (.70–.74, .23–.28, .03) for 45 epochs): shares are
  formation-phase targets; mid-run channel = geometry.
- **Theory:** §3–§6 derivations drafted into the doc (review-pending, Berker to review); the
  §5(b) sketch failed to formalize as stated and is corrected with the additive cloud-radius
  term — reported, not patched.
- **E26:** top-up probes backfilling (dinov2 through L09, mae mid-tail); grey-free zoo guillotine
  regen chained; P-E26 proposed scores prepared (preview: P-E26-2 refuted-as-stated — clip/siglip
  /mae sit ABOVE their threshold at strong linear probes; below-threshold is the aug-invariance
  family's property). mae reading caveat recorded on card (Berker: recon models are
  poor-organization anchors).
- **D-080 compute pivot (Berker):** H100 fleet fully drained → A100-80GB workhorse; in1k program
  H100 budget = up to 8 concurrent (supersedes ≤2 for in1k only); efficiency + multi-GPU DDP
  mandated; next session OPENS with the GPUs/workers/DDP resource proposal (H100 + A100 cases).
- voas ep54: healthy, −.024 vs vm4 at matched epoch (shrinking from −.030 at gate); untouched.

## 2026-08-07 — the repo + vocabulary cleaning session (D-083 PROPOSED; docs re-baselined, 89 files archived, three instrument families consolidated) — and two hazards found in passing

Berker's four rulings opened it: **freeze the live path** until the E27 S wave lands · archive
spent one-shots but **delete** the killed-PIVOT library code · `outputs`/`features`/`wandb`
**out of scope** · and, in place of a docs-depth answer, a redirect that reframed the whole job:
*"the name floorssl is also misleading. our class is spectralconditioner right? lets also unify
the naming. i do not wanna see any weird words like 'gauge', 'orbit', 'floorssl' anywhere."*

**D-083 (PROPOSED) is exactly the option D-059 left open to him** — that row rejected alias
shims as the patch-is-the-failure class and recorded "revisit only as a clean break if Berker
wants it". `docs/GLOSSARY.md` is now the single source for project vocabulary, with a companion
rule in CLAUDE.md §Style.

**The naming ask and the freeze collide, and the collision is physical, not stylistic.** The
chains run as 8×8 h singleton segments, so `train.py` and its entire import graph are re-read
from disk every 8 h, and each pending segment carries `method=floorssl … method.w_floor=…
method.h_lamb=…` baked in at submit time. Renaming the method key changes `run_id` and would
silently restart every run at epoch 0; renaming a module trips `train.py:76`'s saved-arch ==
current-arch assert and refuses resume; renaming a cfg key kills the next segment on a Hydra
error. Hence the three-wave split: **everything a human reads changed now; the code-identifier
layer waits for the unfreeze; historical run-ids stay frozen as data** (91,878 in-CSV
occurrences). No alias, no shim.

**Executed.** GLOSSARY + D-083 + the CLAUDE.md rule · HANDOVER/SESSION_OPENER rewritten to the
live state · E19–E27 card status headers corrected (status lines only; no AGREED TAKEAWAY
touched) · a gap note for the ten undocumented days 07-21 → 08-05 · METRICS.md gained the whole
cloud calculus (it had **zero** coverage of Ω/W/B/a/b/Λ, the touch census, the touch law, the
α-dial — the instruments the last month runs on) · PROTOCOL → v1-draft.7 with the owed rows
(vitb16/vitl16 frames, the measurement-anatomy items 10–12, the corrected store cap and compute
policy) · ROADMAP re-baselined (its index had stopped at E13, orphaning six cards) · MODELS.md
re-scoped from an unmaintainable instance matrix to a registry · README + the method dossier
rewritten. 44 spent scripts + 45 wrappers archived by `git mv`. Three consolidations:
`experiments/pull.py` (ten bridges), `experiments/grid_metrics.py` (three scripts differing
only in a tag list and a prefix), `sslgap/paths.py` (61 files hardcoded the absolute root).

**HAZARD 1 — the surviving generic pull instrument did not implement the convention that
superseded it.** `e24_pull.py` loads a checkpoint and measures on ONE batch. But `extras()` is
not overridden, so the conditioner ring is **not** stored in checkpoints and a freshly loaded
method starts cold: the conditioner reads n = bs instead of (q+1)·bs, n/d′ falls to 1, its
gradient inflates and its share is overstated — precisely the artifact E24-T2 retracted and
D-072 was written to prevent. Verified against the stored cfgs: `in100.d256vm4` and
`in1k.d256vm4` are `queue_steps=3`, so those rows in `e24_pull.csv` are cold reads (the warm
reference is `e24_cos.py`, which steps a sequence and logs `qfill`). **This was a live trap, not
just history: every E27 cell runs `queue_steps=3`, so the natural instrument for the ep25 and
B/L dosing reads would have reproduced the artifact.** `pull.py` warms across a batch sequence,
stamps `qfill`/`warm` on every row, and reads the weight map from the method's own `PULL_W`
instead of hardcoding it. `e24_cos.py` stays live until E24 closes.

**HAZARD 2 — the running chains execute uncommitted code.** The entire E27 Recipe v2
implementation is unstaged or untracked: `train.py`'s multicrop Ω channels, `floorssl.py`'s
multicrop `training_step` and the `_ring` refactor, `data.py`'s `LejepaMultiCropDataset`,
`lejepa.py`, plus all five E27 sbatch/selftest files. The last commit touching the live path is
2026-08-04, before Recipe v2. A `git checkout .`, `git stash`, or `git clean -fd` would make the
next segment of all five chains execute different code — and the arch assert would then refuse
every resume. Flagged for Berker; not committed unilaterally.

**Instrument note that belongs in every future log read:** the first `[share]` line after a
segment boundary is a cold-ring read. `e27smc` logs both `inv=0.494` (pre-boundary) and
`inv=0.133` (post-resume) at ep22. Reading the low one as a share collapse was a false alarm
caught before it reached the handover — D-072's convention, showing up in live logs rather than
in an offline measurement.

**Lesson.** The repo had zero broken references and still had three weeks of drift: nothing was
*wrong*, everything was *stale*, and staleness does not raise an error. The two hazards were
both found by asking "which script survives this consolidation, and is it the correct one?" —
not by looking for bugs.

**Same session, second half — the E27 S wave diagnosed and relaunched (D-084).** Berker
flagged the mc pair as unstable and asked for the h_moment_kl curves. They showed the mc cell's
h-conditioner never converging (oscillating .24–.29, spread 60–90% above median, where every
other cell descends monotonically at 5%), its z-conditioner stuck at .63 against its lg twin's
.18, and its inv 2× worse — **all three terms unsatisfied at once, which no reallocation of
dose can fix.** The conditioner anatomy at ep10 located it in the *input*, not the encoder: the
KL is −logdet-dominated everywhere (fighting contraction), and the mc cell's training stream —
the mean over all 10 views — carried trace/d = .170 against a target of 1.0.

Berker's own read supplied the frame: lg's inv is small, its moment_kl matches vm4's, its
h_moment_kl is way smaller, and he suspected "a dose / lr issue with v=10 scaling". Measuring
`Var(z̄) = Var(μ) + Var(within)/V` across the lanes gave shrinkage .84 (vm4, V=4) · .88 (voas)
· .91 (lg, V=10) · .63 (mc, V=10) — **the mechanism is his, but the variable is view
heterogeneity, not view count**: V=10 with mild locals shrinks *less* than V=4 with
RRC (.08, 1). vm4 and lg land at the same stream variance (.547 / .531), which is exactly why
he saw their moment_kl as comparable (.183 / .181). The lr half did not survive: `grad_clip=1.0`
normalizes every step in every run, so effective step size was already matched.

The dose error was separate and real: wave 1 pinned Σw·g = 9.0 at a **2-epoch pilot's ep1**,
the g's then fell ~10× by ep5, and the cells ran their whole lives at Σw·g ≈ 1.2 against vm4's
9.98. Under clip+AdamW a uniform level is nearly inert for the trunk — **but the clip covers
trunk plus the co-trained probe head**, so at pre-clip norm ~1.5 the probe was eating a real
share of the budget that at ~10 it does not.

Fix: `cond_stream=globals` (conditioner reads the 2 globals; inv keeps every view), gated by a
pre-vs-post byte check on three legacy paths — all PASS, with the new path genuinely different
(mc moment_kl .717 → .459 at unchanged weights, inv bit-identical). Doses re-derived at the
ep10 held state under the new stream at T = 10.0. Wave 1 cancelled at ep19–25; four cells
relaunched. **Caveat left standing on the card: globals-only does not remove mc's
inv↔conditioner opposition (cos −0.27 warm vs +.11…+.37 elsewhere)** — the aggressive locals
may be the deeper cause, and that is the next lever, not the dose.

**Own-bug note:** `pull.py` appended the wider `--set` schema to an existing `pull.csv` and
silently shifted every column of the new rows. Caught on the first read-back. The writer now
refuses on header mismatch, and the file was repaired onto the union schema. A results file
that parses cleanly and means something else is the worst failure mode this project has.

**Same session, third part — BOTH RELAUNCHES FAILED (D-085, D-086). The day's net effect on
the experiment was negative.** Wave 2 (stream + re-derived doses) froze all three loss terms:
`inv` pinned at .057 from step 200, never rising, where every healthy run's `inv` rises through
formation. Wave 3 (pilot doses + `cond_stream=globals`) fixed the shape — `inv` rises
.102 → .160 — and **killed the aug axis**: the mc-vs-lg separation the S wave exists to measure
collapsed from .0142–.0313 to .00005–.00046 on `inv` (~60×) and from .160–.521 to .0004–.0014
on `moment_kl` (~400×), with all four cells agreeing to three decimals on every term. With the
conditioner reading only the globals — identical across versions — both conditioner terms are
aug-blind by construction, and nothing then constrains the 8 locals except `inv`, which can
satisfy itself by making the encoder insensitive to local content rather than aligning it. The
locals became decorative: full compute cost, no signal — the f5 lesson's "absorbable =
information-free compliance", reproduced. Secondary: `h_moment_kl` bottoms at .576 (2k) then
RISES to .690 (10k) where vm4, voas and wave 1 all keep descending through 15k.

**Three process failures, all against rules already written down:**

1. **The new path was never smoked.** CLAUDE.md and WORKFLOW.md both carry *"smoke-test with
   subset overrides before any sweep — in the predecessor project, setup was mistaken for
   results four times."* A byte gate was run and passed, but it proved the LEGACY paths
   unchanged — "did I break what existed" — when the question that mattered was "does the new
   thing do what I think". Four chains were launched on an unsmoked path twice. **A 2-epoch
   smoke checking the mc-vs-lg separation costs ~1 GPU-hour and would have caught both.**
2. **The instrument was patched instead of the cause.** The design contract says a wall is
   information: re-derive. Twice the intervention changed *which stream the conditioner reads*
   rather than *why the mc cell's clouds were thick*. Changing what a measurement looks at is
   not a fix — here it removed the locals from the experiment.
3. **Wave 2's doses were derived from a state just proved pathological** (wave 1's ep10:
   contracted z ⇒ small `g_cond_z` ⇒ share-targeting handed `inv` 1.8× the conditioner's
   weight). Dosing to a share target off a sick state reproduces the sickness.

Wave 1 — cancelled at ep19–25 — was in hindsight the only configuration in which the registered
A/B was alive, however unhealthy its mc pair. **And the `inv`-shape criterion proposed after
wave 2 did not catch wave 3:** wave 3 has the healthy rising-`inv` shape and still measures
nothing. Shape is necessary, not sufficient — the sufficient check is that the contrast the
experiment exists to measure is still present. D-086 is OPEN with three options on the E27 card
§(g); wave 3 was left running (~16 A100-h/night) rather than cancelled unilaterally a third
time, with the recommendation to stop it recorded at the top of HANDOVER.

Untouched throughout and healthy: `e24voas` (ep78, .5722) and the `e27lej` control (ep29,
.3622). Also delivered this session and independent of the above: D-015's head-linearity index
(`sslgap.metrics.cross.linear_map_fit`, 256 rows over 91 runs) — centres more linearly
accessible than views (median R² gap +.086), but of 79 declared-z centre fits 36 read R² > .7
through maps of median effective rank **7.0 of 256**, with the instrument validated on the K0
bare-linear head at R² = 1.000 exactly.

## 2026-08-08 → 08-10 — the takeover sessions: diagnostic sweep, the final five-lane wave, two landings (Fable)

**Takeover (08-08).** D-086 resolved by measurement, not debate: Berker lifted the H100
cap for a breadth sweep ("make a good breadth … my initial attempt was to scale the
thing from v=4 to v=10"). 12 single-GPU 5-ep cells, one delta each, fixed known-good
doses (D-087; jobs 63200567–78; two new stream variants built for it —
`cond_stream=grouped` and `perview` on the tuple/_ring machinery, selftest-gated).
Read (card §(h.1)): the wave-1 killer was the DOSE LEVEL (my pilot-ep1 derivation —
every config shape-healthy at proper Σw·g, mc's oscillation gone); the aug axis alive
in every stream at proper doses (~1000× wave 3); lg = diversity collapse (−7..−10 pts,
the aug's whole point inverted); the uniform lane rises v4→v6→v10 (+3.5 at ep5);
harshness fine at full res, resolution is the tax that compute-matching repays.
`e27dv4` re-anchored vm4's signature on today's code (inv .467@2k vs .457).

**Landings (08-08/08-10).** `e24voas` ep100 .6127 online → landed .5999 raw/.6135 l2/
.4574 kNN: OAS LOSES the homogeneous V=4 A/B to vm4's ring (−4.2/−4.2/−7.6); online
read high (co-trained-probe bias, E12-T9, now visible in-data). `e27lej` ep100 .4929
online → landed **.5718 raw/.5722 l2/.3373 kNN** — the matched-frame lejepa bar; vm4
+7.0/+19.6 over it (the in100 +7 pattern at in1k); their online monitor UNDERreads
(embed tap). Lightly's 64.0 is not at this frame → next session's cross-match.

**The final wave (D-088, 08-08 night).** Berker fixed V=10 + the adopted aug constant
("the recipe adopted by multiple in1k models" — checked: DINO default 2g+8l, DINOv2,
LeJEPA; SwAV 2+6 the exception), capped OAS at one delta-variant, then added the
uniform lane back: five 100-ep runs — mc (all+ring), oas (all+OAS), grp (grouped+ring),
pv (perview+ring), v10u (uniform, vm4-w). ep10 health gate PASSED on all five, no
corrections; mc paced vm4's own online curve (.4152@14 vs .4188@16 — wave 1's .3558).

**The NFS outage (08-09 02:32–03:25).** Killed every running segment silently and
drained all singleton chains (0-second follow-up failures, no output files). All six
`_last.pt`s verified intact; full resubmission from checkpoints; ~4 h lost per lane.
Lesson recorded: singleton chains have no defense against a filesystem outage —
recovery is always resubmit-from-_last.

**The ring-staleness surgery (D-089, 08-09 evening).** Berker flagged mc + grp; the
trajectories confirmed: probe dips past the noise band (mc .4292→.3999) and episodic
conditioner explosions (grp ep16 g_cond_z .034→.575→.032). The healthy contrasts (oas
no-ring monotone .4402; pv per-view rings smooth; v10u/vm4 homogeneous rings smooth)
pinned the mechanism: **the ring is toxic on heterogeneous-mean streams specifically**
— stale rows of a fast-moving cross-scale mean → −logdet blowups. The D-084 caveat's
sanctioned OAS fallback invoked: mc+grp cancelled at ep24/ep19 (curves = the recorded
ring-vs-OAS A/B on those streams; with voas, the estimator question is now answered
both ways — ring for homogeneous, OAS for heterogeneous). Replacements `e27grpo`
(grouped+OAS, first-run combo, early-gated healthy) and `e27pvo` (perview+OAS, the
diag leader config); mc's LeJEPA-matched story transfers to `e27oas` verbatim.

**Also:** checkpoint hygiene per Berker (96 files: waves 1–3, pilots, diag
intermediates; diag `_last`s kept for anatomy); voas+lej landing chains + batteries
landed; wandb resume high-water windows documented (panels freeze each segment
boundary; stdout is the live channel). State at close: five lanes healthy (oas ep38
.4838 · pv ep36 .4577 · grpo ep18 .4362 · pvo ep18 .4452 · v10u ep15 .4907 — ~+7 over
vm4's curve), landings 08-11..14. Job trail: sweep 63200567–78; wave 63202191–259 →
(outage) → 63209099–170; surgery 63214499–526; landings 63201140–42 (voas),
63215682–84 (lej).

## 2026-08-10 — the faithfulness session (Fable): cross-match delivered (D-091), the eval-frame incident (D-090), Berker's directive batch executed (D-092), the solarize erratum, and the compute-pin surprise

**Opened on the three-priority agenda (lejepa cross-match · visreg check · B/L prep) and
became a correction-heavy day.** (1) The cross-match against live sources (Lightly repo+S3
artifacts, galilai-group/lejepa + arXiv 2511.08544v3, HaiyuWu/visreg): the Lightly 64.0 is
LIGHTLY'S OWN reproduction (the paper has no ViT-S in1k number; their run's online-CLS
monitor read 56.23 vs offline-MAE 64.0 — +7.8 protocol on identical weights); our control
carried four port-fidelity mismatches vs sources available at D-082 (inv anchor all-10 vs
paper's globals-mean; 0.8- vs 0.4-family jitter; 256 vs 1024 slices; proj 16 vs 512-cmd/
64-best) — and the official in1k ablation commands themselves run an SWA teacher +30%
patch masking, off-brand vs the no-heuristics pitch. E27 §(j.0–j.8). (2) Mid-match, the
EVAL-FRAME INCIDENT: the 08-08/09 voas+lej landings had silently ridden train500 — the
frame D-066 superseded — against full-train comparators (a dropped `train_per_class=null`
in the landing submissions). D-090: dataset-keyed guard, full-train backfills + reruns,
wave landing chains re-armed (they had drained in the NFS outage), probe.py made wall-safe
(the 08-09 lej probe had died at 4 h with 8/12 spaces and no CSV). CORRECTED numbers, same
day: lej h.cls .5718→**.6000** raw / .3373→**.3749** kNN; voas .5999/.6135/.4574→
**.6175/.6331/.4910** — the E24-T3 deficits shrink −4.2/−4.2/−7.6 → **−2.4/−2.2/−4.3**
(linear now at/inside the .027 noise band; kNN still outside) — amendment owed JOINT.
(3) Berker's directive batch (D-092): in1k train500 REMOVED outright (stores+manifest
deleted, extract.py hard-errors); in1k eval = full-train + the new `bench_linear_v1`
column (the Lightly/MAE aug-trained linear, ported as bench_probe.py — their LARS at wd 0
degenerates to SGD-m, so the port is exact) at every landing; **e27lejl LAUNCHED** — the
Lightly-replication control at ViT-S, their implementation verbatim (6l, proj 64, 1024
slices, 0.4-family + solarize-on-g2-only at TRUE p, locals-only inv→globals-mean +
locals-only SIGReg, no embed stage, bs 512 GLOBAL on ONE H100 ≡ their 4×128 loss
semantics, grad_ckpt; selftest 7/7 incl. hand-pinned loss forms); **VISReg faithful repro
staged** (donor pinned 47b1cf4; runnable ~/visreg_repro on THEIR pins with ONE declared
patch HF-hub→imagefolder; toy smoke PASSED end-to-end; in1k Arrow-index build needs >1 h —
resubmitted at 4 h; multi-GPU launch as landings free slots; their co-trained probe
verified gradient-isolated). (4) The AUG ERRATUM, caught by the new selftest: the house
`RandomApply([v2.RandomSolarize], p=.2)` pattern HALVES the effective probability (v2's
own p=.5; measured .0988 wrapped vs .1983 bare) — every house stack's effective solarize
≈.1 vs declared .2; frozen stacks unchanged (code = v1 definition), declarations
corrected; `_bench_view` = the true-p builder; VISReg's shipped code carries the SAME trap
(their published numbers = effective ≈.1, and the house effective coincidentally equals
theirs). (5) The compute-pin surprise (B/L fit smokes, deliberately wall-bounded): **B =
35 min/ep single H100 (full epoch completed) → ~2.6 d/100 ep; L ≈ 105 min/ep with
grad_ckpt (7678/10009 steps at the 1.5 h cut) → ~7.3 d** — 3–4× under the card estimates;
both fit one 80 GB GPU; DDP demoted to a wall-clock accelerator. The L smoke's wandb
"crashed" state = the wall-cut artifact, not a failure — the pin was the purpose. Bench
A/B partials at close: lej ep15/90 .5600 · vm4 .6329, both climbing. Wave healthy
throughout (v10u ep16 .4990 ≈ +8 over vm4's matched-epoch curve). Job trail: probe
top-ups (cancelled) 63221867–71; backfills 63222500–03; bench 63222504–05; wave landing
chains 63222506–20; lane bench 63225712–17; fit smokes 63222104/05; lejl chain
63225989–226004; visreg smokes 63225769/846/226008/228131.

## 2026-08-10 (evening) — the S-read round (Fable): wave cut to two, the zero-diff pivot, the FLOP collapse, SWA built, the program re-based onto the Lightly frame, B opened early

Berker opened on the four-item agenda but ruled from the live curves instead: **v10u =
the winner; keep only e27v10u + e27oas** (pv/grpo/pvo KILLED at ep44/25/25, chains and
bench cancelled, ckpts trimmed to `_last`; stream record at kill: all-views-mean ≥
grouped ≈ per-view, pv plateaued .45–.46 from ep34). Then three rounds that reshaped
the program, all same evening (D-094/095/096):

**The zero-diff pivot (D-094).** "For lejepa, do what is needed to get the lightly
numbers, i want no diff" → the in-house e27lejl was stood down while still queued (its
declared deviations — bf16, k9 blur, seed, monitor — are exactly what "no diff" rules
out; drop_path 0.1 checked and matched) and the reproduction became THEIR code:
`third_party/lightly` pinned @ f444cf36, runnable `~/lightly_repro` with ZERO patches
(their `main.py --methods lejepa`, 4×128 = global bs 512, 16-mixed, unseeded, their
eval chain), segmented wrapper `slurm/lightly_lejepa.sbatch`. The smoke completed a
full epoch in 21:16 on 4×H100 (their exact 2502-step semantics) — ~36 h/100 ep; the
7-segment chain armed the same evening. Berker pushed back ("you should be able to
exactly match. what is the problem?") and the answer landed as a concession with a
mechanism: nothing fundamental remains now that their code is an importable ORACLE —
the two real diffs (fp16-mixed, blur) are closeable and testable; the oracle-certified
e27lejl relaunch = the our-stack certification twin, held as optional.

**The FLOP collapse (the attribution round).** His own conditional ("if we are sure we
can account this gain to uniform views") failed on measurement: oas-vs-v10u differ on
four axes, and the dominant one is per-step compute (1970 vs 690 tokens/sample, ×2.86;
wall ×1.75). Epoch-matched +7.5 (ep19: .5130 vs .4377) became a FLOP-matched TIE
(v10u ep16 .4990 vs oas ep46 .4984), and the third point sealed it: vm4 at its
FLOP-matched epoch reads .5032–.5053 — **all three lanes collapse onto one
performance-vs-FLOPs curve within ~1 pt** (v10u ~+0.7 above vm4 = the real, small
view-count bonus). The locals confusion dissolved in the same round: locals are never
resized up — 96² enters the ViT as 37 tokens vs 197 (the SwAV compute trick), CLS
summarizes whatever the tokens carry. Berker held that view-compute ≠ epoch-compute in
principle; the disagreement was converted into a PRE-REGISTERED read: if 10-view
invariance compounds, v10u's curve lifts off the shared FLOP curve late — the overlay
(v10u cadence ckpts vs vm4/oas ep100 on the FLOP axis) is owed at the landings.

**SWA (D-095).** "Implement swa, as described in lejepa" — the paper's entire spec is
one line (*"we apply SWA on the encoder producing μ in Equation (6)"*; Izmailov
equal-weight; VISReg checked at source: NO teacher anywhere; lightly: none; je.py still
unpublished — the +2.9–3.4 Table-4 flag is name-only). Implemented in floorssl
(`+method.swa=uniform`): grad-free eval twin (teacher_backbone/teacher_projector)
deepcopied post-init — zero RNG draws, student byte-identical to its parent lane;
per-step equal-weight running average in post_step, BN buffers copied; anchor = the
twin's per-image all-view z-mean at the lane's own anchor set (their V_g-anchored μ
kept as the separate inv-anchor axis); ×2V/(V−1) on the uniform branch pins the init
pull to the calibrated w_inv; swa_k resumes via extras; the twin rides every ckpt →
post-hoc SWA-eval free at landing. Selftest §(9) PASSED on GPU (init-parity EXACT at
drop_path=0, hand-math average, grad-free, round-trip).

**The re-base (D-095) + B opened (D-096).** Convinced by the FLOP table, Berker
re-based the variant program onto the FLOP-matched frame: the new cells ride the EXACT
Lightly view stack under OUR loss — `aug=lightly_mc` in floorssl (2g@224+6l@96,
0.4-family jitter, true-p solarize global-2-only, bicubic; selftest §(10)), OAS
no-ring per the estimator law. The program: S base **e27lmc** + **e27lmc_swa** +
**e27lmc_b512** (flat lr; single-GPU bs-512 = the proven lejl memory path at this
geometry), and the three ViT-B mirrors — **B does not wait for the S landings**
(D-096, his directive; D-079b's joint-read clause superseded; B ≈ 2.2 d/100 ep at 6l).
Both dose pilots launched the same evening (S 63239252, B 63242633; w = s*·9.0/g at
the incumbent held-state). v10u+swa superseded pre-launch; v10u/oas land unchanged as
the scaling story + house-aug control. Flagged-for-veto readings: 6 locals (the
FLOP-match to the bar, vs D-088's V=10 which stays for the landing lanes), the mild
0.4-family vs E27-T1(b)'s harsh-locals result (the price of comparability, knowingly
paid), dose transfer to the twins (init-only law).

Misc: the v10u-b512 fit smoke died on a syntax slip (`+method.grad_ckpt` on an
existing key) and was NOT resubmitted — superseded by the re-base. Budget path at full
sail: 2 lanes + 4 lightly + 3+3 trios = 12, VISReg at 4 = 16 exact (8 when lightly
drains). Bench A/B passed ep25/90 mid-session; corrected 12-space probes and the
VISReg Arrow-index smoke still filling at close. DONORS gained visreg @ 47b1cf4 and
lightly @ f444cf36.

## 2026-08-11 → 08-12 — the incident-and-payoff day (Fable): SWA cured by option (a), four harness fires fought, the repro lands ON TARGET, the frame-matched tie, the combo round

Overnight from the trio launches, four infrastructure fires, each diagnosed to root:
**(1) e27oas crash-drain** — native CUDA abort on gpu270 at ep71; follow-ups died on the
sick node, the drain fired the landing chain onto a missing ep100 (harmless). Resumed
ep70 `_last` with the node excluded. **(2) e27lm4b5b OOM-relay** — every bs512×4g
segment host-OOM-killed after ~1 epoch (1.6 GB decoded batches × 28 workers vs 200G);
Berker's "why does the curve start at ep3" was the symptom (wandb's step high-water
swallowed each relay's points). Fixed 400G + 12 workers — then a SECOND relay of
16-second failures from the `frame.num_workers` vs top-level `num_workers` key slip;
third relaunch TRAINING. **(3) The queue mystery** — Berker asked why jobs pend with
"free space": measured answer — 27 idle GPUs strand three ways (12 behind CPU-exhausted
nodes, ~8 reserved for a top-priority whole-node job, the rest behind MEMORY windows:
gpu265 had 119G free vs our 128G habit-ask). The 110G resubmit scheduled in seconds —
memory-window fitting became the standing trick. **(4) VISReg-B launch** — three
attempts: `num_workers` needs `+` (undeclared key); accelerate needs `--multi_gpu`; then
explicit `--gpu_ids 0,1,2,3` after it refused a 4-GPU allocation it provably held.
Attempt 3 UP, wandb ONLINE (team entity, SSL-ImageNet1K-VIT-B/sbe9q03e). Also: the
lejepa repro was invisible to Berker — their harness is TB-only; `wandb_bridge.py`
republishes to `lightly.lejepa.repro` (v2 after the DeviceStatsMonitor 2M-point parse
stall: allowlist + per-file cache). VISReg's imagefolder crashes root-caused to the
data tree (stray val tarball auto-extracted into a hash-label; split-name inference
"val"→validation) — tar moved, patch takes the sole split; PATCHES.md updated.

**The science:** Berker's wandb read killed the sick SWA lanes (−14/−15, pathological
h_moment_kl) and chose option (a); postmortem = BOTH axes (inherited doses mis-set ×1.45
at init + the paper-literal uniform-from-0 average = a near-init anchor early).
`swa=ema` (DINO τ cosine .996→1, declared deviation) + swa-ACTIVE pilots → relaunched
lanes track AT their bases from ep1 (S ep3 .2458 vs base .2437; B ep1 .1421 vs .1443) —
cure confirmed, then SURPASSING: lmcse +3.0 over base at ep15. **The repro landed its
online monitor at .5584 vs Lightly's published 56.23** — the zero-diff replication on
target; offline evals running. **The matched-epoch table** (D-099): the frame-matched
bs512 pair is a DEAD HEAT ep10–30; bs128 early "edges" are small-batch fast-start;
the real gains at matched batch are swa +3.0 and bs512 +3–4, independently — Berker:
combine them → the 2×2 completes with e27lmcs5 / e27lm4s5b, own pilots submitted
(63394507/8), chains fire on their ep1 lines. oas reached ep96 with its landing chain
re-armed (63394461-64); bench A/B continuations resubmitted after their 24h walls
(cumulative CSVs resume); lightly eval segment mid-linear at close of the round.
Next session declared: THEORY from Berker's uploaded file + the submission plan,
while the arms run.

## 2026-08-12/13 — the theory round + the two-space measurement program (D-100)

Berker's paper draft landed in `docs/paper/` (main.tex + appendix.tex + references.bib).
Read in full, every proof verified line-by-line — all correct (two cosmetic notes: App C.2
reuses Q for two objects; Thm C.2 wants the rank(M_z)=r_z half-sentence). Discussion round
settled the positioning (D-100): Thm 2.1 is NOT tautological — the three-stage construction
is the content and the literature genuinely confuses the two spaces; empirical support =
each method's stated desideratum at its declared loss tap vs h on TRAINED models, choosing
strong failure points without promising h is always worse. Stores PERSISTED (427G, D-005
purge suspended). ICLR 9/24 target; C5 (FLOP-collapse) out of this paper; seeds after a
promising single seed.

Built and validated same-day: `sslgap/metrics/twospace.py` (paired-view B̂/Â with even/odd
cross-group canary, truncated-support whitening, held-out whitened R²_acc + Σ_c spectra,
the Θ(I+Θ)⁻¹ fidelity-law test, Ĝ/Ŝ split with any-predictor lower-bound logic, per-class
organization/decomposition) + driver/sbatch/selftest (18/18 GREEN incl. exact synthetic
recovery of the law). Run over ~60 stores incl. fresh multi-view layered extracts for the
e20f controls and all trajectory checkpoints.

Checklist-driven figure deck (Berker: "work with a checklist… every item a figure"):
`docs/paper/CHECKLIST.md` + `experiments/paper_checklist_figs.py` → results/figures/paper/
C1–C8b rendered. Lane truth pinned after two wrong-lane incidents (C1 first rode treated
e20f arms; e12 pairs were the wrong treatment family): treatment pairs = E20F wave
(control `in100.<m>.s0` vs `in100.<m>.s0.e20f`), C1 = controls only, and — Berker
2026-08-13 standing rule — EVERY treated/untreated measurement includes the ours pair.

The missing experiment (Berker: "in100 run with our loss only after mlp") — `d256vm4zonly`
= vm4 with h_lamb=0, one H100. First launch CRASHED (hand-copied overrides; num_classes
default 10 + bs default 256 — the second one silent). Fix protocol now standing (memory:
twin-launch-config-diff): mechanical resolved-config diff vs the reference ckpt cfg + a
first-epoch watcher. Relaunch verified: diff = {h_lamb, tag} only; landed ep100 online
.7048, full landing chain + trajectory extracts + MLP pass complete same night.

Headline raw results (all landed, artifact-checked; interpretation joint, D-100 records):
- Fidelity law holds per model at corr .990–.9997 across every Θ eigendirection (C4);
  downstream decomposition identity per class (C8b).
- ±h-moment-floor, six single-factor pairs incl. ours: B-rank of h ×2–3 up (ours 181→371),
  trΘ/r UP in all six (Berker's "treatment raises Θ" — confirmed), held-out whitened R²_acc
  up where headroom exists (vicreg .07→.18, simclr .64→.85, ours .44→.49). C8 term-shift:
  the floor buys organization (d_h² down .08–.17) at a small view-fidelity price (+.02–.06).
- C6: standard SSL training DEGRADES h→z accessibility over epochs in both arms
  (vicreg/simclr/dino); ours is the only lane where it RISES. Thickness thins with depth
  ~10× and the bridge forms only past L06 (C7, all 6 columns).
- MLP center-predictor with source-split early stopping FAILS to tighten the linear
  Ĝ bound on 10/11 runs (val≈test, train<val — generalization-limited): the view→center
  conditional mean is essentially linear at N=10k; C5's excess-share lower bounds .45–.87
  ride the closed form.
- Gaussian-family ±treatment table (+ vm4/zonly): floor improves moments/tails/radial in
  ~every cell; sliced-EP moves independently (the E12-T5 lesson alive at small amplitude).
  vm4 z = moment-KL .01 with kurt 195 — moment-perfect, shape-extreme. zonly h: moment-KL
  3.28/kurt 27 → vm4 .51/1.54 — our h-term reproduces the whole treatment signature.
- Marginal-entropy instrument (Berker spec: 10 bins, min/max, Σ per-dim, 5 Haar rotations):
  range-binned AND variance-scaled variants both show every treated arm moving toward the
  Gaussian reference, 10–100× CI; var-scaled is the corrected reading (Gaussian = ceiling
  2.4061); zonly 2.3587→vm4 2.4044 = the largest arrow. results/compare/marginal_entropy_h.csv.
- E20 guillotine regenerated with the 8th row (ours: zonly grey vs vm4 green).

E27 arms meanwhile: lightly repro CLOSED — offline linear 64.11 / kNN 47.06 vs published
64.0/47.1 (the zero-diff S anchor certified). visreg-b attempt 3 TRAINING (first run to
survive the launcher errors; the HF arrow build was first-execution, ~2h; wandb
SSL-ImageNet1K-VIT-B/sbe9q03e; ep19 test .4833). oas landed ep100 .6049 online; extract
200G OK after a 96G OOM; probe TIMEOUT@4h → 24h rerun (partial CSV: h.cls raw_v2 .6268 /
house_v2 .6383); audit OOM@96G → 200G rerun. Both combo chains fired off their own pilot
ep1 lines (S 3.53/4.46/0.180; B 5.07/8.97/0.335). Bench A/B hit their 24h walls at ep87/90
(lej best .6058 vs vm4 .6633, +5.75 frame-matched) — finals resubmitted. byol 4096-d taps
hung twospace 4h → max_tap_d cap. Faithfulness incidents on record: "e27lmc done" claimed
off the plan while wandb read ep87 (Berker caught it; artifact-check-before-claiming now
the standing rule), plus the --wrap/sh-source and doubled-path submit failures — all
repaired same-day, all landings re-verified from disk.

## 2026-08-25 .. 08-31 — the 400 round: launch, trim, and the compute-fairness turn

- D-103 400-round launched 08-25 (six cells + four 100-ep twins + two pilots; dose law
  w = 57·τ_win/g^ep1). Same-day ENOSPC burned the v6b400 chain mid-launch (group fs hit
  100%; the purge freed 391G — docs/PURGE_2026-08-25.md); relaunched, ~2 epochs lost.
- 08-26..30 (recorded from the interim handovers): D-104 dino re-dose resolved (e20fwlo
  the winner) and D-105 visreg pair into zoo+exhibits; the e31 shape/scale split ran at
  IN-100; the OK-AI 12-run transfer block self-run; e27v6b100 landed and swept every B
  readout (bench 74.15 / kNN 64.28 / transfer 80.9); the SWA-twin question closed (0.00
  at B); four assistant errors recorded verbatim in that handover.
- 08-31 (one session): all four twins confirmed finished. Curve forensics across the
  grid: every 400-cell's matched-epoch lag is the cosine phase (phase-matched leads
  positive everywhere); v6b100 terminal slope +0.111pp/ep = undertrained; the small B→L
  delta traced to step starvation (L bs512 = 2,502 steps/ep vs B 10,009 at unscaled lr).
  Berker trimmed the round (D-106: sbetl400/Ls5btl400/Ls5b400 killed at ep188/161/160;
  P-m-3's 400-tail forfeited to the twins' null). The OK-AI compute ledger was verified
  from their lite_ssl code (teacher forwards globals only → 8g+18l/sample; our swa=ema
  twin forwards all views → v6 = 24g = ×2.13, scale-invariant) and the compute-fairness
  program adopted: compute column + acc-vs-FLOPs exhibit; okdinob300/okibotb300 benches
  launched (first attempt died on a missing pubvit adapter arg — relaunched). Paper:
  Ours-B → v6b100 in both tables, OK-AI self-run transfer rows, I-JEPA cite 79.3
  (verified from their Table 1), data2vec bench row dropped (no published LP number).
  e31 closed with the slice-dilution finding (a random 128-slice reports ~27% of the
  full-space shape residual, ×3.7–5.1; full-slice satisfies full-space logdet at matched
  pull; the v6/ring lanes already close the OK-AI rank gap at h). E32 defined AND agreed
  same-day (cloud vs image spread: the aug-quiet core, D-110). The v6s pair (D-107,
  ×2.15 declared) and v6Llr pair (D-109, lr=4e-3 path-match) pre-registered with
  committed predictions, then piloted. Rulings: no seeds (field-consistent); IN-100
  probe convention kept; the D-093 16-H100 cap noted as previously disabled (31 peak in
  the scaling round). ADE20k fetched for the conditional seg run (D-108.5).

## 2026-08-31 (afternoon) .. 09-01 — chains formed, E33, the coverage sweep, tables consolidated

- The four new chains (v6s pair on v10u-τ; v6Llr pair on the D-111 winner-share re-dose)
  launched 08-31 afternoon, reproduced their intended doses exactly at ep0, and all
  passed the ep2 quench gate (wandb moment-KL VALUE .20–.26 vs the .1 kill line;
  h-moment real). S settle locked at ~.64/.35 by ep3, pair members overlaying; L-lr pair
  hovers inv-heavy (.77–.84). Formation watchers closed by Berker ("they kickstarted
  correctly"). On the step axis the L-lr pair walks B's per-step path slightly ahead
  (.454 @47.5k steps vs v6b100 .394 @50k) — the epoch-axis probe lag is schedule phase,
  not sickness. Corrected ETA: v6Llr400 ~09-17 at the measured ~1h/ep (loader-bound).
- E33 (rich vs lazy) built and landed in one day: linear-CKA + empirical-NTK machinery
  (sslgap/metrics/cka.py, sslgap/extract/ntk.py, experiments/feature_drift.py); init
  convention settled by TEST (from_native random-init bit-exact vs the trainer build;
  cross-checked vs the stored null store). Berker ruled IN-100 primary (in1k pair
  canceled pre-launch). RAW: both cells far from lazy (final 1−CKA .87/.89, NTK-align
  .56/.53); the comparative "richer than baseline" does NOT hold (baseline drifts
  slightly further), but the floored kernel keeps moving late (successive alignment
  .98–.99 vs .9993+; ‖K‖ 8k→21.6k vs flat ~466). Figure re-cut to vm4-only on Berker's
  call. AGREED wording pending.
- Seg ruled IN (09-01) after the port validated on printed rows (DINO-B-400 30.39 vs
  29.40; VISReg-B-400 31.36 vs 30.16 — port reads ~+1). The coverage sweep ran 15 jobs
  in the day, zero reruns: seg for the OK-AI six (B trio 31.79/31.93/38.58, S trio
  25.84/26.54/33.64) + officials; transfer for every remaining official — port ≤0.2 vs
  printed on six ckpts (four exact) — + the carried vm4-S row (75.0). v6b100 seg =
  28.11 at 100ep. OK-AI's catalog verified exhausted (6 repos, ep100/300 only, no L).
- Two wall timeouts, both on the crowded gpu238: the v6L100 house probe (closed by the
  new ruling: in1k landing = bench-only) and the v6L100 bench at ep75 (resumed from its
  state file on a clean node → final 72.79/64.44 @ep90, still rising — the L-100 row
  call opened for Berker). seg_visreg.py gained --adapter for the pubvit lifts.
- Tables consolidated: OK-AI rows grouped with midrules in all three tables (Berker's
  ruling) and the ep300 cells completed on his catch (S 66.4/73.8/75.2, B 72.4/75.2/
  78.6); the seg table rebuilt to auto-fill. The D-106.3 compute column was built,
  landed, then REMOVED same-day (D-112: no-sota framing + the OK-AI data confound —
  their cards state 1.43–1.45M images ≈ all IN-1k splits); OK-AI rows retained
  eyes-open as the only epoch-matched S+B family. Their "DINO" identified as a
  modernized hybrid (DiNO+KoLeo, DINOv2 ViT-v2 backbone/optimizer, no registers) —
  methods-text sentence owed. Four stray dead-session watcher notifications verified
  and neutralized across the two days.

## 2026-09-03 (evening) → 09-04 — the video speed-up, a stray session, the toy purge, and the v6s100 tail-storm incident (Fable)

- **The video cell was measured, not tuned (Berker: "dont microoptimize ... workarounds that would materially
  improve").** The first E34 cell (64312322, 2 × 4 H100, clip files on BeeGFS) ran 53 s per optimizer step. Measured
  (scratch/video/e34_*.py, outputs/e34*): BeeGFS gives each CLIENT NODE ~35 whole-file (5 MB) or ~60–80 span (1.2 MB)
  reads/s at ~100 ms per request, scaling with nodes (three nodes at once: ~80 each); one clip costs a core 0.30 s of
  read latency, 0.02 s of JPEG decode, 0.017 s for six crops and 0.38 s of photometric ops; an H100 needs 0.28 s per
  96-clip micro-batch; nvjpeg 0.04 s. NFS is no faster per node (cold 1.2 MB windows 84 vs 52–69 files/s at 64
  readers), so the data stays on BeeGFS. The H100 nodes carry an 879 GB NVMe root (/tmp, 678 GB free) and a 28 TB
  /mnt/GPU100localdata that locatgrp cannot write; local staging was NOT pursued (Berker: no IT request now).
- **Second launch (D-record on the E34 card):** exact batched GPU photometrics (`video/levjepa/data/gpu_views.py`,
  every op 0.000/255 against torchvision; declared deviation: no uint8 rounding between ops), the span read in the
  clip-file loader, an `augmentation.photometrics_on_gpu` switch, checkpoints pinned under the BeeGFS run dir
  (stable-pretraining had redirected the first cell's to ~/.cache). First cell cancelled after epoch 6; run
  `vid.floorssl.s0.k710s2` as job 64377149 (2 × 4 on the freed gpu273/277, 20:02): epochs of 8–12 min instead of 34
  (≈ 3×), landing ≈ 09-05 midday. The 8 × 1 shape (job 64377150, eight BeeGFS clients, resume-capable) is chained
  `afterany` behind it: the gpu100 QOS caps a user at 31 GPUs and the fleet's 16 + 8 leave no room for 8 more while
  the 2 × 4 runs; a switch = cancel the 2 × 4 at an epoch boundary when the pool has eight free nodes (a watcher
  alerts) and needs Berker's word each time.
- **Video H stream (RAW):** shares moved to ~.82–.88 inv / .12–.17 Z / .00x H; kl_z .13 falling; the H stream's
  whitened trace fell .21 → .04 while kl_h rose 1.57 → 1.89 (per-step medians, epochs 21–59). Step-matched against
  ImageNet v6s100 (kl_h 1.43 at 3k updates, flat/oscillating 0–5k, .85 by 20k, .46 at 50k, .52 at 100–150k, .28 at
  800k+) the absolute values are close early; the direction differs. The realized H share matches by the dose law
  (.003–.007 on both). A re-dosed cell (w_h ×2–×7 by the share law) was proposed as the comparison arm on the 8 × 1
  lane; Berker raised it and has not decided. Peek on the epoch-89 EMA weights: frozen attentive probe on a 64k-image
  ImageNet subset, 5 epochs (job 64433768; two evaluator first-run bugs: `scripts/attentive_probe.py` needs
  PYTHONPATH=video/levjepa, and its validation forward under DDP lacks no_grad → run single-process).
- **A second Claude session relaunched E28 (19:28) after Berker had ruled E28 done; he killed it. All e28 jobs
  cancelled.** The E28 card's last rows (19:3x) are that session's. gpu274 ran a benchmark 3–9× slow on 09-03 and is
  excluded from the fleet successors and the video shapes (Berker's word).
- **Toy checkpoints purged** (Berker: "you can dump toy checkpoints"): 525 files, 155 GB, `docs/PURGE_2026-09-03.md`;
  group filesystem 308 → 463 GB free. Purge candidates for the rest are in the 09-04 report (memory).
- **INCIDENT — e27v6s100's tail storm (D-115).** 02:09 on 09-04 the landing segment (gpu271) printed its first clipped
  gradient burst at step 849,724 (epoch 85, lr 6e-5); 291 by 10:55 with pre-clip norms to 6e7. The online probe never
  dropped (ep85–93: .6532 → .6560, new bests at 91 and 93), but the per-step samples (wandb) show burst steps where the
  FORWARD explodes (inv to 7e5, loss to 2e7) and the typical loss drifting 34.5 → 39.5. Berker called it a health
  issue and asked for a checkpoint roll-back. Cause, read from `_ep75.pt` vs the storm state: the projector's two
  pre-BatchNorm weights shrank in norm 44.7 → 32.0 and 104.5 → 78.4 under `mlp_wd` .05 while the second BN's input
  running variance collapsed 11× (1.9e-2 → 1.7e-3); a layer feeding a BatchNorm is scale-invariant, so weight decay
  shrinks it unopposed and its effective lr rises as the nominal lr vanishes, and BatchNorm re-amplifies the collapsed
  signal on 64-per-GPU batches. v6b100, v6L100 and v6Llr100 never burst. Actions: `_last.pt` ← `_ep75.pt` (epoch
  index 74, full state), storm files kept as `*_storm_ep92.pt`, the chain resumed as segment 64435570 (gpu266,
  11:12) with `method.mlp_wd=0` — a no-op on the function at the resume point; `train_ddp.py` now re-applies the
  config's per-group weight decay after `opt.load_state_dict` (the checkpoint's hyperparameters otherwise win
  silently) and prints the groups (`[0.05, 0, 1e-07]` confirmed). Rejected: SyncBN (driver untouched), BN eps
  floor (changes every unit's gain at resume). Landing moves to ≈ 09-05 midday.
- **Lessons.** (1) The INCIDENT trigger's running mean is inflated by the spikes it counts; read typical steps from the
  per-step record. (2) An online probe can rise through an objective drift; the loss medians are the health read.
  (3) Weight decay on scale-invariant (pre-BN) layers is an effective-lr schedule that RISES at the end of a cosine —
  a training-hygiene item for the head. (4) `optimizer.load_state_dict` restores param-group hyperparameters:
  config overrides at resume must be re-applied (now done). (5) Node correlation is not a mechanism: both gpu271
  runs burst, but the run-side read held and the node hypothesis did not. (6) Short loader smokes timed fewer
  clips than the prefetch depth and were not throughput numbers. (7) Watchers every 10 min during a storm are
  noise; Berker cancelled the fleet watcher.

**2026-09-04 late evening (append).** v6Llr100 landed at .6976 (P-v6Llr-2 refuted; Berker concluded the large-lr experiment a
failure; v6Llr400 cancelled at ep98). v6s100 wd0 read per step (tables `results/diag/v6s100_wd0_*.txt`): typical step flat then a slow
rise (median 33 → 38 by epoch 88), rare spikes thickening (~15–26 above 4,000 per epoch), z moment-KL never above 0.16 at any
burst — Berker's breakage indicator (a z moment-KL spike) is clean; the storm in the original read p99 1.5 → 43 with ~1,200 steps
above 0.3 per epoch. Checkpoint copies: one verified clean copy kept per run (Berker: no wasted space; monitors hourly). **v6b400's
own tail storm opened at epoch 264** (25 bursts, probe stalled) with the same pre-BN weight-decay driver; on Berker's word the D-115
fix was applied at once: roll back to the epoch-261 best, `mlp_wd=0`, fresh wandb run — segment 64588207, pending on priority.
Video: the 48-worker takeover of S cost an hour of relaunches for ~18 % faster epochs (BeeGFS-bound); S and B together starve each
other on BeeGFS (S epochs 681 → 1,790 s) — B cancelled and requeued behind S (one video cell at a time). Offline K710 probe on S's
epochs 0–30 queued (64570046 / 64578059) for the epoch-matched comparison with the killed second launch.

**2026-09-05 morning (append).** v6s100 landed clean at 10:45 (final online .6734; 236 clipped bursts over the wd0 segment, z moment-KL
never above 0.16, BatchNorm variances recovered) — the D-115 roll-back held to the end; its landing eval chain (extract/bench/transfer/
seg, the v6b100 recipe) launched on Berker's word. v6b400's fixed segment (D-116) started 09:06 on gpu266 after a gpu271 prolog failure
left it held (lesson: `scontrol release` after any launch failure); first epoch clean (0 bursts). Berker's monitor rules: one quiet
hourly watcher, no verbose start/node chatter. Video S: epoch ~95, 475–555 s per epoch, online K710 probe 16.8–19.3 %; the offline
probe copy runs on a 3090 node (no measurable slowdown of S).

## 2026-09-06 — v6b400's second storm and the guard (D-117); the video ViT-S lands and reads 47.2 on the IN-1k attentive probe

- **v6b400 under wd=0 stormed again at epoch 284** (twenty epochs after the D-115 roll-back): the per-step rows showed a different
  mechanism from v6s100's — outlier batches (invariance loss 3 → 96 → 3,588) spiking the gradient, the z ring carrying the outlier z for
  three more steps (z moment-KL excursions in runs of exactly four), the h stream flat, the projector's BatchNorm variances large; the
  projector's pre-BN norms had grown 4.3 %/epoch without weight decay. Berker: "diagnose then fix and replace" → rolled back to the
  pre-onset epoch-283 copy (protected the night before) and resumed with the outlier-batch skip guard (config-gated in train_ddp.py);
  skips fell 80 → 3 per epoch within five epochs, no z-KL excursion since, probe .7126 at ep290. Read: the knob has no good setting
  (wd 0.05 = thinning/collapse, wd 0 = growth/outliers, 1e-4 ≈ 0) — the Linear → BatchNorm projector block is the wall; block-level
  options recorded for new cells. Rolling clean copies now require a z-KL-clean epoch as well.
- **v6s100 landed clean** (final online .6734) and swept its landing chain: bench 69.7 / kNN 59.2, transfer 77.6 (v6b100: 74.2 / 64.3 /
  80.9), seg queued; the tables in main.tex regenerated (S-100 rows; the generator's L-100 fill reverted pending Berker's call).
- **Video ViT-S landed** (240 epochs; final online K710 probe 24.8 %) and its IN-1k attentive probe (student weights, 4 × 1 H100, batch 64)
  read **47.23** at epoch 20 against LeVJEPA ViT-S 39.4 / V-JEPA 2 38.7 (pixel reads) — RAW, with the batch note; P3 proper needs the
  z-only twin; the EMA read runs. The probe port had dropped V-JEPA's output projection (found via a DDP hang; fixed, PORT_NOTES).
- Lessons (Berker): never touch a healthy run without his word (the 48-worker takeover cost an hour for 18 %); one video cell at a time
  on BeeGFS; quiet hourly watchers; release SLURM-held jobs after launch failures; keep one verified clean checkpoint copy, not many.
- **Afternoon session (15:00 →, Fable): the hourly watcher re-armed** — one quiet Monitor (v6b400 per-epoch probe / skip / z-KL / projector read
  with the verified clean copy, v6s400 and lm4sbe400 probe lines, the EMA probe and seg endings, the video B cell), report-only; the copy rule
  keeps one verified copy and superseded copies are removed by hand (the sandbox refuses a watcher that deletes). v6b400 guarded epochs
  284–292: skips 80 → 3–8 per epoch, no z-KL step > 0.3 since epoch 285, probe .7125, projector norms still +1.2 % per epoch under wd 0
  (E27 card; `results/diag/v6b400_wd0g_epoch_tail.txt`).
- **The video eval audited against V-JEPA's evaluator** (E34 card §fairness audit; PORT_NOTES): classifier, schedules, augmentations and
  input match; the probe's optimization does not — 16× the head updates (batch 64 vs 1,024) and no gradient clipping (the donor clips at 1.0),
  plus a differently initialized head; V-JEPA reads its EMA target, LeVJEPA's convention is unstated. Berker's ruling: the probe batch is not a
  confound → D-118 withdrawn the same hour; the remaining recipe differences are on the pretraining side (4× the optimizer steps at batch
  768, ≈2.1× the encoder FLOPs per epoch from six global views, our own 20 % draw, clipping) — his choices, listed on the card. LeVJEPA's grid
  reports IN-1k only; SSv2 / K400 evaluators are not built (data raw on BeeGFS). The EMA read runs (ep2 30.77 vs the student's 31.21).
- **Tables in the paper:** the generator holds the L-100 cell at --- by rule (Berker's open call), carries the seg S rows and a new video table
  (tab:video: cited pixel reads + Ours student / EMA rows, protocol notes in the caption, RAW); main.tex regenerated; main_lean.tex mirrored by
  hand — its Ours B-100 and L-100 rows still carried the lm4 cells (71.2 / 56.0, 71.8 / 60.8) and are now the v6 values / ---.
- **ADE20k seg for the OK-AI ep100 checkpoints queued** (Berker: "we need ep100 okai runs so that it will stay comparable (both base and small
  vits). not urgent but you can queue on gpu partition"): jobs 64695614–19 (dino / ibot teacher, lejepa student; S and B; `--adapter pubvit`,
  tags `in1k.pub.ok<m><a>100.seg`) on the gpu partition behind seg-v6s100; the seg table carries their rows (--- until landing).
- **lm4sbe400 stormed and was killed (21:30 → 21:5x, Berker's word).** The LeJEPA-family control at B (wd .05 on the projector) showed the
  D-115 signature at ep277: 20 bursts in the epoch (norms to 99k), pre-BN norms −30 % since ep200, BN-2 running variance median 4.5e-3 with every
  channel below 1e-2. Berker: "kill lm4sbe400. we can consider fixing it later. for me b start is the priority right now" — segment and successor
  cancelled, storm state and roll-back points kept. Also today: the EMA IN-1k read of the video S cell landed at 46.76 (student 47.23), the SSv2
  and K400 evaluation stores were built, the video probes were ported (D-119), and the B cell was relaunched on 8 × 1 H100 with node-local staging
  and a 515-epoch chain (LeVJEPA's Table 3 budget), checkpoints 239 / last / best.
- **09-07 morning:** ADE20k seg landed for v6s100 (22.17) and the six OK-AI ep100 checkpoints (S: LeJEPA 24.22, DINO 25.12, iBOT 30.10; B: 30.60 / 31.12 / 37.87) — tab:seg filled; the video B cell trains on 8 × 1 with staging at 4.5 min per epoch (epoch 150 at 11:30); the S SSv2 / K400 probes run; v6b400's guard skips 60–90 steps per epoch with z-KL excursions at 2–4 skipped steps per epoch, Berker's continue decision pending.

- **09-07 evening, the seg question (Berker: why only segmentation?):** patch-token diagnostic built (`sslgap/metrics/patch.py`, `experiments/patch_diag.py`; RAW on the E27 card): the frozen patch tokens of our cells rank exactly as tab:seg under a training-free patch classifier; ranks, locality and norms are healthy; what is missing is the image-level component in the patches (95 % within-image variance, patches orthogonal to the CLS; the public models carry 13–32 % image-level variance and a CLS component); the IN-100 twin pair attributes the decoupling to the h conditioner (cos(patch, CLS) .59 without it, .13 with it), not to the absent local crops. Also: v6s400 re-shaped to 96 CPUs after Berker's '8.6 days is unacceptable' (first pair died on the broken node gpu277, resubmitted); lm4sbe400 killed; B cell past epoch 240; the S SSv2 read 37.31.
- **09-08 07:31, v6s400 wd-0 + guard (D-120; USER-DIRECTED):** Berker: "do the wd=0 fix with the guard. make sure we stay in high speed config." The D-115 fix (projector `mlp_wd=0`) with the D-117 skip guard was applied to v6s400 from its running state at ep197 (pre-storm: zero bursts, BN-2 running variance median .096 / min .044, pre-BN norms 66 / 154 / 107 down from 82 / 192 / 118 at ep100) in the fast shape (96 CPUs / 44 workers per rank, 27.5 min/epoch): new singleton segments queued behind the running one, the wd-.05 successor cancelled, the swap helper cancelled the running segment right after its ep197 checkpoint (07:29); 64926249 resumed at epoch index 197 with `weight_decay per group = [0.05, 0, 1e-07]` on a fresh wandb run `.wd0g` (1ay07kmz, from step 1,971,773; config verified: skip ratios 10 / 3, mlp_wd 0). Declared deviation: this cell's projector has wd 0 from ep197 (v6s100 from ep76, v6b400 from ep262). The hourly watcher now reads both guarded chains identically and appends per-epoch lines to `results/diag/<run>_wd0g_epoch_tail.txt`.
- **09-08 14:38, video B cell landed:** 515 epochs (LeVJEPA's Table 3 budget FLOP-matched to our recipe), job 64747615 COMPLETED after 1 d 14 h 36 min on 8 × 1 H100 with node-local staging (4.4 min/epoch); final online K710 probe 73.24; the three B-row evals (IN-1k attentive, SSv2 attentive, K400 linear-mean; EMA; 8 × 1) queued on the landing per Berker's word (D-119 amendment); insurance segments never ran. Numbers RAW until jointly read.
- **09-08 15:03, eval shape re-derived (Berker: the shape constraint, not the card count, was the blocker):** the B-row probes asked for one card on each of eight distinct nodes while the free cards sat on five usable nodes; the one-card-per-node shape only served the BeeGFS read cap, void since node-local staging. Launchers made layout-free (8 ranks on any 1–8 nodes, memory per GPU), evals resubmitted (65002280 / 81 / 82); the IN-1k probe started within a minute.
- **09-08 15:20 → 15:52, the launch shape: two more walls, then a re-derivation (Opus).** The layout-free ask cost six dead submissions in
  twenty minutes and taught a constraint that is not video-specific. (i) `--gpus-per-task=1` and `--ntasks-per-gpu=1` mask one card per task;
  NCCL 2.28 then dies between ranks that share a node (`transport/shm.cc:590 Cuda failure 101 'invalid device ordinal'`), and with
  `--gpu-bind=none` this SLURM hands every task of an H100 node the SAME card (`Duplicate GPU detected`). (ii) The unmasked flexible ask
  `--gpus=8 --ntasks=8 --nodes=1-8` places tasks by CPU and cards by card, and the two need not agree: the real eval got 4 / 1 / 3 tasks on
  2 / 1 / 4 cards and `LOCAL_RANK` indexed a card that was not there. It had aligned by luck in the 2-CPU smoke, which is exactly how it passed
  review. **The rule: for multi-GPU work here, ask for a UNIFORM shape — `--nodes=N --ntasks-per-node=k --gres=gpu:k`.** Tasks then equal cards on
  every node by construction, a job-level per-node gres leaves all of a node's cards visible to its tasks, `LOCAL_RANK = SLURM_LOCALID` indexes a
  real device, and NCCL keeps its intra-node transports. It is what every training cell in this project already runs; the flexible node count was
  invented that afternoon to dodge a scheduling problem and it brought both walls with it. The three B-row evals went out at 15:52 on
  `--nodes=4 --ntasks-per-node=2 --gres=gpu:2` (65024351 IN-1k / 65024352 SSv2 / 65024353 K400), queued on capacity: only 7 cards were free on the
  nine nodes our asks may use, so no 8-rank layout places immediately. Also settled that afternoon: **gpu277's eight idle H100s are not capacity** —
  eight ranks on it all died with CUDA `Error 802: system not yet initialized` (smoke 65024570), so SLURM reporting the node IDLE with no drain
  reason is why the partition keeps looking emptier than it is. It stays excluded, with gpu274 (slow reads) and gpu271 (prolog failures).

## 2026-09-19 — E38: seed repeats of the IN-100 controlled pairs (Fable)

Berker opened the session with a repeat order: the IN-100 experiment where our regularizer is added
to existing methods (and the untreated twins) at three seeds each — ours, DINO at its later winner
coefficient, VICReg, VISReg, SimCLR, LeJEPA, BYOL — "with confidence bars at least for the controlled
experiment"; then: "we are just repeating the existing experiments. you dont need to adjust
anything", and: the appendix treatment figure (7 families × 11 quantities vs station, I-JEPA and MAE
excluded) is what gets recreated with the seeded evals, 28 jobs. He also declared HANDOVER.md stale.

Done: card `docs/experiments/E38_in100_seeds.md` (P1–P5 pre-registered), row D-126 PROPOSED. The 14
cells are `paper_exhibits.FAMILIES` (Ours = d256vm4 vs d256vm4zonly; "d256proj" does not exist). Every
recipe was recovered from the seed-0 checkpoint's stored cfg (a CPU job unpickled the 14 archived
`_ep100.pt` payloads; `scratch/e38/overrides.py` diffed them against today's composed defaults), not
retyped — which surfaced a discrepancy: the July lanes VICReg, SimCLR and BYOL (control AND treated)
trained at **bs = 256** while the paper's appendix says 128 for all controlled runs; pairs are matched,
the sentence needs per-lane batch sizes (Berker's edit). Recipes byte-matched per lane, only `seed=`
changes (seeds 1, 2; seed 0 = the paper cells). Infrastructure: `slurm/e38_seeds.sbatch` (H100, driver
gate, gpu277 + gpu274 excluded), `slurm/e38_launch.sh` (smoke / chains / land), readers
`experiments/e38_seeds.py` (table: mean ± 95 % t-CI, paired delta; bar figure) and
`experiments/e38_zoo_seeds.py` (the appendix figure with a seed band — reproduces the paper's figure
exactly at n = 1). Landing per run = extract `.extL` → probes + depth metrics (own CSV per run under
`results/diag/e38/`) + thickness rows (append serialized by a singleton job name).

Launch: 14 one-epoch smokes at seed 99 — 12 passed in 5–10 min; the two LeJEPA smokes sat on gpu274
with the GPU at 0 % and every loader worker pegged (the E34 "gpu274 slow" anomaly), were cancelled and
resubmitted with gpu274 excluded. Lesson: a completed job cannot be an `afterok` target ("Job
dependency problem") — the verified cells launched ungated. 28 chains of 3 × 8 h H100 links are out
(ids on the card), landing chains queued behind each. Numbers land RAW; nothing read.

## 2026-09-21 — code names migrated to the paper's (D-127)

The method `floorssl` is `lambdajepa` (class `LambdaJEPA`, `sslgap/methods/lambdajepa.py`,
`method/lambdajepa.yaml`), the regularizer `SpectralConditioner` is `SACReg`, the zoo hook
`h_reg=moment` is `h_reg=sacreg`; the video fork's loss key `sslgap` is `lambdajepa`. Historical run
ids, checkpoints, stores, result CSVs and cards keep their strings; pre-rename checkpoints load through
`sslgap/ckpt/schema.py:modernize` (method, arch class paths, stored cfg). Lesson: a vocabulary break the
glossary records but the code does not execute (D-083's `spectral`) is no break — the names in the
paper, the code and the glossary must be the same string. The anonymized code supplement
(`supplement/build.py`) had applied the same mapping at package time since 2026-09-21 morning. Verification: compile of every module; `METHODS` registry; Hydra composition; CPU load of three
pre-rename checkpoints (`in100.floorssl.s0.d256vm4_ep100`, `in100.simclr.s0.e20f_ep100`,
`in100.lejepa.s0.e20f_ep100`) through `load_payload` with module rebuild + state load; selftest job
66429387 (ALL PASS); 1-epoch IN-100 smoke `method=lambdajepa` job 66430003 (COMPLETED, ep1 probe .0358,
new arch stamps `sslgap.methods.lambdajepa.lambdajepa_head`); artifacts deleted. One miss caught by the
smoke: the renamed method config still said `name: floorssl` (the rewrite rules matched the quoted Python
literal, not the YAML value) — fixed; the first smoke (66429388) failed on that `KeyError`.
Follow-up the same day (Berker: "yes please do remove"): the dead E12/E21 variants `DiagSACReg`
(`h_reg=moment_diag`), `SpectralFloor` (`h_reg=spec_floor`) and `HingeFloor` (`z_floor=hinge`) removed
with their branches, `H_KEYS` entries and the `z_floor` config key; no paper cell ever used them. Selftest
job 66439361 ALL PASS.
