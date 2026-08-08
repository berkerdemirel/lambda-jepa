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
