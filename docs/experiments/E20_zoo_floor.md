# E20 — the calibrated moment floor at h, across the whole zoo

**Status: PRE-REGISTERED 2026-07-18 (Berker: "launch calibrated (specific to the method)
moment floor loss on h to every method in our method zoo without touching their z losses").
Launch upon calibration pulls + smokes. D-row: D-045 (PROPOSED at launch).**

**Current status (as of 2026-08-07): CLOSED.** Takeaways **E20-T1** (dino's miss is dose,
not fit — T=6% is not a universal optimum), **E20-T2** (the four winning view arms are not
converged at 100 ep), **E20-T3** (the calibrated conditioner at h gains two columns in 4/5
view lanes; the benefit is view-lane-shaped) are USER-APPROVED 2026-07-19 (wording FINAL
2026-07-20); **E20-T4** (the zoo2 read: an (Ω, Λ) dial with a gain band) USER-APPROVED
2026-08-03.

## Question

The divergence ladder (E19 card) identified nominal-λ transplantation as the reason the
floor's hold is lane-dependent, and certified the fix: dose by REALIZED SHARE of the tap's
total pull. G-wave (E12-T7) showed the floor transfers at nominal λ=.02 with no tuning
(vicreg +6.1 / dino +3.6 knn); its declared headroom was the per-method λ. E20 runs the
corrected experiment: **the moment floor at every method's h, dosed per-method by the
certified share rule, shipped z losses byte-untouched — the zoo table of the floor as a
universal additive conditioner.**

## Frame

M2 control lanes verbatim (`in100.<m>.s0`, house optimizer, seed 0, 100 ep, IN-100
ViT-S/16@224); ONE additive term per cell: `+method.h_reg=moment +method.h_lamb=λ_m`.
MomentFloor consumes no construction RNG ⇒ init streams byte-match every control.
Tags `in100.<m>.s0.e20f`.

## Dose convention (the design decision; validated before launch)

λ_m = T·A_m / ((1−T)·ĝ_h), **T = 0.06** (the winning reference's formation-window share:
f2 6.2% @ep25; the certified lam50x cell 6.1%), A_m = Σ_shipped w_i·g_enc,i at each
CONTROL's ep25 checkpoint (formation window; jobs `e20p.<m>.ep25`), **ĝ_h = 0.30 = the
HELD-STATE floor pull (declared universal constant)**. AMENDED BEFORE LAUNCH (the original
draft divided by the control-state g_h): the certified formula's g_h was measured on the
ARM's held tap; control-state g_h is inflated wherever the control tap is wild (lejepa
control embed: g_h 6.24 at kl 6.38 → λ would come out .0007, 28× under the lejepa-tuned
.02 — the anchor that exposed the bias). Held-state g_h is empirically near-universal
across every held/partially-held tap measured this project: {f2 .436/.271/.236, gv2
.325/.278/.303, arm-4 .217/.252, hz .336} → ĝ_h ≡ 0.30, range .22–.44 on record.
Anchor checks under the amended rule: lejepa λ = .0146 ≈ the .02 its dose curve selected;
certified vicreg-emb cell λ = .71 vs its held 1.0 — both inside the hold basin (f2 itself
held down to 2.4% late-training share). Rejected alternatives: nominal transplant (the
ladder's identified bug); init-share matching (lam5x-equivalent, diverged); control-state
denominator (the lejepa anchor above); per-lane arm-state g_h where an arm exists (gv2 .325
for vicreg@cls — a refinement Berker may prefer; ĝ=.30 keeps one rule for all seven).
Formation-share is flat where the adversary persists (measured: gv2/hz/emb arms) and
self-amplifies where it dies — exactly f2's own reference behavior, accepted. CAVEAT standing: g_enc is trunk-module-only
([[pull-instrument-genc-convention]] memory; within-lane λ's self-consistent; lejepa's cell
alone carries the embed-inside-encoder difference — flagged for cross-method magnitude
comparisons).

## Per-method taps (declared BEFORE numbers; deviations explicit)

| method | floor tap (grad branch) | audited/declared h | deviation? |
|---|---|---|---|
| simclr | student CLS (pooled views) | same | — |
| byol | student CLS (pooled views) | same (teacher = z-side only) | — |
| vicreg | student CLS (pooled views) | same | — (gv2's tap; new dose) |
| dino | student CLS (both global crops) | teacher CLS (EMA inherits — gd precedent) | EMA indirection |
| lejepa | embed (CONTROL lane) | same | no spec_norm / no embed_calib (≠ f2's lane — lane factors on record) |
| mae | GAP over VISIBLE-token latents of the masked forward | full-image GAP | masked-view tap = the only grad-carrying trunk output |
| ijepa | GAP over context-encoder tokens | teacher GAP (EMA) | context tap + EMA indirection |

The masked/context taps parallel the view methods' situation (their floor also sees the
AUGMENTED distribution, never the eval distribution) — the tap is each method's own
training-time h computation, which is the uniform rule.

## Pre-registered directional predictions (locked before numbers; Claude's picks)

- **P-zoo-A (headline).** In lanes with nominal-dose precedents (vicreg gv2 +2.5 knn @cls
  post-D-036, dino gd +3.6), the calibrated dose ≥ the nominal cell's knn gain at declared h
  with converged-lin flat-or-up. The corrected dose is 1–2 orders above .02 where the
  adversary is strong — the E12 "interior optimum near .02" was measured on LEJEPA ONLY and
  transplanted; the ladder says the optimum is a SHARE, not a λ.
- **P-zoo-B.** Cone-carrying view methods gain most (T1/T3 mechanism): byol (ctrl rand-cos
  .84, deepest cone) ↑↑ · vicreg ↑↑ · simclr ↑ · dino ↑ (EMA-damped) · lejepa ↑ (plain lane;
  f2 remains the lane ceiling).
- **P-zoo-C (the aug-less frontier).** mae ~/↑ and ijepa ~ — smaller/uncertain: no
  view-alignment adversary at h (their cones/geometry differ), EMA dilution (ijepa), masked
  taps. A null HERE with wins elsewhere localizes the floor's benefit to view-method
  geometry; a win here generalizes it. Either outcome is informative.
- **P-zoo-D (health).** No collapse in any lane (floor is anti-collapse-shaped); risk pole is
  over-conditioning tax where λ_m lands high (vicreg-family) — watch converged lin vs
  control; kill = online probe ≤ 2× chance @ep≥10 or grad incident (K1 standing).

## Discipline

Pre-registration (this file) before numbers · λ from measured pulls recorded below before
launch · CPU dry-runs of the four new code paths (simclr/byol/mae/ijepa h_reg=moment) PASSED
against stored control ckpts (arch + state-dict keys byte-stable on default paths) · 2-ep
smokes per cell → 3×8h singleton chains (lejepa+dino → the two free H100 slots; the rest on
gpu) · `num_classes=100` on every launch · numbers land raw; NO takeaway without Berker ·
then extract (`.ext`) → v2 probes → battery → trajectory pulls (e20tp.*) → score vs controls
+ existing nominal cells (gv2/gd/f2).

## Dose measurement record (mechanical fill when pull jobs land)

| method | A_m (Σ w·g @ep25) | g_h @ep25 (control-state, RECORDED not used) | **λ_m = .2127·A_m** | job |
|---|---|---|---|---|
| simclr | 1×.46062 = .4606 | .4959 | **0.098** | 62411768 |
| byol | 1×.09625 = .0963 | .5325 | **0.0205** | 62411769 |
| vicreg | 25·.08583+25·.13456+1·1.76663 = 7.276 | .7559 | **1.548** | 62411770 |
| dino | 1×1.21264 = 1.2126 | .3559 | **0.258** | 62411771 |
| lejepa | .02·.95647+.98·.05071 = .06883 | 6.2356 | **0.0146** | 62411772 |
| mae | 1×.04630 = .0463 | .6467 | **0.0098** | 62411851 (first try 62411773 refused: yaml patch default 8 ≠ M2's 16 — arch assert; method.patch=16 on every mae/ijepa launch) |
| ijepa | 1×.01196 = .0120 | .4037 | **0.0025** | 62411852 (same patch fix) |

Note on the aug-less pair: their objectives pull the trunk weakly at formation (recon g .046,
jepa g .012 — an order below the view lanes), so calibrated λ's land tiny; the floor's
ABSOLUTE pressure is then also small. If their cells read null, the share rule itself (share
of a weak lane total) vs absolute-pressure dosing becomes a discriminating question — flagged
pre-launch, not patched.

## Execution log

- 2026-07-18: calibration pulls landed (table above; mae/ijepa patch-16 refusal + rerun on
  record). Dose convention AMENDED pre-launch (ĝ_h=0.30 held-state denominator; §Dose).
- 2026-07-18: 2-ep smokes ×7 PASSED (ep2 probes: lejepa .1454 · vicreg .1350 · simclr/dino
  .1168 · mae .1078 · ijepa .0862 · byol .0754 — all in family, h_moment_kl logging, no
  incidents). CHAINS LAUNCHED: 3×8h singleton links each — lejepa→h100-slotA,
  dino→h100-slotB, simclr/byol/vicreg/mae/ijepa→gpu (`e20-<m>`). λ's per the dose table;
  control-matching overrides per lane (byol/dino ema_base .996; dino local_size 96; lejepa
  house-hygiene set; mae/ijepa patch 16).

## AGREED TAKEAWAY (rows added only as jointly agreed)

**E20-T1 (Berker 2026-07-19: "for dino the caveat is the strength of the loss right? thats
the takeaway"; wording Claude's, veto open).** dino's miss is a DOSE effect, not a floor-fit
effect: the lane carries a monotone three-point dose ladder — λ=.02 (gd) −0.1 lin / +2.6
knn_c · λ=.02+.0401 two-tap (gd2) −1.4 / +1.9 · λ=.258 calibrated (e20f) −2.5 / −0.4 — the
floor helps dino at gentle dose and the calibrated 6%-share overshoots it. dino is the lane
where the control's lin is strongest (.6864) and the cone headroom smallest (+0.9
centering), i.e., least to condition and most to lose; the share rule's T=6% is not a
universal optimum — per-lane dose-response curvature is real (mirrors E12's "interior
optimum" on lejepa, now measured on the far side in a second lane).

**E20-T2 (Berker 2026-07-19: "simclr, vicreg, byol and lejepa would further benefit from
training more as the curves do not look fully converged" — with his stated simclr
uncertainty; wording Claude's, veto open).** The four winning view arms are still climbing
at the 100-ep cut: last-20-epoch monitor slopes lejepa +2.0 · byol +1.7 · vicreg +1.6 ·
simclr +1.2 (the shallowest, matching Berker's "not 100% sure abt simclr"); dino +0.7
flattening, ijepa ~0. Caveat carried in the row: the CONTROLS also still climb (+2.2–3.1)
— the M2 100-ep budget converges no lane fully; what extension would settle is whether the
DELTAS hold or grow, not just the absolutes. Any longer-budget rerun is a new launch
decision (epochs change = frame change; D-004 budget convention).

**Raw-numbers table (Berker's requested form; lin raw · knn centered both sides):**

| lane | ctrl lin | ctrl knn_c | +floor lin | +floor knn_c |
|---|---|---|---|---|
| lejepa | .6036 | .5286 | .6700 | .6288 |
| byol | .5836 | .4944 | .6448 | .5900 |
| simclr | .6060 | .5144 | .6174 | .5684 |
| vicreg | .6452 | .5648 | .6432 | .5946 |
| dino | .6864 | .6080 | .6610 | .6052 |
| ijepa | .4836 | .3574 | .4284 | .3784 |
| mae | .4344 | .2342 | .3502 | .2332 |

## Numbers land below this line as they arrive; AGREED TAKEAWAY only after joint discussion.

### Mid-training record (RAW; reader `experiments/e20_curves.py` →
### `results/figures/e20/e20_curves.png`; chains at ep15–70)

h_moment_kl: **held in all seven lanes** — running minima current everywhere, no active
divergence (arm-4 signature absent zoo-wide at calibrated dose): vicreg .17@ep33 · mae
.20@ep50 · dino .21@ep15 · simclr .30@ep34 · ijepa .31@ep70 · byol .35@ep29 · lejepa
.37@ep18 (plain-lane embed; f2's own ep18 ≈ .29–.31). Early recovered warmup transients in
simclr/byol (≤ep1.2) noted, not flagged.

Online-monitor deltas, arm − control at MATCHED epochs (CAVEATS RIDE: E12-T9 monitor bias
~+3 toward floor arms, mechanism unresolved; mid-training ≠ converged — no extrapolation,
c015 lesson; offline v2 probes at landing are the arbiter):
view lanes all positive and growing through formation — byol +13.8@ep25 · simclr +11.3@ep25
· vicreg +10.3@ep25 · lejepa +8.0@ep10 · dino +5.0@ep10. The aug-less pair INVERTS
mid-training: mae +5.8@ep5 → **−2.2@ep25 → −1.5@ep50**; ijepa +9.5@ep5 → +2.9@ep25 →
**−0.5@ep50** — early boost, control catches and passes (the P-zoo-C uncertainty zone,
raw). Landing pipeline declaration: standard `.ext` extraction (do_eval+do_pairs defaults —
the all-controls/E19 convention; L-tap/orbit extensions only on request) → v2 probes →
battery → e20tp pulls at cadence.

### simclr + byol + vicreg CONVERGED (2026-07-18 evening; RAW; offline v2, declared h=cls;
### knn Δ vs CENTERED control per convention; lin raw)

| lane | e20f lin / knn | ctrl lin / knn_centered | Δlin | Δknn_c |
|---|---|---|---|---|
| byol (λ=.0205) | .6448 / .5914 | .5836 / .4944c | **+6.1** | **+9.7** |
| simclr (λ=.098) | .6174 / .5646 | .6060 / .5144c | **+1.1** | **+5.0** |
| vicreg (λ=1.548) | .6432 / .5948 | .6452 / .5648c | −0.2 | **+3.0** |

Raw notes: byol = the zoo's largest gains on BOTH columns (+6.1 lin is ~5× any prior floor
cell's lin effect; it was P-zoo-B's deepest-cone pick); vicreg at 77× gv2's dose ≈ 3× gv2's
centered-knn gain (+3.0 vs +1.1) at flat lin, and its raw knn .5948 ≈ c015's .5942 (the
own-term 6× closure cell — calibrated transplant matches converged own-term); simclr's own
E17 uniform cell was +6.3 raw/+4.4 centered — e20f (+7.0 raw/+5.0 centered) edges it.
Arm-side centered values: rerun job 62416256 (expected ≈ raw; floors trained cones away).
Monitor-vs-offline note: online bests .6666/.6276/.6638 vs offline lin .6432/.6174/.6448 —
the E12-T9 bias pattern again (present in vicreg/simclr, ~absent in byol).

### mae + ijepa CONVERGED (2026-07-18; RAW; offline v2, declared h)

| cell | lin_v2 | knn200 | Δlin | Δknn |
|---|---|---|---|---|
| mae ctrl (h.gap) | .4344 | .2396 | — | — |
| **mae e20f (λ=.0098)** | .3502 | .2386 | **−8.4** | −0.1 |
| ijepa ctrl (teacher gap) | .4836 | .3474 | — | — |
| **ijepa e20f (λ=.0025)** | .4284 | .3784 | **−5.5** | **+3.1** |

**Reporting convention (Berker 2026-07-18, sharpening the ask): kNN deltas are cited vs the
CENTERED control** — post-hoc centering is free, so the raw-ctrl comparison overstates the
floor by whatever the control's cone was worth (the E17-T3 convention: "uniform +6.3 knn
(+4.4 centered)"). Linear columns are affine-invariant (bias absorbs the mean) and stay
raw-vs-raw. Instrument: `experiments/e20_centered.py` → results/diag/e20_centered.csv;
view-lane rows auto-fill at landing. Landed cells, corrected: **ijepa e20f knn +2.1 vs
centered ctrl** (.3784 vs .3574c; raw-vs-raw +3.1 included +1.0 of free centering; the arm
itself is exactly centering-invariant — cone trained away) · **mae ≈ flat either way**
(.2332c vs .2342c = −0.1; lane cone-free, the −8.4 lin tax has no mean component).
Controls' free-centering worth, to be subtracted at landing: byol +3.5 > simclr +2.0 >
vicreg +1.4 > dino +0.9 > lejepa +0.5 (the P-zoo-B cone ordering) — byol's raw knn delta
takes the biggest haircut. Mid-training MONITOR deltas are linear-head numbers ⇒ not
cone-inflated (they carry only the E12-T9 online bias caveat).

P-zoo-C direction realized in its DAMAGE branch: the reconstruction/prediction lanes take a
large LINEAR tax at even the tiniest calibrated doses (h_kl was held in both — the floor won
its constraint and the lane paid at lin; ijepa keeps the floor's usual knn gain, mae doesn't
even get that). The floor's h-benefit is NOT aug-family-universal — raw scope boundary for
the zoo table. Candidate mechanisms for the joint discussion only (NOT takeaways): the
masked/context-view tap deviation (the floor conditions a distribution the eval probe never
sees); no view-alignment adversary means the floor's conditioning has nothing to
counterbalance; monitor inversion at ep25–50 foreshadowed both. Berker's standing priority:
these are boundary rows, no further spend.

### ZOO COMPLETE — lejepa + dino converged (2026-07-19; RAW)

lejepa e20f **.6700 / .6284** vs ctrl .6036/.5286c → **+6.6 lin / +10.0 knn_c** — and vs f2
(.6578/.6056): the PLAIN-lane calibrated cell beats the reference winner on both columns
(+1.2 / +2.3). FLAG: lanes differ in lr (e20f = control's 1e-3; f2's E12 lane = 3e-4) — not
a pure accessory ablation; an f2-lane-lr cell would isolate it if wanted. dino e20f
**.6610 / .6038** vs ctrl .6864/.6080c → **−2.5 lin / −0.4 knn_c** — the mid-training
trailing confirmed at convergence: calibrated .258 OVERSHOT dino (nominal gd: −0.1 lin,
+2.6 knn_c — better on both); dino = the rule's one view-lane miss, the dose-response's far
side in the lane with the strongest control lin. Online bests .6824/.6940 vs offline
.6700/.6610 (monitor bias again).

**FULL TABLE (Δlin raw · Δknn vs centered ctrl):** lejepa **+6.6/+10.0** · byol
**+6.1/+9.7** · simclr +1.1/+5.0 · vicreg −0.2/+3.0 · dino **−2.5/−0.4** · ijepa −5.5/+2.1 ·
mae −8.4/−0.1. Directional score vs the locked predictions: P-zoo-B's within-view ordering
largely realized (byol top-2 as picked; dino weakest view lane); P-zoo-A SPLIT (vicreg:
calibrated ≫ nominal ✓; dino: calibrated < nominal ✗ — the share rule is not universally
monotone; whether T=6% needs a per-lane ceiling or dino's optimum share is lower = joint
question); P-zoo-C damage branch (aug-less lin tax); P-zoo-D's over-conditioning pole
realized in dino + the aug-less pair. Mechanical follow-ups running: e20tp cadence pulls
(view lanes ×4 ckpts, jobs submitted 07-19), final centered rerun (62419995), batteries.
NO takeaway rows without Berker.

**E20-T3 (USER-APPROVED 2026-07-19; full row in DECISIONS):** the calibrated floor at h
delivers two-column gains in 4/5 view lanes (lejepa beats f2, lr flag standing); dino =
dose overshoot (T1); aug-less lanes take lin damage — the h-benefit is VIEW-LANE-shaped.

## dino-fix fork (Berker 2026-07-19: "for dino part also launch a fixed version … if there's
## additional complication … resolve")

The clean single-tap dose curve has two points (.02 → −0.1/+2.6c · .258 → −2.5/−0.4; gd2's
middle is two-tap-confounded). Two bracketing cells complete it: `e20f_lam008` (λ=.008) ·
`e20f_lam06` (λ=.06) — everything else e20f-verbatim (same lane overrides). Pre-registered:
the curve peaks in [.008, .06]; the winning cell = the fixed dino. Claude's pick: peak near
.02–.05 (rise 0→.02 measured, decline by .258; if .008 ≥ .02's gain the optimum is even
lower and dino's effective adversary ≈ nil). Companion instrument `e20_opposition.py` (the
complication's mechanism read): cos(∂h_floor/∂trunk, ∂lane/∂trunk) at every control's ep25 —
prediction: dino's lane gradient is floor-ALIGNED-or-neutral (its teacher-centering already
does floor work; control effrank 93 vs 34–37) while the view adversaries are opposed —
the measured direction factor missing from the share rule (magnitude ≠ opposition). If the
cosine separation fails to materialize, the opposition account is refuted and the dose curve
alone carries the fix. 2-ep smokes → 3×8h chains, gpu.

Opposition read LANDED (job rerun after a dino multi-crop list-batch fix; e20_opposition.csv):
cos(lane, floor) at ep25 — ijepa +.159 · mae +.129 · byol +.048 · simclr +.033 · vicreg
+.016 · lejepa −.015 · **dino −.053**. The pre-registered ally-account is REFUTED as stated
(dino is the MOST OPPOSED lane, not the most allied): its overshoot reads as direct gradient
interference at high λ (consistent with E20-T1's strength-of-the-loss framing). Inverse
surprise: the aug-less lin-damaged pair is the most ALIGNED — but their lane gradients are an
order weaker (|g| .046/.012), so their calibrated cells ran hot (realized share ≈8–14% vs the
6% target: ĝ=.30 underestimated their control-state g_h .65/.40). Raw; single-batch cosines
do not dose lanes — the dose curve (lam008/lam06, chains running) carries the dino fix per
the declared fallback.

**Dose cells LANDED (2026-07-20; ep100 → extract 62434437/440 → probes; offline
teacher.h.cls, lin = linear_raw_v2 · knn = knn200 raw, ctrl centered .6080 for the knn_c
convention; RAW, joint read owed):**

| λ | 0 (ctrl) | .008 lam008 | .02 gd | .06 lam06 | .258 e20f | (.02+.04 gd2, two-tap) |
|---|---|---|---|---|---|---|
| lin | .6864 | **.6900** | .6852 | .6738 | .6610 | .6722 |
| knn | .5994 (.6080c) | .6296 | **.6342** | .6286 | .6038 | .6272 |

The five-point single-tap curve is complete: lin is monotone-declining in λ with the zero
crossing between .008 and .02 — λ=.008 is the only lin-positive cell (+0.4 vs ctrl) — while
knn peaks at λ=.02 (+2.6 vs centered ctrl) and stays +2.1..+2.2 at .008/.06. Pre-registered
"curve peaks in [.008, .06]" verdict: TRUE on knn (interior peak .02), EDGE on lin (peak at
the .008 boundary; the option "if .008 ≥ .02's gain the optimum is even lower" is the realized
branch on lin). Online monitors ran .7092/.6980 (lam008/lam06) vs offline .6900/.6738 —
monitor optimism again. T=6%-rule amendment discussion (dose curvature, per-lane optima ≪ T
for dino) is QUEUED for the joint pass; no wording without Berker.

## Comparison pass (2026-07-19, HANDOVER priority ii; RAW — no takeaway)

**(a) Anatomy decomposition at cadence** (e19_floor_anatomy argv mode extended with dino
multi-crop global-crop handling; 32 ckpts: 5 view-lane e20f arms + missing controls ×
ep25/50/75/100; full rows in `results/diag/e19_floor_anatomy.csv`). The floor-KL at each
lane's train-time h tap (cls; lejepa embed), split cone/scale/aniso, arm/ctrl at ep100:

| lane | kl | cone | scale | aniso | R (win-inst var share) | per-dim var |
|---|---|---|---|---|---|---|
| simclr | .20/1.71 | .02/.02 | .01/1.03 | .17/.65 | .39/.26 | .82/.05 |
| byol | .21/2.16 | .00/.05 | .01/1.28 | .20/.84 | .37/.22 | .85/.03 |
| vicreg | .13/1.64 | .00/.07 | .00/1.01 | .12/.56 | .36/.21 | .93/.05 |
| dino | .20/1.13 | .01/.01 | .01/.50 | .19/.61 | .58/.17 | .85/.16 |
| lejepa | .25/2.95 | .00/.06 | .00/.80 | .25/2.09 | .48/.12 | .95/.08 |

Cone and scale are fully won in every arm (per-dim var pinned .78–.95 vs collapsed controls
.02–.16); aniso is the one contested channel (.12–.25 residual) — the E19-T1 anatomy
reproduced zoo-wide. Controls' kl lives in different channels per lane: lejepa = cone+aniso
at embed (6.36 total @ep25), byol = scale (collapse-adjacent per-dim var .02), dino = the
smallest total (1.13 — "least to condition" in anatomy form). R rises 2–3× in every arm (the
floor spends h view-invariance), most extremely dino (.58 vs .17 — the lane that lost).
z-side @ep100: vicreg/dino/byol/lejepa z unmoved (kl/R/pos ≈ ctrl — "z untouched" holds in
anatomy); EXCEPTION simclr raw-z scale ×20 (per-dim var 57 vs 2.8, kl 27 vs 1.7) — NT-Xent
normalizes z so raw scale is loss-free; the h-floor's variance pressure propagated there.

**(b) Battery vs control per lane** (`experiments/e20_battery_table.py` →
`results/compare/e20_battery_vs_ctrl.csv`; declared-h spaces = the headline probe spaces).
At h, arm/ctrl: effrank 195–330 / 14–92 (mae 305/13.6 the extreme); kurt_topeig.worst falls
toward Gaussian everywhere except dino (~flat .91/.89); EP drops an order of magnitude;
uniformity → −3.7..−3.9 (near-uniform sphere) from −0.8..−3.2; pairs: alignment (pos-pair
distance²) WORSENS ~.65 vs .09–.28 ctrl and cos_invariance drops .67 vs .88–.95 — the floor
trades h view-alignment for spread, zoo-wide. z-block: unmoved except simclr scale (above)
and byol/dino mild effrank up (33.9/23, 91.5/80). CAVEAT RESOLVED (2026-07-19):
the last backfill audit (62420969) COMPLETED and the table was regenerated with full
gauss-ctrl columns (`results/compare/e20_battery_vs_ctrl.csv`, 18:21 build; aug-less lanes'
z-block rows are "-" by design — no z heads in those comparisons). Control-flavor note: comparison-pass
controls = `.s0.ext` extractions; the card's headline ctrl numbers for vicreg/dino/lejepa are
e17c-flavor re-extractions of the SAME ckpt (probe deltas ≤ .002 — e.g. vicreg lin .6430 vs
.6452, lejepa embed lin .6022 vs .6036).

**CORRECTION (2026-07-20, found while building the E21/E20f metric figures): this section's
PAIR numbers were frame-mixed.** `e20_battery_table.py`'s pairs loader was last-row-wins over
the multi-manifest pairs files, so the compare CSV's h.pairs rows silently read whichever aug
stack was last per file — `own_<m>` for simclr/byol/vicreg/dino/mae/ijepa, `foveal_v1` for
the lejepa ctrl (no own block), `audit_v1` only for lejepa-e20f. Fixed at source (loader now
filters `@audit_v1`, the PROTOCOL §5 cross-lane stack; `e20_battery_vs_ctrl.csv` regenerated
2026-07-20). Corrected audit_v1 values at declared h — alignment arm/ctrl: lejepa .806/.090 ·
byol .868/.133 · simclr .750/.285 · vicreg .879/.159 · dino 1.309/.639 · ijepa 1.654/1.089 ·
mae 1.758/.563; cos_invariance arm/ctrl: lejepa .597/.955 · byol .566/.934 · simclr
.625/.857 · vicreg .560/.921 · dino .346/.680 · ijepa .173/.456 · mae .121/.719. The
paragraph's ".65 vs .09–.28 / .67 vs .88–.95" ranges are SUPERSEDED (they understated the
arm-side alignment shift and overstated the aug-less controls' invariance — own/foveal stacks
are gentler than audit_v1). The direction — arm less view-aligned than its control, every
lane — is unchanged in the pure frame. Single-space metric rows (single-manifest files) were
never affected. Gotcha carried: any keyed read of a `.pairs.csv` must filter its manifest.

**(c) e17-score deep pass re-pointed** (`experiments/e20_score.py` → full text in
`results/compare/e20_score.txt`; orbit @o8 stores extracted for arms + controls, jobs
62420983–92). Inv-jump (head z-margin − h-margin): the four winning lanes SHRINK it under
the floor (ctrl +.54/+.48/+.41/+.51 → arm +.32/+.14/+.22/+.11 for lejepa/vicreg/simclr/byol)
while the h-margin itself expands +23/+35/+20/+39 pts — entirely by rand_cos → ~0 (cone
gone: mu_share .50–.77 → .01–.06). The head's jump-split job INVERTS: controls' jump was
−Δrand (cone removal, +.48–.70); arms' jump is +Δpos (pos-restoration, +.14–.45). dino
INVERTS the lane pattern: its control h-margin was the zoo's largest (.537) and the floor
collapsed it to .329 (pos .683→.348); head burden GROWS (+.149→+.370). diagKL@h .01–.21 vs
.65–1.5 ctrl zoo-wide. byol seg-stretch printout carries a key collision (proj vs pred both
end "tap1/out"); printed tap1→out 9.52 = pred.tap1→pred.out.

**(d) Class-pair d′ + centered figures** (`experiments/e20_pair_dprime.py` →
`results/diag/e20_pair_dprime.{csv,npz}`, figures `results/figures/e20/e20_pair_dprime.png`
+ `e20_centered.png`, INDEX rows added). Per-pair two-class discriminability along the
mean-difference axis: view lanes DROP or hold (mean d′ lejepa 4.43→4.16 · simclr 4.01→3.79 ·
vicreg 4.38→3.76 · dino 4.91→4.41 · byol 3.78→3.80; frac(arm>ctrl) .22–.56) while their
probes gained; the aug-less pair RISES (mae 1.71→3.06, 98% of pairs; ijepa 2.80→3.17, 85%)
while their lin dropped. The probe deltas do not ride pairwise mean-separation in either
direction — mechanical note: the floor's within-class spread inflation (R↑) sits in the d′
denominator. NO takeaway without Berker.

## E20-T4 — the zoo2 orbit-calculus read (2026-08-03)

**E20-T4 (USER-APPROVED 2026-08-03; zoo2 figure under the D-068 calculus; full row in
DECISIONS):** the e20f h-conditioner moves every aug-based lane up in (Ω_h(cls), Λ): Ω roughly
doubles (simclr .37→.64 · byol .39→.75 · vicreg .39→.78 · lejepa .18→.48 cls / .12→.70 embed;
dino .58→1.98 on the deliberate E20-T1 overshoot dose) and Λ rises 5/5 (1.25→1.66 · 1.47→2.05 ·
1.06→1.54 · 1.39→2.55 · 1.24→3.13). Arms landing at Ω(cls) ≈ .5–.8 / Λ ≈ 1.5–2.1 improve
probes, kNN 3–6× the linear gains (byol +.13 · lejepa +.10 · simclr +.07 · vicreg +.04 kNN);
the one overdosed arm (dino → Ω 1.98, Λ 2.55) is flat-to-down — its PRE-treatment Ω sat
nearest the others' band and the treatment moved it far out: a minimal→bump→overtreat shape at
n=5, hypothesis-grade, tested under control by E23 stage C. **Reading (Berker, jointly
precised):** Λ>1 = the head relaxes (preserves) image-center/content separation relative to
aug-orbit energy — healthy when the aug term does not dominate; both observed extremes (2.55
overshoot, 3.13 bottleneck) sit off the gain band. lejepa's Λ 3.13 is real — Λ is dimension-free
(the d-tail check: d256→d128 sits on the √dim null; a, b levels carry √(D_z/D_h) dimension mass
+ scale gauge, Λ cancels both) — but mechanism-distinct (512→16 hard bottleneck = selectivity by
subspace choice, not reshaping): excluded from band estimation. mae/ijepa excluded (no aug
objective; the conditioner is actively toxic for mae: Ω 3.4→13.4, probes collapse .41→.33 lin).
