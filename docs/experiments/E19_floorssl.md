# E19 — floorssl: the moment floor as the sole anti-collapse term at z (Z-wave arm 1)

**Status: PRE-REGISTERED 2026-07-17 (Berker: "we should focus on our f2 (winner). you have a
theory of why moment works at h, we should move it to z. thats a big priority."). Launch upon
pull + smoke. D-row: D-042 (PROPOSED at launch).**

## Question

E12/E17 established WHY the moment floor organizes h (T6: safety = cluster-blindness × activity
= non-absorbable rotation-swept spectral floor; the winner f2 at ε-dose). E19 moves the term to
the space every published method actually regularizes — z — as the ONLY anti-collapse term of a
minimal method: **does the safe half of the KL suffice as a method-grade non-collapse
mechanism?** Menu note: Z1 ≡ Z3 (the "minimal method" and the "vicreg var+cov→floor swap" are
the same loss, because vicreg's alignment is already plain MSE) — one run fills both roles.

## Frame

vicreg lane end-to-end, byte-matched to the E17 control `in100.vicreg.s0.e17c.ext`: ViT-S/16
@224 IN-100, bs=256, V=2 BYOL pair, expander 384→2048→2048, house AdamW, seed 0, 100 ep.
Module construction identical (MomentFloor consumes no RNG) ⇒ same seed-0 init stream as the
control — the comparison is loss-terms-only.

## Arms (the 2×2; two cells already exist)

|  | z = var+cov (shipped) | z = floor |
|---|---|---|
| **no h-term** | control `e17c` (exists) | **arm 1 `floorssl`** (this launch) |
| **h-floor @cls, λ=.02** | `e12gv2` (exists, scored) | **arm 2 `floorssl_hz`** (this launch; Berker 2026-07-17 "momentfloor on both h and z … you can also launch that") |

Arm 1 `in100.vicreg.s0.floorssl` — `+method.anticollapse=floor +method.w_floor=19.10`:

L = 25·MSE(z_a, z_b) + w_floor · MomentFloor(z_pooled)

Arm 2 `in100.vicreg.s0.floorssl_hz` — arm 1 + gv2's h-side floor VERBATIM
(`+method.h_reg=moment +method.h_lamb=0.02 +method.h_tap=cls`): the full recipe — ε-dose floor
at h (conditioning) + destination-dose floor at z (anti-collapse). Mirroring gv2's λ keeps all
four cells comparable; two independent MomentFloor instances (one per space), each drawing its
own fresh frame per step.

**RNG statement, precise (Berker's catch 2026-07-17):** MomentFloor consumes no RNG **at
construction** (no parameters, no draws in `__init__`) — the seed-0 module-init stream, and
hence every weight init, is byte-identical across all four cells. At TRAINING time it draws a
fresh random frame every forward call (torch global RNG, unseeded per step by design — the
SIGReg convention), so the training-time RNG stream diverges from the control's after the first
step. Identical inits, divergent stochastic trajectories — the same situation as every E12/E17
added-term arm; "byte-matched" refers to init + architecture + data order, not the step-level
noise sequence.

- Floor input POOLS the two views (n = bs·V = 512 on the fresh 128-d slice ⇒ n/d′ = 4, the
  declared operating point; pooling is the floor's own E12/f2 convention — it reads batch
  moments, unlike the per-view var-hinge). Deviation from vicreg's per-view var/cov noted.
- Dose = **destination dose by equal-pull**: w_floor · g_enc(floor) = the SHIPPED var+cov
  bundle's weighted pull (25·g_var + 1·g_cov = 52.18, E17 record, job 62341125, same
  frame/seed/first-batch). This is deliberately NOT ε-dose: at z the anti-collapse term does the
  method's whole non-collapse job; the E12 two-regime theory says whitening-level pressure at a
  DISPOSABLE space is not representational damage at h — what h receives is the conditioning
  through the expander. That transplanted claim is exactly what E19 tests.

## Pre-registered directional predictions (locked before numbers)

- **P1 (headline).** floorssl ≥ the vicreg control at h (control converged: lin 64.5 /
  knn200 55.1): the floor is mean-seeing (removes the z-cone directly; gv2 already showed the
  λ=.02 h-side floor zeroes vicreg's head jump) and cluster-blind (spends no class structure).
  Directional pick (Claude): h-probes within noise or above control; NOT a large jump — the T3
  winners' gains came from h-placement, which this arm deliberately does not use.
- **P2 (z-geometry, mechanical).** cone@z → ~0 (vs control's .715 mu_share@h / z-cone .84
  rand-side); effrank(z) above the control's (logdet barrier vs hinge); B/T@z recorded
  exploratory (vicreg's z was the least class-aligned measured, .16).
- **P3 (failure mode / kill).** If per-step 128-slicing under-covers 2048-d at destination duty
  (f3/D-025: slicing mixes eigenvalues toward the mean), collapse leaks through between slice
  refreshes: watch z-std + probe-vs-chance; K1 grad incident standing; K-collapse = online probe
  ≤ 2× chance at ep ≥ 10 or feature std → 0 ⇒ stop, read curves at onset (house default). This
  failure would itself be the Z4 motivation sharpened (coverage, not fidelity, as slicing's cost).
- **Scope note on aug families (Berker 2026-07-17, after the aug report).** The view pipelines
  are per-method BY DESIGN (D-004 recipe fidelity): lejepa = 4 symmetric strong-photometric
  views; vicreg = BYOL asymmetric pair (jitter 0.4). Consequences, agreed: (i) **floorssl is
  aug-unconfounded** — it rides vicreg's own pipeline against a byte-matched control, so the
  floorssl-vs-control delta is pure loss-term; (ii) **f2's win carries an aug-family scope
  caveat** — the E12/E17 floor-at-h result lives entirely inside lejepa's aug family (which
  also shapes the cloud geometry the connectivity instrument measures); the size of the f2 win
  need not transfer across aug families. E19 is, incidentally, the first floor test under a
  different aug family (at z; an h-placement under vicreg augs = gv2, which exists).
- **Scope.** One seed, one dose convention. NOT launched yet: the Z4 estimator cell (full-cov
  accumulated ≥16 batches at matched pull — the freshness-vs-coverage contrast), dose grid,
  simclr host. They queue on this arm's read.

## Dose measurement record (mechanical fill at launch)

| quantity | value |
|---|---|
| shipped var+cov weighted pull (E17 record 62341125) | 25×0.661 + 1×35.65 = **52.18** |
| fresh g_enc(moment_kl @ z pooled), job 62395517 | 2.73191 (loss 1.1742) |
| **w_floor = 52.18 / 2.732** | **19.10** |
| z-side inv reproduction check (E17: g_inv 0.4172) | 0.41567 (−0.4%; bf16/op-order drift from the added autocast-off floor region — lane confirmed) |
| arm-2 h-side dose | gv2's λ=0.02 verbatim (NOT re-pulled — the 2×2 requires the identical cell) |

## Trajectory scoring requirement (Berker 2026-07-17 — dose matching is init-only)

Same rule as E18: w_floor was equal-pulled to var+cov on the FIRST batch, and the two
functionals saturate differently (the hinge saturates hard — c015's trΣ froze by ep25; the KL
floor's pressure is graded). Scoring records per-term encoder-grad norms + the geometry
trajectory (pos/rand, effrank, d′, connectivity margins, z-cone) at ep25/50/75/100 for both
arms vs control — not just converged probes.

## Discipline

Pre-registration before numbers (this file) · dose by measured pull recorded before launch ·
2-ep smoke → 3×8h chain (gpu partition — both H100 singleton slots carry the E18 chains until
they converge; wall-clock to ep100 is comparable) · grad_clip=1.0 (vicreg lane default) ·
numbers land raw; NO takeaway without Berker · then extract (`.ext`, declaration-agnostic) →
probe → battery (ISO-ladder v2 rides along) → score vs `in100.vicreg.s0.e17c` + hpull_varcov
c015 as the second comparator (own-term-at-h winner).

## Execution log

- 2026-07-17: `anticollapse=floor` switch landed in `sslgap/methods/vicreg.py`; CPU-validated
  (default term set byte-unchanged {inv,var,cov}; floor branch {inv,moment_kl}, backward clean).

## Arms 3–4 — the conduit fork (D-043; Berker 2026-07-17b: "add an affine transform to
## vicreg's trunk and rerun floorssl runs … gpu partition")

The 07-17b anatomy found the reach-back asymmetry (raw record below): f2's floor@embed cleans
CLS through lejepa's BARE affine emb; floorssl's floor@z transmits nothing through the BN
expander; within-lejepa cross-check — sigreg@proj behind the BN'd projector leaves embed dirty.
Candidate mechanism: **BN between tap and trunk = moment firewall; bare on-path affine =
conduit.** Arms 3–4 transplant the lejepa topology into vicreg: `emb = Linear(384→384)` (no BN
— the point; no init calibration — STRICTER than f2's calibrated lane; created AFTER the
expander so trunk/proj init streams byte-match all cells; expander consumes emb-out = shared
duty, mirroring lejepa's embed→projector path).

|  | no h-term | h-floor λ=.02 @emb-out |
|---|---|---|
| **+affine** | arm 3 `floorssl_emb` (confound control) | arm 4 `floorssl_hz_emb` |

z-side = floor destination w=19.10 verbatim in both (pairs with arms 1–2 → a second 2×2:
affine × h-floor under z=floor). λ=.02 and w=19.10 verbatim (comparability convention of arm
2); init pulls measured for the ledger only (tag e19emb), not used to re-dose. Probe monitor
and eval stay at CLS (byte-same instrument as all cells). **Declaration flag for Berker:** with
the affine on-path, D-036's rule ("h = projector input") would move the declared h to emb-out —
not redeclared here unilaterally; extraction captures both spaces.

Pre-registered directions (locked before numbers; Claude's pick = P-emb-A):

- **P-emb-A (conduit confirmed):** arm 4's h_moment_kl@emb falls f2-like — monotone, no hump
  (the affine absorbs the scale part that stays frozen at every direct-CLS tap), converged
  value approaching the estimator floor ≈.07 + a real-aniso residual; AND the CLS monitor
  inherits conditioning (kl@cls < hz's, per-dim var above hz's .185, cone dead) — shared-duty
  conduit transmits, the lejepa mechanism reproduces in vicreg.
- **P-emb-B (absorb-only):** h_kl@emb falls BUT CLS stays at the unfloored state (kl-monitor
  ≈ floorssl's ~2.5, var ~.09) — affine compliance is local; f2's reach-back needed more than
  shared duty (e.g. lejepa's inv routing through embed). Kills the conduit story as stated.
- **P-emb-C (affine confound):** arm 3 vs arm 1 shifts probes/z-geometry materially with NO
  h-term — capacity/plumbing effect; then arm 4 reads against arm 3, not arm 1.
- Health: K1 standing; 2-ep smokes; same kill criteria as arms 1–2.

**Arm 5 `floorssl_hz_emb_cal` (D-044; Berker 2026-07-17b: "i was expecting a well behaving f2
like h_moment_kl from floorssl_hz_emb — create your run accordingly").** The f2-COMPARABLE
cell: arm 4 + lejepa's embed_calib transplanted (`+method.emb_calib=true` — one-shot
first-batch per-dim (μ,σ) fold into the emb Linear; extras-guarded across resume; CPU-validated
once-only + running-chain byte-safety). The tap starts at (0, I): the floor's entire ε-budget
holds ANISOTROPY from step 0 — the arm-4 diagnosis (anatomy @ep10: kl 2.00 = aniso 1.96, scale
.0008, cone .04 — scale/cone already won, the rise is pure aniso from the unconstrained
Linear) is thereby isolated from the init transient. Pre-registered: **P-cal-A** — h_kl@emb
starts near the estimator floor (~.07) + small aniso and stays f2-like low ⇒ f2's behavior
reproduces in vicreg; the arm-4 rise was the init-transient's imprint on early formation.
**P-cal-B** — h_kl climbs from the clean start toward arm 4's level ⇒ aniso production is
lane/architecture-intrinsic (free on-path Linear ahead of BN at ε-dose), and f2's difference
lies elsewhere (candidate: lejepa's inv routes THROUGH embed — shared invariance duty
constrains the spectrum; vicreg's does not). Claude's pick: P-cal-B (the aniso source is the
Linear's own training dynamics, which calibration does not constrain — only re-bases).
Comparators: arm 4 (calibration off/on), f2 (cross-lane shape), gv2/hz (tap contrast).

Arm-5 execution: smoke 62397049 passed health (ep1/2 .0548/.1266, no incidents); chain
LAUNCHED `e19-flr-hzcal` 62397056/57/58 (3×8h singleton, gpu). **CORRECTION (Berker's catch,
same evening):** the in-flight claim "calibration verified — h_kl on f2's early trajectory"
was WRONG — it read only the first ~40 steps. Full record: h_kl bottoms at .69 (~s33–140)
then RE-INFLATES: .94@s279 · 1.19@s418 · 1.41@s556 · 1.62@s988 (end of ep2, ≈ the
uncalibrated INIT level, still climbing toward the twin's ~2.0). Calibration re-based the
start only; the tap escapes the target within ~1 epoch either way. f2 reference: .84@~s1500 →
.19@ep100, monotone, no re-inflation ever. The structural question this poses is priority (i)
of the next session (SESSION_OPENER.md) — stated there with zero direction; the P-cal-A/B
pre-registrations predate this read and must not steer it. Cosmetic artifact for the record:
BOTH emb arms log s1 = 1.63 (the uncalibrated init value) because torch.autocast caches the
pre-fold bf16 weight cast within step 1's region — the fold lands from s2 (cal s2 = 1.02 vs
uncal s2 = 1.51); one step, immaterial, same pattern would apply to lejepa's embed_calib.
CAUTION ROW: the first init-pull record (tag e19embcal, job 62397048) is INVALID — values
byte-duplicate the uncalibrated e19emb row (compute node served a stale vicreg.py seconds
after the edit); superseded by tag **e19embcal2** (job 62397059).

**Arm 5 KILLED ~ep3 (Berker 2026-07-17c: "you can kill hz_emb_cal run thats failed the
constraint but still running"; chain 62397056/57/58 scancelled, verified gone).** The arm's
datum = the ep0–2 record above (calibration re-bases the start only; the tap escapes the
target within ~1 epoch and re-inflates toward the twin's level — P-cal-B direction at the
recorded horizon). Init-transient is thereby ELIMINATED as the explanation of the arm-4 rise;
the structural question (what f2 has that this lane lacks) = priority (i), open.

## Divergence-diagnostic ladder (Berker /goal 2026-07-17c: "identify the issue between f2
## line and this one. try lr match first, then other options (head dimensionality and so
## on) … do not wait until 100 epoch runs finish … be sensitive to the divergence points")

Base = arm 5 VERBATIM (emb_dim=384 + emb_calib + h-floor λ=.02@emb + z-floor w=19.10; clean
(0,I) start isolates the aniso channel). One knob per cell; λ/w never re-dosed. SHORT single
jobs (~2.5 h ≈ ep25, no chains, gpu partition, ≤8 concurrent per the goal grant; kill
criteria standing). **Read protocol:** h_moment_kl vs step, epoch-aligned (vicreg ~495
steps/ep, f2 ~990); failure trace = arm 5 (wandb a68qwozl: bottom .69 @s33–140 → .94@s279 →
1.19@s418 → 1.41@s556 → 1.62@s988, climbing); hold template = f2 (monotone under ~.85, never
re-inflates). Per cell record: bottom value+step, divergence step (first sustained rise ≥.15
above bottom), value at ep1/2/3(/10 where the job reaches it), end slope. Candidate map from
the priority-(i) inventory: lr → C5; head-out dim → C3 (feasibility); spec_norm → C4;
emb-wd0 → C2 (gauge/balanced-split).

| cell | override on arm-5 base | tests | pre-registered direction (Claude) |
|---|---|---|---|
| `embdiag_lr3` | `method.lr=3e-4` (f2's lane lr; schedule shape unchanged, epochs=100) | C5 | partial: bottom later/deeper, still re-inflates (lr scales the dynamics, not the adversary structure); full hold would echo E10's lr-fuse |
| `embdiag_d128` | `method.expander_dim=128` (2048→128; z-demand becomes rank-feasible from 384-d; floor@z = EXACT full-dim KL at unchanged n/d′=4 — d_slice=128 constraint is why not 16) | C3 | strongest single-knob flip candidate: z-pressure dies post-satisfaction → re-inflation weakens or holds |
| `embdiag_spec` | `+method.spec_norm=true` (spectral norm on expander Linears, BN affines free — lejepa's C1/f2 head convention transplanted) | C4 | dampens (slower rise), unlikely full hold alone |
| `embdiag_wd0` | `+method.emb_wd0=true` (emb Linear excluded from weight decay) | C2 | if aniso = wd's balanced-factorization imprint on the gauge freedom, this holds f2-like; my joint pick with d128 |

Dose rung (added same evening, pre-registered before any diag numbers): the ep19/ep25 pulls
put arm-4's realized h-floor share at 0.129% flat vs f2's 6.2% @ep25 (48×, g_enc caveat
standing) — no cell tested dose alone. | `embdiag_lam5x` | `method.h_lamb=0.1` | C1 (dose,
5×) | divergence delayed / crest lowered, not held | · | `embdiag_lam50x` | `method.h_lamb=1.0`
(≈ matches f2's realized share; E12 caveat: ~destination weights scrub class structure — probe
cost acceptable in a diagnostic) | C1 (dose, ~50×) | if THIS holds while batch-1 cells
diverge, the issue is the dose RULE (nominal λ ≠ realized share across lanes), not lane
structure |. Divergence-epoch vs dose = the read.

Caveats declared: w_floor=19.10 verbatim in d128 (realized z-pull changes with dim — shape
and dose move together in that cell; acceptable for a divergence diagnostic, flagged for any
quality claim). spec/wd0 need vicreg.py additions — absent-key paths byte-identical
(running-chain resume safety), CPU dry-run before launch, post-launch activation verified
(spec: parametrized state-dict keys + arch kwargs; wd0: guarded log line), NFS-staleness
wait between edit and submit (e19embcal incident).

**Standing-watch answered en route (reader validation, wandb through ep75.7 of arm 4):**
arm-4 h_kl@emb CRESTS ~2.02 @ep12 then declines ~monotonically to ~1.09 by ep75.7 — the
gv2/hz crest-then-slow-fall pattern at ~2× amplitude, NOT a permanent plateau (the earlier
"plateau ~1.95" was the ep10–20 window). Still ≳5× f2 at matched epochs (f2: .34@ep12,
.19@ep100). f2 template exact marks: e0.5 1.11 · e1 1.05 · e2 .84 · e3 .66 · e5 .49 —
monotone from ~ep0.5, no divergence ever; arm 5 bottoms .67@ep0.14, diverges @ep0.36,
crosses f2's curve UPWARD ~ep1.3. Divergence signature = slope sign after ~ep0.5, not level.

**Ladder ep2 preliminary (RAW; reader `experiments/e19_diag_curves.py` →
`results/figures/e19/e19_diag_curves.png`; EMA .9; epoch-aligned):**

| cell | bottom@ep | diverge@ep | e0.5 | e1 | e2 | vs |
|---|---|---|---|---|---|---|
| arm 5 (failure trace) | .672@.14 | .36 | .94 | 1.33 | 1.64 | — |
| f2 (hold template) | — | never | 1.11 | 1.05 | .84 | falls from above |
| lr3 | .588@.27 | **.87** | .62 | .79 | 1.25 | delayed ≈ lr-ratio, same climb |
| d128 | .700@.14 | .34 | 1.04 | 1.56 | 1.94 | FASTER than arm 5 |
| spec | .684@.13 | .41 | .92 | 1.31 | 1.64 | ≈ arm 5 exactly |
| wd0 | .701@.12 | .43 | .93 | 1.38 | 1.60 | ≈ arm 5 exactly |
| lam5x (ep1.3) | .657@.17 | .48 | .80 | 1.09 | — | dampened, diverging |
| lam50x (ep1.3) | .520@.27 | **none** | .53 | .62 | — | below f2's matched-ep curve, ~flat |

**ep5 confirmation read (2026-07-17c ~21:15):** lam50x running-minimum is CURRENT
(.441@ep5.03): e1 .62 · e2 .62 · e3 .55 · e5 .45 — monotone fall, no divergence ever, BELOW
f2's matched-epoch curve (f2 e5 = .49), same shape. Probe pace inside the envelope the whole
way (ep3 .1366 vs arm-5 .1364 / arm-4 .1446; ep5 .1800 vs arm-4 .1866). lam5x: crest 1.24@ep2
→ 1.13@ep5 falling. Failed cells at e5 all at the uncal twin's level: lr3 1.64 · d128 1.69 ·
spec 1.69 · wd0 1.59 (divergences certified — events happened, no extrapolation). Dose-graded
on BOTH axes: share .13% → crest 2.02@ep12 · ~.7% → 1.24@ep2 · ~6% → none. Realized-share
arithmetic closes: at h_lamb=1.0, share = 1.0×.217/(3.35+.217) ≈ 6.1% ≈ f2's 6.2% @ep25.
Open certs in flight: continuation jobs 62407857/58 extend both lam cells to ~ep14–16
(late-crest exclusion; crest-time shrinks with dose, so a 50× late crest would invert the
observed relation); anatomy decomposition of both lam snapshots (job 62407934, rows append to
e19_floor_anatomy.csv) checks the hold is held-aniso at (0,I), not a traded channel.

**CERTIFICATION READS (2026-07-18 ~00:2x; both residuals closed):** (1) late crest EXCLUDED —
lam50x minimum still current @ep12.3 (.343): e5 .45 · e8 .41 · e12 .36, ON f2's own curve
(f2: .49/.41/.34 at the same marks) while the λ=.02 twin crested 2.01@ep12. lam5x continues
crest-decline (1.03@ep12). Failed cells to ep13: lr3 1.36 (delayed crest-decline) · d128 1.74
· spec 1.63 · wd0 1.64. (2) held-not-traded — anatomy @ep5 (job 62407934, csv rows): lam50x
emb = per-dim var .9986 · cone .0293 · scale .00005 · aniso .4075 (kl .437, falling) — a
genuine (0,I)+small-aniso hold on every channel; lam5x emb residual = pure aniso (1.055) at
var 1.195 — decomposition-level dose grading. REACH-BACK BONUS DATUM: lam50x CLS (one affine
upstream, untapped) kl .772 / rand .450 vs arm-4 CLS 1.798/.675 @ep10 — the f2 conduit
mechanism REPRODUCES in vicreg once the floor wins the tap; the D-043 conduit question was
dose-gated, not lane-gated. Goal resolution stands: issue = nominal-λ transplant (realized
share 48× under f2's); fix = realized-share dosing (λ* = λ_ref × share_ref/share_lane ⇒
h_lamb ≈ 1.0 this lane). AGREED-TAKEAWAY wording awaits Berker.
Mechanical direction check at this horizon: dose-response graded (div@.36 → .48 → none as
h_lamb ×1 → ×5 → ×50); all structural single-knob cells (spec/wd0/d128) inside the failure
envelope; lr rescales the divergence clock (~×2.4 ≈ lr ratio 3.3) without changing the path.
Confirmation horizon: lam50x to ep5–7 (late re-inflation excluded only then; uncal twin
crested at ep12 — cal cells' divergences all fired by ep0.5, so ep5+ hold is ~10× past the
onset window). NO takeaway before that read + Berker.

Ladder execution (2026-07-17c): vicreg.py additions landed (spec_norm kwarg on
vicreg_expander mirroring lejepa_projector, arch-recorded only when set; emb_wd0 param-group
split with guarded log line). CPU dry-run PASSED all three paths: default arch + state-dict
keys byte-stable vs the RUNNING hz_emb _last ckpt; spec builds 9 parametrization keys +
steps/backwards; wd0 groups cover exactly embed (147,840 params) with union complete.
Measured RNG note: torch spectral_norm DOES draw u/v at construction ⇒ the spec cell's emb
Linear init shifts vs arm 5 (trunk/proj unshifted — created before the wrap); acceptable
here: the phenomenon is init-insensitive (arms 4/5 fail from different inits) and emb_calib
re-bases per-dim. LAUNCHED: `e19-dg-lr3` 62407658 · `e19-dg-d128` 62407659 · `e19-dg-spec`
62407670 · `e19-dg-wd0` 62407671 (each a single 2.5 h gpu job, ~ep25 horizon).

Execution: byte-invariance for running chains CPU-verified before edit landed (arch +
state-dict keys stable; emb path grad-checked). Init pull record (job 62396357, tag e19emb):
g_inv .4428 · g_moment_kl@z 3.097 · g_h_moment_kl@emb-out 5.149 (loss 1.62 — the uncalibrated
Linear starts off-target) ⇒ h-floor init share 0.146% of lane total (hz's cls-tap: 0.21%).
First smoke pair failed on launch error (`num_classes=100` omitted — toy default 10 → probe
nll assert; NOT an arm issue); corrected smokes 62396460/61 PASSED: ep1/ep2 .0632/.1276 (emb)
· .0560/.1272 (hz_emb), no incidents, on the arms-1/2 pace. Chains LAUNCHED 2026-07-17b:
`e19-flr-emb` 62396576/77/78 · `e19-flr-hz-emb` 62396579/80/81 (3×8h singleton links, gpu).

## Numbers land below this line as they arrive; AGREED TAKEAWAY only after joint discussion.

### 2026-07-18 — ep100 first scoring pass, arms 1–3 (RAW + mechanical direction check; NO takeaway)

Arms landed ep100 2026-07-17/18 (floorssl 21:31 ·  hz 22:25 wall-file times reversed — see
ckpts; emb 23:54; hz_emb 00:4x, probes pending). Extractions `.ext` (adapter=native, jobs
62408200–02, 62408299) → v2 probe family + audit battery (62408301–06, 62408316/17).
Headline pair at DECLARED h (student.h.cls, D-036), linear_raw_v2 / knn_v1_k200:

| run | lin_v2 | knn200 | Δlin | Δknn |
|---|---|---|---|---|
| e17c control (shipped var+cov) | .6452 | .5506 | — | — |
| **arm 1 floorssl** (z=floor SOLE) | .6244 | .5212 | **−2.1** | **−2.9** |
| **arm 2 floorssl_hz** (+ h-floor@cls λ=.02) | .6246 | .5526 | −2.1 | +0.2 |
| **arm 3 floorssl_emb** (+ free on-path Linear) | .6156 | .5020 | −3.0 | −4.9 |
| **arm 4 floorssl_hz_emb** (+ h-floor@emb λ=.02) | .6224 | .5332 | −2.3 | −1.7 |
| e12gv2 (shipped z + h-floor@cls) | .6508 | .5756 | +0.6 | +2.5 |
| c015 (own var+cov 6× at h, closure) | .6508 | .5942 | +0.6 | +4.4 |

Mechanical direction check vs the locked predictions: **P1 MISSED** — floorssl reads BELOW
the control on both headline probes (prediction was ≥/within-noise). **P2 HELD** (battery,
results/battery/*.csv): cone@z dead (gauss_kl_full.location@z = 0.000; anatomy .00015);
effrank(z) 454–480 vs control 442–450 (mildly above), rankme(z) 875–905 vs 562–567
(strongly above). h.cls geometry ≈ control (effrank 70–75 vs 67–70, rankme 143–145 vs
149–150 — flat); gauss_kl_full@h.cls (D-040, new): floorssl 2.396 = location .066 +
spectrum 2.330 (control rows absent — its battery predates ISO-ladder v2; re-audit of e17c
= optional backfill). P3 kill never approached.
Secondary raw rows: z-space probes floorssl proj.out .5626/.5340 vs control z .5884/.5622
(z also slightly below); tap1 best-probing space in-arm (.6116, E01-T6 pattern again);
h.gap knn floorssl .3884 vs control (.394 E17 record) ≈ flat. Arm-3's CLS reads worst of the
four (−3.0/−4.9): the UNfloored free Linear costs at the trunk readout too (its emb-out
anatomy @ep12: cone 3.19/aniso 2.42 — raw pairing, no causal claim). Arm 4 sits between
arm 3 and arm 2 (+3.1 knn over the free Linear, −1.9 under the CLS-floored twin) — raw
ordering consistent with the ladder's partial-win-at-0.13%-share record; the corrected-dose
cell (h_lamb=1.0, held tap + cleaner CLS at ep5) exists only to ep12 (promotion = Berker's
call). Online-monitor-vs-
offline note (E12-T9 watch): floorssl best-monitor .6246 vs offline lin .6244 — bias ≈ 0 in
this lane at ep100. hz_emb probes + all four batteries land next; per-epoch trajectory pulls
for arms 1–2 at cadence submitted (jobs recorded in e12h_pull.csv tags e19tp.floorssl.ep* /
e19tp.hz.ep*).

### 2026-07-17b — mid-training curve record + the commensurability instrument (RAW; chains at ~ep20)

Figures (read in order): `results/figures/e19/e19_curves_raw.png` ·
`results/figures/e19/e19_hkl_question.png`. Instrument:
`experiments/e19_floor_anatomy.py` → `results/diag/e19_floor_anatomy.csv` (KL decomposition
cone = ½‖μ_Q‖²/d′ · scale = ½(m−1−ln m), m = trΣ_Q/d′ · aniso = ½·logdet Jensen gap — exact,
dimensionless, cross-tap; plus R = within/total variance at each tap, pos/rand, per-dim var;
fixed seed-0 batches, n=512 at every tap in all three lanes, train-mode parity with the logged
curves). Companion: `experiments/e19_curves.py` (wandb pulls + figures);
`experiments/e12h_pull.py` gained the card-mandated `+ckpt=` trajectory extension (needs
`frame=in100_vits16` explicitly — default frame is toy).

Validation: recomputed kl reproduces every live curve at matched steps — f2 .189 (curve .19),
gv2 1.000 (≈1.02), hz@ep20 1.371 (≈1.4), z-floor .427/.423 (≈.43).

Raw observations on record (mid-training for the two E19 arms — no converged claims):

1. **The parked h_kl trajectory question.** hz's h_moment_kl rise (1.25→1.39, steps ~1500→5900)
   sits on the SAME rise→crest(~ep10-15)→slow-fall hump gv2's full curve shows (gv2: 1.29@ep25
   → 1.00@ep100); hz tracks gv2's shape/level at matched epochs. Decomposition: cone is killed
   in the first epochs everywhere the floor acts (hz cone .006 by ep20); the CLS value =
   frozen scale (m≈.15–.19, scale-part .44–.54, per-dim CLS var never approaches 1 at λ=.02)
   + an aniso hump (gv2 aniso .82@ep25→.46@ep100 carries ALL the net decline). f2's fall
   (.84→.19) is the same functional holding scale ≈ 0 THROUGHOUT (calibrated affine embed:
   mbar .92–.93, scale-part .002) — its converged value is ~98% aniso residual (.185).
2. **Unfloored counterfactuals (anatomy monitor).** vicreg ctrl @cls: kl 2.28@ep25 → 1.64@ep100
   (cone .55→.065 — the cone shrinks on its own late, T5 pattern; scale .78→1.01, var .083→.051;
   aniso .94→.56). lejepa ctrl @cls ep100: kl 2.83 (var .0076, rand .757). gv2-vs-ctrl at cls
   ep100: cone 100× lower, per-dim var 3× higher (.146 vs .051), aniso mildly lower — the ε-dose
   CLS floor is not passive, but never wins scale.
3. **Reach-back asymmetry.** f2's floor@embed cleans CLS one affine upstream (rand .039 vs ctrl
   .757; var .543 vs .0076; kl .302 vs 2.834). floorssl's floor@z cleans z (cone .0002,
   rand@z −.002, better than ctrl's z) but transmits NOTHING back through the BN expander to
   CLS (cone .714 / rand .944 / var .089 ≈ ctrl@ep25). The h-floor cell is what removes the
   h-cone (hz cls rand .069 @ep20).
4. **Loss-space alignment, commensurably (R = within/total var).** lejepa proj: .151@ep25 →
   .072@ep100 (f2 lane ≈ ctrl lane). vicreg z ctrl: .314@ep25 → .191@ep100; floorssl .410@ep18,
   hz .396@ep20 (falling). Raw vicreg inv is near-flat across all vicreg arms (~.26 at ep20;
   ctrl .28→.22 over 100 ep) — non-diagnostic without the variance normalizer.
5. **Health.** Both E19 arms: grad_norm rides the vicreg-family envelope (~2.2 at ep20, no
   incidents); online probe @cls tracks the two controls' early trajectory (+1–2 pts at
   ep16–20); P3 kill criteria not approached. z-floor value still falling at ep18–20.

Dose-spread record (init pulls, e12h_pull ledger, per-lane inv-pull convention):
floorssl floor@z 52.18 weighted = 502% of lane inv-pull (83% of lane total) · f2 floor@embed
.0872 = 2.2% (0.67% of total; card record) · gv2/hz floor@cls .131 = 1.26% (0.21% of total).

**Emb-arm mid-training pulls (2026-07-17c, jobs 62397108/09; e19tp.emb.last @ep22 ·
e19tp.hz_emb.last @ep19, pullsnap copies of _last).** Raw g_enc (unweighted): arm 3 — inv
.0925 · moment_kl@z .1294; arm 4 — inv .0708 · moment_kl@z .0825 · h_moment_kl@emb .2166
(loss 1.868 ≈ the live plateau). Weighted at-tap totals: arm 4 = 25·inv 1.771 + 19.1·z-floor
1.576 + .02·h-floor .00433 ⇒ h-floor share **0.129%** of lane total (twin hz @ep20: 0.20%;
f2 @ep25: **6.2%**). The z-side adversary DECAYED from init in this lane too (70.2 → 3.35
weighted, ×21) but stabilizes ~25× above f2's (.132 @ep25, ×98 down from init, flat after).
z-state across cells byte-similar (z-floor loss .423/.421/.426 in hz_emb/hz/floorssl; inv
.262/.239/.242) — the on-path Linear changed ~nothing at z while its own output went aniso.
**INSTRUMENT CAVEAT (flagged 07-17c): g_enc = trunk-module-only.** For vicreg the emb Linear
is a separate module and its OWN parameter gradients are excluded; lejepa's "encoder" includes
its embed Linear. Within-lane shares are consistent; the cross-lane f2-vs-emb-arm share
comparison is biased against the vicreg arms for the h-term specifically (the floor sits
directly on that Linear). Fix candidate: an additional g_enc_full column (trunk+emb) — not
retrofitted mid-ledger without a decision.

**Trajectory pulls LANDED** (e19tp.* rows in `results/diag/e12h_pull.csv`, jobs 62396091–100;
figure `results/figures/e19/e19_pull_traj.png`). Floor share of lane total weighted pull:
f2 0.67%(init) → 6.2%(ep25) → 2.4%(ep100) — RISES ×9 then settles ×3.5 above init (lejepa's
z-side g collapses after init: g_sigreg 454.7→1.68 by ep25). gv2 0.21% → 0.09%(ep25) →
0.036%(ep100) — FALLS ×6 (vicreg's z-side g grows through training — the late grad_norm rise —
while the h-floor's g shrinks as the cone dies). floorssl floor@z 83%(init) → 48%(ep18) —
the destination dose decays toward parity with inv as the floor approaches satisfaction.
hz h-floor 0.21% → 0.20%(ep20) — flat so far. Same-name floors at mid-training span ~2000×
in realized share (.48 vs .0004). Third cross-validation: pull-row loss_values reproduce the
live curves (f2 h_kl .269/.225/.202/.192 at ep25/50/75/100; gv2 1.283/1.100/1.028/1.007).

## AGREED TAKEAWAY (rows in DECISIONS)

**E19-T1 (USER-APPROVED 2026-07-19):** the dose law — nominal-λ transplants invalid; effect
tracks realized share (λ* = λ_ref·share_ref/share_lane; T=6%, ĝ_h=.30); flat channels won at
any dose, aniso ∝ share; lr/head-dim/spec_norm/wd/init/topology eliminated by direct cells;
the D-043 conduit is real but DOSE-GATED. **E19-T2 (USER-APPROVED):** the standalone failed
AS INSTANTIATED (both doses pre-law); ruled failed-but-retrying; target = beat dino/lejepa
variants with inv@z + floor@z + floor@h; → arm 6 below.

## Arm 6 — floorssl_hz_v2 (the dose-corrected standalone; Berker 2026-07-19: "launch the hz
## standalone now")

Arm 2 with BOTH doses corrected by the certified rules, everything else byte-identical:
**w_floor = 45** (z-duty matching: shipped var+cov runs 2.19× inv's pull at formation
(e19tp.gv2.ep25) vs the old z-floor's 0.93× (e19tp.floorssl) ⇒ 19.1×2.35) · **h_lamb = 0.65**
(share rule on the hz arm's own held-state pulls: .0638×3.414/.336). INTERACTION NOTE
(declared): raising w_floor raises the lane total ⇒ λ_h=.65 realizes ≈3.4% share under the
new A (vs 6% under the old) — inside the certified hold basin (f2 held to 2.4%);
self-consistent 6% would be λ_h≈1.2 — the approved spec (.65) launches, the alternative
recorded. Pre-registered directions: P-v2-A h-tap holds f2-like (share ≥3%) AND z-scale
equilibrium rises toward 1 (w×2.35) · P-v2-B probes: beat arms 1–2 at h; the bar Berker set
= dino/lejepa-variant territory (lejepa e20f .6700/.6288 = the zoo top) · kill criteria as
arms 1–2. Smoke → 3×8h gpu chain.

Arm-6 execution: smoke 62420599 PASSED (ep1/2 .0420/.1022 — below arm-2's smoke pace
.056/.127 as expected at the heavier doses, no incidents). Chain `e19-hz-v2` LAUNCHED
2026-07-19 (3×8h singleton, gpu). First read = divergence-window h_kl (does λ_h=.65 hold
CLS f2-like at ~3.4% share) + z-scale equilibrium (does w=45 lift per-dim z-var from .39).
Arm-6 chain MOVED to gpu100/H100 (Berker 2026-07-19: "it should be better if it is gpu100
partition"): gpu links scancelled at ~ep8, resubmitted as h100-slotA 3×8h singleton; resumes
from _last — no training lost.
