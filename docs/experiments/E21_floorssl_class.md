# E21 — FloorSSL as its own method + the BN-free head fork

**Status: PRE-REGISTERED 2026-07-19 (Berker: "an independent class for our method and we
dont rely on vicreg's architecture. i want to see both versions, one copying vicreg part
(expander BN etc) and the other should try the bn free way"). D-row: D-046 (PROPOSED).**

## Question

Two-in-one: (1) the method graduates to its own class (`sslgap/methods/floorssl.py`:
L = 25·MSE(z) + 45·MomentFloor(z pooled) + 0.65·MomentFloor(cls) — the E19-T1/T2 dose
recipe); (2) the head fork — **does removing BN (the measured moment firewall, E19-T1) turn
the head into a full-depth conduit so the destination-dose z-floor conducts conditioning
back into the trunk — and does the method survive without BN's stabilization?** The design
bet from our own record: the z-floor's unit-variance demand replaces BN's scale pin AT z
(the pin whose absence caused lejepa's measured lazy-head minimum); hidden layers keep no
pin — declared risk.

## Arms

| arm | head | status |
|---|---|---|
| **bn** | Linear-BN-ReLU ×2 + Linear (vicreg's expander verbatim) | ≡ the RUNNING `in100.vicreg.s0.floorssl_hz_v2` — NO retrain: class migration verified byte-identical (166 tensors, seed-0 init + pinned-RNG step math; dry-run record below) |
| **nobn** | Linear-ReLU ×2 + Linear | `in100.floorssl.s0.nobn` — NEW cell; head Linears byte-match the bn arm's (BN inits deterministic ⇒ same draw stream) |

Doses VERBATIM across arms (w_floor=45, h_lamb=.65 — the adversary structure shifts without
BN; realized pulls measured at cadence instead of re-dosing, comparability convention). Aug
pipeline stays the BYOL pair; trunk/frame/seed byte-shared. Probe/eval at CLS (D-036).

## Pre-registered directions (Claude's pick: A-partial + survivable)

- **P-bn-A (conduit-on):** nobn's CLS receives z-floor conditioning the bn arm's cannot get
  through the firewall — anatomy read at matched epochs: cls kl/rand/aniso BELOW the bn
  arm's at the SAME λ_h (the h-dose is matched, so any extra CLS conditioning is conducted).
- **P-bn-B (stability price):** hidden-scale drift without BN's pin (watch: grad_norm
  envelope, z-std, per-term curves; K1 incident rule standing; kill = probe ≤2× chance
  @ep≥10). A controlled failure here is itself the firewall-necessity datum.
- **P-bn-C (null):** BN was not binding at these doses — arms indistinguishable at h.

## Discipline

Migration byte-check PASSED before launch (scratchpad dry-run: (1) bn-arm init ≡ vicreg-class
v2, 166 tensors; (2) pinned-RNG training_step terms identical; (3) nobn Linears byte-match +
backward clean). 2-ep smoke → 3×8h singleton chain (gpu; H100 slots carry v2 + dino-fix).
`num_classes=100`. Numbers raw; NO takeaway without Berker. Landing: extract → v2 probes →
battery → anatomy (the conduit read) → cadence pulls (e21tp.*).

## Numbers land below this line as they arrive; AGREED TAKEAWAY only after joint discussion.

## AGREED TAKEAWAY (2026-07-20 session wrap — PARTIAL; mirrored to DECISIONS E21-T1)

**ADOPTED (Berker: "yes"; wording under the delegated trust): the sigreg-configuration
reading.** At every healthy dim 16–512 the loss-terminal z is essentially Σ=I-compliant
(KL/dim ≤ .08, no dead dims) while content parks at h (z-probes .04–.20 below h.cls; h
effrank 184–231 vs both ctrls' 27/67; ep25 R@z monotone in dim) — the method realizes
sigreg's division of labor with its own loss at both places.

**Working position, NOT a takeaway (Berker: "dimension 256 looks good but we are not
married"):** d256 = the current operating point; naming deferred.

**HELD, with triggers (Berker's wrap answers):**
- D* ∈ (512, 2048] boundary → re-examined via view-mean cells at d512/d2048 AFTER d256vm2
  reads out ("we should probably try 512 and 2048 after the change of view means variant,
  but hold on this for now" — cells parked).
- Collapse-mechanism wording → after d256vm2 ("we are not sure competition comes from where
  but i was thinking our slices creating a lot of randomness"). His conjecture is testable
  independently of vm2: the K-draw decomposition — g_z over K fresh slices at a held state;
  mean pairwise cos among draws = the slice-jitter share, cos(averaged g_z, g_inv) = the
  systematic opposition — separates slice randomness from objective-level competition.
  DESIGNED, PARKED on his word.
- Beats-the-bars claim → after d256e200 ("we are running 200 epochs to see the convergence
  run").

**Approvals recorded same wrap:** D-046..050 + D-051 (incl. the 38.7 re-dose) + D-052
USER-APPROVED ("as we run stuff you should have my decisions"); E18-T1 · E19-T1 · E19-T2 ·
E20-T1..T3 vetos RELEASED, wording FINAL ("for wording stuff i trust you"); E20-T1 completed
with the five-point dose curve; E20-T3 absorbed the audit-frame pair correction.

## Arm 3 — lejepa augs (Berker 2026-07-19: "lejepa's augmentations are better for our hz run
## compared to vicreg's")

`in100.floorssl.s0.lejepa_augs`: bn head + the lejepa V=4 symmetric strong-photometric
family (ViewsDataset default stack — the M1-portval-verified official aug), bs=128 ⇒ the
floor's pooled n stays 512. AXIS DISCIPLINE: aug changes on the bn head only (nobn keeps
byol pair) — head and aug effects stay separable; combine if both win. inv generalized to
all-pairs mean MSE (reduces EXACTLY to the lineage's pairwise MSE at V=2 — byte-check
re-passed after the change). Doses verbatim (25/45/.65; adversary shifts under stronger
views — cadence pulls, not re-dosing). Grounds: the floor's two biggest E20 wins live in
this aug family (lejepa +6.6/+10.0, and f2 before it); our method owns its recipe (D-004
binds the audit's per-method fidelity, not FloorSSL). Pre-registered: P-aug-A the aug family
transfers its advantage (h probes above the byol-pair v2 arm at matched epochs) · P-aug-B
family-specific (lejepa's family helps lejepa's loss, not ours — v2 ≥ this arm) · health:
V=4 quadruples per-step trunk passes — pace watched, kill criteria standing. 2-ep smoke →
3×8h chain (gpu).

REDIRECTION (Berker 2026-07-19: "this hz arm that is running, cancel it and rerun with the
new augs"): the v2 chain (byol-pair, vicreg-class) CANCELLED at ep6 — **arm 3 (lejepa_augs)
is the primary hz run**, chain on h100-slotA when its smoke passes. Comparability note: the
stated bar (beat dino/lejepa variants — lejepa e20f .6700/.6288) is now AUG-MATCHED, which
is the fairer frame for that bar; the byol-frame reference (vs e17c/floorssl_hz) is
retired with v2. nobn keeps the byol pair as launched — if Berker wants the head axis clean
inside the new aug frame, nobn moves to lejepa augs too (one word).
nobn MOVED to lejepa augs too (Berker 2026-07-19: "nobn is also with lejepa augs right? if
not make correct it"): byol-pair chain cancelled ~ep2; relaunch = head_norm=none + V=4
lejepa family (fresh run under the SAME tag `nobn` — the byol-pair steps are superseded
from the smoke2-validated relaunch). Head axis now clean inside the lejepa-aug frame:
{bn, nobn} × lejepa_augs; the byol frame is fully retired from E21.

## Launch + early-window record (2026-07-19; RAW)

Smokes PASSED (laug ep2 probe .0714 · nobn.smoke2 ep2 .0956 — both climbing, no incident;
BN-free head trained 2 clean epochs). Chains: laug = h100-slotA 62420881→882→883 · nobn =
gpu e21-nobn 62420884→885→886, both gated `afterok:<smoke>,singleton` (SLURM-native — the
previous session's watchers died with it; ping-only, no duplicate launches). Reader:
`experiments/e21_curves.py` → results/figures/e21/e21_curves.png (h_kl vs f2/lejepa-e20f
template · z-floor+inv terms with the v2 ep6 stub · grad_norm envelope · dino dose panel).

Early window (~ep2, 11:50): h_moment_kl laug .52 (bottom .44@0.6) · nobn .33 (.28@0.9) —
both inside the template band, no divergence flag (rule: sustained > min+.15); z-floor
moment_kl ~.50 both and falling; inv ~.53/.53 (head-axis pair tracking each other); nobn
grad_norm med 64 / max 438 vs laug 47 / 231 — elevated absolute scale (weights 25/45), no
BN pin showing as ~1.9× the bn arm's envelope, max/med ≪ the 100× kill rule. K1 NOT fired.
z-scale equilibrium read queued (anatomy argv on `_last`, job 62421070: does w=45 hold
per-dim z-var above v2's .39?).

PACE (corrected 11:24 — the first flag was a wall-clock arithmetic error, retracted): log
timestamps give laug ep3 in 32 min ≈ 10–11 min/ep (H100, V=4) · nobn ~12 · dino cells ~11
⇒ 100 ep ≈ 18–20 h — ALL four cells fit their 3×8h chains with margin; no extension needed.
Monitor at matched epochs: nobn AHEAD of laug (ep2 .0956 vs .0712; laug ep3 .0868) — raw.

z-scale first point (anatomy on `_last`, job 62421070; ep2/ep1 states — EARLY, equilibrium
read repeats at ep25 cadence): per-dim z-var laug .287 / nobn .265 (v2-era reference .39 was
the w=19 equilibrium; too early to score w=45 against it). cls side: nobn per-dim var
1.0001 with scale-part 2e-05 at ep1 — the BN-free head's cls-floor pinned scale IMMEDIATELY
(bn arm .90 at ep2); nobn z kl .63 (aniso .31) vs laug .51 (aniso .15) at mismatched-epoch
states. RAW.

## COLLAPSE + fix (Berker 2026-07-19 ~12:00: "floorssl runs collapsed. fix them.")

**Kill confirmed (pre-registered criterion met):** monitor laug .0388→.0712→.0868@ep3 →
.0322→.0216@ep5 and falling (2× chance = .02; per-step probe loss ≈4.34 vs ln100=4.61 at
ep13) · nobn .0406→.0956@ep2 → .0400→.0184@ep4. BOTH arms, SAME onset (ep3→4) ⇒ not the
head axis (K1/BN-free instability acquitted: grad_norm envelopes smooth-decaying, no
incident, no NaN — P-bn-B's failure mode is NOT what happened).

**Forensics (per-step window, both cells):** a SMOOTH equilibrium, no divergence event —
inv RISES monotonically (.23→.54 by ep5 and keeps rising = alignment sacrificed), z-floor
stalls at .44–.50, h-floor loses ground (nobn .28→.41), grad_norm decays normally. The
information-starved signature (E15/E16 class), not an instability. Decisive control: **v2
(SAME 25/45/.65 doses, byol pair) was healthy** — monitor .042→.189@ep6 climbing, inv
peaked .467@ep3 then FELL. Same doses, different aug family, opposite equilibrium ⇒ the
card's verbatim-dose bet across aug frames FAILED = E19-T1's certified error class
(nominal-weight transplant across frames; the "adversary shifts under stronger views —
cadence pulls, not re-dosing" convention is retired for cross-FRAME moves).

**Fix derivation (the certified route, not weight-tweaking):** init pulls cannot carry it —
measured init term values are near-identical across frames (inv .234 vs .224: a random
trunk cannot see the aug family; same reason the E20 law measured at ep25 FORMATION
states). Frame-bridge instead (`experiments/e21_pull.py`, job 62421210): hold the healthy
formation state fixed (v2_best ep6; loadable into the floorssl class by the migration
byte-proof) and swap ONLY the aug pipeline — per-term trunk-g under byol pair vs lejepa
V=4. Equal-pull re-dose per term: **w′_t = w_t · g_t(v2_best, byol) / g_t(v2_best,
lejepa)** — every term keeps its healthy-lineage realized pull under the new frame. Also
measured: init both frames (the null) + laug_best ep3 (trajectory record). Chains
CANCELLED (62420881-6); collapsed ckpts KEPT until the fix lands (diagnostic states; _best
= the pull's bridge input). Relaunch = new tags (collapsed cells retired), 2-ep smokes →
chains, collapse-window (ep3–6) watched via e21_curves. If the re-dosed cells collapse
again, the loss FORM (MSE + floors at these ratios) is refuted under strong-view families
— that would be a design datum, not a dose datum.

**Frame-bridge pull LANDED (e21_pull.csv, job 62421210) — the mechanism is now measured:**
at v2_best (ep6 healthy state), swapping ONLY the aug pipeline: g_inv .2166 (byol) → .1650
(lejepa V=4) = **−24%**, while g_z .2226→.2288 (−3% eq.) and g_h .6732→.7096 (+5%) are
frame-invariant — verbatim doses silently cut inv's realized share 34.1%→27.7% at the
formation state; the equilibrium then sacrificed alignment (the observed collapse mode).
Corroborations: (i) trajectory row (laug_best ep3): floor pulls GREW along the collapsing
run (g_z .334, g_h .994 vs v2-state .229/.710) while g_inv stayed .166 — the floors'
dominance amplified as content leaked; (ii) the init-null concretely confirmed: init-state
ratios would have prescribed w_floor 45→62.7 (the WRONG direction; init cannot see the aug
family — inv g ratio at init 1.10 vs 1.31 at formation).

**Re-dose (equal-pull at the healthy formation state):** w_inv 25→**32.8** (×1.313) ·
w_floor 45→**43.8** (×0.973) · h_lamb .65→**0.617** (×0.949). RELAUNCHED as new cells
`in100.floorssl.s0.lejepa_augs2` (bn) + `in100.floorssl.s0.nobn2` (head_norm=none), same
frame/seed/bs, 2-ep smokes 62421226/27 → chains gated afterok+singleton: laug2 =
h100-slotA 62421228→229→230 · nobn2 = gpu e21-nobn 62421231→232→233. Collapse-window
(ep3–6) watched via e21_curves; pre-registered falsification stands: a second collapse at
corrected doses refutes the loss FORM under strong-view families (dose ruled out by the
certified correction). Late note for the record: the nobn v1 tail (post-collapse, ~ep5+)
did eventually spike (grad max 7728 vs med 48, h_kl→1.03) — the BN-free instability
appeared DOWNSTREAM of the shared dose failure, not as its cause; K1's ordering matters.

**Commensurability read (Berker: "inv still doesnt look good, but i dont know how comparable
the curves are to f2") — R = within-instance/total variance at each cell's loss space (the
anatomy instrument's lane-free inv readout; raw inv VALUES are not cross-lane comparable —
different dims/weights/normalization):**

| cell | space | ep | R | pos-cos | per-dim var |
|---|---|---|---|---|---|
| f2 | proj (its aligned space) | 25 | **.151** | .826 | .83 |
| lejepa ctrl | proj | 25 | .163 | .813 | .80 |
| f2 | embed (its FLOOR space) | 25 | .561 | .468 | .92 |
| v2_best (healthy, byol pair) | z | 6 | **.615** | .570 | .35 |
| laug v1 (collapsed) | z | 2 | .944 | .405 | .29 |
| **laug2 (re-dosed)** | z | 2 | **.920** | .421 | .24 |
| **nobn2 (re-dosed)** | z | 2 | .903 | .147 | .24 |

Structural datum: lejepa/f2 SPLIT the duties across spaces (floored embed R .56 · aligned
proj R .15); floorssl presses floor+alignment on ONE z — and even the HEALTHY v2 runs at
R@z .61, four times f2's aligned-space value. The eye-read "inv doesn't look good" is real
on the commensurable scale and PREDATES the collapse (v2 carries it while climbing).
Re-dosed cells at ep2: inv value bending (−17% vs v1 at matched ep4, flattening in v2's
plateau shape; z_kl/h_kl slightly higher = laxer floors, consistent) but R@z only
marginally below v1 (.92 vs .94) and monitors identical through ep2 (.0752/.0980) — the
ep3–5 monitor fork is the verdict (watcher armed). nobn2 raw notes: z pos .147 (much less
aligned than laug2's .421) while its cls kl .386 is the best of any floorssl cell; cls
per-dim var 1.26 = variance OVERSHOOT without BN's pin (v1 pinned at 1.00 — the pin is
dose-sensitive). RAW.

## ep3 fork + the ε-structure arm (D-048; Berker: fix now, don't waste compute)

Fork at ep3 (the v1 collapse epoch): **laug2 .0426→.0752→.0944 — above v1's peak and
rising (KEPT, earns time; ep4–5 verdict pending)** · **nobn2 .0432→.0980→.0536 — falling,
v1's pattern (chain CANCELLED at ep3)**. Under corrected doses the head axis
differentiates: BN's stabilization is doing work the dose correction can't replace.

`in100.floorssl.s0.laug_eps` (NEW; = the deferred Z5 ε-dose grid cell, now derived): the
commensurability table says floorssl's aligned space is majority-owned by its floor (65%
share at z; even healthy v2 runs R@z .61 vs f2's aligned .15). The ε-arm inverts ownership:
inv owns z (91% share), z-floor demoted to the certified 6% conditioner share (T, E19-T1),
h-floor kept (.617, 2.8%), total trunk pull preserved at the bridge state (15.87) ⇒
**w_inv 87.8 · w_floor 4.16 · h_lamb .617**. The floor's −ln(var) barrier still hard-blocks
scale collapse; v1-scaling predicts z-var equilibrium ~.1 (sqrt(F/W) law: v1 sqrt(19/25)→.39
observed). BN head — the dose-STRUCTURE axis stays clean vs laug2. Smoke 62421496 → gpu
chain 62421497→498→499 (`e21-laug-eps`). Pre-registered: P-eps-A the inv-owned z aligns
(R@z falls toward f2's aligned band, monitor healthy through ep6) · P-eps-B scale-starved z
(z-var → ≪.1, cls probe suffers despite own floor) · P-eps-C same collapse anyway (the
strong-view family defeats the form regardless of ownership — the deepest read). Claude's
pick: P-eps-A.

**laug2 KILLED at ep4 (Berker confirms the fall; 2026-07-19 ~12:55):** ep4 .0728 < ep3
.0944 — the v1 fork one epoch later, softer slope, same direction; chain 62421228-30
cancelled. With nobn2's ep3 fall this closes the dose axis: **share restoration (the
certified equal-pull correction, D-047) does NOT prevent the collapse under lejepa V=4 —
dose alone is refuted; the question moves to the functional/structural level.** Berker
directive: dedicated fix session; question = why do var+cov and sigreg work at z but
MomentFloor does not (→ standalone floorssl). Evidence base + 2×2 synthesis
(demand-type × share; all floorssl failures in shape×owner) + discriminating-experiment
menu handed over in SESSION_OPENER.md. laug_eps (shape×minority, 91/6/3) left RUNNING —
its ep3–6 is the next session's first read.

**CORRECTION (Berker 2026-07-19: "momentfloor demands for first two moment match no? while
sigreg asks for complete [isotropic] gaussian at z"): CONFIRMED — the demand ordering is
vicreg-var ⊂ MomentFloor (μ=0, Σ=I only; higher-order-blind by construction) ⊂ SIGReg
(full distribution via CF). The earlier "2×2" mislabeled the axis; the record's synthesis
is a MONOTONE tradeoff: more demanded structure ⇒ smaller content-safe share (vicreg owns
at ~50% · floor helps ≤6% / kills at owner · sigreg works at 2% / taxed at 10%). Estimator
rider (measured on perfect N(0,I): floor phantom .070/.158 at n=512/256, ~all aniso, MP
eigs [.26,2.18], unbounded logdet vs sigreg CF .0075–.027 bounded, vicreg hinge .013–.019)
— per-weight severity ≫ the two-moment definition suggests; phantom does NOT explain the
frame fork (byol frame = more phantom, healthy). Full corrected synthesis in
SESSION_OPENER.md.**

**vicreg-vs-floorssl contrast (Berker's ask; code-verified; full table in
SESSION_OPENER.md):** same trunk/expander/inv — the anti-collapse differs on FOUR axes:
(1) target set: diagonal-Σ-with-diag≥1 (huge feasible set, anisotropy allowed axis-aligned)
vs Σ=I exactly (a point); (2) pressure: one-sided-up + decorrelate vs symmetric
shrink+equalize; (3) coordinates: fixed axes (gauge-exploitable) vs rotation-invariant
slices; (4) **payment: per-view batches (one view/image ⇒ only ACROSS-IMAGE spread pays;
aug variance cannot) vs pooled views (aug variance pays)**. Smoking gun (measured):
collapsed cells at ep2 have across-image var ≈ .02/dim (std .14) — the floor ~92%-paid in
aug noise while vicreg's hinge would read .86/dim; healthy byol-frame cells paid 71% in
image spread. The payment axis explains the aug-frame fork the demand hierarchy alone
could not (strong augs = abundant aug-payment = cheap content-free equilibrium), and the
law-of-total-variance budget (within+across = 1/dim) is the inv-coupling vicreg simply
does not have. Design wall carried to the fix session: pooling exists FOR the slice
estimator (n/d′=4) — a view-mean floor (across-image payment, keeps Σ=I identity) makes
n=bs=d′ singular ⇒ payment rule and estimator must be co-designed (d′≤64, or diagonal/1-d
bounded floor, or bigger bs). New menu item (2b): view-mean floor; composes with
one-sidedness (2).

**CORRECTION (2026-07-19 fix session, found while deriving the 2b payment arm; verified
against `vicreg.py:115-127` + law of total variance): the contrast's PAYMENT axis (axis 4)
is WRONG as stated.** vicreg's var-term batch holds one view PER IMAGE, but each sample
carries its own augmentation draw: Var_i[z_i^{(a)}] = Var_i[μ_i] + E_i Var[ε_i^{(a)}] =
across + within — **aug variance inflates vicreg's variance reading exactly as it does the
pooled floor's** (in expectation; pooling only reduces estimator noise on the within part).
The ".86 vs .24 smoking gun" is therefore invalid arithmetic (it credited vicreg with
reading across-var only; at the collapsed state vicreg's hinge would read 1−√.24 ≈ .51 —
it fires like the floor does). What SURVIVES measured: the state description (across-image
var .02/dim, R@z .92 — the collapsed z really is 92% aug noise), the g_inv −24% frame
effect, and axes 1–3 (target set / sidedness / coordinates). The frame-fork explanation
shifts from payment ELIGIBILITY to within-variance MAGNITUDE: a stronger view family makes
within-var larger, and under a SYMMETRIC cap (total ≈ 1/dim) larger within crowds out
across — the budget, not the payment rule, is the frame-dependent lever; one-sidedness
removes the budget ceiling (axis-2 mechanism, unaffected). Two downstream consequences:
(i) menu 2b's rationale downgrades — a view-mean floor reads across + within/V (lever ÷V,
not eliminated; vicreg itself reads full within); (ii) **"vicreg works at z" is an UNTESTED
premise under the killer frame** — vicreg has only ever run under its own/byol-class
families in our record; whether var+cov survives lejepa V=4 at z is unknown (candidate
cheap control cell if the mechanism question stays open after the dimension move).

## One-sided floor arm (fix session 2026-07-19 afternoon; menu item 2; D-049 PROPOSED)

**Derivation (THE QUESTION, worked from the record):** the 4-axis contrast + the monotone
demand-share tradeoff localize the suspected poison to the SYMMETRIC half of the floor's
demand. Above-floor penalization (scale push-down + aniso equalize) is a CAP on every
direction's variance; under pooled-view payment the cap enforces the total-variance trade
(within + across ≈ 1/dim) — inv progress must be BOUGHT from across-image spread. Under
abundant aug-payment (V=4 strong family) the cheap equilibrium — views spread, zero content,
the measured 92%-aug-paid state — sits INSIDE the cap, and content growth would violate it.
One-sidedness removes the CAP, not the payment: across-image spread above the floor becomes
FREE, so vicreg's healthy exit (views align while across-var takes over the floor's demand)
is no longer blocked. `HingeFloor` (`sslgap/methods/_common.py`): per fresh slice direction,
relu(1 − std) + MomentFloor's cone term. Population target set: **Σ ⪰ I (λ_min ≥ 1) — a
literal floor in the PSD order** — one-sided, rotation-invariant (no vicreg per-dim gauge),
two-moment. Diagonal-of-slice (NOT eigenvalue) hinge by construction: at n/d′=4 the MP
eigen-spread [.26, 2.18] would hand an eigen-hinge a ~10× phantom.

**Measured properties (scratch `hinge_props.py`, 2026-07-19; D=2048, d′=128, 20 fresh
slices, mean over draws):**

| state | hinge | KL floor |
|---|---|---|
| perfect N(0,I), n=512 / 256 (PHANTOM) | **.0135 / .0202** | .070 / .158 |
| collapsed-cell state Σ=.24·I | **.511 — fires HARDER than KL** | .404 |
| mid-collapse Σ=.5·I | .293 | .166 |
| healthy-aniso powerlaw, effrank 21.6% (vicreg-z-like) | **.0185 ≈ its own phantom = FREE** | .105 |
| healthy-aniso, effrank 9% (byol-z-like) | .0245 | .143 |
| rank-64 / 16 / 4 redundancy, tr=D | .040 / .083 / .172 — graded pressure, NO cov term needed (slice mixing spreads direction-vars χ²-like; below-1 mass fires) | 2.22 / 3.90 / 4.41 |

The hinge is a STRONGER floor below 1 and a ZERO force above it ⇒ a survival is attributable
to the demand axis, not to a weakened anti-collapse. Byte-check dry-run PASSED (scratch
`hinge_bytecheck.py`): (1) init byte-identity 166/166 tensors across z_floor=kl/hinge (the
switch perturbs no seed-0 draws — the arm's init ≡ the whole E21 lineage's); (2) RNG-stream
parity (identical per-forward draw pattern ⇒ h-floor slices matched across arms at matched
steps); (3) pinned-RNG training_step: inv + h_moment_kl IDENTICAL, only the z-term semantics
differ (kl 3.92 vs hinge .69 at init); backward clean to trunk; (4) one-sidedness live
(×3 → .007 = cone-only, ×1 → .016 = phantom, ×0.3 → .70).

**Arm `in100.floorssl.s0.laug_hinge`:** z_floor=hinge; h-floor UNCHANGED symmetric KL (the
≤6%-share conditioner is the certified-GOOD regime — only the z OWNER term changes form);
lejepa V=4, bn head, bs=128 — **pooled payment deliberately HELD (one axis per arm; 2b is
the payment arm if needed)**. Doses **32.8 / 45 / .617**: inv and h carry their
D-047-certified frame corrections (both terms unchanged from laug2 ⇒ the correction
transplants), the NEW z-term takes the lineage's nominal owner dose 45 (no bridge is
measurable for a term that didn't exist at the bridge state; realized pulls at cadence).
This makes the cell an EXACT single-axis contrast to laug2 (43.8·KL → 45·hinge, ±3%
nominal): collapse ⇒ form insufficient even at certified doses; survival ⇒ the symmetric
demand was the poison (dose already ruled out by laug2). Deliberate divergence from the
menu's literal "v1 doses" — the menu's intent is the owner REGIME; raw v1 doses would
re-import the certified −24% inv-share miscalibration and blur both outcomes. Logging key
`moment_kl` retained for the z-term (reader compat; semantics per-arm, this card).

**Pre-registered predictions:**
- **P-hinge-A (Claude's pick):** healthy through the ep3–6 fork (monitor climbing past v1's
  .0868 ep3 peak at ep4+); R@z leaves the .92 collapse band toward vicreg's ~.19 as
  across-var grows; z goes anisotropic/heavy-tailed (effrank down, kurt up — the healthy-z
  signature) at direction-var ≳ 1. Mechanism read: cap-removal dissolves the trap.
- **P-hinge-B:** same ep3→4 collapse ⇒ one-sidedness under pooled payment insufficient —
  payment axis (2b view-mean, estimator co-designed) becomes primary suspect.
- **P-hinge-C:** no collapse but no alignment — R@z stays ≥ .9, monitor sideways: the floor
  lives on aug spread and inv never wins. Distinct from A by R@z + monitor slope.

Kill criteria standing (probe ≤2× chance @ep≥10; ep3→4 fall = the fork signature; grad
incident rule). Launch: 2-ep smoke → 3×8h chain on h100-slotA (free), afterok+singleton.

**laug_eps VERDICT (2026-07-19 ~14:25; Berker: "fails hard"): COLLAPSED — P-eps-C.**
Monitor .0356→.0854→**.1100@3**→.0668@4→**.0160@5** (≤2× chance = the pre-registered kill;
per-step probe loss 4.64–4.65 ≥ ln100 = 4.61 — at/below chance). Same ep3→4 fork as every
lejepa-frame cell; the ep3 peak was the family's highest before the same fall. Forensics
(per-step): probe-loss reversal at ep3.25 (4.147→4.318→4.65); inv rose off its trough
during the fall (.0238→.0293, +20%) then RE-SETTLED to .0231 by ep5 — content-free
alignment reached (the starved equilibrium's terminal state); moment_kl FLAT 1.81–1.84
throughout (read at the time as "NOT P-eps-B — z-var never crashed"); h_kl stayed in the
healthy band (.27–.28); no gradient incident.
**SUPERSEDED by the terminal anatomy (job 62421833, row laug_eps ep5 z): per-dim z-var
= .0117 — the z DID scale-crash (10× below even the sqrt-law's ~.1), and the flat 1.83–1.87
moment_kl IS the crashed steady state (KL scale channel at m=.012 computes to 1.73 of the
1.87 total). Corrected mechanism: w_floor=4.16 never held z-scale (crash completed within
warmup), inv's ownership was VACUOUS on a near-zero z (MSE trivially small at zero scale —
the pinned .023 inv was satisfaction by implosion, not alignment), and the ep2–3 probe
climb was h living on the residual trickle until rising lr consumed it. laug_eps = P-eps-B
(scale starvation), NOT P-eps-C: the ε-arm's failure is the floor being too WEAK for its
scale duty at 6% share on a 2048-d z — it does not implicate the form at every share.
R@z .90 at terminal (within-dominated at tiny scale, pos-cos .049). ep3 _best row (job
62422212): z mbar .0142 at the .1100 peak — z was ALREADY DEAD while the monitor climbed;
the climb was h alone (cls mbar .90, kl .287), consumed once lr rose.**

**Hinge escape-state anatomy (job 62422212, laug_hinge ep6 — the killed diagnostic's kept
ckpt; completes P-hinge-A's unread clauses, ALL REALIZED):** z row — **R@z .186** (out of
the .9 collapse band, INSIDE the aligned band: f2 proj .151, lejepa ctrl .163; v2's
healthy-but-elevated .615 left far behind) · per-dim var **.842** (scale held at the
floor's target; no implosion, no aug-spread payout) · pos-cos .763 (real view alignment) ·
KL decomposition cone .002 / scale .010 / **aniso 3.606** — the z is maximally anisotropic
and otherwise floor-compliant: the healthy-z signature (vicreg z effrank 21.6%, kurt 245)
produced BY the one-sided floor at owner dose. cls row healthy (mbar .384, kl .762, R .63).
**The mechanism answer to THE QUESTION is now measured at every corner of the 2048-d
square: symmetric-at-owner = alignment sacrificed (v1/laug2, R@z .92+) · symmetric-at-6% =
scale implosion under owner-inv (laug_eps, mbar .012) · one-sided-at-owner = aligned +
scale-held + anisotropy-free = healthy (hinge). A z that carries content must either be
ALLOWED anisotropy (one-sided — vetoed as identity) or be SMALL enough that full Σ=I
compliance costs no content because content lives upstream at h (the sigreg configuration
= the D-050 bracket, verdict pending).** RAW; joint interpretation with Berker owed. **Axis closure: dose/share is now refuted at THREE
structures — floor-owner 65% (v1), restored-inv 35% (laug2), inv-owner 91% (laug_eps) —
all fork at ep3→4 under lejepa V=4, two of these structures healthy under the byol pair.
The collapse onset is invariant to term balance and switched by the aug family ⇒ the
mechanism lives in the loss FORM × strong-view interaction (the cap and/or payment axes),
exactly what laug_hinge (running, single-axis form change) discriminates next.** Chain
killed at ep5 (62421497-99); ckpts kept (_best = ep3 .1100 peak state — the family's best
pre-collapse state, bridge-quality forensics input). Delayed anatomy 62421833 kept: reads
the terminal R@z for the R-table's failure family.

**Gradient-direction rider (menu-4 residue, measured; scratch `floor_grad_direction.py`:
per-original-dim variance velocity −2·E[g·x] under each floor, D=2048, n=512, 10 slices):**
on perfect isotropy the KL floor's noise-chasing NETS OUT (velocity ±3e-9 on the true
spectrum — phantom gradients average away, no systematic contraction; hinge +1e-6 mild up).
On the healthy-aniso spectrum the KL floor is an active ERODER — top-decile true dims pushed
DOWN (−1.6e-6/step·unit-lr) while dead dims inflate (+1.2e-7): the isotropizing action
measured at gradient level. The hinge on the same spectrum pushes UP ∝ current variance
(+4.0e-6 top, +4.2e-7 bot — rich-get-richer along the EXISTING shape, vanishing once all
slice directions clear 1): shape-preserving. At the collapsed state both restore (+1.5e-6 /
+0.9e-6 all dims). RAW.

**Launch record:** smoke 62421787 (gpu) → h100-slotA chain 62421788→789→790, gated
afterok:62421787,singleton. Reader updated (`e21_curves.py`): LIVE = laug_eps + laug_hinge,
killed cells as dashed refs, monitor-fork table added (probe-by-epoch ≤ ep12).

**RE-SCOPE (Berker 2026-07-19 ~14:40): hinge VETOED as method identity ("if you add
hingefloor to floorssl then it becomes vicreg with slicing we do not want that") — arm
downgraded to DIAGNOSTIC-ONLY: harvest the ep3–4 fork datum (does cap-removal stop the
2048-d collapse — the mechanism read), then kill the chain. Method direction → D-050.**

**DIAGNOSTIC OUTCOME (2026-07-19 15:35; RAW): the hinge cell did NOT fork — monitor
.0462 → .0698 → .0906@3 → .0910@4 (the transition where v1/laug2/laug_eps fell −63%/−23%/
−39%) → .1124@5 → .1300@6 (final line before the scancel landed) CLIMBING (KL cells were
.016–.022 at ep5, dead). Escape, not delay:
**cap-removal alone stops the ep3→4 collapse at 2048-d, same frame, same corrected doses —
the symmetric above-floor demand is the collapse mechanism at owner share under strong
views** (P-hinge-A's fork clause realized; the R@z/anisotropy clauses unread — arm killed
at ep5 per the veto, chain 62421788-90 scancelled, ckpts kept). Triangulation with the
D-050 smokes (all four small-z KL cells opening at ~2× the 2048-d family): the collapse
requires BOTH the symmetric cap AND a large owned space; the method drops the second leg
(small z, identity intact) — the first leg's confirmation is mechanism knowledge, not a
method change.

## D-050 — the small-z dimension bracket (Berker 2026-07-19: "we are somewhere in between
## sigreg and vicreg... vicreg relies more on the expander and sigreg has an advantage with
## small output dim... start with lejepa and increase a bit"; "you can try many, we are not
## compute bound. only cap is 2 h100s")

**Derivation:** lejepa's projector is MLP 512→[2048,2048,16] — hidden ALREADY equals
vicreg's expander width; the entire architectural boundary vicreg↔lejepa is the OUT-dim
(2048 vs 16). Read through the demand hierarchy: **out-dim is the demand-affordability
dial** — vicreg's minimal demand affords 2048 owned dims; sigreg's full demand affords
complete compliance only on 16 with content upstream (embed, unregularized there); our
two-moment demand at owner share should afford an intermediate dim. FloorSSL's upstream
story is the method's certified advantage: content lives at h under the calibrated floor
(the 4/5-lane zoo improver), so a small fully-owned z is the sigreg CONFIGURATION with our
loss at both places and identity intact (Σ=I KL floor; hinge vetoed). Estimator dissolution
rides for free: at D_z ≤ 128 the floor reads the EXACT full covariance (`floorssl.py`:
d_slice = min(128, expander_dim); byte-identical at 2048) — no slice subspace sampling,
and the phantom shrinks with n/d (measured, n=512, N(0,I)): **d16 .0090 · d32 .0171 ·
d64 .0340 · d128 .0702** (the 2048/128-slice setup's .0702 = the bracket's top). Slice
noise was separately acquitted as a collapse/instability cause (fixed-batch fresh-slice
std < 1% even on spiked spectra vs the observed 8% batch/heavy-tail jitter; v2 healthy at
MORE phantom n=256; smooth-equilibrium forensics) — Berker's noise flag, measured record
in scratch + conversation 2026-07-19.

**Cells `in100.floorssl.s0.d{16,32,64,128}`:** expander_dim ∈ {16,32,64,128}, ALL ELSE
laug2-verbatim — bn head, hidden 2048, lejepa V=4, bs=128 (pooled n=512), z_floor=kl,
doses **32.8 / 45 / .617** (the D-047-certified frame corrections carried; single-axis:
only the space changes). Trunk init byte-identical across all four (seed-0, trunk drawn
before head — verified) ⇒ a trunk-matched family; d16 = the lejepa anchor point under our
loss; 2048 = the refuted anchor. Dry-run PASSED all dims (exact floor, finite step,
backward clean). Health flag: init floor pull grows as dim shrinks (trunk-grad@init 180
(d16) / 101 / 74 / 63 vs 20 (d2048)) — clip=1.0 + 10-ep warmup standing, ep0–1 envelope
watched.

**Pre-registered predictions:**
- **P-dim-A (Claude's pick, the affordability frame):** survival is monotone in dim — a
  threshold D* separates healthy small-z cells (climb through the ep3–6 fork; R@z falls
  toward the aligned band as inv+floor co-own a small space) from starved large ones
  (2048's fork returns). The method result = D* and the h-probe-vs-dim curve (bar:
  aug-matched lejepa e20f .6700/.6288).
- **P-dim-B:** all four fork at ep3→4 ⇒ dimension is not the axis — the strong-view frame
  defeats inv+floor at z at ANY dim ⇒ next diagnostics: the untested vicreg control under
  lejepa V=4 (does the reference survive?) and the family-vs-V split.
- **P-dim-C:** all four survive ⇒ only the 2048 point was broken (affordability with a very
  high D*); bracket extends upward (256/512) to find it.
- Sub-reads: d16 vs d32/64 h-quality orders the "increase a bit" intuition; z-var
  equilibrium per dim vs the sqrt(F/W) law; the floor-value trajectory now phantom-clean
  at small dims (exact estimator) — the aniso/scale channels become trustworthy per-step.

Kill criteria standing (probe ≤2× chance @ep≥10; ep3→4 fall; grad incident). Compute per
Berker's directive: gpu-partition chains for d16/d64/d128, h100-slotA for d32 (the primary)
after the hinge diagnostic dies; ≤2 H100 respected (slotB = dino lam06, hands off).

**Launch record (2026-07-19 ~14:55):** smokes d16 62422025 · d32 62422029 · d64 62422033 ·
d128 62422037 (gpu, 2-ep) → chains gated afterok+singleton: d16 62422026-28 (`e21-d16`) ·
d32 62422030-32 (**h100-slotA**, queued behind the hinge diagnostic's singleton) · d64
62422034-36 (`e21-d64`) · d128 62422038-40 (`e21-d128`).

**Smokes ALL PASSED (~15:20; RAW):** ep1 / ep2 — d16 .0660/.1324 · d32 .0878/**.1452** ·
d64 .0780/.1436 · d128 .0844/**.1458**. Context: the 2048-d family's ep2 band is .070–.085
and its ALL-TIME peaks were .087–.110 (ep3, pre-fork) — every bracket cell EXCEEDS every
2048-d peak by ep2, and d32–d128 are flat within noise (.1436–.1458; d16 −.013 below).
CAVEAT STANDING: all 2048-d cells also climbed through ep2–3 before the ep3→4 fork — the
bracket's verdict is the CHAIN fork window (~16:00–16:45), not the smokes; chains re-cover
ep1–2 seed-identically first.

**FORK VERDICT (2026-07-19 16:10–16:30; RAW): ALL FOUR THROUGH — ep3→ep4: d16 .1500→.1938
(+29%) · d32 .1762→.2200 (+25%) · d64 .1960→.2202 (+12%) · d128 .1822→.2340 (+28%), at
the transition where every 2048-d KL cell fell −23%…−63%. The symmetric Σ=I floor at owner
share is HEALTHY under lejepa V=4 at every dim ≤128 tried ⇒ P-dim-C realized within the
bracket (no interior D*; the affordability threshold sits in (128, 2048]). Cross-arch note:
d32's chain (H100) is its own trajectory vs its A40 smoke (seed reproduction is
arch-local); d16/d64/d128 chains seed-exact to their smokes. Attribution caveat: d128 vs
2048 changes BOTH space size and estimator exactness (slice sampling returns only above
128) — slice noise stands acquitted (<1% reading noise), so dimension is presumptive;
a d256 cell would separate them if the attribution is wanted. ep5+ continuation, fork-state
anatomy (R@z / z-var / aniso per dim: sigreg-configuration check), and the h-probe-vs-dim
ordering vs the lejepa e20f bar (.6700/.6288 at ep100) are the remaining reads; kill
criteria stay armed through ep10.**

**Upward extension LAUNCHED (16:35, the P-dim-C clause + estimator attribution): d256 =
the discriminator (slicing returns above 128 — d256 forking while d128 lives implicates
the exact↔sliced estimator boundary; d256 healthy pushes D* toward 2048 on the dimension
axis) · d512 fills the curve. Same doses/frame, gpu chains: smokes 62422482/62422486 →
chains 62422483-85 (`e21-d256`) / 62422487-89 (`e21-d512`). Fork-state anatomy for
d16–d128 queued (job 62422490, ~ep5 states). ep5 continuation: d64 .2620.**

**LANDINGS (2026-07-20 morning; final ONLINE monitors — offline v2 probes vs the bars are
the comparison of record, pipeline rows below as they land):** d32 ep100 monitor **.6536**
(rising to the end; run best .6568) → landed via e21_land.sh: extract 62430289 → probe
62430290 / battery 62430291 (`in100.floorssl.s0.d32.ext`). d64 ep100 monitor **.6886** (best
.6940) → extract 62432056 → probe 62432057 / battery 62432058 (`.d64.ext`). d16 ep100
**.5824** (best .5834) → extract 62432061 → probe 62432062 / battery 62432063
(`.d16.ext`). d128 ep100 **.6848** (best .6850) → 62432547/549/550. d256 ep100 **.6984**
(best .6984) → 62432551/552/553. d512 ep100 **.6848** (best .6862) → 62432542/543/544.

**ALL SIX LANDED — the online-monitor probe-vs-dim curve at ep100 (RAW; the OFFLINE v2 lin
probes vs the bars are the number of record, landing when the probe jobs finish):**

| dim | 16 | 32 | 64 | 128 | 256 | 512 | (2048) |
|---|---|---|---|---|---|---|---|
| ep100 monitor | .5824 | .6536 | .6886 | .6848 | **.6984** | .6848 | collapsed |
| best-ck | .5834 | .6568 | .6940 | .6850 | .6984 | .6862 | — |

**LANDING BUG (fixed 2026-07-20): the first landing wave's extract jobs all FAILED —
`KeyError: 'floorssl'` in `sslgap/ckpt/adapters.py` `_NATIVE_ASM` (the independent FloorSSL
class is new this session; the E19/E20 floorssl cells ran under the vicreg CLASS, so
extraction had never seen `method="floorssl"`). Fixed at the design level, not patched:
floorssl's native extraction layout IS `_asm_projector` — same backbone+projector roles,
h=trunk-CLS (D-036), z=projector taps (`TVMLPTaps` walks its nn.Sequential head identically
to vicreg's expander) — so it registers to the SAME assembler (`"floorssl": _asm_projector`),
no floorssl-specific branch. Verified: native load of d32_ep100 returns probed=student.h.cls.
All six re-landed (extract 62432600/603/606/609/612/615). Gotcha carried: a new independent
method class needs its `_NATIVE_ASM` entry before landing.**

Shape: rises steeply 16→64, a broad flat plateau 64–512 (.685–.698, all within ~.014),
peak at **d256 .6984**. Online-monitor bars for context (lejepa e20f .6700 lin / f2 .6578):
the whole 64–512 plateau sits at/above the lejepa online monitor; d256 is +.028 over it on
this instrument. INTERPRETATION DEFERRED to the offline v2 probes (the comparison of
record; online monitor is optimistic and lane-shared) and to joint discussion — NO takeaway
without Berker.

**EXTENSION FORK VERDICT (19:15–19:30; RAW): BOTH THROUGH — d256 .1698→.2082 (+23%) ·
d512 .1582→.2092 (+32%) at the killer transition. With the ≤128 bracket: EVERY dim
16–512 survives; only 2048 collapses ⇒ **D* ∈ (512, 2048]**. Estimator attribution
resolved: d256/d512 run the SLICED estimator (50%/25% coverage) and live — the
exact↔sliced boundary is NOT the mechanism; the axis is the dimension of the owned space
itself. ep10 kill-horizon cleared by all four original cells (d128 .3384 > d64 .3264 >
d32 .2970 > d16 .2568, ep11 climbing .265–.356). Remaining reads: ep100 h-probe-vs-dim
(the method result) + anatomy cadence at ep25/50/75/100 via the landing pipeline; optional
D*-pinning cells (d1024, d1536) if the boundary's location matters beyond (512, 2048].**

## OFFLINE LANDINGS (2026-07-20 — the numbers of record; RAW, joint read owed)

All six d-cells landed end-to-end (extracts 62432600–615 after the adapter fix → probes
`results/probes/in100.floorssl.s0.d*.ext.csv` + batteries `results/battery/…d*.ext.*`).
Centered-kNN companion `results/diag/e21_centered.csv` (job 62434490; instrument =
`experiments/e21_centered.py`, the E20 convention): centering moves floorssl h.cls by
≤±0.3pt — cones trained away, the E20 arm-side pattern. The collapsed 2048-d contrast
column = `lejepa_augs2` landed at its ep4 kill state (extract 62434434; NOT an ep100 cell).

**Offline v2 probe-vs-dim @ trunk h.cls (lin = linear_raw_v2 · knn = knn200 CENTERED):**

| dim | 16 | 32 | 64 | 128 | 256 | 512 | 2048 laug2@ep4 |
|---|---|---|---|---|---|---|---|
| lin | .5792 | .6428 | **.6850** | .6682 | .6830 | .6668 | .1482 |
| knn_c | .5196 | .5782 | .6188 | .5882 | **.6280** | .6222 | .1062 |

Bars, read from their own record files: lejepa e20f @z.embed (declared) **.6700 / .6288** ·
lejepa e20f @h.cls .6750 / .6114 · f2 @z.embed .6578 / **.6030 knn_c** (store intact after
all — same-instrument centered 2026-07-20; the earlier ".6056-RAW, store purged" note was
wrong on both counts). Same pass: lejepa ctrl h.cls knn centers .5134→**.5312** (+1.8 — the
one non-trivial centering delta; its audit rand-cos .767 = the cone the arm cells lack).
The OFFLINE shape ≠ the online monitor's (monitor: broad 64–512 plateau, peak d256 .6984):
offline is TWO-PEAKED — d64 tops lin, d256 tops knn_c — with a REAL d128 dip on both columns
(consistent across probe variants raw/house/l2) and a d512 fade the monitor did not show.
Against the declared lejepa bar: d64 +.0150 / d256 +.0130 on lin; knn_c d256 −.0008 (par).
Against lejepa's arch-matched h.cls read: d64/d256 above on lin, d64/d256/d512 above on knn_c.
z-space probes (content-parking gradient; lin @z.proj.out): .3818 / .5042 / .5824 / .5816 /
.6182 / .6238 — z stays .04–.20 below h at every dim.

**Battery structural gradient (file-exact; z = each run's loss-terminal `z.proj.out`, h =
`student.h.cls` 384-d; pairs on the fixed audit_v1 stack):**

| cell | z effrank/d | z KL/dim | z kurtw | h effrank | h kurtw | h KL/dim | h cos_inv |
|---|---|---|---|---|---|---|---|
| d16 | .988 | .022 | 58 | 184 | 1.25 | .77 | .522 |
| d32 | .988 | .016 | 37 | 196 | 5.10 | .62 | .606 |
| d64 | .988 | .014 | 68 | 208 | 4.31 | .51 | .668 |
| d128 | .976 | .026 | 53 | 226 | 3.85 | .45 | .684 |
| d256 | .933 | .040 | 163 | 231 | 0.59 | .44 | .666 |
| d512 | .884 | .082 | 150 | 229 | 0.66 | .43 | .670 |
| lejepa ctrl (z=16) | .994 | .005 | 2.1 | 27 | 0.71 | 3.74 | .967 |
| vicreg ctrl (z=2048) | .216 | 2.351 | 245 | 67 | 0.86 | 2.33 | .921 |

RAW observations for the joint read (no interpretation): the z stays essentially Σ=I-compliant
at EVERY dim (KL/dim ≤ .08; effrank/d ≥ .88 — never approaching vicreg's .216 configuration)
while z worst-kurt jumps 3× at d256/512 (moment-compliant, shape-heavy); h effrank RISES with
dim (184→231, all 3–8× both ctrls); h worst-kurt spikes at d32–d128 (5.1/4.3/3.9) and drops
to ~.6 at d256/512; h view-metrics sit mid-band (cos_inv .52–.68 vs ctrls .92–.97 under the
same audit stack — the d-cells train under lejepa V=4, ctrls under their own recipes).

**Metric figures (Berker's JOB-1 ask; FS-only + INDEX; RAW, no takeaway on-figure):**
`results/figures/e20/e20f_metric_panel.png` · `results/figures/e21/e21_metric_panel.png` ·
`results/figures/e21/e21_probe_vs_dim.png` (renders `experiments/e20f_metric_fig.py` /
`e21_metric_fig.py`, defaultp; final chain 62435296 with the laug2 column).
**Pair-panel conversion (Berker 2026-07-20: "do they show the same thing?"): CONFIRMED —
Wang–Isola alignment ≡ 2−2·cos_invariance on L2-normalized pairs (exact affine; e.g. vicreg
arm 2−2(.560)=.880 vs .879), one number rendered twice. Both figures' pair panels replaced by
POS-cos and RAND-cos separately (rand pair = cross-view different-image pair, same stack —
the M1 pair_margin instrument), computed fresh from the intact audit_v1 pair stores:
`experiments/e2x_posneg.py` → `results/diag/e2x_posneg.csv` (E20 zoo arm+ctrl · dino dose
cells · d-cells · laug2 · f2; no re-extraction was needed — the "purged" assumption was
wrong). Berker's quad ask → `results/figures/e21/e21_quad.png` (`e21_quad_fig.py`): d64 ·
d256 · lejepa e20f · lejepa ctrl — probes + battery + pos/rand dumbbells at h.cls / declared
embed / loss-terminal z; convergence caveat on-figure (all four monitors still climbing at
the 100-ep cut — Berker's flag, E20-T2 pattern). Dino dose cells landed same day →
`results/figures/e20/e20_dino_dose.png` + numbers on the E20 card §dino-fix.**

**Quad v2 (Berker session-wrap asks, regenerated 2026-07-20 evening): + the dino arms
(dose-winner λ.008 + dino ctrl; full dose family stays on e20_dino_dose.png) + TWO NEW
INSTRUMENTS, both cross-method scale-invariant:**
- **z-from-h linear predictability** (`experiments/e2x_zpred.py` →
  `results/diag/e2x_zpred.csv`): val R² of OLS h→z on the training-time head pair (student
  h.cls → loss-terminal z; 1 = the head is affine in effect). Measured: lejepa e20f **.962**
  / ctrl .949 (the 2048-2048-16 MLP is nearly affine in effect) · d64 .881 · **d256 .578** ·
  dino λ.008 .391 / ctrl .407 (the GELU+l2norm bottleneck head is the zoo's "beast").
- **class-conditioned cosine at trunk h** (`experiments/e2x_classcos.py` →
  `results/diag/e2x_classcos.csv`; exact all-pairs via class sums, train split): same/diff —
  lejepa ctrl .885/.776 (gap .109 on a .78 base — the cone) · e20f .295/.024 (gap .271) ·
  d64 .268/.013 (.255) · d256 .213/.014 (.198) · dino ctrl .358/.159 (.199) · λ.008
  .261/.083 (.178). Floor-family cells put the diff-class base at ~0 with the largest gaps.
- Centered-kNN extended to the dino pair (same instrument): λ.008 .6336c vs ctrl .6090c
  (+2.5c — matches the E20-T1 completion read).

**z-pred DIMENSION-CONFOUND CONTROLS (Berker's flag: "this linearity thing could be
confounded by the dimension difference"; job 62444630, rows in e2x_zpred.csv):** the
RANDOM-HEAD null (seed-0 norm-free MLP on the d256 run's fixed h) is FLAT across d_z —
.6513/.6531/.6520/.6517 at d 16/64/256/2048 — dimension per se does not move the metric
(random functionals of one shared hidden have d-independent linear share). The TRAINED
bracket is steeply d-dependent — d16 .9234 · d32 .9207 · d64 .8808 · d128 .8066 · d256
.5782 · d512 .4380 — so the gradient is TRAINED-IN (the Σ=I demand over more owned dims
recruits the head's nonlinearity), and it CROSSES the null: d16–d128 run MORE linear than
random (the loss satisfiable in the hidden's linear corner), d256/d512 LESS. Cross-dim
comparisons on the quad therefore mix head identity with loss demand (Berker's suspicion
confirmed in direction); MATCHED-DIM contrasts survive clean: floorssl d16 .923 vs lejepa
16-d .949/.962 (real, small) · floorssl d256 .578 vs dino bottleneck-256 .391/.407 (dino's
head is the genuine beast at matched dim) · vicreg 2048-d expander .367 = the far anchor.
Quad panel updated with the null as a dashed d-flat reference line (bars above = head more
linear than random, below = beastlier).
**Instrument correction found while building them (2026-07-20): `e20_battery_table.py`'s
pairs loader was last-row-wins over multi-manifest pairs files — the compare CSV's h.pairs
rows mixed frames (own_<m> for most lanes, foveal for lejepa ctrl, audit_v1 for lejepa-e20f).
Fixed: pairs reads filter to `@audit_v1` (PROTOCOL §5 cross-lane convention);
`e20_battery_vs_ctrl.csv` regenerated in the final render job; E20-card §(b) pair numbers
(".65 vs .09–.28" block) are superseded — correction noted there once the regen lands.**

## View-mean payment arm (D-051; Berker 2026-07-20: "we make batch size x views as batch
## dimension then impose our floorssl loss, but a cleaner version would be getting view-wise
## means and then applying our loss … right now how invariant they will be across views versus
## the spread is competing very obviously. (the current version can have its own merit, this
## is why we need to test for the d256 arm with the view-wise mean version) launch it on an
## h100 if you agree with my point")

**Status: PRE-REGISTERED 2026-07-20, numbers land below the predictions. Agreement recorded:
Claude agrees with the decoupling point — the pooled payment was an estimator-driven side
effect (pooling exists FOR the slice n, the card's own wall note), and the competition is
measured (ep25 R@z .19–.36 = the floor's budget share inv is actively fighting; the 2048
collapse was the 92%-aug-paid extreme of the same channel).**

**Payment algebra (the derivation Berker's two proposals are judged by; z_i^v = μ_i + ε_i^v,
A = Cov(μ) across-image, W = E Cov(ε) within/aug, μ ⊥ ε):**

| rule | floor input | reads (per direction, expectation) | n for the estimator |
|---|---|---|---|
| pooled (current) | [bs·V, D] | A + W | bs·V = 512 |
| **view-mean (proposal 1)** | [bs, D] means | **A + W/V** | bs = 128 |
| per-view, shared Q, averaged (proposal 2) | V × [bs, D] | A + W (each view batch carries one aug draw per image — the law-of-total-variance split lands the SAME aug term; the fix-session axis-4 CORRECTION's own algebra) | bs = 128 per call |

All three demand A = I in the aligned limit (ε→0); they differ in the MISALIGNED regime —
exactly where the ep3–4 equilibria form. Proposal 2 re-creates pooled's demand in expectation
with a 4×-smaller estimator batch (thin-sample logdet bias does not average out over V), so
it does not implement the decoupling motivation; its shared-Q element is adopted anyway
(below), and the h-floor is ALREADY a single shared-Q pooled call per step. Proposal 1 cuts
the aug payment by V: inv's progress on within-var costs the floor 1/V as much, the
content-free payment channel (the collapse mechanism's fuel) shrinks ×4, and the floor's
demand lands on image spread from the start. Declared risk: the z-floor no longer reads
per-view scatter AT ALL — within-var control rests on inv (w=32.8) alone (P-vm-B's channel).

**Estimator co-design (the card's recorded wall, now mandatory):** view-means drop the
floor's n to bs=128 ⇒ the d′=128 slice is SINGULAR (n/d′=1, rank-deficient logdet). The cell
runs **z_d_slice=32** (n/d′=4 — the 2048-family's own healthy phantom ratio, ~.07/dim;
12.5%/step fresh-slice coverage). Stream-parity construction (the hinge-arm discipline):
`MomentFloor(d_slice=32, d_draw=128)` still DRAWS the canonical 128-frame and uses its first
32 columns (Householder QR's leading k columns ≡ the fresh k-frame draw) ⇒ the per-step RNG
stream is IDENTICAL to the pooled lineage's — h-floor slices stay matched across arms at
matched steps (byte-check clause 2), and the vm z-slice is a sub-frame of the pooled cell's
same-step frame (common random numbers). h-floor UNCHANGED pooled cls (single axis; ≤6%-share
conditioner = certified-GOOD regime).

**Cell `in100.floorssl.s0.d256vm`:** d256-verbatim (bn head, hidden 2048, lejepa V=4, bs=128,
z_floor=kl, seed 0) + `z_floor_batch=view_mean z_d_slice=32`. Doses **32.8 / 45 / .617** —
the hinge-arm convention: unchanged terms (inv, h) carry their D-047-certified frame
corrections; the changed z-term keeps the lineage's nominal owner dose 45 (no bridge exists
for a term-form that wasn't at the bridge state; realized pulls at cadence). Init-pull note:
the vm floor reads a SMALLER input variance at any misaligned state (A + W/4 < A + W) ⇒
larger init floor value/pull — clip=1.0 + 10-ep warmup standing, ep0–1 envelope watched
(D-050's health-flag machinery).

**Pre-registered predictions:**
- **P-vm-A (Claude's pick):** decoupling wins — through the ep3–6 window healthy; R@z falls
  BELOW the pooled-d256 trajectory at matched cadence (floor demand landing on across-var);
  ep100 offline probes ≥ the pooled d256 bar (.6830 lin / .6280 knn_c).
- **P-vm-B:** within-scatter unmanaged — with the floor blind to per-view scatter, within-var
  stays high (R@z ABOVE pooled trajectory), z pos-cos falls, probes below pooled: the pooled
  version's "own merit" (Berker's caveat) is exactly the scatter backstop.
- **P-vm-C:** null — at d256 the equilibrium composition self-corrects either way; curves and
  endpoints within noise of pooled ⇒ the payment axis only matters at large D (2048 rescue
  candidate).
Comparison instrumentation: pooled d256/d512 anatomy cadence BASELINE launched (job 62435554:
d256 ep25/50/75/100 + d512 ep25/ep100 — these cells post-date the ≤128 cadence wave and had
no rows); vm gets the same cadence at landing. Kill criteria standing (probe ≤2× chance
@ep≥10; ep3→4 fall; grad incident).

**Byte-check dry-run PASSED (scratch/vm_bytecheck.py, job 62435560; 62435553 died on a
sh-vs-bash `source` in the --wrap, no code issue):** (1) init byte-identity 166/166 tensors
vs the pooled d256 build; (2) pinned-RNG step — inv .238964 IDENTICAL and h_moment_kl
3.497058 IDENTICAL across arms (the d_draw stream-parity construction verified live: both
arms draw the canonical 128-frame, h-floor slices matched), z-term differs (pooled 3.197006
vs vm 3.398903 — the vm floor reads the smaller mean-input variance ⇒ larger KL, the
predicted direction), vm backward clean to trunk; (3) default path historical
(d_slice=d_draw=128, pooled input — the landed zoo is byte-unaffected); (4) Householder
sub-frame property confirmed (32-slice ≡ fresh 32-frame).

**Launch record (2026-07-20 ~15:45):** smoke 62435580 (gpu, 2-ep, tag=d256vm.smoke) → chain
62435581→582→583 on **h100-slotA** (gpu100/H100, afterok:smoke + singleton; slotB stays
free, ≤2-H100 cap respected). Overrides = the d256 cell's recorded wandb args verbatim +
`method.z_floor_batch=view_mean method.z_d_slice=32`. Collapse-window (ep3–6) watched;
cadence anatomy at landing; pooled-baseline anatomy job 62435554 running. Smoke PASSED
(ep1 .0718 · ep2 .1114, climbing — above the 2048 family's all-time band by ep2).

**DOSE CHECK + re-dose (Berker 2026-07-20: "since we changed the way we compute the loss,
hyperparam sensitivity can occur — briefly check if chosen params are good"):** instrument
`experiments/e21_vmpull.py` (job 62436061 → `results/diag/e21_vmpull.csv`) — the D-047
bridge construction with the LOSS-FORM swapped instead of the aug: hold the pooled d256
cell's own states (init null · **ep25 = the certified formation-state dosing convention** ·
ep50 stability), same batch, same slice frame (d_draw parity ⇒ common random numbers), and
read per-term trunk pulls under both forms. Internal controls PASSED in-data: inv and h
g_enc bit-identical across forms at every state — only the z-term moves. Result: **g_z(vm)/
g_z(pooled) = ×2.42 at init (transient; clip=1.0 + 10-ep warmup standing) · ×1.162 at ep25 ·
×1.273 at ep50 (drift — realized pulls at cadence, not chased).** At nominal 45 the vm
z-floor's ep25 share becomes 51.2% (the floor becomes the majority term) vs the pooled
cell's healthy 47.5% — the arm would confound payment with a hotter dose. **Equal-pull
correction at ep25: w′_floor = 45/1.162 = 38.7** (restores shares 51.7/47.4/0.8 and total
pull 12.99 ≈ pooled's 13.00). Applied under the standing fix-now directive (D-048 class):
chain 62435581-83 CANCELLED at ~ep6 (hot-dose partial ckpts left on disk, tag d256vm —
45-dose record), relaunch = **`in100.floorssl.s0.d256vm2`, w_floor=38.7, all else
unchanged**: smoke 62436143 → chain 62436144→145→146 (h100-slotA, afterok+singleton).
P-vm-A/B/C carry unchanged; the ep3–6 window re-watched on the corrected cell.

**GRADIENT-INTERACTION READ (Berker's follow-up: "since they are not competing maybe it is
still fine? the effective diff is the learning rate like change right? how's the interaction
between floor pull vs inv pull?" — instrument `experiments/e21_vmcos.py`, job 62436375 →
`results/diag/e21_vmcos.csv`; trunk-space cosines between per-term gradients, same weights /
batch / slice frame across forms; controls exact — bitwise-identical inv and h grads, cos
+1.0000; the first two runs died on a harness bug then fp32 dot slop, both fixed, f64):**

| read | init | ep25 | ep50 |
|---|---|---|---|
| cos(g_inv, g_z) **pooled** | −.116 | **−.861** | **−.811** |
| cos(g_inv, g_z) **view-mean** | −.131 | **−.412** | **−.281** |
| cos(g_z pooled, g_z vm) | +.278 | .728 | .652 |
| cos(g_inv, g_h) | −.023 | −.319 | −.346 |
| cos(g_h, g_z) pooled / vm | .335/.133 | .339/.232 | .358/.228 |
| pre-clip ‖Σw·g‖ pooled / vm | 134/283 | 3.1/7.2 | 4.7/9.2 — clip=1.0 ACTIVE at every state |

Three answers this fixes on record: (1) the pooled competition is not metaphor — at formation
states the z-floor's trunk gradient runs near-ANTI-PARALLEL to inv's (−.86/−.81), and
view-mean cuts the opposition to −.41/−.28, FALLING with training (the within/V residue
shrinking) — Berker's decoupling intuition confirmed in direction, not eliminated in degree;
(2) the form change is NOT lr-like — cos(z_p, z_v) .73/.65: a different trunk direction, not
a rescale; (3) anti-inv cancellation budget at ep25 (w·g·|cos| against inv's own 6.72):
pooled@45 = 5.31 (79% of inv's pull cancelled) · vm@45 = 2.95 (44%) · vm@38.7 = 2.54 (38%)
— i.e. even at nominal 45 the vm form fights inv LESS than the healthy pooled reference
does; the 45→38.7 re-dose is an ATTRIBUTION choice (hold the mix, single-axis payment test),
not a safety necessity — and a vm@45 cell is now a motivated dose-axis probe if wanted.
Riders: the h-floor mildly opposes inv too (−.32/−.35 — the certified 2.8%-share tax made
visible); the two floors decorrelate under vm (cos(h,z) .34→.23). Single-batch cosines are a
read, not a doser (E20-opposition caveat standing).

**vm2 FORK VERDICT (2026-07-20 evening; RAW): THROUGH — ep1 .0730 → ep2 .1150 → ep3 .1704 →
ep4 .2034 (+19% at the transition where the pooled-2048 family fell −23..−63%) → ep5 .2152 →
ep6 .2272, climbing. Trajectory sits ON the pooled d256's at matched epochs (pooled ep3
.1698 → ep4 .2082 vs vm2 .1704 → .2034) — early monitors do not separate the forms; the
P-vm-A/B discriminator is the ep25 anatomy (R@z vs the pooled column .386). Bonus datum from
the cancelled 45-dose partial (tag d256vm, killed ~ep6): it too was fork-healthy — ep3 .1278
→ ep4 .1782 (+39%) — consistent with the cosine read that even the hot vm dose fights inv
less than pooled does. d256e200 (slotB) pacing ~11 min/ep, ep11 .3354, kill-horizon ep10
cleared.**

## 200-ep budget arm — d256e200 (D-052; Berker 2026-07-20: slot-B menu presented, "200 ep
## for d256 it is")

**Status: PRE-REGISTERED 2026-07-20, numbers land below the predictions.** The convergence
flag (Berker on the offline read: "the curves were continuing climbing for those runs (no
convergence)" — the E20-T2 pattern) made measurable on the method's own knn-peak cell.

**Cell `in100.floorssl.s0.d256e200`:** d256-verbatim (pooled floor, bn head, lejepa V=4,
bs=128, z_floor=kl, doses 32.8/45/.617, seed 0) + `frame.epochs=200`. Design note: a FRESH
200-ep run with its own cosine (warmup 10 → anneal over 200), NOT a continuation — resuming
the landed d256 past its fully-annealed 100-ep schedule would produce a franken-trajectory
no recipe uses; the budget axis IS the question, so the run owns its schedule. Comparability
declared: its ep100 state ≠ the d256 cell's ep100 (different lr trajectory); the read is
ep200-endpoint vs the 100-ep record (budget-confounded BY DESIGN) + the shape of the offline
curve at the longer budget. Cadence ckpts land at ep50/100/150/200 (frame quarter-points).

**Pre-registered predictions:**
- **P-e200-A (Claude's pick):** more budget converts — ep200 offline above the 100-ep record
  (.6830 lin / .6280 knn_c), monitor slope positive through ep150+ (the E20-T2 slope
  expectation transfers to our lane).
- **P-e200-B:** the 100-ep numbers were the plateau — ep200 within noise of ep100 (or
  down: late-schedule overfit/erosion); the "no convergence" read was monitor optimism.
- **P-e200-C:** budget changes the structure read — z/h battery gradient at ep200 shifts
  (e.g. R@z keeps falling, h effrank keeps rising) even if probes move little: the
  equilibrium is still forming at 100 ep.
Kill criteria standing (probe ≤2× chance @ep≥10; grad incident; ep3→4 fork signature).

**Launch record (2026-07-20 ~16:05):** smoke 62435688 (gpu, 2-ep, tag=d256e200.smoke) →
chain 62435689–694 (6×8h, **h100-slotB**, afterok:smoke + singleton — 200 ep ≈ 37 h at the
measured ~11 min/ep; a link starting after ep200 exits immediately on resume, harmless).
Both H100 slots now occupied: slotA = d256vm (D-051) · slotB = d256e200 (D-052).
**Smoke PASSED (~16:20): ep1 .0750 · ep2 .1290, climbing — and ep1 .0750 EQUALS the landed
d256 cell's ep1 (warmup lr is total-epochs-independent: the e200 run's first ~10 epochs are
the SAME trajectory as d256's on matched arch — a free reproduction check, and the declared
"ep100 not comparable" caveat gets its complement: comparable THROUGH warmup, diverging as
the cosine phases separate).** Chain proceeds on slotB.

## vm3 — symmetric view-mean payment at BOTH taps (D-058; Berker 2026-07-21: "do it as vm3.
## i thought when we were talking about d->32 you meant h level. you cannot justify such
## asymmetry. use one h100 and do imagenet100")

Cell `in100.floorssl.s0.d256vm3` = d256vm2-VERBATIM + `h_floor_batch=view_mean` +
`h_d_slice=32`: the h-floor's input moves from pooled view-CLS (n=N·V=512, d′=128) to
per-image view-mean CLS (n=N=128, d′=32) — the z-side estimator co-design applied
IDENTICALLY at h, with the 32-slice drawn as the first-32 sub-frame of the canonical
128-frame so per-step RNG streams stay aligned with vm2 at matched steps. Everything else
vm2-verbatim (z view-mean d′=32; w_inv 32.8, w_floor 38.7; lejepa V=4; bn head; fast loader
defaults; eval every epoch for curve comparability with vm2).

**Dose rule (pre-declared before the bridge numbers):** h_lamb′ = 0.617 ·
g_h(pooled)/g_h(view_mean) at the vm2 ep25 held state (`e21_vm3pull`, job 62469436; the
vmpull precedent — measured ratio applied outright); w_inv/w_floor unchanged (their terms
are byte-identical across the h axis). **Launch gate:** the default-path regression (new
keys absent ≡ explicit pooled/128, pinned-RNG step) must print PASS.

**Pre-registered predictions (Claude's pick: A):**
- **P-vm3-A** — at the ~6%-share conditioner dose the h payment axis is second-order:
  probes + ep25 anatomy within noise of vm2 (the asymmetry was harmless).
- **P-vm3-B** — symmetry helps h: view-means take the aug noise out of the h-floor's moment
  estimate → cone removal per unit dose improves; h probes ≥ vm2, h-cone ≤ vm2.
- **P-vm3-C** — the h-floor weakens or destabilizes: n=128 estimator noise at h / reduced
  effective demand → the cone persists at h, or early-window instability.

Kill criteria: house incident rule; monitor ≤2×chance at ep≥3 falling; z-var implosion
watch; h_moment_kl runaway watch. Compute: 1 H100 (slotA lane, free post-vm2), 2-ep smoke
gate (NEW code path — the additive h knobs) → 2×8h singleton links; at the tuned loader
≈4 min/ep → ~7 h, lands tonight.

### vm3 bridge record (job 62469436; e21_vm3pull.csv; RAW)

Default-path regression PASS (absent keys ≡ explicit pooled/128, pinned-RNG step). Held-state
h-floor pulls at vm2 ep25: g_h(pooled, d′128) = 0.1967 · g_h(view_mean, d′32) = 0.3585 — the
view-mean h-floor pulls ×1.82 per unit weight (aug-averaged means, stronger trunk gradient).
Pre-declared rule applied: **h_lamb = 0.617 × 0.1967/0.3585 = 0.339** (conditioner share
preserved); w_inv 32.8 / w_floor 38.7 verbatim. Launch: smoke → 2×8h slotA chain.

## vm3 dose arm — d256vm3x2 (D-062; Berker 2026-07-21: "launch one on gpu partition (higher
## dose h/z)", following the z tail-cliff read: vm2's clean-frame z has 97/256 dims below
## 10⁻³·λ₁, slope −3.19 vs pooled 0/256, −0.36 — the sliced view-mean floor under-enforces
## the native spectrum, E12-T5 lesson 3 at the estimator-co-design wall)

Cell `in100.floorssl.s0.d256vm3x2` = vm3-verbatim + BOTH conditioner doses ×2:
**w_inv 32.8 · w_floor 77.4 · h_lamb 0.678** (×2 = the declared conservative first step above
1×; base = vm3 not vm2 — the symmetric payment is the method identity per D-058; the pairing
baseline is vm3-in100 @1× landing tonight, making this a clean single-axis dose contrast).
gpu partition (all three H100s busy), 2×8h afterok-chained links, fast loader, no smoke (no
new code/frame; dose-only on the validated vm3 path) — early-curve watch + standing kills
instead (the E22 pattern). Est. ~7 min/ep on A40-class → ~12 h.

**Pre-registered predictions (Claude's pick: B):**
- **P-x2-A** — the tail lifts AND probes hold: clean-frame z dims<10⁻³·λ₁ falls toward the
  pooled cell's 0 while lin/knn stay ≥ vm3-1× ⇒ "we were regularizing less than we should"
  (Berker's instinct) — the cliff was recoverable for free.
- **P-x2-B** — the tail lifts but probes DROP vs vm3-1× ⇒ the E12-T2 interior optimum on the
  dose axis: the cliff is where the content lives-or-doesn't-care; more spectrum policing
  taxes it.
- **P-x2-C** — early-window collapse: doubling the z-owner share squeezes inv below its
  ownership threshold (the D-048 lesson inverted). Kill criteria standing (monitor ≤2×chance
  ep≥3 falling; z-var implosion; incident rule).

Readout: probes + the spectra instrument (e21_spectra_fig / the tail stats) on x2 vs vm3-1×
vs vm2 vs pooled — the dose-response of the tail cliff.

### vm3x2 fork verdict (ep3–8, 2026-07-21 ~15:20; RAW)

CLEARED: ep3 .1446 → ep8 .2594 rising (single flat step ep5), no incident, ≫ the 2×-chance
kill — P-x2-C did not fire. Matched-epoch anchor: vm2 read .1704/.2798 at ep3/8 — vm3x2
slightly below, same healthy shape. Dose verdict (P-x2-A vs B) waits on the ep100 landing +
the spectra/tail read vs vm3-1×.

## z-only ablation — d256vm3zonly (D-063; Berker 2026-07-21: "launch an ablation on gpu
## partition where we apply our loss only after mlp (not on h). very low priority")

Cell `in100.floorssl.s0.d256vm3zonly` = vm3-VERBATIM with **h_lamb=0**: the loss lives only
after the MLP (inv@z + z-conditioner); the h-conditioner still computes (weight zero) so the
per-step RNG stream stays vm3-aligned — the cleanest single-axis h-ablation. w_inv 32.8 /
w_floor 38.7 verbatim (the h term carried ~1% of total pull; no re-balance). gpu partition,
2×8h afterok links, no smoke (weight-zero axis on the validated path; standing kills).

**Pre-registered predictions (pick: A):** P-zonly-A — h probes drop vs vm3-1× (the ~6%-share
h-conditioner's value, the E20 zoo direction, survives at the vm3 config) · P-zonly-B — no
difference (redundant: the z-conditioner's backflow through the BN head already conditions
h) · P-zonly-C — h improves (the h term was taxing at this config). Readout: probes + the
depth/spectra instruments vs vm3-1× (both land ~tonight/tomorrow).

### floorssl-family guillotine ARMED (Berker 2026-07-21: "when in100 runs land, do the same
### figure between the models: vm3zonly, vm3x2, vm3, vm2, d256, d128 and d64 runs")

Full pipeline session-independent: 4 landed cells extracting now (H100 audit lanes); the 3
training cells' extract chains gated afterany on their final links (vm3 62469564 · vm3x2
62475575 · zonly 62493742) — everything flows to `e21_vm_depth_metrics.csv` and
`results/figures/e21/e21_guillotine_vm.png` (7 cell rows × 8 quantities, zoo2 rules: no gap
station, 0–1 axes, ranks/d + d annotations, per-row gauss). Figure job 62523400 auto-fires
when the last cell lands (~04:00). The rows read as: h-dose triplet (zonly 0× / vm3 1× /
vm3x2 2×) · payment pair (vm2/vm3) · dim-bracket tail (d256/d128/d64).
