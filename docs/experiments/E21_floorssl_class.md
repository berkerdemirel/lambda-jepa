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
