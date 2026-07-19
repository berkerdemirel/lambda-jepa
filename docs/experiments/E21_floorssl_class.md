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
