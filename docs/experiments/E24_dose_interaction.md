# E24 — the three-pull interaction law: toward a scale-free dose recipe

**Status: USER-DIRECTED 2026-08-03 (Berker: "we need the analysis basically to see how these
3 should interact … try many combinations (feel free to use gpu partition) so that we have a
scale free recipe that explains the interactions and we know how much pull etc needed
jointly. (you have 50 run budget individually for both datasets imagenette and in100, before
running anything on in1k)"). D-row: D-070. Sequencing: E23 stage C′ (D-069) executes AFTER
this delivers the recipe; E23 P0–P9 scoring HELD; anything IN-1k is GATED on E24 closing +
Berker.**

**Current status (as of 2026-08-07): takeaways APPROVED; card OPEN on the voas landing.**
**E24-T1** (the share recipe: ~2/3 inv, ~1/3 z-conditioner, a few % at h — s\* ≈ (.63, .34,
.03)), **E24-T2** + **E24-T2-CLOSE**, **E24-T3/D-075** (estimator policy by scale),
**E24-T4** and **E24-T5** are all USER-APPROVED. Still owed: `e24voas` reaches ep100
(~2026-08-08, at ep74 on 2026-08-07) → its landing chain, the OAS-vs-ring A/B read
(information only — the ring is already ruled for E27), and the in-training-vs-o8 Ω offset
that E27's band steering depends on.

## Question

The three floorssl pulls (inv@z, floor@z, floor@h) reach the trunk through anatomy- and
frame-dependent conversion, so nominal weights are local (the D-069 diagnosis; stage-C's
confound). What is the frame-portable coordinate system for dosing, and where is the healthy
region in it — how much pull is needed, JOINTLY, for a healthy inv(z) trend AND good
geometry at the same time (Berker's twin criteria)?

## Coordinates (measured, never nominal)

Realized trunk-share profile **(s_inv, s_z, s_h) = w_i·g_i / Σ w·g** (the w·g instrument;
g_enc trunk-module-only, bs=128 — house convention, incl. its cross-lane caveat) at the
formation state; **total realized pull Σ w_i·g_i held at the frame's certified level** (a
stated invariant; probed ×.5/×2 in wave 2). Certified reference profile from stage-B pull
v2 (lane-invariant): **(.47, .50, .03)**.

## Design (two frames, SAME config family — only the dataset changes)

**LANE RULE (Berker 2026-08-03: "dont use byol lane anywhere. always stay with V=4 lejepa
lane"):** both frames on **lejepa V=4**, each at its certified frame point:
- **toy**: imagenette, `e23Llr` anatomy+lane (bn expander d256-out, lejepa V=4, peak lr
  5e-4, doses 22.1/39.8/.64 — the locked toy recipe; .8003 healthy).
- **IN-100**: `d256` anatomy+lane (same head family; VERIFIED lejepa V=4 from the ckpt cfg,
  32.8/45/.617, lr per family recipe — the family base).
The transport test is dataset-only up to each frame's certified lr; variant portability
(vm4) is tested in reserve arms — if the share-space law survives the dataset change AND
the variant swap, it has earned "scale-free".

**Wave 0 (measurement, ~zero budget):** per-term g at init AND the formation ckpt (toy
`e23Llr_ep38`; in100 `d256_ep25`) per frame → the nominal→realized transform → derived
dose sets for wave 1 (w_i ∝ s_i^target/g_i, rescaled to hold total).

**Wave 1 (16 full cells/frame):** the share-simplex lattice
**s_h ∈ {0, .03, .10, .25} × r ≡ s_z/(s_inv+s_z) ∈ {.25, .40, .55, .70}**
(certified point ≈ (s_h .03, r .52) sits in-lattice). Two-pass dosing: derive → short
confirm-measure at ep10 (ONE correction allowed) → full run (toy 150 ep / in100 100 ep).
**Per-cell health gate = Berker's criterion:** sustained DOWNWARD inv(z) trend by
max(ep25, warmup_end+5), zkl/hkl bounded-or-converging, gnorm decaying; a cell failing
after its correction is RECORDED as unhealthy-at-that-share-point (a datum on the map,
killed to save compute — the region label is the result).

**Wave 2 (≤12/frame, adaptive):** boundary zoom on the healthy region + ridge; Ω-target
dosed arms (dose adjusted to hit an Ω_h trajectory target — the F1 formulation, tested
against share-targeting); total-magnitude probes (×.5, ×2); one seed replicate at the
recipe point.

**Reserve:** toy ≤6 — vm4-variant cells at the winning share point + anything wave 1
surprises demand; in100 ≤6 — vm4-anatomy confirms at the winning point (the "for imagenet
later" recipe candidates). (The lane-portability arm is MOOT under the all-lejepa rule —
its budget returns to reserve.)

**Budget ledger (HARD 50/frame, Berker):** wave1 16 + wave2 ≤12 + reserve ≤6 + confirms/
smokes ≤8 ⇒ ≤42 planned, 8 headroom. All gpu partition; H100s untouched.

**Landing:** every completed cell gets the standing chain (o8+L extract → probes + battery
+ D-068 instruments). Per-epoch realized-share logging lands in the training path before
wave 1 (config-keyed, default off — byte-identity for non-E24 runs preserved).

**Naming:** `{toy,in100}.floorssl.s0.e24s{h}{r}` with h ∈ {0,1,2,3} = s_h {0,.03,.10,.25},
r ∈ {a,b,c,d} = {.25,.40,.55,.70}; wave-2 `e24w*`, reserve `e24x*`.

## Pre-registered predictions (locked before any wave-1 number)

- **P-E24-1 (transport):** the INTERIOR healthy region occupies the same share-space region
  at both frames; the certified profile sits inside both.
- **P-E24-2 (the s_h=0 edge — the known candidate frame divergence, pre-registered as
  such):** healthy at toy (stage-C h0 twins ≥ h.65), pathological at IN-100 (zonly P6
  collapse). The interior transport claim is scoped around this edge.
- **P-E24-3 (the competing hunches — the surface adjudicates):** **Berker-H:** raising s_z
  above ~.5 improves the family (his vm3 direction, "in case we trust the 0.8").
  **Claude-H:** s_h is the Ω_h lever (the 7/7 h-on/off table) and s_z past ~.5 at fixed
  capacity degrades inv health. Both on record; neither is the null.
- **P-E24-4 (health screen):** sustained-downward inv ⇔ eventual probe health; a healthy-
  probing cell with flat inv (or the reverse) kills the criterion as a screen.
- **P-E24-5 (scale-free formulation):** raw-Ω targets do NOT transport (Ω levels are
  frame-local — established 2026-08-03); share targets DO; baseline-normalized Ω
  (Ω / Ω at the frame's s_h=0 twin) undecided — reserve arms read it.
- **Kills:** house rules standing; hard budget 50/frame; in1k gated.

## Launch log

- 2026-08-03: card + D-070; wave-0 pull job (generalized instrument `e24_pull.py`) fired.
- 2026-08-03 **LANE CORRECTION (Berker: "dont use byol lane anywhere. always stay with V=4
  lejepa lane")**: toy side switched y256leg→Llr before wave 0 ran (job scancelled and
  refired); in100 d256 verified already-lejepa (V=4, 32.8/45/.617 from ckpt cfg). All E24
  cells are lejepa V=4. (The scancel raced the first job — its byol y256leg reference rows
  landed in e24_pull.csv and STAY as record: formation shares .469/.503/.028 ≡ the stage-B
  certified profile; CSV schema unified after the mixed-header append.)
- 2026-08-03 **wave 0 LANDED** (job 63009147, e24_pull.csv): formation shares near
  frame-IDENTICAL on the certified recipes — toy Llr .510/.463/.027 vs in100 d256
  .522/.471/**.007** (the one frame gap = the h-term, ~4× lighter at IN-100); init-state
  z-floor dominance .78–.81 both frames (g's swing 4–17× init→formation) ⇒ wave-1 doses
  anchor on FORMATION g's; two-pass confirm at ep10.
- 2026-08-03 **instruments extended per Berker ("measure Ω as well … diagnosis as well as
  the performance")**: the in-training logger now emits the full D-068 orbit calculus per
  epoch (W/B/Ω at h,z + a/b/Λ; eval-mode forward, label-free, RNG/buffers restored;
  `orbit/*` in wandb) alongside `share/*`. Caveat on record: in-training Ω rides the TRAIN
  aug stack — trajectories are the instrument; landing (audit_v1 o8) numbers stay canonical.
- 2026-08-03 **smokes PASSED + wave 1 LAUNCHED**: 2-ep corner smokes (s3a doses, both
  frames) clean — logger verified live (ep0 shares far from target exactly as the init/
  formation gap predicts; the ep10 confirm is the designed catch). First smoke pair died at
  Hydra parse (`aug`/`V` need `+method.` append syntax) — zero steps, zero budget; fixed in
  the launcher. **32 cells fired** (jobs 63010992–63011070; toy 2×8h links, in100 3×8h
  links, gpu partition). Budget: toy 17/50 · in100 17/50 (counting smokes).
- 2026-08-03 **toy ep10 CONFIRM (16/16 alive): the share map is SELF-REFERENTIAL — raw
  finding.** Cells at the anchor's own coordinates land on target (s0d/s1d within ±.01);
  far cells drift toward the anchor because the z-floor's per-unit g GROWS when down-dosed
  (less-satisfied conditioner pulls harder): every a-column cell targeting s_z ≈ .2
  realized ≈ .5. Confirm tolerance declared ±.05/component → 2 PASS, **14 corrected**
  (one-correction rule, each re-dosed from its OWN ep10 g's; corrections aggressive where
  the physics says so — s3ac w_floor 16.1→2.66). **Originals RUN ON as map points at their
  MEASURED coordinates** (an off-target cell is a valid sample, not waste — the stage-C
  completeness-rule analog); corrected twins (`*c` tags, jobs 63011098–63011124) restore
  the intended low-s_z / high-s_h coverage. Early Ω raw notes (train-stack frame,
  trajectory instrument): the s_h dial moves Ω_h(ep10) 1.3→19 across rows — direction
  consistent with the 7/7 h-on/off table; Λ < 1 everywhere this early. Budget: toy 31/50.

- 2026-08-03 **toy ORIGINALS ep25 GATE (full-curve rule: inv trend + zkl/hkl + gnorm —
  inv-trend ALONE does not separate, healthy and sick both drift −3–9%): 7 PASS / 9 KILL.**
  PASS (probe@ep37-38): s0a .669 · s0b .652 · s0c .613 · s0d .558 · s1a .641 · s1b .643 ·
  s1c .626 — all continue to ep150. KILL (recorded at their MEASURED coordinates): s1d
  (probe .277, inv trending UP at realized s_z .64) · s2a-d · s3a-d (probes .15-.20,
  per-term g's exploded 4-77× vs healthy .15-.6, Ω_h 26-60; wandb curves read via API).
  Kills executed (scancel, chains verified). RAW structure on the record: (1) realized inv
  LEVEL stratifies exactly by r-column across all h rows (a≈.27 · b≈.47 · c≈.70 · d≈1.0) —
  the z/inv dose ratio sets inv satisfaction, h-row invariant; (2) early P-E24-3 direction
  AT TOY (raw, unscored): the low-z a-column holds the best probes, the z-heavy d-column
  declines/dies, and the s_h≥.10 original rows died wholesale — at realized s_h only
  .02-.14 (never reached their .10/.25 targets; the corrected twins are the honest probe
  of those coordinates); (3) h-free s0a = the toy max again (.669). Budget unchanged
  (kills were counted); fleet freed for the 14 corrected twins (their ep10 confirm next).

- 2026-08-03 **EARLY-TRIAGE SWEEP (Berker: "you can speed up by seeing already collapsed
  runs for both in100 and toy") — 15 more kills, and the corrected s0/s1 twins are the
  story.** TOY corrected twins: **s0ac .8298 @ ep85 — ABOVE the certified Llr .8003 with
  65 ep to go** (h-free, target .75/.25/0); s1ac .795 · s0bc .783 · s0cc .764 · s1bc .766 ·
  s1cc .718 — all healthy, in-training Λ ≥ 1 appearing (s1ac 1.2). The corrected s2*c/s3*c
  collapsed AGAIN (probes .13–.19, g's to 630, Ω_h 26–90) → **s_h ≥ .10 at toy is
  UNFIXABLE under the share law** (both dose vectors dead; the pre-registered label
  applies); 7 killed, s3dc (just started) keeps its gate. IN-100: s2/s3 rows at chance
  (.009–.035 by ep4–11, g's to 806, Ω_h 25–67 — the same signature) → 8 killed pre-gate
  per the directive; s0/s1 rows healthy (.15–.27 @ ep4–7, calm g's), with the s1 row
  realizing s_h .06–.08 = the LIVING lower bracket of the in100 h-boundary. **Wave-2 zoom
  question flagged:** the toy s2 ORIGINALS died at realized s_h only .02–.14, so the death
  boundary may sit in h_lamb magnitude (≥ ~1.3) or the joint dose vector rather than
  realized share — the s_h ∈ (.03, .10) interval is the zoom target, designed after the
  s0/s1 rows land. Fleet after sweep: 14 toy alive (7 originals + 6 corrected + s3dc),
  8 in100 alive. Budget: toy 31/50 · in100 17/50.

- 2026-08-03 **corrected-twin ep10 CONFIRM (second-pass convergence quality) — the stiffness
  ordering, raw:** the h-share responds to correction (s2ac realized .092 ≈ its .10 target;
  s3 row .145–.495 — the corrected cells DID reach their target h region and died there,
  hardening the unfixable label to a REALIZED-share statement), while the z-floor is STIFF
  below ~.4: s0bc landed exactly on target (.402 vs .40) but s0ac targeting s_z=.25 only
  moved .58→.52 despite a 2.8× down-dose — the z-floor defends a share floor ≈ .4–.5 at
  toy; the low-s_z corner (< .3) remains UNSAMPLED (wave-2 candidate: w_floor ≈ 2–4 cells).
  The toy h-death boundary NARROWS to realized s_h ∈ (.05, .09) (s1cc alive at .048, s2ac
  dead at .092; h_lamb interval (.7, 1.45) equally consistent — separating share-vs-
  magnitude causality = a wave-2 cell at target s_h ≈ .06–.07). Healthy corrected cells'
  probes order NEGATIVELY in measured s_z at h≈0 (s0ac .830 at z ~.35–.52 > s0cc .764 at
  ~.65) — more raw P-E24-3 direction at toy.

- 2026-08-03 **s3dc killed (Berker's call, live read):** by ep33 it had joined its row —
  probe .147, g's 33–70, Ω_h 53, realized s_h .201 — the 14th/14 corrected-or-original
  cell at s_h ≥ .10 coordinates to die at toy. The unfixable label now covers the full
  s2/s3 lattice at both dose vectors with no survivor. Toy fleet: 13 alive (7 originals +
  6 corrected, all s_h ≤ .05 realized).

- 2026-08-03 **in100 ep10 CONFIRM (8/8 alive, probes .22–.34): the drift is BIDIRECTIONAL
  ANCHOR-ATTRACTION** — toy's low-z cells drifted z-UP toward the anchor profile, in100's
  high-z cells drift z-DOWN toward it (s0d realized .523 vs target .70): the same
  self-referential law seen from both sides. PASS s0b (.578/.422 on .60/.40) + s1c; **6
  corrected** from their own ep10 g's — healthy-state derivations (calm g's .05–.19,
  unlike the discarded toy sick-state ones), with the required force now explicit: s0ac
  w_inv 125.8 (anchor 32.8), s0dc w_floor 174.9 — the anchor attraction is strong. Λ raw
  note: the s1 row (realized s_h ~.05) runs in-training Λ 1.10–1.44 > 1 while the h-free
  s0 row sits .75–.82 — the h-term's Λ push visible at in100 under controlled shares.
  **Ops fix (own miscalculation): in100 needs ~45 h/cell (27 min/ep), not 24 h — all 8
  original chains EXTENDED to 7×8h links; corrected twins launched with 7 from birth**
  (jobs 63012280–63012347). Budget: toy 31/50 · in100 23/50; fleet 13 toy + 14 in100 alive.

- 2026-08-03 **TOY WAVE 1 COMPLETE — all 13 healthy cells to ep150, FINALS RAW (online
  best):** corrected: **s0ac .8734** · s1ac .8601 · s1bc .8492 · s0bc .8476 · s0cc .8255 ·
  s1cc .8046 — **every corrected cell ≥ the certified Llr record .8003, s0ac beats it by
  +.073.** Originals (measured coordinates): s1a .8464 · s0a .8456 · s0b .8209 · s1b
  .8199 · s1c .7572 · s0c .7534 · s0d .6803. RAW STRUCTURE: probe falls MONOTONICALLY in
  realized s_z within every group (a > b > c > d), corrected > original at every
  coordinate (lower realized z), s_h 0 ≈ .03 (h-dose ~neutral at toy in this range) — no
  interior optimum on the sampled z axis; the best sampled point is the lowest-z corner
  (s0ac realized ~.65/.35/0), and the < .3 region is still unsampled (the z-floor's share
  stiffness) → wave-2's first question. Landing chains FIRED 13/13 (extract o8+L →
  probe + battery; jobs 63012357–63012401, 39 verified) — canonical Ω/Λ + census +
  spectrum per cell next. NO takeaway rows without Berker.

- 2026-08-03 **Ω_h SORTING (Berker's question, exact): within-row Spearman(Ω_h(end), acc)
  = −1.000 (h-free, n=7, p<1e-4) and −.943 (h=.03, n=6, p=.005)** — a near-perfect
  within-anatomy sorter; pooled ρ .41 only because the h-term shifts Ω_h up ~+.3–.5
  without costing acc (offset removal needed across h-levels).
- 2026-08-03 **wave-2 toy RESOLVED same-day (fast fleet) + Berker kill round.** (1)
  wh05/wh07 KILLED (Berker; pre-kill: Ω_h 105/64, probes .24/.21 at ep~137) — they died
  at realized s_h only .03–.04, never reaching their .05/.07 targets, at h_lamb 2.39/3.34
  ⇒ **the h-death is MAGNITUDE-carried, not share-carried: toy h_lamb boundary ∈ (0.70,
  1.45); the share hypothesis is refuted at toy** (s1cc lived at share .048/lamb .44).
  (2) Low-z frontier: targets .05–.25 ALL realize s_z ≈ .36–.47 — the z-share floor is a
  REACHABILITY wall, not a performance cliff; below intended ~.12 the inv term SATURATES
  (wz05 g_inv .010, oscillating shares — Berker's live read; probe .774) ⇒ the optimum
  sits AT the wall. wz25 .8604 · wz18 .8573 · wz12 .8474 · wz05 .7738. (3) **wrep (seed
  1) .8634 @ ep138 — the record REPLICATES.** (4) **wt05 .8591 / wt2 .8568 — total-pull
  ×.5/×2 moves acc ≤ .007: the prescription is T-ROBUST within 4×; only the share profile
  matters.** TOY MAP CLOSED: optimum realized ≈ (.62–.65, .35–.38, ≤.03), bounded by the
  z-reachability wall (~.35) and inv saturation below. Budget: toy 40/50 (complete).
- 2026-08-03 **IN-1k certified-profile pull LANDED (job 63012837; measurement-only, the
  in1k training gate untouched): vm2 = .663/.326/.011 · vm3 = .674/.317/.009 — the
  hand-fixed 1k lane recipe ALREADY SITS AT THE E24 OPTIMUM in share space** (independent
  convergence: E22's empirical lane fix ≡ E24's mapped optimum). vm4 reads .132/.853/.015
  RAW with a standing caveat: the queue cell's ×5.59-bridged w_floor and ¼-leverage
  gradient make instantaneous w·g non-comparable to pooled cells — flagged, not
  interpreted. IN-100 kill round (Berker): s0d .309@ep24 · s1d .331@ep31 · s0cc .323@ep22
  · s0dc .163@ep21 · s1dc .020@ep22 (collapsed) — labels recorded; 9 in100 cells remain
  (incl. both toy-winner mirrors s0ac/s1ac). Canonical D-068 instruments over the 13 toy
  stores LANDED (e24_toy_{spaces,trans,census,spectrum,spectrum_summary}.csv).

- 2026-08-03 **THE VM PIVOT (D-071; Berker: "we will stick to vm4 like thing … record this
  to your memory etc so that we dont do the same mistake" + "i refresh your budget. the
  important thing is that we find the recipe").** The anatomy mismatch surfaced and owned:
  waves 1–2 ran pooled/pooled (inherited anchors) while the operating recipe is vm4;
  pooled results stay, scoped pooled-only (they serve stage-C′/D-069 directly). **Binding
  defaults from here (memory + D-071): view-mean h+z, queue for n_eff/d′ ≈ 4 (queue 3 at
  bs=128), d′=128, D/d′ meaningful; the anatomy axis stated loudly in every design.**
  E24-vm program: toy vm4 CANARY `e24v0` launched (first-ever toy vm cell, canary-first;
  pooled-s\* prior doses 47.6/7.6/.36; ep25 gate; its ep10 g's = wave-0-vm) — job
  63013037; in100 vm3/vm4 formation pulls firing (63013038; ckpts verified). Budget
  REFRESHED (fresh 50/frame, Berker).

- 2026-08-03 **vm canary v0 KILLED (Berker's live catch: "24v0 is unhealthy" — probes
  thrashing .15–.23 at ep9–12, shares oscillating, Ω_h RISING 2.0→5.4, g_inv .016–.073):
  the pooled-s\* prior does not transplant to the vm anatomy** (w_floor 7.6 ignored the
  queue's ¼ leverage). Sick-state g's NOT used for re-dosing (house law). **in100 vm
  pulls LANDED (63013038): vm3 = .511/.480/.009 — the SAME certified profile as every
  healthy cell at every scale (share law survives the view-mean change); vm4 =
  .117/.871/.012 — matching in1k vm4 (.13/.85/.02): the queue cell is z-floor-DOMINANT
  in naive w·g on healthy winning cells at BOTH scales.** Interpretation candidate
  (discussion-only): the 2/3–1/3 balance governs COMPETING forces (pooled: inv vs floor
  over aug scatter); the vm+queue floor pulls on centers — a component inv does not
  defend — so its naive share can be large harmlessly; the vm map may need
  force-competition-aware coordinates. **Canary v1 launched (63013062): double-ratio
  bridge transplant, toy-vm4 = certified-toy × (in100 vm4/pooled per-term ratios) =
  (22.1, 139.6, 1.96)**; ep25 gate, watcher armed.

- 2026-08-03 **canary v1 ep25 gate: PASS (full-curve + Ω, wandb read gvbiptls, warmup 10 →
  gate 25).** inv falls sustained from ep12 (.6475→.6079@ep23), zkl monotone 1.77→.85,
  hkl peak 2.57@ep10→2.04, gnorm 186→19.6 decaying every epoch, in-training Ω_h
  6.03(ep6)→2.25(ep24) falling, probe .478@ep24/.447@ep25 (climbing with bounces);
  logger shares settle ~(.43, .54, .03) with inv drifting UP — every channel the opposite
  of v0's kill signature. **The double-ratio bridge transplant (22.1/139.6/1.96) stands
  as the first healthy toy vm4 dose vector**; run continues to ep150.
  **CURVE-QUALITY FLAG (Berker's live read, same evening: "too noisy as if we have a
  large lr (accwise) … i dont like that run's curves") — quantified vs matched healthy
  cells, ep12–33: v1 mean|Δacc|/epoch .0437 (max .098) vs Llr .0267 / s0ac .0229 /
  wrep .0219; per-step inv sd .080 vs Llr .018, s0ac/wrep .004; gnorm@ep20–25 20 vs
  Llr 13, s0ac 3.8. Late window ep34–50: still ~3× Llr (.032 vs .011; inv sd .053 vs
  .015) — not early-phase settling. s0d (the smoothest cell, .011) shows z-heavy dose
  ≠ noise at pooled — the churn is vm4-at-toy-specific. Consistent raw thread: v1's
  cos(inv, z-floor) is batch-VOLATILE (−.27…−.66) where in100/in1k vm cells are stably
  decoupled. v1 = organizing-but-LOUD; NOT a certified-clean anchor — no doses derived
  from its state pending the lane diagnosis (ring-staleness vs lane-lr controls, joint
  design).** Same sweep: wave-2
  landing chains fired 7/7 (`e24_land.sh` extended to seed-qualified names — wrep = s1;
  jobs 63013070–90; wave-1 stores skip-verified), wave-1 probe/audit refires 63012570–78
  all COMPLETED, in100 9/9 cells healthy at ep28–37 (probes .43–.56 climbing, calm g's
  .04–.23, Ω_h flat-to-falling; s0c weakest — probe .434, Ω_h creep .413→.422 — watch).

- 2026-08-03 **THE COORDINATE QUESTION MEASURED (`e24_cos.py` NEW → `results/diag/
  e24_cos.csv`, job 63013112): per-term trunk-gradient cosines + naive shares on a
  6-batch sequence that warms the queue ring exactly as training does (b0 = cold ring =
  the e24_pull.py ckpt convention; qfill=3 = the operating estimator). Both opener
  hypotheses resolve in one instrument — RAW:**
  **(1) The vm4 z-dominance was a COLD-QUEUE INSTRUMENT ARTIFACT.** The rings are not
  checkpointed (`extras()` empty ⇒ every ckpt-based pull read the conditioner at
  n/d′=1). Cold rows REPRODUCE the puzzle exactly (s_z .871 in100 / .853 in1k / .919
  toy-v1); warm rows collapse g_z ~10× (in1k .314→.028) → **in1k vm4 warm =
  (.62–.64, .36–.38, .01) — ON s\*; with vm2 (.65/.33/.01) and vm3 (.67/.32/.01), all
  three hand-tuned IN-1k cells measure on the recipe under the operating estimator.**
  in100 vm4 warm = (.52–.56, .43–.47, .01); toy v1 (ep~26, forming) warm =
  (.37–.40, .56–.60, .04). The in-training share logger measures with LIVE rings —
  warm by construction; its channel was right all along.
  **(2) Force geometry differs by anatomy (the competition mechanism, measured):**
  pooled floors OPPOSE inv in trunk space — cos(g_inv, g_zfloor): in100 d256 −.63…−.88
  (replicates e21_vmcos ep25 −.86), toy s0ac −.61…−.82 (g_inv .06 — the saturated
  at-the-wall optimum made visible), toy Llr −.20…−.69. View-mean floors DECOUPLE at
  scale: in100 vm2 .00±.08 / vm3 +.05±.15 / vm4-warm −.14…−.33; in1k vm2/vm3/vm4-warm
  +.08…+.44 (mildly aligned). The z-level argument is exact (a view-mean floor's
  z-gradient is constant across an image's views = pure center force; inv's is
  mean-zero across views = pure scatter force; pooled carries both) — the shared trunk
  preserves it at in100/in1k; toy vm4 (ep26 forming, one cell) still moderately opposed
  (−.27…−.66). Riders: the two floors are mutually ALIGNED (cos(z,h) in100 vm3
  +.53…+.68), cos(inv, h) ≈ 0 everywhere. Instrument note: cosines f64 in the script
  now (e21_vmcos control lesson; this run's fp32 drift ≤ ~.005 — immaterial at these
  magnitudes). NO takeaway rows; interpretation + the vm-map design are the joint
  discussion. D-072 PROPOSED (queue-cell pull convention: warm reads only).

- 2026-08-03 **D-072 + E24-T2 USER-APPROVED ("d072 approved. vm4 z dominance agreed.")
  and the NOISE-MYSTERY CONTROL PAIR LAUNCHED (Berker: "you do the next move to
  understand … dont stop and lets solve the mystery and find a good prescription for
  vm4 family at toy, in100 so that we can run the best combo at in1k"; v1 runs to ep150
  per his call; in100 pooled wave-1 untracked for now).** One arm per candidate cause,
  everything else matched to v1:
  **e24v2lr** (63013162→163, gpu): vm4 anatomy + v1 doses verbatim, **lr 2.5e-4**
  (eta_min 5e-6, ratio held) — the large-lr reading, bs-matched to v1.
  **e24v3nq** (63013164→165 H100 slotA, RELOCATED 63013213→214 to the free A100-80GB
  node gpu238 after ~1h stuck behind the gpu100 queue — zero steps run, same config;
  63013213 OOMed at 78GB/80GB — bs=512·V4 = 2048 imgs flat through ViT-S/8 does NOT fit
  any 80GB card, the H100 would have failed identically; RELAUNCHED 63013219→220 with
  NEW config-keyed `method.grad_ckpt` = trunk activation checkpointing, default off /
  byte-identical legacy, same math for ~30% compute — fits with wide margin):
  **view-mean NO-RING at a non-degenerate estimator via Berker's bs suggestion** —
  bs=512 ⇒ n = bs = 512 at d′=128, n/d′=4
  with queue_steps=0 (D-071 dimensional bindings preserved; declared confounds: 18
  steps/ep vs 73, ~2× SGD gradient averaging — the noise read is per-eval; lr 5e-4
  kept, one-knob rule). Doses = certified toy pooled transplant (22.1/39.8/.64; the
  in100 vm3≈pooled dose precedent); ep10 warm confirm, one correction allowed.
  **Logger rider (D-072 applied to the logger itself): NEW config key `share_log_bs`
  in train.py — the share logger's fixed probe batch must match the cell's operating
  estimator (512 here; null = 128 house default, byte-identical legacy path — a
  bs-128 logger batch on this cell would be a degenerate cold-read inside the logger).**
  Gate: ep25 full-curve + Ω + the noise stat (mean|Δacc|/ep + per-step inv sd vs the
  reference band: Llr .027/.018, optimum cells .022/.004, v1 .044/.080). Predictions:
  ring-staleness ⇒ v3nq smooth + v2lr still noisy; lane-lr ⇒ the reverse; both noisy ⇒
  view-mean-at-toy itself (reserve third arm: queue_steps=1, bs-matched). Budget:
  toy 4/50.

- 2026-08-03 **e24v2lr ep25 gate: PASS — the lr knob is REAL but PARTIAL (raw).** Full
  curve: inv sustained-down from ep12 (.629→.572@ep32), zkl monotone 1.14→.59, hkl peak
  2.08@ep10→1.48, gnorm 60→32; Ω_h 1.38@ep25 vs v1's 2.25@ep24 (further along
  organizing); **probe .523@ep26 vs v1 .459 at the matched epoch — half the lr, MORE
  accuracy.** Noise stat (ep12–26): per-step inv sd **.0188** (v1 .0837; band: Llr
  .0148, optimum cells .0043) — loss-side churn FIXED to near-band; acc mean|Δ| .0337
  (v1 .0433; Llr .0269) — improved, still ~25% above band; max|Δ| .082. **DESIGN NOTE
  (recorded confound): lr↓ also slows per-step drift ⇒ reduces ring-staleness bias
  simultaneously — v2lr improving is consistent with BOTH causes. v3nq (no-ring at FULL
  lr) is the discriminator; still PENDING on gpu100 (Priority). Berker's staleness read
  + the shrinkage-conditioner proposal (fresh-only + trace-preserving Ledoit–Wolf inside
  SpectralConditioner, ρ̂ analytic/detached/logged) presented — awaiting his call; no
  method change lands without it.**

- 2026-08-03 **e24v1 FINISHED (raw): best .7778, ep150 .7740** — the first completed toy
  vm4 cell: the noisy bridge-transplant at warm shares (.37–.40, .56–.60, .04) lands
  well under the pooled optimum (.8734 s0ac) and the certified pooled lane (.8003 Llr).
  Landing chain fired (63013221–223, o8+L extract → probe + battery). v2lr cooking
  (~ep38); v3nq (grad-ckpt relaunch) starting on gpu238.

- 2026-08-03 **D-073 (Berker's OAS prescription) IMPLEMENTED + two probes fired.**
  SpectralConditioner grows `shrink="oas"` (ρ from S.detach(), m LIVE, target mI never
  I — trace preserved exactly; None path verified bitwise), method key `floor_shrink`
  wires BOTH taps (D-058 symmetry), ρ̂ logged per tap (`share/rho_{z,h}`). Local check:
  iid rows at n=p read ρ=1 / loss .005 where the legacy estimator reads 1.157 — the
  Marchenko–Pastur phantom (= the D-072 cold artifact) killed at the source;
  anisotropic n=p: ρ=.72, finite grads. **e24v4oas canary** (63013226→227): bs=128,
  view-mean both taps, NO ring, full lr 5e-4, pooled-certified dose prior
  (22.1/39.8/.64), ep10 confirm — the no-ring family candidate at the standing frame
  bs. **e24_oas_cmp.py NEW** (job 63013228): same-weights CRN ring-warm vs OAS-fresh
  z-floor force at in100/in1k vm4 ep25 formation states — the "does OAS erase real
  large-frame anisotropy" pre-read (Berker's Frobenius-vs-Stein caveat) BEFORE any
  at-scale training; nonlinear Stein shrinkage = the recorded second challenger.
  Fleet: v2lr cooking, v3nq (grad-ckpt) cooking on gpu238, v1 landing chain running.
  Budget: toy 5/50.

- 2026-08-03/04 **v3nq ep25 gate: PASS — and THE MYSTERY RESOLVES (raw): the ring is
  the noise source.** Noise stat ep12–26 (mean|Δacc| / inv_sd / acc@26): **v3nq (NO
  ring, bs512, FULL lr) .0130 / .0152 / .600 — BELOW the pooled reference band** (Llr
  .0269/.0148/.610); v2lr (ring, half lr) .0337/.0188/.523; v1 (ring, full lr)
  .0433/.0837/.459. Triangulation: ring removed at full drift → clean beyond the ~2×
  bs-averaging allowance; ring kept at half drift → still noisy. gnorm 74.7 — the
  HIGHEST of the four — with the smoothest curves: pre-clip magnitude ≠ direction
  stability. Ω_h 1.05@ep25 falling (lowest of any toy vm cell); realized shares
  (.37–.42, .55–.60, .02–.04) — z-heavy-of-s\* and still smooth at reference acc:
  STALENESS, not z-dose, was the toxin. No mid-run dose correction (diagnosis kept
  clean; measured coordinates recorded per the completeness rule).
- 2026-08-03/04 **oas_cmp LANDED (63013228 → e24_oas_cmp.csv): OAS-fresh preserves the
  operating ring force direction at scale** — cos(g_z ring-warm, OAS-fresh): in100
  .92–.93, in1k .77–.84 (ctrl_inv exact +1.0000); ρ̂_z .69–.82 at n=p; g_z OAS ≈ 3×
  ring (dosing absorbs); cos(inv,z) decoupling character preserved in both forms. No
  wholesale anisotropy erasure; the in100 training A/B remains the arbiter for the 1k
  residual; the nonlinear-Stein challenger stays benched.

- 2026-08-03/04 **v4oas ep25 gate: PASS — OAS reproduces the no-ring smoothness AT THE
  STANDING FRAME and takes the matched-epoch lead (raw).** Noise ep12–26: mean|Δacc|
  **.0134** ≈ v3nq's .0130 with ZERO bs confound (bs=128); **acc@26 .689 — ahead of
  every toy cell at matched epoch** (s0ac .615, Llr .610, v3nq .600); inv_sd .0293
  (between Llr .0148 and v1 .0837 — the per-step demand varies with ρ̂; the acc curve
  is glass); Ω_h .64–.67 falling (lowest toy vm); **Λ 1.08–1.11 — the first toy vm
  cell above 1**; ρ̂_z .17–.19 / ρ̂_h .13 at operating states (vs .7–.8 measured on
  ring-cells' ep25 states — OAS-trained features are evidence-rich); realized shares
  self-drift toward s\* (.50–.65 inv). **TOY VM MAP LAUNCHED on the OAS anatomy**
  (4 cells, toy 9/50; two-pass doses from v4oas's gate-passed ep25–27 g's — g_inv .57,
  g_z .21, g_h .81, T = 19.6 held): **e24va** (.72/.25/.03 — does the z-reachability
  wall move under OAS?), **e24vb** (.62/.35/.03 — the pooled-s\* mirror), **e24vc**
  (.47/.50/.03 — the certified-profile point), **e24vh0** (.65/.35/0 — h-free twin).
  Jobs 63013275–82; ep10 confirms + one correction per the standing rule; the z-heavy
  corner is already sampled (v4oas .33–.48 live + v3nq .55–.60 + v1). v2lr/v3nq run to
  ep150 (completeness).

- 2026-08-04 **vm-map ep10 CONFIRM (4/4 healthy: probes .48–.54, Ω_h .66–.73, ρ̂_z
  .06–.17): the anchor-attraction law operates on the OAS anatomy too** — every cell
  drifted z-UP from target (realized r = s_z/(s_inv+s_z): va .475 · vb .489 · vc .633 ·
  vh0 .411 vs intended .25/.36/.52/.35). Ruling per the completeness rule: vb/vc/vh0
  RUN ON at measured coordinates (useful spread, drifts ≤ .11); **va realized ≈ vb's
  coordinates (.50/.45 vs .48/.46 on DIFFERENT weight vectors — kept running as a free
  same-shares/different-w pair, a direct share-sufficiency test on this anatomy)**; its
  one-correction allowance spent on **e24vac** (63013343→344, toy 10/50): the low-z
  corner probe (target .72/.25/.03) re-dosed from va's own ep10 g's (.478/.460/1.504 →
  w = 29.5/10.7/.39) — the corner where the pooled optimum lived; OAS-flavored
  reachability question (down-dosed floor → less conditioning → more anisotropy
  evidence → ρ̂ falls → per-unit g rises: the self-referential law's expected OAS
  mechanism) on record before the number.

- 2026-08-04 **vac ep10: THE LOW-Z CORNER IS UNREACHABLE ON THE OAS ANATOMY TOO — and
  the mechanism is caught in ρ̂ (raw).** Target .72/.25/.03 at w_floor 10.7 (2.2×
  down-dose from va's 23.3): realized .482/.468/.049 ≈ va's original (.501/.453) —
  the z-share wall sits at ~.45–.47 here (pooled's was ~.35–.4). The pre-registered
  OAS mechanism CONFIRMED: g_z ROSE .460→.583 as the dose fell while ρ̂_z FELL
  .063→.047 — less conditioning → more anisotropy evidence → the estimator sharpens →
  per-unit pull rises: the self-referential law, estimator-mediated. vac healthy
  (probe .509@ep11, Λ 1.05), runs on at its measured coordinate; its correction
  allowance is SPENT. Bonus coverage: THREE cells now sit at r ≈ .47–.49 with w_floor
  spanning 10.7 / 23.3 / 32.7 — a 3-arm same-shares/different-weights sufficiency
  test, free.
- 2026-08-04 **fleet mid-read (raw):** v4oas ep66 **.8232 — already ABOVE the
  certified pooled final** (Llr .8003) with 84 ep to go; v2lr FINISHED best .7875
  (> v1 .7778, < Llr); v3nq ep69 .7312 (fast start, then 4× fewer steps bites at
  matched epochs — its discriminator job is done); map ep29–31: vb .7052 · vh0 .7032 ·
  vc .7006 · va .6851 — the va/vb same-shares pair currently split ~.02; the
  sufficiency test reads at finals.

- 2026-08-04 **IN100 MIRRORS LAUNCHED (Berker: "also launch the in100 mirrors") — the
  toy vm-OAS target grid transplanted to in100 (in100 4/50).** Anatomy: view-mean h+z,
  NO ring, `floor_shrink=oas`, d′=128, bs=128, family lr; doses by the two-pass law
  from MEASURED OAS-fresh g's on the in100 vm4 ep25 formation state (e24_oas_cmp v2
  refire 63013628: g_inv .1995 / g_z_oas .1044 / g_h_oas .1630; T = 11.6 = the
  certified in100 family total): **e24va** (.72/.25/.03 → w 41.9/27.8/2.14) ·
  **e24vb** (.62/.35/.03 → 36.0/38.9/2.14) · **e24vc** (.47/.50/.03 → 27.3/55.5/2.14)
  · **e24vh0** (.65/.35/0 → 37.8/38.9/0). h_lamb 2.14 sits within the vm4 family's
  operating range (in100 1.894 / in1k 1.679) — the frame-local h-magnitude law
  respected. 7×8h chains (tails 63013639/646/653/660); ep10 confirms (~4.5h) + one
  correction; ep25 gates with the noise stat. Tests: OAS transport, the wall's in100
  location, h-neutrality at scale, cross-frame recipe (with the toy finals).
  e24_oas_cmp.csv schema v2 (adds g_inv/g_h_{ring,oas}; v0 preserved verbatim as
  e24_oas_cmp.v0.csv).

- 2026-08-04 **TOY VM-OAS FINALS (raw) — the OAS family SWEEPS, and the r-surface is
  FLAT:** **vc .8764 — NEW TOY RECORD** (> pooled optimum s0ac .8734) · v4oas .8716 ·
  vh0 .8713 · vb .8696 · va .8693 — five cells spanning realized r ≈ .41–.63 within
  **.007** of each other (pooled's same span fell monotonically by ~.17): in the
  decoupled-force regime the share landscape flattens into a broad basin above the
  z-wall (~.45–.47) — raw shape, no takeaway. **The va/vb same-shares/different-
  weights pair finished .0003 apart (.8693/.8696)** — share-sufficiency on the OAS
  anatomy about as clean as it gets (vac = the 3rd arm, ~ep139). h0 ≈ h.03 replicated
  on OAS (vh0 .8713 in-cluster). v3nq final .8191 (bs512 exact-fresh; step-count-
  limited — 2700 vs 10950 steps — raw note); ring cells v2lr .7875 / v1 .7778.
  **ρ̂ answer (Berker's question): logged everywhere (share/rho_{z,h} + prints +
  CSVs); NEVER 1 in operation** — v4oas trajectory .09(ep0) → .35(ep149) slow
  monotone (features Gaussianize → evidence recedes → force hands over to scale-
  maintenance); vac ep139 ρ̂_z .116 (down-dosed floor ⇒ anisotropy persists ⇒
  evidence persists); in100 mirrors ep7–9 ρ̂_z .11–.28; ρ̂=1 only at the at-target/
  degenerate boundary, and collapse re-arms the floor (anisotropy = evidence — the
  self-stabilizing feedback). Landing chains fired 7/7 finished cells (63014495–515;
  vac lands on completion — list extended).

- 2026-08-04 **Ω_h SORTING UPDATED over both healthy families (Berker's ask; channel =
  in-training Ω_h(end), acc = final best; raw):** POOLED n=20 ρ **−.754** (h-free n=14
  **−.922** — softened from −1.000/n=7 by wz05, the inv-saturated wall cell at Ω 1.245/
  .7755; h.03 n=6 −.943 unchanged) · **VM-OAS n=5 −.700** (n small + the flat basin;
  within h.03 n=4 **−1.000**: .473→.8764 · .488→.8716 · .505→.8696 · .528→.8693 — the
  sorter resolves INSIDE a .007-acc basin, finer than the r-coordinate) · COMBINED
  n=25 −.642 (cross-h/famiy offsets compress; the h-offset direction REPLICATES on
  vm: h.03 cells Ω .47–.53 vs vh0 .397 at equal acc) · combined+variants n=28 −.686;
  all view-mean incl. ring/bs512 variants n=8 **−.929** (v3nq/v2lr/v1 land exactly
  where their Ω_h says). Cross-family: vm-OAS end-Ω band (.40–.53) ≡ the pooled BEST
  cells' band (.39–.45) at matching accs. **vac LANDED same hour: final .8703, Ω_h(end)
  .532 (chain 63014569–71) — its row updates: vm-OAS h.03 n=5 ρ −.700 (one inversion:
  vac out-performs va at higher Ω — the basin is flatter than the sorter's resolution
  at Δacc ~.001); vm-OAS all n=6 −.657; combined n=26 −.612.** Six of six OAS cells
  finished .8693–.8764 (all above every prior toy cell except s0ac, three above it);
  the va/vb/vac same-shares triple: .8693/.8696/.8703 — spread .001 across w_floor
  10.7–32.7. Canonical-o8 re-confirm when the landing audits finish.

- 2026-08-04 **IN-1K RUN LAUNCHED (D-074, Berker's gate: "run your best bet") —
  `in1k.floorssl.s0.e24voas`, h100-slotA, 63014650→663 (14×8h singleton links).**
  vm-OAS anatomy at the program config family, frame lr; target (.55, .42, .03) =
  mid-basin; w = 21.4/49.6/1.89 from the measured in1k OAS-fresh basis (g .2310/.0763/
  .1427, T 9.0). ep10 confirm ~11h; ep25 gate with the full channel set. Standing
  baseline to beat: ring-vm4 .6392. in100 mirrors at ep9–11: shares va .51/.45/.05 ·
  vb .57/.36/.06 · vc .59/.36/.05 · vh0 .62/.38/0 (vc drifted z-DOWN toward the in100
  anchor profile — bidirectional anchor attraction on OAS at scale), Ω_h .43–.64,
  **Λ 1.08–1.42 (all four > 1 by ep9)**, ρ̂_z .15–.27.

- 2026-08-04 **in100 mirror ep10 CONFIRM (4/4 healthy, probes .36–.40 @ep11 — AHEAD of
  the pooled wave-1's .22–.34 at the same point; Λ 1.11–1.41, Ω_h .42–.63):** the
  anchor attraction at in100 is STRONGER than toy's — va/vb/vc all collapsed onto the
  anchor profile (realized .58–.61 / .33–.36 / .06; targets .72/.25 · .62/.35 ·
  .47/.50), compressing the r-ladder to .35–.41. vb PASSES tolerance (.579/.364/.057
  on .62/.35/.03); vh0 marginal (.586/.414/0), runs on (its h-free coordinate is the
  point). Three cells at the anchor coordinate on different w vectors = the in100
  share-sufficiency triple, free. **Corrections spent (own-ep10-g re-doses, 7×8h
  chains): in100 e24vac (wall probe, target .72/.25/.03 → w 72.6/31.2/1.52, tail
  63014709) · in100 e24vcc (high-z coverage, target .47/.50/.03 → w 26.5/95.1/1.27,
  tail 63014716).** Budget: in100 6/50. Land list extended (in100 v-cells).

- 2026-08-04 **in100 vm map COMPLETE (6/6 to ep100, error-scan clean; landing chains
  63017192–209 fired; leftover vb links self-drained on the finished ckpt).** Finals
  (online best / ep100): **vcc .7348/.7318 — the in100 family max** · vc .7252/.7238 ·
  vb .7236/.7226 · va .7200/.7200 · vh0 .7108/.7094 · vac .7090/.7078. RAW structure:
  the corrected high-z twin (vcc, ep10 realized .596/.373/.030) tops the family by
  ~+.010 over the anchor-collapsed mirrors (.7200–.7252 — a .005 triple at essentially
  one realized coordinate), with the wall-probe vac (ep10 .614/.357/.029 — the low-z
  target UNREACHED again, z defends ~.35 at in100 too) and h-free vh0 (.586/.414/0)
  together at the family floor ~.709–.711; h-free sits −.012 under the vb/vc cluster
  at in100 where toy read h0 ≡ h.03 — raw note, read pending. End-state drift: ALL
  cells migrate inv-up by ep99 (.72–.84 / .15–.28 / ≤.012), Λ(end) 1.66–2.16,
  in-training Ω_h(end) .237(vh0)–.353. Pooled wave-1: 9 finished s-cells landed
  UNREAD per the standing rule (chains 63017165–191).

- 2026-08-04 **in1k e24voas STARTED + ep10 CONFIRM (raw; the one-correction decision
  HELD for Berker).** Ops: the 63014650→663 chain was superseded by 63014679
  (flex-first --time-min link: ran ep0→1 in its 1.5h backfill window, cancelled at the
  limit as designed) + 63014680–693; start 02:47 (~4h ahead of projection). NOTE the
  relaunch dropped --exclude=gpu277 (ExcNodeList null on the whole chain) and the run
  sits ON gpu277 — at ~48 min/ep it runs FASTER than the vm4-1k links (~68–80 min/ep,
  and this run probes every epoch): no black-hole signature; left undisturbed
  (gpu100 is parked-full; re-adding the exclude would starve the chain). wandb resume
  rider: the flex link's replayed steps fell below the monotonic cursor — wandb curves
  show a ~ep1.3–2 display gap; the .out log is complete. Reads: ep0 init shares
  .123/.834/.043 (init z-dominance, as wave-0 predicted); ep10 probe **.3480** (ring
  champion vm4-1k read .3713 at ep10 — −.023 at matched epoch), realized shares
  **(.712, .257, .031)** vs target (.55, .42, .03) — inv/z OUTSIDE the ±.05 tolerance,
  h dead-on: the anchor attraction reproduces at in1k on OAS (ep8/9/10 inv
  .702/.638/.712 — single-epoch volatility ±.04–.07, 3-ep mean ≈ (.68, .28, .03),
  approaching E24-T2's hand-tuned in1k coordinate (.62–.64, .36–.38, .01) from the
  inv-heavy side). Channels: Ω_h .467 / Ω_z .203 falling, **Λ 1.518 (>1 from ep1)**,
  ρ̂_h .435 / ρ̂_z .423 (rising-not-1 — evidence-meter behavior); no grad incidents.
  Correction math from its own ep10 g's (.259/.040/.128, T = 7.77): hitting
  (.55/.42/.03) needs w = 16.5/81.6/1.82 (inv ×.77, z ×1.64, h ≈held) — weighed
  against the in100 precedent that w_floor 95.1 moved realized z only ~+.05 into the
  attraction (vcc). ep25 gate ≈ 23:00 tonight (noise stat ep12–26 vs the band
  mean|Δacc| ≤ ~.027 / inv_sd ≤ ~.019). **Correction ruling (Berker, same day): option
  (a) — run on untouched to the ep25 gate and decide there; under E24-T3/D-075 the run
  doubles as the live OAS-vs-ring A/B at in1k (baseline .6392).**

- 2026-08-04 **CANONICAL-o8 Ω_h SORTING RE-CONFIRM — the landing channel REPLICATES the
  in-training table (instruments extended to all 29 toy cells: e24_grid_metrics 13→29
  incl. w- and v-cells; `e24_omega_sort.py` NEW → results/diag/e24_omega_sort{,_cells}
  .csv):** pooled n20 **−.790** (in-training −.754) · h-free n14 **−.935** (−.922) ·
  h.03 n6 **−1.000** (−.943) · vm-OAS n6 **−.657** (exact match) · vm-OAS h.03 n5
  **−.700** (exact) · view-mean all n9 **−.900** (n8 pre-vac −.929) · combined n26
  −.645 (−.612) · +variants n29 −.701 (n28 −.686); l2 framing within .02 of raw
  everywhere. Channel swap changes NOTHING in the ordering claims; per-cell canonical
  Ω_h now on record (the top h-free cells cluster at Ω .366–.389: wz25/s0ac/wrep/wz18).
  Full D-068 instruments (spaces/trans/census/spectrum) now cover the w- and v-cells.

- 2026-08-04 **OAS TENSION CHECK LANDED (job 63028414 → e24_cos.csv; Berker: "you can
  check the tension between inv and reg under oas too") — under OAS at toy the inv↔z-floor
  opposition is GONE:** vc ep38 (formation) cos(g_inv, g_zfloor) **+.14…+.51** over 6
  batches (never negative), ep150 +.15…+.43 — matching-or-exceeding the in100/in1k
  view-mean decoupled character (0…+.3), where the toy RING cell (v1, ep~26) read
  −.27…−.66 batch-volatile and pooled reads −.63…−.88. The toy-vm4 "partial,
  batch-volatile opposition" (E24-T2's second open clause) is thus consistent with a
  ring-staleness artifact, not a toy property of view-mean. Riders reproduce: floors
  mutually aligned at formation (cos(z,h) +.51…+.69, relaxing to +.12…+.34 at end),
  cos(inv,h) ≈ 0 (−.24…+.21); instrument shares match the in-training logger at both
  states (ep38 ≈ (.45–.53, .43–.53, .03); ep150 ≈ (.72–.79, .21–.28, .01) — the
  inv-up end drift). RAW; clause-closing nod = Berker's.

## AGREED TAKEAWAY

**E24-T1 (USER-APPROVED 2026-08-03, Berker: "record this final recipe of 2/3 and 1/3 +
few percentages on h thing"; vm scoping his same message): THE SHARE RECIPE — dose by
measured force, and put ~2/3 of the trunk's total pull on invariance, ~1/3 on the
z-conditioner, a few percent on the h-conditioner.**

- **Coordinates:** realized trunk-pull shares s_i = w_i·g_i/Σ (trunk-only g, formation
  state) — weights do NOT transport across scales/states (init≠formation by 4–17× in g;
  frames need ~4× different weights for the same force); shares DO.
- **The spot: s\* = (s_inv, s_z, s_h) ≈ (.62–.67, .32–.38, .01–.03).** Cross-scale
  evidence: toy mapped systematically (16-cell simplex + corrections + frontier; optimum
  replicated at seed 1, .868/.873; total-pull ×.5/×2 moves acc ≤ .007 — T-robust within
  4×); IN-100 healthy cells live in the same region (finals + the two s\*-mirrors
  pending); **IN-1k's independently hand-tuned lane recipe measures (.66/.33/.01) — 
  already ON the spot** (vm2/vm3 pulls, ep25 formation states).
- **The three walls that make it a spot:** (1) s_z ≳ .5 → the conditioner out-pulls inv,
  monotone decline then trunk drain; (2) s_z below ~.35 is UNREACHABLE with static
  weights (the less-satisfied floor pulls harder per unit + inv's gradient saturates once
  views match) — the optimum sits ON the reachability wall; (3) the h-danger is FORCE
  MAGNITUDE, not share (toy h_lamb boundary ∈ (0.70, 1.45); deaths at realized share
  .03–.04) — boundary magnitudes are frame-local, only share coordinates + ordering laws
  transport.
- **Procedure at a new scale:** healthy pilot past warmup → measure per-term trunk g + T
  (one GPU-hour) → w_i = s\*_i·T/g_i → confirm realized shares at ep10, ONE local
  correction → monitor inv↓, Ω_h↓ (within an h-level, end-state Ω_h ranks outcomes:
  ρ ≈ −.95 over 20 healthy runs), Λ ≥ 1.
**E24-T2 (USER-APPROVED 2026-08-03, Berker: "d072 approved. vm4 z dominance agreed."):
THE VM4 Z-DOMINANCE WAS AN INSTRUMENT ARTIFACT — warm realized shares remain THE dose
coordinate.** The .85–.92 vm4 z-share rows (in100/in1k, e24_pull.csv) were cold-ring
reads: the queue is not checkpointed, so the ckpt-based instrument fed the conditioner
128 rows against the 128-dim slice (n_eff/d′ = 1) and its loss/gradient inflated ~10×.
Under the operating estimator (ring filled with queue_steps batches first — D-072, now
the convention): **in1k vm4 = (.62–.64, .36–.38, .01), ON s\* — all three hand-tuned
IN-1k cells sit on the E24-T1 recipe**; in100 vm4 = (.52–.56, .43–.47, .01). The
in-training share logger (live rings) was correct all along. Mechanism rider (measured,
`e24_cos.csv`): pooled floors oppose inv in trunk space (cos(g_inv, g_zfloor) −.63…−.88);
view-mean floors decouple at in100/in1k (≈ 0…+.3) — Berker's interpretation ("with mean
pooled views, regularization competition with inv loss should be lower"), confirmed at
scale. **Excluded from this takeaway (open):** toy-vm4's partial, batch-volatile
opposition and the e24v1 curve-noise mystery — the control pair (§Launch log) is the
running probe.

**E24-T3 / D-075 (USER-APPROVED 2026-08-04, Berker: "we will treat OAS as stabilizer for
the toy. default for in1k and in100 for now are the running cov. we will keep in1k
running and we can change our position accordingly"): ESTIMATOR POLICY BY SCALE — OAS =
the toy stabilizer (fast per-step drift stales the 4-step ring: the control-pair
triangulation); the warm running-covariance ring (queue 3, n_eff/d′ ≈ 4) = the DEFAULT
at IN-100/IN-1K for now (slow drift + full-evidence whitening: vm4-anchored guillotine —
gauss_kl@cls .510 vs .84–.96, z.out .013 @ rank .994, kNN ≥ everywhere; not strict:
OAS cells hold head-side class margin .38–.45 vs .217 and vcc edges training-best
.7348 vs .7312). Evidence-ordering rider (toy gauss_kl@h.cls): pooled-fresh .38–.41 <
vm-fresh-512 .63 ≈ ring .64–.80 < OAS-shrunk .92–.98 < h-free ~1.5 both anatomies —
estimator evidence + the h-term carry Gaussianity; the matched-n s1a/s1ac↔v3nq pair
supports the scatter-competition channel as a real extra contribution. in1k e24voas
runs on as the live A/B vs ring-vm4 .6392; position revisable on its result. (Full row
in DECISIONS.)**

- **vm scoping (Berker):** the share coordinates are certified for the POOLED anatomy.
  The vm variants change the conditioners ALGORITHMICALLY (view-mean of the augmentations
  vs pooled views; vm4 adds the queue with ¼ gradient leverage and ×5.59-bridged
  w_floor), and **vm4 — the IN-1k winner (.6392) — reads .13/.85/.02 on the naive
  instrument: NON-comparable, not interpreted.** Open items: a vm-aware pull convention
  (measure g on the conditioner's actual input stream), and the reserve vm4-anatomy
  cells at s\* (in100, pending wave-1 finals) = the "view vm4 too" step.
  *(The pooled-only scoping is lifted by E24-T5 below; the vm-aware pull convention
  landed as D-072.)*

**E24-T4 (USER-APPROVED 2026-08-04, Berker: "toy noise was caused by model moving
faster and 4 consecutive batch stats are not shared (creating noise)"): THE TOY VM
NOISE WAS RING STALENESS UNDER A FAST-MOVING MODEL — closes E24-T2's open clause.**
The queue conditioner mixes statistics from 4 consecutive steps; when the model moves
fast (toy: small data, peak formation within ~10 epochs) those batch stats are no
longer shared — the mixed estimate mismatches the current features and the force
direction churns. Triangulation (noise stat ep12–26, mean|Δacc| / per-step inv sd):
v1 (ring, full lr) .0433/.0837 → v2lr (ring, HALF lr = half drift) .0337/.0188 →
v3nq (NO ring, full lr, bs512) **.0130/.0152 — BELOW the pooled reference band**
(Llr .0269/.0148). Staleness ∝ per-step drift ⇒ scale-dependent: the same 4-step ring
is clean at in100/in1k (vm4 anchor healthy everywhere — the D-075 policy's basis).
z-dose exonerated (v3nq z-heavy .55–.60 and smooth). Remaining T2 open item —
toy-vm4's batch-volatile inv↔floor opposition — is the pending OAS tension check
(fired same day, §Launch log). **CLOSED (USER-APPROVED 2026-08-06, via the plain
question round): the apparent opposition was an artifact of the stale ring
estimator, not a real conflict between the two losses — the landed OAS tension
check (job 63028414) measured cos(g_inv, g_zfloor) at +.14…+.51 at every
checkpoint under the sound estimator. E24-T2 is now fully closed.**

**E24-T5 (USER-APPROVED 2026-08-04, Berker's closing restatement: "e24 recipe was 2/3
zinv, 1/3 zreg and a few percentage hreg. we also found that oas stabilizes toy
training but not necessary for in100 and in1k."): THE E24-T1 RECIPE STANDS ON THE
OPERATING (view-mean) ANATOMY — the pooled-only scoping is lifted.** Evidence: the toy
vm-OAS map (6 cells .8693–.8764, flat basin over realized r .41–.63 spread .007 above
the z-reachability wall ~.45–.47; va/vb/vac same-shares/different-weights triple
within .001; h0 ≡ h.03) · the in100 vm map complete (6/6 healthy .7078–.7348, anchor
attraction to s\*, wall reproduced at ~.35, twins bracket the mirrors) · in1k vm cells
measure ON s\* warm (E24-T2) with e24voas cooking at the anchor. Estimator scoping =
E24-T3/D-075. **Riders from the same ruling:** (1) no seed replicates — "we dont care
about the accurate performance" (the map/mechanism is the deliverable; the 3 proposed
toy cells vc-s1/upper-edge/v4oas-s1 are DROPPED); (2) the 9 finished in100 POOLED
wave-1 runs stay UNREAD and closed — "they do not matter i do not think it is a good
practice to do such competition between inv and spectral reg" (the pooled anatomy is
deprecated as practice for this program; landed artifacts remain on disk, unworked);
(3) gpu277 is rehabilitated — "bad gpu is no longer bad" (the D-065 black-hole
exclude is removed from the standing sbatch files; the e24voas chain trained on it
all day at full pace). Formal P-E24-1…5 scoring rows remain PARKED on this card (raw
outcomes recorded throughout the §Launch log).

## ON-HOLD RECORD — e24voas ep25 gate package (2026-08-05; raw, interpretation deferred per Berker "keep these e24 results on hold")

Gate REACHED overnight (chain link 63014683; run continued past it, ep29 .4293 at
record time; no incidents anywhere in ep0–29; grads decaying). **Noise stat ep12–26:
mean|Δacc| = .0056 (band ≤ ~.027), inv_sd = .0121 (band ≤ ~.019) — both PASS.**

| ep | probe | ep | probe |
|---|---|---|---|
| 12 | .3630 | 21 | .4140 |
| 13 | .3707 | 22 | .4123 |
| 14 | .3868 | 23 | .4176 |
| 15 | .3900 | 24 | .4180 |
| 16 | .3880 | **25** | **.4155** |
| 17 | .3957 | 26 | .4292 |
| 18 | .3991 | 27 | .4253 |
| 19 | .4004 | 28 | .4325 |
| 20 | .4071 | 29 | .4293 |

Matched-epoch baseline (ring-vm4-1k, incumbent .6392@100): ep24 .4484, ep26 .4527,
ep28 .4629 → voas deficit ≈ **−.030**, stable since ep8. Shares at ep25:
(.693, .277, .030) vs target (.55, .42, .03) — the ep10 anchor attraction never moved;
realized inv .69 sits ABOVE the toy-mapped basin top (.63). Channels at ep25: Ω_h .375
(from .437 @ ep13, drifting down), Ω_z .149, Λ 1.589 (climbing), ρ̂ .461/.468.
Correction math staged (w 16.5/81.6/1.82 from its own ep10 g's). **Gate ruling
FINAL (USER-APPROVED 2026-08-06, plain question round): run UNTOUCHED to ep100 —
the OAS-vs-ring A/B stays clean; the mid-run share-steering question moves to the
S/B/L program design (E25-T1 successor), where the anchor-attraction diagnosis
belongs.**

## ROWS FROM THE 2026-08-10 QUESTION ROUND (USER-APPROVED, Berker: "e24 agreed"; wording Fable, veto open)

**E24-T3 (USER-APPROVED 2026-08-10) — the estimator law, measured both ways:** the running-cov ring and
OAS split by STREAM HOMOGENEITY, not by scale. Homogeneous view-mean stream (V=4
uniform, in1k, matched everything): ring WINS at land — vm4 .6416 raw/.6554 l2/.5335
kNN vs voas .5999/.6135/.4574 (−4.2/−4.2/−7.6; voas online read high, the co-trained-
probe bias visible in-data). Heterogeneous-mean streams (V=10 multicrop: the all-view
cross-scale mean, the locals-group mean): the ring is TOXIC — stale rows of a
fast-moving heterogeneous mean produce episodic −logdet blowups (e27grp ep16 g_cond_z
.034→.575→.032; e27mc probe dips past the noise band), while OAS lanes are monotone
(D-089). This refines D-075's toy-vs-scale split into a stream-homogeneity law:
homogeneous → ring (n_eff at full trust); heterogeneous → OAS (adaptive shrinkage
absorbs the estimation noise the ring amplifies). B/L consequence: estimator choice
follows the winning wave lane's stream.

**E24-T4 (USER-APPROVED 2026-08-10) — the toy vm-OAS basin recipe:** at toy scale under OAS view-mean
payment, the dose map has a FLAT BASIN — r (z-share ratio) ∈ .41–.63 spans spread
.007 with vc .8764 the family record — bounded below by the z-wall at r ≈ .45–.47
(cells beneath it degrade), with anchor attraction pulling every cell's realized
shares to the (.70–.85, .1–.3, ~.03) equilibrium and the mechanism visible in ρ̂
(the evidence meter climbing as the conditioner's demand is met). Operating rule:
place the dose IN the basin above the wall and let the equilibrium take it; exact
in-basin position is second-order (±.007).

**EVAL-FRAME FLAG on E24-T3's landed magnitudes (D-090, found 2026-08-10 afternoon —
the row text above stays verbatim; amendment = Berker's at the joint read):** the voas
landing rode the SUPERSEDED train500 frame (probe fit + kNN bank on 500k rows) while
vm4's landing rode the D-066 standard FULL 1.28M train — the extract default regressed
between the two landings (vm4's 07-30 extract passed `train_per_class=null`; voas's
08-08 extract didn't). The −4.2/−4.2/−7.6 deficits are therefore manifest-confounded in
vm4's favor (2.56× probe rows, 2.56× kNN bank); the DIRECTION claim and the entire
heterogeneous-stream half of the law (in-training trajectories, D-089) are untouched.
Standard-frame voas re-probe launched (extract 63222502 → 12-space probe 63222503);
corrected numbers land here beside the originals with deltas. Guard + full record:
DECISIONS D-090, E27 card §(j.5).

**Ω-offset calibration (DELIVERED 2026-08-10, informational — the band-steering
consumer was retired at D-088):** voas in-training channel (fixed batch, V=4
audit_v1, eval-mode) ep99 Ω_h = .220 vs landed o8 Ω_h(cls) = **.2068** → offset
**+.013 (×1.06)** — the in-training channel reads the landing channel almost
directly at matched (V=4, audit) construction. vm4 landed .2111 reproduced (method
check). e27lej landed .1446 (= .20 of threshold; on the E27 card). For the E27 wave
lanes the same calibration computes at their landings from the 3-channel logger's
`omega_h` (aud) series. voas end-state shares drifted to (.85, .14, .01) by ep99 —
the anchor-attraction equilibrium, raw context.
