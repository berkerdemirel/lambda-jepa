# E23 — MLP leakage: head depth as the orbit-radius dial, and the overlap sweet spot

**Status: PRE-REGISTERED + USER-APPROVED 2026-07-29 (Berker: "lets go"; design settled over
three discussion rounds this date). D-rows: D-067 (design), D-068 (instruments). Thesis and
brief: SESSION_OPENER.md (2026-07-22→29) — Saunshi et al. 2022 orbit-overlap reading; the
field tunes leakage via the MLP-after-h without naming it.**

## Question

A shallow projector cannot satisfy the SSL objective inside the head, so invariance pressure
leaks back into h — h's orbits shrink. Residual capacity added to the head absorbs the
objective and relieves h. Is there a sweet spot of leakage — controlled orbit radius at h →
controlled *class-conditional* orbit overlap → downstream? **Depth K of an identity-init
residual MLP is the dial; everything is measured in the Λ = b/a calculus (below) plus the
E17 census instruments.**

## The calculus (fixed for this card)

Per space s ∈ {h, z}, over V-view orbit stores: orbit energy W_s = E‖s_iu − s_iv‖² (view
pairs), center energy B_s = E‖m_i − m_j‖² (image pairs, m_i = view-mean), thickness
Ω_s = W_s/B_s; radius r_rms = √(W/2) (exact: E‖s_u − s_v‖² = 2σ̄²). Transmission
a² = W_z/W_h (aug secants), b² = B_z/B_h (center secants), Λ = b/a; identity Ω_h = Ω_z·Λ²
(identity — never a prediction). **V-debias is exact and closed-form: B = B̂ − W/V**
(sample view-means carry σ̄²/V of orbit noise into center distances) — verified empirically
on the three o32 stores: B̂ drifts ~30% across V ∈ {2,8,32}, debiased B is V-invariant to
3 digits (`results/diag/e23_retro_spaces.csv`, o32 rows). o8 is sufficient forever.

Conditioner placement facts (variant-resolved; corrects an overstatement caught by Berker
2026-07-29): POOLED conditioner ⇒ per-direction budget B-var + W_z/2 ≈ d′ (orbit and
center variance compete for trace — the E21 92%-aug-paid collapse lived here); VIEW-MEAN
conditioner (vm3/vm4) conditions centers only (a 1/V-damped orbit echo). **Neither variant
floors W_z.** A residual-conditioner on {z_iv − m_i} would be the τ-band — METHOD-CANDIDATE
only, no run authorized under this card.

## Design (locked; **rev2 2026-07-30** — rev1's residual scaffold, BN-free lane, 2048-d
width, and byol fallback are all dead, §Launch log; every rev2 ruling is Berker's
2026-07-30 answers)

**Frame/lane:** toy_vits8 (imagenette, 150 ep, grad_clip 1.0) · **aug=lejepa V=4 · bs=128**
(Berker: "we will continue with v=4"; the lane holds the best toy numbers on record —
toy-lejepa .912 lin/.897 kNN — floorssl's DOSES on it are what stage B certifies) ·
z-floor POOLED, h-floor POOLED · queue off · **expander out = 256** with the two named
conditioning ratios held at the family point (Berker's frame): **exploration D/d′ =
256/128 = 2, estimator n/d′ = 512/128 = 4**. (2048 = D/d′ 16 = the recorded collapse
regime at both scales.)

**Head (rev2):** `floorssl_stage_head` — bare Linear(384→2048) adapter + K
**BottleneckStages** [Linear(2048→m)–BN–ReLU–Linear(m→2048)–BN–ReLU] + bare
Linear(2048→256) out. Every path nonlinear (the rev1 skip's always-open linear path is
what defeated the depth dial); fully BN-pinned per stage (BN "essential" — Berker; every
healthy floorssl ever is BN). **m = the per-stage bottleneck width — the FINE capacity
dial** answering the K=0→1→2 coarseness (Berker's granularity requirement, replacing
residual identity-init). K=0 = bare adapter∘out = the linear projector (folklore anchor;
max-leakage end; declared-risk cell, held by the h-term). Taps per stage + out (per-K
within-head Λ profile free). Invertible/normalizing-flow heads: parked (open menu).

**Axes (final; Q4 resolved 2026-07-30 "ok do h off experiments too"):** depth
**K ∈ {0,1,2,3,4,6} at m=256** (the evolution axis — Berker: "doing k anyway could help
as well to see the evolution") + fine **m ∈ {64,128,512,1024,2048} at K=2** · **h_lamb =
.65 primary everywhere**; **h_lamb=0 controls at K ∈ {1,3,6}** (the folklore twin — the
leakage-only readout at low/mid/high depth) · no head_norm axis (BN always). Main program
= 6 + 5 + 3 = **14 cells** after lane certification + wd grid.

**wd protocol:** projector param group (`mlp_wd`); wd is an EMPIRICAL dial (BN gauge caveat
recorded once: under BN, weight scale is partly gauge; the bare out-Linear keeps real
bite; Berker expects measurable effect regardless = P8′). Grid {0, 1e-3, 1e-2, 5e-2,
2e-1} at the base cell (K=2, m=256, h.65) ON the certified lane; coefficient then frozen.

**Stage B rev2 (lane certification first — Berker approved BOTH 2a and 2b):**
- **B-L1 (share-measured re-dose):** realized per-term pulls (g_enc instrument, E17/E19
  lineage) on the healthy byol-lane cell + a short lejepa-lane run at the base scaffold →
  re-dose w_inv/w_floor by the realized-share law to match the healthy profile → 1–2
  confirm runs. Plus ONE arm at the toy-lejepa optimizer point (lr 2e-3, warmup 1) as the
  alternative hypothesis.
- **B-L2 (blind grid):** ~6 dose pairs around the two known anchors on the lejepa lane at
  the base scaffold.
- **Success criterion (pre-declared):** the e23y256leg health envelope — monotone probe to
  ≥.70 @150ep, all terms converging, gnorm decaying. Dose point picked JOINTLY.
- **B-W:** the wd grid on the certified lane → **joint stage-B review** → C (mains).

Landing per cell: o8 audit extraction (audit_v1 stack, toy pairs manifest) at ep150 →
energies + census + spectrum + probes; cadence ckpts ep38/75/113/150 for trajectory reads.

**Instruments at landing (D-068):** `sslgap/metrics/orbit_energy.py` (W/B/Ω/a/b/Λ, debiased,
raw + per-view-L2 framings, label splits) · the E17 census revived (touch% = 2r ≥ d_intra,
overlap depth ω, **class-conditioned**: touch-same vs touch-diff, enrichment p_pos/p_neg) ·
the {a_k} transmission spectrum (least-squares J: h-residuals → z-residuals over view pairs;
gain profile over h-PCs; selectivity = spread of {a_k}) — replaces the α-grouped-projector
control, which was REJECTED (Berker: it treats h as fixed, amputates head–trunk co-training;
its flat spectrum destroys the selectivity that plausibly carries the semantic effect) ·
frozen probes (lin + kNN at h).

**Run naming:** `toy.floorssl.s0.<tag>`: lane cert `e23L*` (B-L1 measure/confirm, B-L2
grid, `e23Lopt` optimizer arm); wd grid `e23w*`; mains `e23K{K}` (m=256, h.65), `e23M{m}`
(K=2), controls `e23K{K}h0` (pending Q4). Smokes `e23smoke` (rev1), `e23smoke2` (rev2).

## Pre-registered predictions (rev2, locked before any rev2 number exists; a < 1 expected
everywhere — Berker: watch a, no target for b, Λ carries the reading)

- **P0 (premise):** a < 1 in every trained cell and falls monotonically with K (m=256). If
  a does not move with K, the depth-as-leakage-dial premise fails — the informative null.
- **P1:** inv_final falls with K (≡ W_z up to 1/d; pooled lane ⇒ budget-stabilized).
- **P2 (core coupling):** W_h and r_rms(h) rise with K; the (log a², log W_h) trajectory
  moves up-left across iso-W_z diagonals — depth shifts who pays for invariance from trunk
  to head.
- **P3 (sweet spot):** kNN@h vs K peaks as a PLATEAU EDGE (T4's threshold shape), located
  where touch-same saturates while touch-diff is still low; enrichment peaks there.
- **P4:** linear@h flatter than kNN across K (E17/E21 pattern).
- **P5′:** Ω_h rises with K at stable b (selective relief); under the h.65 primary the
  low-K scale-collapse mode is damped by the h-conditioner — it shows as strain (h_kl
  effort ↑ at low K), not collapse.
- **P6 (conditional on Q4 controls):** the h0 control cells show the invented-dimensions
  signature at low K — b↑ with effrank_h↓ (zonly = the existing extreme: a=5.9, b=9.5) —
  suppressed in their h.65 twins.
- **P7 (selectivity):** the {a_k} spectrum grows more UNEQUAL with K, not merely lower in
  mean.
- **P8′ (Berker):** mlp_wd shows a measurable empirical effect on (a, W_h, kNN@h) despite
  BN's scale gauge.
- **P9 (fine dial):** at fixed K=2, growing m lowers a smoothly — the m-axis moves the
  same physics as the K-axis, in finer steps.
- **Kills:** house rules standing; W_z collapse in any arm; single-step grad-norm >100×
  running median = INCIDENT.

- 2026-07-30 **rev2 code landed** (`floorssl_stage_head` + BottleneckStage + tap rule +
  config keys `head_stages`/`stage_m`; `head_depth` retired from the training path;
  `floorssl_res_head`/ResBlock kept assembly-only with REJECTED notes; audit.py pairs loop
  now skips orbit manifests — the o8-as-pairs crash that felled the three bump audits,
  reruns 62857493-95). Verify battery: taps per stage ✓ K=0 linear ✓ round-trip ✓ m-dial
  param scaling ✓ legacy intact ✓. Smoke `toy.floorssl.s0.e23smoke2` (2 ep, lejepa lane,
  base cell) = job 62857842.

- 2026-07-30 **B-L launched** (smoke2 PASSED, ep2 .2206): anchor `e23Lbyol` 62857858 (stage
  head on the healthy byol lane; formation source) · blind grid `e23Lg1-6` 62857859-64
  (lejepa lane: (25,45),(32.8,38.7),(50,45),(25,22),(50,22),(12,60)) · `e23Lopt` 62857865
  (lejepa optimizer point). **B-L1 pull MEASURED** (job 62858212, Lbyol ep38 formation,
  bs=128 both lanes, trunk-only g_enc): inv 2.612(byol) vs 5.302(lejepa) — the V=4 family
  pulls inv on the trunk **2.03×** harder; z-floor 2.927 vs 3.547 (1.21×); h-floor 3.121
  vs 2.995 (~parity). **Equal-pull re-dose: w_inv 25→12.3 · w_floor 45→37.1 · h_lamb
  .65→.68**; confirm arm `e23Lshare` 62858283 launched. (Direction note: the IN-100 lane
  fix landed at (32.8, 38.7) — the toy measurement points LOWER on inv; dose laws are
  frame-local, the house dose-curvature lesson.) Rows: results/diag/e23_pull.csv.

- 2026-07-30 **B-L rev2 KILLED 9/9 (Berker: "kill the failed runs")**: the ANCHOR
  `e23Lbyol` failed on the healthy byol lane itself (ep5 .285 → ep136 .22) — the rev2
  stage anatomy is the isolated factor (its BARE adapter left the 2048-d entry unpinned;
  every healthy floorssl head BN-pins after every hidden Linear); all lejepa arms + the
  share-derived arm + the optimizer arm died with it, so the lane-dose question is
  UNATTRIBUTED under rev2 and the first pull measurement (sick-state) is DISCARDED.
  **Berker's curve diagnosis (2026-07-30): "regularization dominating inv"** — the
  operating mechanism: bare/skip segments widen the conduit through which the z-floor's
  pull reaches the trunk (the E19-T1 BN-firewall story), the regularizer out-pulls inv,
  h gets bent to serve z-conditioning (hkl↑, inv unsatisfied). Weighted realized pulls
  (w·g) become the reported quantity; the healthy cell's SHARE PROFILE is the dose target.
- 2026-07-30 **PROTOCOL (Berker): ep20 KILL GATE** — a toy arm with nothing healthy at
  ep20 is killed (gate: probe ≥ .35 at ep20, then a FULL-curve read — inv, z-space
  moment_kl, h-space h_moment_kl — before it continues; collapsed arms sat at .11–.15 by
  ep20, healthy at ~.5: wide separation). **Canary-first**: no batch on unproven anatomy —
  single gated runs first. "Be more careful while setting things" — acknowledged.
  **REVISED same day (Berker, after e23Lw's post-gate collapse): the gate sits at
  max(ep25, warmup_end+5)** — gating DURING the ramp is blind (Lw passed .5108 at ep20
  with its warmup still running, then collapsed to .1475 by ep25 the moment lr peaked).
- 2026-07-30 **rev3 ladder = the LEGACY ANATOMY itself** (`floorssl_ladder_head`): K
  hidden BN-ReLU layers; **(K=2, width=hidden) is BYTE-IDENTICAL to the certified bn
  expander (verified: state-dict equality at seed 0)** — the ladder grows from the
  healthy .745/.751 cell; `head_width` = the fine dial (middle-layer width); K=0 = bare
  linear anchor (declared risk). rev2 `head_stages` retired from training (assembly-only).
  Canaries `e23K1` 62858755 + `e23K4` 62858756 (byol, h.65) with ep20 gates ARMED; pull
  v2 62858754 re-measuring on the CERTIFIED y256leg ep38 state with w·g shares. Full rev3
  grid + lejepa lane rerun wait on canary gates + the share numbers.

- 2026-07-30 **rev3 canaries PASSED their ep20 gates + full-curve reads** (K1 probe .5434,
  K4 .5259; gnorms DECAYING 28→16/13, zkl converging, hkl stable-to-falling — the healthy
  signature; both continue to ep150). **Pull v2 on the CERTIFIED state landed
  (results/diag/e23_pull.csv): the share profiles are LANE-INVARIANT** — byol inv/.469
  zfloor/.503 h/.028 vs lejepa .471/.504/.025; the sick-state 2.03× was artifact. Derived
  correction small: w′ = 22.1/39.8/.64 (~12% down). ⇒ the lejepa disease is NOT a
  formation-share imbalance — it lives in the ep0–20 warmup phase. Three ep20-gated
  early-phase canaries launched (legacy-identical K=2, derived doses): `e23Ld` 62858762
  (doses alone), `e23Lw` 62858763 (+warmup 20), `e23Llr` 62858764 (+lr 5e-4).

- 2026-07-30 **THE LANE SOLVED (ep20 gates + full curves): the lejepa-at-toy disease is
  the lr TRANSIENT, not doses.** `e23Ld` (derived doses alone) GATE-KILLED at .1704 —
  confirming the share result; **`e23Lw` (+warmup 20) PASSED .5108 and `e23Llr` (+lr
  5e-4) PASSED .5827** — both with the full healthy signature at ep20. **SUPERSEDED
  hours later (Berker caught it live): Lw COLLAPSED at ramp-end** (.5108@ep20 →
  .1475@ep25 → KILLED@ep36; the ep20 gate had tested it mid-ramp — protocol revised to
  max(ep25, warmup_end+5)). With warmup-10 dying at ep~10 and warmup-20 at ep~25 — both
  at ramp-end — while **Llr climbs (.6369@ep36)**: the toy-lejepa lane cannot survive
  PEAK lr 1e-3 regardless of ramp length; **lr 5e-4 is the lane recipe candidate**
  (formal pick at the joint review; note it carries a peak-lr difference vs the byol/
  family reference cells). Three canaries to ep150: K1, K4, Llr.

- 2026-07-30 **stage-B review round (Berker): LANE RECIPE LOCKED = lejepa V=4 + peak lr
  5e-4** ("thats good"). Dead-run PURGE executed on his order: 131 ckpt files / 51 GB
  deleted; kept = e23y256leg, e23z3legh0, e23K1, e23K4, e23Llr (the healthy set). IN-1k
  3-cell guillotine BUILT under the standard frame:
  `results/figures/e23/in1k_guillotine_3cell.png` (+INDEX) — raw shape: monotone rise
  L03→cls in both probes, decline through the head; vm4 rides highest through the head
  (z.out .534 lin / .432 kNN vs vm2/vm3 ~.486/.375); kNN separates vm3 down at cls.
- 2026-07-30 **stage-C RESHAPE PROPOSED (Berker's intent: minimal→bump→overtreat, and the
  metric↔performance relation as the deliverable):** capacity axis from genuinely minimal
  heads — **width ladder at K=2**: expander_hidden ∈ {32, 64, 128, 256, 512, 2048(=Llr)}
  — + **depth ladder at hidden 2048**: K ∈ {0, 1, 3, 4, 6} (K=2=Llr) + h0 controls at
  {hidden64-K2, hidden2048-K2, K6}; all on the locked lane recipe; **wd DEMOTED** to fixed
  5e-2 (a 3-point wd probe at the base cell later covers P8′) — capacity is the scientific
  axis, wd was the residual-era dial. 14 new gated runs; every cell lands with the full
  battery; the headline analysis = (a, b, Λ, Ω_h, W_h, touch-same/diff, enrichment) vs
  (kNN, linear) along the capacity ladder — P0–P9 read on the capacity axis. CONFIRMED
  by Berker ("did you launch the grid jobs?") → **stage C LAUNCHED 2026-07-30**: 13 cells
  on the locked recipe (lejepa V=4, lr 5e-4, doses 22.1/39.8/.64): width e23jW{32,64,128,
  256,512} = 62866643-47 · depth e23jK{0,1,3,4,6} = 62866648-52 · h0 controls
  {e23jW64h0, e23jK2h0, e23jK6h0} = 62866653-55 (+ e23Llr as the 2048/K2 cell). All
  ep25-gated (persistent monitor emits per-cell verdicts); landing chain per survivor
  after full runs.

- 2026-07-30 **stage-C ep25 gate round complete (8 PASS / 4 KILL + 1 NFS-flake retry
  pending)**. PASS: W32 .590 · W64 .585 · W128 .488 · K1 .550 · K3 .619 · K4 .554 ·
  K2h0 .577 · K6h0 .469. KILL: W256 .246 / W512 .258 (the classic disease: inv stuck ~.61,
  gnorm 52/93) · K0 .229 (gnorm 133 — the linear-conduit max-leakage cell, as declared) ·
  K6 .213 (inv STUCK .62 at hkl .96 — NO cliff; possibly premature: deep stacks ramp slow,
  and its h0 twin PASSED). Raw structure: (1) **tiny heads run an inverted economy** —
  W32 inv .12 / zkl 3.1 (invariance paid, anti-collapse dumped: rank-32 z can't fill 256
  isotropic directions) yet healthiest ep25 probes; (2) **the width axis crosses three
  regimes** — tiny (healthy, inverted), mid 256–512 (collapse), 2048 (healthy, pays both);
  (3) **K6 h-pair datum**: h.65 stalls inv and dies, h0 progresses by SPENDING h (hkl 3.14
  climbing, optimization healthy, probe .47) — leakage live on the thermometer; h×depth
  interaction vs ramp artifact = open (later-gate K6 rerun offered to Berker).
  **POLICY OVERRIDE (Berker, same hour): grid cells COMPLETE regardless of gate verdicts**
  — the collapse region is part of the metric↔performance map; kills stay for
  canary/diagnostic contexts only. The four killed cells RELAUNCHED with identical tags →
  auto-resumed from their ~ep25 checkpoints (W256 62867107 · W512 62867108 · K0 62867109 ·
  K6 62867110); the W64h0-retry gate killer disarmed. All 13 cells run to ep150. Figures:
  e21_guillotine_vm + e20_guillotine_zoo2 EXTENDED with Ω/a/b/Λ columns (station→z.out);
  the IN-1k 12-panel likewise (job 62866843).

- 2026-07-30 **STAGE-C GRID COMPLETE — all 13 cells to ep150 (Berker's completeness rule;
  killed cells resumed and finished). FINAL ONLINE BESTS, RAW:** width@K2: W32 .7944 ·
  W64 .7873 · W128 .7052 · **W256 .3679 · W512 .3735** · W2048(Llr) .8003 | depth@2048:
  **K0 .4451** · K1 .7562 · K2 .8003 · K3 .7987 · K4 .7969 · **K6 .4492** (never recovered
  over its full run) | h0 controls: **W64h0 .8046 (the grid maximum)** · K2h0 .7952 ·
  K6h0 .7434. Raw shapes for the interpretation session: the mid-width CRATER (.37 flanked
  by .79/.80 — the ep25 three-regime structure held to convergence); the depth PLATEAU
  with craters at both ends (K0 linear, K6 deep); **the h-free lane at-or-above its h.65
  twins everywhere** (incl. the K6 pair .74 vs .45 — the h-conditioner HURTS the deep
  cell; and the byol echo .751 vs .745). NO TAKEAWAYS — next session, jointly. Landing
  chains FIRED for all 14 cells (extract o8+L-taps one pass → probe + battery; jobs
  62868609-50, counts verified 14/14/14).

## Grounding (retro-analysis, FREE, landed 2026-07-29 — numbers RAW)

`results/diag/e23_retro_spaces.csv` (1086 rows) + `e23_retro_trans.csv` (1215 rows); job
62808982; ~45 o8/o32 stores (family, e20f zoo ctrl/arm pairs, E17 arms, lejepa cadence),
zero skips. Raw observations carried into the design: family 256-d cells a≈.49–.61,
b≈.91–1.07, Λ≈1.6–2.2; d128/d64 tail a=.38/.22 (out-dim as a-dial → held fixed here);
BW_cls(h) vm4 .497 vs d256 .359 (E21-T2 direction reproduced under this estimator); zonly =
the P6 signature on disk; within-head Λ climbs tap1→out in every 256-d cell (selectivity
built layer by layer); zoo ctrl→e20f arm raises Λ in 6/7 methods (ijepa flat); lejepa
cadence walks a .098→.072 at flat b over training. Interpretation: discussion-only.

## Launch log

- 2026-07-29: code landed (`floorssl_res_head` + ResBlock + TVMLPTaps tap rule + `mlp_wd`
  param group + config keys; legacy path byte-identical, defaults null). Ladder invariant,
  taps, arch round-trip, zero-init dynamics, legacy layout: all verified (CPU battery).
  Smoke `toy.floorssl.s0.e23smoke` (2 ep) = job 62831881 — PASSED.
- 2026-07-29 **stage-B dose arms FAILED health** (`e23dA` 62836228, `e23dB` 62836229; 150 ep
  landed): probes peak ~.33 at ep5 then COLLAPSE to ~chance by ep10–40, end .166/.187.
  Forensics (wandb per-step, curve-forensics default): h_moment_kl (zero-weighted, logged —
  the free h thermometer) climbs .29→~3.0 monotonically; grad_norm healthy ~30–50 through
  ep3–7 then cliffs (97→3397; dB spike 30686 at ep~60 — the >100×-median kill condition
  occurred; runs lived fully clipped from ep~10); inv NEVER satisfied (rises .29→.63 dA /
  .26→.45 dB, then flat). Final-ckpt check: residual blocks fully awake (W₂ ~43–46 ≈ W₁) —
  identity-trap hypothesis REJECTED. Prime suspect: **unpinned BN-free residual head**
  (E21's declared hidden-scale risk, amplified by residual adds); second: lane-dose
  mismatch (lejepa V=4 at toy uncertified). NO dose pick; **wd grid ON HOLD**.
- 2026-07-29 four discriminating probes launched (dB dose held fixed; ~1.4 h each):
  `e23x0` 62841768 (K=2 BN-free mlp_wd=0 — wd's role), `e23xb` 62841769 (K=2 bn — do BN
  pins restore health; P8's flip side), `e23xleg` 62841770 (LEGACY expander bn on the SAME
  lane — separates head-fault from lane-fault), `e23xh` 62841771 (K=2 BN-free h_lamb=.65 —
  does h-enforcement alone rescue). If the BN-free lane needs a stabilizer, that is a
  DESIGN re-derivation with Berker (contract) — not a patch.
- 2026-07-29 **probe verdict: the HEAD IS EXONERATED — the toy-lejepa lane is the fault.**
  ALL FOUR collapse identically (same ep5-peak→ep10-20 crash): e23x0 .199 / e23xb .173 /
  **e23xleg (LEGACY certified bn expander) .201** / e23xh .131 (h-enforcement does NOT
  rescue; worst). The identical config family is certified-healthy at IN-100 → the fault is
  the LANE-AT-TOY-FRAME transfer (aug=lejepa V=4 pooled at imagenette/128px/9.5k imgs),
  not the residual head, not mlp_wd, not BN-freeness, not h-freedom. Discriminating pair
  launched: `e23xcert` 62843664 (E19-certified toy cell VERBATIM: byol V=2 bs=256
  25/45/.65 bn — healthy expected, else repo REGRESSION) + `e23xres2` 62843665 (res K=2
  BN-free h_lamb=0 mlp_wd=5e-2 ON the certified byol lane — tests the fallback lane for
  the whole program). Lane decision returns to Berker after these land (his lejepa-lane
  call hits a real wall at toy; the wall is information).
- 2026-07-29 **e23xcert ALSO collapsed (.198) → premise archaeology → TWO FALSE PREMISES
  found (Claude's, owned):** (1) "the 25/45 toy-certified v2 recipe" — E19 was an IN-100
  experiment (its card's own launch-error note proves it); NO floorssl run ever existed at
  the toy frame before today; (2) "toy-certified 2048-d expander" — same false source; the
  family's ACTUAL certified operating point is d256, and 2048-d is the RECORDED E21
  collapse regime (92%-aug-paid). Meanwhile the toy frame itself is fine: the M0 toy zoo
  trained here — toy.vicreg.s0 at the IDENTICAL optimizer point (lr 1e-3/wd 5e-2/warmup
  10/clip 1, bn expander, byol pair) probes h.cls lin .78-.80, kNN .72-.75. So: optimizer
  point exonerated, frame exonerated, architecture exonerated → the floorssl LOSS at
  out=2048 with IN-100 doses is the standing suspect at toy. ALL eight failed runs shared
  expander_dim=2048.
- 2026-07-29 d256 probes launched (the never-tried certified operating point at toy):
  `e23y256leg` 62845336 (legacy bn byol 25/45/.65 d256), `e23y256res` 62845337 (res K=2
  BN-free h0 d256), `e23y256lej` 62845338 (legacy bn lejepa V=4 32.8/38.7/.65 d256 — the
  vm2-minus-frame transplant). If healthy → stage B resumes at out=256 (D-067 amendment
  needed: design said out=2048 — Berker sign-off); if sick → the realized-share dose-transfer
  instruments come out (measured re-dose at toy, E19-T1 law).
- 2026-07-29 **d256 verdict: `e23y256leg` (legacy bn, byol, 25/45/.65) HEALTHY — .745 at
  ep150, monotone, textbook signatures (inv .54→.43, zkl .40→.19, hkl .92→.27, gnorm
  27→6). THE FIRST HEALTHY TOY FLOORSSL EVER; 2048 was strictly toxic (same config died
  there).** Still dying at d256: `e23y256res` (res K2/BN-free/h0: hkl .89→2.07, gnorm
  cliffs — the h-degeneration disease; h_lamb=.65 visibly fights and wins in the healthy
  cell) and `e23y256lej` (lejepa V4 38.7-doses). Factorization round launched (byol/d256
  unless noted): `e23z1resh65` 62845797 (res BN-free + h.65 — h-freedom vs head),
  `e23z2resbn` 62845798 (res bn + h.65 — BN-freeness), `e23z3legh0` 62845799 (legacy +
  h0 — h-freedom on the certified head), `e23z4lej2545` 62845800 (lejepa + 25/45 — dose
  vs lane), plus `e23z5resh0k8` (res K=8 BN-free h0 — does DEPTH relieve the h-free
  instability; if yes, the stability boundary K*(h_lamb=0) becomes a design observable —
  the leakage mechanism at its extreme).
- 2026-07-29 **FACTOR MAP COMPLETE (toy/d256): h-freedom VIABLE — `e23z3legh0` (legacy bn
  byol h_lamb=0) = .751, the best cell; `e23y256leg` (h.65) = .745. THE RESIDUAL HEAD IS
  THE ISOLATED KILLER — 6/6 res arms died** (h.65/bn/h0 × K2/K8 × byol/lejepa; z1 .19,
  z2 .24, z5 .15), with mlp_wd arithmetically exonerated (5e-2 both groups ≡ the legacy
  single group under AdamW). **Mechanistic read (the wall is information): the additive
  skip is a PERMANENT LINEAR CONDUIT from h to z** — z = linear(h) + learned(h) at every
  depth, so inv pressure reaches h through a K-INDEPENDENT bypass: the leakage dial cannot
  dial (collapse at K=2 and K=8 alike ✓), while the legacy plain MLP is opaque (every path
  crosses ReLUs — and every healthy cell has full-path BN pinning). The residual scaffold
  architecturally short-circuits the very quantity E23 manipulates. **Separately: lejepa
  lane at toy died 3/3** (legacy head, both dose flavors; z4 final .2196).
  Stage-B review package → Berker (scaffold re-derivation: PLAIN-MLP depth ladder; lane;
  out=256 amendment). NOTHING further launches pending his ruling.

- 2026-08-04 **STAGE-C′ LAUNCHED (D-069 USER-APPROVED-AS-AMENDED, Berker: "can we redo
  our grid search on toy: mainly playing with width and depth repeating the previous
  failed analysis. always using the mean of the views for the regularization on both h
  and z") — the capacity grid on the vm-OAS anatomy** (view-mean h+z, `floor_shrink=oas`
  per D-075's toy policy; the toy estimator consolidation is what unblocked this).
  Cells (12, `slurm/e23_wave_c2.sh`): WIDTH ladder K=2 hidden {32, 64, 128, 256, 512} ·
  DEPTH ladder hidden-2048 K {0, 1, 3, 4, 6} · h0 twins {W64h0, K6h0}; the 2048/K2
  references = the LANDED E24 v-cells (vc .8764 / v4oas .8716 / vb .8696; vh0 .8713 =
  the h0 reference) — no reruns. Uniform prior dose = vc's measured basis
  **16.2/46.7/0.73** (the (.47/.50/.03)-pinned W2048K2 cell — D-069's certified profile
  transplanted to this anatomy); per-epoch share/ρ̂/orbit logging from birth; ep10
  confirm reads vs (.47/.50/.03) ±.05; ep150 completeness rule (stage-C precedent —
  cells run to the end, kill only on Berker's call). **Declared deviations from D-069's
  letter (walls-are-information presentations, not patches):** (1) the init+1ep
  pre-measure pass is DROPPED — init g's swing 4–17× vs formation (the E24 wave-0
  lesson postdates D-069); the prior anchors on vc's formation basis and the ep10
  confirm + per-case correction carries the share-pinning; (2) corrections are
  per-case at ep10 (E24-vm precedent: originals run on as map points, twins only where
  coverage demands), not automatic. P0–P9 remain the held pre-registrations and score
  against C′; stage-C fixed-nominal pooled grid stays the documented contrast arm.
  Smokes first (W32/K0/K6 × 2 ep: `[share]`+ρ̂ lines, checkpoint, no incident) → wave.
  Spend: 12 cells + 3 smokes, E23-C′ ledger (separate from the E24-vm toy 10/50).

## AGREED TAKEAWAY

**(the stage-C grid takeaways come after the Block-2 joint read; T1–T3 below are the IN-1k
3-cell + calculus-methods set, USER-APPROVED 2026-08-03 in-conversation)**

**E23-T1 (the 3-cell conjunction):** IN-1k cells separate on the conjunction (pos−rand
margin × effrank) at h.cls, not on either alone — vm3 takes max margin (.834) at collapsed
rank (.392) → worst kNN (.492); vm2 holds rank (.732) at the lowest margin (.732) → .529;
vm4 alone holds both (.813, .739) → best everything (.534 kNN / .642 lin / .6392 online),
corroborated z-side by the full-rank queue signature at scale (z.out effrank .977·d,
gauss_kl .014 — E21-T2 reproduced at IN-1k). Linear probes barely separate
(.629/.629/.642) — kNN and the paired plane carry the separation. Berker's conjecture
stands: high margin × high rank at h is the recipe.

**E23-T2 (head anatomy at 1k):** all three cells share the anatomy — probes rise monotone
L03→cls and decline through the head; the first head stage is the choke point (tap1:
gauss_kl ≈4.9, effrank/d →.13, Ω drop; the remaining path re-expands with residual Λ only
1.24–1.37, so selectivity is set at or before stage 1). One late-head inversion: vm4 rides
highest through the head in both probes yet its class margin at z.out dips below the other
two (.384 vs .440/.441) — the best-h cell does not carry the best class margin at out.

**E23-T3 (Λ portability + the metric ruling):** Λ(cls→z.out) is a property of the cell,
not the datascale — vm2 2.21→2.26, vm3 1.78→1.63, vm4 1.73→1.60 across IN-100→IN-1k, vm2
the family max at both. With the dimensionality result (d-tail: d256→d128 sits ON the
√dim null; a, b levels carry √(D_z/D_h) dimension mass + scale gauge; Λ = √(Ω_h/Ω_z)
cancels both), the standing usage rule: **cross-space and cross-model reads ride on Ω and
Λ; a, b are within-cell decompositions only** (per-direction â = a·√(D_h/D_z) when a
cross-D level is unavoidable). Mirrored as the D-068 addendum with the label-blindness
rule (Berker: the metric lens stays aug-accessible/label-free; class-conditioned
quantities are evaluation-side only; caveat — orbit overlap does not guarantee same-class
connectivity).

## ON-HOLD RECORD — stage-C′ finals, 12/12 to ep150 (2026-08-05; raw, P0–P9 scoring + joint read deferred per Berker)

All 12 vm-OAS share-pinned cells completed ep150 (chains 63025261–284, completeness
rule; zero dead cells). Final/best online probe:

| width lane (K2) | final | best | depth lane (2048) | final | best |
|---|---|---|---|---|---|
| W32 | .8313 | .8352 | K0 | .8420 | .8474 |
| W64 | .8555 | .8555 | K1 | .8609 | .8645 |
| W128 | .8629 | .8645 | **K3** | **.8790** | **.8808** |
| W256 | .8703 | .8718 | K4 | .8589 | .8617 |
| W512 | .8622 | .8673 | K6 | .8504 | .8507 |
| W64h0 | .8502 | .8530 | K6h0 | .8194 | .8239 |

2048/K2 references = the landed E24 v-cells (vc .8764 · v4oas .8716 · vb .8696;
vh0 .8713). Raw contrast vs pooled stage-C, same cells: W256/W512 were .37/.37,
K0/K6 were .45/.45 — all four land .84–.87 here. K3 .8808 = grid max, above every
E24 v-cell. Numbers only; no takeaway.

**Landing-chain incident (2026-08-04 ~19:05):** the self-lander's 12 extract chains
tripped the D-005 500 GB store cap mid-flight (store logical bytes 500.0 GB vs 294 GB
physical — the NFS backend compresses; `store.py` counts logical) → W64 extract dead,
6 cells' stores partial (missing meta.json), cascading audit FileNotFoundError +
probe StopIteration (no val manifest); only W32 landed fully, K3/K6/W64h0/K6h0
partially. Cap raised per Berker's 2026-08-05 ruling (D-077); partial stores wiped
and `e23_land_grid.sh` refired same day (idempotent).
