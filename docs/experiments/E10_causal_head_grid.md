# E10 — Causal head interventions: move the loss, keep everything else

> Derived from docs/report/ssl-projector-gap-report.html §5.3 E10 — "the decisive experiment".

**Status:** draft · **Phase:** M4 (IN-100 retrain grid) · **Gate:** requires M2/M3 conclusions
USER-APPROVED (G-M2)
**Pre-registered:** ❏ pending (frontier predictions to be locked before the grid launches)

## Hypothesis

The h↔z dissociation is **caused** by head placement, and it trades off: enforcing desiderata at h
increases their satisfaction there while reducing nuisance retention and (for misaligned pretexts)
transfer breadth — a measurable frontier, not a cliff.

## Grid (~15–18 configs × 2 seeds, IN-100 ViT-S/16)

1. **VICReg**: var/cov terms at z (standard) | at h (expander kept for invariance only) | at both.
2. **SimCLR**: projector depth 0/1/2/3 (0 = DirectCLR-style loss on backbone sub-vector).
3. **BYOL**: trained predictor → DirectPred closed-form spectral predictor (stretch arm).
4. **LeJEPA**: standard (SIGReg on 3-layer projector, as shipped) | SIGReg moved onto the backbone
   (projector = identity) | projector depth 0/1/2/3 — *the* intervention that tests "impose the
   desideratum at h".
5. **DINO**: + KoLeo regularizer at h (the DINOv2 hint, isolated).

Full E01 battery + E04 ledger + (available) transfer suite on every run.

## LeJEPA SIGReg-on-h arm — detailed design (drafted 2026-07-02; **arms A–D USER-APPROVED
Berker 2026-07-08**; pilot launch pending D-016 sign-off)

**Motivating M1 measurement** (toy.lejepa.s0, kurt_topeig.worst, matched-Gaussian nulls ~0.1–0.2):
z.embed 3.62 → proj.tap1 16.08 → proj.tap2 7.47 → proj.out 1.52 — the head-depth profile is
NON-monotone; the trained statistic is satisfied only at the exact layer the loss touches, and the
hidden layers are wilder than the embed. Direction note: z.embed sits UPSTREAM of the projector
(CLS → Linear → z.embed → MLP → proj.out); what M1 shows is no *upstream* transfer of the
constraint — nothing exists downstream of proj.out to measure.

**Pilot design (superseded economy note; D-016 APPROVED 2026-07-08):** `toy.lejepa.s0` was
trained PORT-EXACT (lr 2e-3, no grad clip, eta_min lr/2 — D-011), so reusing it as arm A would
confound recipe with placement. Per Berker's ruling ("we treat every model equally… d011 was just
to do the reproduction"), the pilot trains **all three arms A/B/C under the house recipe**
(AdamW 1e-3, wd 5e-2, warmup 10 ep, eta_min 1e-5, grad_clip 1.0; lamb/V/proj_dim/emb_dim
unchanged; fresh per-step slices kept — loss mechanics, not hygiene). Placement
(`method.sigreg_at` ∈ proj/embed/cls, sslgap/methods/lejepa.py) is the only mover. Side product:
arm A vs toy.lejepa.s0 = a lejepa recipe-sensitivity datum. Run ids: `toy.lejepa.s0.e10{A,B,C}`.

**The linear-absorption confound, and why "on h" is three arms, not one.** F4 rules that a linear
map adds no *probe* capacity (hence LeJEPA h = z.embed). But SIGReg is not linear-invariant: a
trainable Linear can whiten any full-rank cloud, so a distributional constraint imposed after a
Linear can be absorbed BY that Linear, leaving the trunk anisotropic. "Impose isotropy at h"
therefore splits into inequivalent placements:

| arm | SIGReg applied to | inv term | trainable map after trunk | question it isolates |
|---|---|---|---|---|
| A (standard) | proj.out (16-d) | proj.out | Linear(384→512) + MLP(2048,2048,16) | baseline, as shipped |
| B | z.embed (512-d) | proj.out (projector kept) | Linear only, under the constraint | is ONE linear map enough buffer for a distributional constraint? |
| C | trunk CLS (384-d) | proj.out (projector kept) | none under the constraint | pure "desideratum at h", invariance still buffered |
| D (depth sweep) | output of proj depth d ∈ {0,1,2,3} | same point | d MLP layers | both losses move together — the report's head-free axis (OP-4/OP-5) |

B/C isolate SIGReg placement (single-factor); D moves the whole loss (comparable to the SimCLR
depth sweep). Arm-C h stays z.embed (the Linear remains, trained by inv-gradient only) — the
probed space is unchanged across A/B/C, which is what makes their battery rows comparable.

**Battery cells that decide it** (all E01 rows, both spaces + taps): `kurt_topeig`/worst-direction
isotropy at {CLS, z.embed, proj.out}; E04 aug-info retention at h; headline probes.
Pre-registered readings (PROPOSED):
- A vs C at h-isotropy: if C ≫ A at CLS-isotropy with equal probes, the projector was never needed
  for the *constraint* — only for the *invariance* term (sharpens the report's decision cell).
- B's signature: z.embed isotropic while CLS stays anisotropic = the Linear absorbed the
  constraint → the "buffer" for distributional pressure is as shallow as one linear map; then the
  F4 ruling (probe-side) and loss-side buffering formally dissociate — worth a THEORY_MAP note.
- If C degrades probes markedly where B does not, the trunk pays for isotropy in retained
  information (E04 ledger quantifies what was spent) — the frontier point the hypothesis predicts.

**Recipe discipline for the moved arms:** these are no longer the official recipe, so D-011's
exemptions lapse — house hygiene reapplies (grad_clip 1.0, eta_min ≤ lr/20, warmup ≥ 1 ep; needs
its own DECISIONS row before launch). λ stays fixed at 0.02 in the first pass — placement is the
only moving factor; SIGReg's slice statistics are dim-sensitive (16 vs 384/512-d, PROTOCOL §6.2),
so a λ mini-sweep is pre-authorized ONLY as separately-labeled arms if a fixed-λ arm collapses
(kill-triggers: probe < chance+5pts @ ep25, or kurt_topeig diverging).

**Toy pilot (PROPOSED, cheap):** arms A/B/C × 1 seed on the toy frame (~3 × 1 h H100) once M1
wraps, to de-risk λ-transfer and hygiene before committing H100-days at IN-100. Pilot numbers
calibrate the run plan only; frontier claims stay M4-gated.

## Rescue arms B′/C′/D0′ — measured-λ rebalance (pre-registered 2026-07-09, BEFORE launch)

**Authorization:** Berker 2026-07-09: "i will follow your recommendation on this. try to build as
healthy arms as possible for B/C/D. and you can use 3 h100s for this exp." (3 concurrent H100
slots granted for this experiment — exception to the standing 2-slot cap, D-021.)

**Design (minimal intervention):** official loss form and per-arm placement unchanged; the ONLY
mover vs the failed pilots is λ, chosen by measurement, not sweep. Rule: λ_arm is set so the
**init-time trunk-gradient ratio** ‖∇(λ·sigreg)‖ / ‖∇((1−λ)·inv)‖ on shared (trunk) parameters
matches **arm A's ratio at init** — A is the empirically healthy operating point, and
SIGREG_DIM_SCALING §6 says λ can fix this balance and nothing else. Ratios measured by replaying
training-condition batches on arch-matched seed-0 inits (`results/diag/e10_grad_share.csv`, job
62169172). λ values appended below once computed, before any submission.

**Escalation rule (pre-committed):** if λ_matched for an arm lands within ~2× of the pilot's
0.02, the balance hypothesis is wrong for that arm — do NOT burn the slot on a near-replica;
escalate to the projector-degeneracy contingency (spectral-normalized projector B″/C″) as a
separately-proposed arm instead.

**Hygiene:** house recipe per D-016 (unchanged); ep5 health gate as run for D0 (probe above bar,
both terms in live tension); kill-trigger probe < chance+5 pts @ ep25; grad-norm incident rule.
Runs `toy.lejepa.s0.e10{Br,Cr,Dr}`, 150 ep, 3-ep smoke first, slots h100-slot{A,B,C} singleton.
Post-chain: extract → audit → probe (probes = v2 family per D-020).

**Predictions (LOCKED):**
1. B′/C′ complete training without the decoupling collapse: no kill-trigger; offline linear at
   h.gap ends ABOVE the untrained baseline (randinit h.gap linear_raw ≈ .40), where pilots
   B/C ended at .16–.22.
2. **Shape prediction (the SIGREG_DIM_SCALING §6.2 test):** a healthy B′'s embed remains
   shape-non-Gaussian — `kurt_topeig.worst` at embed ≫ matched-Gaussian null (>10×) even with
   its train-sigreg settled near equilibrium — i.e. SIGReg at K=512/N=256 buys moments +
   anti-degeneracy, not shape. If B′'s embed comes out shape-tame, the scaling analysis is
   wrong somewhere and says so.
3. D0′: exceeds D0's best (.3358) and holds to ep150 without the late decay IF that decay was
   the balance mode arriving slowly; a decay under matched-λ means the mode is not
   λ-addressable. Either outcome is informative.
4. Buffer hypothesis: B′/C′ h-space probes ≤ A's h-space probes (the report's decision cell,
   now on healthy runs).

**λ values (filled from measurement before launch):** *(job 62169172 partial → rerun 62169472)*

**AMENDMENT (2026-07-09, after the ep0 measurement, BEFORE any rescue launch).** The measured
init rows refute the A-matched rule's premise: at init B's trunk balance is already A-like
(sig_share_trunk 0.854 vs A 0.899; λ_A-matched(B) = 0.030, within 2× of the pilot 0.02 → the
pre-committed escalation trigger FIRES). The pilot deaths were **dynamic**, not init-imbalance:
B's λ-scaled sigreg trunk-grad grows 7.2 → 78 by ep38 (share 0.9995, inv's grad 0.038) while
A's trunk share FALLS 0.90 → 0.32 — the projector is, measurably, a **gradient sink** that
absorbs the unattenuated moment-channel yank (init slice_var at embed = 0.11: sigreg's first
job is a ~10× scale correction, buffered in A, trunk-borne in B/C). Revised λ rule, replacing
A-matching: **equal-pull at init** — λ_arm s.t. λ·‖∇sigreg‖_trunk = (1−λ)·‖∇inv‖_trunk at the
seed-0 init (trunk share 0.5; between A's init 0.9, safe only when buffered, and A's
steady-state 0.3). From measured ep0 ratios (job 62169472, full CSV): r_B = 287.2, r_C = 699.2, r_D0 = 661.7 →
**λ_B = 0.00347 · λ_C = 0.00143 · λ_D0 = 0.00151** (λ = 1/(1+r)). Full-table readings for the
record: A resolves the init conflict into the projector (trunk share 0.90→0.32; init
cos(g_inv,g_sig) = −0.94 → −0.2); B/C = acute takeover (share ≥0.999 from ep38, inv ~1e-5,
C's sigreg thrashing 19→46→34); D0 = chronic balanced conflict (share 0.48–0.60, cos ≈ −0.65,
inv alive and rising 0.38→0.43 while sigreg falls) — its late probe decay happened UNDER
tension, so D0's λ targets a less sigreg-tilted equilibrium rather than preventing a runaway. Contingency
re-aimed by the same mechanism: if an equal-pull arm still collapses, the fix class is
init-scale calibration of the constrained space / gradient routing — NOT projector spectral
norm (the projector degeneracy in B was downstream of the trunk runaway, not its cause);
any such arm changes the official loss and needs its own proposal. Predictions 1–4 above
UNCHANGED and remain locked.

## Pre-registered expectations (report)

Desiderata-at-h variants: higher metric satisfaction at h, lower augmentation-info retention,
equal-or-better aligned-task accuracy, worse misaligned-task transfer — the Guillotine alignment
story, made causal. **LeJEPA decision cell**: if standard (projector) LeJEPA beats SIGReg-on-backbone,
the buffer is real even for a distributional constraint; if close, isotropy is gentle enough to
impose at h and the projector is dispensable for it. *Either answer is a finding.* [O]

## Compute

≈15–18 H100-days per seed pass at the 2-slot cap (ROADMAP M4).

## Results

*(numbers only)*

**Toy pilot live log (2026-07-08, runs toy.lejepa.s0.e10{A,B,C}, house recipe, λ=0.02):**
- ep43/150 online probe (chance 0.10): A (sigreg@proj) **0.604** · B (sigreg@embed) **0.102** ·
  C (sigreg@cls) **0.126** → the pre-registered kill-trigger (probe < chance+5pts @ ep25) HAS
  FIRED for B and C.
- wandb last-30-step term means @ep43: A inv 0.191 / sigreg 3.06 · B inv **0.00005** / sigreg
  69.9 · C inv **0.0001** / sigreg 38.0. (For reference, M1 randinit OFFLINE probes at h.gap:
  linear_raw 0.401 / knn 0.333 — offline probes of B/C pending run completion.)
- Disposition (Berker, in-session): B/C ran to completion for the autopsy battery. Finals:
  **A best .8168** (vs port-exact grid run .782 — recipe-sensitivity datum) · B best .237 /
  final ~.13 · C best .253 / final ~.12 (B/C "best" = pre-collapse peaks; battery on _ep150).
  gpu partition congested (30 pending ahead) → autopsy re-queued SEQUENTIALLY on h100-slotB
  (Berker-directed): extracts 62129785/86/87 → **arm D0** (toy.lejepa.s0.e10D0, job 62129788;
  proj_depth=0 → projector=Identity, BOTH terms on the 512-d embedding — the no-buffer
  configuration, where the B/C decoupling collapse cannot occur by construction; Berker's rule:
  health check at ep5, kill if unhealthy) → probes 62129790/92/94; audits 62129789/91/93 on
  defaultp CPU after each extract.
- **D0 ep5 gate: PASSED** (2026-07-08). Online probe .156/.235/.265/.239/.238 (ep1–5; bar .20);
  terms: sigreg 44.6 → 18.9 (descending; B's was stuck ~70), inv 0.19 → 0.36 (alive; B/C ~1e-4)
  — the two losses are in live tension at the embedding, no collapse signature.
- **D0 ep20 check: PASSED** — probe .326 and climbing (.29–.31 through ep13–19); vs the
  discriminator (B/C ≤ .15 by ep20; A ≈ .5). Learning, slower than arm A. Runs to completion.

**Rescue-arm live log (2026-07-09, runs toy.lejepa.s0.e10{Br,Cr,Dr}, jobs 62169804/06/08):**
- First launch attempt (62169750–55) died in smoke on Hydra override syntax (`+method.` prefix
  needed for sigreg_at/proj_depth) — zero GPU-hours lost, smokes doing their job.
- Resubmitted 62169803–08: all three smokes COMPLETED; fulls running.
- **ep5 gates**: Br .151/.244/.234/.225/.220 — PASS (pilot B flatlined ~.10); Cr
  .115/.241/.242/.251/.251 — PASS, monotone; Dr .111/.159/.148/.177/.182 — **judgment call:
  below D0's ep5 bar (.20) but monotone-climbing; the bar was calibrated at λ=0.02 where
  sigreg's early feature-conditioning push is 13× hotter, so Dr runs on; the pre-registered
  ep25 kill-trigger (probe < .15) remains binding.**
- **Br KILLED at ep26 (job 62169804 scancelled): ep25 probe .1261 < .15 trigger.** Trajectory:
  .20–.24 through ep11 → cliff ep12–13 (.138→.103, immediately after the 10-ep lr warmup
  reaches peak) → chance-flatline. **Prediction 1 FAILS for placement B at measured λ
  (0.00347, 5.8× below pilot): the init-balance/λ lever does not prevent the embed-placement
  collapse — it delayed it ~0 epochs relative to warmup.** λ is refuted as the rescue lever
  for B; remaining levers are official-loss changes (init-scale calibration, per-slice
  standardization, N↑, targeted slices — SIGREG_DIM_SCALING §6.3), each needing its own
  proposal. Ckpts on disk for optional autopsy: toy.lejepa.s0.e10Br_{best,last}.pt (best ≈
  ep2–4 pre-collapse).
- **Cr KILLED at ep31 (job 62169806): ep25 probe .1070 < .15.** Trajectory .20–.25 through
  ep18 → cliff ep19 → flatline. Same failure class as Br, cliff 7 epochs later. **Prediction 1
  FAILS for placement C at measured λ (0.00143) as well — the λ lever is refuted for BOTH
  buffered-constraint placements (embed and cls).**
- **Dr PASSES the ep25 trigger (.1730 ≥ .15) — runs on.** Caveat logged as numbers: the
  trajectory OSCILLATES .09–.19 with no trend through ep31 (D0 at λ=0.02 was at .33 by ep20,
  climbing) — the gate is satisfied as written; no other trigger applies before ep150.
- **Dr COMPLETED (job 62169808): best .2530 @ ~ep100, final .1878.** Milestones: ep50 .193 ·
  ep75 .179 · ep100 .253 · ep112 .192 · ep125 .182 · ep150 .188. **Prediction 3 resolves on
  its pre-registered branch: Dr did NOT exceed D0's best (.2530 < .3358) and the post-peak
  decay recurred (−26% from peak; D0's was −57%) — per the locked conditional, the no-buffer
  decay mode is NOT λ-addressable; additionally the 13× colder λ cost peak height (slower
  climb, lower peak).** Joint reading pending; wandb bi88weim. Post-chain queued:
  extract/audit/probe 62170277/78/79 (run_id toy.lejepa.s0.e10Dr.ext, v2 probes). All three
  H100 slots now free; Br/Cr autopsy batteries remain an offered option (ckpts on disk).

**Arm B‴ (e10Bi) — init-calibrated embed (pre-registered 2026-07-09 BEFORE launch; Berker: "i
still think we can do a trick to make it optimizable… i reckon we can solve it for healthier
training").** Design: pilot B EXACTLY (sigreg@embed, λ=0.02 official, house hygiene) + ONE
change, an optimization trick with the loss untouched: one-shot data-dependent init of the emb
Linear (first-batch per-dim μ,σ folded into W,b → embed starts unit-scale/zero-mean; guards
resume via extras). Root cause it targets: timm's head init is trunc_normal std .02, NOT
fan-in-scaled → embed starts ~10× under-scaled (CPU check: std .14 → 1.00 post-calib; matches
the measured .11 slice variance) — removing sigreg's unbuffered opening yank is the direct
causal test of the grad-share mechanism. **Predictions (LOCKED):** (1) mechanism-complete ⇒ no
runaway: survives ep25 (≥.15) and ends offline h.gap above the untrained baseline; (2) shape
prediction unchanged — embed ends kurt-wild ≫ matched null even at low sigreg; (3) if it still
collapses, the opening yank is not the (sole) trigger — mechanism story incomplete, informative
either way. Kill rules as before (ep25 < .15; grad-norm incident). Run toy.lejepa.s0.e10Bi,
h100-slotB, 3-ep smoke first.
- **e10Bi KILLED at ep25 (job 62173841): .1302 < .15.** Trajectory: climbed to .22 by ep6 (vs
  Br's cliff at ep12), then decayed through the post-warmup window to .11–.14 by ep15 —
  **softer slope, same destination. Prediction 3's branch resolves: the init scale-yank is a
  trigger but NOT the sole driver; unit-scale/zero-mean start delays, does not prevent, the
  embed-placement collapse.** With λ∈{0.02,0.0035,0.0014} and calibrated-init all failed, the
  tested optimization levers (weight, init) are exhausted for placement B — remaining candidate
  drivers are intrinsic to the constraint-at-512-d itself: the CONTINUOUS unbuffered
  moment-drift pressure at the trunk as features evolve (not just the opening yank), and the
  near-orthogonal (cos≈0) sigreg/inv gradient geometry measured at this placement (no
  productive tension). Any further arm changes the official loss (per-slice standardization /
  slice-count / N) and needs its own proposal. Ckpts: e10Bi_{best,last}.pt on disk.

**Instability forensics (2026-07-09, Berker's hypothesis; figure:
results/figures/e10/rescue_instability.png; wandb per-step pulls, warmup ends step 370):**
- **Confirmed: the B-family failure is an lr-triggered instability, timed exactly as Berker
  guessed.** Br and Bi train HEALTHILY through warmup — inv and sigreg descend jointly (Br ep8:
  inv .0009, sigreg 14.6↓, grad-norm max 1.35) — then the first grad-norm spikes land at steps
  433/435 (ep12) in BOTH runs, ~70 steps after lr peaks; storms reach 96–2439× (vs clip 1.0).
- **lr × placement, not lr alone:** arm A runs the identical schedule (peak 1e-3) with ZERO
  spikes over 145 ep (grad-norm max 6.2). Slice-sampling noise RULED OUT as driver: pre-collapse
  sigreg per-step CV 0.12 (Br/Bi) vs 0.17 (A) — no elevation at the embed placement.
- **Dr is a different failure entirely: zero spikes, stable grads — pure variance collapse.**
  λ=0.0015 too weak as the anti-collapse force: inv → 1e-5 (total victory) while sigreg pins at
  EXACTLY the derived σ²→0 ceiling (103) from ep10 — the SIGREG_DIM_SCALING prediction visible
  in a live training curve.
- Correction to the earlier live-log claim: "optimization levers exhausted" was premature — the
  lr regime was untested; Berker's curve-reading identified it.

**Arm e10Blr — low-peak-lr B (pre-registered 2026-07-09 BEFORE launch).** Engineering-mode arm
(healthiest-possible, per the standing mandate), not single-factor: sigreg@embed, λ=0.02
official, calibrated init KEPT (it bought the clean warmup), house recipe except **peak lr
3e-4** (warmup 10 ep and cosine shape unchanged, eta_min 1e-5 ≤ lr/20 ✓). Rationale: spikes
ignite only above some lr* for the unbuffered placement; 3e-4 tests whether lr* is practically
reachable. **Predictions (LOCKED):** (1) no spike storm (grad-norm stays <10× trailing median
through ep30); (2) passes the ep25 gate and trains to ep150 with a rising probe; (3) the shape
prediction stands (embed ends kurt-wild ≫ matched null); (4) if it storms even at 3e-4, the
instability threshold sits below practical lr and the placement verdict returns, airtight.
Kill rules unchanged. Run toy.lejepa.s0.e10Blr, h100-slotB, 3-ep smoke first.

**D-family forensics (2026-07-09, Berker's request — figure:
results/figures/e10/dfamily_forensics.png; wandb 6bv5b5cn/bi88weim, epoch table in session log):**
- **CORRECTION to the earlier live-log reading: D0's "late decay" was NOT decay-under-tension —
  it is the SAME gradient storm as B/C, igniting at ep32** (first >10×-median spike step 1151;
  storms to 10³–10⁵ max/ep through ep145; sigreg jumps off its floor 11→45 and acc crashes at
  the same epoch). The grad-share replay (fixed ckpts, fresh batches) could not see this —
  storms are on-trajectory events; the standing lesson "diagnose from the per-step curves at
  onset" (Berker) is now in memory as default practice. The metastability picture: the two-loss
  tension AT the embed holds ~22 epochs at peak lr (acc reaching .33 — best embed-space number
  in the family) before tipping; B tips at ep12, C at ep19, D0 at ep32.
- **Berker's "performance tracks sigreg" — quantified: corr(acc, −sigreg) = 0.63 for D0** —
  and the scaling note says why: at K=512, sigreg is effectively a live variance/degeneracy
  meter, so it moves one-to-one with representation aliveness. For Dr the correlation collapses
  to 0.24 because sigreg is PINNED at the ceiling (no signal) while the probe wobbles on
  LN-amplified residual structure.
- **Dr late-window anomaly (unscored, for discussion):** from ~ep100, as cosine lr decays,
  sigreg finally wins ground (103 → 70; variance reopening) yet acc DROPS .25 → .18 — the
  reopening appears to reshape the small structure the probe had been reading.
- Implication for e10Blr's watch: D0 proves storms can ignite ~20+ epochs after the lr peak —
  passing the ep25 gate does NOT clear Blr; the no-storm prediction is judged over the full run.

**e10Blr KILLED at ep52 (job 62174151; rule stated in-session at ep41, executed after ep50
confirmation).** The healthiest embed-placement trajectory ever recorded — ep25 gate PASSED
(.2571, first in family), sigreg reached 5.2 (siblings: 11–45), inv 2e-4, probe .28 at ep23 —
then the SAME storm ignited in slow motion: gnorm median 1.1→11.7 over ep29–41, sustained 8–20
with maxima to 1100 through ep52; probe .28→.16. **Prediction 4 resolves: ignition time scales
with lr but ignition is not eliminated — measured fuse lengths: ep12 (B, 1e-3) / ep19 (C, 1e-3)
/ ep32 (D0, 1e-3) / ~ep38 (Blr, 3e-4). The no-buffer placement carries a finite-time
instability at every tested (λ, init, lr); the projector removes the fuse (A: 145 ep, max gnorm
6.2). Notably the storm ignites FROM the jointly-best loss state (sigreg 5.5 + inv 2e-4) — the
tension state is reachable and good but not stable without the buffer.** Ckpts for the "best
achievable without buffer" cells: e10Blr_{best≈ep23, ep38}.pt.

**Arm e10Dlr — the D-story arm (pre-registered 2026-07-09 BEFORE launch; Berker: "making both
good enough on arm D tells a better story").** Both losses at the representation (proj_depth=0,
no projector anywhere), λ=0.02 (the value that produced D0's real tension and .33 peak),
calibrated init, peak lr 3e-4 (the fuse-stretcher measured on Blr). Companion diagnostic
launched with it (experiments/sigreg_ref_check.py, job 62176059 → results/diag/sigreg_ref.csv):
decomposes each arm's sigreg value into N(0,I) floor (1.053) + second-moment part
(covariance-matched-Gaussian excess) + shape/degeneracy residue — the reference Berker asked
for to interpret "sigreg ≈ 5". **Predictions (LOCKED):** (1) fuse-scaling: if Blr's ×3 stretch
transfers, ignition lands ~ep90–110 (D0's ep32 × ~3) — either a LATE storm (second placement
point on the fuse-length law) or survival to ep150 (the D-story realized); (2) "both good
enough" defined measurably: at end/best, sigreg's shape/degeneracy residue over its
covariance-matched Gaussian < 1.0 AND inv at-or-below D0's pre-storm plateau (~0.35) with probe
≥ D0's best (.3358) or still climbing; (3) shape prediction unchanged (embed kurt-wild ≫ null
— moments-only theory); (4) kill rules: ep25 probe < .15, OR sustained storm = grad-norm
epoch-median > 10 for 3 consecutive epochs (the rule improvised for Blr, now pre-committed).
Run toy.lejepa.s0.e10Dlr, h100-slotB, smoke 62176060 → full 62176061.
- NOTE (2026-07-09, mid-run, before Dlr endpoint data): the sigreg_ref measurement
  (results/diag/sigreg_ref.csv) shows the shape-residue bar in prediction 2 is near-automatic
  at K=512 (all trained embeds read ≈ their covariance-matched twin). The criterion stands as
  locked, but the DISCRIMINATING quantity for the D-story is the MOMENT part (Blr_best 6.2,
  D0-end 16.0): how close Dlr's embed covariance gets to I, and whether it stays there.

**e10Dlr COMPLETED (2026-07-09 evening, job 62176061): best .5432 (~ep145), final .5383 — NO
storm, NO decay, monotone 150 epochs.** Milestones .259(5)/.334(20)/.372(30)/.400(40)/
.449(65)/.490(95)/.513(110)/.543(145); gradient checks at ep32 and ep91–99 quiet (median ~1.3,
max <3.5 — quietest run in the family incl. A); terms co-descending throughout (end: sigreg
~7.7, inv ~.25). vs D0 (best .336→final .145) and every killed sibling. Fuse-scaling storm
branch falsified for this arm; predictions 1–2 quantitative reads + prediction 3 (shape) await
the battery. Post-chain queued: extract/audit/probe 62178368/69/70 (toy.lejepa.s0.e10Dlr.ext);
moment-part trajectory on cadence ckpts queued (sigreg_ref_check + Dlr_ep{38,75,112,150} cells,
job 62178372). ALL numbers land raw — the joint analysis is next session's item 2 (Berker:
"we will wait untill Dlr finishes and then analyze the results"). H100 3-slot grant now LAPSED
(D-021); deitlite still holds slotA until ~ep100.

**e10Dlr moment-part trajectory (results/diag/sigreg_ref.csv, job 62178570) — numbers only:**

| Dlr ckpt | T_actual | cov-matched twin | moment part | shape residue |
|---|---|---|---|---|
| ep38 | 11.56 | 10.28 | 9.22 | 1.29 |
| ep75 | 10.00 | 8.64 | 7.59 | 1.36 |
| ep112 | 9.09 | 7.37 | 6.32 | 1.71 |
| ep150 | 7.93 | 6.36 | 5.31 | 1.57 |

Two patterns for the joint reading: (1) the moment part descends MONOTONICALLY 9.2 → 5.31,
ending below Blr_best's 6.23 — the flagged discriminating quantity improves all run and never
reverses; (2) unlike every other 512-d cell (residue ≈ 0 ± 0.5), Dlr's shape residue is
consistently POSITIVE and ~1.3–1.7 (locked prediction-2 bar "<1" is therefore NOT met as
written) — a shape deviation large enough to survive slicing at K=512 typically indicates
low-dimensional/cluster structure (cf. the MC's 10-cluster cell reading 25–27; a partial
cluster geometry reads ~1–2). Whether that residue is the GOOD kind (semantic clustering
emerging in the representation) is exactly a battery question — toy.lejepa.s0.e10Dlr.ext is on
disk. UNSCORED; joint analysis next session.

## AGREED TAKEAWAY

*(empty)*
