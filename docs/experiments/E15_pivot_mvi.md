# E15 — PIVOT rung-2: the MVI training arms (does the mechanism beat its own frozen target?)

**Status:** pre-registered in-conversation 2026-07-13 and USER-APPROVED (Berker: "we have 2 h100s
right do one with l transport and one without. context position should be uniform. rest is good.
… lets go!"). DECISIONS D-032. First PIVOT-native training; rungs 0–1 (E13/E14) exhausted the
zero-training checks — this tests the mechanism, undiluted.

## Question

Does direct regression of a retained token block onto sketched frozen-reconstruction-target
descriptors of withheld context — from scratch, with view-consistency and a variance floor —
produce an h that EXCEEDS what its frozen target carries (the capped-distillation ceiling)?
Per proposal §5.7/§6.1 (MVI) at the M2 frame.

## Arms (Berker's call: both transport variants, one per H100 slot)

| arm | run_id | terms | slot |
|---|---|---|---|
| A | `in100.pivot.s0.e15a` | L_pred + λ_view·L_view + λ_var·L_var | h100-slotA |
| B | `in100.pivot.s0.e15b` | A + λ_T·L_transport (hflip token-reindex) | h100-slotB |

## Frame & recipe

M2 `in100_vits16` verbatim where applicable: ViT-S/16 @224, 100 ep, seed 0, AdamW lr 1e-3
wd 5e-2, house scheduler (warmup 10 ep, cosine η_min 1e-5), grad_clip 1.0, bs 128, bf16,
drop_path 0.1, online-probe monitor on detached h.gap, ep{25,50,75,100}+best+last, wandb online,
8h chunks × 3 singleton links per slot.

## Construction

- **Events:** foveal_v1 geometry (sharp 96², ×4 surround). Per image per step: ONE fovea position
  c uniform over the 81 aligned slots; base = Resize(224)+CenterCrop(224) + shared hflip(p=.5)
  BEFORE position sampling (views+ctx live in one frame); two event views = independent light
  photometric variants (ColorJitter .4/.4/.2/.1 p=.8, grayscale p=.2) of the same event —
  §5.4's equivalent-event condition (different fovea positions are DIFFERENT events). Declared
  risk carried from E14-addendum: the foveal input has zero rung-1 support for frozen models;
  rung-2 is the test of the trained story.
- **Context/target:** c_B uniform over slots, independent of c (Berker's call) ⇒ h regresses
  toward the position-marginal conditional expectation (§5.3 stochastic-target logic). ctx =
  CLEAN sharp 96² crop at c_B → 224 (no photometric — target stability, declared).
  z = concat of two 192-d RFF blocks (σ ∈ {1,2}·σ_med) of the standardized frozen
  `in100.mae.s0` h.gap on ctx; stopgrad; **total sketch = 384 = the retained block dim (§5.2
  step 3)**. Standardization stats, σ_med, γ, and RFF W/b measured ONCE from the stored E14 ctx
  features (`in100.mae.s0.ext/pairs100@foveal_v1_ctx`, 20k samples of exactly the training ctx
  distribution), frozen to `outputs/e15_target_v1.npz` (seed 15), sha recorded in run cfg.
- **Head-less (§5.1):** L_pred = SmoothL1(h.gap(view_i) − z), both views regress the same z; NO
  projector/predictor/EMA — the trained block IS the audited h (D-003 unchanged: h = trunk
  forward_features; extraction adapter entry `pivot` = backbone-only branch).
- **L_view** = ||h₁ − h₂||². **L_var** = mean_j [γ − Std_batch(h_j)]²₊ over concat(h₁,h₂), γ =
  0.5× median per-dim std of the target sketch (from prep; inactive unless collapse-ward;
  constant λ_var, no annealing discretion).
- **L_transport (arm B):** g = hflip only at MVI (exact token reindex on the 14×14 grid, M_g ≡ 1;
  crop/warp transport deferred — declared deviation from §5.7's four-term-with-warps). Implemented
  as SmoothL1 between column-flipped patch tokens of hflip(view₁) and patch tokens of view₁.
- **λ protocol (D-021/D-026 lineage):** λ_pred = 1; λ_view and λ_T by equal-pull-at-init measured
  on real batches (`experiments/e15_pull.py`, H100, values recorded HERE pre-launch); λ_var = 1 on
  the hinge (inactive at init by γ construction). No mid-flight changes.

## The ceiling C (computed pre-launch, scoring-only)

C = house `linear_raw_v2` / `knn_v1_k200` on the position-marginal target itself: φ-blocks of
frozen mae h.gap over 2 ctx draws per image, averaged — extracted for train500+val manifests
(`experiments/e15_ceiling.py` → pseudo-run `in100.mae.s0.e15phi`), probed by the standard driver.
This is the capped-distillation comparator (includes the sketch's nonlinear-readout gain).
Reference points: mae lane raw h.gap .434 lin / .240 knn; vicreg .590/.394; lejepa .602/.524;
dino .686/.599.

## Pre-registered predictions (locked with D-032, before any training)

- **E15-P1 (stability):** both arms train 100 ep, no incident trigger; floor never gamed —
  effrank(h) stays above the randinit null band (f4 signature = declared failure mode).
- **E15-P2 (THE mechanism bar):** converged linear_raw_v2(h.gap) > C on both columns. Kill K2 if
  ≤ C: capped-distillation outcome — the mechanism added nothing beyond its frozen target at this
  frame; reported as such, no rescue arms without a new decision.
- **E15-P3 (tiered context bet):** P3a clears C and mae raw decisively (mechanism alive); P3b
  reaches vicreg .590 lin (competitive with aug-invariance geometry at matched frame); P3c dino
  .686 (flagship-grade; not expected at MVI).
- **E15-P4 (instrument coherence, descriptive):** within-run far-channel D_read (T=mae, E14
  machinery on cadence ckpts) falls over training and anti-tracks probe growth; D_kern vs own
  target becomes meaningful (E13-T2's surviving "loss pins the rotation" hypothesis, first test).
- **E15-P5 (transport, descriptive):** arm B ≥ arm A on kNN (transport pins token geometry) —
  weak directional lean, recorded not gated.

## Kill criteria

- **E15-K1:** single-step grad-norm > 100× running median ⇒ INCIDENT, stop and read (house rule).
- **E15-K2:** converged probe ≤ C ⇒ capped-distillation outcome (see P2).
- **E15-K3:** L_var satisfied + effrank below randinit null band ⇒ f4-class variance-floor gaming;
  recorded, arm stops at the next cadence point.

## Cheapest-zeroing-path audit (per term, read at the 2-ep smoke; suspects from D-029)

- L_pred: constant-h floor = Var(z) measured at prep — "predict the mean sketch" is a quantified
  floor, watched via pred-loss vs Var(z) ratio.
- L_view: zeroable by collapse — guarded jointly by L_pred (z varies per image) + L_var.
- L_var: gameable by noise dims (E12-f4) — effrank + kurt-of-top-eig monitored per epoch.
- Tokenizer leakage: T=mae trained on the same IN-100 train split — declared; target-only
  transfer already on the books (.434/.240).

## Sequence

prep (CPU, stored arrays) → ceiling extraction+probe (H100 + gpu) → pull measurement (H100) →
λ's recorded here → CPU dry-run → 2-ep smoke both arms (H100 slots) → zeroing-path read → chain
launch (3×8h links per slot). Numbers land raw; AGREED TAKEAWAY only after joint discussion.

## Numbers land below this line as they arrive.

### Pre-launch measurements — 2026-07-13

- **Target freeze** (`outputs/e15_target_v1.npz`, sha256[:16] d613544ceeb25dec, prep run local
  CPU over stored ctx arrays): σ_med 26.440 (E14's independent per-direction estimates: 26.51 /
  26.30 — consistent); γ = .01645; Var(z) = .001446; target per-dim std min/med/max
  .0057/.0329/.0691.
- **Equal-pull at init** (job 62249062, A40, bs 128 real batch): grad norms pred 1.145 / view
  1.516 / transport 0.682 / var 0 (floor inactive as constructed) ⇒ **λ_view = .7554,
  λ_T = 1.6804** (frozen into `configs/method/pivot.yaml`). Init readings: pred .460 (= 318× the
  constant-h floor — the first learning signal is scale-shrink onto the sketch, expected for the
  head-less design), view .382, transport .157, init h-std .585.
- **THE BAR — C (ceiling probe, jobs 62249061 extract + 62249073 probe):**
  **C = .3780 linear_raw_v2 / .2376 knn_v1_k200** (φ-marginal over 2 uniform ctx draws,
  train500/val, pseudo-run `in100.mae.s0.e15phi`). C < raw mae h.gap (.434/.240): position
  marginalization + RFF compression cost linear content. P2/K2 read against C; mae raw is the
  secondary reference.

### Smoke read (2 ep, jobs 62249071 A / 62249072 B) — PASS; chains launched

wandb 20o8hc6h (A) / fsa608i2 (B). Per-term zeroing-path read: **no gaming signature on any
term.** pred .460 → .0064 (A, 4.4× constant-h floor, still falling) / .0017 (B, 1.19× floor);
h_std .585 → .0335 (A, ≈ target scale) / .020 (B, 20% above γ — floor armed, never fired,
var ≡ 0 both); view → ~.004/.0003 via photometric invariance, not collapse (std ≫ 0); transport
(B) .157 → .0006; grad_norm smooth, max excursion 16× median (A), well under the 100× trigger.
Raw observation (no reading): the transport arm optimizes visibly faster at fixed step count.
`pred_over_varz` (1.0 = constant-h floor; <1 = image-specific prediction) is the primary
training-time mechanism dial — at smoke end neither arm is below 1 yet (warmup regime, lr at 20%).
**Chains: e15a = 62249105/6/7 (h100-slotA), e15b = 62249108/9/10 (h100-slotB), 3×8h links,
~13h to ep100.**

### Mid-training forensics — 2026-07-13 ~ep71 (Berker flag: "loss too small, not enough tension")

Raw read (wandb duln2haw / l26e167n): `pred_over_varz` crossed BELOW the constant-h floor (1.0)
at ~ep10 and is still descending — A .441, B .515 at ep~71 — so the objective is NOT trivially
satisfied; h carries image-specific predictive content (~56% / ~49% of target variance beyond the
mean). h_std .0227/.0226 (≈⅔ target std, conditional-mean shrinkage; floor quiet, var ~1e-5).
Online probe PLATEAUED: A ~.34–.35, B ~.33 since ~ep63 (lr in final decay).

**The tension diagnosis (new number):** the 2-draw estimate from stored ctx pairs gives
E Var(z|image) = .000434 ⇒ **perfect-image-predictor ratio bound = .300** — 30% of target
variance is irreducible position-draw noise even with the full sharp image (the event-limited
bound is HIGHER). Arm A has therefore extracted ≈80% of the maximally extractable signal
((1−.441)/(1−.300)); remaining headroom ≤ .141 ratio, partly irreducible. Berker's read holds in
sharpened form: not "objective satisfied trivially" but "target nearly mined out by ep70" — the
binding constraint is the TARGET'S information content (one frozen mae, single 96² ctx, K=1 draw),
not optimization. Projection (not a verdict): online ~.35 sits near-below C (.378) and below mae
raw (.434); converged offline probes on final ckpts decide P2/K2. No mid-flight changes (per
D-032); levers for the landing discussion (E16-class): K>1 ctx draws per step (§5.3's own
variance-reduction), multi-channel/larger contexts (§3.2), richer reconstruction tokenizer.

### KILLED at ~ep74 — 2026-07-13 (Berker: "they're wasting my compute. with that accuracy they
### shouldnt be a part of any comparison. fix it and rerun 2 variants.")

Runs cancelled (62249105–110); ep25/50 cadence ckpts retained, NO comparison membership, no
probes/audit. Verdict (in-conversation, joint): the E15-MVI instantiation is INFORMATION-STARVED
— my pooled-h.gap reading of §5.3/§6.1 collapsed the position-indexed family of conditional laws
into its scene-average (c_B unknown to the event ⇒ optimal h = average local mae texture; ~32
effective target dims, 30% draw noise; loss ≈ solved by ep10; probe plateau ~.35). §6.1's
"obtain corresponding dense tokens by known crop maps" reads as position-conditioned dense
prediction — E16 rebuilds on that. Salvage kept: head-less direct regression is STABLE without
EMA/projector (P1 evidence), and the target-budget toolkit (C, Var(z), E Var(z|·)) becomes a
mandatory pre-training gate (D-033). E15 P2–P5 are VOID (never tested at a meaningful ceiling).
