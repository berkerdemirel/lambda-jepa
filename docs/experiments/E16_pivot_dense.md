# E16 — PIVOT dense rebuild: position-conditioned prediction (the E15 fix), 2 variants

**Status:** pre-registered in-conversation 2026-07-13 on Berker's direction ("since we see them
failing i think they're wasting my compute … it is a failure. the question is how do we save the
pivot" → "fix it and rerun 2 variants"). DECISIONS D-033. Supersedes E15's void P2–P5.

## The fix (from the E15 post-mortem, joint)

E15 collapsed the position-indexed family of conditional laws into its scene-average
(uniform c_B unknown to the event ⇒ optimal h = average local mae texture; ~32 effective target
dims, 30% draw noise; loss solved by ep10, probe plateau ~.35). §6.1's "obtain corresponding
dense tokens by known crop maps" = position-conditioned DENSE prediction: the context at
position q supervises the event's token block AT q. Position-specific targets are deterministic
given (image, q) — the E15 draw-noise floor becomes signal.

## Construction (both variants)

- **Events:** foveal_v1 as E15 (sharp 96² fovea uniform over 81 slots, ×4 surround; shared hflip;
  two light-photometric views of the same event).
- **Dense local channel:** K=4 query positions q_k uniform over the 81 slots per image-step
  (shared across views). Target z_k = φ(frozen mae h.gap(sharp 96²@q_k → 224)) with the E15
  target freeze REUSED (`e15_target_v1.npz` — same crop distribution; σ_med 26.44, blocks
  2×192 σ∈{1,2}σ_med). Predictor side (head-less, §5.1 "fixed pooling operators"): the GAP of
  the event's 6×6 token block at q_k — implemented as avg_pool2d(kernel 6, stride 1) over the
  14×14 patch grid = the 9×9 slot grid exactly; gather at q_k. L_pred = mean_k over both views
  of SmoothL1(block_k − z_k).
- **L_view** = ||h.gap₁ − h.gap₂||²; **L_var** = hinge floor on h.gap (γ = .01645 unchanged,
  backstop role). **Transport: dropped both variants** (deferred; axis kept clean).
- **Variant e16a:** dense local channel only.
- **Variant e16b:** + **global anchor channel** (§3.2 "low-resolution whole-image view"; §6.3
  "stable low-frequency target blocks"): the CLS token regresses φ_g(frozen mae h.gap(×4
  down/up whole image)), own stats/RFF (`e16_global_v1.npz` from the budget job). DECLARED
  RE-ENCODE ANCHOR, not prediction — the event's surround already contains the low-res pixels
  (E14-T1 lesson kept explicit); its role is retaining global structure in h.
- λ_pred = 1; λ_view (+ λ_g for e16b) equal-pull-at-init re-measured on the dense objective;
  λ_var = 1 on the hinge. Frame/recipe = E15 verbatim (M2, bs 128, 100 ep, seed 0); ~4×8h
  singleton links per slot (K=4 ctx forwards raise step cost ~1.5–2×).

## Budget gate (D-033 METHOD RULE: no training on an unmeasured target)

Pre-launch numbers from `experiments/e16_budget.py` (mae φ at the fixed 3×3 slot grid per
train500/val image + low-res whole-image descriptors) + standard probes on pseudo-run
`in100.mae.s0.e16grid`:
- **C_gist** = probe of mean-of-9 φ (the gist ceiling; E15's C with better MC),
- **C_pos** = probe of concat-of-9 (3456-d; position-aware information range),
- **C_glob** = probe of φ_g(low-res image) (e16b's anchor ceiling),
- variance decomposition: within-image (position) share of Var(z).
**GATE: launch iff C_pos ≥ C_gist + .05 linear** (position-specific semantic content is real —
what the dense objective mines). All numbers land below before any training job.

## Pre-registered predictions (locked with D-033, before training)

- **E16-P1 (stability):** as E15-P1 (no incident; floor quiet or backstopping only).
- **E16-P2 (bar):** converged linear_raw_v2(h.gap) > C_gist on both columns (kill K2 if not).
- **E16-P3 (tiers):** P3a > mae raw .434; P3b ≥ vicreg .590 (aug-geometry competitive — the
  first genuinely interesting tier); P3c ≥ dino .686 (flagship; not expected).
- **E16-P4 (tension, descriptive):** pred_over_varz stays materially above its floor deep into
  training (the E15 failure signature — solved-by-ep10 — must NOT recur); position-conditioned
  tension persists.
- **E16-P5 (channel axis, descriptive):** e16b ≥ e16a on linear (global anchor retains gist);
  weak lean, recorded not gated.

## Kill criteria

**K1** incident trigger (house). **K2** converged probe ≤ C_gist ⇒ dense mechanism adds nothing
over gist re-encoding at this frame — reported, no rescue arms without a new decision.
**K3** floor-gaming signature (E12-f4). **K-gate** budget gate fails ⇒ no launch, back to
discussion.

## Sequence

budget job (H100, ~15 min) + probes → gate read → pull (λ's recorded here) → 2-ep smokes both
variants → zeroing-path read → chains (~4×8h links per slot). Numbers land raw; AGREED TAKEAWAY
only after joint discussion.

## Numbers land below this line as they arrive.

### Budget gate — 2026-07-13 (jobs 62251779 + 62251789): **PASS**

Var(z) total .001489; within-image position share **34.8%** (the signal E15 averaged away;
2-draw estimate said 30% — consistent). Probes (pseudo-run `in100.mae.s0.e16grid`):

| space | lin | knn | role |
|---|---|---|---|
| phi.mean9 (**C_gist**) | **.4420** | **.2494** | P2/K2 bar |
| phi.concat9 (**C_pos**) | .5096 | .2838 | position-aware range |
| phi.global (**C_glob**) | .3476 | .2368 | e16b anchor ceiling |

**GATE: C_pos − C_gist = +.068 ≥ .05 ⇒ LAUNCH.** Calibration note (declared): even C_pos sits
below vicreg (.590) — a static linear read of 9 sketches doesn't reach aug-method territory, so
P3b requires the trained trunk to exploit position-integration beyond static sketch content;
C_pos is a reference range, not a cap on a trained encoder. C_gist (.442) ≈ mae raw (.434):
the 9-draw marginal recovers what E15's 2-draw C (.378) undershot.

### Pre-launch λ (dense objective, job 62253782): λ_view = .6298, λ_g = 1.1228; floor inactive
at init (var 0, h_std .585, block_std .646); init pred .532 = 368× constant-h floor; global .856.

### Smoke read (2 ep, jobs 62253786 a / 62253787 b) — PASS with declared watch-items; chains launched

wandb shpmi4c2 (a) / 65nd7si5 (b). **The E15 failure signature does NOT recur**: pred_over_varz
ends the smoke at **26.1 (a) / 27.7 (b)** vs E15's 4.4/1.2 — the dense objective is far from
solved at ep2 (P4 tension confirmed at smoke scale). Floor quiet (var ≡ 0; h_std .038–.059 ≫ γ);
block_std .054–.070 approaching target scale; probes .017/.020 (warmup regime). Zeroing-path:
nothing gamed. **Watch-items (declared, with responses):** (1) grad spikes — a 7 isolated
steps >20 (max 227), b RECURRENT 32/1978 steps (max 286, five >130); the trainer's K1 EMA check
did not fire (EMA absorbs isolated spikes — known softness), grad_clip 1.0 bounded every update,
no curve destabilization. (2) e16b's view term RISES late in the smoke (.005→.113) while e16a's
settles — the global channel adds early cross-view tension. RESPONSE RULE: if e16b's view term
is still growing at peak-lr (ep10+) or any K1 fires, the arm stops at the next cadence ckpt and
goes to discussion; λ's stay frozen per protocol. **Chains: e16a = 62256149,62256150,62256151,62256152 (h100-slotA), e16b = 62256153,62256154,62256155,62256156 (h100-slotB), 4×8h links each; ~23h to ep100.**

### Norm audit — 2026-07-13 (Berker challenge: "shrinking the norm would satisfy MSE")

`results/figures/e16/e16_target_norms.png`. Target side: raw mae h.gap on ctx crops healthy
(norms 5.00±0.76, 0 dead dims); sketch = constant part ‖z̄‖ 1.19 + informative rms .73 (38%
energy) ⇒ one-sided shrinkage RAISES L_pred (frozen target = scale anchor; the BYOL co-shrink
mode is structurally impossible). Live finding: a REAL transient shrink phase ep~3–8 (h_std min
.0065/.0073, below γ) — the variance floor ENGAGED (active ~90% of steps since ep1.5, max 1e-4)
and with the pred anchor pulled scale back; the smoke's "floor quiet" did not hold past ep2.5 —
floor reclassified from backstop to LOAD-BEARING for this objective. Recovered equilibrium =
lawful conditional-mean scale: predicted σ_target·√(1−ratio) = .0233 at ratio .50 vs measured
block_std .022–.025 (both arms). Probes scale-free and rising (a .163@ep18, b .200@ep16; ratio
.50 and descending vs E15's exhausted 1.1 at same point). Standing unguarded subspace (declared):
within-block token structure — checked at extraction via token-level effrank.

*Precision + angle-loss audit (Berker Qs, 2026-07-13): bf16 quantization at our scales = .5% of
residual at ratio .5 (loss delta 0.00%; binds only ~ratio 1e-3–1e-4; z fp32, blocks cast fp32,
trunk internals O(1) via final-LN gamma). Angle loss rejected on measurement: cos(z, z̄)=.847±.068
— raw-z cosine is DC-dominated (informative angular spread .13 lives in centered space);
centered-cosine ≈ normalized MSE minus the scale anchor (anti-co-shrink), Theorem-1
identification, and instrument continuity. Cosmetic lever if precision ever binds: global sketch
rescale ×10 (Adam-invariant).*

### KILLED at ep66/60 — 2026-07-13 (Berker: "they failed as well, and we need a more fundamental
### change … this turns into a distillation setting where we try to recover mae semantics which
### is not even good")

Chains cancelled (62256149–156); cadence ckpts ep25/50 retained (ep75 not reached), NO comparison
membership. State at kill: e16a ep66 probe .344 (max .356) ratio .260; e16b ep60 probe .356
(max .362) ratio .307. The mechanical fact supporting the diagnosis (to CONFIRM next session):
E16 extracted the target far more deeply than E15 (ratio .26–.31 vs .44; position-conditional vs
marginal) yet the probe plateaued at the SAME ~.35 — mining mae semantics harder does not raise
representation quality; the ceiling is the teacher's content (C_gist .442 / C_pos .510, mae raw
.434), i.e., a distillation-bounded setting with a weak teacher. E16 P2–P5 VOID. Products
retained: budget-gate rule, tension dial, dense position-conditioned machinery, norm/precision
audits.
