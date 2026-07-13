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
