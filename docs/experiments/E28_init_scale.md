# E28 — backbone initialization scale as an h-space knob (×10 init, no h regularizer)

**Status: USER-DIRECTED 2026-08-13 (Berker, verbatim): "Run the existing IN100 SSL pipeline
with ViT-Small. Use the current setup unchanged except for the intervention below: keep the
existing invariance loss on post-MLP (Z); keep our existing post-MLP covariance/log-det loss
on (Z); do not regularize (H); multiply all learned affine weights in the ViT backbone by 10
at initialization (attention Q/K/V/output weights, MLP/FFN linear weights); do not scale
LayerNorm or BatchNorm parameters, biases, positional embeddings / CLS token, projector
weights. … We want (H) to be the unnormalized final ViT representation. Move the ViT's final
LayerNorm to the start of the projector … Verify that for the ordinary unscaled
initialization this relocation leaves (Z) numerically unchanged. … For this experiment,
please do not contaminate the existing codebase. Keep the script as separate as possible …
You can use gpu100 partition."**

**No DECISIONS row is written this session** (Berker: "do not change anything else in the
repo") — one is OWED if this lane is kept: the LN relocation changes what `h` means for the
extraction/landing pipeline (see §Landing implications).

**CLOSED 2026-08-13 (Berker: "it was a good try but e28 failed") — every job stopped, H100
budget returned to the E27 program.** See §AGREED TAKEAWAY. Final states at cancellation:
wd=0 arms ×1 ep12 (h effrank 10.4) · ×5 ep9 (34.3) · ×10 ep12 (20.9); earlier wd=5e-2 arms
×10 ep~45 (37.6, probe .085 peak at ep15) · ×5 ep25 (h_ln 45.5, probe **.4898** — ahead of the
×1 control's .4638). No lane reached ep100; the negative is read from matched-epoch
comparisons against the landed ×1 control, under three controls (decay off, temperature held,
×1 attribution arm).

Code: `experiments/e28_scaleinit.py` (standalone), `slurm/e28_scaleinit.sbatch`,
`slurm/e28_launch.sh`, results in `results/e28/`. **Nothing under `sslgap/`,
`experiments/train.py` or `experiments/configs/` was modified** (verified by `git diff`) — the
script imports and subclasses only. The diagnostics written for E28 (spectra, gauge
decomposition, view alignment, per-family displacement) are experiment-local by request and
would have to move to `sslgap/metrics/` before any reuse (D-054).

## Question

Does raising the BACKBONE's initialization scale alone — with no regularizer of any kind on
h — induce stronger feature learning and a higher-rank / higher-stable-rank h?

This is the D-100 h-moment question asked from the opposite side. The ±h-moment pairs bought
h-rank by paying a term in the loss (B-rank ×2–3, ours 181→371). E28 asks whether a free
parameter of the initialization buys the same geometry for nothing.

## The lane (held fixed) and the two deltas

**Lane = `in100.floorssl.s0.d256vm4zonly` VERBATIM** — the D-100 z-only twin: MSE invariance
AND the two-sided spectral conditioner both behind the MLP (`h_lamb=0`), which is exactly
Berker's "keep the invariance loss on Z, keep the covariance/log-det loss on Z, do not
regularize H". Overrides recovered from the reference checkpoint's stored cfg (not retyped):
`aug=lejepa` V=4 · bs=128 · `expander_dim=256`, bn expander · w_inv 32.8 / w_floor 157.8 /
h_lamb 0.0 · view-mean payment at both taps, `z_d_slice=h_d_slice=128`, `queue_steps=3` ·
AdamW lr 1e-3, wd 5e-2, 10-ep warmup, cosine → 1e-5 · drop_path .1 · 100 epochs · seed 0 ·
the online-probe monitor. **Enforced mechanically at every job start**: the resolved config
is diffed against the reference checkpoint's cfg and the job dies unless the only differing
keys are `tag` and `e28.*` (the twin-launch rule; the zonly lane itself was born from a
hand-copied override set that silently took `bs=256`/`num_classes=10`).

**Delta 1 — h := the unnormalized final representation.** timm applies the trunk's final
LayerNorm inside `forward_features`. That module is MOVED (same object, same weights, no new
RNG draw) to the front of the projector:

    h = ViTBlocks(x),    z = MLP(LN_final(h))

so z is the same function of x as before while h is read pre-norm. `training_step`'s `cls` is
now the raw h: it feeds the detached probe features and the zero-weighted h-conditioner (the
term is still computed, as in the reference lane, for RNG-stream parity — it contributes
neither loss nor gradient at `h_lamb=0`).

**Delta 2 — ×10 on the backbone's weight matrices at init.** Exactly 49 tensors: the 48
`blocks.*.{attn.qkv, attn.proj, mlp.fc1, mlp.fc2}.weight` (Q/K/V are fused in timm's `qkv`)
plus `patch_embed.proj.weight`. Untouched: every LayerNorm/BatchNorm parameter, every bias,
`pos_embed`, `cls_token`, and the whole projector including the moved LN.

> **`patch_embed.proj.weight` is INCLUDED** (Berker 2026-08-13: "i think patch embed proj
> should be scaled too") — 49 tensors, not 48. Attempt 1 ran ~10 epochs without it under the
> declared reading of the spec and was superseded; its artifacts are parked in
> `results/e28/attempt1_blocksonly/`. Measured consequence: essentially none at ×10 (init h
> effrank 182.0 with vs 180.3 without) — block 0 sees LN(x₀), which is blind to x₀'s scale, so
> what including it actually changes is the image-content : positional-signal ratio inside
> x₀ = patch_embed(img) + pos_embed, since pos_embed and cls_token stay unscaled per the spec.

## Verification (gate; re-run before every training segment)

`+e28.mode=selftest` — §1 relocation, §2 scaling audit, §3 health at ×10. It runs at the head
of every segment; the numbers below are from the arm's own gate (`outputs/e28x10_63437776.out`,
H100) and the standalone gate 63437128 (A40) — identical on every check:

| check | result |
|---|---|
| §1 `max｜z_ref − z_new｜` at ordinary init, fp32 | **0.0 (bit-identical)** |
| §1 same, under bf16 autocast | **0.0** |
| §1 `max｜h_ref − LN_moved(h_new)｜` | **0.0** |
| §1 params | trunk bit-identical, final LN moved verbatim, expander bit-identical |
| §2 scaled tensors | **49** = 4 block families × 12 + patch_embed.proj; every other tensor bit-identical to the ×1 build |
| §2 untouched (asserted) | biases, block LayerNorms, cls_token, pos_embed, projector incl. the moved LN |
| §3 h rms at init | 0.580 (×1, raw) → **92.5 (×10)**; **z rms 0.098 → 0.097** — LN pins z's scale |
| §3c one AdamW step at ×10 | terms finite (inv .225, z-kl 3.32 on a degenerate bs=32 batch), grad_norm 1.9e4 → clipped at 1.0, 0 non-finite params |

**Berker's verification request is answered: the relocation is bit-identical for Z**, in fp32
and in bf16. The only things it changes are what we read as h and what the online probe's own
LayerNorm receives.

## The spectator wall (DECLARED DIVERGENCE — the one place this run is not the lane verbatim)

The first gate (63436960) **crashed** in §3:
`torch._C._LinAlgError: linalg.cholesky … not positive-definite` — inside `h_moment_kl`, the
term this lane weights at **exactly 0**.

Diagnosis. The reference lane still COMPUTES the h-conditioner at `h_lamb=0`, deliberately, so
the per-step RNG stream stays aligned with the h-treated arms (the D-063 zonly convention).
With `queue_steps=3` the ring is empty on the first step of every job, so the estimator sees
n = bs = 128 rows in a d′ = 128 slice — after centering that scatter matrix is *structurally*
singular. `SpectralConditioner` carries an **absolute** ridge `eps=1e-4`, which is ≈3e-4
relative to h's eigenvalues at ×1 (Cholesky survives, this is what the reference lane did all
along) but ≈1e-8 once h is read unnormalized at ×10 — below fp32 resolution. Measured:

| tap | rms | n=128 (step 1) | n=256 | n=512 |
|---|---|---|---|---|
| h, ×1 | 0.58 | 0.720 | 0.373 | 0.285 |
| h, ×10 | 93.7 | **raises** (CPU probe + gate 63436960); **4426.9** in gate 63437128 | 4405 | 4407 |
| z (the objective's term) | 0.098 | 2.86 | 1.97 | 1.89 |

The ×10 h row is the signature of a numerically singular matrix: whether `cholesky` returns or
raises is decided by rounding, so this is a **stochastic** crash at the first step of every
segment — including every requeue/resume, where the ring re-warms from empty.

**Resolution (re-derived, not patched).** At `h_lamb=0` the term exists only for RNG parity,
and Cholesky/QR consume no RNG. `experiments/e28_scaleinit.py:SpectatorConditioner` therefore
draws the *identical* random slice — same shape, same call order — and returns a gradient-free
0 instead of factorizing a matrix nobody uses. **Verified: the global RNG state after the call
is bit-identical to the stock conditioner's.** The loss, every gradient, the optimizer path and
the RNG stream are the reference lane's; h's geometry is measured far better by the per-epoch
spectra than by the KL this would have logged. The only visible consequence is that
`train/h_moment_kl` logs a flat 0 for this run.

`sslgap/methods/_common.py` was NOT touched. Standing observation for Berker (information, not
a change): **the conditioner's ridge is absolute, so the estimator is not scale-free.** Any
future arm that conditions on an unnormalized tap — including this lane with `h_lamb > 0` —
needs a ridge relative to tr(Σ)/d′ before it can be trusted at all.

## The ×1 control already exists (and is free)

At scale=1 this script's trunk+projector gradients are **identical** to `d256vm4zonly`'s: z is
unchanged (§1), the h-conditioner is zero-weighted, and `probe_feats` is detached — so only
the online probe head sees a different input. The landed run is the control (ep100 online
**.7048**). Its h_raw/h_ln/z geometry and displacement are recomputed with the SAME code at
ep 0/25/50/75/100 (`+e28.mode=control`, ep0 = the seed-0 init rebuild) → the treated lane's
per-epoch curve is read against a 5-knot control curve, not against nothing.

## Diagnostics (per epoch, full val split, deterministic eval transform, fp32, eval mode)

Trajectory-invisible: no grad, BN buffers untouched in eval mode, RNG forked and restored.

- **eigenspectrum** (float64 centered covariance) of **h_raw** (unnormalized), **h_ln**
  (= LN_final(h_raw), the ordinary ViT h) and **z** — full spectra to
  `results/e28/<run_id>_spectra.npz`, keyed `ep{e}.{space}`.
- per space: rms, ‖mean‖, trace, λ₁, **effective rank** (exp-entropy) and effrank/d, **stable
  rank** tr(Σ)/λ₁, participation ratio, **RankMe**, α (−slope of log λ vs log rank),
  top-10 variance fraction, #{λ > 1e-3·λ₁} → `results/e28/<run_id>_diag.csv`.
- **relative parameter displacement from init** ‖θ−θ₀‖/‖θ₀‖ and the weight-norm ratio
  ‖θ‖/‖θ₀‖, for: whole backbone · the scaled 48 · qkv/attn_proj/fc1/fc2 · block LNs · biases
  · patch_embed · pos+cls · the moved final LN · the projector MLP · every block →
  `results/e28/<run_id>_disp.csv`. θ₀ is snapshotted to disk before the first step so the
  reference survives requeue.
- everything above also streams to wandb (`sslgap`, run `in100.floorssl.s0.e28x10`) as
  `diag/<space>/<metric>`, `disp/<group>`, `wnorm/<group>`.

Reading the CSVs: they are APPEND-only, so a requeue that resumes at epoch k re-measures
epochs ≥ k. Take the LAST row per (epoch, space) / (epoch, group).

## Pre-registered predictions (committed 2026-08-13, before any training number exists)

The mechanism to beat: with AdamW the per-step parameter update is ~lr in magnitude almost
independently of the gradient scale, while ‖θ₀‖ is 10× larger — so the *relative* motion of
the backbone should shrink by roughly 10×. Large init scale is the textbook lazy/kernel
regime (Chizat–Bach), i.e. LESS feature learning, not more. Meanwhile a network that stays
near its random init has an h spectrum close to a random-feature spectrum, which is FLATTER
(higher effective rank) than a trained one. **High rank and weak feature learning can
therefore arrive together, and the rank number alone cannot tell them apart.**

- **P-E28-A (mine, primary):** effrank/stable rank of h_raw UP vs the ×1 control at matched
  epoch, `disp/backbone` DOWN by ~an order of magnitude, and online probe accuracy DOWN. The
  rank is bought by staying near init, not by feature learning.
- **P-E28-B (the interesting outcome):** rank UP with probe accuracy held or improved and
  displacement NOT collapsed — init scale is a genuine free knob on h geometry, and the
  h-moment term has a cheap substitute.
- **P-E28-C:** the run is unhealthy — attention logits are ×100 at init (near-hard softmax),
  so grad-norm incidents / probe collapse / a non-decreasing inv(z) are live risks. §3 shows
  finite loss, finite grads and a finite AdamW step, which is a birth check, not a
  convergence check.
- **P-E28-D (secondary, mechanism):** `disp/scaled48` ≈ 0.1 × the control's at matched epoch;
  `wnorm` follows the pure-decay curve more closely than the control (relative decay rate is
  init-scale-independent under decoupled wd, so a decay-dominated `wnorm` = little learning).
- **P-E28-E:** stable_rank(h_raw) < stable_rank(h_ln) throughout — the raw space carries a
  large common-mode/scale component that LN removes.

**Read protocol:** matched-epoch (ep 25/50/75/100) comparison of the treated per-epoch curve
against the control's 5 knots, on (i) online probe accuracy, (ii) effrank + stable rank +
RankMe + α of h_raw AND h_ln, (iii) `disp/backbone` and `disp/scaled48`. The claim "stronger
feature learning" requires (i) and (iii) to move WITH (ii); rank alone does not license it.

## Declared risks / confounds

1. **Weight decay is a live actor.** wd 5e-2 with lr 1e-3 shrinks weights at a *relative*
   rate independent of their scale, so the ×10 lane's weights decay toward the ordinary scale
   over training (~e^-2.5 over the cosine schedule if learning does not oppose it). The
   intervention is genuinely an *initialization* intervention, not a sustained scale change —
   the `wnorm` trace is what tells us how long the ×10 survives. Changing wd was NOT done:
   "everything else … unchanged".
2. **Attention saturation at init.** q·k is ×100 ⇒ the softmax is near one-hot on random
   directions for the first steps. Warmup (10 ep, start factor .01) is the only thing damping
   it. Kill-trigger armed (grad-norm > 100× running median prints INCIDENT).
3. **The h-conditioner term is computed but zero-weighted** (RNG parity with the reference
   lane). At ×10 its VALUE is large (h rms ~93) but it enters neither loss nor gradient; the
   logged `train/h_moment_kl` for this run is a spectator quantity and must not be compared
   across lanes.
4. **The online probe reads unnormalized h** through its own LayerNorm, so its accuracy is
   comparable to the control's but not identical-by-construction even at ×1.
5. **One seed, one condition.** Berker asked for condition 1 only.

## Landing implications (if this run is taken to the full landing chain)

`extract.py` reads h from `trunk.forward_features` — which is now UNNORMALIZED for this
checkpoint — and the projector taps are shifted by one index (the LN is `projector.0`). The
checkpoint rebuilds correctly (`arch` blocks point at `experiments.e28_scaleinit.e28_trunk` /
`e28_projector`, round-trip verified), but any cross-run h-space table that mixes this run
with the standard lanes is comparing pre-norm to post-norm features. Use the `h_ln` column of
the E28 diagnostics for like-for-like, or state the difference. `from_native(random_init=True)`
rebuilds the arch WITHOUT the ×10 (the null model is an ordinary-init model).

## Launch log

| when | what | jobs |
|---|---|---|
| 2026-08-13 14:48 | gate attempt 1 — **FAILED** in §3 (the spectator wall, above); chain killed by `DependencyNeverSatisfied`, cancelled | 63436960 (+ 61–63 cancelled) |
| 2026-08-13 14:53 | gate attempt 2 — selftest ALL GREEN + ×1 control diagnostics (`gpu`, A40, 1 h) | 63437128 |
| 2026-08-13 15:0x | training chain attempt 1 (blocks-only ×10) — CANCELLED at ~ep10 on Berker's patch_embed ruling; artifacts parked in `results/e28/attempt1_blocksonly/` | 63437129–31 |
| 2026-08-13 15:2x | init-scale sweep + per-block rank profile (`gpu`, A40) | 63437775 |
| 2026-08-13 15:2x | **the arm**: ×10 on 49 tensors (blocks + patch_embed), 3×8 h H100 (`gpu100`), `afterany` links; ~6 min/epoch ⇒ 100 ep ≈ 10 h; the selftest re-runs as a gate at the head of every segment | **63437776** → 63437777 → 63437778 |

run_id `in100.floorssl.s0.e28x10`; checkpoints `outputs/in100.floorssl.s0.e28x10_{ep25,ep50,ep75,ep100,best,last}.pt`.

## Why inv destabilised, and what the rank at ×10 actually is (Berker 2026-08-13: "inv became very unstable … do you think grad norm being >2k is the issue? … trying to figure out how to make this rich learning regime")

**The grad norm is not the disease.** Per-step wandb history, ×10 arm 63437776 vs the ×1
control:

| | inv (first 200 steps → ~step 400 → last 200) | grad_norm median | z-KL median (first → last) |
|---|---|---|---|
| ×1 control | 0.459 → 0.369 → **0.193** (monotone down) | 38.8 → 6.4 → 14.0 | 0.723 → 0.139 |
| ×10 arm | 0.252 → 0.450 → **0.492** (rising, max 0.955) | 874 → 773 → 140 | 1.395 → 1.060 |

Both lanes exceed `grad_clip=1.0` on essentially every step (the control's *median raw* norm
is 38.8), so both are clipped-every-step, and **AdamW is invariant to a uniform rescaling of
the gradient** — it divides by its own running RMS. Clipping is therefore neither the harm nor
the help here; the norm is a readout of landscape steepness, not the cause. The disease is
that **inv rises** while z-KL sticks at ~1.0 (7× the control's converged 0.14).

**Two independent mechanisms, both measured.**

*(1) The objective got harder at init — the rank IS augmentation scatter.* Ω = W/B (orbit pair
energy over centre pair energy, `sslgap/metrics/orbit_energy.py`): how far two views of one
image land apart, relative to how far apart two images land. Job 63438532,
`results/e28/e28_viewsweep.csv`, at init on one fixed V=4 lejepa batch:

| scale | families scaled | Ω_h | h effrank |
|---|---|---|---|
| ×1 | — | 4.64 | 6.6 |
| ×3 | all | 6.00 | 18.2 |
| ×5 | all | 11.09 | 89.2 |
| **×10** | **all (the arm)** | **23.73** | **182.0** |
| ×10 | **no_qkv** (proj+fc1+fc2+patch) | **5.12** | **11.2** |
| ×10 | **qkv_only** | **29.59** | **251.7** |
| ×10 | mlp_only | 5.01 | 11.3 |
| ×20 | all | 28.51 | 199.2 |
| ×20 | qkv_only | 40.49 | **283.6** |
| ×20 | no_qkv | 5.05 | 10.6 |

Two things fall out. **The rank comes entirely from `qkv`** — scaling every other weight family
by 10 leaves h at effrank 11 (vs 6.6 at ×1), while scaling qkv alone gives 251.7, *higher* than
scaling everything. Attention logits scale as scale² (q and k are both scaled), so this knob is
attention temperature, not weight magnitude. And **Ω tracks effrank monotonically in all 21
cells**: h rank and view-decorrelation are the same quantity here. Near-hard attention makes
two crops of one image select different tokens, so their representations decorrelate — that
decorrelation IS the high effective rank. It is augmentation noise spread over directions, not
feature content, and the invariance term's whole job is to remove it. Hence inv rising, and
hence the rank collapsing as soon as training gets traction (effrank 204 → 107, stable rank
22 → 4.2 by ep3).

This also corrects the ceiling I reported earlier: ~204 is the ceiling of the *combined* knob;
qkv-only reaches 283.6 at ×20 — but at Ω 40.5, i.e. views nine times more scattered than at ×1.

*(2) At fixed lr the arm is 10× LAZIER by construction — the opposite of rich.* Under Adam the
per-step update is ~lr per coordinate regardless of gradient size, so relative motion per step
≈ lr·√d/‖θ‖; a ×10 init divides it by 10. Measured (`_disp.csv`):

| | disp/backbone | wnorm/backbone | disp/projector_mlp |
|---|---|---|---|
| ×1 control @ep25 | **3.05** | **3.04** (weights GREW 3×) | — |
| ×10 arm @ep3 | **0.061** | **0.978** (shrinking — decay-dominated) | 0.360 (6× the trunk) |

So the trunk is nearly frozen and shrinking while the projector does the adapting: textbook
lazy. P-E28-D is confirmed on both counts.

**The lr pilot** (63439130, `in100.floorssl.s0.e28x10lr`, declared deviations lr 1e-3→1e-2,
wd 5e-2→5e-3 to hold lr·wd so decoupled decay does not eat the ×10 init 10× faster): at ep1
`disp/backbone` = **0.121** vs the fixed-lr arm's **0.0119** — exactly the predicted ×10, and
`wnorm` 1.004 instead of 0.997. The update-to-weight ratio is the richness dial and it behaves
exactly as the theory says. **But the pilot collapsed**: h effrank 182 → 56.0 → **4.6** (stable
rank 1.4) and probe .0116 → .0080, i.e. at chance, by ep2 — while warmup still had lr at 20% of
peak. CANCELLED at ep2. The richer the trunk, the faster it destroys the init rank, which is
exactly what mechanism (1) predicts: that rank is the thing inv exists to remove.

*Below ×1 there is nothing* (job 63439967): at ×0.3/×0.5/×0.7 every family set gives Ω_h
4.35–4.74 — indistinguishable from ×1's 4.64 — and h effrank 3.0–6.5. The ordered side is a
plateau; Ω_h ≈ 4.4 is the augmentation's own difficulty, not something the init can lower.

**Uniformity does not change the picture** (Berker 2026-08-13: "the more uniform the better. it
is hard to justify why we leave some parts out"). Three uniformity levels, ×10 at init:

| selector | tensors | Ω_h | h effrank |
|---|---|---|---|
| `all` — every affine WEIGHT (the arm) | 49 | 23.7 | 182.0 |
| `w_and_b` — + every affine BIAS (each layer's map exactly ×α) | 98 | 25.2 | 181.3 |
| `everything` — + LN gains, pos_embed, cls_token; no exclusions at all | 148 | 32.1 | 203.7 |

Adding the biases is invisible (181.3 vs 182.0), so the arm's exclusion list is not
load-bearing — **the arm as running already IS the uniform-over-weights intervention.** Adding
the LN gains only moves *faster* along the same curve (`everything` at ×3 = Ω 22.7 / rank 174.6
≈ `all` at ×10), because γ×10 scales the branch input and compounds with qkv×10 into the
attention logits.

**Across all 31 cells — uniform and selective, ×0.3 to ×20 — h effective rank is a monotone
function of Ω** (no cell with rank > 40 below Ω 8; none with rank < 20 above Ω 8). There is one
curve, and every scaling choice only slides along it: **the rank you buy is exactly the
view-decorrelation you pay for.** Init scale cannot buy rank at fixed invariance. The h-moment
floor can, and does (D-100: B-rank ×2–3 with views still agreeing) — which is the contrast E28
exists to draw: **rank-by-decorrelation vs rank-under-invariance**, and note the h-floor acts on
the per-image view-MEANS (`h_floor_batch=view_mean`), i.e. on the image centres, while the
init's rank lives in the view scatter that inv exists to destroy.

**The arm's own trajectory confirms the erasure** (63437776, per-epoch): h effrank
182 → 204 (ep1) → 107 (ep3) → 81 (ep8) → **36.0 (ep10)**, stable rank 12.9 → 4.8, `wnorm`
0.81 (backbone weights 19% below init). By ep10 the ×10 arm's h rank has fallen BELOW the ×1
control's ep25 value (41.1), with probe .0308 against the control's .0370 at ep1. The init
geometry is not a fixed point of this objective and is being removed as fast as the trunk
moves.

## "The network does not look like it is training" — what it was, and what the fix is not (Berker 2026-08-13)

**Part of it was warmup.** `warmup_ep=10` with start factor .01, and a ×10 backbone moves 10×
less per step at a given lr, so the first ten epochs did essentially nothing. The arm took off
the moment warmup ended: probe .0308 (ep10) → .0374 → .0508 → .0552 → .0742 → **.0848 (ep15)**,
wobbling to .0596 (ep17). It is training — roughly where the ×1 control was at ep2–3.

**On "inv is not going down": raw inv is the wrong readout.** The z-conditioner inflates z's
scale in the first epoch (z rms .05 → .54), which raises the view MSE mechanically. The
scale-free form is `inv / (2·var z)` — the value two INDEPENDENT draws would give, so 1.0 means
the views carry no information about each other:

| | ep1 | ep3 | ep25 | ep50 | ep75 | ep100 |
|---|---|---|---|---|---|---|
| ×10 arm | **0.775** | **0.901** | | | | |
| ×1 control | | | 0.321 | 0.270 | 0.191 | **0.172** |

So the worry is real and sharper than "inv is flat": at ×10 the two views are **nearly
independent and getting more so**. Raw inv itself rose .24 → .55 by ep8 and has since
plateaued (.5534 → .5514 → .5455), and the early jaggedness is gone — inv IQR .037 → .008,
grad-norm median 2781 → 257. Stable, smooth, and unproductive. This is the Ω result from the
init sweep, now measured during training. **New instrument** (`view_alignment`, own CSV
`results/e28/<run>_align.csv`): per-epoch Ω_h, Ω_z and `inv/(2·var z)` on a fixed V=4 batch —
live from the ×5 arm onward.

**Re-dosing + more lr is NOT the fix.** House instrument (`+e28.mode=doses`, job 63441066,
rings warmed): realized trunk shares and total pull,

| cell | g_inv | g_zkl | share_inv | share_z | T = Σ w·g |
|---|---|---|---|---|---|
| ×1 init | 0.957 | 1.122 | .151 | .849 | 208.5 |
| **×1 ep25 (formation)** | 0.192 | 0.033 | **.550** | **.450** | **11.47** |
| ×10 init | 3.436 | 5.731 | .111 | .889 | 1017.0 |
| ×10 ep12 | 0.543 | 0.155 | .421 | .579 | 42.25 |

⇒ re-dose onto the ×1 profile: **w_inv 32.8 → 11.62, w_floor 157.8 → 33.36**. Two pilots with
those doses, differing only in how hard the trunk may move (wd held so lr·wd is constant):

| pilot | lr | probe @ep5 | h effrank @ep5 |
|---|---|---|---|
| the arm | 1e-3 (original doses) | — (.0308 @ep10, .0848 @ep15) | 18–30 @ep11–15 |
| P1 63441259 | 3.16e-3 | **.0200** | **16.9** (133 @ep1 → 37.8 @ep4) |
| P2 63441260 | 1e-2 | (ep1 .0124) | 127.9 @ep1 |
| earlier pilot (no re-dose) | 1e-2 | collapsed ep2 | 4.6 |

**Both higher-lr cells are worse than the lr-1e-3 arm.** So the binding constraint is NOT
laziness — it is stability: the ×10 landscape (near-hard attention) tolerates only small
relative steps, and the step size needed to learn is above what it tolerates. Raising lr moves
faster in a jagged landscape and degenerates h.

**And the intervention is dissolving.** `wnorm/backbone` 1.000 → 0.780 (ep11) → **0.659 (ep18)**
under wd 5e-2, while h effrank fell 182 → **20.2 (ep18)** — already below the ×1 control's ep25
value of 41.1. At this rate the ×10 weights reach ~0.1× init by ep100, i.e. the arm converges
to an ordinary run, from a worse start and ~15 epochs behind. If a scaled network is to be
STUDIED for 100 epochs, weight decay on the scaled weights is the next thing that has to be
justified.

## The ×5 arm, and what "h_raw effrank = 2" actually is (Berker 2026-08-13: "with 5x it collapses harder … never seen h_raw effrank as low as 2")

**It is not a collapse — ×5 is the best arm we have.** Online probe at matched epochs:

| ep | ×1 control | ×5 | ×10 |
|---|---|---|---|
| 1 | .0370 | .0434 | .0196 |
| 5 | .1864 | .1614 | — |
| 10 | .3046 | .2108 | .031 |
| 15 | .3626 | .3406 | .085 |
| 20 | .4260 | **.4292** | — |
| 25 | .4638 | **.4898** | — |

×5 falls behind through warmup (it is 5× lazier per step) and then **catches up and passes the
control by ep20**. At ep25 its h_ln effrank is 45.5 (control 44.1) and z effrank 186.2 (control
185.2) — everything the loss and the probe actually see is healthy and slightly ahead.

**The effrank-2 reading is a gauge artifact, and the gauge is the per-sample SCALE.** Since
z = MLP(LN(h)), the loss is invariant to h → a·h + b·1 for per-sample scalars a>0, b: the
common mode and the norm of each h_i are exact gauge freedoms, pinned by nothing. Job 63449541
(`gauge_row`, `results/e28/e28_gauge_ckpts.csv`) splits them:

| lane | ep | h_raw effrank | common-mode share | **scale share** | effrank ⊥ common-mode | h_ln effrank |
|---|---|---|---|---|---|---|
| ×1 control | 25 | 41.1 | .0001 | .083 | 41.1 | 44.1 |
| ×1 control | 100 | 56.8 | .0001 | .155 | 56.8 | 54.4 |
| **×5** | **25** | **2.6** | **.0061** | **.741** | **2.6** | **45.5** |
| ×10 | 25 | 25.6 | .0001 | .057 | 25.6 | 25.8 |

In the ×5 run **74% of h_raw's total variance is variation in ‖h_i‖** (control: 8%). Once the
norms are heavy-tailed a handful of images dominate the second moment and the effective rank of
h_raw goes to ~2 — while the DIRECTION structure is untouched (h_ln 45.5 ≈ control 44.1).
Projecting out the all-ones mode changes nothing (share .006), so it is specifically the scale.
The ×10 arm's low rank at ep25, by contrast, is REAL: h_raw 25.6 ≈ h_ln 25.8, no gauge inflation.

**Consequence for the experiment's premise:** "rank of unnormalized h" is not a well-defined
property of the learned representation in this geometry — it is the representation's rank plus
two unconstrained scalar modes, and the scale mode can swamp it. The honest h-rank column for
E28 lanes is **h_ln** (or h_raw with the per-sample scale divided out); `gauge_share` /
`scale_share` / `effrank_perp` now ride every epoch (`results/e28/<run>_gauge.csv`). If a high
rank at UNNORMALIZED h is to be a target at all, something has to pin ‖h‖ — the objective does
not, and this is exactly why the project's declared h is the trunk's normalized output (D-003).

## THE ARMS — ×N init, backbone weight decay OFF

Goal (Berker 2026-08-13): "we just want a training model with weight scaling on vit which leads
to high rank (as if we regularized h). i dont care about anything else … my expectation of
accuracy is not 70% … it can be slightly lower. but rank should stay high according to the
theory."

**REJECTED first attempt — scale hold.** I first made the ×N permanent by renormalizing every
backbone tensor to its init norm after each step. Berker killed it, correctly: *"lazy / rich
learning is a dynamics claim. if you keep changing the scale, then it does not deliver the
claim."* A projection onto a fixed-norm set is an external force; it makes "the scale persists"
true by construction, so nothing about the dynamics can be read off the result. The `hold`
code path stays in the file, disabled and labelled, as the record of a rejected design — it is
not an arm.

**Re-derived: the force annealing the scale away was weight decay, and wd is a recipe
hyperparameter, not a constraint I invent.** Decompose the observed norm change into
decay × learning, using `r_decay = ∏(1−lr_t·wd)` (identical for every tensor, computed on the
real schedule) — what remains is what the LEARNING did:

| ×10 arm ep | r_decay | qkv obs / learn | LN γ obs / learn | eff qkv scale (with wd) | **implied at wd=0** |
|---|---|---|---|---|---|
| 12 | .706 | .769 / **1.09** | .664 / **0.94** | 5.11 | **10.25** |
| 18 | .526 | .659 / **1.25** | .464 / **0.88** | 3.06 | **11.07** |
| 30 | .301 | .508 / **1.69** | .291 / **0.96** | 1.48 | **16.26** |
| 45 | .168 | .471 / **2.80** | .244 / **1.45** | 1.15 | **40.60** |

Under gradient dynamics ALONE the effective attention temperature holds at ~10 and then grows;
every bit of the collapse to 1.15 was decoupled weight decay. So the arm is the ×N init with
**backbone wd = 0** — plain AdamW, no projection, one hyperparameter set to zero. The
projector keeps wd 5e-2 through the repo's existing `mlp_wd` axis, so the delta is confined to
the ViT. Declared deviations: `method.wd`, `method.mlp_wd`.

**Pre-registered (before any number):** if the decomposition is right, ×10 holds an effective
qkv scale ≥ ~10 and an h effective rank ≳ 180 through ep25+, instead of the 3.06 / 20.2 the
wd=5e-2 arm showed at ep18. If the rank still decays, the annealing is being driven by the
learning signal itself and initialization scale cannot deliver persistent rank at all.

### RESULT: the pre-registration is FALSIFIED (ep12 read, 2026-08-13)

wd=0 does exactly what the decomposition said it would to the TEMPERATURE — and the rank
collapses anyway:

| ×10, wd=0 | ep0 | ep1 | ep3 | ep6 | ep9 | ep12 |
|---|---|---|---|---|---|---|
| effective qkv scale | 10.00 | 10.00 | 10.01 | 10.05 | 10.13 | **10.18** |
| h effrank | 182.0 | 197.9 | 89.2 | 118.9 | 52.7 | **20.9** |
| probe | — | .027 | | .037 | | .046 |

The temperature is held perfectly by the dynamics alone (no decay, no projection, `wnorm` 1.04
— the norms even grow). **So the temperature was never what drove the rank down.** The rank
decay is driven by the LEARNING SIGNAL itself: the network discards the high-rank structure
because the objective has no use for it. My registered prediction ("×10 holds effrank ≳180
through ep25+") is wrong.

Companion arms at ep8–12, same recipe:

| arm | eff qkv scale | h effrank | probe |
|---|---|---|---|
| ×1, wd=0 | 1.00 → 2.85 (qkv grows 3.1×) | 6.6 → 31.5 (ep6) → **10.4** (ep12) | .3284 |
| ×5, wd=0 | 5.00 → 4.93 | 89.2 → 142.3 (ep1) → **35.8** (ep8) | .2218 |
| ×10, wd=0 | 10.00 → 10.18 | 182.0 → **20.9** (ep12) | .0462 |

Every lane — starting at 6.6, 89 or 182, with the temperature falling, held, or rising —
converges into the same h effrank 10–40 band, and accuracy orders inversely with the scale.
The only intervention that has ever put h at ~200 is the h-moment term itself (`d256vm4`,
202.6). Ranks are non-monotonic in every lane (the ×1 wd=5e-2 control runs 6.6 → 41 → 62 → 57;
the unheld ×10 recovered 18 → 37.6 by ep45), so the ep25/50 read is the one that closes it —
but the direction is now measured under three separate controls (decay off, temperature held,
attribution control at ×1).

| arm | run_id | jobs | init h effrank |
|---|---|---|---|
| ×1, wd=0 (attribution control) | `in100.floorssl.s0.e28w1` | 63449904→05 | 6.6 |
| ×5, wd=0 | `in100.floorssl.s0.e28w5` | 63449906→07 | 89.2 |
| ×10, wd=0 | `in100.floorssl.s0.e28w10` | 63449908→09 | 182.0 |
| ×20, wd=0 | `in100.floorssl.s0.e28w20` | 63449910→11 | 199.3 |

The ×1 arm is not optional: without it, a high rank at ×10 cannot be attributed to the scaling
rather than to switching decay off.

## (rejected) SCALE HOLD — kept for the record

Everything before this section is diagnosis. The goal is now one thing: **weight scaling on the
ViT that yields high h-rank AND trains**, accuracy allowed to sit below the control's .7048.

**The change that makes it possible — SCALE HOLD.** The measurements above showed the run
annealing the ×N away (effective qkv scale 10.0 → 1.15 by ep45, h rank following it down
182 → 18). An initialization cannot resist that attractor, so the scale is now HELD: after
every optimizer step each backbone tensor is renormalized to its **init** Frobenius norm. The
optimizer may ROTATE the weights freely; it may not resize them. Uniform and self-selecting —
76 tensors pinned (all affine weights, all LN gains, pos_embed, cls_token), the 72 zero-init
biases free because there is no norm to hold. The projector is untouched ("weight scaling on
the ViT"). It is a projection onto a constraint set, not a loss term, so it changes the fixed
points rather than the starting point — and it makes weight decay a no-op on the held tensors,
so wd stays at the reference lane's 5e-2. Selftest §4 gates it (max |‖θ‖/‖θ₀‖ − 1| = 1.2e-7
over pinned tensors, free tensors still move, direction still rotates).

| arm | run_id | jobs | ep0 h effrank | ep1 h effrank | ep1 probe |
|---|---|---|---|---|---|
| ×5 held | `in100.floorssl.s0.e28h5` | 63449697→98 | 89.2 | **138.7** | .0348 |
| ×10 held | `in100.floorssl.s0.e28h10` | 63449699→700 | 182.0 | **193.3** | .0230 |
| ×20 held | `in100.floorssl.s0.e28h20` | 63449701→702 | 199.3 | **199.2** | .0156 |
| — reference | ×1 control | | 6.6 | — | .0370 |
| — target | h-treated `d256vm4` | | | **202.6** (ep100) | (.7048 control) |

All three sit at or near the h-treated arm's rank from epoch 1, `wnorm/backbone` is exactly
1.0000 (the hold is binding), and the rank is REAL not gauge (`effrank_perp` ≈ `effrank`,
gauge share .002). The open question is whether it holds through training and what it costs in
accuracy — read at ep20–25.

Cancelled to make room: the unheld ×10 (63437776, reached ep~45) and unheld ×5 (63442708,
reached ep25, probe .4898 — it had passed the control) plus both re-dose pilots. Their
checkpoints and per-epoch CSVs are kept.

## The target number, reconciled (Berker 2026-08-13: "40 is still miserable. d256zonly control arm has >150 eff rank")

From the landed battery (`results/battery/*.extL.csv`, n=50k train, ep100, `raw|full`):

| run | space | **effective_rank** | rankme | PR | α |
|---|---|---|---|---|---|
| **d256vm4zonly** (z-only CONTROL, no h term) | h.cls | **53.9** | 135.8 | 18.1 | 1.42 |
| **d256vm4** (h-TREATED) | h.cls | **202.6** | 301.1 | 150.7 | 0.64 |
| d256vm4zonly | z.proj.out | 180.6 | 208.5 | 152.5 | 0.41 |
| d256vm4 | z.proj.out | 250.5 | 254.5 | 245.6 | 0.16 |

**The >150 belongs to `d256vm4` — the arm WITH the h regularizer — not to the z-only control,
which sits at 53.9.** (My per-epoch instrument agrees with the battery: h effrank 56.8 at ep100
on the 5k val split vs 53.9 on the 50k train store.) So the scoreboard, all on
`effective_rank` of h:

| lane | h effrank | note |
|---|---|---|
| **h-treated d256vm4** | **202.6** | the target E28 is trying to reach without an h term |
| z-only control | 53.9 (ep100), 44.1 (ep25) | the untreated baseline |
| ×5, ep25 | 45.5 (h_ln) | on par with the control at the same epoch |
| ×10, ep25 | 25.8 | below it |
| **×10 AT INIT** | **182.0** | in the target neighbourhood — for one step |

That last row is the whole story of E28: the ×10 initialization does land h in the treated
arm's rank neighbourhood, and then the network spends the run annealing it away (next section).
Berker's reading is right — 40 is miserable against a target of 200.

## Weight decay, checked properly (Berker 2026-08-13: "did you check the weight decay?")

I had claimed decay was dissolving the intervention. **That was imprecise — decay is identical
in both lanes and the ×1 lane grows straight through it.** Pure decay with NO learning
(∏(1−lr_t·wd) over the real schedule, wd 5e-2): 0.526 by ep18, 0.377 by ep25, 0.082 by ep100.
Observed `wnorm/backbone`: ×10 arm 0.659 at ep18 = **1.25×** the pure-decay floor; ×1 control
3.037 at ep25 = **8.1×** it, and 2.683 at ep100 = 32.6×. Same decay — the ×10 lane's learning
simply adds almost nothing to the norm. The shrinkage is a symptom, not a cause.

**But checking it did surface the real mechanism.** This recipe puts LayerNorm gains and biases
in the SAME decay group (`floorssl.param_groups`: one group, no no-decay list), and attention
logits scale as (γ·W_qkv)². Tracking that product against an ORDINARY init:

| ×10 arm ep | ‖qkv‖/init | ‖LN γ‖/init | effective qkv scale | logit factor | h effrank |
|---|---|---|---|---|---|
| 0 | 1.000 | 1.000 | 10.00 | 100.0 | 182.0 |
| 6 | .927 | .904 | 8.38 | 70.2 | 96.4 |
| 12 | .769 | .664 | 5.11 | 26.1 | 31.9 |
| 18 | .659 | .464 | 3.06 | 9.4 | 20.2 |
| 30 | .508 | .291 | 1.48 | 2.2 | 30.5 |
| 45 | .471 | .244 | **1.15** | **1.3** | 37.6 |

and the ×1 control moves the OTHER way — 1.00 → 1.62 (ep25) → 1.20 (ep100). **Both lanes
converge to an effective qkv scale of ~1.2.** The ×10 arm spends its first ~40 epochs annealing
its own attention temperature back to the same operating point the control climbs to, and its
h rank tracks that annealing down (182 → 18 at ep15) and then back up along the ordinary
trajectory (37.6 at ep45). The intervention has a half-life, set by decay on γ and W together —
which is a far more precise statement than "wd dissolves it", and it is the reason ×10 cannot
hold its init geometry for 100 epochs.

## Amendment to the pre-registration (2026-08-13, after the control ran, before any treated trajectory)

P-E28-A's stated mechanism contains a premise that the control measurement **falsifies**: I
assumed a random-init h has a flat, high-rank spectrum, so a lazy run would show high rank for
the wrong reason. Measured, the ordinary ViT-S init is the opposite — h at ep0 has
**effrank 6.6 / 384** and stable rank 2.0 (top-10 directions carry 93% of the variance), and
training RAISES it to ~57–67. So at ×1 "lazy" would mean rank stays LOW, and rank and accuracy
are expected to move together. The predictions stand as written (they were committed first);
this is the corrected reasoning to read them against.

## Results (raw; nothing interpreted here — AGREED TAKEAWAY is Berker's call)

### ×1 control, recomputed with this code (`results/e28/in100.floorssl.s0.d256vm4zonly_diag.csv`, gate 63437128)

| ep | h_raw rms | h_raw effrank | h_raw srank | h_ln effrank | z effrank | disp/backbone | wnorm/backbone |
|---|---|---|---|---|---|---|---|
| 0 (init) | 0.58 | 6.6 | 2.04 | 6.9 | 14.6 | 0 | 1.00 |
| 25 | 8.76 | 41.1 | 6.05 | 44.1 | 185.2 | 3.05 | 3.04 |
| 50 | 4.97 | 62.0 | 10.10 | 52.1 | 175.5 | 3.25 | 3.19 |
| 75 | 2.94 | 66.7 | 7.44 | 56.2 | 171.8 | 2.91 | 2.83 |
| 100 | 2.33 | 56.8 | 5.04 | 54.4 | 167.4 | 2.78 | 2.68 |

(z effrank/256 ≈ .65–.72 throughout; α: h 2.00 → 1.43, z 1.62 → 0.56. Full spectra in
`_spectra.npz`, per-block displacement in `_disp.csv`.) Note `wnorm/backbone` ≈ 3: the ×1
backbone's weights GROW to ~3× their init norm — this lane is nowhere near decay-dominated,
which is the baseline P-E28-D has to be read against.

### Init-scale sweep — how far can this knob push h's rank? (job 63437775, `results/e28/e28_initsweep.csv`)

Berker 2026-08-13: *"effective rank is 180 is a good start … i was expecting to reach a higher
number."* Measured at step 0 (no training), same val set, both patch_embed variants:

| init scale | h_raw effrank /384 | h stable rank | α (h) | z effrank /256 | h rms |
|---|---|---|---|---|---|
| ×1 | 6.6 (.02d) | 2.0 | 2.00 | 14.6 | 0.58 |
| ×1.5 | 8.8 | 2.2 | 1.83 | 17.7 | 1.3 |
| ×2 | 11.0 | 2.5 | 1.73 | 20.1 | 2.3 |
| ×3 | 18.2 | 3.0 | 1.49 | 27.9 | 4.8 |
| ×5 | 89.2 (.23d) | 5.9 | 1.02 | 91.3 | 13.4 |
| **×10** | **182.0 (.47d)** | **12.9** | **0.78** | **139.8** | 62 |
| ×20 | 199.2 (.52d) | 15.6 | 0.74 | 146.7 | 257 |
| ×50 | 203.2 (.53d) | 16.5 | 0.73 | 148.3 | 1.6e3 |
| ×100 | 203.5 (.53d) | 16.6 | 0.73 | 148.4 | 6.5e3 |

**The knob saturates.** The rise is between ×3 and ×10 (18 → 182); ×10 → ×100 buys 182 → 204
while h's rms explodes 100×. ~**204 / 384 (.53d)** with α → 0.73 is the ceiling of init scale
on this architecture+data — the limit is the hard-attention / ReLU-limit random function, not
the size of the weights. A number materially above ~204 needs a different knob, not a bigger
one. (Trained reference points: the ×1 lane reaches h effrank 66.7 at ep75; z reaches ~185.)

**patch_embed makes almost no difference** at ×10 (182.0 with vs 180.3 without) and none at
≥×20; the visible gap is in the transition region (×5: 89.2 vs 82.3) and at ×3 (18.2 vs 16.2).
Expected: block 0 sees LN(x₀), which is blind to x₀'s scale — including patch_embed only
changes the image-content : positional-signal ratio inside x₀ (pos_embed and cls_token stay
unscaled per the spec). It is INCLUDED in the arm per Berker's ruling.

Depth profile (`results/e28/e28_initdepth.csv`) — effrank of the raw residual-stream CLS after
each block:

| scale | b1 | b2 | b3 | b4 | b6 | b8 | b10 | b12 |
|---|---|---|---|---|---|---|---|---|
| ×1 | 4 | 5 | 6 | 6 | 6 | 6 | 7 | 7 |
| ×3 | 10 | 13 | 14 | 15 | 16 | 17 | 18 | 18 |
| ×10 | 105 | 140 | 154 | 163 | 173 | 177 | 179 | 182 |
| ×100 | 108 | 147 | 165 | 177 | 191 | 196 | 200 | 204 |

The rank is made in the FIRST block (105 of the final 182 at ×10) and depth only trims the
tail; at ×1 no block ever builds any.

### ×10 treated, initialization only (attempt 1, blocks-only, job 63437129)

| | ×1 init | ×10 init |
|---|---|---|
| h_raw rms | 0.58 | **61.5** |
| h_raw effrank / 384 | 6.6 (.017) | **180.3 (.470)** |
| h_raw stable rank | 2.04 | **12.6** |
| h_ln effrank | 6.9 | 180.8 |
| z effrank / 256 | 14.6 | 139.4 |

i.e. the ×10 initialization alone puts h at ~3× the effective rank the ×1 lane reaches after
100 epochs of training — **at step 0, with no h regularizer**. Whether that survives training,
and what it costs, is the experiment. Online probe ep1: **.0196** (×1 lane's ep1: .0370); by
ep1 h effrank had moved 180 → 204 and z 139 → 206, so the trajectory is not frozen.

The canonical arm (63437776, patch_embed included) opens at the same place: ep0 h_raw effrank
**182.0**, stable rank 12.9, rms 62; z effrank 139.8.

_(the per-epoch trajectory fills `results/e28/in100.floorssl.s0.e28x10_{diag,disp}.csv` as the
chain runs; ep25/50/75/100 checkpoints land in `outputs/`)_

## AGREED TAKEAWAY

**Berker 2026-08-13: "it was a good try but e28 failed."** Written from that verdict plus the
measurements below — correct or amend it, and it still owes its `docs/DECISIONS.md` mirror
(not written this session per "do not change anything else in the repo").

**E28-T1 — initialization scale does not substitute for the h-moment term.** ×N init places h's
effective rank anywhere between 6.6 and 199 at step 0, but training removes it, and the removal
is not an artifact of the recipe: with backbone weight decay OFF and the effective attention
temperature held at ×10 by the dynamics themselves (10.00 → 10.18 through ep12, norms growing,
no projection), h effrank still falls 182 → 20.9. Every lane — ×1, ×5, ×10, starting at 6.6,
89 or 182 — converges into the same h effrank 10–40 band, with online accuracy ordering
inversely against the scale (.328 / .222 / .046 at ep8–12). The only intervention that has ever
put h at ~200 is the h-moment term itself (`d256vm4`: 202.6 vs the z-only control's 53.9).

**E28-T2 — what the init buys is rank-by-decorrelation, not rank-under-invariance.** Across 33
initialization cells (×0.3–×20, uniform and family-split), h's effective rank is a MONOTONE
function of Ω = view-scatter / image-scatter: no cell has rank > 40 below Ω 8, none has rank
< 20 above Ω 8, and the ceiling of the knob is ~204 at Ω ≈ 33. The rank is the network
amplifying augmentation differences into orthogonal directions — precisely the structure the
invariance term exists to destroy — which is why it cannot survive training and why the
h-term's rank is a different object. Mechanistically the whole effect is attention temperature:
scaling every weight family EXCEPT qkv leaves h at effrank 11 (vs 6.6 at ×1).

**E28-T3 (measurement lesson, applies beyond E28) — "rank of unnormalized h" is not well
defined under a relocated final LN.** z = MLP(LN(h)) is invariant to h → a·h + b·1 for
per-sample scalars, so the common mode and ‖h_i‖ are exact gauge freedoms of the objective. In
the ×5 arm 74% of h_raw's variance became norm variation and h_raw effrank read **2.6** while
the direction structure was healthy (h_ln 45.5, z 186, probe .4898 — ahead of the control).
Report h_ln, or h_raw with the per-sample scale divided out. This is why the project's declared
h is the trunk's NORMALIZED output (D-003) — E28 is the exhibit for that ruling.

**E28-T4 (estimator, owed as a DECISIONS row if anyone conditions on an unnormalized tap):**
`SpectralConditioner`'s ridge is an ABSOLUTE `eps=1e-4`, so the estimator is not scale-free.
On unnormalized h at ×10 the step-1 slice (n = d′ = 128, structurally singular) fails Cholesky
stochastically. Any future arm with `h_lamb > 0` on an unnormalized tap needs a ridge relative
to tr(Σ)/d′ first.

**Not claimed:** that ×N cannot help at all. ×5 under the standard recipe passed the ×1 control
on the online probe at ep20–25 (.4292/.4898 vs .4260/.4638) before being cancelled — an
accuracy effect, with h rank at control level. That is a separate question from the one E28
asked and it was not run to ep100.
