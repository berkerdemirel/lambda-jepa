# v6s400 — self-contained brief for an outside opinion (2026-09-10)

## Setup

Self-supervised pretraining of a ViT-S/16 on ImageNet-1k, 400 epochs, batch 128, 6 global
augmented views per image, AdamW, cosine learning rate with ~1 epoch warmup (peak ~1e-3,
currently 2.3e-4 at epoch 300).

The loss has three terms:
1. **invariance** on the projector output `z` (pull views of the same image together), weight 31.07
2. **a conditioner on `z`**: a moment-matching KL between the batch's `z` statistics and an
   isotropic Gaussian target, weight 225.50
3. **the same conditioner on `h`**, the trunk's CLS token before the projector, weight 1.671

Both conditioners are estimated over a *ring*: the current step's rows plus the previous 3 steps'
rows, detached (`queue_steps=3`).

**The sample count is small, and this matters.** Both conditioners are fed the **mean over the 6
views** of each image, not the individual views, so a step contributes 128 rows (one per image), not
768. The ring therefore holds **4 × 128 = 512 rows**. Both conditioners act on a **128-dimensional
slice** (`z` is 256-dim, `h` is 384-dim; both are sliced to 128). So every moment estimate is
**512 points in 128 dimensions — 4 samples per dimension.**

Projector: `Linear(384→2048) → BatchNorm1d → ReLU → Linear(2048→2048) → BatchNorm1d → ReLU →
Linear(2048→256)`. Call the three weight matrices W0, W3, W6. W0 and W3 feed a BatchNorm, so
their scale cannot change the network's function; W6 is the output and its scale does matter.

Two sibling runs inform what follows: the same recipe at 100 epochs (v6s100) and at ViT-B / 400
epochs (v6b400).

## Phase A — epochs 1–197, weight decay 0.05 on everything

Everything trained normally. The online linear probe rose steadily to ~0.615.

The concerning trend was in the projector. The two pre-BatchNorm weight matrices were **shrinking**:
|W0| 82 → 66 and |W3| 192 → 154 between epoch 100 and epoch 196. BatchNorm-2's running input
variance fell with them, median 0.277 → 0.096.

Mechanism, established on the siblings: a layer feeding a BatchNorm is scale-invariant, so weight
decay shrinks it unopposed while its *effective* learning rate (lr/|w|²) rises as the cosine drives
lr down. Eventually the BatchNorm is normalising a nearly-constant signal and amplifying noise.
v6s100 hit exactly this at epoch 85 of 100: 291 clipped-gradient bursts in 22 hours, the forward
pass exploding (invariance loss to 7e5, total loss to 2e7), BatchNorm-2's running variance
collapsing 11× to a minimum of 2.8e-4. v6b400 hit it at epoch 264 of 400.

The h conditioner term was **improving** throughout this phase, about 0.0010 per epoch
(0.4185 at ep170 → 0.3956 at ep194).

## Phase B — epochs 198–277, projector weight decay set to 0, plus a skip guard

Applied pre-emptively before any storm, from the running state (setting weight decay to zero is a
no-op on the function at the moment you do it). The guard discards an optimizer step whose
rank-max invariance loss exceeds 10× its running mean, or whose z conditioner exceeds 3× its
running mean; all ranks skip together.

What we observed:

| | ep199 | ep231 | ep276 |
|---|---|---|---|
| \|W0\| | 105 | 263 | 335 |
| \|W3\| | 245 | 626 | 786 |
| BatchNorm-2 variance, median | 0.9 | 47.5 | 141 |
| BatchNorm-2 variance, minimum | 0.21 | 3.60 | 1.83 |
| BatchNorm-2 variance, 1st percentile | 0.67 | 20.5 | 14.4 |
| steps per epoch with z conditioner > 0.3 | 0 | **0** | 154 |
| steps per epoch discarded by the guard | 0 | 127 | 272 |
| online probe | .616 | .631 | .649 |

So removing weight decay reversed the shrinkage into **unbounded growth** — |W0| tripled in 77
epochs, and BatchNorm-2's variance rose 157×. Meanwhile the *bottom* of the variance distribution
turned around at about epoch 231: the minimum went 3.60 → 1.83 and the 1st percentile 20.5 → 14.4
while the median tripled. The z conditioner spikes were exactly **zero until epoch 231** and appear
only after that turn.

The h conditioner term dipped at the switch, then **degraded** for ~25 epochs (about 0.0008 per
epoch), then resumed improving from ~epoch 224 at roughly its old rate.

Per-step forensics comparing three 6-epoch windows (ep219-224, ep240-245, ep271-276):

- typical invariance loss **falling**: 0.363 → 0.352 → 0.339
- typical z conditioner **flat**: 0.1207 → 0.1219 → 0.1209
- typical gradient norm **rising**: 26.5 → 29.2 → 35.6
- steps with z conditioner > 0.3: 10 → 79 → **655**
- steps with invariance loss > 5× median: 418 → 578 → 640 (only ×1.5)
- at the spiking steps the invariance loss is at its **typical** value, not elevated
- the spikes arrive in runs of **exactly 4 steps** (128 of 197 runs) — the ring length

For contrast, when the ViT-B sibling stormed under zero weight decay, each 4-step run *was* opened
by an outlier batch with invariance loss 3 → 96 → 3588 against a typical 0.29. That is no longer
the pattern here.

## Phase C — epoch 278 to now, projector weight decay 8e-4

Reasoning: with weight decay the scale-invariant weight settles where decay balances gradient
noise, roughly `|w|² = lr·d/(2·wd)`. At 0.05 that settling point collapses as lr decays (phase A's
failure); at 0 there is no settling point at all (phase B's failure). We picked the decay so the
settling point equals the norm the run already had: `wd = lr·d/(2|w|²)` gives 8.2e-4 for W0 and
7.9e-4 for W3, so 8e-4 for both. Cross-checked against the measured growth rate (0.095 per step for
|W0|², consistent with the formula once the lr drift is accounted for).

Observed over epochs 278–300:

- **|W0| flat at 336** for twenty epochs, having been gaining ~1 per epoch. |W3| 788 → 785 and
  |W6| 320 → 317, both now drifting down. The intended effect.
- **h conditioner fell steadily** — 0.3357 → 0.3221 over eighteen epochs (steps of −0.0024,
  −0.0027, −0.0037) — then **reversed +0.0046** at epochs 297–298.
- **z conditioner spikes kept climbing**: 546 → ~750 → **1311** per epoch. Guard now discards
  ~10% of all optimizer steps.
- BatchNorm-2 minimum variance drifted 1.75 → 0.81 → 0.94.
- **The online probe kept improving throughout and is at its best of the run, 0.6564.**

Current BatchNorm-2 variance distribution: min 0.813, 0.1% 1.58, 1% 12.5, 10% 69.9, median 138,
max 376. BatchNorm-1 has carried exactly one channel at 5.6e-45 (denormal zero) since epoch 199.

One hypothesis was tested and rejected: that a few collapsed-variance channels are amplified by
BatchNorm and drive the spikes. The bottom 1% of channels carry only 6.1% of z's energy, and the
largest BatchNorm gain is 9.3× the median — too small to explain 1300 spikes per epoch.

## The situation in one paragraph

The centre of every loss is healthy and the probe is at its best. What has grown, monotonically and
across all three phases, is the **tail** of the z conditioner: its median has not moved at all
(0.1207 → 0.1209) while the count of large excursions went from 0 to 1311 per epoch. Neither
weight-decay setting stops it — 0.05 shrinks the projector to collapse, 0 grows it without bound,
and 8e-4 holds the norm exactly as intended yet the spikes continue. About 100 epochs remain.

## Questions

1. **What is actually producing the growing tail?** The median is pinned while the tail explodes,
   the spikes land on ordinary batches, and they come in runs equal to the estimator's ring length.
   Note the estimator has only **4 samples per dimension** (512 rows, 128 dims). Is the growth a
   real pathology in `z`, or is it a heavy-tailed estimator at marginal sample size becoming heavier
   as `z`'s covariance grows more anisotropic — i.e. an artefact against a fixed 0.3 threshold?
   What measurement would separate those two?

2. **Is discarding 10% of steps harmful or protective here?** The guard's z trigger is a ratio to a
   running mean, so a heavier tail trips it more often even if nothing is wrong. Should the trigger
   be re-calibrated, replaced with an absolute threshold, or removed for the z term while keeping it
   for the invariance term?

3. **Does the h conditioner's reversal at epoch 297 matter?** It reversed once before (at the
   phase-B switch) and recovered on its own after ~25 epochs. Is there a reason the two reversals
   would share a cause?

4. **Is there a better control for a scale-invariant layer than weight decay at all?** Candidates we
   have considered: fixing the norm explicitly after each step and rescaling the following
   BatchNorm's running statistics to match (exactly function-preserving); weight standardisation;
   replacing BatchNorm with LayerNorm in the projector. Which of these actually addresses a growing
   tail as opposed to a drifting norm?

5. **Should the estimator's ring be widened** (`queue_steps` 3 → 7, doubling the rows behind each
   estimate) as a cheap test of the sample-noise explanation? What would you expect to happen to the
   spike count if the noise explanation is right, and what if it is wrong?

6. **Given ~100 epochs remain and the probe is at its best, is the right move to intervene at all?**
   What evidence would justify stopping versus letting it finish?
