# Brief: what we have, what we are after, and what has already failed

Self-contained. No project vocabulary is assumed.

## 1. The setting

Two self-supervised vision models, identical in every respect except one term in the loss.

- Backbone: ViT-S/16 at 224px. Data: ImageNet-100 (100 classes, 126,689 train images, 5,000
  val). 100 epochs, seed 0, identical augmentations (random resized crop, colour jitter,
  grayscale, blur, solarize, flip), V = 4 views per image per step, batch 128.
- **h** = the backbone's CLS output, 384-d. This is the representation we care about: it is what
  a downstream user gets, and it is the input to the projector.
- **z** = the projector's output, 256-d. The projector is
  `Linear(384→2048) → BN → ReLU → Linear(2048→2048) → BN → ReLU → Linear(2048→256)`.
  **The training loss is computed only on z.**
- The loss has two terms:
  - `inv` — invariance. Mean squared distance between the z's of the 4 augmented views of the
    same image, averaged over the 256 dimensions of z. Weight 32.8.
  - `cond_z` — anti-collapse. A Gaussian moment KL that pushes the batch distribution of the
    per-image mean z toward N(0, I) on a fresh random 128-d subspace each step. Weight 157.8.
    Without it, `inv` alone is minimised by mapping everything to a single point.
- **Untreated model** (`d256vm4zonly`): exactly the loss above.
- **Treated model** (`d256vm4`): the same, plus the *same* anti-collapse term applied to h as
  well, weight 1.894. This is the intervention whose value we are trying to explain.

Both are trained; nothing here retrains anything unless a design asks for it.

## 2. Where the two models end up

Numbers below are ours, measured, and are the baseline any new design should reproduce.

| | untreated | treated |
|---|---|---|
| linear probe on h (top-1, 50k train → 5k val, plain Linear, AdamW, converged) | 70.24 | 72.02 |
| kNN-200 on h | 57.84 | — |
| invariance loss at z (`inv`) | 0.134 | 0.191 |
| view-spread ÷ image-spread at z (scale-free invariance) | 0.138 | 0.127 |
| view-spread ÷ image-spread at h | 0.255 | 0.373 |
| effective rank of h (of 384) | 53.9 | ~200 |

Read plainly: **both models achieve the same invariance at z, and the treated one is 1.8 points
better downstream.** The treatment also leaves far more structure in h (higher rank, more
view-variation) which the projector then removes.

## 3. The claim we want to test

*The treated representation retains content that the training objective never required, and that
content is useful for classification.*

Equivalently, in the sharpest form we could not establish: **there exist directions in h that are
harmful or irrelevant to the invariance objective, yet useful for a linear classifier** — and the
treated model has more of them than the untreated one.

## 4. What we can measure cheaply (no retraining)

- Both checkpoints, so the projector can be applied to any modified h; the frozen-head recompute
  reproduces the stored z to cosine .999992.
- Stored features: clean h for 50k train and 5k val images; **8 augmented views of h for all
  126,689 train images**, drawn from the training augmentation family. So the invariance loss and
  the anti-collapse term can be re-evaluated exactly, offline, on any modified representation, at
  the training estimator's batch shape.
- A linear probe run takes about a minute on one GPU; a full re-evaluation of the loss over 4,096
  images takes about a second. Everything below ran in 10–20 minute jobs.
- Retraining a model is ~10 GPU-hours and is possible but has not been used here.

## 5. What we already tried, and exactly why each attempt failed

**(a) Delete the lowest-variance directions of h, watch loss and probe.** Failed: with a linear
probe, deleting directions is identical to forbidding the probe to use them, so accuracy can only
fall — the direction of the result is guaranteed and only its rate carries information. Worse, the
representation is shaped only by the objective, so the low-variance tail is residue: the 200
directions whose deletion moves the loss least hold 1.7% of the variance and cost 0.34 points when
all are deleted together.

**(b) Score each direction by how much the loss changes when it alone is deleted, then compare to
how much the probe loses.** Failed as a test of the claim: the two scores correlate at +0.76
(untreated) and +0.42 (treated), and the directions the loss ignores are within the probe's noise
floor (±0.1 point, set by the directions whose removal *raises* accuracy).

**(c) Count directions where removal leaves the invariance loss no worse and costs the probe.**
Failed by construction: the invariance loss is a sum of squared distances, so removing *anything*
lowers it. The condition is satisfied by 383/384 and 384/384 directions — the filter does no work,
and the count degenerates into "directions the probe uses".

**(d) Rank directions by how much they move across views of one image (what the invariance term
would like removed), then remove them.** Failed in the other direction: the probe barely uses
those. Above a view-noise share of 0.7 there is not a single direction the classifier uses, in
either model, and the top-25 worst-for-invariance directions change the invariance loss by 0.0–0.6%
because the projector already discards them.

**(e) Split h into the subspace the projector transmits and the subspace it kills (least-squares
fit of the head, singular directions), probe each half.** Inconclusive: the linear fit explains
only ~55% of z (so "killed" may still reach z nonlinearly), the null space is 128-d by arithmetic
alone (384 in, 256 out), and the result came out reversed — the *untreated* model's killed half
reads 67.24 on 51% of its variance, the treated model's 64.78 on 33%.

## 6. The structural obstacle a new design must solve

The invariance term, alone, has no preference that anything can violate. It is minimised by
throwing everything away. So any statement of the form "the objective does not want this direction
but classification does" is vacuous: the objective does not want *any* direction. Preference is
supplied only by the anti-collapse term — which is the very thing the treatment adds. Every
framing we tried collapsed back into this circularity.

A design that works must therefore either
1. make the objective able to get **worse** under the intervention, not only better (deletion
   cannot do this; replacement, re-fitting, or re-training can), or
2. compare **at matched objective value** — hold what the loss achieves fixed and ask what differs
   downstream, or
3. abandon localisation to directions and test the claim at the level of the whole
   representation, where it already holds (same invariance at z, +1.8 points at h).

## 7. What a good answer would look like

A measurement, computable from the two frozen models (or from one extra training run), whose
outcome could plausibly have come out either way, and which distinguishes:

- **H1:** the treated h carries class-relevant content that the objective did not require, and
  that is where its +1.8 points come from; versus
- **H0:** the treated h is merely a better-conditioned version of the same content, and the +1.8
  points come from the probe finding it easier to read, not from extra content being present.

Nothing we ran separates H1 from H0. That separation is the ask.
