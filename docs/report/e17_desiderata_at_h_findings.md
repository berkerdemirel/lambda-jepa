# What does the SSL projector actually do — and which desiderata belong in the backbone?

**Findings report (E12/E17), 2026-07-16.** Self-sufficient summary for a reader with a deep-learning
background but no knowledge of this project. All numbers are single-seed (0), ImageNet-100,
ViT-S/16 @224, 100-epoch controlled retrains under one training frame; every comparison is against
a byte-identical control (same code, config, seed — the added term is the only factor). Approved
takeaways: T1–T7 (§6); ledger row D-039.

## 1. Background and question

Modern self-supervised methods (SimCLR, BYOL, DINO, VICReg, LeJEPA) share an architectural
convention: the loss is applied not to the backbone representation `h` (here: the ViT trunk output
actually used downstream) but to `z`, the output of a small MLP projector that is thrown away after
training. The field's justifications conflict. Some methods state *distributional claims about the
representation itself* (VICReg: decorrelated unit-variance dimensions; LeJEPA: isotropic
Gaussianity) yet enforce them only at `z`. Others treat the projector as a *protective buffer*
(Bordes et al.): the constraint would damage `h`, so it is kept at arm's length. If the buffer view
is right, the constraint should hurt when moved to `h`; if the methods' own claims are right, `h`
should be able to carry its method's property — and perhaps benefit.

**E17 tests this directly: for each method, add a small copy of the method's OWN loss term(s) at
`h`, on top of the unchanged full-strength loss at `z`, and measure what changes.** "Small" =
dosed to ~10% of the shipped z-term's measured gradient pull on the encoder (per-term encoder-grad
norms measured on the real first batch; doses fixed before launch). Two arms per method where the
loss decomposes: the **non-collapse term** (the part that prevents representational collapse:
uniformity / variance+covariance / SIGReg) and, separately, **+ the invariance term** (the part
that pulls augmented views of one image together). BYOL and DINO have entangled objectives and get
one faithful arm each (BYOL: predictive alignment of trunk features onto the EMA teacher through a
single trainable linear map; DINO: a small weight-normed linear prototype head at `h` with
centering/sharpening/cross-entropy — a faithful mini-DINO, K=512, its own EMA teacher head and
center). A companion E12 result enters throughout: **f2**, a *method-agnostic proxy* term at
LeJEPA's `h` — a Gaussian moment floor (defined in §5) at very small dose — which had produced the
largest neighborhood gains we had seen and which E17 was designed to explain.

**Probes and instruments** (all computed on frozen features, 50k train / 5k val):
- `lin` = converged linear probe accuracy (shift-invariant); `knn` = weighted-cosine kNN, k=200
  (the standard SSL leaderboard probe; NOT shift-invariant — this matters, §3.1).
- **View triple** on an 8-view augmentation orbit of 10k images: `pos` = mean cosine between views
  of the same image; `rand` = mean cosine between random pairs; `margin = pos − rand`. Measured at
  `h` and at `z`; the **head's margin jump** = margin@z − margin@h, decomposed into its pos-side
  and rand-side parts.
- **Spectrum/shape battery**: effective rank of the (centered) covariance; worst-direction excess
  kurtosis over the top covariance eigendirections; a sliced Epps–Pulley statistic (distance of
  random 1-d projections to Gaussian, after per-dimension standardization); diagonal moment-KL
  (per-dimension mean/variance vs N(0,1) — deliberately not centered).
- **Class-geometry instruments** built for this study: class-pair separation d′ (project on the
  difference of two class means; d′ = |μ₁−μ₂|/√((σ₁²+σ₂²)/2); 1000 random class pairs — mean and
  10th percentile); **augmentation-cloud connectivity** (per image: view-cloud radius r vs distance
  to the nearest same-class image centroid; `touch%` = fraction of images whose cloud reaches its
  same-class neighbor, 2r ≥ d); head-map geometry preservation (CKA / neighbor-Jaccard /
  Procrustes between `h` and `z`).

## 2. Headline findings

### F1. The projector's celebrated "invariance" work is, concretely, mean-removal.

The low view-margin at `h` is not caused by poor view alignment — it is caused by a large **shared
mean vector**: in every method, the mean cosine between *unrelated* images (`rand`) equals the
mean's share of feature energy to three decimals (`rand ≈ ‖μ‖²/E‖x‖²`; centering makes rand ≈ 0).
Decomposing each control's head jump: it is **entirely rand-side** — the head removes this cone —
while the pos-side contribution is *negative* (view alignment gets slightly worse across the head)
in SimCLR, BYOL, VICReg and LeJEPA. Only DINO's head does mostly pos-side work (its cone is already
small). So the quantity the projector visibly manufactures is not invariance; alignment is born in
the trunk. What the head does is remove a mean offset and spread the remainder.

### F2. Every method's own term can take that job over at `h` — but the head is never "relieved".

Dosed at 10% pull, each method's non-collapse term removes the cone at `h` (SIGReg, uniformity:
rand → .002–.005; BYOL's predictive alignment: .77 → .19 emergently, with no explicit spread term;
VICReg see F8), and the head's margin jump collapses accordingly (e.g. +.54 → +.15 LeJEPA,
+.41 → +.16 SimCLR, +.51 → +.10 BYOL). Adding the method's own invariance term drives the jump to
≈ 0 — the head does no measurable margin work at all. Yet the **final z-loss values end equal or
worse in every arm except BYOL's** (LeJEPA's projector-SIGReg +21%/+43%; SimCLR's NT-Xent +.06;
DINO's CE +.06; BYOL −4% — the one case where the h-term is the same functional as the z-term).
The z-space geometry (pos/rand at z) is reconstituted almost unchanged in every arm. The projector
behaves as an **adapter**: constrain its input and it re-morphs to deliver the same `z`, at equal
or higher cost — the role it drops (cone removal) is replaced by a new one (re-aligning the views
its input lost).

### F3. Making `h` satisfy the loss makes the head map *gentler* — and gentleness is independent of quality.

Head-map geometry preservation (h~z CKA / Procrustes) rises monotonically with how *fully* the
z-desiderata are satisfied at `h`: SimCLR CKA .49 → .74 (uniformity) → .88 (+alignment); LeJEPA's
+inv arm reaches CKA .955 / Procrustes .18 — a near-isometry. But: (i) *partial* satisfaction makes
the head wilder (decorrelated-but-disaligned input adds re-alignment work), (ii) the off-desideratum
proxy f2 produced the **wildest head measured (CKA .82 → .46) on the best representation**, and
(iii) the gentlest head (LeJEPA +inv) sits on the worst. All four cells of
{gentle, wild} × {good, bad} occur. A gentle projector is purchasable; it certifies nothing.

### F4. Whether the transfer *helps* splits the terms into three families.

Vs byte-identical controls, at the declared `h`:

| family | arm | Δlin | Δknn(k=200) | signature |
|---|---|---|---|---|
| **spread** | SimCLR + uniformity | +0.3 | **+6.3** | rank 51→134, worst-tail d′ ↑ |
| **spread** | BYOL + predictive-align | **+4.1** | **+5.4** | both spaces improve; z-loss improves |
| **spread (proxy)** | LeJEPA + moment floor (f2) | +1.2 | **+7.5** | rank 22→217, p10-d′ +15% |
| **spread (slow)** | VICReg + own var+cov @6× dose | +0.6 | **+4.4** | cone broken at convergence (F8) |
| **shape** | LeJEPA + own SIGReg | −1.4 | **−4.3** | mean class-pair d′ −17%, tax reaches the trunk |
| **assignment** | DINO + proto-CE | **+2.6** | **−4.2** | class margin fattens; neighborhoods fragment |

The winners share one signature: cone removed + covariance rank expanded + the **worst-separated
class pairs decrowded** (10th-percentile d′ rises; the mean barely moves) + higher-moment structure
left intact. The one loser among distributional terms, SIGReg, is precisely the one whose objective
*sees distribution shape beyond the second moments* — and a well-separated class pair IS a bimodal
1-d projection, so its gradient literally reduces class separation (measured: mean d′ −17%).
The assignment loss (DINO) is a third regime: it *reshapes* — pulling same-class images toward
shared prototypes fattens the local class margin (best linear gain of the study) while fragmenting
the fine neighborhood structure kNN uses (the kNN tax halves at k=20 — an over-clustering
signature, K=512 prototypes for 100 classes).

A caveat on the kNN gains: our kNN is cosine-based, hence mean-sensitive. Re-running kNN with
train-mean removal shows the cone-removal share of each gain (eval-time centering recovers +0.5 to
+3.4 on controls); **every gaining arm still beats its centered control** (e.g. uniformity
+6.3 raw → +4.4 after centering; f2 is centering-insensitive because its lane was already
mean-calibrated — its +7.5 contains no cone component). The gains are real reorganization, not a
metric artifact.

### F5. Alignment and neighborhoods trade — through augmentation-cloud connectivity.

The single best summary of the +invariance arms: they buy back view alignment and pay in
*connectivity*. Per image, the 8 views form a cloud; kNN quality lives on whether these clouds
reach the nearest same-class neighbors (intra-class nearest-neighbor distances are only 85–96% of
inter-class ones — margins are thin, cloud geometry decides neighbors). **Within every family, at
fixed spread term, kNN tracks touch% monotonically as the invariance dose rises**: the moment-floor
family C1 → f2 → f8 (weak inv) → f7 (strong inv) gives touch 87.6 → 99.7 → 95.3 → 91.2 against
knn 53.1 → 60.5 → 60.2 → 57.6; likewise SimCLR (96.9 → 81.0, knn 55.9 → 52.1) and LeJEPA's own-term
pair. Figure: `results/figures/e17/e17_touch_vs_knn.png` — every +inv leg points down-left. Two
sharpenings: the benefit is **threshold-like** (f8 keeps all of f2's kNN at touch 95.3 — clouds
must touch, overlap need not be maximized), and a **shape-dominant mixing ratio** (f8: floor:inv
pull = 4.5:1) adds linear-probe gain at zero kNN cost — the favorable corner of the trade. Driving
the head's burden to exactly zero is *pyrrhic*: the strong-inv arms collapse effective rank
(LeJEPA +inv: 24.5 → 14.9, below control) and post the worst kNN of the study (−11.6).

This connects the finding to the theory line that grounds contrastive generalization in
augmentation-graph connectivity (HaoChen et al. 2021; Wang et al. 2022): the invariance term at
`h`, pushed hard, disconnects the graph the guarantees need; spread reconnects it.

### F6. The metrics disagree because they measure disjoint things (and that is a result).

On LeJEPA's +inv arm: diagonal moment-KL says "best-calibrated ever" (.041), Epps–Pulley says
"least Gaussian ever" (1106, worse than control), kurtosis says "fine" (1.09). No contradiction:
diag-KL reads only marginal location/scale (blind to all joint structure — a rank-15 covariance can
have unit diagonal), kurtosis reads only 4th moments along top directions, and the sliced EP test —
normally foolable by the CLT on high-rank data — becomes *hyper-sensitive* at low rank (a random
slice of a rank-15 object sums too few factors to Gaussianize). None of the three reads the
covariance spectrum, which is where this arm actually failed. Practical consequence adopted here:
isotropy claims need a ladder — mean share, spectrum (effective rank / correlations), directional
tails, and slice-shape *conditioned on rank* — not any single statistic.

## 3. The mechanism of the winner (why the "moment floor" organizes features)

The proxy term behind f2 — the largest clean gain of the study — is, per training step:

1. draw a **fresh random orthonormal frame** Q ∈ ℝ^{512×128} (unseeded; new directions each step);
2. project the batch, take mean μ_Q and full covariance Σ_Q of the projection;
3. penalize KL(N(μ_Q, Σ_Q) ‖ N(0, I))/128 = ½(‖μ_Q‖² + tr Σ_Q − 128 − log det Σ_Q)/128,
   with total weight λ = .02 (≈ 2% of the shipped loss's encoder pull).

Two properties explain its behavior:

- **Graded, weakest-first floor.** The per-eigenvalue cost k(λ) = ½(λ − 1 − ln λ) has slope
  ½(1 − 1/λ): magnitude ≈ 24.5 at λ = .02 but only 0.38 at λ = 4 — a ~65× asymmetry. At ε-dose,
  the only gradients that survive are from near-dead directions and the mean: the term acts as a
  **spectral floor plus mean-remover**, not a whitener. At destination dose the residual pressure
  equalizes healthy directions — whitening — and the same term becomes destructive (measured:
  −13.9 lin at 25× the dose). One loss, two regimes, selected by dose.
- **Non-absorbable by construction.** Because the frame rotates every step, there is no fixed basis
  in which the encoder can satisfy the constraint parametrically. The controlled contrast: the same
  moments enforced in a *fixed coordinate basis* are absorbed by a diagonal rescale of the last
  linear layer while the variance flees into correlated heavy-tailed directions — no reorganization,
  no gain, even at 27× the dose.

Why does a floor *organize* anything? At effective rank ~22, a hundred classes' worth of structure
shares directions (superposition); collisions manufacture spurious neighbors. The floor revives
~200 directions and — being a function of batch mean and covariance only — **cannot dictate what
fills them**: it is cluster-blind. The measured footprint says the data fills them with structure:
class-assignment consistency holds (kmeans-NMI .53), the *worst-separated* class pairs gain the
most (p10-d′ +15%, mean flat), coordinate kurtosis stays leptokurtic (higher moments remain
data-driven), and connectivity jumps (touch 99.7%). The comparison with SIGReg is then exact:
SIGReg = the same sliced machinery matching the full characteristic function = **this term's
channel PLUS a shape channel** — and the shape channel's gradient points at the class structure.
The winner is not a better regularizer but a *strictly smaller* one.

**Is N(0, I) essential?** μ = 0: yes — the mean direction carries no class information but dominates
cosine geometry; removing it never hurt anywhere. Σ = I: only half — the essential half is the
floor (no dying directions, swept over all bases); the equalizing half is harmful at full dose and
approximately inert at ε-dose; the unit scale is convention. N(0,I) earns its place operationally:
it is the unique target making the sliced statistic (a) closed-form in batch moments, (b) convex
with a one-sided-infinite collapse barrier, and (c) coherent under random slicing (rotation
invariance means every slice shares one target). The *declared* desideratum our data supports is:
zero mean, spectrum bounded away from zero, everything else free.

### F8 (correction en route). VICReg's own term: mean-blind, hence slow — but sufficient at convergence.

VICReg's variance+covariance term is computed on batch-centered features: it cannot see the mean
that carries the cone. Mid-training reads suggested it could only dilute the cone; the converged
6×-dose run corrects this: the variance hinge recruits centered variance to saturation
(tr Σ 19 → 362 ≈ d), diluting the mean share to ~.12 on its own, the residual mean halves through
drift, and by epoch 100 the cone is broken (rand .035) with the winners' signature (knn +4.4,
rank 67 → 157). So the correct statement is **dose-inefficiency, not impossibility**: the
mean-*seeing* floor achieves the same at λ = .02 (~300× less relative pull — separately verified:
it zeroes VICReg's head jump on its own). VICReg remains the method whose z-space is the least
class-aligned we measured (between-class variance share .16 at z vs .31 at h).

## 4. What this says about the projector debate

The "protective buffer" and "the desideratum belongs at h" views are **both right, per component**.
The buffer is genuinely protective against *shape* pressure (SIGReg at h taxes into the trunk) and
against *hard invariance* (connectivity collapse) — the projector is where geometry too violent for
a residual-stream backbone gets done (per-layer operator norms 15–150, segment stretches 0.07–10).
But the *affine/spread* component — mean removal plus a spectral floor — belongs in the backbone:
every method tested improves its representation by carrying it (the one apparent exception, VICReg,
joins at sufficient dose). The field's practice of ablating projector depth while holding the trunk
sacred searches the layer axis; this study searched the component axis at fixed head, and the
component axis is where the structure was.

## 5. Approved takeaways (T1–T7, verbatim from the experiment card)

1. **T1** The projector's "invariance jump" is mean-removal; alignment is born in the trunk; any
   mean-seeing own-term at h takes the job over.
2. **T2** h-satisfaction gentles the head map (∝ total satisfaction) but never relieves it
   (z-losses equal-or-worse except BYOL); gentleness ⊥ representation quality (all four cells
   realized).
3. **T3** The transferable component is cone-removal + spread with clusters intact (rank ↑,
   worst-tail class separation ↑); shape enforcement taxes; assignment reshapes (lin/knn
   dissociation).
4. **T4** Alignment and neighborhoods trade through augmentation-cloud connectivity; threshold-like;
   shape-dominant mixing is the favorable corner; zero head-burden is pyrrhic.
5. **T5** (amended at convergence) VICReg's mean-blind term removes the cone indirectly and
   dose-hungrily; at 6× it joins the winners; the contrast with mean-seeing terms is
   dose-efficiency, not possibility.
6. **T6** "Gaussianity at h" decomposes into safety (cluster-blindness) × activity (non-absorbable
   swept spectral floor); Gaussian shape is incidental; higher moments should stay data-driven.
7. **T7** Guillotine per method: LeJEPA protective; SimCLR's spread half belongs at h; BYOL fully
   portable (anti-buffer); DINO reshapes; VICReg dose-inefficient. The buffer survives as the
   shape-work hostel.

## 6. Limitations

Single seed; one dose point per arm (the SIGReg arm carries an over-dose flag — its view-alignment
dropped to .615 mid-run; lower doses unmapped); ImageNet-100 scale; ViT-S/16 only; the moment-floor
comparisons (f2/f7/f8) ride a slightly different training lane than the E17 arms (each family is
internally controlled; cross-family effect sizes carry a lane offset); kNN convention is
weighted-cosine (mean-sensitive — quantified and controlled via centered re-runs); the
connectivity measure uses first-moment cloud proxies (radius/centroid), not support overlap.
Augmentation families are per-method by design (recipe fidelity): the moment-floor-at-h result
(f2) is established under LeJEPA's aug family (4 symmetric strong-photometric views) and its
effect size need not transfer across aug families — the view pipeline also shapes the
augmentation-cloud geometry that the connectivity findings measure.

## 7. Open next steps (agreed direction)

(i) upgrade touch% to a per-class reachability-with-cost analysis (travel all instances through
cloud overlaps; MST cost over cloud-gap edges) and read it jointly with centered view-margin and
kNN; (ii) test the floor as a standalone z-space non-collapse term (it is, to our knowledge, not a
published SSL regularizer in this sliced-KL form; nearest relatives: CorInfoMax's log-det barrier,
W-MSE's whitening, VICReg's axis-aligned moments, MCR²'s coding-rate expansion — none sliced,
ε-dosed, or placed at h); (iii) motivate mean-removal as a first-class design element; (iv) the
declared-prior arm — replace the Gaussian slice-target with Student-t_ν (ν fit from the measured
coordinate kurtosis) at identical dose: if the SIGReg tax persists, shape enforcement per se harms;
if it vanishes, the target (not shape enforcement) was wrong.
