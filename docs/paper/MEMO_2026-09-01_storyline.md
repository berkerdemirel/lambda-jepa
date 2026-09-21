# Paper storyline memo — 2026-09-01 (DRAFT for discussion; nothing here is AGREED)

> **Revisions after Berker's read (09-01 evening).** (1) Accessibility to the appendix:
> agreed. (2) Thickness: keep the flow gap -> invariance is not the target -> define
> thickness -> thickness decides what the objective keeps and feature learning makes the
> backbone follow -> regularizer; the "second axis" framing is dropped in favor of that
> single line (now in `skeleton.tex` Sec. 3). (3) No paired IN-100 transfer run: the paper
> is already experiment-heavy; item 5.1 is withdrawn. (4) E31: no evidence from it in the
> paper and no strong statements; the dose panel and item 5.3 are withdrawn. (5) ViT-L:
> wait for the running arms; not in the main table (no OK-AI L cells, paper heavy); an
> appendix row at most. (6) Method name: open, not "floor". Sections 2, 3 and 5 below
> should be read with these overrides.

Scope: the redo of the ICLR 2027 research memo after Berker's correction of 09-01
("the paper starts with distributional claims not holding at h; invariance is not
what we want either, so thickness matters; task-agnostic features should be dense;
rich learning aligns features to the task; regularize to keep active dimensions
high"). Base document: `docs/paper/skeleton.tex`. Vocabulary per `docs/GLOSSARY.md`
(method = `spectral`; the term is a two-sided spectral conditioner; "floor" is used
below only for the linear-theory lower bound sigma_min >= sqrt(c), which is literally
a floor). Numbers are quoted from the CSVs and cards as of 09-01; RAW vs AGREED is
marked where it matters. Nothing in `docs/paper/*.tex` was changed.

## 0. Thesis

Self-supervised objectives constrain a projected space z, and their distributional
guarantees do not flow to the retained backbone h; invariance is not the target
either, since z is the invariant object and nobody deploys it. What h should be is
characterized on two axes, stable image-to-image spread and relative augmentation
thickness, and a task-agnostic h must be broad on the first axis and moderate on the
second. Rich feature learning aligns h to the pretext task and concentrates it onto
the few directions view agreement needs; a two-sided spectral conditioner on the
backbone's per-image-center covariance, derived from the layer-balance mechanism of
two-layer networks, gives every stable direction a lower bound on its variance while
the directions the loss uses keep the variance the loss gives them. Learning stays rich, capacity does not concentrate, downstream gains come from
better center organization rather than from less invariance, and the same objective
is a competitive standalone SSL method at ImageNet-1k scale.

Title candidates (not decided): "What the retained representation should be:
capacity and thickness beyond the projector" / "Keep every direction: spectral
conditioning of the backbone in projector-based SSL".

## 1. The causal chain (one storyline, five beats)

**Beat 1 — Claims made at z do not flow to h.** LeJEPA's Gaussianity, VICReg's
variance floor, VISReg's isotropy and SimCLR's alignment are imposed at the loss
space. Theorem 2.1 (Z-only objectives identify only the factorization class) says
they cannot certify a non-invariant property of h; `fig_discrepancy` (C1, AGREED)
shows the same three desiderata met at z and failed at h on the untreated controls
(LeJEPA moment-KL .005 at z vs 3.74 at h; VICReg variance .99 vs .46; SimCLR margin
.84 vs .40). In the two-layer linear model with h = W1 x and z = W2 h this is literal:
any loss of the end-to-end map W2 W1 conserves the imbalance Delta = W2^T W2 - W1
W1^T, so the objective never chooses how capacity is split between backbone and
projector. That lemma is already in the coauthor draft and is the right one-line
bridge from Beat 1 to Beat 3.

**Beat 2 — Invariance is not the target either; h differs from z on two axes.** If
invariance were the goal we would deploy z. The paper needs a formal object for
"what h keeps beyond z", and it has two coordinates per stable direction u of h:
stable variance b(u) = u^T B_h u (image-to-image) and relative thickness
theta(u) = u^T A_h u / u^T B_h u (within-image cloud over image spacing; Theta_h =
B_h^{-1/2} A_h B_h^{-1/2} is the matrix form). z is the degenerate corner of that
plane: few active directions and thin. `fig_treatment_main` shows both drops at the
CLS-to-z boundary for every method (the RankMe/d and Theta columns). The two-sided
theorem gives the two failure modes on the thickness axis and is the formal version
of "invariance is not what we want": too thin, d_h(y) >= (sqrt(E_A(y)) -
sqrt(||Theta_h||))_+, a representation that is too invariant cannot organize
distinctions that augmentations obscure; too thick, the view-to-center fidelity term
a^T Theta (I + Theta)^{-1} a is monotone in Theta for fixed organization. E32-T1
(AGREED, D-110) ties the two axes: the usable size of h is the number of directions
that are both active and quiet (theta < .25), and the class signal concentrates there.
So the target h is "many stable directions above a barrier, sitting in a moderate
thickness band"; the method controls the first coordinate and the second is measured.

**Beat 3 — Task-agnostic use wants h dense, and the objective will not give that.**
"Dense" is the design premise: the downstream task family is unknown, so h should not
discard stable directions merely because view agreement does not need them. Rich
feature learning aligns representations to the training objective: in supervised
deep linear nets the balanced small-init solution allocates encoder variance
a_i^2 = s_i to the task's singular modes and nothing to the rest (Saxe et al.;
silent alignment, Atanasov et al.; neural collapse, Papyan et al.), and in SSL the
same product-of-layers implicit bias is the dimensional-collapse mechanism (Jing et
al. 2022; Tian et al. 2021 for the non-contrastive case; verify each statement's
setting before citing). In SSL the training objective is view agreement, which needs
few directions of h relative to d_h, so alignment is concentration: h ends with about
as many active directions as the anchor needs. This must never be written as "rich
learning is sparse or low-rank" in general; it is "rich learning aligns h to the
pretext task, and the pretext task is narrow". Evidence that the base objectives do
this: the C2 controls sit at low stable-center rank (LeJEPA 43, BYOL 110, SimCLR 128
of 384), and E28 (AGREED, failed) shows init scale alone cannot buy the rank back.

**Beat 4 — A conditioner that keeps every stable direction without freezing
learning.** In the linear model, L2 on both layers plus -gamma/2 log det(W1 W1^T)
drives Delta -> -c I with c = gamma/lambda_reg, independent of init and of the loss
(the task terms cancel in the balance ODE), and the allocation becomes

    a_i^2 = (c + sqrt(c^2 + 4 s_i^2)) / 2,   so   a_i^2 ~ s_i for s_i >> c   and   a_i^2 >= c always.

That single equation is the thesis: rich in the modes the objective uses, floored
elsewhere, never lazy, never collapsed; sigma_min(W1) >= sqrt(c). It sits in the
coauthor draft's appendix marked "not needed" and is the most important equation in
the paper. In the nonlinear network the balance identity does not survive but the
spectral barrier does: apply the same KL-to-N(0, I) conditioner to the covariance of
per-image centers at h on random orthonormal slices. Centers, not views, because the
pooled-center covariance is B + A/V, so at V = 6 only a sixth of the augmentation
spread pays the conditioner; the method floors stable spread and leaves thickness to
the invariance term, which is exactly what Beat 2 asks for. The same conditioner at
the z tap is the loss-space anti-collapse; it is a function of the end-to-end map and
never touches the balance, which is why "loss in z only" leaves h narrow (C2
controls) and the h tap restores it. Operating dose: a few percent of trunk pull at h
(realized share ~.01-.03; v6b400 settled at .007), ~1/3 at z. The conditioner's
isolated optimum is isotropic, but the operating regime is a barrier: the E31 arms
show that pushing h toward whiteness costs ten points (Beat 5).

**Beat 5 — Evidence.** Capacity protected across seven methods (C2, x1.8-5.4 kept
stable-center rank, AGREED); learning stays rich (E33, RAW: 1-CKA .87, NTK alignment
to init .56, both far from lazy, and not richer than the untreated twin); probe gains
come from center organization, not from less invariance (C8, AGREED: organization
-.08..-.17 dominates a fidelity shift of +.01..+.06 in all six pairs; Theta rises
while accuracy rises in `fig_treatment_arrows`); the aug-quiet core grows (E32,
AGREED: ours 39 -> 70 directions, class energy .54 -> .69); more conditioner is not
better (E31, no card: baseline 73.7, h-dose x4 71.2, full-slice 63.1 IN-100 linear);
and the standalone objective is competitive at ImageNet-1k under stated confounds
(B-100: 74.2 linear / 64.3 kNN / 80.9 transfer / 28.1 mIoU).

**Main vs supporting.** Main: Beats 1-5. Supporting, one sentence in the main text
and a full treatment in the appendix: accessibility (C3, the bridge diagnostic; C3b
shows random-init twins are near-perfectly accessible at low rank, so it is not a
quality criterion and cannot motivate the conditioner), the fidelity law (C4), the
G/S split (C5), the too-thin per-class exhibit (C11, reading pending), depth
profiles (C7), trajectories (C6).

## 2. Recommended structure (base: skeleton.tex)

| Sec | Title | Content | Exhibits | Change vs skeleton |
|---|---|---|---|---|
| 1 | Introduction | Beats 1-3 in three paragraphs, Beat 4 in one, contributions | none | reorder; accessibility reduced to one clause |
| 2 | What the retained representation should be | 2.1 the projector gap: Thm 2.1, the conserved-imbalance remark, C1; 2.2 two axes: B_s, A_s, Theta, z as the degenerate corner, the two-sided theorem (compact statement), the dense premise | `fig_discrepancy`; optional small schematic of the (b, theta) plane with z / control h / treated h | merges skeleton 3.1 and 5; skeleton 3.2 (accessibility) moves to the appendix |
| 3 | The objective will not give it; a conditioner that does | 3.1 linear model: conserved balance, rich allocation a_i^2 = s_i, SSL reading; 3.2 the regularizer: Delta -> -cI, the allocation equation, sigma_min >= sqrt(c), regime remark; 3.3 nonlinear: centers on slices, B + A/V, two taps, the objective, the dosing rule | new toy figure: a_i^2 vs s_i for regularized / balanced / lazy, plus feature drift under the regularizer (replaces `Figure_1.png`) | replaces skeleton 4; promotes the singular-mode proposition to the main text; Gaussianity and XOR corollaries to the appendix |
| 4 | Experiments at IN-100 (the controlled block) | 4.1 setup, one plain sentence per metric; 4.2 capacity; 4.3 learning stays rich; 4.4 where the gain comes from and thickness stays in band; 4.5 dose: barrier, not whitening | `C2_capacity` (+ `C2b` trajectory); `e33_feature_drift` (single cell per Berker); `fig_treatment_arrows` + `C8_term_shift` + the E32 quiet-core panel; new E31 dose panel (spectrum flatness and probe vs dose) | `fig_treatment_main`, `in100_two_space_audit`, `C3b_accessibility_profile`, `fig_org_vs_sensitivity` move to the appendix |
| 5 | ImageNet-1k | main table S/B/L x 100/400; transfer; segmentation; one confound paragraph (data split, views, compute, recipe modernity, epochs); the matched-view S cells | `tab:in1k-main`, `tab:transfer`, `tab:seg` | OOD table removed (empty); placement table to the appendix once Ours-B/L rows exist |
| 6 | Related work | projector gap; collapse and rank regularizers; feature-learning regimes with supervised results labeled supervised and the SSL collapse literature cited for SSL; invariance vs information preservation | none | merge skeleton 2.1-2.4 |
| App | | proofs; corollaries; accessibility (C3, C3b); C4; C5; C11 if read; C6; C7; treatment grid; placement table and scatter; estimator and dosing details (slice, ring, OAS, share law); OK-AI recipe and data notes (D-112) | | |

Page budget (9 pages): 1 intro / 1 sec 2 / 1.25 sec 3 / 2.25 sec 4 / 1.5 sec 5 / 0.5
related / 0.5 limitations and conclusion.

## 3. Evidence map

| Claim | Status | Supporting now | Missing or weak |
|---|---|---|---|
| Z-claims do not flow to h | AGREED (C1) | Thm 2.1; `fig_discrepancy`; conserved-Delta lemma | nothing; keep to half a page |
| h differs from z on both axes | zoo closed (E20) | `fig_treatment_main` boundary columns (RankMe/d, Theta) for 5 methods; C7 depth profiles | a one-panel (b, theta) schematic would replace the 25-panel grid |
| Too thin / too thick | theory; C4 AGREED-LOCKED; C11 pending | Thm 5.2 (i)/(ii); C4 law holds out of sample; `fig_treatment_arrows` (Theta up, accuracy up) | no thickness sweep exists; C11 reading owed (pre-registered shape inverted) |
| Rich learning concentrates h onto the pretext task | literature + controls | linear allocation a_i^2 = s_i; C2 controls low; E28 init scale cannot fix it | toy panel; the r_z-vs-control-rank check (optional appendix figure) |
| The conditioner floors without freezing | theory; E33 RAW | balance ODE; allocation equation; `linear_figure` agreement eps = .01; E33 both cells rich | E33 AGREED wording; regime sign vs Domine/Kunin conventions; drift under the regularizer on the toy |
| Capacity protected at h | AGREED (C2, C2b) | x1.8-5.4 kept rank, all six pairs; ours 180 -> 370; in1k `ckpt_rank.csv` rows exist | single seed; Ours-B/L absent from the placement table |
| Gains via organization; thickness in band | AGREED (C8, E32) | C8 six pairs; E32 quiet core 5/6 methods; arrows | linear deltas for DINO and SimCLR (+1-2) are within seed noise; kNN deltas (+5..+11) are safe |
| Barrier, not whitening | RAW, no card | E31: 73.7 / 71.2 (x4) / 72.3 (scale x4) / 72.6 (shape x4) / 63.1 (full slice) / 63.0; `e31_fullspace_spectra` | joint read; figure; three dose points only |
| Task-agnostic premise ("dense") | premise | in1k transfer B-100 80.9 vs OK-AI DINO-B-100 78.3, iBOT-B-100 80.7 (not paired) | paired transfer on the IN-100 zoo (see 5.1) |
| Competitive standalone at IN-1k | landed B-100; rest pending | B-100 74.2/64.3, transfer 80.9, seg 28.1; S-100 66.3/54.5 (V=4), 68.5/57.7 (V=10); L-100 72.8/64.4 | S-100/400, B-400, L-400/Llr pending; L-100 below B-100 on linear; seg at 100 ep vs field at 300-400 |
| Accessibility never hurts | AGREED (C3) | delta R2 in [-.001, +.209] | appendix only; C3b null cuts against it as motivation |

## 4. Risks and overclaims

- "Rich = sparse/low-rank" in general. Only supported in specific supervised models.
  Write: rich learning aligns h to the pretext task, which is narrow; cite Jing et al.
  for SSL. Never claim the conditioner makes learning richer (E33: the untreated twin
  drifts slightly further).
- "The important thing is thickness." A reviewer will ask for the method that sets
  thickness and the sweep that shows the band; we have neither, and C8 says the gain
  is organization. Thickness is the second axis and the shield, not the lever.
- Rank as a virtue. DINO-S has higher RankMe than ours and the placement scatter shows
  no monotone rank-to-accuracy relation across methods; E31 full-slice loses ten
  points; E32-T1 says usable size is the quiet core. Claims stay within-method and
  paired: the conditioner prevents the concentration the base objective would cause.
- Isotropy vs barrier. The conditioner's optimum is N(0, I) and reviewers will read it
  as SIGReg at h. Differentiators to state: retained h, per-image centers with the
  B + A/V correction, small-share regime, derived from the balance mechanism; E31 is
  the exhibit that whitening h is harmful.
- Theory-to-method gap. The linear result regularizes the encoder weight Gram; the
  method regularizes activation covariance on random slices with a ring buffer. Equal
  only for linear whitened inputs. Say "design principle" and keep theorems inside the
  linear model.
- Regime sign. The draft never says which regime Delta = -cI is under the
  Domine/Kunin sign conventions (encoder Gram larger than projector Gram). State it,
  cite it, show it on the toy.
- Why h is deployed. The literature offers several accounts (guillotine, feature
  reweighting, bottleneck). Do not single out invariance; say h differs on both axes.
- External comparisons. OK-AI trains on 1.43-1.45M images vs our 1.28M, uses
  modernized recipes and 2g+8l views at x2.13 less compute per epoch than ours; VISReg
  uses 4 globals; seg baselines are at 300-400 ep vs our 100. Context only, confounds
  listed once, no leaderboard sentence. Use the matched-view S cells for the views
  argument (lmc 64.5 at the Lightly frame vs their anchor 64.1; lmcse 65.7).
- Negative L scaling. L-100 linear 72.8 sits below B-100 74.2. If v6Llr100 does not
  clear P-v6Llr-2, present L as a step-starvation case in the appendix, not a main row.
- Single seed (D-108). Fine at IN-1k; the exposed numbers are the IN-100 linear deltas
  of one to two points.
- Two thicknesses. Omega = W/B (glossary), Theta (paper), A_s (absolute). One
  normalizing sentence in setup.
- CLS-only conditioning vs patch-token segmentation. One sentence in limitations, or a
  patch-token capacity readout.
- Naming and vocabulary. No method name in any draft; the glossary's `spectral` is
  official; "moment floor" is retired but appears throughout `main.tex` and the
  checklist.

## 5. Smallest set of additional work (ranked by acceptance value per GPU-hour)

1. Paired transfer on the IN-100 zoo: the VISReg Table-5 evaluator on the seven
   treated/control pairs (14 ViT-S checkpoints, 8 datasets, 10-epoch probes; about one
   GPU-day). This turns the "dense" premise into paired evidence that the retained
   capacity carries transferable information. Pre-register the direction first.
2. Toy figure for sec 3: a_i^2 vs s_i for regularized, balanced and lazy nets, plus
   1-CKA drift under the regularizer. CPU minutes. Replaces the Gram heatmaps.
3. E31 dose exhibit: spectrum flatness and IN-100 probe vs h-dose using the five
   existing arms; joint read; optional x0.25 and x2 arms (two IN-100 runs) if a curve
   is wanted. Give it a card or fold it into E24.
4. E33 AGREED wording; optional E33 on the x4-dose checkpoints if per-epoch ckpts
   exist (where would the barrier start freezing features).
5. Table hygiene, zero compute: regenerate the placement table with Ours-B/L rows;
   either produce an artifact for the LeJEPA Lightly-repro row (64.1/47.1 has no CSV)
   or drop it; quote the settled value for OK-AI DINO-B-300 (the max sits at bench
   epoch 3, a warm-up spike).
6. IN-1k landings per the calendar: B-400 with seg and transfer is the must-have; S
   rows from v6s; L by the v6Llr100 arbitration; v6L100 transfer and seg if its row is
   taken.
7. Optional: two extra seeds of the ours pair at IN-100 for an error bar on the
   headline paired delta; a patch-token rank readout for the segmentation story.

## 6. Comments on skeleton.tex (line-anchored)

- `:72` title, `:82-86` abstract: lead with Beats 1-3, then the conditioner; the
  current abstract leads with faithfulness and accessibility.
- `:97-100` projector gap and accessibility: keep the gap as Beat 1; accessibility
  becomes one clause ("projected organization stays linearly accessible from h and
  the term never hurts, appendix").
- `:102` "sufficient rank for the bridge to be meaningful": drop this link. C3b shows
  accessibility does not need capacity, so the conditioner cannot be motivated through
  it. Motivate through the dense premise and Beat 3.
- `:104` the feature-learning bullet: this is Beat 4. Add the allocation equation.
- `:106-108` and `:315-352` thickness: move up into sec 2 as the second axis, with the
  two-sided theorem stated compactly (as in `main_lean.tex:354-386`) and the E32
  quiet-core reading; no separate experimental section.
- `:131-135` related work on feature-learning regimes: label supervised results as
  supervised; cite the SSL collapse literature for SSL; do not write "rich = sparse".
- `:196-204` sec 4.1.1 bullets: "task agnostic reps should be dense" and "rich learning
  does not impose sparse/dense directly" are right. Replace "dense" with "broad on the
  stable axis", and replace the second bullet with the precise statement: the balance
  is conserved, so the objective never sets h's spectrum, and the small-init solution
  allocates a_i^2 = s_i.
- `:221-234` the K(mu, C) equation: write it as the KL to N(0, I), state two-sidedness
  (it also caps variance), state that the isotropic optimum is not the target and the
  operating dose is a few percent of trunk pull.
- `:252-274` image vs augmentation spread: keep; it is why the conditioner reads
  centers; the "applying to B x V views competes with invariance" bullet is the
  pooled-vs-view-mean lesson and earns one sentence.
- `:280-309` dual-space objective: add that the z-tap term is a function of the
  end-to-end map while the h-tap term is what pins the balance; add the dosing rule.
- `:357-404` experiment figures: keep `fig_discrepancy`, `C2_capacity`,
  `fig_treatment_arrows`; move `in100_two_space_audit`, `C3b_accessibility_profile`,
  `fig_org_vs_sensitivity`, `fig_treatment_main` to the appendix; add the e33 panel,
  the E31 dose panel, the E32 quiet-core panel and the IN-1k tables, none of which the
  skeleton places.
- `:408-424` standalone SSL: add the confound paragraph and the matched-view cells;
  strike "robustness" (the OOD table is empty).
- `:429-443` conclusion: mirror the five beats.
- Missing entirely: a method name; a limitations paragraph; the seed statement; the
  CLS-only conditioning caveat; the (b, theta) object definition.

## 7. Housekeeping that touches the paper (from the 09-01 inventory)

- `placement_table.csv` has 14 rows and no `e27v6b100` / `e27v6L100`; the structure
  metrics have no Ours-B/L point while `tab:in1k-main` does.
- The LeJEPA Lightly-repro row (64.1 / 47.1) is hard-coded as a cite tuple in
  `experiments/paper_exhibits.py`; no `in1k.lejepa.s0.lightly.*` file exists.
- `in1k.pub.okdinob300.ext.bench.csv` peaks at bench epoch 3 (75.2); the settled tail
  is 74.6. The max convention propagates the spike into `tab:in1k-main`.
- Three cards are RAW despite landed numbers: E29 (no takeaway section), E30
  (predictions owed), E33 (section empty). E22 has no takeaway section.
- Retired vocabulary ("moment floor", `h_lamb=0`) lives in `main.tex` and the checklist
  prose; run-ids are exempt, prose is not.
- Calendar for the tables: v6Llr100 ~09-04 (arbitration), v6s100 ~09-05, v6b400
  ~09-09, v6L400 ~09-11, sbe400 ~09-12, v6s400 ~09-13, v6Llr400 ~09-17 (bench ~09-19;
  ICLR 09-24).

## 8. Decisions this memo needs from Berker

1. Accessibility to the appendix (a change from the skeleton's sec 3.2).
2. Thickness as the second axis in sec 2 with the compact theorem, not a section.
3. The paired IN-100 transfer run (item 5.1): pre-registration and launch.
4. The E31 exhibit: card or fold-in, and the joint read.
5. A method name for the paper (`spectral` is the glossary's word).
6. The L-100 row call, pending the v6Llr100 arbitration.
