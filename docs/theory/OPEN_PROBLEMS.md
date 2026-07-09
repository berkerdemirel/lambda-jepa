# OPEN PROBLEMS — the [O] ledger

> Derived from docs/report/ssl-projector-gap-report.html (compiled 2026-07-01, quote-verified there). Tags: [E] established · [P] plausible interpretation · [O] open question.

Every [O]-tagged gap named in the report, each with: the problem, why it matters, and which of our
experiments touches it (E1–E11 per docs/ROADMAP.md). All entries are [O] — no dedicated literature
found as of the report's compile date.

## From §2.6 — the head/projector gap

### OP-1 · A quantitative transfer law
- **Problem:** nobody has measured how much of any z-space property (invariance level,
  decorrelation, uniformity, rank) survives at h as a function of head depth/width/nonlinearity.
  All existing evidence is qualitative (VICReg D.8, RCDM, CASSLE probes).
- **Why it matters:** it is the report's core missing quantity; without it, every "the head
  buffers X" claim stays directional.
- **Ours:** E1 (transfer ratios τ per metric×method), E2 (decay profiles along the head), E10
  (the causal version — "the partial-transfer law", §2.6's top open problem).

### OP-2 · Principled readout-layer selection
- **Problem:** Guillotine shows the optimal layer varies by task (ImageNet peaks at trunk;
  EuroSAT/CLEVR/IN-1% at the first projector layer); no a-priori criterion exists. RankMe/α-ReQ
  rank whole models, not layers.
- **Why it matters:** "which layer to cut" is currently folklore; a label-free rule would be
  immediately practical.
- **Ours:** E2 — fit accuracy-vs-layer from metrics-vs-layer (leave-one-method-out); a label-free
  layer-selection rule falls out if the fit is tight.

### OP-3 · Prototype-head anatomy at scale
- **Problem:** no dedicated analysis of what DINOv2/iBOT's 65k–131k prototype layers absorb, nor
  guillotine curves for masked-token heads.
- **Why it matters:** the categorical head is the loosest coupling to backbone geometry yet
  produces the best frozen features — the least understood, most successful head design.
- **Ours:** partially — E2 guillotine over DINO's head taps (MLP → bottleneck → prototypes,
  recomputed per D-005/D-008) at toy/IN-100 scale; iBOT/DINOv2-toy in the M5+ roster expansion.

### OP-4 · The nonlinearity mystery
- **Problem:** the quote "we still do not completely understand the impact of non-linear
  projections" is from **Dubois, Hashimoto & Liang, *Evaluating SSL via Risk Decomposition*,
  ICML 2023 (arXiv 2302.03068) §5.3.3** [re-attributed 2026-07-08 — previously mis-cited to
  Dubois '21/'22; the founding report's citation needs the same fix], where nonlinear heads add
  no effective-dimensionality benefit over linear ones; yet SimCLR's +3% nonlinear-over-linear
  is robust; Xue et al. explain part of it in stylized models only. Related [verified]: Dubois
  '22 proves the *need* for heads theoretically, and the 2023 paper failed to confirm '22's
  asymmetric-head gains — the theory and the measurements disagree within one group's own line.
  None of the five §2.5 theory families predicts this cleanly.
- **Why it matters:** it is the one datum no buffer/rank/bottleneck account unifies — a wedge into
  what the head actually does.
- **Ours:** E10 projector-depth sweeps (SimCLR depth 0/1/2/3; LeJEPA depth 0–3) with the full
  metric battery, so the linear-vs-nonlinear difference is measured in desiderata, not accuracy
  alone.

### OP-5 · Head-free at SOTA scale
- **Problem:** DirectPred/DirectCLR replace or shrink the head only in (near-)linear regimes and
  trail full nonlinear heads (62.7 vs 66.5; 72.4 vs 72.5). No surveyed method — including LeJEPA,
  which keeps a wide 3-layer MLP projector — trains head-free at scale without loss.
- **Why it matters:** "is the nonlinear projector eliminable at SOTA scale?" is the decisive
  missing experiment for the whole gap literature.
- **Ours:** E10's depth-0 and SIGReg-on-backbone arms answer it at IN-100 scale; SOTA scale sits
  beyond our data ladder (rung-4 gate G-M4) and is flagged, not claimed.

## From §3 — metrics

### OP-6 · The h↔z rank map: is the monotonicity method-dependent?
- **Problem:** RankMe (on z) works because "performance on embeddings and representations scales
  almost monotonically" — an empirical regularity, not a theorem; α-ReQ is its backbone-side
  complement. No paper bridges the two spaces formally.
- **Why it matters:** rank-based label-free model selection silently assumes the bridge.
- **Ours:** E1 computes RankMe/α at both spaces (the h↔z rank map); E3 tests whether monotonicity
  holds within but not across method families.

### OP-7 · k-NN consistency between spaces (novel diagnostic)
- **Problem:** does x's neighborhood in h match its neighborhood in z (Jaccard@k of neighbor sets;
  label agreement)? A diagnostic composition with no existing literature.
- **Why it matters:** low overlap + higher h-probe accuracy = the head actively destroys semantic
  neighborhoods — a direct, geometry-level picture of what the head reorders.
- **Ours:** E1 battery (kNN-consistency metric).

### OP-8 · Neighborhood purity under distribution shift
- **Problem:** whether z's "cleaner" invariant geometry is actually more robust than h's richer
  geometry under shift (k-NN on IN-R/IN-A/ObjectNet queries against an IN-1k gallery, per space).
  The buffer theory predicts h wins on-distribution; the ordering under shift is open.
- **Why it matters:** decides whether the head's filtering is robustness-relevant or only
  loss-relevant.
- **Ours:** E1 (shift repeat on COCO val) + E3 (z-space metrics predicted to decouple under shift;
  IN-R/A, ObjectNet targets in the §5.0 robustness suite).

## From §4 — failure modes (incl. the §4.5 named absences: "all searched, none found")

### OP-9 · SSL group-robustness audit
- **Problem:** no canonical group-robustness audit of SSL encoders exists
  (Waterbirds/CelebA-blond, worst-group accuracy).
- **Why it matters:** background/context shortcuts are documented (§4.1) but never measured in the
  standard group-robustness harness; a named paper-sized hole.
- **Ours:** E6 (its two-space twist — is z more group-robust than h for view-based methods? — is
  itself flagged [O] in the report).

### OP-10 · Controlled IN-C/R/A comparison across families
- **Problem:** no controlled corruption/rendition/adversarial-example comparison across SSL
  families under matched backbone + probe.
- **Why it matters:** existing robustness claims ride on mismatched architectures and probes —
  protocol masquerading as representation quality.
- **Ours:** the §5.0 fixed frame carries IN-C/R/A/ObjectNet/IN-Sketch/Stylized-IN/ImageNet-W for
  every model under fixed probes; consumed by E3 and E11.

### OP-11 · Shape-bias psychophysics for MIM/JEPA
- **Problem:** the Geirhos cue-conflict literature covers supervised + contrastive-era SSL only;
  MIM/JEPA models are absent.
- **Why it matters:** Park et al.'s frequency analysis predicts MIM texture reliance — untested at
  the behavioral level; family-level inductive-bias fingerprints are missing rows.
- **Ours:** E7 (completes the matrix, and measures shape bias at h vs z — the first two-space bias
  measurement).

### OP-12 · iBOT-specific failure analyses
- **Problem:** iBOT has no dedicated failure literature at all — "it is the least-audited strong
  model" (Table 4).
- **Why it matters:** best-in-class low-shot (81.0 IN-1%) with zero autopsy.
- **Ours:** not in the core-7; scheduled with the M5+ roster expansion (then E5–E9 rerun over it).

### OP-13 · SSL memorization beyond déjà vu
- **Problem:** déjà-vu memorization (Meehan et al.) is the only SSL memorization result; it
  "cannot be detected by conventional techniques for evaluating representation quality"; nothing
  beyond it.
- **Why it matters:** memorization is invisible to every standard eval in the field's toolkit.
- **Ours:** E5 is déjà-vu-adjacent (object-absent crop → class/source recoverability) "without the
  privacy framing"; a dedicated memorization study is out of current scope.

### OP-14 · Object-absent local views in multi-crop (§4.2)
- **Problem:** DINO's local 96px crops train local-to-global correspondence; on scene data local
  views may contain a different object than the global view. Object-absent local views remain
  unstudied (first task-conflict analysis, De Plaen Feb 2026, does not cover them).
- **Why it matters:** it is the crop-shortcut mechanism at its purest — the loss is satisfied while
  the manufactured supervision is wrong.
- **Ours:** E5 (COCO-crops bank: object-present / object-absent / pure-background / IoU-logged RRC
  pairs).

### OP-15 · Prototype pathologies (§4.2)
- **Problem:** dead/duplicated prototypes, long-tail prototype-class mismatch, what 131k prototypes
  encode at DINOv2 scale — no dedicated failure study found.
- **Why it matters:** the prototype layer is the loss's actual interface in the strongest frozen
  models; its pathologies are unaudited.
- **Ours:** not directly scheduled; nearest hooks are E2's DINO head taps and the M5+ roster
  (SwAV/iBOT/DINOv2-toy).

### OP-16 · Content multimodality of masked prediction (§4.3)
- **Problem:** StoP addresses *location* ambiguity only; nobody quantifies content multimodality
  (how many completions a mask admits) or links that entropy to representation damage.
- **Why it matters:** [P] latent-space mode-averaging would produce blurred *semantics* (I-JEPA)
  vs visible blurred pixels (MAE) — plausibly worse and definitely less visible.
- **Ours:** E9 (the first content-ambiguity measurement: predictor-ensemble disagreement,
  in-filler entropy proxy, structural proxies → correlate with per-region representation quality).

## From §6 — theory

### OP-17 · LDM identifiability × the head
- **Problem:** Latent Distribution Matching's identifiability theorem (latents recovered up to
  affine maps) is stated at the loss layer; the paper does not discuss projection heads (verified
  absence). Does identifiability compose with a head — i.e., if z identifies latents up to affine
  maps and h retains strictly more information, does h identify them too, plus nuisances?
- **Why it matters:** it would formalize the report's central conjecture — "sufficiency survives,
  minimality doesn't" — turning the audit's empirical pattern into a theorem shape.
- **Ours:** our E1 (geometry) + E4 (information ledger) data are exactly what such an argument
  needs; the theory work itself is unassigned. See THEORY_MAP.md §"What our measurements would
  inform" and the problem note [LDM_HEAD_COMPOSITION.md](LDM_HEAD_COMPOSITION.md) (drafted
  2026-07-02: class-gap statement, linear-head special case, head-linearity index PROPOSAL).

## Minor [O] flags also recorded in the report (for completeness)

- Adversarial fragility's location — hypersphere-uniformity-linked fragility (Gupta et al.): does
  it live in h or only z-adjacent geometry? (§4.1; an E1/E3 audit cell.)
- Imbalance robustness vs curation dependence — SSL is *relatively* robust to imbalance (Liu) yet
  *absolutely* degraded on uncurated data (Tian; DINOv2's pipeline): compatible claims whose joint
  shape is unmeasured. (§4.2; touched by the IN-100-LT arm of the §5.0 frame.)
- View-predictability at h for non-predictive methods — do contrastive backbones support latent
  prediction anyway, making "predictive" emergent rather than distinguishing? (§3 battery; E1.)
- Pattern B of §3.2 — whether desideratum-transfer τ tracks downstream quality within families
  (would extend RankMe beyond rank). (E1 → E3.)
- No independent I-JEPA autopsy — every published critique is a method paper justifying its fix
  (§4.4); our two-space, layer-resolved audit of I-JEPA is the direct response (E1/E2 + E9).
