# Citation sweep: Dubois '22 / '23 and adjacent literature vs. our novelty claims

**AGENT-GENERATED, UNREVIEWED — 2026-07-10. Citation sweep for novelty claims; nothing here is an agreed takeaway.**

Target papers: Dubois, Hashimoto, Ermon, Liang, *Improving Self-Supervised Learning by
Characterizing Idealized Representations* (NeurIPS 2022, arXiv:2209.06235); Dubois, Hashimoto,
Liang, *Evaluating Self-Supervised Learning via Risk Decomposition* (ICML 2023, arXiv:2302.03068).

---

## §1 Method and coverage limits

**What was searched.**

- Full Semantic Scholar citation lists for both papers, with abstracts: **50 citers** of
  2209.06235 and **11 citers** of 2302.03068 indexed by S2 as of 2026-07-10 (S2's own
  `citationCount` matches — this is the full S2-indexed set, but S2 undercounts vs. Google
  Scholar; expect the true count for the NeurIPS paper to be 2–3×).
- Full S2 **citation contexts** (the sentences in which each citer cites the target) for all 61
  citers — this is how the claim-5 adoptions and the claim-4 near-extensions were found.
- ~15 web searches across the five claim angles: asymmetric projection head; encoder-vs-projector
  geometry (3 phrasings); dimensional collapse at both spaces; Gaussianity/isotropy audit
  benchmarks; V-information / usable-information probe families (4 phrasings); MLP-vs-linear probe
  gap metrics; epsilon-invariance / equivalence-relation relaxation (3 phrasings);
  compression/MDL-based representation evaluation; LeJEPA/SIGReg theory; DISSL/CISSL follow-ups.
- ~20 candidate papers fetched at abstract level; **9 full PDFs downloaded and grepped**
  (2209.06235's companion 2302.03068, 2301.12189, 2206.13378, 2210.02885, 2206.02574, 2309.02265,
  2305.10229, 2511.08544, 2604.15557).

**Coverage limits.** (i) S2 indexing lag: very recent (2026 H1) citers may be missing; keyword web
searches partially compensate. (ii) Google Scholar was reachable only via general web search, not
enumerated exhaustively. (iii) Papers whose overlapping content lives only in appendices may be
under-detected when I only had the abstract (flagged per-entry below: "abstract-level" vs
"full-text-grepped"). (iv) Non-arXiv venues (pure IEEE/ACM) sampled, not enumerated.

Quotes marked **[verbatim]** were grepped from full text; others are close paraphrases from
abstract-level fetches.

---

## §2 Findings per claim

### Claim 1 — Composition: rate/minimality against a declared prior over the embedding marginal ⊗ probe-family-tiered usable-information guarantees over the G-defined task family

**No paper composes both.** The components each have owners; the two closest compositions are
LeJEPA (prior ⊗ probe families, no rate, no measurement) and DIB (family ⊗ minimality, supervised,
no marginal prior).

| Paper | Venue/yr | arXiv | Overlapping content | Verdict |
|---|---|---|---|---|
| **LeJEPA: Provable and Scalable SSL Without the Heuristics** (Balestriero & LeCun) | preprint 2025 | 2511.08544 | Declares a prior over the embedding marginal (isotropic Gaussian) and justifies it by downstream-probe theory spanning **multiple probe families**: [verbatim] "embeddings f_θ(x) should follow an isotropic Gaussian distribution to minimize worst-case risk across downstream tasks"; theorems cover linear/ridge probes AND nonlinear probes — "Theorem 7: k-NN isotropic Gaussian Optimality", "Theorem 8: Kernel isotropic Gaussian Optimality" [verbatim headings]; intro claims coverage of "and nonlinear probes (Section 3.2), providing the first…" [verbatim fragment]. **But**: no rate/compression/minimality term anywhere (SIGReg is distribution-matching to a max-entropy target — anti-compression); no per-family usable-information accounting; a training objective, not an audit/guarantee composition. | **PARTIAL-OVERLAP** (takes the "declared prior justified across probe tiers" half; the rate⊗tiers composition and the measurement side remain open) — full-text-grepped |
| **Learning Optimal Representations with the Decodable Information Bottleneck** (Dubois, Kiela, Schwab, Vedantam) | NeurIPS 2020 | 2009.12789 | Composes a **V-family-relative minimality** with V-sufficiency ("information retention and compression from the perspective of the desired predictive family… e.g. linear classifier"). Closest formal composition of minimality+probe-family in the literature — by the same first author. **But**: supervised (labels define the task, not an augmentation group G); minimality is V-information-relative, not a rate against a declared prior over the embedding marginal; single family, not tiered. | **PARTIAL-OVERLAP** (must be cited and differentiated explicitly) — abstract-level |
| **Evaluating Representations with Readout Model Switching** | ICLR 2023 | 2302.09579 | MDL/description-length evaluation of frozen representations with a **family of readout models** and Bayesian switching; "accounts for model complexity, as well as data efficiency"; shows accuracy-based comparisons "are inconsistent when multiple readout models are used". **But**: the codelength is over labels given Z (readout complexity), not a rate on the embedding marginal against a prior; families are marginalized by switching, not tiered into per-family guarantees. | **PARTIAL-OVERLAP** on the evaluation-methodology side — abstract-level |
| Compressive Visual Representations (Lee et al.) | NeurIPS 2021 | 2109.12909 | Adds a CEB rate term to SimCLR/BYOL (rate exists in SSL training). No probe tiers, no audit. | ADJACENT |
| MCR² / coding-rate line (Yu et al.) | NeurIPS 2020 | 2006.08558 | Coding rate against a Gaussian codebook as the training principle. No probe-family accessibility guarantees. | ADJACENT |
| A Theory of Usable Information under Computational Constraints (Xu et al.) | ICLR 2020 | 2002.10689 | The tier machinery itself (monotone in V). No rate term (as already stated in our claim). | ADJACENT (component owner) |
| DCI-ES (Eastwood et al.) | ICLR 2023 | 2210.00364 | "the functional capacity required to use a representation is an important but thus-far neglected aspect of representation quality, which we quantify using explicitness or ease-of-use (E)" — probe-capacity-tiered ease-of-use via loss-capacity curves. Disentanglement evaluation; no rate-against-prior; no G-defined task family. | ADJACENT (component precedent for the tiers, see claim 3) — abstract-level |
| Matrix Information Theory for SSL | ICML 2024 | 2305.17326 | Matrix-entropy quantities as SSL analysis/losses. No probe tiers, no declared prior semantics. | ADJACENT |
| Projection Head is Secretly an Information Bottleneck | 2025 | 2503.00507 | Rate-like quantity is **I(Z₁;Z₂) between h and z** ("downstream performance improves with less mutual information between encoder and projector features"), with guarantees for pre-projector features. Not a prior over the marginal; no probe tiers. | ADJACENT |

**Not found anywhere:** a framework that (a) measures a rate of the embedding marginal against a
*declared* reference prior and (b) issues per-probe-family usable-information guarantees over the
augmentation-defined task family, in one object.

### Claim 2 — Two-space audit: same statistical/geometric desiderata measured at h AND z, matched frame, across many SSL methods

**The qualitative core is pre-empted in scattered, small-scale form; the systematic matched-frame
multi-method battery is not.** This is the claim with the most accumulated adjacent evidence.

| Paper | Venue/yr | arXiv | Overlapping content | Verdict |
|---|---|---|---|---|
| **Deciphering the Projection Head: Representation Evaluation in SSL** (Ouyang? et al.) | arXiv 2023 (no venue found) | 2301.12189 | [verbatim] "After investigating the **layer-wise alignment & uniformity**, we reveal that the projection head, in essence, targets the uniformity objective" — Figure 4 computes alignment+uniformity "based on each intermediate layer within the SSL architectures. (a) SimCLR (b) MoCo-V2 (c) SimSiam"; also [verbatim] "the encoder outputs (representation vectors) exhibit superiority in terms of **augmentation robustness, lower entropy**, and better downstream task performance than the outputs of the projection head". Proposes RED, a shortcut connection between representation and projection — a placement intervention. | **PARTIAL-OVERLAP, the single closest paper to claim 2**: 4 desiderata (alignment, uniformity, invariance/robustness, entropy) at both spaces, 3 methods, plus an intervention. Limits: CIFAR-scale, 3 same-family methods, no isotropy/rank/Gaussianity battery, no matched cross-method frame, goal is a method not an audit — full-text-grepped |
| **Toward a Geometrical Understanding of SSL Contrastive Learning** | 2022 | 2205.06926 | "the **projector, rather than the encoder, is more strongly driven to become invariant** to the augmentations… by learning to project it into a low-dimensional space" — invariance + dimensionality contrasted between encoder and projector. Contrastive-only. | **PARTIAL-OVERLAP** (invariance placement, one method family) — abstract-level |
| **Guillotine Regularization** (Bordes, Balestriero, Garrido, Bardes, Vincent) | TMLR 2023 | 2206.13378 | Known to the project as the accuracy-per-layer paper, but note it goes further: Fig. 2 trains linear probes for *latent variables* per layer and concludes [verbatim] "the **predictor is responsible for a lot of the invariance to augmentation**, and that the information is most easily retrievable before it". | **PARTIAL-OVERLAP** (invariance/usable-info placement per layer; not a statistical battery, largely SimCLR+supervised) — full-text-grepped |
| **RankMe** (Garrido, Balestriero, Najman, LeCun) | ICML 2023 | 2210.02885 | Rank measured at z, downstream measured at h, with the cross-space link made an explicit validated hypothesis: [verbatim] "(ii) **embeddings and representations performance are monotonically linked**" and [verbatim] "Performance of JE-SSL representations (encoder output) … against the embeddings (projector output) RankMe values". | **PARTIAL-OVERLAP** (single statistic bridged across the two spaces over many JE-SSL runs) — full-text-grepped |
| **AdaDim** | 2025 | 2505.12576 | Tracks dimensionality H(R) at the representation space and I(R;Z) between representation and embedding space through training; "best performing SSL models do not have the highest H(R) nor the lowest I(R;Z), but effectively arrive at a balance". Method/dynamics paper, not an audit. | **PARTIAL-OVERLAP** — abstract-level |
| Reverse Engineering Self-Supervised Learning | NeurIPS 2023 | 2305.15614 | Layer-wise clustering/alignment-to-classes analyses ("alignment increases during training and when moving deeper into the network") across "diverse models". Depth-wise, but not framed as h-vs-z desiderata placement. | ADJACENT — abstract-level |
| Understanding Dimensional Collapse in Contrastive SSL (Jing et al.) | ICLR 2022 | 2110.09348 | Spectra of representation and embedding spaces for contrastive learning; DirectCLR intervention. Single method family, single statistic. | ADJACENT (classic) |
| Projection Head is Secretly an IB | 2025 | 2503.00507 | Theory of the h→z pair; empirics are **accuracy-only** at both spaces (SimCLR, Barlow Twins; CIFAR/IN-100): "does not report rank, spectral statistics, or dimensional collapse measurements across projection configurations". | ADJACENT |
| DIET (2302.10260) / Occam's Razor for SSL (2406.10743) (Ibrahim, Balestriero et al.) | 2023/2024 | — | Articulate the *motivation*: [verbatim from context] "all existing studies have derived optimality conditions at the projector's output … which is not the output of interest since the projector is thrown away". They remove the head rather than audit both spaces. | ADJACENT (motivation pre-empted, execution different) |
| Speech SSL Benchmarking pair | Interspeech'23 / CSL'24 | 2306.00452, 2308.14456 | Multi-model matched-frame *benchmark* but probe-based (accuracy under probe families), not statistical desiderata, and speech. | ADJACENT |
| Evaluating the Representation Space of Diffusion Models via SSL Principles | 2026 | 2606.09718 | Invariance/contamination (Fisher-based ICR) across noise levels of diffusion features. Not SSL h-vs-z. | ADJACENT |

**Searched for and not found:** any 2022–2026 paper measuring a *battery* (isotropy, uniformity,
rank, invariance, Gaussianity, …) at both h and z under a matched frame across many SSL methods,
or making per-desideratum placement claims across method families. The last targeted search
(Gaussianity/isotropy audit across many SSL checkpoints, 2025–2026) returned nothing of the kind.

### Claim 3 — Cross-probe-tier gap object Δ = I_{V2}(Z→Y) − I_{V1}(Z→Y) as a representation-quality metric

**The gap object exists in accuracy form in an adjacent field; the tier idea is established as an
evaluation concern (partly credited to Dubois '22 itself). The V-information formalization inside
an SSL audit is not taken.**

| Paper | Venue/yr | arXiv | Overlapping content | Verdict |
|---|---|---|---|---|
| **Predicting Where Steering Vectors Succeed** (Billa) | preprint 2026 | 2604.15557 | [verbatim] "The **probe gap ∆(ℓ) = Amlp(ℓ) − Alin(ℓ)** quantifies how much concept information is present at layer ℓ but not output-aligned. … A large probe gap indicates nonlinear encoding at that layer". Reported per layer, per concept family, on Gemma-2-2B/Llama-3.1-8B, used to predict steerability. | **PARTIAL-OVERLAP**: exactly our Δ object, but accuracy-based (not V-information), LLM-interpretability context, purpose is steering prediction not SSL representation quality — full-text-grepped |
| **DCI-ES** (Eastwood et al.) | ICLR 2023 | 2210.00364 | Explicitness E: representation quality as a function of probe **capacity** (loss-capacity curve over a probe hierarchy). Integrates over capacities rather than differencing two named families; disentanglement field. | **PARTIAL-OVERLAP** — abstract-level |
| **Speech SSL Benchmarking: Are We Doing it Right? / A Case for Larger Probing Heads** (Zaiem et al.) | Interspeech 2023 / Comput. Speech Lang. 2024 | 2306.00452, 2308.14456 | Show SSL model *rankings change* between probe families, explicitly building on Dubois '22: [verbatim from citation context] "Dubois et al. have shown that **changing the probe family from linear to MLP leads to different optimal choices** in the hyperparameters of the SSL models and enables smaller SSL representations." | **PARTIAL-OVERLAP** (tier-sensitivity as finding; no Δ object as metric) |
| Conditional probing (Hewitt et al.) | EMNLP 2021 | 2109.09234 | V-information **differences over conditioning baselines** (same family, different inputs) — the transpose of our object. | ADJACENT |
| Pareto Probing (Pimentel et al.) | EMNLP 2020 | 2010.02180 | Reports accuracy–complexity Pareto frontiers across linear vs MLP probes; recommends reporting both, never the difference as a metric. | ADJACENT |
| MDL probing (Voita & Titov) | EMNLP 2020 | 2003.12298 | Codelength per probe architecture, compared for robustness, not differenced as a quality metric. | ADJACENT |
| Evaluating repr. with Readout Model Switching | ICLR 2023 | 2302.09579 | Marginalizes over the readout family into one codelength rather than exposing per-family gaps. | ADJACENT |
| Understanding Probe Behaviors through Variational Bounds of MI | ICASSP 2024 | 2312.10019 | Theory for linear probing vs MI bounds; no family-difference metric ("equates linear probing with fine-tuning"). | ADJACENT |

**Honest reading:** the *number* "MLP-minus-linear" is folklore and now has at least one named,
formalized instance (2604.15557). What is not taken: casting Δ as a difference of **V-informations**
(with the guarantees that inherit from Xu '20 monotonicity), tying V1/V2 to the audit's declared
probe tiers, and using it as a per-method, per-space quality axis in an SSL audit.

### Claim 4 — Approximate/agnostic extension of Dubois '22: ε-invariance under stochastic (non-equivalence-relation) augmentations with graceful degradation of the sample-optimality/worst-case-ERM machinery

**No one extends Dubois '22's machinery.** But several frameworks already handle stochastic,
non-partition augmentations quantitatively — they are the bar any "ε-extension" will be measured
against.

| Paper | Venue/yr | arXiv | Overlapping content | Verdict |
|---|---|---|---|---|
| A Fiber Criterion for Representation Identifiability in Supervised Learning | 2026 | 2606.01092 | Cites Dubois '22 for augmentation-induced equivalence relations ([verbatim context] "Theoretical analyses of SSL also formalize idealized representations through augmentation-induced equivalence relations…") and studies identifiability obstructions from predictor-preserving augmentations — but works with **exact** properties, supervised setting; no ε-relaxation. | ADJACENT (closest direct theoretical engagement found) |
| InfoNCE: Identifying the Gap Between Theory and Practice (Rusak et al.) | AISTATS 2025 | 2407.00143 | Empirically tests CL-ICA identifiability theory when assumptions are violated; engages Dubois '22 comparatively ([verbatim context] "This setting is similar to that in Dubois et al. (2022)…"; "the stronger the augmentation, the more prevalent the collapse"). Assumption-violation empirics, not ε-guarantees, and for the Zimmermann line not the ERM machinery. | ADJACENT-to-PARTIAL |
| Understanding Augmentation-based SSL via RKHS Approximation and Regression (Zhai et al.) | ICLR 2024 | 2306.00788 | Handles stochastic augmentations via an augmentation operator; "augmentation complexity" enters generalization bounds — quantitative degradation with augmentation quality, alternative machinery. | ADJACENT (alternative solution to the same problem) |
| An Augmentation-Aware Theory for Self-Supervised Contrastive Learning | 2025 | 2505.22196 | Downstream risk "bounded by the unsupervised risk, but also explicitly by a trade-off induced by data augmentation" under a "semantic label assumption" — augmentation-quality degradation terms in the bound. | ADJACENT (same problem, contrastive-specific machinery) — abstract-level |
| HaoChen et al. spectral contrastive (2106.04156) + augmentation-graph generalizations (e.g. Wang et al. 2024, cited in 2410.04959's context) | NeurIPS 2021→ | 2106.04156 | The canonical non-equivalence-relation treatment (soft augmentation graph) with downstream guarantees. Pre-dates and bypasses, rather than extends, Dubois's partition assumption. | ADJACENT (must be positioned against) |
| Joint Embedding vs Reconstruction (2505.12477) | 2025 | 2505.12477 | Cites Dubois '22 in passing only; Gaussian/linear augmentation model; "analyzes exact recovery conditions asymptotically… rather than finite epsilon-bounds on invariance violations". | ADJACENT — full-text-checked for the citation |
| On Improving the Sample Efficiency of Non-Contrastive SSL | (no id in S2) | — | Reuses Dubois '22 as a *kernel-preservation* statement ([verbatim context] "SSL aims to learn features that preserve the covariance kernel structure … [Dubois et al., 2022]") — a soft/stochastic reading of the framework, but for sample efficiency, no ε-invariance results. | ADJACENT |

**Not found:** any paper doing "Dubois '22 with ε-invariance" — i.e., keeping the probe-family
characterization + worst-case-ERM sample-optimality and degrading it gracefully when the
augmentation relation is stochastic/non-transitive.

### Claim 5 — Empirical follow-ups testing the asymmetric-projection prescription (project one branch only)

**No third-party systematic test exists. The authors' own follow-up is negative; two third-party
papers silently adopt the prescription without ablating it.**

| Paper | Venue/yr | arXiv | Overlapping content | Verdict |
|---|---|---|---|---|
| **Dubois et al., Risk Decomposition** (the companion itself) | ICML 2023 | 2302.03068 | [verbatim, Appx.] "Furthermore, we did not see any significant impact on alignment as suggested by (Gupta et al., 2022) **or gains from using one-linear projection head as suggested by (Dubois et al., 2022)**. This shows that our understanding of the impact of non-linear projection heads is still lacking." Confirms the caller's framing verbatim. | Baseline fact (authors' own non-confirmation) — full-text-grepped |
| **How Does Contrastive Learning Organize Images?** | WACV 2024 | 2305.10229 | [verbatim] "We adopt the framework and augmentation proposed in [7] and **incorporate the prediction head improvements suggested by [14]**" where [14] = Dubois et al. 2022 (verified in reference list). No ablation of the head choice; features evaluated "before the prediction head". | **PARTIAL-OVERLAP**: third-party *adoption*, zero *evaluation* — full-text-grepped |
| Exploring Inductive Biases in Contrastive Learning: A Clustering Perspective | arXiv 2023 | (no id in S2) | Same group/text: [verbatim context] "We base our framework on [1] and enhance it with [14]'s prediction head, using a 512-dimension". Adoption again, no test. | PARTIAL-OVERLAP (same nature) |
| PESTO | ISMIR 2023 | 2309.02265 | Related-work nod only: [verbatim] "Finally, [30] incorporates asymmetry to contrastive- and clustering-based representation learning" ([30] = Dubois '22). PESTO's own head is a *deterministic* linear form replacing a learnable head — a different design. | ADJACENT |
| Understanding and Improving the Role of Projection Head in SSL (Gui et al.) | NeurIPS 2023 | 2212.11491 | Reformulates the head as parametric component of InfoNCE; alternative optimization scheme. No one-branch asymmetry test found (abstract-level + search snippets). | ADJACENT |

**Not found:** any paper whose *contribution* is testing project-one-branch-only vs symmetric heads
across methods/settings. The prescription sits untested outside the authors' own negative note.

---

## §3 Near-miss list (checked, not relevant)

One line each; all were fetched or context-checked and rejected for the stated reason.

- **2304.12210 A Cookbook of SSL** — survey; cites Dubois '22 descriptively, no audit/no theory extension.
- **2210.02989 SynBench / ICML'24 "What Would Gauss Say…" (Ko et al., PMLR 235)** — synthetic-Gaussian *task-based* probing of pretrained h only; engages Dubois '22's desiderata list but no rate term, no two-space measurement.
- **2305.17326 Matrix Information Theory for SSL** — loss design via matrix entropy; no probe tiers (listed in §2.1 as adjacent, nothing more).
- **2309.17281 Information Flow in SSL** — proves losses implicitly optimize matrix MI/entropy; theoretical unification + new method, no per-space measurement.
- **2304.05369 Expand or Narrow your representation (Bordes et al.)** — backbone-width intervention for transfer; cites Dubois '22 for "larger backbone representations → better linear probe"; not a two-space audit.
- **2304.03456 Rethinking Evaluation Protocols of Visual Representations** — protocol/hyperparameter sensitivity of LP/kNN/TL; no probe-family gap object, no projector measurements.
- **2307.05610 Substance or Style** — probes embeddings for style vs content factors; single space, factor-probing.
- **2401.10474 LDReg** — local-dimensionality regularizer; cites Dubois '22 among collapse–quality links only.
- **2410.04959 Collapse-Proof Non-Contrastive SSL** — method with guarantees; cites Dubois '22 in a related-work list.
- **2407.00935 Look Ahead or Look Around?** — AR-vs-masked pretraining theory; reuses Dubois '23's *experimental designs* (few-shot protocol), not its audit machinery.
- **CREMA (2407.07110) & Foundation Models for ECG** — *apply* Dubois '23 risk decomposition to ECG models as an evaluation tool; domain application, no methodological overlap with claims 1–5.
- **2405.20456 Scaling Laws for Individual Data Points / 2407.08351 AutoBencher / AlpacaEval confusions** — cite Dubois '23 (or a different Dubois '23) for unrelated reasons.
- **2306.03440 Quantifying Variability Collapse** — neural-collapse metric; cites Dubois '22 in a design-list.
- **2303.03307 Maximum Manifold Capacity Representations** — cites Dubois '22 re: sETF minima; geometric objective, no rate/tier composition.
- **2212.04858 Implicit variance regularization / 2309.16109 Feature Normalization Prevents Collapse** — non-contrastive dynamics theory; no overlap.
- **2110.15288 Hyper-Representations; 2205.10643 speech SSL review; 2106.11054 Visual Probing; 2209.03268 Reverse Probing** — probing/eval adjacent but none report probe-tier gaps or two-space statistical audits.
- **2206.01251 CLID (expressiveness + learnability)** — kNN-on-kmeans learnability + intrinsic dimension; unsupervised eval, single space, no prior/rate.
- **2503.15484, 2510.00298, 2408.02919 (V-information applications: value profiles, image quality, dataset unit-testing)** — use PVI/V-information in other domains; none difference two families.
- **2605.08241 TinySSL** — teacher-student MCU distillation; "distilled" ≠ DISSL; no relation to the asymmetric-head prescription.
- **2412.03314, 2411.06508, 2302.10283 (equivariant SSL line)** — equivariance-vs-invariance design; no ε-degradation of Dubois-style guarantees.
- **2211.00460 Augmentation Invariant Manifold Learning** — kernel-integration statistical framework for augmentation invariance; adjacent theory, no probe families, no Dubois machinery.
- **2603.27631 Asymptotics of SSL Pre-training (two-stage M-estimation)** — asymptotic statistics of pretraining; checked title/abstract, no ε-invariance of downstream-task families.
- **2501.03469 Information-Maximized Soft Variable Discretization** — info-max discretization method (IEEE TIP); cites Dubois '22 generically.
- **2606.21590 RBF Projection Heads / 2408.14514 autoencoder-embedding heads** — projection-head architecture swaps benchmarked on accuracy; no statistical two-space audit, no one-branch asymmetry.
- **2303.00633 Info-theoretic view of VICReg (Shwartz-Ziv et al.) & 2304.09355 "To Compress or Not to Compress" review** — discuss compression⊗SSL conceptually; no declared-prior rate, no tiers, no composition theorem.

---

## §4 Bottom line per claim

- **Claim 1 (rate-against-declared-prior ⊗ probe-tiered usable info): NOVELTY STANDS, with mandatory positioning.** No composition found. Must cite and differentiate: LeJEPA (declared isotropic-Gaussian prior justified by worst-case risk theorems across linear AND nonlinear probe families — but no rate term, no measurement/audit), DIB (V-family minimality composition, supervised), Readout-Model-Switching (MDL ⊗ readout family, label-side), DCI-ES (capacity-tiered ease-of-use). The claim survives **as a composition claim only** — every component individually is owned.
- **Claim 2 (two-space statistical audit, matched frame, many methods): NEEDS RE-SCOPING, then stands.** The qualitative placement findings (uniformity lives in the head — 2301.12189; the head/predictor absorbs invariance — 2205.06926, 2206.13378 Fig. 2; z-rank ↔ h-performance — RankMe) each exist at small scale. Nobody has the matched-frame, multi-method, multi-desideratum battery (incl. isotropy/Gaussianity worst-direction stats) at both spaces with provenance control. Re-scope from "nobody measures desiderata at both spaces" to "no systematic matched-frame audit; existing two-space evidence is fragmented, 2–3 methods, 1–2 statistics, CIFAR-scale" — and cite 2301.12189 prominently or a reviewer will.
- **Claim 3 (Δ = I_{V2} − I_{V1} as quality metric): THREATENED in its raw form; stands as V-information object inside the audit.** The accuracy version of the exact gap object is published (probe gap Δ(ℓ)=Amlp−Alin, arXiv 2604.15557, LLM steering, 2026 preprint), probe-tier ranking-sensitivity is established (speech benchmarking line, explicitly crediting Dubois '22), and DCI-ES already sells probe-capacity curves as a quality axis. Claim only: the V-information formalization + G-tied tiers + per-space Δ within an SSL audit.
- **Claim 4 (ε-invariance extension of Dubois '22): NOVELTY STANDS, narrowly.** No one extends the probe-family/worst-case-ERM machinery to stochastic non-equivalence augmentations. But the augmentation-graph/RKHS lines (HaoChen '21; Zhai '23; 2505.22196) already deliver graceful-degradation bounds under stochastic augmentations by *other* machinery — the contribution must be framed as "ε-extension of the probe-family characterization and sample-optimality results specifically", and compared against those bounds.
- **Claim 5 (testing the asymmetric-head prescription): NOVELTY STANDS.** Verified verbatim that Dubois '23 itself reports no "gains from using one-linear projection head as suggested by (Dubois et al., 2022)". Since then: two third-party papers adopt the prescription without ablation (2305.10229 + sibling), PESTO nods in related work, and no paper tests it systematically. A careful test would be the first.

---

## §5 Review annotations (Berker ⊕ Claude, 2026-07-10 — after spot-verification)

The two load-bearing findings were independently verified (anchors_projhead_probegap.md; both
papers are arXiv-only):

- **Claim 2 (2301.12189):** quotes CONFIRMED; scale/depth as reported (CIFAR, 3 sibling methods,
  4 statistics, correlational mechanism evidence only; RED = a loss-reweighting heuristic, not a
  mechanism-grounded fix; unpublished since 2023). Berker's ruling: the paper's "deciphering" is
  not earned — cite it as the closest fragmentary precedent AND treat its gap (battery breadth
  with matched nulls across families + CAUSAL placement interventions) as explicitly ours to
  fill (E1 battery + E10/M4 arms). Claim-2 verdict stands as "re-scope, then stands", with the
  re-scoped wording recorded in DUBOIS22_VS_TRD_PI §9 item 5.
- **Claim 3 (2604.15557):** definition quote CONFIRMED, but A_lin is the untrained LOGIT LENS,
  not a trained linear probe — the object is trained-MLP-minus-untrained-readout, LLM-only, no
  V-information, no Xu/Dubois citations. Verdict DOWNGRADED from "THREATENED (raw form)" to
  ADJACENT/supporting precedent (Berker: different context, supporting work if we gain
  something from it). Citation obligations already on the E11 card (Deliverable 4).
