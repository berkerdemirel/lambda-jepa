# THEORY MAP — which space do the theorems govern?

> Derived from docs/report/ssl-projector-gap-report.html (compiled 2026-07-01, quote-verified there). Tags: [E] established · [P] plausible interpretation · [O] open question.

Transcription of report §6. Almost every formal result in SSL is a statement about the function the
loss touches. When practice inserts a trainable head g and evaluates f, the theorem binds g∘f and
says nothing direct about f. Below: the major theory lines mapped to their spaces, the claims that
weaken across the gap, the results that survive it (which share a shape), and where the two
2025–26 anchors sit.

## The map — result × formal object × space × transfers to h?

| Result | Formal object | Space | Transfers to h? |
|---|---|---|---|
| Alignment/uniformity — Wang & Isola, ICML'20 | ℓ2-normalized encoder output on S^(m−1); InfoNCE asymptotics | loss layer | No formal claim. Their own experiments probe both output and fc7 — the modern split didn't exist yet; correlation observed, nothing proven. |
| Contrastive generalization bound — Arora et al., ICML'19 | any f in hypothesis class, loss computed on f; mean classifier on f | loss layer | No — with a head, the theorem certifies g∘f; f is an arbitrary pre-image. |
| Vacuity of loss-only analyses — Saunshi et al., ICML'22 | the analysis paradigm itself | — | The enabling result: same loss, same augmentations, different function class ⇒ different downstream. The head is precisely such a function-class device. [P] |
| Spectral contrastive guarantee — HaoChen et al., NeurIPS'21 | population minimizer of spectral loss = top-k eigvecs of augmentation graph; linear probe on those features | loss layer | No — the guaranteed probe sits on the loss features; practice probes one space below. HaoChen & Ma ICLR'23 add function-class effects but keep the loss-layer object. |
| SSL = spectral embedding — Balestriero & LeCun, NeurIPS'22 | closed-form optima of VICReg/SimCLR/BT losses | loss layer | No — "the learned embedding is a Laplacian/MDS embedding" is true of z; the head absorbs the whitening/orthogonality structure of that solution. [P] |
| DirectPred dynamics — Tian et al., ICML'21 | linear predictor W_p eigenspace vs input correlation | predictor space | Not addressed — backbone quality checked only empirically. |
| Prediction-head mechanism — Wen & Li, NeurIPS'22 | the encoder beneath a trainable head (nonlinear, end-to-end) | backbone | Yes — by construction. Substitution + acceleration effects make the encoder learn all features; the head is a training device. |
| Rank differential — Zhuo et al., ICLR'23 | spectra of dual-branch outputs | branch outputs | Explicitly not connected to backbone rank (named limitation). |
| Implicit variance regularization — Halvagal et al., NeurIPS'23 | eigenmodes of closed-form linear predictors | predictor space | Not addressed. |
| Multi-view IB / sufficiency — Federici et al., ICLR'20; Tsai et al., ICLR'21 | the latent z the objective shapes | loss layer | Split verdict: sufficiency survives at h (data processing: h ⊇ info(z)); minimality does not — h is non-minimal by design. [P] |
| Minimal sufficiency is harmful — Wang et al., CVPR'22 | task-relevant info not shared between views | loss layer | The danger is maximal at z; the head is arguably why practice escapes it — h keeps the non-shared information. [P] |
| VICReg information bound — Shwartz-Ziv et al., NeurIPS'23 | the regularized embedding's statistics | expander output | No — the bound certifies entropy/decorrelation properties h is never forced to have. |
| Stepwise spectral learning — Simon et al., ICML'23 | linearized BT dynamics; loss-layer embeddings in closed form | loss layer | Not addressed (kernel-PCA picture of z). |
| MAE mask theory (U-MAE) — Zhang et al., NeurIPS'22 | pixel objective reduced to encoder-level alignment of mask-induced positive pairs | encoder (via decoder assumptions) | Yes — reverse-direction bridge: loss in pixel space, guarantee pulled back to h; predicts and fixes h-space dimensional collapse. |
| MAE identifiability — Kong et al., CVPR'23 | which latents of a hierarchical model the encoder recovers; masking ratio selects the level | encoder content | Content-level claim (what is represented), not geometry — the kind that survives heads. [P] |
| JEPA implicit bias — Littwin et al., NeurIPS'24 | deep linear self-distillation feature dynamics | encoder (linear model, no head) | Yes within its model class; nonlinear-head regime open. |
| LeJEPA optimality — Balestriero & LeCun '25 | embedding distribution minimizing worst-case downstream risk; SIGReg statistics | projector output (implementation); notation f_θ/z ambiguous in the paper | Gap present, not dissolved. SIGReg is applied to the projector; the backbone is probed. The optimality theorem characterizes the enforced (projector) distribution — its relevance to the probed backbone rests on an unstated isotropy-transfer assumption. |
| Latent Distribution Matching — Mikulasch & Zenke, ICML'26 Spotlight | F_LDM = −D_KL[R(z,z′)∥P_θ(z,z′)]: alignment to an assumed latent model + entropy; identifiability up to affine maps | loss layer (heads not discussed — verified absence) | Identifiability is a content claim and plausibly survives composition with a head on the data manifold — the most gap-robust kind of loss-layer theorem. [P] |

## Eight claims that weaken across the gap (§6.1 — all [P])

1. **"Good alignment + uniformity ⇒ good representation."** Both metrics live on the loss sphere;
   Guillotine's core observation is that the layer where invariance is enforced is not the best
   layer; Xue et al. show h contains features z lacks entirely. Loss-sphere geometry
   under-determines h.
2. **Contrastive-loss-value generalization bounds.** They bind g∘f; Saunshi et al. prove the loss
   value alone can be vacuous — the head is exactly the architectural degree of freedom their
   critique licenses.
3. **"Learned features are the augmentation graph's spectral embedding."** True of z at the
   optimum; h is a pre-image that need not carry the spectral solution's whitening/orthogonality
   structure — Guillotine's layer-wise probes show the trunk does not behave like the loss-optimal
   embedding.
4. **"Invariance ⇒ minimal sufficient representation."** By data processing, h is a superset of
   z's information — non-minimal by design (Ouyang: the head "filters out the information
   irrelevant to the contrastive objective"). Sufficiency-type conclusions survive at h;
   minimality/compression-type conclusions do not. And per Wang et al., that's fortunate.
5. **Isotropy/whitening optimality prescriptions.** Theorems of the form "the probed distribution
   should be isotropic/uniform" are enforced in a space nobody probes — including in LeJEPA, whose
   SIGReg targets isotropy at the projector output while the backbone is probed. LeJEPA's
   isotropy-optimality theorem, taken literally, binds the discarded space; its applicability to
   the backbone requires the (unproven) assumption that isotropy transfers through the projector.
6. **Dimensional-collapse diagnoses at the embedding.** Rank at z ≠ rank at h; the head buffers
   collapse pressure (Jing: no projector ⇒ collapse moves into h). Rank-based model selection
   (RankMe) rests on an empirical, not formal, h↔z monotonicity.
7. **Non-contrastive stability theorems.** DirectPred/IsoLoss/orthogonality results characterize
   predictor eigenmodes — the loss-adjacent machinery — while the object of interest is the encoder
   those dynamics shape (Wen & Li is the corrective, and its very existence marks the others'
   scope).
8. **Information-theoretic bounds on VICReg-style objectives.** The certified quantities (entropy
   floor, decorrelation) are satisfied at z exactly and at h only directionally — the bound's
   premises fail in the probed space.

## The results that survive — and their common shape (§6.2, [P])

Five kinds of result cross the gap:

1. theorems whose object is the **encoder-under-a-head** (Wen & Li);
2. **reductions** that pull a far-space loss back to encoder-level structure (U-MAE);
3. explicit **backbone-side guarantees** with the head modeled as a filter (Ouyang et al.);
4. **architectures that shrink the gap** (DirectCLR — loss on a backbone sub-vector; note LeJEPA
   does *not* belong here, as it retains a full projector);
5. **content claims** — identifiability up to affine maps (LDM; Kong et al.) — which concern *what*
   is encoded rather than *how it is arranged*, and so are stable under the head's
   reparametrization.

The pattern: **geometry-claims break at the gap; content-claims and head-aware claims survive.**
That is also exactly the E1/E4 dichotomy — measure geometry desiderata (expect dissociation) and
information content (expect survival) separately.

## Where the two 2025–26 anchors sit (§6.3)

**LeJEPA** supplies a normative target: among distributions with a scalar covariance constraint,
the isotropic Gaussian uniquely minimizes integrated squared bias of downstream probes — for linear
and nonlinear probes — so this is the distribution to aim for. But LeJEPA enforces it on a
projector, not the backbone, so the theorem currently certifies the wrong space; closing that gap
(SIGReg directly on h) is an experiment, not an established property. Its plausible empirical bet —
that a distributional constraint transfers through the projector more faithfully than pointwise
invariance — is exactly what E10's SIGReg-on-backbone arm tests. [P]

**Latent Distribution Matching** (Mikulasch & Zenke, ICML 2026 Spotlight — verified) unifies the
families at the loss layer: every surveyed objective is alignment (log-likelihood under an assumed
latent model) plus an entropy estimator, and the choice of estimator generates the family tree —
KDE → contrastive, parametric-Gaussian → VICReg, conditional-entropy-with-predictor →
BYOL/SimSiam's stop-gradient, with CPC and JEPA also mapped. Its identifiability theorem
(representations recover true latents up to affine transformation, even with nonlinear predictors)
is the strongest available answer to "what do these objectives actually promise?" — and notably,
the paper does not discuss projection heads at all. [P] Composing LDM's loss-layer identifiability
with the head question is an open theory problem this study's data would directly inform: if z
identifies latents up to affine maps and h (≈ g⁻¹-ish pre-image) retains strictly more information,
then h identifies them too, plus nuisances — formalizing "sufficiency survives, minimality
doesn't." [O]

## What our measurements would inform (OURS)

- **E1 (two-space audit)** → the isotropy-transfer question (does SIGReg's constraint reach h?)
  and, more broadly, per-method desiderata-at-h transfer ratios τ — the empirical bridge every
  loss-layer theorem above lacks.
- **E10 (SIGReg-on-h arm)** → a direct test of LeJEPA's normative claim in the probed space: if
  projector-LeJEPA beats SIGReg-on-backbone, the buffer is real even for a distributional
  constraint; if they are close, isotropy is gentle enough to impose at h and the projector is
  dispensable for it. Either answer is a finding.
- **LDM identifiability × the head** = open theory problem [O]: our sufficiency-vs-minimality
  measurements (E1 geometry battery + E4 information ledger) are the data an affine-identifiability-
  through-heads argument would need. Problem note: [LDM_HEAD_COMPOSITION.md](LDM_HEAD_COMPOSITION.md).
