# TRD-π v1.3 — review notes (Claude, 2026-07-08)

**Status: discussion basis for a dedicated session — NOT locked, NOT a decision. Saved at
Berker's request (dress-rehearsal discussion). Adoption question deliberately deferred; nothing
here changes M2's launched jobs (battery upgrades below are post-hoc computable from stored
features).** Subject: `docs/theory/trd_pi_theory_framework_v1_3.md`.

## Verdict (one paragraph)

TRD-π is the mature form of the program this project has been assembling — its D0–D5 map
one-to-one onto DESIDERATA_FRAMEWORK's D1–D5 (D0/rate-vs-declared-prior added), its §7
predictions are largely our E-cards, and its §0 empirical hook *is* our M0 measurement
(lejepa-lamb002 proj.out: EP 24.9 vs 1.98, kurt-worst 3.54 vs 0.13). It adds real theorems where
we had prose. Recommended posture: adopt its **measurement layer** into M2 now (cheap, post-hoc,
listed below), treat its **constructive/training layer** as the ROADMAP Q6 synthesis to be
decided after G-M2, and do not re-badge sslgap mid-milestone — the audit program's value is that
it does not presuppose the framework it is testing.

## What is genuinely strong

1. **R4 (random-slice blindness, quantitative).** The Beta(k/2,(K−k)/2) attenuation lemma turns
   the Diaconis–Freedman critique into arithmetic with a falsifiable estimator (defect rank k̂).
   Spot-checked against our numbers: worst-direction κ₄ 3.54 at K=16, rank-1 defect ⇒ mean-slice
   κ₄ = 3.54·3/(16·18) ≈ 0.037 — matches the doc. This is the theoretical sharpening of our
   SIGReg honesty cells.
2. **R6(d/e″) name our measured configuration.** Expander singularity (dim z > dim h ⇒
   KL(q_z‖N)=+∞ — "z ~ Gaussian" unsatisfiable in principle for VICReg-class heads); and
   fiber-blindness of dimension-reducing heads = exactly proj-16-healthy / embed-rank-9.4.
3. **Pipeline discipline** (declare → select → train → estimate → audit; ν_train vs ν̂ never the
   same symbol; Lilliefors-style per-draw re-estimation in the bootstrap; "never reports
   passed"). Better statistics than anything published in SSL; self-applies our EP critique.
4. **Claim-class hygiene** ([T]/[L]/[PS]/[F]/[D]/[P]/[C]) — the culture this repo wants. The [T]s
   spot-checked pass (R1 identity; R2's VICReg signed-permutation argument incl. the diag(1,4)
   counterexample; R8(c) Fisher-info I(5)=1.25, κ₄(5)=6).

## The five risks (raise before adopting anything load-bearing)

1. **The fork hangs on a [PS].** R8(a) — linear-probe results depend on the embedding law only
   through covariance — is "pending appendix", yet R8(d)'s reversal ("finite ν is the preferred
   design point") inherits it everywhere. If the Balestriero–LeCun lemmas use Gaussianity beyond
   second moments, the framework's most distinctive recommendation deflates to a design taste.
   → **RESOLVED 2026-07-08 (verification agent, arXiv 2511.08544 v3, quotes in hand): R8(a)
   SURVIVES with three amendments.** (i) The linear-probe results (Lemmas 1–2, proofs B.1–B.2)
   are functions of the covariance spectrum alone — ridge bias −λ(XᵀX+λI)⁻¹β and OLS variance
   σ²tr((XᵀX)⁻¹); nothing beyond second moments of the embedding law enters, so any
   unit-covariance law (factorial Student-t included) earns the identical guarantee. The paper's
   own §3.1→3.2 transition concedes the split: "the distribution of features must be isotropic.
   We now move to nonlinear probing where the standard Gaussian will emerge as the unique
   optimum." (ii) Amendments: J(p) is the FISHER-INFORMATION functional ∫‖∇log p‖²p (Cramér–Rao
   equality iff Gaussian; Lemma 6/Theorem 9 in B.4), not ∫‖∇p‖²; "minimax-equivalent" softens to
   "indistinguishable under the paper's fixed-design, second-moment analysis" — spectrum
   comparisons, not a minimax theorem; a RANDOM-design analysis would re-admit heavy tails
   through fourth moments of (ZᵀZ)⁻¹; Gaussian uniqueness is exact for the leading p-dependent
   kNN-ISB term under an isotropic task prior, but only an UPPER BOUND for the kernel case;
   probe class = {radius-kNN, Nadaraya–Watson} — no MLP result exists. (iii) Bonus, verified:
   Table 1(d) confirms the 64-d projector as best (75.65 top-1, beats 128–1024 in every column);
   the official MINIMAL.md code applies BOTH losses at projector output while probing a
   backbone-side 512-d Linear embedding (the same Linear-after-trunk configuration as our
   D-003 lore); and the paper contains NO sentence acknowledging that the theorem's space
   (projector output) differs from the probed space (frozen backbone) — the project's founding
   observation, now quote-verified. Consequence for TRD-π: R8(d)'s "finite ν costs nothing at
   tier 1" holds at the population/fixed-design level; finite-sample random-design constants
   remain an honest open flank (file with OP-h).
2. **The identifiability payoff imports A2.** R3 assumes affine identifiability is *achieved* —
   imported from results whose conditions (predictive/temporal settings; Zimmermann-class
   assumptions) are not established for crop-based image SSL. The finite-ν arm's promised payoff
   (axes = factors) may simply not materialize on our data; E6 nulls would be doing their job.
3. **Audit cost.** Composite-null bootstrap with per-draw re-estimation + Monte-Carlo
   per-direction reference laws + §6.0 defect-injection calibration per (K,N,budget) ≈ building
   the battery a second time. Adopt selectively. Note the doc's own rule (§6.0 precedes any
   audited claim) would retroactively gate the honesty cells it quotes in §0 — we should hold
   ourselves to that: run §6.0 at our (K,N) before those cells go external.
4. **Its placement rule presumes what E10 tests.** Rate+marginal at h̃ is E10-arm-C-like pressure
   on the trunk. If arm C pays a real probe cost, TRD-π's differentiator (3) inherits it and §4
   needs the head back. Our E10 toy pilot is the cheapest de-risk for both programs — the
   convergence of the two designs (arm C ≈ TRD-π at the ν=∞ corner, minus the bank) is itself
   evidence the placement question is the right one. Unproven engineering: adversarial direction
   bank (OP-g cycling), λ_marg carrying anti-collapse + calibration alone, σ/normalization
   conventions for cross-method rate comparability (the same convention problem that sank
   cross-method loss comparisons).
5. **Governance nits.** `TRD_PI_PROVENANCE.md` (review lineage / corrections / declined-changes)
   is referenced but absent from `docs/theory/` — must land in-repo if adopted. Label discipline
   (F-labels at selection, priced) is right but means all claims are relative to declared
   (G, F, π, σ): the framework's force is honesty + measurement, not a solution to "what should
   be invariant" — a fit with this project's thesis, but do not oversell.

## Tiered adoption proposal (to discuss)

- **Now (M2 battery, post-hoc over stored features — no effect on training jobs):** k̂ defect-rank
  estimator (fit slice-κ₄ spectrum to the Beta prediction); ν̂ + CI on 1/ν as battery stats;
  Δ = I_V₂ − I_V₁ tier-gap protocol as E11's formal spine (mlp2_v1 is already in the E11 grid);
  head-conditioning index (= D-015); which-null discipline (= D-014); §6.0 audit-power
  calibration at our (K, N) regimes (CPU, synthetic).
- **Now (zero compute):** R8(a) verification (agent running); extend the DESIDERATA
  quote-verification pass to TRD-π's [F] imports.
- **After G-M2:** the constructive track (declared-prior training, bank, HSIC gate, Matryoshka
  rider) as the Q6 synthesis decision — informed by E10's placement frontier.

## Provenance-doc consistency check (2026-07-08 — Berker asked for a check, no discussion)

`TRD_PI_PROVENANCE.md` landed in docs/theory; checked against v1.3. **Verdict: consistent.**
Specifics:
- Every correction in the record appears in v1.3 as described: C6e→R6e″ (the H(v)→−∞
  counterexample, in-text Remark ✓); C2/VICReg minimizer set (diag(1,4)+45° counterexample,
  R2 item 3 verbatim ✓); R8(a) demotion to [PS] with single-home rule ✓; the v1.1 audit-null
  correction (moment-matched Gaussian → parametric bootstrap, Gaussian retained only at the
  ν=∞ corner) matches §5 Stage 4 ✓.
- All §5 standing redlines are respected in v1.3 (spot-checked: "we know of no alternative
  frame with this property"; "consistent with rank > 1 — k̂ decides"; "worst witness found
  under the declared search budget"; the +25% Fisher-functional scoping; "most informative
  per unit cost").
- Pleasant cross-validation both ways: (i) our battery's `null_gauss` usage on the LeJEPA
  honesty cells is exactly the simple-null ν=∞ corner case the provenance record says was
  deliberately retained; (ii) pre-submission checklist item 1 (re-derive R8(a) against the
  reference lemmas) is now substantially discharged by our verification agent — R8(a) survives
  with the amendments recorded above (Fisher-information naming for J(p); fixed-design/
  second-moment scoping; local-smoothers-only probe class). Checklist item 2 (BYOL ablation
  cell) DISCHARGED 2026-07-08: collapse-to-near-chance is robust but cell-specific —
  minus-predictor = 0.2% (Table 5(b) row 7), minus-TARGET-network with predictor kept = 0.3%
  (row 6; also τ=0 stop-grad, Table 5(a)), predictor+sg(θ) = 5.5% (Table 19); the paper never
  reports the loss value at the collapsed cells, so v1.3 §0's "while satisfying their
  objective" should read "whose objective admits collapsed solutions" — full quotes in
  docs/theory/verification/anchors_ib_mi_byol.md.
- Two loose ends, both filename-level: the provenance doc references its framework companion
  as `TRD_PI_FRAMEWORK_v1_3.md` (repo file is `trd_pi_theory_framework_v1_3.md`), and cites
  `TRD_PI_EDIT_MANIFEST_v1_2_to_v1_3.md`, which is not in the repo.
- From the identifiability verification pass (2026-07-08, verification/anchors_identifiability.md)
  — three v1.3 citation-level fixes for the author: (i) "Mikulasch et al. '26" → **Mikulasch &
  Zenke** (two authors); (ii) §2's "per-method derivations from the latent-distribution-matching
  reading" is overbroad — LDM's own text excepts real DINO ("might not directly be related to
  any entropy estimator") and never surveys Barlow Twins/SwAV/MAE, yet v1.3's corners table
  carries BT and DINO/SwAV rows under that attribution — the reading-rule hedge helps but the
  attribution should name which rows are LDM-derived vs our extension; (iii) R3's A2 temporal
  scoping clause is VERIFIED accurate (LDM theorems stated over ∏ₜ P(z_t|z_{:t}); pairwise case
  cited to prior work, not proven). Bonus quote for the project: LDM §2 calls LeJEPA
  "single-variable LDM" that "does not provide identifiability guarantees."

## Cross-references

DESIDERATA_FRAMEWORK.md §4–5 (the composed program TRD-π matures) · THEORY_MAP.md (placement of
LeJEPA/LDM anchors) · OPEN_PROBLEMS.md OP-17 → TRD-π OP-i · E10 card (arm C ≡ placement rule at
the Gaussian corner) · E01-T4 (the measured cells TRD-π's §0 quotes).
