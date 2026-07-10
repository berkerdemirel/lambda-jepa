# TRD-π — Provenance, corrections record, and review lineage

Companion to `trd_pi_theory_framework_v1_3.md` (v2.0 draft: `trd_pi_theory_framework_v2_0_draft.md`). This document carries everything the paper does not need to argue: version history, the corrections record, the declined-changes register, name maps, and the pre-submission checklist. The framework document states claims; this document states how they got there.

---

## 1 · Lineage

**Pre-consolidation.** Five working documents → full consistency pass (found one genuine error, C6e; one simplification, the closed-form dithered rate; a set of phrase-level overclaims redlined) → one external review round → adjudication (accepted nearly all; caught the second genuine error, the VICReg minimizer-set claim; produced protocol v2.1 with the three-way split and the boundary-estimation note; rejected the reviewer's implementation of the background prior B in favor of the C1-identity reading, which became D0).

**v1.0** (2026-07-04). Consolidation of the five working documents. Survived three adversarial review rounds with two substantive corrections retained in-text as Remarks (R6e″: dimension-reducing heads can increase differential entropy; R2: VICReg's minimizer set is neither {Cov = I} nor O(K)-closed — setwise symmetry is the signed permutations), one precision fix (setwise symmetry vs per-solution stabilizer), and systematic de-overclaiming (tripwire language; evidence-not-certificates; conditional tags).

**v1.1** (2026-07-04; one external review round + editor pass).
1. The target/estimate circularity around ν̂ resolved: the symbol split into ν_train (in-loss, frozen at §5 Stage 1 from the pilot) and ν̂ (post-hoc, audit-side); protocol Rule 0 replaced by the five-stage pipeline.
2. Audit null corrected from moment-matched Gaussian to a parametric bootstrap from the fitted declared prior with per-draw re-estimation (Lilliefors-type); Gaussian null retained only at the ν = ∞ corner and as a separate Gaussianity diagnostic; sliced references by Monte Carlo (no convolution closure).
3. D0 reworded — I(X;Z) is measured against q_Z, not the external prior; companion identity added to R1.
4. G generalized from group to stochastic nuisance mechanism with the content-variable reading; "maximal invariant" scoped to the group case.
5. Battery F moved into the declared tuple; R7(a)'s finiteness restated as a property of the declaration.
6. R8(a) demoted to [PS pending appendix]; R8(d) sharpened into a directional default claim conditional on (a); simple-vs-composite audit semantics added (R8f); R3 takeaway (1) re-worded to match the demotion.
7. Corners table: idealization reading rule; DINO/SwAV row added; BYOL row cites implicit whitening (Tian et al. '21); MAE marked as contrast, not corner.
8. Compression provenance added to R1 (Theis/Ballé/Minnen line); "who fights whom" equilibrium added to §4.
9. §6.0 defect-injection calibration, OP-h, and E6 added; E-cards renumbered contiguously; §6.2 promoted to the selection stage.
10. Cosmetic: fourth-cumulant tensor renamed C⁽⁴⁾; M0's 16-d cell scoped against the 64-d best configuration; sketch table retitled "a reader's map" (reversed in v1.3, see below).
11. The ideal-program / surrogate / audit separation for sufficiency made explicit (R1 three levels; D1; R7 retitled; R6 placement rule; §4): tier-1 sufficiency is a constraint of the ideal program; view alignment is its label-free surrogate in L; the battery F enters only through selection and audit.

**v1.2** (2026-07-04; second external review round).
1. Scale convention pinned (D5, §5): the declared family is unit-variance for the dithered variable at the declared σ; Stage-3 per-coordinate scales are frozen nuisance standardization reported as diagnostics, never absorbed into the null; the Stage-4 bootstrap replicates the standardization step.
2. "Who picks the basis" added to §4: the prior imposes that a basis exist (R2's symmetry break), the data select which (Darmois–Skitovich) inside the A2 regime; outside A2 no factor semantics are claimed; discovery vs imposition adjudicated by pilot-read symptoms, E6, and the seed null.
3. Self-supervision scope stated in §1: label-free in L, factor-aware in governance; pre-registered label use contrasted with silent probe-based selection in mainstream practice.
4. R8(d)'s conditionality moved inside the claim sentence; "rational default" → conditional preferred design point.
5. Head-conditioning exposure declared in R6 (spectral normalization bounds only the upper constant; σ_min(g) unconstrained); σ_min / effective rank added to the reported battery; an h-space alignment term considered and rejected — it would re-import augmentation-variant content the head exists to absorb.
6. Density-level dither hygiene sentence added to §2.
7. Notation: source latents renamed s★; K defined once as the audited ambient dimension.
8. Version lineage fixed.

**Declined after v1.2's round, with reasons on record**: gutting §0's M0/BYOL hook (the cell check is already gated in the submission list); repeating the R6(d) qualifier in E-cards; canonical-basis machinery (learnable frames, pre-whitening+Varimax before the prior); demoting the fork below its conditional status.

**v1.3** (2026-07-05; structural revision — restore ambition, keep every correction). No mathematical claim, claim class, number, assumption, or citation changed. The sentence-level edit manifest was not carried into the repo (noted 2026-07-08); this summary is the surviving record. Summary:
1. Provenance split: §10 and all version narration moved to this document; the framework no longer argues with its own history.
2. One-hedge rule: every caveat is stated once, at the claim that owns it; §8 became an index of one-liners with pointers. R8(a)'s [PS] status now has a single home (R8a itself); R3, R8(d), and §8 point to it instead of restating it.
3. Desiderata restructured as want + carrier: each D states the unqualified want, then names the R that carries it. Operationalization caveats moved to their R-sections.
4. The spine table added to §1 (D → carrier → where existing methods stand → what TRD-π adds), directly linking the desiderata to the results and to the corners map.
5. Sketch analogy restored from "a reader's map" to "the framework's conceptual heart" (reverses v1.1 item 10, cosmetic half).
6. "We impose the demand and instrument the discovery" promoted to §0 as the program's stance; §4's "who picks the basis" now closes on it; the courtroom framing ("the anticipated objection … has a precise answer") cut.
7. Defensive-rhetoric strip: "This is discipline, not concession" → label use reframed as pricing what mainstream SSL consumes silently; "a declared exposure, not an inconsistency" → "handled structurally" with the three structural answers kept verbatim.
8. §0 M0 hook trimmed to one scoping clause + pointer to R6 (single home of the 16-d vs 64-d discussion).
9. Rule 0 de-historicized (states the invariant, not the v2.1 history); §6/§7 "(new)" and "(promoted from …)" tags and E-card renumbering apparatus moved here.
10. §2 gains the closing frame: every corner is a projection of the program; the four differentiating axes numbered.
11. Contribution triple (from the consistency-pass verdict, redlined phrasing) stated in §0.
12. OP-17′ renamed OP-i.

---

## 2 · Corrections record (genuine errors, kept visible)

1. **C6(e) → R6(e″)**: "spectral normalization ⟹ H(h) ≥ H(z) − K·log L" was wrong for dimension-reducing heads. Counterexample: h = (u, v), H(v) → −∞, z = u — a dimension-reducing head can increase differential entropy. Corrected to the two-case statement with full regularity hypotheses (absolute continuity, L-Lipschitz, a.e.-injectivity, nonvanishing Jacobian, finite entropies); the dimension-reducing case certifies spread of the pushforward only, with no bound along the fibers. Retained in-text as a Remark.
2. **C2 / VICReg minimizer set**: the original claim (minimizer set = {Cov = I}, hence O(K)-closed) was wrong on both halves. The hinge only enforces per-coordinate std ≥ γ; Cov = diag(1, 4) plus a 45° rotation re-activates the covariance penalty. Corrected to: signed permutations as the setwise symmetry, with solution-dependent stabilizer gauge in equal-variance blocks. Retained in-text as a Remark.
3. **Precision fix**: setwise symmetry vs per-solution stabilizer distinguished.
4. **Systematic de-overclaiming** (consistency pass + adjudication): tripwire language; evidence-not-certificates; "+25%" scoped to the matched Fisher functional; "consistent with rank > 1, k̂ decides"; "candidate mechanism" prefix on the head-warp explanation; "most informative per unit cost."

## 3 · Name maps

- **E-cards**: v1.0 E10 → E4; v1.0 E11 → E5; E1–E3 unchanged; E6 introduced in v1.1.
- **Open problems**: OP-17′ (head-composition placement of implicit entropy estimators) → OP-i as of v1.3. OP-a…OP-h unchanged.
- **Claim labels**: pre-consolidation C1…C8 correspond to consolidated R1…R8 (C6e″ → R6e″ etc.).
- **File/label offset**: the PDF named v1_3 carried internal label v1.2; from v1.3 onward filename and internal label coincide.

## 4 · Pre-submission checklist (carried forward, unchanged in substance)

> STATUS (2026-07-10, Berker-approved corrections pass): items 1–3 DISCHARGED as recorded in
> the v2.0 draft Appendix D — R8(a) folded back as [F-verified, scoped] with the three
> amendments (Fisher-functional naming; fixed-design/second-moment scoping; kNN/kernel-only
> tier-2 classes) plus the new R8(a′) random-design lemma; BYOL cell wording corrected with
> exact cells; R7(b) and R4's signal reading proven. Items 4–6 remain open (4 = ours to run;
> 6 = thresholds locked per-card as cards launch).

1. Re-derive R8's constant against the reference ISB functional; quote their lemma forms in an appendix. This discharges the [PS] on R8(a) — and with it the conditionality of R8(d) — or forces its demotion. Decide by the math, not by additional hedging.
2. Verify the exact BYOL ablation cell (collapse-to-near-chance is robust across sources; the specific figure needs one primary-source check).
3. Prove-or-demote R7(b) (explicit modulus for Lipschitz φ) and R4's signal reading (quadratic-regime lemma).
4. Run §6's experiments plus E1-lite as the minimal empirical section — §6.0 precedes any audited claim.
5. Structure the first submission around the R1/R4–R6 measurement spine with R8 as a conditional extension (strategy adopted from review round 2) — revisit if item 1 lands as a theorem, in which case R8 can be promoted to a co-headline.
6. Pre-register audit thresholds; state E-claims as predictions.

## 5 · Standing redlines (phrasing law, from the consistency pass and adjudication)

| Never write | Write instead |
|---|---|
| role split "dissolves"/"kills" Goodhart | "makes gaming visible and reportable; a tripwire, not a guarantee" |
| "the only frame in which all desiderata are statable" | "a frame in which all are jointly statable; we know of no alternative with this property" |
| "observed EP = 24.9 ⟹ defect rank > 1" | "consistent with rank > 1; the k̂ estimator decides" |
| "the cheapest decisive experiment" | "the most informative experiment per unit cost" |
| "+25%, exact" | "+25% in the matched Fisher-information functional; transfer to the ISB constant pending re-derivation" |
| "sparsity is free at second order" (unqualified) | "…for linear probing, at matched covariance, in the complete regime, conditional on R8(a)" |
| "passed" (audit) | "worst witness found under the declared search budget" |
