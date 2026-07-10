# Spot-verification: the two load-bearing citation-sweep findings

**Claude, 2026-07-10, Berker-directed ("if both of them are just arXiv papers, do not directly
trust them, your check is valuable").** Verifies the two papers the citation sweep
(citation_sweep_dubois22.md) leaned on for its Claim-2 re-scoping and Claim-3 threat verdict.
Method: arXiv abs pages + ar5iv full text fetched 2026-07-10; sweep quotes checked against the
body; both fetches agree with the sweep's verbatim quotes.

---

## 1 · arXiv 2301.12189 — Ma, Hu, Wang, "Deciphering the Projection Head: Representation
Evaluation Self-supervised Learning"

**Status: arXiv-only since 2023-01-28 (v1, sole version) — unpublished after 3.5 years.**

Quotes VERIFIED: "the projection head, in essence, targets the uniformity objective" (§3);
"the encoder outputs (representation vectors) exhibit superiority in terms of augmentation
robustness, lower entropy, and better downstream task performance than the outputs of the
projection head" (abstract + intro).

**What they actually do:** layer-wise alignment & uniformity (InfoNCE decomposition; Fig. 4,
Table 1) for SimCLR / MoCo-V2 / SimSiam; augmentation-robustness cosine (SimCLR only, Table 2);
discrete entropy (Table 3, Fig. 5); correlation with downstream error (Table 4). Analysis scale:
**CIFAR-10/100, ResNet-18, 200 ep** (ImageNet appears in downstream evaluation only). Theory: one
proposition (uniformity ≈ KL-to-uniform ⇒ entropy increase). **Evidence for the mechanism claim
is correlational layer curves — no intervention, no ablation-based causal test.** Their
mitigation, RED, is a loss-reweighting term (w_i from representation-space similarity
percentiles) that lets representation-level similarity gradients bypass the head — a heuristic
partial loss-on-h move, +2.2–3.2 kNN points on CIFAR.

**Assessment (matches Berker's prior: "i am not remotely convinced that they actually
deciphered — otherwise they would've proposed a good mechanism/method to mitigate"):** the title
is not earned by the evidence. Fragmentary two-space measurement (4 statistics, 3 sibling
methods, CIFAR, no nulls, no isotropy/rank/Gaussianity, no matched cross-family frame),
correlational mechanism story, and a mitigation that is a reweighting heuristic rather than a
mechanism-grounded placement fix. It is the closest prior to our Claim-2 and must be cited —
AND it defines the gap we fill: battery breadth with matched nulls across method families, and
causal placement interventions (E10 pilot done; M4 grid) where they have correlations.
Incidentally, RED's partial loss-on-h is an uncontrolled cousin of our E10 arms — our grid is
the controlled version of exactly their move.

## 2 · arXiv 2604.15557 — Billa, "Predicting Where Steering Vectors Succeed"

**Status: arXiv-only, v1 2026-04-16, single author.**

Quote VERIFIED: "The probe gap Δ(ℓ)=A_mlp(ℓ)−A_lin(ℓ) quantifies how much concept information
is present at layer ℓ but not output-aligned" (§3.3). **But the construction differs from ours
more than the sweep suggested: A_lin is the LOGIT LENS — the frozen unembedding applied to
hidden states, "no training is required" — not a trained linear probe.** A_mlp is a trained
residual MLP (d→512→d). So their gap = trained-MLP minus untrained-readout: not a two-trained-
family tier gap, no V-information, no budget matching, LLM residual streams only, and neither
Xu '20 nor Dubois is cited.

**Assessment:** re-classified from the sweep's "THREATENS (raw form)" to **ADJACENT /
supporting precedent** — per Berker's ruling ("i dont think it is threatened; object could be
used before in another context (llm, speech); we are doing vision; it would be a supporting
work if we are inspired") and reinforced by the construction difference. Cite as related work
when Δ ships (alongside DCI-ES 2210.00364 and the speech tier-sensitivity pair); our Δ remains
the V-information formalization with the sign condition, trained budget-matched declared tiers,
per-space, inside the SSL audit.

## 3 · Consequences filed

- citation_sweep_dubois22.md §5 review annotations (this file is its evidence).
- DUBOIS22_VS_TRD_PI.md §9 item 5: precedent scoping + the fill-the-gap commitment.
- E11 card Deliverable 4 already carries the Δ citation obligations.
