# How the SIGReg training statistic scales with dimension K

**DRAFT for discussion (Claude, 2026-07-09, at Berker's request during the E10 A/B/C/D0
autopsy) — NOT agreed; no takeaway.** Question: when SIGReg moves from proj.out (K=16) to the
embed (512) or CLS (384), is "higher loss" just "harder constraint in higher dimension"?
Everything below is derived from OUR implementation (`sslgap/methods/lejepa.py:26-47,125`) and
Monte-Carlo-verified with that exact module at the trained batch size. Companions: E10 card
(arms + autopsy numbers), `results/diag/e10_grad_share.csv` (per-term gradient shares, job
62169172), TRD_PI_REVIEW_NOTES §strong-1 (the R4 Beta-attenuation lemma this note extends to
the loss level).

## 1 · The statistic, as coded

Per step: M=256 fresh directions u ~ unif(S^{K−1}); per view and slice, the Epps–Pulley CF
distance of the N projections against **standard** N(0,1) — no per-slice standardization, so
means/variances are constrained, not normalized away:

    T(u) = N · Σ_knots w(t) [ (Ê cos(t·u᷀ᵀz) − e^{−t²/2})² + (Ê sin)² ],   L = mean_{views,slices} T

with 17 knots on [0,3] and w(t) = (2Δt, endpoints Δt)·e^{−t²/2} — note the **2Δt convention:
the code's quadrature is 2×∫₀³**, which matters for absolute values below. Loss =
λ·L + (1−λ)·inv, λ=0.02. Training N = bs = 256 per view (V=4 views averaged).

## 2 · Exact/perturbative anatomy [derivation; MC-verified §4]

With φ(t) = e^{−t²/2} and a slice s = uᵀz with mean m, variance σ², excess kurtosis κ_s:

- **Null floor** (s truly N(0,1)): E[T] = 2∫₀³(1−e^{−t²})φ dt = **1.053**, independent of N, K,
  and slice count (the ×N cancels the CF estimator variance exactly). Same family as the audit's
  `epps_pulley` — cf. its matched-Gaussian nulls ≈ 0.52–0.58 at pca64 (audit uses plain-Δt).
- **Mean defect**: ΔT ≈ N·C₂·m², C₂ = 0.482. Random slices: E[m²] = ‖μ‖²/K → attenuates ~1/K.
- **Variance defect**: ΔT ≈ N·C₄·(σ²−1)², C₄ = 0.121 (small deviations; exact CF distance
  beyond). Typical slice variance concentrates at tr(Σ)/K = mean eigenvalue — **no K
  attenuation**: if the average variance is off, every slice reads it. Beyond-perturbative
  anchors (N=256): var×2 in all dims → T ≈ 17.5; **fully collapsed slice (σ²→0) → T =
  N·2∫(1−φ)²φ dt ≈ 103** — the statistic's collapse ceiling.
- **Shape (kurtosis-type) defect**: CF perturbation is quadratic in κ_s: ΔT ≈ N·C₈·κ_s²,
  C₈ = 0.0033. Slicing attenuates κ_s before the square:
  - factorial defect, κ in every coordinate (CLT regime): κ_s = 3κ/(K+2) → ΔT ∝ N·κ²/K².
    K=16→512 costs ×(514/18)² ≈ **815×**.
  - rank-r defect subspace: with c² = ‖P_def u‖² ~ Beta(r/2,(K−r)/2), E[κ_s²] ~ κ²·E[c⁸];
    rank-1: E[c⁸] = 105/(K(K+2)(K+4)(K+6)) → ΔT ∝ N·κ²/K⁴ — **~10⁶× weaker at 512 vs 16**.
    (TRD-π R4's Beta lemma is the linear-in-κ₄ audit version of this; the loss senses the same
    attenuation squared.)
- **Anti-CLT regime** (low effective rank / clustered cloud): slices are r-summand mixtures, no
  Gaussianization; T = O(N)·O(1) in most directions, **K-independent and large** — dominated by
  the near-zero slice variances of a degenerate cloud (→ the ≈103 ceiling above).

## 3 · What is NOT true

"Higher dimension ⇒ mechanically higher loss" is not the mechanism: for fixed smooth shape
defects the sliced statistic *falls* with K (Diaconis–Freedman; the battery's known EP
foolability). What survives at K=512 is the **moment/collapse channel** (unattenuated) while
the **shape channel dies** (1/K²–1/K⁴, and already marginal at training N — see §4). High-d
placement therefore doesn't make the Gaussianity constraint "harder-as-scored"; it makes the
score *blind to shape* and sensitive only to first/second moments and degeneracy.

## 4 · MC verification (exact SIGReg module, N=256 as trained, 20 reps, mean ± sd)

| synthetic input | K=16 | K=512 | prediction |
|---|---|---|---|
| N(0,I) (H0) | 1.02 ± 0.15 | 1.07 ± 0.08 | floor 1.053, K-indep ✓ |
| var ×1.2 all dims | 2.21 ± 0.23 | 2.17 ± 0.09 | 1.05+1.24=2.29, K-indep ✓ |
| var ×2 all dims | 17.5 ± 0.9 | 17.3 ± 0.5 | K-indep ✓ (beyond-pert.) |
| rank-1 kurt κ=3 | 1.03 ± 0.19 | 1.08 ± 0.05 | +0.006 / +2e-5 — invisible ✓ |
| factorial kurt κ=3 | 1.21 ± 0.24 | 1.06 ± 0.07 | +0.21 / +0.0003 ✓✓ |
| 10-point cluster + 0.1σ noise | 25.6 ± 4.3 | 27.5 ± 1.7 | anti-CLT, K-indep, large ✓ |

Read the fourth row twice: **at the training batch size, even at K=16 a rank-1 kurtosis defect
is invisible to the per-step statistic** (the audit sees such defects because audit N is
10k–50k, and ΔT ∝ N). Per-step shape signal at K=512 is zero to noise precision (H0 sd 0.08).

## 5 · Reading the arm values with this lens [numbers; interpretation for discussion]

- A (proj, K=16): train sigreg ≈ 3.06 vs floor 1.05 — a genuine but modest deviation; at K=16
  and N=256, visible mass must be moments + strong-shape (E01-T9's audit-N decomposition put
  87–94% of the *audit* deviation in shape at proj.out — different N, both can hold).
- B (embed, K=512): stuck ≈ 70 ≈ ⅔ of the σ²→0 ceiling (103) — consistent with its measured
  rank-11/512 embed: typical slices of a near-degenerate cloud have tiny variance. The value is
  **the collapse reading itself**, not "20× more shape violation than A". C (cls, 384): same
  regime.
- D0 (embed, no projector): descent 44.6 → 18.9 ≈ moving from the degenerate/moment regime
  toward var-mismatch levels — the (unattenuated) moment channel being optimized first, as the
  scaling predicts.
- Slice-moment columns in `e10_grad_share.csv` (slice_var_mean/min/max, |slice mean|) decompose
  each arm's value into these channels at each cadence epoch; grad-share columns show who owned
  the trunk gradient (measurement running, job 62169172).

## 6 · Implications for the rescue-arm decision (E10 B′/C′) — discussion inputs, not takeaways

1. **λ cannot repair the internal ratio.** λ multiplies the moment, cluster, and shape channels
   equally; the shape:moment sensitivity ratio falls like 1/K²–1/K⁴ regardless of λ. A λ-sweep
   can fix the gross inv-vs-sigreg balance (if B/C's failure was an instability of that
   balance — grad shares will say), but at K=384–512 and N=256, SIGReg-as-implemented is a
   **first-two-moments + anti-degeneracy regularizer, not a shape-Gaussianizer**, at any λ.
2. Pre-registerable prediction for a λ-rescued B′: it may train (no collapse, probes healthy),
   but its embed should stay shape-non-Gaussian (kurt_topeig.worst at embed ≫ matched-Gaussian
   null) even where its sigreg value is low — the constraint satisfied in its own currency,
   moments-only. If instead B′'s embed comes out shape-tame, this analysis is wrong somewhere.
3. Alternatives that DO move shape pressure at high K, if we want them as arms: per-slice
   standardization before the CF (kills the moment channel, isolates shape — changes the
   official loss), N↑ (shape signal ∝ N), slice count/importance (targeted, not uniform,
   directions — the TRD-π adversarial-bank idea), or imposing at a whitened/normalized space.
   Each is a different intervention with its own confound; none is "just λ".
4. Scale-comparability caveat for E10-D (depth sweep): sigreg values across arms/depths with
   different K are not comparable numbers even at equal "violation" — matrix cells should carry
   the floor (1.05) and, where quoted, the moment-vs-shape split rather than raw values.

## 6b · Measured on the trained arms (2026-07-09, results/diag/sigreg_ref.csv, job 62176231)

Covariance-matched-Gaussian decomposition of the training statistic, replayed under training
conditions (T_actual = floor 1.053 + moment part + shape/degeneracy residue):

| cell | K | T_actual | moment part | shape residue |
|---|---|---|---|---|
| A @proj | 16 | 3.80 | 0.44 | **2.32 (84% of deviation)** |
| Blr_best @embed | 512 | 7.07 | 6.23 | −0.22 ≈ 0 |
| D0 end @embed | 512 | 16.58 | 16.01 | −0.48 ≈ 0 |
| Dr end @embed | 512 | 71.69 | 69.69 | 0.95 ≈ 0 |

§2–§4's central claim, now measured on real trained networks: at K=16 the statistic does real
shape work (84% — independently matching E01-T9's audit-N 87–94% at proj.out); at K=512 it is a
**pure second-moment meter** — every trained embed reads exactly its Gaussian twin (residue 0 ±
MC noise). "sigreg ≈ 5–7 at the embed" therefore means: Gaussian-indistinguishable GIVEN its
covariance; the entire excess is covariance-not-I (scale/anisotropy), and shape is neither asked
nor answered. Corollary for arm success criteria: a shape-residue bar (<1) is near-automatic at
512-d — the discriminating quantity is the MOMENT part's trajectory.

## 7 · Open items

- Grad-share + slice-moment measurement (job 62169172) → fold numbers into §5.
- If adopted into the battery docs: METRICS.md note that the training-loss EP and audit EP
  differ by N and by quadrature convention (2Δt vs Δt) — values are not directly comparable.
- The ep43 kill-trigger reading in the E10 card ("B/C inv ≈ 1e-4 = satisfied") deserves a
  re-read under §5: inv collapsed *because the projector degenerated*, and sigreg's 70 was the
  degeneracy reading — one mechanism, two symptoms. PENDING grad-share confirmation.
