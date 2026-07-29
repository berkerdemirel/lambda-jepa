# E23 — MLP leakage: head depth as the orbit-radius dial, and the overlap sweet spot

**Status: PRE-REGISTERED + USER-APPROVED 2026-07-29 (Berker: "lets go"; design settled over
three discussion rounds this date). D-rows: D-067 (design), D-068 (instruments). Thesis and
brief: SESSION_OPENER.md (2026-07-22→29) — Saunshi et al. 2022 orbit-overlap reading; the
field tunes leakage via the MLP-after-h without naming it.**

## Question

A shallow projector cannot satisfy the SSL objective inside the head, so invariance pressure
leaks back into h — h's orbits shrink. Residual capacity added to the head absorbs the
objective and relieves h. Is there a sweet spot of leakage — controlled orbit radius at h →
controlled *class-conditional* orbit overlap → downstream? **Depth K of an identity-init
residual MLP is the dial; everything is measured in the Λ = b/a calculus (below) plus the
E17 census instruments.**

## The calculus (fixed for this card)

Per space s ∈ {h, z}, over V-view orbit stores: orbit energy W_s = E‖s_iu − s_iv‖² (view
pairs), center energy B_s = E‖m_i − m_j‖² (image pairs, m_i = view-mean), thickness
Ω_s = W_s/B_s; radius r_rms = √(W/2) (exact: E‖s_u − s_v‖² = 2σ̄²). Transmission
a² = W_z/W_h (aug secants), b² = B_z/B_h (center secants), Λ = b/a; identity Ω_h = Ω_z·Λ²
(identity — never a prediction). **V-debias is exact and closed-form: B = B̂ − W/V**
(sample view-means carry σ̄²/V of orbit noise into center distances) — verified empirically
on the three o32 stores: B̂ drifts ~30% across V ∈ {2,8,32}, debiased B is V-invariant to
3 digits (`results/diag/e23_retro_spaces.csv`, o32 rows). o8 is sufficient forever.

Conditioner placement facts (variant-resolved; corrects an overstatement caught by Berker
2026-07-29): POOLED conditioner ⇒ per-direction budget B-var + W_z/2 ≈ d′ (orbit and
center variance compete for trace — the E21 92%-aug-paid collapse lived here); VIEW-MEAN
conditioner (vm3/vm4) conditions centers only (a 1/V-damped orbit echo). **Neither variant
floors W_z.** A residual-conditioner on {z_iv − m_i} would be the τ-band — METHOD-CANDIDATE
only, no run authorized under this card.

## Design (locked)

**Frame/lane:** toy_vits8 (imagenette, 150 ep, grad_clip 1.0) · aug=lejepa V=4 · bs=128
(pooled conditioner n = bs·V = 512, d′=128, n/d′=4 — the certified estimator regime
preserved) · z-floor POOLED, h-floor POOLED · queue off · expander hidden/out = 2048 fixed
(out-dim is itself an a-dial per the retro — held fixed to keep depth the only moving axis).

**Head:** `floorssl_res_head` — Linear(384→2048) adapter + K identity-init ResBlocks
(x + W₂ReLU([BN]W₁x), W₂ zero-init) + Linear(2048→2048) out. Adapter/out built before
blocks ⇒ same seed-0 draws at every K ⇒ **every depth starts as the same function** (the
ladder invariant; verified, incl. bn lane). K=0 = linear projector (the SimCLR linear-cell
anchor; max-leakage end). BN lives inside block branches only ⇒ the K=0 cell is lane-shared.
Taps: `z.proj.tap{k}` = block-k output, `z.proj.out` (TVMLPTaps extended; per-K within-head
Λ profile comes free).

**Axes:** K ∈ {0,1,2,3,4,6,8} × h_lamb ∈ {0, 0.65} (measure AND enforce at once — Berker's
call) × head_norm ∈ {none, bn}. bn lane runs h_lamb=0 only, plus one bn anchor at
K=2/h_lamb=.65. Main grid = 14 (BN-free) + 6 (bn, K≥1; K=0 shared) + 1 anchor = **21 cells**.

**wd protocol:** projector in its own param group (`mlp_wd`); backbone wd 5e-2 untouched.
Stage-B grid at K=2/BN-free/h_lamb=0: mlp_wd ∈ {0, 1e-3, 1e-2, 5e-2, 2e-1}; pick by
**process health** (stable grad norms, no W_z collapse, terms converged, no dead blocks) —
declared BEFORE numbers; coefficient then held fixed across all cells (bn lane inherits it;
its insensitivity there is itself P8 evidence).

**Doses:** stage-B sanity at K=2 (mlp_wd=5e-2 start): arm A = toy-v2 25/45 (toy-certified,
byol-lane provenance) vs arm B = family equal-pull 32.8/38.7 (IN-100 provenance, E21 card
§re-dose). Same health criteria; winner carries the sweep.

**Stage gates:** A (this paper trail + code) → B (dose 2 + wd 4 runs; smoke first) →
**joint review of B with Berker** → C (21 cells). Landing per cell: o8 audit extraction
(audit_v1 stack, toy pairs manifest) at ep150 → energies + census + spectrum + probes;
cadence ckpts ep38/75/113/150 kept for trajectory reads on pivotal cells.

**Instruments at landing (D-068):** `sslgap/metrics/orbit_energy.py` (W/B/Ω/a/b/Λ, debiased,
raw + per-view-L2 framings, label splits) · the E17 census revived (touch% = 2r ≥ d_intra,
overlap depth ω, **class-conditioned**: touch-same vs touch-diff, enrichment p_pos/p_neg) ·
the {a_k} transmission spectrum (least-squares J: h-residuals → z-residuals over view pairs;
gain profile over h-PCs; selectivity = spread of {a_k}) — replaces the α-grouped-projector
control, which was REJECTED (Berker: it treats h as fixed, amputates head–trunk co-training;
its flat spectrum destroys the selectivity that plausibly carries the semantic effect) ·
frozen probes (lin + kNN at h).

**Run naming:** `toy.floorssl.s0.<tag>`: stage B `e23dA`, `e23dB`, `e23w{0,1e3,1e2,2e1}`;
main grid `e23k{K}` (BN-free h0), `e23k{K}h` (h_lamb=.65), `e23k{K}b` (bn h0), `e23k2hb`
(anchor). Smoke `e23smoke`.

## Pre-registered predictions (locked before any E23 number exists)

- **P0 (premise):** measured mean-a falls monotonically with K. If a does not move, the
  depth-as-leakage-dial premise fails — the informative null.
- **P1:** inv_final falls with K (≡ W_z up to 1/d; pooled lane ⇒ budget-stabilized).
- **P2 (core coupling):** W_h and r_rms(h) rise with K; the (log a², log W_h) trajectory
  moves up-left across iso-W_z diagonals — depth shifts who pays for invariance from trunk
  to head.
- **P3 (sweet spot):** kNN@h vs K peaks as a PLATEAU EDGE (T4's threshold shape), located
  where touch-same saturates while touch-diff is still low; enrichment peaks there.
- **P4:** linear@h flatter than kNN across K (E17/E21 pattern).
- **P5′ (two regimes):** low K = scale collapse (W_h, B_h fall together, Ω_h ~flat, b
  rises); high K = selective relief (W_h recovers at stable B_h, Ω_h rises, b stable).
- **P6 (Berker's non-collapse hypothesis):** the low-K/h_lamb=0 corner shows the
  invented-dimensions signature — b↑ with effrank_h↓ (the zonly retro row is the existing
  extreme case: W_h collapsed 100×, a=5.9, b=9.5). h_lamb=.65 lane suppresses it.
- **P7 (selectivity):** the {a_k} spectrum grows more UNEQUAL with K, not merely lower in
  mean.
- **P8 (Berker's BN kill):** the bn lane kills the mechanism — head behavior insensitive to
  mlp_wd (weight scale is gauge under BN) and the depth→a→W_h law degraded or absent.
- **Kills:** house rules standing; W_z collapse in any arm; single-step grad-norm >100×
  running median = INCIDENT.

## Grounding (retro-analysis, FREE, landed 2026-07-29 — numbers RAW)

`results/diag/e23_retro_spaces.csv` (1086 rows) + `e23_retro_trans.csv` (1215 rows); job
62808982; ~45 o8/o32 stores (family, e20f zoo ctrl/arm pairs, E17 arms, lejepa cadence),
zero skips. Raw observations carried into the design: family 256-d cells a≈.49–.61,
b≈.91–1.07, Λ≈1.6–2.2; d128/d64 tail a=.38/.22 (out-dim as a-dial → held fixed here);
BW_cls(h) vm4 .497 vs d256 .359 (E21-T2 direction reproduced under this estimator); zonly =
the P6 signature on disk; within-head Λ climbs tap1→out in every 256-d cell (selectivity
built layer by layer); zoo ctrl→e20f arm raises Λ in 6/7 methods (ijepa flat); lejepa
cadence walks a .098→.072 at flat b over training. Interpretation: discussion-only.

## Launch log

- 2026-07-29: code landed (`floorssl_res_head` + ResBlock + TVMLPTaps tap rule + `mlp_wd`
  param group + config keys; legacy path byte-identical, defaults null). Ladder invariant,
  taps, arch round-trip, zero-init dynamics, legacy layout: all verified (CPU battery).
  Smoke `toy.floorssl.s0.e23smoke` (2 ep) = job 62831881.

## AGREED TAKEAWAY

(empty — filled only jointly, per the collaboration contract)
