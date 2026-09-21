# E33 — Rich vs lazy: CKA drift + empirical-NTK alignment, baseline vs h-floored

## Why

Berker's agenda (2026-08-31): test whether training remains RICH — features and the
tangent kernel keep evolving — rather than LAZY (the network function moving only in
the tangent space of its init, kernel ~frozen), for the baseline vs h-floored
contrast. Goal: show the regularized model still exhibits substantial feature/kernel
evolution while retaining its higher representation rank (rank numbers come from the
existing audit battery, not recomputed here). A regularizer that bought rank by
freezing feature learning would be a hollow win; this card tests that alternative.

## Frame (anatomy stated loudly)

Single-view eval anatomy, student trunk, **h.cls** (the probed space). Two drifts,
both measured against the run's **TRUE init**:

- **Init convention**: `from_native(random_init=True, seed = the run's training
  seed)` — VERIFIED 2026-08-31 bit-exact (worst |Δ| = 0.0) against the trainer's own
  `seed_everything(seed) → build_modules()` path for all four cells. Both members of
  each pair share arch + seed s0, so each pair has ONE common true init; the in100
  job additionally cross-checks its reconstructed init features against the stored
  `d256vm4null.extL` random-init store (initcheck_cos row in the CSV).
- **CKA drift** (features): linear CKA (Kornblith 2019) between features at init and
  at epoch t, per station (L03/L06/L09 + final; both cls and gap ride along), on a
  fixed seeded val subset n=4096. Report 1−CKA over epochs × stations.
- **Empirical-NTK alignment** (kernel): K_t on the nested first 256 subset images,
  trunk params only (the g_enc trunk-module-only convention), output = h.cls via 8
  unit-norm random output projections fixed across checkpoints and cells (common
  random numbers). Report ⟨K_t,K_0⟩_F/(‖K_t‖‖K_0‖), its centered variant (Cortes
  2012), successive-checkpoint alignment, and ‖K_t‖_F. fp32 grads, bf16 per-sample
  grad storage (A100-80 for ViT-B).

Code: `sslgap/metrics/cka.py` + `sslgap/extract/ntk.py` + `experiments/
feature_drift.py`; outputs `results/diag/feature_drift.<cell>.csv` + kernels npz
under `outputs/feature_drift/` (untracked; re-derivable).

## Cells

| pair | baseline | regularized | ckpt epochs |
|---|---|---|---|
| IN-100 ViT-S (PRIMARY) | `in100.floorssl.s0.d256vm4zonly` | `in100.floorssl.s0.d256vm4` | 25/50/75/100 |

(vit_small_patch16_224 per the checkpoints' own `frame` blocks — read from the ckpt,
never the tag. The in1k lm4sbe/v6b100 pair was in the draft roster but Berker ruled
IN-100 primary and canceled the in1k cells before any launch, 2026-08-31.)

## Pre-registered predictions (committed before any numbers; smoke job 63942510)

- **P1 (both cells rich):** every cell shows substantial drift — final-station
  1−CKA(init, ep_final) well above the lazy regime (≥ .5) and NTK alignment vs init
  decaying markedly below 1 and monotonically in epoch gap.
- **P2 (the contrast; the claim at stake):** the h-floored cell's feature/kernel
  evolution is comparable to or LARGER than its baseline's at h — the floor does not
  push training toward the lazy regime. Refuted if the regularized cell's drift is a
  large factor smaller (kernel alignment staying near 1 while the baseline's decays).
- **P3 (depth):** drift grows with station depth (final ≥ L09 ≥ L06 ≥ L03), per the
  guillotine picture of where method pressure lives.
- **P4 (schedule):** most of the drift is early — the ep25 checkpoint captures the
  majority of the final 1−CKA (cosine-lr + the settle precedents); successive
  alignment ⟨K_t,K_{t−1}⟩ rises toward 1 across the ladder.

## Numbers (RAW — landed 2026-08-31, jobs 63943898/63943899; CSVs are the record)

`results/diag/feature_drift.in100.floorssl.s0.d256vm4{,zonly}.csv` + kernels in
`outputs/feature_drift/`; figure `results/figures/e33_feature_drift.png`
(script `scratch/e33_figure.py`; single-cell vm4 per Berker 2026-08-31 — the figure's
job is the rich-regime evidence for OUR cell, not the pair contrast; both cells'
numbers stay in the CSVs and the RAW tables above). Init cross-check vs the stored null store: cos min .999951
both cells. Both cells share the exact init kernel norm (2363106.6 — the identical
init + common probes, consistency check).

CKA(init, ep t) at h.cls (final | L09 | L06 | L03):

| ep | vm4 (regularized) | zonly (baseline) |
|---|---|---|
| 25 | .206 \| .266 \| .357 \| .482 | .183 \| .215 \| .306 \| .474 |
| 50 | .174 \| .236 \| .344 \| .519 | .155 \| .186 \| .246 \| .461 |
| 75 | .140 \| .189 \| .295 \| .463 | .125 \| .174 \| .240 \| .443 |
| 100 | .127 \| .179 \| .288 \| .460 | .106 \| .161 \| .224 \| .399 |

Empirical NTK at h.cls:

| ep | align_init vm4 / zonly | align_prev vm4 / zonly | ‖K‖_F vm4 / zonly |
|---|---|---|---|
| 0 | — | — | 2363107 / 2363107 |
| 25 | .563 / .531 | .563 / .531 | 8006 / 570 |
| 50 | .562 / .529 | .981 / .9993 | 8996 / 454 |
| 75 | .559 / .527 | .983 / .9992 | 14263 / 446 |
| 100 | .562 / .529 | .992 / .9998 | 21577 / 466 |

## AGREED TAKEAWAY

**E33-T1 (USER-AGREED, Berker 2026-09-02: "e33 agreed"; the point is that training
remains rich, the comparison with the untreated twin is not stressed):** *Training with the
h conditioner remains in the rich regime. At h.cls the regularized IN-100 cell ends far from
its initialization (1−CKA .87 at the final station) with the empirical tangent kernel moved
away from the init kernel (alignment .56), i.e. features and kernel keep evolving; the
untreated twin sits within a few percent on both measures. The regularizer did not buy its
capacity by freezing feature learning.* Scope: IN-100 ViT-S/16, single seed, h.cls, the
E33 instruments (linear CKA vs true init; empirical NTK alignment on the 256-image subset).
