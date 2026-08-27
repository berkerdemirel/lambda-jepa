# E27 ranked at optimization step 100,000

Read 2026-08-27 from `causal-learning-ai-ista/sslgap`. Each column ranks the E27 runs by
one quantity, read from the single logged state at wandb `_step` **= 100000 exactly** —
no averaging, no smoothing, no interpolation. 21 runs ranked. Underlying values, run ids
and configs: `results/diag/e27_step100k.csv`.

**Direction.** Rank 1 is the **smallest** value in the four term columns (↓) and the
**highest** value in `test acc` (↑). Low is not automatically good in the term columns —
a collapsed moment term is exactly the pattern the 400-epoch round's P-m-2 quench gate
watches for, so these are orderings, not scores.

**What the quantities are.** `h_moment_kl` / `moment_kl` / `inv` are the loss's three raw
term values as logged (`train/*`), **unweighted** — not the `w·g` contributions the share
logger reports. `probe loss` is `train/probe`, the cross-entropy of the co-trained linear
head on that step's training views (gradient-isolated from the trunk). `test acc` is
`test/acc`, that same head's top-1 on the full ImageNet-1k validation set, at the last
epoch boundary at or before step 100k (it is logged per epoch, not per step).

**Step 100k is not the same place in every run.** At bs 128 it is epoch 10; at bs 512 it
is epoch 40. Batch size is given per run in the legend, and it is the dominant structure
in the `test acc` column.

**Vocabulary.** `moment_kl` and `h_moment_kl` are the frozen wandb curve names; the
current terms are `cond_z` and `cond_h` (`docs/GLOSSARY.md` §1). Run-ids likewise still
spell the method `floorssl`; the method is `spectral`.

| rank | h_moment_kl ↓ | moment_kl ↓ | inv ↓ | probe loss ↓ | test acc ↑ |
|---:|---|---|---|---|---|
| 1 | `lmcb5` | `v6L400` | `lm4Ls5btl100` | `v6L400` | `lm4Ls5b` |
| 2 | `lm4Ls5b` | `lm4Ls5b` | `lmcb5` | `lm4Ls5btl100` | `v6L400` |
| 3 | `lmcs5` | `lmcs5q2` | `lm4Ls5b400` | `lm4Ls5b400` | `lm4Ls5btl100` |
| 4 | `lm4Ls5btl100` | `lm4Ls5b400` | `lm4b` | `lm4Ls5b` | `lm4Ls5btl400` |
| 5 | `lm4s5b` | `lm4Ls5btl100` | `lm4Ls5b` | `lm4Ls5btl400` | `lm4Ls5b400` |
| 6 | `lm4Ls5b400` | `lm4s5b` | `lm4Ls5btl400` | `lm4s5b` | `lm4s5b` |
| 7 | `lm4Ls5btl400` | `lm4Ls5btl400` | `lm4s5b` | `v10u` | `lmcs5q2` |
| 8 | `lmcse` | `lmcs5` | `lmc` | `lmcs5` | `lmcs5` |
| 9 | `lm4sbetl100` | `lm4sbe` | `lmcs5q` | `lmcs5q2` | `lmcb5` |
| 10 | `lm4sbetl400` | `lm4sbetl100` | `lmcs5` | `lmcb5` | `lmcs5q` |
| 11 | `lm4sbe400` | `lmcs5q` | `v10u` | `v6b100` | `lm4sbe` |
| 12 | `lm4sbe` | `lmcb5` | `lm4sbe` | `v6b400` | `v6b400` |
| 13 | `lmc` | `lm4sbe400` | `lmcse` | `lm4sbe` | `v6b100` |
| 14 | `oas` | `lm4sbetl400` | `lm4sbetl400` | `lm4sbetl100` | `lm4sbetl100` |
| 15 | `v6L400` | `v6b100` | `oas` | `lm4sbetl400` | `lm4sbe400` |
| 16 | `lmcs5q` | `v6b400` | `lm4sbe400` | `lm4sbe400` | `lm4sbetl400` |
| 17 | `lm4b` | `lm4b` | `lm4sbetl100` | `lmcs5q` | `lm4b` |
| 18 | `lmcs5q2` | `lmcse` | `lmcs5q2` | `lm4b` | `v10u` |
| 19 | `v10u` | `oas` | `v6L400` | `lmcse` | `lmcse` |
| 20 | `v6b400` | `lmc` | `v6b100` | `lmc` | `oas` |
| 21 | `v6b100` | `v10u` | `v6b400` | `oas` | `lmc` |

**Excluded, per instruction — the three runs without a complete state at step 100k.**
`v6L100` (the 100-epoch twin of L2′, launched last, only at step ~48k) · `lm4b5b` (the B
base at batch 512; killed under D-103, and its wandb history returns no rows at any step
through the API) · `lej` (the LeJEPA control at ViT-S — a different loss with no moment
terms; its z-space term is `train/sigreg` = 1.0859 at this step, which is not the same
functional as `moment_kl` and so cannot be ranked against it).

## Batch 512 only (10 runs)

Within this table step 100k is **epoch 40 for every run** — the schedule position is
matched, so the orderings are not carrying a batch-size confound.

| rank | h_moment_kl ↓ | moment_kl ↓ | inv ↓ | probe loss ↓ | test acc ↑ |
|---:|---|---|---|---|---|
| 1 | `lmcb5` | `v6L400` | `lm4Ls5btl100` | `v6L400` | `lm4Ls5b` |
| 2 | `lm4Ls5b` | `lm4Ls5b` | `lmcb5` | `lm4Ls5btl100` | `v6L400` |
| 3 | `lmcs5` | `lmcs5q2` | `lm4Ls5b400` | `lm4Ls5b400` | `lm4Ls5btl100` |
| 4 | `lm4Ls5btl100` | `lm4Ls5b400` | `lm4Ls5b` | `lm4Ls5b` | `lm4Ls5btl400` |
| 5 | `lm4s5b` | `lm4Ls5btl100` | `lm4Ls5btl400` | `lm4Ls5btl400` | `lm4Ls5b400` |
| 6 | `lm4Ls5b400` | `lm4s5b` | `lm4s5b` | `lm4s5b` | `lm4s5b` |
| 7 | `lm4Ls5btl400` | `lm4Ls5btl400` | `lmcs5q` | `lmcs5` | `lmcs5q2` |
| 8 | `v6L400` | `lmcs5` | `lmcs5` | `lmcs5q2` | `lmcs5` |
| 9 | `lmcs5q` | `lmcs5q` | `lmcs5q2` | `lmcb5` | `lmcb5` |
| 10 | `lmcs5q2` | `lmcb5` | `v6L400` | `lmcs5q` | `lmcs5q` |

Architectures present: ViT-L `lm4Ls5b`, `lm4Ls5b400`, `lm4Ls5btl100`, `lm4Ls5btl400`,
`v6L400` · ViT-B `lm4s5b` · ViT-S `lmcb5`, `lmcs5`, `lmcs5q`, `lmcs5q2`.

## Batch 128 only (11 runs)

Within this table step 100k is **epoch 10 for every run**.

| rank | h_moment_kl ↓ | moment_kl ↓ | inv ↓ | probe loss ↓ | test acc ↑ |
|---:|---|---|---|---|---|
| 1 | `lmcse` | `lm4sbe` | `lm4b` | `v10u` | `lm4sbe` |
| 2 | `lm4sbetl100` | `lm4sbetl100` | `lmc` | `v6b100` | `v6b400` |
| 3 | `lm4sbetl400` | `lm4sbe400` | `v10u` | `v6b400` | `v6b100` |
| 4 | `lm4sbe400` | `lm4sbetl400` | `lm4sbe` | `lm4sbe` | `lm4sbetl100` |
| 5 | `lm4sbe` | `v6b100` | `lmcse` | `lm4sbetl100` | `lm4sbe400` |
| 6 | `lmc` | `v6b400` | `lm4sbetl400` | `lm4sbetl400` | `lm4sbetl400` |
| 7 | `oas` | `lm4b` | `oas` | `lm4sbe400` | `lm4b` |
| 8 | `lm4b` | `lmcse` | `lm4sbe400` | `lm4b` | `v10u` |
| 9 | `v10u` | `oas` | `lm4sbetl100` | `lmcse` | `lmcse` |
| 10 | `v6b400` | `lmc` | `v6b100` | `lmc` | `oas` |
| 11 | `v6b100` | `v10u` | `v6b400` | `oas` | `lmc` |

Architectures present: ViT-B `lm4b`, `lm4sbe`, `lm4sbe400`, `lm4sbetl100`, `lm4sbetl400`,
`v6b100`, `v6b400` · ViT-S `lmc`, `lmcse`, `oas`, `v10u`.

---

## What each ranked run is

Shared unless stated: ImageNet-1k, seed 0, AdamW lr 1e-3, wd .05, 10-epoch warmup, cosine
to `eta_min` 1e-5, grad-clip 1.0, 100 epochs, and the `spectral` loss
`w_inv·inv + w_cond_z·cond_z + w_cond_h·cond_h` with per-config doses from that config's
own 2-epoch pilot. "SWA" is the D-095 twin as actually implemented after the D-098
postmortem: a grad-free EMA teacher (τ cosine .996→1) whose all-view z-mean is the
invariance anchor, not Izmailov equal-weight averaging.

**ViT-S rung**

- **`oas`** — the D-088 estimator cell: our loss on the house multicrop stack (2 globals
  @224 + 8 locals @96), all-views mean, OAS shrinkage with no queue. bs 128, no SWA.
  Doses 21.4 / 49.6 / 1.89.
- **`v10u`** — the D-088 uniform-view lane: V=10 uniform globals @224 (scale .08–1)
  instead of multicrop, ring estimator with a 3-step queue. bs 128, no SWA. The S winner
  on the epoch axis. Doses 26.9 / 129.2 / 1.679.
- **`lmc`** — the D-095 re-base and the base corner of the S 2×2: our loss on Lightly's
  exact LeJEPA view stack (2 globals @224 + 6 locals @96, 0.4-family jitter), OAS
  no-queue, bs 128, no SWA. The FLOP-matched frame against the Lightly reproduction.
  Doses 11.43 / 13.85 / 0.451.
- **`lmcse`** — `lmc` + SWA, one delta. bs 128. Doses 8.31 / 12.14 / 0.467.
- **`lmcb5`** — `lmc` + batch 512 at the same flat lr, one delta. Same doses as `lmc`.
- **`lmcs5`** — `lmc` + SWA + batch 512, the "both" corner of the S 2×2 (D-099). Doses
  3.53 / 4.46 / 0.180.
- **`lmcs5q`** — the estimator A/B: `lmcs5` with the estimator swapped to the ring with a
  3-step queue (the vm4-family estimator) at the bs512+SWA point, everything else
  identical. Doses 28.03 / 106.65 / 1.831. Killed by directive at the 400-epoch launch,
  but it has a complete state at step 100k.
- **`lmcs5q2`** — the second shot of that estimator twin, re-dosed from the winner's
  realized forces (τ from `v10u`'s ep-1 shares) rather than from its partner's. bs 512.
  Doses 61.75 / 1020 / 3.19.

**ViT-B rung** — all at the D-097 geometry, 4 globals @224 + 6 locals @96, the FLOP-match
to the VISReg-B anchor (the per-rung anchor principle: every architecture level gets a
frame- and FLOP-matched external).

- **`lm4b`** — the B base: 4g+6l, OAS no-queue, bs 128, no SWA, `h_d_slice` 256. Doses
  11.91 / 17.39 / 0.57.
- **`lm4sbe`** — `lm4b` + SWA. **The paper's Ours ViT-B/16 100-epoch row.** bs 128. Doses
  10.14 / 26.38 / 0.971.
- **`lm4s5b`** — `lm4b` + SWA + batch 512, the "both" corner of the B 2×2. Doses 5.07 /
  8.97 / 0.335. (The `s` in the tag is `swa`, not ViT-S — this cell is a ViT-B.)
- **`lm4sbe400`** — B1 of the 400-epoch round (D-103): `lm4sbe` re-run for 400 epochs,
  nothing else changed. Tests whether epoch scaling buys the VISReg-band +3…+5.
- **`lm4sbetl400`** — B3: `lm4sbe400` with `eta_min` raised to 5e-5 = lr/20, one delta.
  The learning-rate-tail arm of the saturation question.
- **`lm4sbetl100`** — the 100-epoch twin of B3, a cheap early referee for the same
  question.
- **`v6b400`** — B2′, the all-global cell: `aug=lejepa` V=6 @224 (no locals), ring
  estimator with a 3-step queue, SWA, bs 128, `h_d_slice` 256, 400 epochs, ×1.17 the
  anchor's token budget. The moment-channel hypothesis — hold the moment term instead of
  letting it quench. Doses 22.81 / 222.26 / 4.054.
- **`v6b100`** — the 100-epoch twin of B2′.

**ViT-L rung**

- **`lm4Ls5b`** — the L cell: 4g+6l, SWA, batch 512, `h_d_slice` 384. **The paper's Ours
  ViT-L/16 100-epoch row.** Doses 4.65 / 10.28 / 0.35.
- **`lm4Ls5b400`** — L1: the same cell for 400 epochs, re-dosed to 5.07 / 8.97 / 0.335.
- **`lm4Ls5btl400`** — L3: L1 with `eta_min` 5e-5 = lr/20, one delta (the L mirror of B3).
- **`lm4Ls5btl100`** — the 100-epoch twin of L3.
- **`v6L400`** — L2′, the all-global cell at L: `aug=lejepa` V=6 @224, ring estimator with
  a 3-step queue, SWA, batch 512, `h_d_slice` 384, 400 epochs. Doses 14.15 / 411.9 / 10.2.
