# E38 — seed repeats of the IN-100 controlled pairs (three seeds per cell)

**Status: PRE-REGISTERED 2026-09-19, launched the same day at Berker's direction (verbatim: "i would
like you to repeat the in100 experiments where we add our regularization to existing models (also
without regularization should be repeated) to get 3 seeds each. the models of interest are: ours
(winner recipe i think it was d256proj), dino (it had a special regularization coefficient we did
later), vicreg, visreg, simclr, lejepa, byol. it would be nice if we include confidence bars at least
for the controlled experiment of in100." / "we are just repeating the existing experiments. you dont
need to adjust anything"). D-row: D-126 (PROPOSED).**

## What is repeated

The 14 cells behind the paper's Section "Controlled experiments on ImageNet-100" (the FAMILIES list of
`experiments/paper_exhibits.py`): six methods, each as released and once with our conditioner at its
retained representation, plus our own objective with and without the backbone term. "Ours" in the
paper is the `d256vm4` recipe (h+z) against its z-only twin `d256vm4zonly` — there is no `d256proj`
cell on record; d256vm4 is what the paper's numbers (32.8 / 157.8 / 1.894) come from. DINO's treated
arm is the D-104 winner dose `e20fwlo` (λ = .01), not the 6 %-rule arm.

| family | control (seed-0 run) | treated (seed-0 run) | h tap for the reads | bs |
|---|---|---|---|---|
| LeJEPA | `in100.lejepa.s0` | `in100.lejepa.s0.e20f` (λ .0146) | 512-d embedding | 128 |
| VICReg | `in100.vicreg.s0` | `in100.vicreg.s0.e20f` (λ 1.548, cls) | CLS | **256** |
| SimCLR | `in100.simclr.s0` | `in100.simclr.s0.e20f` (λ .098) | CLS | **256** |
| DINO | `in100.dino.s0` | `in100.dino.s0.e20fwlo` (λ .01) | CLS (student) | 128 |
| BYOL | `in100.byol.s0` | `in100.byol.s0.e20f` (λ .0205) | CLS | **256** |
| VISReg | `in100.visreg.s0` | `in100.visreg.s0.visregf` (λ .0123) | 512-d embedding | 128 |
| Ours | `in100.floorssl.s0.d256vm4zonly` | `in100.floorssl.s0.d256vm4` | CLS | 128 |

## Frame — byte-matched recipes, only the seed changes

Every new run uses the override set recovered from its seed-0 checkpoint's stored `cfg`
(`scratch/e38/dump_cfgs.py` → `scratch/e38/overrides.py`, a diff against today's composed defaults;
the sets are written out in `slurm/e38_launch.sh`). Nothing is retuned: same λ, same lane overrides
(DINO/BYOL `ema_base=.996`, DINO `local_size=96`, LeJEPA on the house-hygiene set lr 1e-3 / warmup
10 / eta_min 1e-5 / grad_clip 1.0), same batch size. Loader keys (D-057 fast defaults) are not part
of the recipe. Seeds **1 and 2** are trained; seed 0 = the existing paper cells. `seed` drives the
trunk/head init, the data order and the augmentation streams (`seed_everything` + the loader
generator + `seed_worker`).

**Discrepancy found while recovering the recipes (reported, not smoothed):** the paper's appendix
says the controlled runs use "batch size 128". The July lanes — VICReg, SimCLR and BYOL, control and
treated alike — trained at **bs = 256** (494 steps/epoch, 49,400 steps at ep100 in every one of
those six checkpoints); LeJEPA, DINO, VISReg and Ours trained at 128. Each pair is internally
matched, so the pairwise comparison is unaffected; the appendix sentence needs a per-lane batch size.
The repeat keeps each lane's own batch size (Berker: "you dont need to adjust anything").

Trainer drift since the July checkpoints (git 35c5bd90 → today): the training math on these code
paths is unchanged by the record (every method/floorssl addition since is a keyed-off optional path,
legacy byte-identical; the trainer diff is the kill-trigger bookkeeping and the share logger, off
here). The E28 `x1` rerun of the `d256vm4zonly` recipe on 2026-09-03 is the standing verbatim-rerun
check for the floorssl lane. Seed-to-seed spread will absorb any residual drift for the July lanes;
if a seed-1/2 control lands far outside the seed-0 value, that is read as drift, not as noise.

Compute: one H100 per run (`slurm/e38_seeds.sbatch`: gpu100, gpu277 excluded, driver-health gate,
3 × 8 h links, resume via `_last.pt`; an IN-100 run took 5 h 53 on one H100 on 2026-09-17), 28 runs
concurrently — inside the 31-concurrent peak Berker accepted on 2026-08-31. Gate: a 1-epoch smoke of
every cell at `seed=99` must complete before that cell's chains start. Landing per run = the paper
cells' pipeline, i.e. every input of the appendix treatment figure (`fig_treatment_appendix`, 7 families ×
11 quantities vs station — Berker 2026-09-19: "we will be recreating this figure with the seeded evals"):
extract `.extL` (h_layers [3,6,9] + 8-view orbits) → probes (`linear_raw_v2`, `knn_v1_k200`, every
station) + depth metrics (`e20f_depth_metrics.py`: RankMe, effective rank, Gaussian KL, pos/rand and
class cosines; one CSV per run under `results/diag/e38/`) + thickness rows (`e23_retro_append.py` →
W/B per station, hence Θ, a, b, Λ; appends serialized by a singleton job name). No twospace/audit
unless asked. The seeded figure = `experiments/e38_zoo_seeds.py` (mean line, shaded 95 % band, the
appendix figure's harmonization rules verbatim).

## Readers and the statistic

Per family and probe: control, treated, and the **paired** delta (treated − control at the same
seed), each as mean ± the 95 % t-interval of the mean over the landed seeds (n = 3: t = 4.303, so
the half-width is 2.48 × the sample std). Reader script `experiments/e38_seeds.py` →
`results/e38/e38_seeds.csv` (per seed), `results/e38/e38_summary.csv`, tex block
`docs/paper/blocks/e38-seed-table.tex`, figure `results/figures/e38_seed_bars.png` (bars = means,
whiskers = 95 % CI, dots = seeds). Stations: the 512-d embedding for LeJEPA and VISReg, CLS otherwise
(the paper's `H_STATION`). Seed-0 values come from the existing store names listed in the script.

## Pre-registered predictions (committed 2026-09-19 before any seed-1/2 number exists)

Seed-0 paired deltas as the reader computes them (linear / kNN-200, points, raw both sides, at the
stations above; `experiments/e38_seeds.py` on the existing cells): LeJEPA +6.6 / +10.5 · VICReg −0.2 / +4.4 ·
SimCLR +1.1 / +7.0 · DINO (λ .01) +0.6 / +3.4 · BYOL +6.0 / +13.2 · VISReg +7.6 / +8.8 · Ours (h+z vs z-only)
+1.8 / +8.4.

- **P1 (noise scale):** the seed-to-seed sample std of a single cell is ≤ 1.0 point on linear and
  ≤ 1.5 points on kNN for every cell (ViT-S/16, 100 epochs, 5,000 val images).
- **P2 (kNN sign):** the paired kNN delta is positive in all three seeds for every family; its 95 %
  interval excludes zero for LeJEPA, BYOL, SimCLR, VISReg and Ours.
- **P3 (linear):** the paired linear delta's 95 % interval excludes zero for LeJEPA, BYOL and VISReg
  (seed-0 gains of 6–8 points); for VICReg it contains zero (the "unchanged" cell of the paper);
  SimCLR, DINO and Ours are the uncertain band (seed-0 gains of 0.6–1.8 points against an expected
  paired half-width of ~1–2 points) — the interval may include zero there, and the paper's wording
  then softens to "within noise" for those cells.
- **P4 (the paper's sentence):** "kNN improves for every method, by 3 to 13 points, and linear for
  all but VICReg" survives as a statement about means; the interval form is what the table shows.
- **P5 (health):** no collapse or kill-trigger incident in any of the 28 runs.

## Discipline

Card before numbers (this file) · recipes from stored cfgs, not memory · smokes gate the chains ·
`num_classes=100` on every launch · numbers land RAW in `results/e38/`; the AGREED TAKEAWAY and any
paper edit only after the joint read with Berker.

## Launch record

- 2026-09-19 18:5x: recipes recovered (`scratch/e38/cell_cfgs.json`, `scratch/e38/overrides.json`); gpu277 re-probed BAD (cuInit 802), gpu271/274 healthy; 14 one-epoch smokes at `seed=99` submitted on gpu100/H100 (`e38s-<cell>`): lejepa 66289429 · lejepa.e20f 66289430 · vicreg 66289431 · vicreg.e20f 66289432 · simclr 66289433 · simclr.e20f 66289434 · dino 66289435 · dino.e20fwlo 66289436 · byol 66289437 · byol.e20f 66289438 · visreg 66289439 · visreg.visregf 66289440 · floorssl.d256vm4zonly 66289441 · floorssl.d256vm4 66289442. Smoke checkpoints (`in100.*.s99*`) are deleted once read.
- 2026-09-19 ~19:05: 12 of 14 smokes COMPLETED (5–10 min each, ep1 probe .039–.074, no incidents); the two LeJEPA smokes on gpu274 had no epoch line after 12 min (GPU 0 %, every loader worker pegged — the known slow node), cancelled and resubmitted with gpu274 excluded on every E38 job: lejepa 66289463 · lejepa.e20f 66289464. Smoke checkpoints of the 12 passed cells deleted (36 files). Gate lesson: a COMPLETED job is no longer a valid `afterok` target ("Job dependency problem"), so the 12 verified cells launched ungated and only the LeJEPA chains carry the smoke gate.
- 2026-09-19 ~19:08: CHAINS LAUNCHED — 28 runs, 3 × 8 h H100 links each (`e38-<run_id>`), landing queued `afterok` the third link (extract → probe / depth metrics / thickness). Job ids (`scratch/e38/launch_chains.log`):
  - `in100.vicreg.s1`: links 66289467 → 66289468 → 66289469; extract 66289470 → probe 66289471 / depth 66289472 / thickness 66289473
  - `in100.vicreg.s2`: links 66289474 → 66289475 → 66289476; extract 66289477 → probe 66289478 / depth 66289479 / thickness 66289480
  - `in100.vicreg.s1.e20f`: links 66289481 → 66289482 → 66289483; extract 66289484 → probe 66289485 / depth 66289486 / thickness 66289487
  - `in100.vicreg.s2.e20f`: links 66289488 → 66289489 → 66289490; extract 66289491 → probe 66289492 / depth 66289493 / thickness 66289494
  - `in100.simclr.s1`: links 66289495 → 66289496 → 66289497; extract 66289498 → probe 66289499 / depth 66289500 / thickness 66289501
  - `in100.simclr.s2`: links 66289502 → 66289503 → 66289504; extract 66289505 → probe 66289506 / depth 66289507 / thickness 66289508
  - `in100.simclr.s1.e20f`: links 66289509 → 66289510 → 66289511; extract 66289512 → probe 66289513 / depth 66289514 / thickness 66289515
  - `in100.simclr.s2.e20f`: links 66289516 → 66289517 → 66289518; extract 66289519 → probe 66289520 / depth 66289521 / thickness 66289522
  - `in100.dino.s1`: links 66289523 → 66289524 → 66289525; extract 66289526 → probe 66289527 / depth 66289528 / thickness 66289529
  - `in100.dino.s2`: links 66289530 → 66289531 → 66289532; extract 66289533 → probe 66289534 / depth 66289535 / thickness 66289536
  - `in100.dino.s1.e20fwlo`: links 66289537 → 66289538 → 66289539; extract 66289540 → probe 66289541 / depth 66289542 / thickness 66289543
  - `in100.dino.s2.e20fwlo`: links 66289544 → 66289545 → 66289546; extract 66289547 → probe 66289548 / depth 66289549 / thickness 66289550
  - `in100.byol.s1`: links 66289551 → 66289552 → 66289553; extract 66289554 → probe 66289555 / depth 66289556 / thickness 66289557
  - `in100.byol.s2`: links 66289558 → 66289559 → 66289560; extract 66289561 → probe 66289562 / depth 66289563 / thickness 66289564
  - `in100.byol.s1.e20f`: links 66289565 → 66289566 → 66289567; extract 66289568 → probe 66289569 / depth 66289570 / thickness 66289571
  - `in100.byol.s2.e20f`: links 66289572 → 66289573 → 66289574; extract 66289575 → probe 66289576 / depth 66289577 / thickness 66289578
  - `in100.visreg.s1`: links 66289579 → 66289580 → 66289581; extract 66289582 → probe 66289583 / depth 66289584 / thickness 66289585
  - `in100.visreg.s2`: links 66289586 → 66289587 → 66289588; extract 66289589 → probe 66289590 / depth 66289591 / thickness 66289592
  - `in100.visreg.s1.visregf`: links 66289593 → 66289594 → 66289595; extract 66289596 → probe 66289597 / depth 66289598 / thickness 66289599
  - `in100.visreg.s2.visregf`: links 66289600 → 66289601 → 66289602; extract 66289603 → probe 66289604 / depth 66289605 / thickness 66289606
  - `in100.floorssl.s1.d256vm4zonly`: links 66289607 → 66289608 → 66289609; extract 66289610 → probe 66289611 / depth 66289612 / thickness 66289613
  - `in100.floorssl.s2.d256vm4zonly`: links 66289614 → 66289615 → 66289616; extract 66289617 → probe 66289618 / depth 66289619 / thickness 66289620
  - `in100.floorssl.s1.d256vm4`: links 66289621 → 66289622 → 66289623; extract 66289624 → probe 66289625 / depth 66289626 / thickness 66289627
  - `in100.floorssl.s2.d256vm4`: links 66289628 → 66289629 → 66289630; extract 66289631 → probe 66289632 / depth 66289633 / thickness 66289634
  - `in100.lejepa.s1`: links 66289635 → 66289636 → 66289637; extract 66289638 → probe 66289639 / depth 66289640 / thickness 66289641
  - `in100.lejepa.s2`: links 66289642 → 66289643 → 66289644; extract 66289645 → probe 66289646 / depth 66289647 / thickness 66289648
  - `in100.lejepa.s1.e20f`: links 66289649 → 66289650 → 66289651; extract 66289652 → probe 66289653 / depth 66289654 / thickness 66289655
  - `in100.lejepa.s2.e20f`: links 66289656 → 66289657 → 66289658; extract 66289659 → probe 66289660 / depth 66289661 / thickness 66289662

## Numbers — RAW (landed 2026-09-20 ~09:00; `python experiments/e38_seeds.py --fig`, `e38_zoo_seeds.py`)

All 28 runs completed (28 `done` lines; DINO seed-2 pair overran the 8 h link at ep 90/89 and resumed cleanly
in link 2, online best .7144/.7088). Every landing job completed: 28 probe CSVs, 28 depth-metric CSVs
(`results/diag/e38/`), 28 × 20–44 thickness rows appended to `e23_retro_spaces.csv`. Outputs:
`results/e38/e38_seeds.csv` (per seed), `results/e38/e38_summary.csv`, tex block
`docs/paper/blocks/e38-seed-table.tex`, figures `results/figures/e38_seed_bars.png` (bars, 95 % CI whiskers,
seed dots) and `results/figures/e38_treatment_appendix_seeds.png` (the appendix figure with 95 % band;
`_std.png` = ±1 std band).

**Seed table (n = 3; mean ± 95 % t-interval half-width = 2.48 × std; stations: embedding for LeJEPA/VISReg,
CLS otherwise; Δ = paired treated − control):**

| family | linear ctrl | linear + reg | Δ linear | kNN ctrl | kNN + reg | Δ kNN |
|---|---|---|---|---|---|---|
| LeJEPA | 60.6 ± 0.7 | 66.9 ± 0.7 | **+6.3 ± 0.8** | 52.4 ± 0.7 | 62.5 ± 0.8 | **+10.1 ± 1.3** |
| VICReg | 65.1 ± 1.3 | 64.5 ± 0.6 | −0.6 ± 1.0 | 54.9 ± 0.5 | 59.5 ± 0.2 | **+4.6 ± 0.8** |
| SimCLR | 60.1 ± 1.6 | 62.1 ± 1.4 | +2.0 ± 2.0 | 49.3 ± 1.7 | 56.7 ± 1.7 | **+7.4 ± 0.9** |
| DINO (λ .01) | 69.2 ± 0.9 | 68.8 ± 1.3 | −0.3 ± 2.1 | 60.8 ± 1.6 | 63.2 ± 0.6 | +2.4 ± 2.3 |
| BYOL | 58.5 ± 0.6 | 64.3 ± 0.7 | **+5.8 ± 0.5** | 46.3 ± 0.7 | 58.5 ± 1.7 | **+12.2 ± 2.3** |
| VISReg | 57.2 ± 1.0 | 64.6 ± 0.6 | **+7.5 ± 1.0** | 50.7 ± 0.9 | 59.7 ± 0.8 | **+9.0 ± 1.7** |
| Ours (z-only → h+z) | 70.5 ± 0.7 | 71.5 ± 1.5 | +1.0 ± 1.7 | 58.0 ± 0.6 | 66.2 ± 1.0 | **+8.1 ± 1.6** |

Per-seed paired deltas (s0 / s1 / s2): LeJEPA lin +6.6/+5.9/+6.3, kNN +10.5/+9.6/+10.4 · VICReg lin
−0.2/−1.0/−0.5, kNN +4.4/+4.5/+5.0 · SimCLR lin +1.1/+2.5/+2.4, kNN +7.0/+7.6/+7.6 · DINO lin +0.6/−0.7/−0.9,
kNN +3.4/+1.8/+1.9 · BYOL lin +6.0/+5.6/+5.8, kNN +13.2/+11.3/+12.1 · VISReg lin +7.6/+7.7/+7.0, kNN
+8.8/+9.8/+8.5 · Ours lin +1.8/+0.9/+0.5, kNN +8.4/+8.5/+7.4. Seed-0 (July) controls sit inside the
seed-1/2 spread everywhere (largest gap: DINO kNN control 60.1 vs 61.2/61.2) — no drift signature.

**Pre-registered predictions scored (mechanical, not a takeaway):** P1 holds (largest single-cell std 0.7 on
either probe; every half-width ≤ 1.7 on cells, ≤ 2.3 on deltas). P2 holds: the kNN delta is positive in all
three seeds for every family; intervals exclude zero for LeJEPA, VICReg, SimCLR, BYOL, VISReg and Ours; DINO's
[+0.1, +4.7] just clears it. P3 holds as written: linear intervals exclude zero for LeJEPA, BYOL, VISReg;
VICReg contains zero; SimCLR [0.0, 4.0] touches zero, DINO [−2.4, +1.8] and Ours [−0.7, +2.7] contain it.
P4 — the paper's sentence does NOT survive verbatim: "kNN improves by 3 to 13 points" → the DINO mean is
+2.4 (its seeds +3.4/+1.8/+1.9); "linear improves for all but VICReg" → DINO's linear mean is −0.3 (two of
three seeds negative) and SimCLR/Ours are within their intervals. P5: no collapse; **12 kill-trigger lines**
(single-step grad-norm 100× the running mean, clipped at 1.0): Ours z-only s1 ×5 (ep 13–22), Ours h+z s2
×1 (ep 31), Ours z-only s2 ×5 (ep 18–34), SimCLR control s1 ×1 (ep 20); probe curves continued through every
one and the affected runs landed in family. Whether the July seed-0 logs carry the same spikes is unchecked.

**Figure note:** at n = 3 the 95 % band (2.48 × std) is visible only where the seed spread is real —
SimCLR/BYOL early-layer transmission (a, b), DINO's per-layer Θ, Ours' z-tap ranks; the probe columns'
bands are narrower than the marker size for most families. The ±1 std variant is the tighter picture.

## AGREED TAKEAWAY

*(empty by contract)*
