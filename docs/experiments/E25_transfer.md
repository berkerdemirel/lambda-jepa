# E25 — ssl-transfer linear benchmark on the in1k d256vm4 checkpoint

**Opened 2026-08-05 (Berker verbatim: "theres this ssl transfer benchmark, including
dtd, arcr., cars, cifar10, cifar100, flowers102, food, pets datasets … we have one
good checkpoint for in1k from the d256vm4 experiment. lets put that onto a test!").**
Decision row D-078. Status when opened: EXECUTING.

**Current status (as of 2026-08-07): CLOSED.** **E25-T1** USER-APPROVED 2026-08-06 — the
transfer verdict plus the multi-ViT IN-1k program it launched. Benchmark-faithful linear
16/16 and LeJEPA-protocol few-shot 32/32 landed on d256vm4-1k. Successor: **E27**.

## Protocol of record

Faithful port of the linear track of **"How Well Do Self-Supervised Models Transfer?"
(Ericsson, Gouk, Hospedales, CVPR 2021; github.com/linusericsson/ssl-transfer, read
2026-08-05; donor code reference-only per house rules).** Frozen features → sklearn
logistic regression (lbfgs, multinomial, sklearn defaults), C swept over
logspace(−6, 5, 45) ascending with warm_start; C selected on val; final model refit on
train+val at C\*; scored on test. Metrics: **mean per-class accuracy** for aircraft /
flowers / pets, **top-1** for cifar10 / cifar100 / dtd / cars / food. Preprocessing:
Resize(224, bicubic) + CenterCrop(224) + ImageNet normalization; features
un-normalized. Datasets via torchvision at `~/data/ssltransfer`.

**Declared deviations from the donor** (both recorded here before any number landed):
1. Features = our declared h (trunk CLS, D-003) as primary, with the GAP tap as a
   non-primary rider column (donor: ResNet50 avgpool 2048-d — architecture forces the
   substitution).
2. Datasets without an official torchvision val split (cifar10/100, cars, food, pets)
   get a stratified 20% carve from train at seed 0 (donor ships custom split files;
   dtd/aircraft/flowers use their official val splits).
3. **Cars source = the `tanganke/stanford_cars` HF mirror** — torchvision's
   downloader is upstream-dead (ValueError, first launch 63045160, 2026-08-05).
   Canonical 8144/8041/196 split ASSERTED at load. Mirror-selection trail: the
   first-choice `Donghyun99/Stanford-Cars` was REJECTED by that assert (8143/8040 —
   one image short per split; job 63045502); tanganke verified exact via the HF
   datasets-server API before the switch (ClassLabel = the standard 196-name list).

Checkpoint: `outputs/in1k.floorssl.s0.d256vm4_best.pt` — the warm-ring view-mean
IN-1k cell (ViT-S/16@224, 100 ep, in1k online-probe best .6392; the D-075 at-scale
default estimator). Entry point `experiments/transfer_probe.py`, launcher
`slurm/transfer.sbatch` (one dataset per job); results append to
`results/transfer/in1k.floorssl.s0.d256vm4.csv`.

## Pre-registered predictions (committed 2026-08-05, BEFORE any benchmark number)

- **P-E25-1 (uniform deficit):** on the natural-object datasets (cifar10, cifar100,
  food, pets) our accuracies sit below the benchmark's published in1k-SSL R50 rows
  (SimCLR-v2/BYOL era) by a roughly constant offset tracking our in1k-probe gap
  (.6392 vs their ~.70–.74).
- **P-E25-2 (fine-grained extra deficit):** aircraft and cars show a LARGER deficit
  than the natural-object average (the benchmark's own SSL finding, amplified by
  ViT-S capacity and our shorter budget).
- **P-E25-3 (label-free Ω link):** Ω at h, computed per transfer dataset from
  o8-style augmentation views under the canonical estimator, ranks our per-dataset
  transfer accuracy within the natural-image datasets; DTD (texture) exempted —
  domain shift breaks the within-family shape-stability premise. (Rider job after
  the benchmark rows land; label-free per the E23-T3 metric ruling.)
- **P-E25-4 (space):** cls ≥ gap on most datasets (cls is the trained probe target);
  gap may lead on DTD (texture favors spatial pooling).

Known risk, recorded at launch: torchvision's StanfordCars downloader has a dead
upstream historically — a cars crash is a recorded blocker, not silently skipped.

## Few-shot track (added 2026-08-05; Berker: "i think few shots make more sense because lejepa has the matched epoch option at least")

**Reference = LeJEPA Table 2** (arXiv:2511.08544): 1-shot/10-shot/full linear probes
on the SAME 8 datasets — this is evidently where the dataset list originates — with
LeJEPA ViT-L (304M) and ConvNeXtV2-H (660M) both pretrained **100 ep on IN-1k:
epoch- and pretraining-matched** to d256vm4 (our arch is 14× smaller, accepted).
Their k-shot mechanics are UNPUBLISHED (main text, appendix and repo eval/ checked
2026-08-05) → ours is a **declared reconstruction**: support = k stratified
images/class from the full labeled pool (train + official val), 20 draws (seeds
0–19), logistic (lbfgs, C=1.0, max_iter=1000), scored on official test; top-1
primary (their likely metric) + the E25 per-dataset metric as rider; features h CLS
(+ gap); LeJEPA probed concat-CLS-of-last-2-layers — a recorded feature-choice
difference. Entry `experiments/transfer_fewshot.py` →
`results/transfer/in1k.floorssl.s0.d256vm4.fewshot.csv`.

Reference rows (LeJEPA Table 2, transcribed 2026-08-05; top: 1-shot, bottom: 10-shot):

| model | DTD | Aircr | Cars | C10 | C100 | Flow | Food | Pets | avg |
|---|---|---|---|---|---|---|---|---|---|
| ViT-L 1s | 33.21 | 9.37 | 3.40 | 51.65 | 27.01 | 48.53 | 17.14 | 46.11 | 29.55 |
| CnvV2-H 1s | 32.15 | 8.07 | 4.28 | 50.95 | 31.48 | 48.74 | 17.95 | 58.98 | 31.58 |
| ViT-L 10s | 64.72 | 35.25 | 22.25 | 85.15 | 59.77 | 92.53 | 50.90 | 77.00 | 60.95 |
| CnvV2-H 10s | 61.84 | 30.67 | 24.46 | 85.74 | 63.29 | 91.78 | 49.32 | 78.53 | 60.70 |

**Pre-registered (before any few-shot number):**
- **P-E25-5a:** our 10-shot per-dataset ordering mirrors our full-shot ordering.
- **P-E25-5b:** our 10-shot average lands BELOW LeJEPA ViT-L's 60.95 (14× parameter
  gap at matched epochs); flowers is the most likely per-dataset exception (our
  full-shot already sits in the published band there).
- **P-E25-5c:** 1-shot fine-grained (cars, aircraft) is near-floor for us as it is
  for them (their ViT-L: 3.40 / 9.37).

### Few-shot numbers — COMPLETE 2026-08-05 (cls space, top-1 mean±std over 20 draws; LeJEPA ViT-L 100ep ref in parens)

| dataset | 1-shot | ref | 10-shot | ref |
|---|---|---|---|---|
| aircraft | .0608±.006 | (9.37) | .2552±.009 | (35.25) |
| cars | .0414±.003 | (3.40) | .2422±.004 | (22.25) |
| cifar10 | .5431±.046 | (51.65) | .8070±.012 | (85.15) |
| cifar100 | .2715±.016 | (27.01) | .5517±.006 | (59.77) |
| dtd | .3013±.027 | (33.21) | .5825±.011 | (64.72) |
| flowers | .3820±.018 | (48.53) | .8379±.006 | (92.53) |
| food | .1304±.011 | (17.14) | .3783±.007 | (50.90) |
| pets | .4096±.026 | (46.11) | .7044±.009 | (77.00) |
| **average** | **.2675** | (29.55) | **.5449** | (60.95) |

Context row, raw: ours = ViT-S/16, 22M, 100 ep; the reference = ViT-L, 304M, 100 ep
(14× parameters, same pretraining data + epochs). Gap-space rows in the CSV,
systematically below cls. P-E25-5a/b/c scoring reserved for the joint read.

### THE COMBINED TABLE (Berker's ask 2026-08-06: "few shot all shot table (with lejepa comparison)") — all %; ours cls space

| dataset | ours 1s | LJ‑L 1s | ours 10s | LJ‑L 10s | ours full | LJ‑L full |
|---|---|---|---|---|---|---|
| aircraft | 6.08 | 9.37 | 25.52 | 35.25 | 45.98 | 57.01 |
| cars | **4.14** | 3.40 | **24.22** | 22.25 | 47.32 | 57.28 |
| cifar10 | **54.31** | 51.65 | 80.70 | 85.15 | 93.36 | 96.50 |
| cifar100 | **27.15** | 27.01 | 55.17 | 59.77 | 76.98 | 83.71 |
| dtd | 30.13 | 33.21 | 58.25 | 64.72 | 70.48 | 78.30 |
| flowers | 38.20 | 48.53 | 83.79 | 92.53 | 90.17 | 91.21 |
| food | 13.04 | 17.14 | 37.83 | 50.90 | 69.20 | 82.05 |
| pets | 40.96 | 46.11 | 70.44 | 77.00 | 83.31 | 89.74 |
| **average** | **26.75** | 29.55 | **54.49** | 60.95 | **72.10** | 79.48 |

LJ‑L = LeJEPA ViT-L (304M, IN-1k, 100 ep — matched data+epochs, 14× params; Table 2,
arXiv:2511.08544). Bold = ours above the LJ-L reference. Footnotes: (i) our few-shot
= the declared reconstruction (their mechanics unpublished); (ii) our "full" = the
benchmark C-swept logistic — their full-column protocol unspecified; (iii) metric mix
on aircraft/flowers/pets "full": ours per-class mean (benchmark rule; aircraft test
balanced → ≡ top-1), theirs likely top-1; few-shot columns are top-1 both sides.
ConvNeXtV2-H reference rows on this card above. ALL RAW — reading joint.

## Launch log

- 2026-08-05: 8 jobs fired via `slurm/transfer.sbatch` — 63045153 cifar10 · 63045154
  cifar100 · 63045155 dtd · 63045156 aircraft · 63045157 flowers · 63045158 food ·
  63045159 pets · 63045160 cars.

## Numbers

*(rows land here from results/transfer/in1k.floorssl.s0.d256vm4.csv as jobs finish;
columns: space, metric, C\*, val, test)*

| dataset | space | metric | C\* | val | **test** |
|---|---|---|---|---|---|
| aircraft | cls | perclass | 3.16 | .4197 | **.4598** |
| aircraft | gap | perclass | 31.6 | .2950 | .3315 |
| cars | cls | top1 | 5.62 | .4599 | **.4732** |
| cars | gap | top1 | 17.8 | .2697 | .2696 |
| cifar10 | cls | top1 | 0.056 | .9381 | **.9336** |
| cifar10 | gap | top1 | 5.62 | .9070 | .8850 |
| cifar100 | cls | top1 | 0.032 | .7573 | **.7698** |
| cifar100 | gap | top1 | 1.78 | .6941 | .6905 |
| dtd | cls | top1 | 0.56 | .6601 | **.7048** |
| dtd | gap | top1 | 0.18 | .6420 | .7016 |
| flowers | cls | perclass | 10 | .8569 | **.9017** |
| flowers | gap | perclass | 31.6 | .7902 | .8555 |
| pets | cls | perclass | 0.32 | .8642 | **.8331** |
| pets | gap | perclass | 56.2 | .7325 | .6512 |
| food | cls | top1 | 0.1 | .6374 | **.6920** |
| food | gap | top1 | 5.62 | .5660 | .5941 |

(food landed 2026-08-06 on resubmit 63065037; the first job 63045158 TIMEOUT'd — its
full 6h went to the 5 GB download + NFS unpack, racing the few-shot job's unpack of
the same dir. Linear track COMPLETE: 16/16 rows.)

Published linear reference band (Ericsson et al. CVPR 2021 — all R50/23.5M, IN-1k
pretraining, epochs UNCONTROLLED in their study, 120–1000; transcribed 2026-08-05):
supervised / SimCLR-v2 / BYOL / SwAV / DC-v2 = aircraft 43.59/46.38/53.87/54.04/54.49
· cars 44.92/50.37/56.40/54.06/58.60 · cifar10 91.42/92.53/93.26/93.99/94.02 ·
cifar100 73.90/76.78/77.86/79.58/79.61 · dtd 72.23/76.38/76.91/77.02/78.62 · flowers
89.93/92.90/94.50/94.62/94.72 · food 69.49/73.08/73.01/76.62/77.94 · pets
91.45/84.72/89.10/87.60/89.36.

## AGREED TAKEAWAY

**E25-T1 (USER-STATED 2026-08-06, Berker verbatim): "our takeaway for the transfer
job is that this is promising for initial results but we are not quite there.
lightly benchmark reports 64% in1k top1 accuracy for lejepa (vit s16, 100 epochs,
512 batch size) and we beat that recipe. our comparison with ssl transfer has an
architecture confound this is why we will look for training of multiple vits on
in1k next. we shouldnt do it blindly, this is where our target recipe for joint
loss forces come into the play while our R6 read provides an additional free
guide."**

Grounding numbers on record: d256vm4 (ViT-S/16, 100 ep, **bs 128** — batch setting
differs from the Lightly row's 512, otherwise matched arch/epochs/data) landed
linear at declared h: raw **.6416** / l2 **.6554** (online best .6392) vs the
Lightly-benchmark LeJEPA reference **64.0** (Berker's citation, 2026-08-06). The
matched-frame margins: toy +4.9 (floorssl .8808 vs lejepa-retrain .8318), in100
+7.0 lin / +14.3 kNN (vcc .7244/.6568 vs lejepa .6544/.5134); both sides of the
toy/in100 comparisons ran the lejepa authors' own recipe. The ssl-transfer band
comparison carries the R50-vs-ViT-S architecture confound and is a reference band
only. **NEXT: the multi-ViT in1k program (S/B/L) with dose-placement ON TARGET —
guided by the joint-loss-force recipe (E24) + the R6 threshold-normalized band as
the free geometric guide.** Guillotine takeaways deferred to the joint read.
