# E26 — public pretrained-zoo guillotine (IN-1k val reference band)

**Opened under D-076 (USER-APPROVED 2026-08-04; prioritized by Berker 2026-08-05:
"(i) public checkpoint guillotines").** The theory track's sanity anchor: strong
public backbones with NO engineered nondegeneracy guarantee, measured under our
guillotine columns. Status when opened: WAVE 1 (fetch + selftests) EXECUTING.

**Current status (as of 2026-08-07): LANDED, READ PENDING.** All 6 models fetched,
selftested, and measured; anchors reproduced; the cloud-thickness stations are on record
(`results/diag/e26_stations.csv`). Owed: the grey-free zoo guillotine figure, then the joint
read — P-E26-1/2/3 scores are prepared but **not** agreed, and no takeaway is written until
that round happens.

## Rules of record (all from D-076)

- Provenance-separated: own track, own stations CSV, own figure — never mixed into
  retrain comparisons.
- **Trunk stations only** ("l03, l06, l09 and cls will continue. post mlps are
  canceled"): full guillotine column set (lin/knn/rank/effrank/gauss_kl/pos/rand/
  class margin + Ω) at L03/L06/L09/cls; NO head stations, NO a/b/Λ→z.out
  (CLIP/SigLIP projection heads included in the cancel).
- Adapters verified before extraction (selftest per model); native input resolution;
  base-size variants ("across s/b/l use b"); D-005 store rules; network fetch on
  compute nodes (HF_HUB_OFFLINE overridden in the zoo sbatch only).

## Roster (wave 1) and selftest tiers

| model | timm weights | selftest | anchor |
|---|---|---|---|
| clipb16 | vit_base_patch16_clip_**quickgelu**_224.openai | zero-shot IN-1k val (open_clip end-to-end) | published 68.3 (±0.8) |
| siglipb16 | vit_base_patch16_siglip_224.webli | zero-shot IN-1k val | published 76.0 (±0.8) |
| dinov2b14 | vit_base_patch14_dinov2.lvd142m | timm-vs-official-hub feature cosine ≥ .999 on 64 fixed val inputs | + published kNN 82.1 verified at battery stage |
| dinob16 | vit_base_patch16_224.dino | same hub-agreement check | + published kNN 76.1 at battery stage |
| maeb16 | vit_base_patch16_224.mae | cfg + feature-stat smoke only (weakest tier, marked) | published linear 68.0 at battery stage |
| dinov3b16 | vit_base_patch16_dinov3.lvd1689m | fetch attempt LAST (HF license-gated — a refusal is a recorded skip, costs nothing else) | paper numbers at battery stage |

Shelves: web-scale (clip, siglip, dinov2, dinov3) vs **in1k-native SSL** (dino v1,
mae) — the theory should hold on both. moco-v3 dropped (no clean loader at B).
I-JEPA excluded by the use-b rule (no public B checkpoint).

## Pre-registered predictions (committed 2026-08-05, before any zoo number)

- **P-E26-1:** across the zoo, published probe quality (linear/kNN) orders with Ω at
  cls in the same direction as our retrain families (lower Ω ↔ stronger model).
- **P-E26-2:** every strong backbone sits BELOW its own fitted touching threshold
  (M < 0 at cls) on IN-1k val — no engineered moment matching required.
- **P-E26-3:** both shelves (web-scale and in1k-native) show the pattern —
  provenance-robustness of the state variable.

## Launch log

- 2026-08-05: `experiments/pubzoo_selftest.py` via `slurm/pubzoo_selftest.sbatch`
  (single GPU job, wave 1: fetch + selftests → `results/diag/pubzoo_selftest.csv`).
  Job 63045161. Extraction wave fires only after selftests are reviewed.
- 2026-08-05 **wave 2 canary:** dinob16 full chain fired smoke-first (house rule) —
  extract 63046343 → probe 63046344 + audit 63046345 (afterany). The canary verifies
  the `pub.*` spaces ride the standing probe/audit machinery before the 5-model fleet
  fires; dinov2b14 (@518) is the long pole of the fleet (~7 h).
- 2026-08-05 **canary PASSED + battery-stage anchor reproduced:** extraction [done]
  (train/val 8 spaces + o8); standing probe accepted the pub store — `pub.h.cls`
  kNN k=20 **.7574 vs published DINO-B/16 kNN 76.1** (k=20 is DINO's own protocol k),
  linear_l2 .7784 vs published linear ~78.2. **5-model fleet fired:** dinov2b14
  63047437/8/9 · clipb16 63047440/1/2 · siglipb16 63047443/5/6 · maeb16 63047447/8/9 ·
  dinov3b16 63047450/1/2 (extract/probe/audit each).
- 2026-08-05 **fleet incident + preemption:** clip/siglip/mae extracted clean (~35
  min each, probes/audits chained on). dinov3b16 (63047450) host-OOM'd at the ORBIT
  stage (evals had completed) — cause: 12 loader workers × prefetched 8-view pixel
  batches vs the 96G cgroup at 256px. Fix: orbit loader workers 12→4 (code comment
  cites the job); dinov3 rerun 63047740/1/2 at 180G. dinov2b14 @518px would have hit
  the same wall ~6h in (≈80 GB of view buffers alone) — its running chain was
  preempted 45 min in and resubmitted on the patched script: 63047746/7/8 at 180G.
- 2026-08-05 **selftest catch #1 (the tier working as designed):** run 63045161 —
  clip zero-shot 64.42 vs 68.3 **FAIL**, cause = QuickGELU mismatch (open_clip's own
  warning: openai weights on a quick_gelu=False config; the roster's timm name had
  the same trap — `vit_base_patch16_clip_quickgelu_224.openai` is the correct
  variant, roster fixed both sides). siglip selftest crashed on a missing
  `transformers` (open_clip's SigLIP tokenizer is HF-backed; dependency added),
  which ended the job before dinov2/dino/mae/dinov3 ran. Wave 1 refired.

## Numbers

**Wave-1 selftest verdicts (run 63045519, 2026-08-05; `results/diag/pubzoo_selftest.csv`):**

| model | selftest | measured | reference | verdict |
|---|---|---|---|---|
| clipb16 (quickgelu) | zero-shot IN-1k val | **68.32** | 68.3 | PASS |
| siglipb16 | zero-shot IN-1k val | **76.05** | 76.0 | PASS |
| dinov2b14 | hub-agreement cos | 1.00000 | ≥.999 | PASS |
| dinob16 | hub-agreement cos | 1.00000 | ≥.999 | PASS |
| maeb16 | cfg smoke | norm 16.9 | — | SMOKE |
| dinov3b16 | cfg smoke (gated fetch SUCCEEDED) | norm 16.8 | — | SMOKE |

Native resolutions (timm-resolved, D-076 rule 4): clip/siglip/dino/mae 224 · dinov2
**518** · dinov3 **256**.

**Wave-2 design (recorded before extraction):** run_id `in1k.pub.<key>`; STANDARD
in1k manifests reused byte-identical (train.v1 1.28M / val.v1 50k / pairs10 o8 V=8 —
same image lists as the retrain cells); spaces `pub.h.{cls,gap}[.L03|.L06|.L09]`
(+ `pub.h.pool` at the final station for siglip, whose native h is attention-pooled;
its L-taps are gap-only — no prefix token exists); eval preprocessing = each model's
own timm-resolved transform (the selftest-verified path — native size/interp/
crop_pct/norm, a declared deviation from the house 224 eval_transform forced by rule
4); orbit views = the FIXED audit_v1 stack at native size with ONLY the normalization
tail affinely re-based to the model's constants (exact composition, geometry
byte-identical). pairs50 not extracted (its consumers are z-side instruments).

**Zoo stations — Ω (o8, raw; l2 rider within .05 everywhere) + cls probe headlines
(2026-08-06, `results/diag/e26_stations.csv`, probes from the fleet logs; ALL RAW,
P-E26 scoring joint):**

| model | Ω L03 | Ω L06 | Ω L09 | **Ω cls** | lin (best) | pub. lin | kNN20 | pub. kNN |
|---|---|---|---|---|---|---|---|---|
| dinov2b14 | 2.62 | 0.88 | 0.31 | **.320** | .8478 | 84.5 | .8274 | 82.1 |
| dinov3b16 | 3.64 | 1.17 | 0.67 | **.372** | .8473 | — | .8322 | — |
| dinob16 | 2.24 | 0.81 | 0.54 | **.388** | .7784 | ~78.2 | .7574 | 76.1 |
| siglipb16 (pool) | 2.22 | 1.06 | 0.81 | **1.020** | .8259 | — | .7936 | — |
| clipb16 | 3.51 | 3.70 | 2.19 | **1.038** | .8014 | 80.2 | .7501 | — |
| maeb16 | 4.58 | 4.59 | 4.33 | **2.649** | .6586 | 68.0 | .2705 | — |

Bare stats, recorded without reading: Spearman(probe-lin rank, Ω-cls rank) n6 =
**+.829** in the low-Ω-better direction; the three augmentation-invariance models
(dino family) sit at Ω cls .32–.39, the two language-supervised at ≈1.02–1.04, the
reconstruction model at 2.65. **Reading caveat (Berker 2026-08-06): "be careful when
you interpret mae or recons based models as their downstream perf and latent space
organization is also poor"** — reconstruction models are poor-organization anchors,
not counterexamples to organization claims. Every model's Ω is monotone decreasing L03→cls except
mae (flat-high) and clip (bump at L06). Threshold-normalized positions need the
zoo's own touch-census fit (c per D-076 track) — not yet computed.

## AGREED TAKEAWAY

*(joint only — empty until discussed)*
