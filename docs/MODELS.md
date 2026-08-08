# MODELS.md — the model-instance registry

Re-scoped 2026-08-07. The program now has **1100+ checkpoints across ~250 run-ids**, so a
row-per-instance matrix is not maintainable and was three weeks stale. This file is now the
**registry**: what tracks exist, what validated each one, and which card carries the per-cell
record. **The experiment cards are the authority for individual runs** — every cell's config,
dose, and numbers live there, not here.

Binding rule (PROTOCOL §6.8): **never mix provenance within a comparison.** The three tracks
below are separate tables in every figure and every read. Terminology: [GLOSSARY.md](GLOSSARY.md).

---

## Track A — reference checkpoints (M0 inputs; legacy formats via `sslgap/ckpt/adapters.py`)

| run_id | method | frame | source | probed branch | validation |
|---|---|---|---|---|---|
| `toy.lejepa-lamb002.ext` · `toy.lejepa-lamb0.ext` · `toy.infonce.ext` | LeJEPA λ=.02 / λ=0 / InfoNCE | toy ViT-S/8@128 | `../lejepa` | student | extracted+probed 2026-07-02; h.cls linear 91.2% ≈ official ballpark |
| `in100.dino-ctrl.ep{25,50,100}.ext` (+ `.parity30k`) | DINO | IN-100 ViT-S/16@224 | `../ssl_explore` | teacher | **M0 parity PASSED exactly** once the true reference protocol was identified (HISTORY_ARCHIVE 2026-07-02) |
| `toy.randinit-s0.ext` · `in100.randinit-s0.ext` | random-init nulls | per frame | generated (adapter, seed 0) | student | PROTOCOL §6.6 anchors — every battery table carries one |

## Track B — controlled retrains (ours)

**Frames:** `toy_vits8` (Imagenette ViT-S/8@128, 150 ep) · `in100_vits16` (ViT-S/16@224, 100 ep)
· `in1k_vits16` (ViT-S/16@224, 100 ep) · `in1k_vitb16`/`in1k_vitl16` (declared, not yet run).
**Uniform house optimizer** for all methods (D-018 Option A) with the per-method escape clause;
house hygiene (grad_clip 1.0, eta_min ≤ lr/20, warmup ≥ 1 ep, cadence checkpoints, per-step
grad-norm) applies to every arm (D-016). **Single seed 0 throughout** — D-024 ruled out seed
replicates.

### B1 · The core-7 uniform trainers (M1 toy → M2 IN-100)

All seven ported and certified at toy: `lejepa` · `simclr` · `vicreg` · `byol` · `dino` · `mae` ·
`ijepa`, plus the `deitlite` supervised anchor (D-007). **The port that certifies the stack is
LeJEPA:** ep800 toy portval 0.9037 vs the official 0.90217 (Δ +0.15 pt), sigreg shape correlation
0.9997 — the trainer stack is validated against an exact external reference (HISTORY_ARCHIVE
2026-07-03). Per-method recipes, donors, and deviations: `docs/methods/<m>.md`. The IN-100 grid
carries every lane plus its E20 conditioner arm. One historical trap on record: the toy DINO run
has a superseded predecessor (`toy.dino.s0` — probe-label incident) and the run that feeds tables
is **`toy.dino.s0.probefix`**.

### B2 · The house method (`floorssl`, renaming to `spectral` under D-083)

The program's main lane, ~630 run-ids across toy and IN-100 and 6 at IN-1k. Per-cell records live
on their cards — do not look for them here.

| lane | frame | card | note |
|---|---|---|---|
| E19 arms 1–6 | in100 | [E19](experiments/E19_floorssl.md) | the vicreg-class ancestors; doses predate the dose law |
| E21 class + dim/aug/payment arms | in100 | [E21](experiments/E21_floorssl_class.md) | own class; the view-mean + ring anatomy settled here |
| E23 capacity grids (stage C, stage C′) | toy | [E23](experiments/E23_mlp_leakage.md) | 14 + 12 cells; C′ is the share-pinned one |
| E24 dose grid (waves 0–2, v-cells) | toy, in100 | [E24](experiments/E24_dose_interaction.md) | ~40 cells; produced the share recipe |
| `d256vm2` · `d256vm3` · `d256vm4` | **in1k** | [E22](experiments/E22_floorssl_in1k.md) | read as E23-T1…T3; **vm4 is the winner** and E25's transfer subject |
| `e24voas` | **in1k** | [E24](experiments/E24_dose_interaction.md) | the OAS cell; ep74/100 on 2026-08-07 |
| `e27smc` · `e27smcb` · `e27slg` · `e27slgb` | **in1k** | [E27](experiments/E27_guided_sbl.md) | **RUNNING**; multicrop Recipe v2, anchor/band × two local-scale families |
| `in1k.lejepa.s0.e27lej` | **in1k** | [E27](experiments/E27_guided_sbl.md) | **RUNNING**; the matched-frame LeJEPA ViT-S control |

**External reference for the IN-1k lane:** Lightly's LeJEPA ViT-S/16 @ 100 ep, bs 512 → 64.0
top-1. E25-T1 records the comparison and its confounds; the matched-frame control (`e27lej`) is
running precisely because that published number is not at our frame.

## Track C — public IN-1k checkpoints (E26, D-076)

Six models, base size throughout ("across s/b/l use b"), native input resolution, **trunk
stations only** (L03/L06/L09/cls; no head stations, no transmission into z). Full record:
[E26](experiments/E26_public_zoo.md); stations: `results/diag/e26_stations.csv`.

| model | timm weights | selftest tier | anchor reproduced |
|---|---|---|---|
| `dinov2b14` | `vit_base_patch14_dinov2.lvd142m` | hub-agreement, feature cos ≥ .999 | linear .8478 vs 84.5; kNN .8274 vs 82.1 ✓ |
| `dinov3b16` | `vit_base_patch16_dinov3.lvd1689m` | hub-agreement (license-gated fetch) | paper numbers at battery stage |
| `dinob16` | `vit_base_patch16_224.dino` | hub-agreement | kNN20 .7574 vs 76.1 ✓ |
| `clipb16` | `vit_base_patch16_clip_**quickgelu**_224.openai` | zero-shot IN-1k val, open_clip end-to-end | .8014 vs 80.2 ✓ |
| `siglipb16` | `vit_base_patch16_siglip_224.webli` | zero-shot IN-1k val | published 76.0 ± .8 |
| `maeb16` | `vit_base_patch16_224.mae` | **cfg + feature-stat smoke only — weakest tier, marked** | loose |

**Two catches worth carrying forward.** (1) The CLIP **QuickGELU trap**: the non-quickgelu
variant scored 64.42 against a published 68.3 — a 4-point "failure" that was a config mismatch,
not a model or pipeline problem; using the quickgelu variant on **both** sides gave 68.32 PASS.
(2) SigLIP needs the HF tokenizer dependency. Both were caught by the per-model selftest gate
*before* any battery number was trusted — which is the whole reason the gate exists.

**Standing caveats.** Public DINOv2/v3 ship **no heads**, so no z-space claim is possible for
them — h-only rows, flagged. MAE is a **poor-organization anchor** by construction (Berker's
words): reconstruction models are expected to read badly on organization metrics, and that is
the point of including one, not a defect to explain away.

Covariates recorded per public row: pretrain data (IN-1k vs LVD-142M/1689M vs WIT/WebLI), epochs,
backbone, resolution, license. **Published linear/kNN must be reproduced before any battery
number from a checkpoint is trusted** — that rule is what caught the QuickGELU trap.

## Retired

**PIVOT** (`in100.pivot.*`, 16 checkpoints; E13–E16): the external-teacher target line, KILLED at
the distillation ceiling (D-032…D-034, ceiling ruled D-038). Its library code is scheduled for
removal under D-083 Wave B. **M1.5** (`rn18_in100` solo-learn port validation): declared in
PROTOCOL §2, never launched, deferred indefinitely (D-017).
