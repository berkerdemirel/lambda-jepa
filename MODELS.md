# MODELS.md — model-instance matrix

One row per model instance (a checkpoint lineage under one frame+method+seed). Filled as instances
are created. Rule (PROTOCOL §6.8): **never mix provenance within a comparison.**

## Track A — existing checkpoints (M0 inputs; legacy formats via `sslgap/ckpt/adapters.py`)

| run_id | method | frame | source | heads present | probed branch | validation status |
|---|---|---|---|---|---|---|
| `toy.lejepa-lamb002.ext` | LeJEPA (λ=0.02) | toy ViT-S/8@128 Imagenette | `../lejepa/ckpt_lamb002.pt` | `proj` (16-d) + `z.embed` Linear(384→512) | student | **extracted+probed 2026-07-02** — h.cls linear 91.2% ≈ official minimal ballpark |
| `toy.lejepa-lamb0.ext` | LeJEPA (λ=0) ablation | same | `../lejepa/ckpt_lamb0.pt` | same | student | extracted+probed 2026-07-02 |
| `toy.infonce.ext` | InfoNCE variant | same | `../lejepa/ckpt_infonce.pt` | same | student | extracted+probed 2026-07-02 |
| `in100.dino-ctrl.ep25/50/100.ext` | DINO (classic) | IN-100 ViT-S/16@224 | `../ssl_explore/outputs/inv_dino-in100_ep{25,50,100}.pt` | student+teacher backbones & DINO heads, centers | teacher | extracted+probed 2026-07-02; parity ref = diag_dino teacher/30k-linspace protocol (see docs/HISTORY.md — NOT the "@4k" table label) |
| `in100.dino-ctrl.ep100.parity30k` | DINO parity instance | same | same ep100 ckpt | same | teacher | dedicated parity run: linspace-30000 probe manifest, battery on val (N-matched RankMe) |
| `toy.randinit-s0.ext` / `in100.randinit-s0.ext` | random-init nulls | per frame | generated (adapter random_init, seed 0) | same arch as parent | student | extracted+probed 2026-07-02 (PROTOCOL §6.6 anchors) |

## Track B — controlled retrains (core-7; M1 toy → M2 IN-100)

*(rows added as runs land; every row must fill: recipe donor+commit, deviations, pixels/epoch,
ckpt paths, wandb id, git sha, validation-vs-donor status)*

| run_id | method | frame | seed | recipe donor | deviations | pixels/ep ratio | validation | status |
|---|---|---|---|---|---|---|---|---|
| — | LeJEPA | toy | 0 | official minimal (`../lejepa`, exact) | none intended | 4 views | must reproduce official curve (loss within amp noise, probe ~1 pt) | planned M1 (ported first) |
| — | SimCLR | toy | 0 | solo-learn @9187ea3 (reviewed port) | ViT trunk (paper is RN50) | 2 views | M1.5 RN18-IN-100 vs published 66.2ish | planned M1 |
| — | VICReg | toy | 0 | solo-learn @9187ea3 + paper | ViT trunk | 2 views | M1.5 | planned M1 |
| — | BYOL | toy | 0 | solo-learn @9187ea3 + paper | ViT trunk; EMA base scaled to steps/ep | 2 views | M1.5; collapse canary = 0.3% signature | planned M1 |
| — | DINO | toy | 0 | sslx `train_dinov2.py` (restructured) | documented in dossier | 2g+Vl crops (~1.7×) | vs existing IN-100 control at M2 | planned M1 |
| — | MAE | toy | 0 | solo-learn/MMSelfSup donor (no paper ViT-S recipe) | ViT-S decoder scaled | 1 view, 25% visible | weak linear probe is EXPECTED — validate via kNN + finetune-lite | planned M1 |
| — | I-JEPA | toy | 0 | official repo + sslx `ijepa.py` modules | ViT-S scale-down from ViT-H paper | 1 view, multi-block masks | target-variance collapse monitor from step 0 | planned M1 |
| — | supervised DeiT-lite | in100 | 0 | timm recipe | anchor | 1 view | — | planned M2 |

## Track C — public IN-1k checkpoints (M3 validation rung; head inventory TO VERIFY at M3 entry)

| method | expected source | ships heads? (expectation — verify) | known blockers |
|---|---|---|---|
| DINO ViT-S/16 | facebookresearch/dino "full checkpoint" | student+teacher+heads — likely | — |
| iBOT ViT-S/16 | bytedance/ibot | full ckpt w/ heads — likely | — |
| MAE ViT-B/16 | facebookresearch/mae | "visualize" ckpts include decoder — likely | ViT-H full ckpt availability unclear |
| I-JEPA ViT-H/14 | facebookresearch/ijepa | encoder+predictor+target-encoder snapshots — likely | ViT-H scale (inference ok) |
| MoCo v3 ViT-S/B | facebookresearch/moco-v3 | full ckpts (resume-able) — likely | — |
| VICReg RN50 | facebookresearch/vicreg | `resnet50_fullckpt.pth` — likely | ResNet (arch gap vs ViT rows — separate table) |
| Barlow Twins RN50 | facebookresearch/barlowtwins | full ckpt — likely | ResNet |
| SwAV RN50 | facebookresearch/swav | prototypes+head in ckpt — likely | ResNet |
| SimCLR | google-research/simclr | TF checkpoints w/ head | TF→PyTorch conversion friction |
| BYOL | deepmind-research/byol | JAX pickle w/ projector+predictor | JAX→PyTorch conversion friction |
| DINOv2 | facebookresearch/dinov2 | **backbone only — no public heads** | z-space impossible publicly; h-only rows, flagged |
| LeJEPA | rbalestr-lab/lejepa | training code; we have own toy ckpts | retrain-only at IN-1k |

Covariates recorded per public row: pretrain data (IN-1k vs LVD-142M etc.), epochs, backbone,
resolution, license. Published linear/kNN numbers must be reproduced within ~1 pt under the paper's
own protocol before any battery number from that checkpoint is trusted.
