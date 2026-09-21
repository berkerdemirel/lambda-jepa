# λ-JEPA / SACReg — code supplement

Anonymous code supplement for *λ-JEPA: Spectral Anti-Collapse Regularization for Self-Supervised
Learning*: the training code for λ-JEPA and for the six controlled baselines with and without SACReg
at the backbone, every evaluation protocol used in the paper, the loss-weight calibration, the video
(LeVJEPA-based) trainer, the data-preparation scripts, and the figure scripts.

**Names.** The paper's *SACReg* is the class `sslgap.methods._common.SACReg`; the paper's
*λ-JEPA* is the method `lambdajepa` (`sslgap/methods/lambdajepa.py`, class `LambdaJEPA`). Adding
SACReg at the backbone of an existing method is the option `h_reg=sacreg` with weight `h_lamb`.
Internal experiment and decision codes (`E20`, `D-058`, …) appear in comments; they are
provenance notes and carry no information a reader needs.

| paper | code |
|---|---|
| SACReg(X), Eq. (3) | `SACReg` module: KL of the batch mean/covariance of a fresh random `d_slice`-dimensional slice to N(0, I), divided by d′ |
| β_h, β_z, invariance weight | `method.h_lamb`, `method.w_floor`, `method.w_inv` |
| view centers (mean over V views) | `method.z_floor_batch=view_mean`, `method.h_floor_batch=view_mean` |
| slice widths d′ (z 128; h = D/3) | `method.z_d_slice`, `method.h_d_slice` |
| ring buffer of q previous steps | `method.queue_steps` (z), `method.h_queue_steps` (h) |
| projector: 2 hidden BN-ReLU layers of 2048, output 256 | `method.head_layers=2 method.expander_dim=256` (`expander_hidden: 2048`, `head_norm: bn`) |
| V global views, LeJEPA augmentation family | `+method.aug=lejepa +method.V=6` |
| averaged (EMA) invariance target | `+method.swa=ema` |
| backbone term added to another method | `+method.h_reg=sacreg +method.h_lamb=<β_h>` (`+method.h_tap=cls` for VICReg) |
| logged term names | `inv`, `moment_kl` (= β_z term), `h_moment_kl` (= β_h term) |
| representation h | `student.h.cls` in the feature store (CLS token of `forward_features`); `student.z.embed` = the 512-d embedding of LeJEPA/VISReg |

## Layout

```
sslgap/            the library: methods/ (one recipe class per method), models/, data.py, ckpt/ (checkpoint schema
                   + adapters for public checkpoints), extract/ (feature store), metrics/, probes/
experiments/       Hydra entry points (train.py, train_ddp.py, extract.py, probe.py, twospace.py, feature_drift.py)
                   and plain-argv scripts (bench_probe.py, transfer_visreg.py, pull.py, depth_metrics.py, …)
experiments/configs/   train.yaml + method/*.yaml + frame/*.yaml (dataset × backbone × epochs)
slurm/             sbatch wrappers and the two launchers (in1k_launch.sh, in100_controlled_launch.sh)
video/levjepa/     our fork of the public LeVJEPA trainer (MIT; UPSTREAM_README.md, LICENSE) with the λ-JEPA loss
data/              the 100 ImageNet-100 class ids (CMC split)
```

## Setup

```bash
uv sync                      # Python 3.11, torch 2.11.0 (CUDA 13.0 wheels), timm, hydra; installs sslgap editable
source .venv/bin/activate
sbatch slurm/selftest.sbatch # the two gates the training jobs run first (one GPU, a few minutes) -> SELFTEST_OK
```

The video trainer has its own environment (`cd video/levjepa && uv sync`, Python 3.12, PyTorch Lightning,
`stable-pretraining`). Its `pyproject.toml` is the upstream one; `timm` and `pandas`, used by our probe and
subset scripts, are installed as transitive dependencies of the lock file.

Every compute step is written as a SLURM job. The sbatch files reference the package root through
`$SLURM_SUBMIT_DIR`, so submit from the package root (or from `video/levjepa/` for the video jobs). Partition
and constraint lines (`gpu`, `gpu100`, `H100`, `A100|L40S|A40`) are our cluster's and need adapting. Weights &
Biases logging defaults to `wandb_mode=offline`; `disabled` turns it off.

## Data

| dataset | expected location | note |
|---|---|---|
| ImageNet-1k | `~/data/imagenet/{train,val}` (ImageFolder) | 1,281,167 / 50,000 images |
| ImageNet-100 | `~/data/imagenet100/{train,val}` | the CMC 100-class subset; class ids in `data/imagenet100_classes_cmc.txt` (126,689 / 5,000 images) |
| transfer sets | `~/data/ssltransfer` | DTD, Aircraft, Cars, CIFAR10/100, Flowers, Food, Pets; downloaded by `transfer_visreg.py` (torchvision; Cars from the HF mirror `tanganke/stanford_cars`) |
| Kinetics-710 20 % subset | `$LEVJEPA_DATA_ROOT` | `video/levjepa/scripts/k710_subsample.py` (union of K400/600/700-2020 train lists, duplicates and validation ids removed, 20 % per class, seed 0) → `build_lance_kinetics.py` → `lance_to_clipfiles.py` |
| SSv2, K400 evaluation clips | `$SSV2_CLIPFILES`, `$K400_CLIPFILES` | `video/levjepa/scripts/build_clipfiles.py` (`--ssv2`, `--kinetics`; sbatch wrappers in `video/levjepa/slurm/`) |
| public checkpoints | `~/ckpts_public/` | loaded by `sslgap/ckpt/adapters.py:from_pubvit`; sources: OK-AI (HF collection `OK-AI/imagenet-1k-self-supervised-vit-baselines`, ViT-S/B, 100/300 epochs), MoCo v3 (`dl.fbaipublicfiles.com/moco-v3/vit-{s,b}-300ep/`), iBOT teacher checkpoints (official release), VISReg (authors' release), DINO via `timm:vit_base_patch16_224.dino` |

## Reproducing the paper

### ImageNet-1k λ-JEPA (the ImageNet-1k linear/kNN table and the transfer tables)

```bash
bash slurm/in1k_launch.sh vits 100     # ViT-S/16, 100 epochs   -> outputs/in1k.lambdajepa.s0.v6s100_*.pt
bash slurm/in1k_launch.sh vits 400
bash slurm/in1k_launch.sh vitb 100
bash slurm/in1k_launch.sh vitb 400
```

Each call submits a chain of 120-hour DDP segments (`slurm/in1k_ddp.sbatch`: the two selftests, then
`torchrun` with one worker per GPU; global batch 128 over two GPUs; a segment resumes from `_last.pt`). The
override sets inside the launcher are the ones recorded by the runs themselves; the loss weights are
(31.07, 225.50, 1.671) for ViT-S and (22.81, 222.26, 4.054) for ViT-B, shared by the 100- and 400-epoch runs.

Evaluation of a checkpoint `outputs/<run>_ep100.pt`:

```bash
sbatch slurm/bench_probe.sbatch outputs/<run>_ep100.pt <run>          # Lightly-recipe linear probe (90 ep) + kNN k=200
sbatch slurm/transfer_visreg.sbatch native:outputs/<run>_ep100.pt <tag>   # VISReg transfer protocol, 8 datasets
sbatch slurm/extract.sbatch ckpt=~/ckpts_public/<file> adapter=pubvit run_id=<id>   # public checkpoints -> same protocols
```

`bench_probe.py` is a port of the Lightly benchmark's linear evaluation (BatchNorm without affine + linear head on
the frozen CLS token, 90 epochs, SGD momentum 0.9 at lr 0.1·batch/256, cosine with 10 warm-up epochs, best
validation top-1; kNN with k = 200, cosine similarity, temperature 0.07). `transfer_visreg.py` is a port of the
VISReg downstream protocol (CLS tokens of the last four blocks concatenated, a linear head per learning rate over
13 rates for 10 epochs, best test accuracy per dataset). Results land in `results/probes/<run>.bench*.csv` and
`results/transfer/<tag>.visreg_lp.csv`.

### Video (the video table)

```bash
cd video/levjepa
export LEVJEPA_DATA_ROOT=/path/to/kinetics                       # K710-20 % clip files
sbatch slurm/train_k710_vits.slurm                              # ViT-S/16, 240 epochs, 8 GPUs, batch 768
MAX_EPOCHS=515 sbatch slurm/train_k710_vitb.slurm               # ViT-B/16; then resume with MAX_EPOCHS=1085
sbatch slurm/attentive_probe_in1k.slurm <run dir>               # ImageNet-1k attentive probe
sbatch slurm/video_probe.slurm ssv2 attentive <run dir> <out csv>   # SSv2 attentive probe
sbatch slurm/video_probe.slurm k400 linear_mean <run dir> <out csv> # K400, linear on the token mean (20-epoch head)
sbatch slurm/k400_feature_cache.slurm <cache dir> name=<ckpt>   # then, from the package root: sbatch slurm/k400_features_read.sbatch <cache dir>
```

The loss lives in `video/levjepa/sslgap_reg.py` and `main.py` (`loss.type=sslgap`); the run configurations are
`conf/sslgap_vits.yaml` and `conf/sslgap_vitb.yaml` (weights = the ImageNet-1k values of the same backbone). The
ViT-B 240-epoch row is the `epoch-0239.ckpt` of the same run (`+keep.epochs=[239]`); the run was trained to 515
epochs and then resumed to 1,085. `k400_features_read.py` fits the converged L-BFGS logistic regression on cached
features (the number reported for K400) alongside the CLS-token and attentive reads.

### Controlled experiments on ImageNet-100 (the constraint-mismatch, two-space audit, thickness and organization figures; the appendix treatment figure)

```bash
bash slurm/in100_controlled_launch.sh smoke            # 1-epoch check of all 14 cells
bash slurm/in100_controlled_launch.sh chains "0"       # seed 0 = the paper's cells; "1 2" adds seeds
bash slurm/in100_controlled_launch.sh land <run_id>    # (re)run the landing chain of one run
```

The launcher holds the full override set of every cell (six methods as released and with
`+method.h_reg=sacreg +method.h_lamb=<β_h>`; λ-JEPA with `method.h_lamb=1.894` and its z-only twin with
`method.h_lamb=0.0`). Batch sizes are per lane: VICReg, SimCLR and BYOL train at 256, LeJEPA, DINO, VISReg and λ-JEPA at
128 (control and treated alike). Each run's landing chain is extract → probe (linear on raw CLS with AdamW
lr 1e-3 wd 1e-7 until validation stops improving; kNN k = 200, temperature 0.1; 500 training images per class)
→ `depth_metrics.py` (RankMe, effective rank, Gaussian KL, pair and class cosines per station) →
`thickness_append.py` (augmentation thickness Θ = tr A_h / tr B_h from 8 views of 10,000 training images).

```bash
sbatch slurm/twospace.sbatch run_id=<run>.extL          # per-class organization vs view-sensitivity figure
sbatch slurm/feature_drift.sbatch cell=in100.lambdajepa.s0.d256vm4 'epochs=[25,50,75,100]'   # rich-vs-lazy (appendix)
```

### Loss weights (appendix "Setting the loss weights")

```bash
sbatch slurm/pull.sbatch <label> outputs/<run>_ep25.pt ckpt        # per-term backbone gradient norms g and shares
python experiments/loss_weights.py add <G_SSL>                      # beta_h = T/(1-T) * G_SSL / g_R, T=.06, g_R=.30
python experiments/loss_weights.py own <g_inv> <g_z> <g_h>          # w_k = 57 * s_k / g_k, s = (.441, .542, .017)
```

For λ-JEPA the g's are the `[share] ep1` line of a one-epoch pilot (`share_log_every=1`); `own 0.809 0.137 0.580`
returns the ViT-S weights and `own 1.102 0.139 0.239` the ViT-B weights of the paper.

### Appendix "A closer look at LeJEPA"

```bash
sbatch slurm/train.sbatch method=lejepa bs=128 frame=in100_vits16 num_classes=100 \
    +method.h_reg=sigreg +method.h_lamb=0.003763 tag=hpull_sigreg
```

(LeJEPA's own SIGReg added at the 512-d encoder output, then the same landing chain.)

### Figures

The landing chains write `results/probes/<run>.csv` (ImageNet-100 `linear_raw_v2`, `knn_v1_k200`),
`results/probes/<run>.bench*.csv` (ImageNet-1k), `results/transfer/<tag>.visreg_lp.csv`, `results/diag/` (depth
metrics, thickness) and `results/twospace/`. The figure scripts read them:

```bash
python experiments/paper_exhibits.py          # backbone-vs-loss-space bars, thickness-vs-accuracy, organization-vs-sensitivity,
                                              # appendix treatment figure (seed 0) -> results/figures/
python experiments/in100_treatment_figure.py  # appendix treatment figure with seed bands
python experiments/in100_seed_table.py --fig  # seed table
python experiments/feature_drift_figure.py    # appendix rich-regime figure
```

## Notes

- **Loss-weight shares.** The exact shares behind the reported weights are (0.441, 0.542, 0.017); the rounded
  values do not reproduce them to the printed precision.
- **Seeds.** The paper's cells are seed 0; `in100_controlled_launch.sh chains "1 2"` adds seeds, and the seed
  table / banded figure scripts read them.
- **Not included:** the two-layer linear and two-layer ReLU experiments of the theory appendix (separate code
  base of the theory part), the drivers of the constraint-mismatch and two-space capacity audit figures (the
  metrics they plot are in `sslgap/metrics/`), pretrained checkpoints and result files.

## Third-party code

No third-party package is vendored or imported at runtime. `video/levjepa/` is a fork of the public LeVJEPA
trainer (MIT license, commit `3ea0dda`); our additions are `sslgap_reg.py`, the `loss.type=sslgap` branch of
`main.py`, the clip-file loaders in `data/`, the probe and data scripts in `scripts/`. The Lightly benchmark
recipe, the VISReg transfer protocol, the V-JEPA attentive-probe protocol and the SimCLR/BYOL/VICReg/DINO/LeJEPA/
VISReg training recipes are re-implemented in `sslgap/` from the respective public repositories; each method's
module docstring records the reference commit and every deviation.
