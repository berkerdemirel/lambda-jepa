# λ-JEPA / SACReg

Code for *λ-JEPA: Spectral Anti-Collapse Regularization for Self-Supervised Learning*: the training code for
λ-JEPA and for the six baselines with and without SACReg at the backbone, the evaluation protocols of the paper,
the loss-weight calibration, the video (LeVJEPA-based) trainer, the data-preparation scripts, and the two-layer
linear and ReLU experiments of the theory appendix.

**Names.** The paper's *SACReg* is the class `sslgap.methods._common.SACReg`; the paper's *λ-JEPA* is the method
`lambdajepa` (`sslgap/methods/lambdajepa.py`, class `LambdaJEPA`). Adding SACReg at the backbone of an existing
method is the option `h_reg=sacreg` with weight `h_lamb`.

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
                   and plain-argv scripts (bench_probe.py, transfer_visreg.py, pull.py, depth_metrics.py, ...)
experiments/configs/   train.yaml + method/*.yaml + frame/*.yaml (dataset x backbone x epochs)
scripts/           the selftest and the two launchers with the paper's override sets (in1k.sh, in100_cells.sh)
video/levjepa/     our modified copy of the public LeVJEPA trainer with the λ-JEPA loss; scripts/*.sh = the video commands
data/              the 100 ImageNet-100 class ids (CMC split)
theory/            the two-layer experiments of the theory appendix: two_layer_linear/ (block Gram matrices; numpy)
                   and two_layer_relu/ (teacher-student sweeps with SACReg on the hidden layer; torch, CPU)
```

## Setup

```bash
uv sync                    # Python 3.11, torch 2.11.0 (CUDA 13.0 wheels), timm, hydra; installs sslgap editable
source .venv/bin/activate
bash scripts/selftest.sh   # the two gates every training job runs first (one GPU, a few minutes) -> SELFTEST_OK
```

The video trainer has its own environment (`cd video/levjepa && uv sync`, Python 3.12, PyTorch Lightning,
`stable-pretraining`). Its `pyproject.toml` is the upstream one; `timm` and `pandas`, used by our probe and
subset scripts, are installed as transitive dependencies of the lock file.

Every step below is a plain command. We ran them as scheduler jobs (one GPU per ImageNet-100 cell, two per
ImageNet-1k run, eight per video run); the multi-process steps expect the usual `torchrun`/`srun` environment
(`RANK`, `WORLD_SIZE`, `LOCAL_RANK`). Weights & Biases logging defaults to `wandb_mode=offline`; `disabled`
turns it off.

## Data

| dataset | expected location | note |
|---|---|---|
| ImageNet-1k | `~/data/imagenet/{train,val}` (ImageFolder) | 1,281,167 / 50,000 images |
| ImageNet-100 | `~/data/imagenet100/{train,val}` | the CMC 100-class subset; class ids in `data/imagenet100_classes_cmc.txt` (126,689 / 5,000 images) |
| transfer sets | `~/data/ssltransfer` | DTD, Aircraft, Cars, CIFAR10/100, Flowers, Food, Pets; downloaded by `transfer_visreg.py` (torchvision; Cars from the HF mirror `tanganke/stanford_cars`) |
| Kinetics-710 20 % subset | `$LEVJEPA_DATA_ROOT` | `video/levjepa/scripts/k710_subsample.py` (union of K400/600/700-2020 train lists, duplicates and validation ids removed, 20 % per class, seed 0) → `build_lance_kinetics.py` → `lance_to_clipfiles.py` |
| SSv2, K400 evaluation clips | `$SSV2_CLIPFILES`, `$K400_CLIPFILES` | `video/levjepa/scripts/build_clipfiles.sh k400|ssv2 train|val` |
| public checkpoints | `~/ckpts_public/` | loaded by `sslgap/ckpt/adapters.py:from_pubvit`; sources: OK-AI (HF collection `OK-AI/imagenet-1k-self-supervised-vit-baselines`, ViT-S/B, 100/300 epochs), MoCo v3 (`dl.fbaipublicfiles.com/moco-v3/vit-{s,b}-300ep/`), iBOT teacher checkpoints (official release), VISReg (authors' release), DINO via `timm:vit_base_patch16_224.dino` |

## Reproducing the paper

### ImageNet-1k λ-JEPA (the ImageNet-1k linear/kNN table and the transfer tables)

```bash
bash scripts/in1k.sh vits 100     # ViT-S/16, 100 epochs   -> outputs/in1k.lambdajepa.s0.v6s100_*.pt
bash scripts/in1k.sh vits 400
bash scripts/in1k.sh vitb 100
bash scripts/in1k.sh vitb 400
```

Each call runs the two selftests, then `torchrun` with one process per GPU (`NGPU`, default 2; global batch 128).
The trainer resumes from `outputs/<run>_last.pt`, so a wall-time-limited scheduler can run it as a chain of
segments. The override sets in the script are the ones recorded by the runs themselves; the loss weights are
(31.07, 225.50, 1.671) for ViT-S and (22.81, 222.26, 4.054) for ViT-B, shared by the 100- and 400-epoch runs.

Evaluation of a checkpoint `outputs/<run>_ep100.pt`:

```bash
python experiments/bench_probe.py outputs/<run>_ep100.pt <run>            # Lightly-recipe linear probe (90 ep) + kNN k=200
python experiments/transfer_visreg.py native:outputs/<run>_ep100.pt <tag>  # VISReg transfer protocol, 8 datasets
python experiments/extract.py ckpt=~/ckpts_public/<file> adapter=pubvit run_id=<id>   # public checkpoints -> same protocols
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
export LEVJEPA_DATA_ROOT=/path/to/kinetics                        # K710-20 % clip files
bash scripts/train_k710_vits.sh                                   # ViT-S/16, 240 epochs, 8 GPUs, batch 96 per GPU
MAX_EPOCHS=515 bash scripts/train_k710_vitb.sh                    # ViT-B/16; then MAX_EPOCHS=1085 resumes the same run
bash scripts/attentive_probe_in1k.sh <ckpt> <out csv>             # ImageNet-1k attentive probe
bash scripts/video_probe.sh ssv2 attentive <ckpt> <out csv>       # SSv2 attentive probe
bash scripts/video_probe.sh k400 linear_mean <ckpt> <out csv>     # K400, linear on the token mean (20-epoch head)
bash scripts/k400_feature_cache.sh <cache dir> name=<ckpt>        # then, from the repo root:
python experiments/k400_features_read.py --cache <cache dir>      # the converged L-BFGS logistic regression (the K400 number)
```

The training scripts run one process per GPU (`DEVICES` x `NUM_NODES`, default 1 x 8). The loss lives in
`video/levjepa/lambdajepa_reg.py` and `main.py` (`loss.type=lambdajepa`); the run configurations are
`conf/lambdajepa_vits.yaml` and `conf/lambdajepa_vitb.yaml` (weights = the ImageNet-1k values of the same backbone).
The ViT-B 240-epoch row is the `epoch-0239.ckpt` of the same run (`+keep.epochs=[239]`); the run was trained to 515
epochs and then resumed to 1,085.

### Controlled experiments on ImageNet-100

```bash
bash scripts/in100_cells.sh list                    # the 14 cells and their overrides
bash scripts/in100_cells.sh train <cell> [seed]     # one cell, one GPU, resumable -> outputs/<run_id>_ep100.pt
bash scripts/in100_cells.sh land <run_id>           # extract -> linear/kNN probes, depth metrics, augmentation thickness
```

The script holds the full override set of every cell (six methods as released and with
`+method.h_reg=sacreg +method.h_lamb=<β_h>`; λ-JEPA with `method.h_lamb=1.894` and its z-only twin with
`method.h_lamb=0.0`). Batch sizes are per lane: VICReg, SimCLR and BYOL train at 256, LeJEPA, DINO, VISReg and
λ-JEPA at 128 (control and treated alike). The paper reports means over seeds 0, 1 and 2. `land` runs extract →
probe (linear on raw CLS with AdamW lr 1e-3 wd 1e-7 until validation stops improving; kNN k = 200, temperature 0.1;
500 training images per class) → `depth_metrics.py` (RankMe, effective rank, Gaussian KL, pair and class cosines
per station) → `thickness_append.py` (augmentation thickness Θ = tr A_h / tr B_h from 8 views of 10,000 training
images).

```bash
python experiments/twospace.py run_id=<run>.extL       # per-class separability and view-sensitivity statistics
python experiments/feature_drift.py cell=in100.lambdajepa.s0.hz 'epochs=[25,50,75,100]'   # feature and kernel drift (appendix)
```

### Loss weights (appendix "Setting the loss weights")

```bash
python experiments/pull.py <label> outputs/<run>_ep25.pt ckpt       # per-term backbone gradient norms g and shares
python experiments/loss_weights.py add <G_SSL>                      # beta_h = T/(1-T) * G_SSL / g_R, T=.06, g_R=.30
python experiments/loss_weights.py own <g_inv> <g_z> <g_h>          # w_k = 57 * s_k / g_k, s = (.441, .542, .017)
```

For λ-JEPA the g's are the `[share] ep1` line of a one-epoch pilot (`share_log_every=1`); `own 0.809 0.137 0.580`
returns the ViT-S weights and `own 1.102 0.139 0.239` the ViT-B weights of the paper.

### Appendix "A closer look at LeJEPA"

```bash
python experiments/train.py method=lejepa bs=128 frame=in100_vits16 num_classes=100 \
    +method.h_reg=sigreg +method.h_lamb=0.003763 tag=sigreg_h
bash scripts/in100_cells.sh land in100.lejepa.s0.sigreg_h
```

(LeJEPA's own SIGReg added at the 512-d encoder output, then the same landing chain.)

### Two-layer experiments (theory appendix)

Both run on a CPU from their own directory and need no data.

```bash
cd theory/two_layer_linear && python test_reg_encoder.py     # ~2 min; the 8-8-8 linear network of the appendix
cd theory/two_layer_relu
WANDB_MODE=offline python run_grid_hreg.py                    # 4 (tau, lambda) points x 5 w_reg x 5 seeds
WANDB_MODE=offline python run_grid_hreg_densegrid.py          # 17 x 17 (tau, lambda) grid at w_reg in {0, 0.1}
WANDB_MODE=offline python run_grid_hreg_wregscale.py          # 17 tau x 7 w_reg at four lambda values
```

`test_reg_encoder.py` trains the linear network by full-batch gradient descent with weight decay λ_L2 ∈ {0, 0.1}
and the encoder log-determinant term λ_det ∈ {0, −0.2}, and compares the block Gram matrix Q of the regularized
network with the theoretical Q at λ_bal = λ_det/λ_L2 and with the Q of a network initialized at that balance and
trained without regularization. The ReLU drivers train the width-50 student on the width-3 teacher (n = 1,000
unit-sphere inputs, 20,000 full-batch steps, learning rate 5e-3/τ²) with `w_reg · SACReg(h)` on the post-ReLU
hidden layer (`d_slice = m = 50`), one process per grid point (`SLURM_CPUS_PER_TASK` workers when set). Each run
appends `results/Relu/<run_id>_diag.csv`, one row per checkpoint (steps 0, 100, 1k, 5k, 10k, 20k): train and test
loss, the regularizer value, the kernel distance of the two-layer NTK from initialization, and the rank measures of
h on the fixed test set.

## Notes

- **Loss-weight shares.** The exact shares behind the reported weights are (0.441, 0.542, 0.017); the rounded
  values do not reproduce them to the printed precision.
- **Not included:** plotting scripts, pretrained checkpoints and result files.

## Third-party code

Nothing third-party is vendored or imported at runtime. The baselines and protocols are re-implemented in
`sslgap/` from the public repositories, which are the reference for every recipe:

| what | reference |
|---|---|
| SimCLR, BYOL, VICReg recipes (ImageNet-100 settings) | [solo-learn](https://github.com/vturrisi/solo-learn) (`9187ea3`) |
| DINO | [facebookresearch/dino](https://github.com/facebookresearch/dino) |
| LeJEPA | the authors' public code; the ViT-S/16 benchmark row and the linear-probe recipe from [Lightly](https://github.com/lightly-ai/lightly) (`f444cf3`) |
| VISReg and its transfer protocol | [HaiyuWu/VISReg](https://github.com/HaiyuWu/VISReg) (`47b1cf4`) |
| attentive probe | the V-JEPA protocol as used by LeVJEPA |

`video/levjepa/` contains our modified copies of `main.py`, `module.py` and `callbacks.py` from the public
[LeVJEPA](https://github.com/MLO-lab/LeVJEPA) trainer (MIT license, `LICENSE`; commit `3ea0dda`) plus our
additions: `lambdajepa_reg.py`, the `loss.type=lambdajepa` branch, the clip-file loaders in `data/`, the probe,
data and launch scripts in `scripts/`. For everything else (the ViT with token dropping, the Lance builders, the
Walking Tours pipeline) refer to the upstream repository.
