# λ-JEPA

Code for *λ-JEPA: Spectral Anti-Collapse Regularization for Self-Supervised Learning*. It contains the training
code for λ-JEPA and for the baselines of the paper (SimCLR, BYOL, VICReg, DINO, LeJEPA, VISReg), each with the
option of adding SACReg at the backbone, the evaluation protocols, the video trainer, and the two-layer experiments
of the theory appendix.

The regularizer is `SACReg` in `sslgap/methods/_common.py`; λ-JEPA is `sslgap/methods/lambdajepa.py`. Adding
SACReg to another method's backbone is `+method.h_reg=sacreg +method.h_lamb=<weight>`. In the code the paper's
β_z and β_h are `w_floor` and `h_lamb`, and the logged terms are `moment_kl` and `h_moment_kl`.

## Setup

```bash
uv sync
source .venv/bin/activate
bash scripts/selftest.sh      # one GPU, a few minutes
```

The video trainer under `video/levjepa/` has its own environment (`uv sync` there).

Datasets are expected at `~/data/imagenet`, `~/data/imagenet100` (the CMC split; class list in `data/`) and
`~/data/ssltransfer` (downloaded on first use). For video, `LEVJEPA_DATA_ROOT`, `SSV2_CLIPFILES` and
`K400_CLIPFILES` point at the clip stores built by the scripts in `video/levjepa/scripts/`.

## Running

Every step is a plain command; we ran them as cluster jobs. Multi-GPU steps expect the usual `torchrun` or `srun`
environment.

**ImageNet-1k.** `bash scripts/in1k.sh vits|vitb 100|400` trains λ-JEPA with the paper's settings on two GPUs and
resumes from the last checkpoint if interrupted. Evaluate a checkpoint with `python experiments/bench_probe.py
<ckpt> <run>` (linear probe and kNN) and `python experiments/transfer_visreg.py native:<ckpt> <tag>` (transfer).

**ImageNet-100.** `bash scripts/in100_cells.sh list` prints the fourteen cells: each baseline with and without
SACReg, and λ-JEPA with and without its backbone term. `train <cell> [seed]` trains one cell on one GPU and
`land <run_id>` runs the probes and the representation statistics. The paper averages seeds 0, 1 and 2.
`experiments/twospace.py` computes the class-separability and view-sensitivity statistics, and
`experiments/feature_drift.py` the feature and kernel drift.

**Video.** From `video/levjepa/`, `scripts/train_k710_vits.sh` and `scripts/train_k710_vitb.sh` train on the
Kinetics-710 subset on eight GPUs. `scripts/attentive_probe_in1k.sh`, `scripts/video_probe.sh` and
`scripts/k400_feature_cache.sh` followed by `python experiments/k400_features_read.py` are the evaluations.

**Loss weights.** `experiments/pull.py` measures each loss term's backbone gradient norm at a checkpoint, and
`experiments/loss_weights.py` turns the norms into the weights with the rule of the appendix.

**LeJEPA with SIGReg at the encoder output** (appendix). `python experiments/train.py method=lejepa
frame=in100_vits16 bs=128 num_classes=100 +method.h_reg=sigreg +method.h_lamb=0.003763 tag=sigreg_h`, then
`land` as above.

**Theory.** `theory/two_layer_linear/test_reg_encoder.py` and the drivers in `theory/two_layer_relu/` run on a CPU
and need no data.

## Third-party code

Nothing is vendored. The baselines follow the public implementations: [solo-learn](https://github.com/vturrisi/solo-learn)
for SimCLR, BYOL and VICReg, the official [DINO](https://github.com/facebookresearch/dino) code, the authors'
LeJEPA code and its [Lightly](https://github.com/lightly-ai/lightly) benchmark, and the
[VISReg](https://github.com/HaiyuWu/VISReg) code for the method and its transfer protocol. `video/levjepa/` is our
modified copy of a few files of the [LeVJEPA](https://github.com/MLO-lab/LeVJEPA) trainer (MIT, see its `LICENSE`;
commit `3ea0dda`) with our loss and loaders added; for the rest of the trainer see the upstream repository.

Checkpoints and result files are not included.
