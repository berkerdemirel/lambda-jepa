"""Assemble the anonymized code supplement of the lambda-JEPA paper from this repository.

The package must stay in sync with the code that produced the paper's numbers, and its anonymity
must be verifiable. The builder copies a declared manifest (only what the paper reports), applies
the paper's names (SpectralConditioner -> SACReg, FloorSSL -> LambdaJEPA, method key floorssl ->
lambdajepa, the zoo's backbone hook h_reg=moment -> h_reg=sacreg), rewrites cluster paths into
$SLURM_SUBMIT_DIR / environment variables, strips attributions and the wandb entity, replaces the
long module docstrings and sbatch comment blocks with one-liners, keeps only the paper's rows of
the result CSVs, then FAILS if any identifying string survives.

  python supplement/build.py            # -> supplement/dist/lambda_jepa_code/ and the .zip next to it
  python supplement/build.py --no-zip   # tree only;  --out DIR builds elsewhere
"""
import argparse
import ast
import io
import os
import re
import shutil
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
HERE = Path(__file__).resolve().parent
PACKAGE = "lambda_jepa_code"
HOME = Path.home()

# --------------------------------------------------------------------------------------------------
# Manifest: (source, destination). Sources relative to the repository root unless absolute; a
# trailing "/" copies a tree (minus DROP_*); "dir/*.csv" globs one level.
# --------------------------------------------------------------------------------------------------
COPY = [
    ("supplement/README.md", "README.md"),
    ("pyproject.toml", "pyproject.toml"),
    ("uv.lock", "uv.lock"),
    (".python-version", ".python-version"),
    ("sslgap/", "sslgap/"),
    # training
    ("experiments/train.py", "experiments/train.py"),
    ("experiments/train_ddp.py", "experiments/train_ddp.py"),
    ("experiments/train_ddp_selftest.py", "experiments/train_ddp_selftest.py"),
    ("experiments/e27_selftest.py", "experiments/lambdajepa_selftest.py"),
    ("experiments/configs/train.yaml", "experiments/configs/train.yaml"),
    ("experiments/configs/extract.yaml", "experiments/configs/extract.yaml"),
    ("experiments/configs/probe.yaml", "experiments/configs/probe.yaml"),
    ("experiments/configs/twospace.yaml", "experiments/configs/twospace.yaml"),
    ("experiments/configs/feature_drift.yaml", "experiments/configs/feature_drift.yaml"),
    # loss-weight calibration
    ("experiments/pull.py", "experiments/pull.py"),
    ("supplement/loss_weights.py", "experiments/loss_weights.py"),
    # evaluation
    ("experiments/extract.py", "experiments/extract.py"),
    ("experiments/probe.py", "experiments/probe.py"),
    ("experiments/bench_probe.py", "experiments/bench_probe.py"),
    ("experiments/transfer_visreg.py", "experiments/transfer_visreg.py"),
    ("experiments/e20f_depth_metrics.py", "experiments/depth_metrics.py"),
    ("experiments/e23_retro_append.py", "experiments/thickness_append.py"),
    ("experiments/twospace.py", "experiments/twospace.py"),
    ("experiments/e34_k400_features.py", "experiments/k400_features_read.py"),
    ("experiments/feature_drift.py", "experiments/feature_drift.py"),
    # exhibits
    ("experiments/paper_exhibits.py", "experiments/paper_exhibits.py"),
    ("experiments/e38_seeds.py", "experiments/in100_seed_table.py"),
    ("experiments/e38_zoo_seeds.py", "experiments/in100_treatment_figure.py"),
    ("scratch/e33_figure.py", "experiments/feature_drift_figure.py"),
    # SLURM
    ("slurm/_env.sh", "slurm/_env.sh"),
    ("supplement/selftest.sbatch", "slurm/selftest.sbatch"),
    ("slurm/train.sbatch", "slurm/train.sbatch"),
    ("slurm/e27_ddp400.sbatch", "slurm/in1k_ddp.sbatch"),
    ("supplement/in1k_launch.sh", "slurm/in1k_launch.sh"),
    ("slurm/e38_launch.sh", "slurm/in100_controlled_launch.sh"),
    ("slurm/e38_seeds.sbatch", "slurm/in100_cell.sbatch"),
    ("slurm/extract.sbatch", "slurm/extract.sbatch"),
    ("slurm/probe.sbatch", "slurm/probe.sbatch"),
    ("slurm/bench_probe.sbatch", "slurm/bench_probe.sbatch"),
    ("slurm/transfer_visreg.sbatch", "slurm/transfer_visreg.sbatch"),
    ("slurm/pull.sbatch", "slurm/pull.sbatch"),
    ("slurm/twospace.sbatch", "slurm/twospace.sbatch"),
    ("slurm/feature_drift.sbatch", "slurm/feature_drift.sbatch"),
    ("slurm/e34_k400_features.sbatch", "slurm/k400_features_read.sbatch"),
    # data preparation
    (str(HOME / "data/imagenet100/classes_cmc.txt"), "data/imagenet100_classes_cmc.txt"),
    ("scratch/video/k710_subsample.py", "video/levjepa/scripts/k710_subsample.py"),
    # the video fork (our additions to the public LeVJEPA trainer)
    ("video/levjepa/lambdajepa_reg.py", "video/levjepa/lambdajepa_reg.py"),
    ("video/levjepa/main.py", "video/levjepa/main.py"),
    ("video/levjepa/module.py", "video/levjepa/module.py"),
    ("video/levjepa/callbacks.py", "video/levjepa/callbacks.py"),
    ("video/levjepa/conf/", "video/levjepa/conf/"),
    ("video/levjepa/data/", "video/levjepa/data/"),
    ("video/levjepa/scripts/attentive_probe.py", "video/levjepa/scripts/attentive_probe.py"),
    ("video/levjepa/scripts/video_probe.py", "video/levjepa/scripts/video_probe.py"),
    ("video/levjepa/scripts/k400_feature_cache.py", "video/levjepa/scripts/k400_feature_cache.py"),
    ("video/levjepa/scripts/build_clipfiles.py", "video/levjepa/scripts/build_clipfiles.py"),
    ("video/levjepa/scripts/build_lance_kinetics.py", "video/levjepa/scripts/build_lance_kinetics.py"),
    ("video/levjepa/scripts/lance_to_clipfiles.py", "video/levjepa/scripts/lance_to_clipfiles.py"),
    ("video/levjepa/scripts/lambdajepa_smoke.py", "video/levjepa/scripts/lambdajepa_smoke.py"),
    ("video/levjepa/slurm/e34cf3_shape_h100x8x1.slurm", "video/levjepa/slurm/train_k710_vits.slurm"),
    ("video/levjepa/slurm/e34cf3_shape_h100x8x1_vitb.slurm", "video/levjepa/slurm/train_k710_vitb.slurm"),
    ("video/levjepa/slurm/attentive_probe_in1k.slurm", "video/levjepa/slurm/attentive_probe_in1k.slurm"),
    ("video/levjepa/slurm/video_probe_nx1.slurm", "video/levjepa/slurm/video_probe.slurm"),
    ("video/levjepa/slurm/k400_feature_cache.slurm", "video/levjepa/slurm/k400_feature_cache.slurm"),
    ("video/levjepa/slurm/build_k400_clipfiles.slurm", "video/levjepa/slurm/build_k400_clipfiles.slurm"),
    ("video/levjepa/slurm/build_ssv2_clipfiles.slurm", "video/levjepa/slurm/build_ssv2_clipfiles.slurm"),
    ("video/levjepa/pyproject.toml", "video/levjepa/pyproject.toml"),
    ("video/levjepa/uv.lock", "video/levjepa/uv.lock"),
    ("video/levjepa/.python-version", "video/levjepa/.python-version"),
    ("video/levjepa/LICENSE", "video/levjepa/LICENSE"),
    ("video/levjepa/README.md", "video/levjepa/UPSTREAM_README.md"),
]
COPY += [(f"experiments/configs/method/{m}.yaml",) * 2 for m in
         ("lejepa", "vicreg", "simclr", "dino", "byol", "visreg", "lambdajepa")]
COPY += [(f"experiments/configs/frame/{f}.yaml",) * 2 for f in ("in100_vits16", "in1k_vits16", "in1k_vitb16")]

DROP_DIRS = {"__pycache__", ".venv", "wandb", "outputs", ".git", ".ipynb_checkpoints", "audit"}
DROP_FILES = {"sslgap/methods/mae.py", "sslgap/methods/pivot.py", "sslgap/methods/supervised.py",
              "sslgap/metrics/census.py", "sslgap/metrics/defect_rank.py", "sslgap/metrics/patch.py",
              "sslgap/metrics/relrep.py"}
DROP_SUFFIX = {".pyc", ".pt", ".pth", ".npy", ".ckpt", ".png", ".pdf"}


# --------------------------------------------------------------------------------------------------
# Names: the paper's names for the code's frozen identifiers (contents AND file names)
# --------------------------------------------------------------------------------------------------
IDENT = [("SpectralConditioner", "SACReg"), ("FloorSSL", "LambdaJEPA"), ("floorssl", "lambdajepa")]
HREG = [('"h_reg") == "moment"', '"h_reg") == "sacreg"'),
        ('"sigreg": "h_sigreg", "moment": "h_moment_kl"', '"sigreg": "h_sigreg", "sacreg": "h_moment_kl"'),
        ('"moment": lambda: self.floor(emb_in)', '"sacreg": lambda: self.floor(emb_in)'),
        ('h_reg=moment', 'h_reg=sacreg'), ('"h_reg": "moment"', '"h_reg": "sacreg"'),
        ('h_reg="moment"', 'h_reg="sacreg"'), ('h_reg: moment', 'h_reg: sacreg')]
NAMES = {os.path.basename(s): os.path.basename(d) for s, d in COPY
         if not s.endswith("/") and "*" not in s and os.path.basename(s) != os.path.basename(d)
         and not s.startswith(("supplement/", "results/", "video/levjepa/README"))}
STEMS = {k.rsplit(".", 1)[0]: v.rsplit(".", 1)[0] for k, v in NAMES.items() if k.endswith(".py")}

# --------------------------------------------------------------------------------------------------
# Anonymization
# --------------------------------------------------------------------------------------------------
REPO = "/nfs/scistore19/locatgrp/bdemirel/ssl_project"
VID = REPO + "/video/levjepa"
BEEGFS_DATA = "/mnt/beegfs/locatgrp/shared/datasets"
BEEGFS_WORK = "/mnt/beegfs/locatgrp/shared/bdemirel"
SHELL_RULES = [
    (re.compile(r"^#SBATCH --job-name=e27ddp400", re.M), "#SBATCH --job-name=in1k_ddp"),
    (re.compile(r"^#SBATCH --job-name=e38$", re.M), "#SBATCH --job-name=in100_cell"),
    (re.compile(r"^#SBATCH --job-name=e34cf3_vits_k710_h100x8x1", re.M), "#SBATCH --job-name=lambdajepa_k710_vits"),
    (re.compile(r"^#SBATCH --job-name=e34cf3_vitb_k710_h100x8x1", re.M), "#SBATCH --job-name=lambdajepa_k710_vitb"),
    (re.compile(r"^#SBATCH --output=" + re.escape(REPO) + r"/outputs/", re.M), "#SBATCH --output=outputs/"),
    (re.compile(r"^#SBATCH --exclude=.*\n", re.M), ""),
    (re.compile(r"^export WANDB_ENTITY=.*\n", re.M), ""),
    (re.compile(r"source " + re.escape(REPO) + r"/slurm/_env\.sh"), 'source "$SLURM_SUBMIT_DIR/slurm/_env.sh"'),
    (re.compile(r"source " + re.escape(REPO) + r"/\.venv/bin/activate"), 'source "$SLURM_SUBMIT_DIR/.venv/bin/activate"'),
    (re.compile(r"cd " + re.escape(VID) + r"\b"), 'cd "$SLURM_SUBMIT_DIR"'),
    (re.compile(r"cd " + re.escape(REPO) + r"\b"), 'cd "$SLURM_SUBMIT_DIR"'),
    (re.compile(r"^LEV=" + re.escape(VID) + r"\s*$", re.M), 'LEV="$SLURM_SUBMIT_DIR"'),
    (re.compile(r"PYTHONPATH=" + re.escape(VID)), 'PYTHONPATH="$SLURM_SUBMIT_DIR"'),
    (re.compile(r"^export LEVJEPA_DATA_ROOT=" + re.escape(BEEGFS_DATA) + r"/kinetics\s*$", re.M),
     'export LEVJEPA_DATA_ROOT="${LEVJEPA_DATA_ROOT:?set to the Kinetics store root}"'),
    (re.compile(r"RUNDIR=" + re.escape(BEEGFS_WORK) + r"/levjepa_runs/\$RUN"),
     'RUNDIR="${LEVJEPA_RUNS:-$SLURM_SUBMIT_DIR/outputs/runs}/$RUN"'),
    (re.compile(re.escape(BEEGFS_DATA) + r"/ssv2/ssv2_clipfiles"), '"${SSV2_CLIPFILES:?set to the SSv2 clip-file store}"'),
    (re.compile(re.escape(BEEGFS_DATA) + r"/kinetics/k400_eval_clipfiles"), '"${K400_CLIPFILES:?set to the K400 clip-file store}"'),
    (re.compile(re.escape(BEEGFS_DATA) + r"/kinetics"), '"${LEVJEPA_DATA_ROOT:?set to the Kinetics store root}"'),
    (re.compile(re.escape(BEEGFS_DATA) + r"/ssv2"), '"${SSV2_ROOT:?set to the SSv2 root}"'),
    (re.compile(r"HF_HOME=/nfs/scistore19/locatgrp/bdemirel/\.cache/huggingface"), 'HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"'),
    (re.compile(re.escape(REPO)), "$SLURM_SUBMIT_DIR"),
]
PY_RULES = [
    (re.compile(r'^\s*sys\.path\.insert\(0, "' + re.escape(REPO) + r'"\)\s*\n', re.M), ""),
    (re.compile(r'^ROOT = "' + re.escape(REPO) + r'"\s*$', re.M), 'ROOT = str(__import__("sslgap.paths", fromlist=["ROOT"]).ROOT)'),
    (re.compile(r'^B = "' + re.escape(BEEGFS_DATA) + r'/kinetics"\s*$', re.M), 'B = os.environ["LEVJEPA_DATA_ROOT"]'),
    (re.compile(re.escape(BEEGFS_DATA) + r"/kinetics"), "/path/to/kinetics"),
    (re.compile(re.escape(BEEGFS_DATA) + r"/ssv2"), "/path/to/ssv2"),
    (re.compile(re.escape(BEEGFS_WORK)), "/path/to/work"),
    # the method registry without the methods the paper does not use
    (re.compile(r"^from sslgap\.methods\.(ijepa|mae|pivot|supervised) import .*\n", re.M), ""),
    (re.compile(r"MAE, IJEPA, DeiTLite, Pivot, "), ""),
    (re.compile(r"^from sslgap\.models\.heads import BottleneckStage, ResBlock\n", re.M), ""),
    (re.compile(r"^(FOVEAL_SIZES|_FOVEAL_DOWN|_FOVEAL_GRID) = .*\n", re.M), ""),
    # the exhibits generator: figures go to results/figures; MAE and I-JEPA rows are not in the paper
    (re.compile(r"^FIG = .*$", re.M), 'FIG = os.path.join(ROOT, "results/figures")'),
    (re.compile(r'\n    \("(MAE|I-JEPA)", \[.*?\], "#[0-9a-f]{6}"\),', re.S), ""),
]
YAML_RULES = [
    (re.compile(re.escape(REPO) + r"/"), ""),
    (re.compile(r"^wandb_mode: online\s*$", re.M), "wandb_mode: offline"),
    (re.compile(r"^  - frame: toy_vits8\s*$", re.M), "  - frame: in100_vits16"),
    (re.compile(r"^num_classes: 10 .*$", re.M), "num_classes: 100"),
    (re.compile(r"^foveal: .*\n", re.M), ""),
    (re.compile(r"^adapter: lejepa_minimal .*$", re.M), "adapter: native"),
    (re.compile(re.escape(BEEGFS_DATA) + r"/kinetics"), "/path/to/kinetics"),
    (re.compile(re.escape(BEEGFS_DATA) + r"/ssv2"), "/path/to/ssv2"),
    (re.compile(re.escape(BEEGFS_WORK)), "/path/to/work"),
]
TEXT_RULES = [
    (re.compile(re.escape(REPO)), "<repo-root>"),
    (re.compile(r"/nfs/\.\.\./"), "/path/to/"),
    (re.compile(re.escape(BEEGFS_DATA)), "/path/to/datasets"),
    (re.compile(re.escape(BEEGFS_WORK)), "/path/to/work"),
    (re.compile(r"/nfs/scistore19/locatgrp/bdemirel"), "<home>"),
    (re.compile(r"Berker's"), "the authors'"),
    (re.compile(r"Berker"), "the authors"),
    (re.compile(r"bdemirel"), "user"),
    (re.compile(r"BeeGFS"), "the shared file system"),
    (re.compile(r"causal-learning-ai-ista"), "<wandb-entity>"),
]
FORBIDDEN = re.compile(r"berker|bdemirel|scistore|locatgrp|causal-learning|beegfs|/nfs/|whynot|"
                       r"\bISTA\b|ist\.ac\.at|floorssl|FloorSSL|SpectralConditioner", re.I)
TEXT_SUFFIX = {".py", ".yaml", ".yml", ".sh", ".sbatch", ".slurm", ".md", ".txt", ".json", ".csv", ".toml", ".lock", ""}

# --------------------------------------------------------------------------------------------------
# One-line docstrings (package path -> {None: module, "Name": class/function}); untouched elsewhere
# --------------------------------------------------------------------------------------------------
DOC = {
    "sslgap/methods/_common.py": {
        None: "Shared pieces: SACReg, the warmup + cosine schedule, trunk construction, the EMA momentum schedule.",
        "SACReg": "KL(N(mu, Sigma) || N(0, I)) / d' of the batch mean and covariance on a fresh random d'-dimensional orthonormal slice per step (the batch covariance is rank-deficient at full width); eps stabilizes the log-determinant, shrink=\"oas\" replaces the slice covariance by its OAS-shrunk estimate; fp32."},
    "sslgap/methods/base.py": {None: "Frame (what is identical across methods within a comparison) and the SSLMethod recipe interface."},
    "sslgap/methods/lambdajepa.py": {None: "lambda-JEPA: view-to-mean invariance at the projector output, SACReg on the per-image view centers at the backbone (h) and at the projector output (z), with ring buffers and random slices."},
    "sslgap/methods/lejepa.py": {None: "LeJEPA (SIGReg + invariance at the projector output) with the optional backbone term (h_reg=sacreg | sigreg)."},
    "sslgap/methods/vicreg.py": {None: "VICReg as released, with the optional SACReg term at the backbone (h_reg=sacreg, h_tap=cls)."},
    "sslgap/methods/simclr.py": {None: "SimCLR as released, with the optional SACReg term at the backbone (h_reg=sacreg)."},
    "sslgap/methods/byol.py": {None: "BYOL as released, with the optional SACReg term at the backbone (h_reg=sacreg)."},
    "sslgap/methods/dino.py": {None: "DINO as released, with the optional SACReg term at the backbone (h_reg=sacreg)."},
    "sslgap/methods/visreg.py": {None: "VISReg as released, with the optional SACReg term at the 512-d embedding (h_reg=sacreg)."},
    "sslgap/methods/ijepa.py": {None: "I-JEPA (kept for the checkpoint adapters; not used in the paper)."},
    "sslgap/__init__.py": {None: "Backbone (h) and loss-space (z) representations of self-supervised methods: training, feature stores, probes, metrics."},
    "sslgap/data.py": {None: "Datasets, the deterministic eval transform, augmentation stacks, versioned feature manifests."},
    "sslgap/paths.py": {None: "Package paths derived from this file's location (override with SSLGAP_ROOT)."},
    "sslgap/metrics/battery.py": {None: "Metric battery: pure functions over stored feature arrays."},
    "sslgap/metrics/cka.py": {None: "Linear CKA and Frobenius alignment (feature and kernel drift)."},
    "sslgap/metrics/cross.py": {None: "Cross-space metrics between two representations of the same images."},
    "sslgap/metrics/isotropy.py": {None: "Isotropy and Gaussianity: moment KL to N(0, I), the Epps-Pulley sliced statistic."},
    "sslgap/metrics/nulls.py": {None: "Null references (random-init backbone, Gaussian match)."},
    "sslgap/metrics/orbit_energy.py": {None: "Cloud energies over V-view stores: within-image W, debiased between-image B, thickness Theta = W/B, transmission a, b, Lambda."},
    "sslgap/metrics/pairs.py": {None: "Positive-pair metrics: alignment, positive and random-pair cosines, class-conditioned cosine margin."},
    "sslgap/metrics/single.py": {None: "Single-matrix metrics: uniformity, variance floor, redundancy, collapse margin."},
    "sslgap/metrics/spectra.py": {None: "Spectral diagnostics: covariance eigenvalues, effective rank, participation ratio, spectral-tail exponent, RankMe."},
    "sslgap/metrics/twospace.py": {None: "Two-space estimators over V-view stores: whitening, the Theta spectrum, per-class organization vs view-sensitivity."},
    "sslgap/extract/extractor.py": {None: "Frozen-feature extraction: one pass per (checkpoint, manifest) writing every requested space to the store."},
    "sslgap/extract/ntk.py": {None: "Empirical NTK of the trunk at the CLS token (random output projections, trunk parameters only)."},
    "sslgap/extract/store.py": {None: "Feature store: one fp16 .npy per (run_id, manifest, space) plus meta.json."},
    "sslgap/models/backbones.py": {None: "ViT trunks (timm) and the backbone readout."},
    "sslgap/models/heads.py": {None: "Projection heads and tap wrappers."},
    "sslgap/models/posembed.py": {None: "Fixed 2D sin-cos position embeddings."},
    "sslgap/models/vitops.py": {None: "Manual forward paths through a timm VisionTransformer and the EMA update."},
    "sslgap/ckpt/adapters.py": {None: "Adapters lifting checkpoint formats (native, public ViTs, LeJEPA, VISReg, DINO) into LoadedCkpt."},
    "sslgap/ckpt/schema.py": {None: "Checkpoint schema and the uniform in-memory form every checkpoint loads into."},
    "sslgap/probes/__init__.py": {None: "Frozen probe protocols."},
    "sslgap/probes/knn.py": {None: "Weighted-cosine kNN classifier."},
    "sslgap/probes/linear.py": {None: "Linear probes on frozen features: AdamW with early stopping, L-BFGS ridge logistic regression, LDA."},
    "experiments/train.py": {None: "Single-GPU trainer: one loop for every method (seeding, bf16, online probe, checkpoint cadence, resume)."},
    "experiments/train_ddp.py": {None: "Multi-GPU DDP trainer for lambda-JEPA: SACReg on the all-gathered view centers with rank-shared random slices; checkpoints in the schema of train.py."},
    "experiments/train_ddp_selftest.py": {None: "Gate: at world_size 1 the DDP loss reproduces the single-GPU training step exactly."},
    "experiments/lambdajepa_selftest.py": {None: "Gate: trunk resolution parity, the view-to-mean == all-pairs invariance identity, the ring buffer, one full training step."},
    "experiments/pull.py": {None: "Per-term backbone gradient norms and realized shares at a checkpoint: the measurement behind the loss weights."},
    "experiments/extract.py": {None: "Checkpoint -> feature store: train/val features per station and the 8-view orbit stores."},
    "experiments/probe.py": {None: "Frozen probes on a run's feature store: linear on raw features (AdamW, early stopping) and kNN (k = 200)."},
    "experiments/bench_probe.py": {None: "ImageNet-1k linear probe and kNN in the Lightly benchmark recipe (BatchNorm + linear head, 90 epochs; kNN k = 200)."},
    "experiments/transfer_visreg.py": {None: "Transfer linear probes under the VISReg protocol (concatenated CLS of the last four blocks, 13 learning rates, 10 epochs, eight datasets)."},
    "experiments/depth_metrics.py": {None: "Representation statistics per station from the feature store: RankMe, effective rank, Gaussian KL, pair and class cosines."},
    "experiments/thickness_append.py": {None: "Augmentation thickness per station (W, B, Theta = W/B) from the 8-view orbit store; appends to results/diag/e23_retro_spaces.csv."},
    "experiments/twospace.py": {None: "Two-space estimators over the 8-view stores: capacity, the Theta spectrum, per-class organization vs view-sensitivity (results/twospace/)."},
    "experiments/k400_features_read.py": {None: "Optimizer-free readers on cached K400 clip features: weighted kNN and converged L-BFGS logistic regression."},
    "experiments/feature_drift.py": {None: "Rich-vs-lazy diagnostic: per-layer CKA and empirical-NTK alignment of a run's checkpoints against its true initialization."},
    "experiments/feature_drift_figure.py": {None: "Feature and kernel drift figure from results/diag/feature_drift.<cell>.csv."},
    "experiments/paper_exhibits.py": {None: "The paper's figures from the landed CSVs (results/probes, results/diag, results/twospace, results/compare) -> results/figures/."},
    "experiments/in100_seed_table.py": {None: "Seed table of the ImageNet-100 cells: mean +/- 95 % t-interval of control, treated and the paired delta."},
    "experiments/in100_treatment_figure.py": {None: "The appendix treatment figure with seed bands (mean line, 95 % interval, seed dots)."},
    "video/levjepa/lambdajepa_reg.py": {None: "lambda-JEPA pieces for the LeVJEPA trainer: SACReg on the all-gathered per-clip view centers, ring buffers, the share logger, the online K710 probe, named checkpoints."},
    "video/levjepa/scripts/attentive_probe.py": {None: "ImageNet-1k attentive probe on a video checkpoint (V-JEPA frozen-evaluation protocol)."},
    "video/levjepa/scripts/video_probe.py": {None: "SSv2 (attentive) and K400 (linear on the token mean) frozen probes on clip-file stores, V-JEPA protocol."},
    "video/levjepa/scripts/k400_feature_cache.py": {None: "Cache pooled / patch-mean / CLS features of the K400 clips for the optimizer-free readers."},
    "video/levjepa/scripts/build_clipfiles.py": {None: "Build the SSv2 and K400 evaluation clip-file stores (one file of JPEG frames per clip)."},
    "video/levjepa/scripts/build_lance_kinetics.py": {None: "Build the K710-20 % training store from the subset list."},
    "video/levjepa/scripts/lance_to_clipfiles.py": {None: "Convert a Lance store to one file per clip."},
    "video/levjepa/scripts/k710_subsample.py": {None: "The K710-20 % pretraining list: union of the K400/600/700 train lists, duplicates and validation ids removed, 20 % per class (seed 0)."},
    "video/levjepa/scripts/lambdajepa_smoke.py": {None: "Smoke test of the lambda-JEPA loss branch of main.py on random clips."},
}
# comment headers of the shell scripts (inserted after the #SBATCH block; every other comment line goes)
USAGE = {
    "slurm/selftest.sbatch": ["# The two gates the DDP jobs run first:  sbatch slurm/selftest.sbatch   -> SELFTEST_OK"],
    "slurm/train.sbatch": ["# Single-GPU trainer:  sbatch slurm/train.sbatch method=<m> frame=<f> [overrides]"],
    "slurm/in1k_ddp.sbatch": ["# One DDP segment (torchrun, one worker per GPU); submitted by slurm/in1k_launch.sh"],
    "slurm/in1k_launch.sh": ["# ImageNet-1k lambda-JEPA chains:  bash slurm/in1k_launch.sh vits|vitb 100|400",
                             "# (the override sets recorded by the paper's runs; bs is the global batch over two GPUs)"],
    "slurm/in100_controlled_launch.sh": [
        "# ImageNet-100 controlled cells: six methods as released and with SACReg at the backbone, lambda-JEPA with and without it.",
        "#   bash slurm/in100_controlled_launch.sh smoke              # 1-epoch check of every cell",
        "#   bash slurm/in100_controlled_launch.sh chains \"0\"         # seed 0 = the paper; \"1 2\" adds seeds; each run lands (extract -> probe, depth metrics, thickness)",
        "#   bash slurm/in100_controlled_launch.sh land <run_id>      # (re)run one landing chain"],
    "slurm/in100_cell.sbatch": ["# One ImageNet-100 cell (single GPU, resumable); submitted by in100_controlled_launch.sh"],
    "slurm/extract.sbatch": ["# Checkpoint -> feature store:  sbatch slurm/extract.sbatch ckpt=<pt> adapter=native|pubvit run_id=<id> 'h_layers=[3,6,9]' orbit_v=8"],
    "slurm/probe.sbatch": ["# Frozen probes on a feature store:  sbatch slurm/probe.sbatch run_id=<id>"],
    "slurm/bench_probe.sbatch": ["# ImageNet-1k linear probe + kNN (Lightly recipe):  sbatch slurm/bench_probe.sbatch <ckpt> <run_id> [epochs=90] [adapter=native|pubvit]"],
    "slurm/transfer_visreg.sbatch": ["# VISReg transfer protocol:  sbatch slurm/transfer_visreg.sbatch native:<ckpt>|pubvit:<ckpt>|timm:<model> <tag>"],
    "slurm/pull.sbatch": ["# Per-term backbone gradient norms at a checkpoint:  sbatch slurm/pull.sbatch <label> <ckpt> ckpt"],
    "slurm/twospace.sbatch": ["# Two-space estimators over the 8-view stores (CPU):  sbatch slurm/twospace.sbatch run_id=<id>.extL"],
    "slurm/feature_drift.sbatch": ["# Rich-vs-lazy diagnostic:  sbatch slurm/feature_drift.sbatch cell=<run_id> 'epochs=[25,50,75,100]'"],
    "slurm/k400_features_read.sbatch": ["# Optimizer-free K400 readers over a feature cache:  sbatch slurm/k400_features_read.sbatch <cache dir> [out csv]"],
    "video/levjepa/slurm/train_k710_vits.slurm": ["# lambda-JEPA ViT-S/16 on K710-20 %: 240 epochs, 8 x 1 GPU, batch 96 x 8; resumes the run's newest last.ckpt"],
    "video/levjepa/slurm/train_k710_vitb.slurm": ["# lambda-JEPA ViT-B/16 on K710-20 %: 8 x 1 GPU, batch 64 x 8; MAX_EPOCHS env (515, then 1085 by resuming); epoch-0239.ckpt is kept"],
    "video/levjepa/slurm/attentive_probe_in1k.slurm": ["# ImageNet-1k attentive probe:  sbatch slurm/attentive_probe_in1k.slurm <lightning .ckpt> <out csv> [ema|student]"],
    "video/levjepa/slurm/video_probe.slurm": ["# SSv2 / K400 frozen probes on 8 ranks:  sbatch slurm/video_probe.slurm ssv2|k400 attentive|linear_mean <run dir or ckpt> <out csv> [ema|student] [bs]"],
    "video/levjepa/slurm/k400_feature_cache.slurm": ["# K400 feature cache:  sbatch slurm/k400_feature_cache.slurm <out_dir> name=<ckpt> [name=<ckpt> ...]"],
    "video/levjepa/slurm/build_k400_clipfiles.slurm": ["# K400 evaluation clip files (array job):  sbatch slurm/build_k400_clipfiles.slurm <train|val> [out dir]"],
    "video/levjepa/slurm/build_ssv2_clipfiles.slurm": ["# SSv2 clip files (array job):  sbatch slurm/build_ssv2_clipfiles.slurm"],
}
# the exhibits generator keeps its figure functions only (the tables of the paper are typed by hand)
EXHIBIT_KEEP = {"_spines", "h_station", "station_of", "load_zoo_data",
                "make_treatment_fig", "fig_treatment_arrows", "fig_org_vs_sensitivity", "fig_hz_bars"}
EXHIBIT_MAIN = '''if __name__ == "__main__":
    os.makedirs(FIG, exist_ok=True)
    ALL_Q = ["Linear", "kNN-200", "RankMe / d", "EffRank / d", "Gaussian KL",
             "cos margin (pos$-$rand)", "class margin", "$\\\\Theta$ (W/B)",
             "a (s$\\\\to$z)", "b (s$\\\\to$z)", "$\\\\Lambda$ (s$\\\\to$z)"]
    make_treatment_fig([f[0] for f in FAMILIES], ALL_Q, f"{FIG}/fig_treatment_appendix.png")
    fig_treatment_arrows()
    fig_org_vs_sensitivity()
    fig_hz_bars(("VICReg", "VISReg", "LeJEPA"), "fig_hz_bars_no_simclr")
'''


# comments go from our library and scripts (tokenizer-based: code and strings untouched); the
# internal docstrings of functions/classes go where they only carry provenance
STRIP_COMMENTS = ("sslgap/", "experiments/", "video/levjepa/lambdajepa_reg.py", "video/levjepa/scripts/",
                  "video/levjepa/data/clipfile_loader.py", "video/levjepa/data/gpu_views.py",
                  "video/levjepa/data/mp4_loader.py")
DROP_INNER_DOCS = ("sslgap/methods/", "experiments/", "video/levjepa/lambdajepa_reg.py", "video/levjepa/scripts/")
KEEP_INNER_DOCS = {"sslgap/methods/base.py"}
DROP_DEFS = {"sslgap/methods/lambdajepa.py": {"lambdajepa_res_head", "lambdajepa_stage_head"},
             "sslgap/data.py": {"PivotTrainDataset", "FovealPairDataset", "_foveal_rng", "foveal_iou",
                                "foveal_boxes", "foveal_event", "foveal_ctx", "make_source",
                                "supervised_stack", "build_manifest_imagenette"}}
# dead `if` branches: (function, substring of the test) -> the else-branch takes its place
DROP_IFS = {"sslgap/data.py": [("_Source.__init__", "imagenette"), ("_Source.__call__", "imagenette"),
                               ("_FullSplit.__init__", "imagenette"), ("_FullSplit.__call__", "source")],
            "experiments/extract.py": [("_manifests", "imagenette"), ("main", "foveal")]}
# exact edits after every other pass: (regex, replacement)
EDITS = {"experiments/extract.py": [
             (r"from sslgap\.data import \(EvalDataset, FovealPairDataset, OrbitDataset, PairDataset, STACKS,\n"
              r"\s+_Source, build_manifest_imagefolder, seed_everything,\n\s+build_manifest_imagenette\)",
              "from sslgap.data import (EvalDataset, OrbitDataset, PairDataset, STACKS, _Source,\n"
              "                         build_manifest_imagefolder, seed_everything)"),
             (r'ds = frame\.get\("dataset", "imagenette"\)', 'ds = frame["dataset"]')],
         "sslgap/data.py": [
             (r'"""Resolves manifest refs to PIL images\. kind: hf-imagenette \(ref = row index\) or\n\s+imagefolder \(ref = relpath under root\)\."""',
              '"""Resolves manifest refs (relative paths under root) to PIL images."""'),
             (r'"""Raw PIL access to a full split \(no manifest\): imagenette via HF, else ImageFolder\."""',
              '"""Raw PIL access to a full split (no manifest) via ImageFolder."""'),
             (r"\n\s+self\.source = None\n", "\n")]}


# --------------------------------------------------------------------------------------------------
def is_text(p: Path):
    return (p.suffix in TEXT_SUFFIX and p.name != ".python-version") or p.name == ".gitignore"


def rewrite(text, dst_rel):
    kind = Path(dst_rel).suffix
    rules = (SHELL_RULES if kind in (".sh", ".sbatch", ".slurm") else PY_RULES if kind == ".py"
             else YAML_RULES if kind in (".yaml", ".yml") else [])
    for rx, rep in rules:
        text = rx.sub(rep, text)
    for rx, rep in TEXT_RULES:
        text = rx.sub(rep, text)
    for old, new in HREG:
        text = text.replace(old, new)
    for old, new in IDENT:
        text = text.replace(old, new)
    for old, new in NAMES.items():
        text = re.sub(r"(?<![\w.])" + re.escape(old) + r"\b", new, text)
    if kind == ".py":
        for old, new in STEMS.items():
            text = re.sub(r"\b" + re.escape(old) + r"\b", new, text)
    return text


def _docstring(body):
    if body and isinstance(body[0], ast.Expr) and isinstance(body[0].value, ast.Constant) \
            and isinstance(body[0].value.value, str):
        return body[0]


def set_docstrings(src, spec):
    tree = ast.parse(src)
    lines = src.splitlines(keepends=True)
    edits = []
    if None in spec and (n := _docstring(tree.body)):
        edits.append((n.lineno, n.end_lineno, 0, spec[None]))
    for node in tree.body:
        if isinstance(node, (ast.ClassDef, ast.FunctionDef)) and node.name in spec \
                and (n := _docstring(node.body)):
            edits.append((n.lineno, n.end_lineno, n.col_offset, spec[node.name]))
    for a, b, col, text in sorted(edits, reverse=True):
        lines[a - 1:b] = [" " * col + f'"""{text}"""\n']
    return "".join(lines)


def strip_comments(src):
    import tokenize
    lines = src.splitlines(keepends=True)
    spans = {}
    for t in tokenize.generate_tokens(io.StringIO(src).readline):
        if t.type == tokenize.COMMENT:
            spans.setdefault(t.start[0], []).append((t.start[1], t.end[1]))
    for ln, sp in spans.items():
        line = lines[ln - 1]
        for a, b in sorted(sp, reverse=True):
            line = line[:a] + line[b:]
        nl = "\n" if line.endswith("\n") else ""
        lines[ln - 1] = None if line.strip() == "" else line.rstrip() + nl
    out = re.sub(r"\n{3,}", "\n\n", "".join(l for l in lines if l is not None))
    ast.parse(out)
    return out


def drop_inner_docstrings(src, keep=()):
    tree = ast.parse(src)
    lines = src.splitlines(keepends=True)
    edits = []
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)) and node.name not in keep \
                and (n := _docstring(node.body)):
            edits.append((n.lineno, n.end_lineno, n.col_offset, len(node.body) == 1))
    for a, b, col, lone in sorted(edits, reverse=True):
        lines[a - 1:b] = [" " * col + "pass\n"] if lone else []
    out = "".join(lines)
    ast.parse(out)
    return out


def _find_func(tree, qualname):
    node = tree
    for part in qualname.split("."):
        node = next(n for n in node.body if isinstance(n, (ast.FunctionDef, ast.ClassDef)) and n.name == part)
    return node


def drop_if(src, qualname, test_substr):
    tree = ast.parse(src)
    lines = src.splitlines(keepends=True)
    fn = _find_func(tree, qualname)
    for st in fn.body:
        if isinstance(st, ast.If) and test_substr in ast.get_source_segment(src, st.test):
            body = []
            if st.orelse:
                a, b = st.orelse[0].lineno, st.orelse[-1].end_lineno
                cut = st.orelse[0].col_offset - st.col_offset
                body = [l[cut:] if l.strip() else l for l in lines[a - 1:b]]
            lines[st.lineno - 1:st.end_lineno] = body
            out = "".join(lines)
            ast.parse(out)
            return out
    raise KeyError(f"{qualname}: no if on {test_substr!r}")


def drop_defs(src, names):
    tree = ast.parse(src)
    lines = src.splitlines(keepends=True)
    for node in sorted((n for n in tree.body if isinstance(n, (ast.FunctionDef, ast.ClassDef))
                        and n.name in names), key=lambda n: -n.lineno):
        a = node.lineno - 1
        while a > 0 and lines[a - 1].strip() == "":
            a -= 1
        lines[a:node.end_lineno] = []
    return "".join(lines)


def strip_yaml_comments(text):
    kept = [re.sub(r"\s+#.*$", "", l) for l in text.splitlines() if not l.strip().startswith("#")]
    return re.sub(r"\n{3,}", "\n\n", "\n".join(kept)).strip("\n") + "\n"


def filter_exhibits(src):
    tree = ast.parse(src)
    lines = src.splitlines(keepends=True)
    out = []
    for node in tree.body:
        if isinstance(node, ast.If) or (isinstance(node, ast.FunctionDef) and node.name not in EXHIBIT_KEEP):
            continue
        sep = "\n\n" if isinstance(node, ast.FunctionDef) else ""
        out.append(sep + "".join(lines[node.lineno - 1:node.end_lineno]))
    return "".join(out).rstrip("\n") + "\n\n\n" + EXHIBIT_MAIN


def strip_shell(text, usage):
    kept = []
    for line in text.splitlines():
        s = line.strip()
        if s.startswith(("#!", "#SBATCH")):
            kept.append(line)
        elif s.startswith("#"):
            continue
        else:
            kept.append(re.sub(r"\s+# .*$", "", line))
    text = re.sub(r"\n{3,}", "\n\n", "\n".join(kept)).strip("\n") + "\n"
    if usage:
        lines = text.split("\n")
        idx = max((i for i, l in enumerate(lines) if l.startswith(("#!", "#SBATCH"))), default=-1)
        lines[idx + 1:idx + 1] = usage
        text = "\n".join(lines)
    return text


def rename_path(rel):
    for old, new in IDENT:
        rel = rel.replace(old, new)
    return rel


def copy_file(src: Path, dst: Path, out: Path):
    dst.parent.mkdir(parents=True, exist_ok=True)
    rel = str(dst.relative_to(out))
    if not is_text(src):
        shutil.copy2(src, dst)
        return
    try:
        text = src.read_text()
    except UnicodeDecodeError:
        shutil.copy2(src, dst)
        return
    text = rewrite(text, rel)
    if rel == "experiments/paper_exhibits.py":
        text = filter_exhibits(text)
    if rel in DROP_DEFS:
        text = drop_defs(text, DROP_DEFS[rel])
    if rel in DOC:
        text = set_docstrings(text, DOC[rel])
    if dst.suffix == ".py" and rel.startswith(DROP_INNER_DOCS) and rel not in KEEP_INNER_DOCS:
        text = drop_inner_docstrings(text, keep=[k for k in DOC.get(rel, {}) if k])
    if dst.suffix == ".py" and rel.startswith(STRIP_COMMENTS):
        text = strip_comments(text)
    for qual, sub in DROP_IFS.get(rel, []):
        text = drop_if(text, qual, sub)
    for rx, new in EDITS.get(rel, []):
        text, n = re.subn(rx, new, text)
        assert n == 1, (rel, rx[:50], n)
    if dst.suffix in (".sh", ".sbatch", ".slurm"):
        text = strip_shell(text, USAGE.get(rel))
    if dst.suffix == ".yaml" and rel.startswith("experiments/configs/"):
        text = strip_yaml_comments(text)
    dst.write_text(text)
    shutil.copymode(src, dst)


def copy_tree(src: Path, dst: Path, out: Path, prefix: str):
    for p in sorted(src.rglob("*")):
        rel = p.relative_to(src)
        if any(part in DROP_DIRS for part in rel.parts) or f"{prefix}{rel}" in DROP_FILES:
            continue
        if p.is_file() and p.suffix not in DROP_SUFFIX:
            copy_file(p, dst / rename_path(str(rel)), out)


def build(out: Path):
    if out.exists():
        shutil.rmtree(out)
    for s, d in COPY:
        src = Path(s) if os.path.isabs(s) else ROOT / s
        if s.endswith("/"):
            copy_tree(src, out / d, out, s)
        elif "*" in s:
            hits = sorted(src.parent.glob(src.name))
            if not hits:
                sys.exit(f"[build] no match: {s}")
            for p in hits:
                copy_file(p, out / d / rename_path(p.name), out)
        else:
            if not src.exists():
                sys.exit(f"[build] missing source: {src}")
            copy_file(src, out / rename_path(d), out)
    for d in ("outputs", "features", "results/figures"):
        (out / d).mkdir(parents=True, exist_ok=True)
    (out / ".gitignore").write_text("outputs/\nfeatures/\nwandb/\n.venv/\n__pycache__/\n*.pt\n*.pth\n*.npy\n*.ckpt\n")


def gate(out: Path):
    hits = []
    for p in sorted(out.rglob("*")):
        if p.is_file() and is_text(p):
            try:
                for i, line in enumerate(p.read_text().splitlines(), 1):
                    if FORBIDDEN.search(line):
                        hits.append(f"{p.relative_to(out)}:{i}: {line.strip()[:140]}")
            except UnicodeDecodeError:
                pass
    return hits


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(HERE / "dist" / PACKAGE))
    ap.add_argument("--no-zip", action="store_true")
    a = ap.parse_args()
    out = Path(a.out)
    build(out)
    hits = gate(out)
    n = sum(1 for p in out.rglob("*") if p.is_file())
    size = sum(p.stat().st_size for p in out.rglob("*") if p.is_file())
    print(f"[build] {out}: {n} files, {size / 1e6:.1f} MB")
    if hits:
        print("[build] ANONYMITY GATE FAILED:")
        print("\n".join(hits))
        sys.exit(1)
    print("[build] anonymity gate passed (0 hits)")
    if not a.no_zip:
        z = out.with_suffix(".zip")
        with zipfile.ZipFile(z, "w", zipfile.ZIP_DEFLATED) as zf:
            for p in sorted(out.rglob("*")):
                if p.is_file():
                    zf.write(p, Path(PACKAGE) / p.relative_to(out))
        print(f"[build] wrote {z} ({z.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
