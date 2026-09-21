#!/bin/bash
#SBATCH --job-name=e34cf_smoke_gpuviews
#SBATCH --partition=gpu
#SBATCH --constraint=A100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=24
#SBATCH --mem=200G
#SBATCH --time=00:45:00
#SBATCH --output=/nfs/scistore19/locatgrp/bdemirel/ssl_project/outputs/%x_%j.out
# Smoke of the E34 second-launch data path on one card, before slurm/e34cf_shape_h100x8x1.slurm is queued:
# (1) data/gpu_views.photometrics on a real worker batch — dtype, range, determinism per seed, per-sample variation;
# (2) the trainer for 60 micro-batches with photometrics_on_gpu=true, span reads, 22 workers: the [share] ep0 line,
# the step-50 rate line, finite losses, checkpoints landing under the hydra run dir.
cd /nfs/scistore19/locatgrp/bdemirel/ssl_project/video/levjepa || exit 1
export LEVJEPA_DATA_ROOT=/mnt/beegfs/locatgrp/shared/datasets/kinetics HYDRA_FULL_ERROR=1
nvidia-smi -L | head -n 1
.venv/bin/python - <<'PY'
import time, torch
from hydra import compose, initialize
import main as M
from data.gpu_views import photometrics
with initialize(version_base=None, config_path="conf"):
    cfg = compose(config_name="lambdajepa_vits", overrides=["data=k710_clipfiles", "loader.num_workers=8", "loader.batch_size=16", "augmentation.photometrics_on_gpu=true"])
loader = M.build_video_loader(cfg, split="train", shuffle=True, hflip=True)
t0 = time.perf_counter(); b = next(iter(loader)); v = b["views"].cuda(); print(f"worker batch in {time.perf_counter() - t0:.0f}s: views {tuple(v.shape)} {v.dtype}", flush=True)
torch.cuda.synchronize(); t0 = time.perf_counter(); y = photometrics(v, seed=1); torch.cuda.synchronize(); dt = time.perf_counter() - t0
y2, y3 = photometrics(v, seed=1), photometrics(v, seed=2)
d = (y.float() - v.float()).abs().flatten(2).mean(-1)
print(f"gpu views {tuple(y.shape)} {y.dtype} range [{y.min().item()}, {y.max().item()}] in {dt:.2f}s | same seed identical: {torch.equal(y, y2)} | other seed differs: {not torch.equal(y, y3)} | per-sample mean |change| min/median/max {d.min().item():.1f}/{d.median().item():.1f}/{d.max().item():.1f} (0 = untouched sample)", flush=True)
print("SMOKE_VIEWS_OK", flush=True)
PY
.venv/bin/python -u main.py --config-name lambdajepa_vits data=k710_clipfiles \
  trainer.devices=1 trainer.num_nodes=1 accumulate_grad_batches=4 trainer.max_epochs=1 trainer.limit_train_batches=60 \
  loader.num_workers=22 augmentation.photometrics_on_gpu=true checkpoint.every_n_epochs=1 checkpoint.every_n_train_steps=0 \
  wandb.enabled=false hydra.run.dir=/mnt/beegfs/locatgrp/shared/bdemirel/levjepa_runs/smoke_gpuviews_$SLURM_JOB_ID 2>&1 | grep -v -E "UserWarning|warnings.warn|return func"
echo "SMOKE_TRAIN_EXIT rc=${PIPESTATUS[0]}"
ls -la /mnt/beegfs/locatgrp/shared/bdemirel/levjepa_runs/smoke_gpuviews_$SLURM_JOB_ID/spt_cache/runs/*/*/*/checkpoints/ 2>&1 | tail -n 3
