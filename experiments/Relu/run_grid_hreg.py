"""Does the repo's actual regulariser -- `SpectralConditioner` (sslgap/methods/_common.py, the
two-sided KL-to-isotropic-Gaussian "moment conditioner" behind FloorSSL's `w_floor`/`h_lamb`
terms) -- sustain h's rank under training when applied to this toy two-layer teacher-student
ReLU net, the way E28 (docs/experiments/E28_init_scale.md) found it does for a ViT backbone
(where raising init scale alone raises h's rank at step 0 but training erases it; only the
explicit conditioner keeps it up)?

Here the conditioner is applied directly on `h = relu(fc1(x))`, the toy net's one and only
hidden layer -- a placement neither production tap uses (`cond_h` sits pre-projector after a
LayerNorm; `cond_z` sits after the projector's final Linear, not immediately post-ReLU).

Reuses, does not reimplement:
  - `sslgap.methods._common.SpectralConditioner` -- the actual regulariser, constructed the
    same way production does for h: `d_slice=m, d_draw=max(128, m)` (`floorssl.py:145`).
  - `experiments.e28_scaleinit.spectrum_row` / `append_csv` -- the same rank-at-h metrics
    (effrank, stable_rank, rankme, pr, alpha) and append-only CSV convention as
    `results/e28/*_diag.csv`.
  - This folder's own `get_alpha`/`StudentNetwork`/`TeacherNetwork`/`neuron`/
    `get_kernel_trajectory`/`kernel_distance_from_initial`/`test_one_hidden_layer_relu`/`relu`.

Grid is intentionally small (see `__main__`): 5 `w_reg` values x 4 representative
`(scale, delta)` points x 5 seeds x 20k iterations -- NOT parity with `run_grid.py`'s
17x17x16-seed x 1M-iteration sweep, since toy-scale rank/kernel dynamics saturate fast and this
is meant to run locally in minutes. Adding the KL-conditioner term changes the loss landscape;
if a run's `train_loss`/rank metrics blow up or NaN, `w_reg`/`lr` need joint retuning -- this
is directly visible in the output CSV.

Output: one `results/Relu/<run_id>_diag.csv` per grid point (own file per run, so parallel
workers never contend on the same file), one row per checkpoint. Also logged live to wandb
(project matches the rest of the repo's runs, `experiments/train.py`/`e28_scaleinit.py`; no
explicit `entity` -- same as those scripts, so it uses whichever account/team is logged in;
`group="relu-hreg"` keeps these toy sweep runs distinguishable from real training runs). Set
`WANDB_MODE=offline` (or `disabled`) in the environment to skip network calls, e.g. for a
local smoke test.
"""
import multiprocessing
import os
import time
from functools import partial

os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import wandb

from experiments.e28_scaleinit import append_csv, spectrum_row
from sslgap.methods._common import SpectralConditioner
from utils import (
    StudentNetwork, TeacherNetwork, get_alpha, get_kernel_trajectory,
    kernel_distance_from_initial, neuron, relu, test_one_hidden_layer_relu, train)

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUTPUT_DIR = os.path.join(REPO_ROOT, "results", "Relu")
WANDB_PROJECT = "sslgap"


def get_results(scale, delta, w_reg, seed, lr, n_iter, n_samples, m, input_size,
                checkpoints_to_save, test_seed=201, n_test=10_000):
    run_id = f"relu.hreg.scale{scale:.4g}.delta{delta:.4g}.wreg{w_reg:.4g}.s{seed}"
    run = wandb.init(
        project=WANDB_PROJECT, name=run_id, group="relu-hreg", job_type="hreg-sweep",
        config={"scale": scale, "delta": delta, "w_reg": w_reg, "seed": seed, "lr": lr,
                "n_iter": n_iter, "n_samples": n_samples, "m": m, "input_size": input_size})

    torch.manual_seed(seed)
    m0 = 3
    teacher = TeacherNetwork(input_size, m0)

    alpha = get_alpha(delta, scale)
    student = StudentNetwork(input_size, m, scale, alpha=alpha, symmetrize=True)

    inputs = torch.randn(n_samples, input_size)
    inputs = inputs / torch.norm(inputs, dim=1, keepdim=True)
    with torch.no_grad():
        labels = teacher(inputs)

    criterion = nn.MSELoss()
    optimizer = optim.SGD(student.parameters(), lr=lr)
    # w_reg != 0 (not just > 0): a negative w_reg is a valid, separate experiment -- it flips the
    # conditioner's loss sign, actively rewarding deviation from isotropy instead of penalizing
    # it, rather than being silently ignored the way `> 0` would leave it.
    conditioner = SpectralConditioner(d_slice=m, d_draw=max(128, m)) if w_reg != 0 else None

    t0 = time.time()
    if conditioner is None:
        trajectory, losses, _preds = train(
            student, criterion, optimizer, inputs, labels, n_iter=n_iter,
            checkpoints_to_save=checkpoints_to_save)
        reg_losses = [float("nan")] * len(losses)
    else:
        trajectory, losses, _preds, reg_losses = train(
            student, criterion, optimizer, inputs, labels, n_iter=n_iter,
            checkpoints_to_save=checkpoints_to_save, conditioner=conditioner, w_reg=w_reg)
    print(f"[scale={scale:.3g} delta={delta:.3g} w_reg={w_reg:.3g} seed={seed}] "
          f"training took {time.time() - t0:.1f}s")

    Ws, as_ = neuron(trajectory, input_size, m)
    Ws, as_ = np.float64(Ws), np.float64(as_)

    K = get_kernel_trajectory(Ws, as_, np.float64(inputs.detach().clone().numpy()), mode=None)
    kernel_distance = kernel_distance_from_initial(K)  # first checkpoint should be ~0

    # Fixed held-out test set: identical construction/seed to test_one_hidden_layer_relu's own
    # internal sampling, so the two calls below see the same X_test (that function reseeds
    # `test_seed` itself every call, so this is bit-identical, not just distributionally so).
    torch.manual_seed(test_seed)
    test_inputs = torch.randn(n_test, input_size)
    test_inputs = test_inputs / torch.norm(test_inputs, dim=1, keepdim=True)
    X_test = np.float64(test_inputs.detach().numpy())

    rows = []
    for _ci, step in enumerate(checkpoints_to_save):
        test_loss = test_one_hidden_layer_relu(Ws[_ci], as_[_ci], teacher, n_test=n_test, test_seed=test_seed)
        h_test = relu(X_test @ Ws[_ci].T)  # post-nonlinearity hidden activations on the fixed test set
        spec, _eigs = spectrum_row(h_test)  # effrank, stable_rank, rankme, pr, alpha, ... at h

        row = {
            "run_id": run_id, "scale": scale, "delta": delta, "w_reg": w_reg, "seed": seed,
            "step": step, "train_loss": losses[_ci], "reg_loss": reg_losses[_ci],
            "test_loss": float(test_loss), "kernel_distance": float(kernel_distance[_ci]),
        }
        row.update(spec)
        rows.append(row)

        # Mirrors experiments/e28_scaleinit.py's run_diagnostics logging convention
        # (diag/<space>/<metric>, n/d excluded since they're constant per run).
        wandb.log({
            "train/loss": row["train_loss"], "train/reg_loss": row["reg_loss"],
            "test/loss": row["test_loss"], "kernel/distance": row["kernel_distance"],
            **{f"diag/h/{k}": v for k, v in spec.items() if k not in ("n", "d")},
        }, step=step)

    append_csv(os.path.join(OUTPUT_DIR, f"{run_id}_diag.csv"), rows)
    try:
        wandb.finish()
    except OSError:
        # Observed under high worker concurrency (32-way multiprocessing.Pool writing to a
        # shared NFS-mounted wandb/ run directory): wandb's cleanup can hit a transient "Stale
        # file handle" here well after this run's own data is already durably written above
        # (append_csv is a separate, already-completed plain file write) -- letting it propagate
        # would fail the whole SLURM job (and every run still queued behind it in the pool) over
        # what is purely a wandb-cleanup race, not a lost result.
        print(f"[{run_id}] wandb.finish() hit an OSError (likely NFS stale-handle cleanup race); "
              f"result CSV was already written, continuing.")
    return run_id


if __name__ == "__main__":
    # multiprocessing.cpu_count() reports the node's full core count, not this job's SLURM
    # allocation (observed: 256 vs. an actual --cpus-per-task=32) -- honor SLURM_CPUS_PER_TASK
    # when set, matching the convention in experiments/train.py/e28_scaleinit.py.
    n_cpus = int(os.environ.get("SLURM_CPUS_PER_TASK", multiprocessing.cpu_count()))
    print(f"We have access to {n_cpus} cores")

    # 4 representative (scale, delta) points spanning run_grid.py's grid range, rather than its
    # full 17x17 -- "lazy/balanced", "rich/balanced", "imbalanced" and "rich/imbalanced" (Chizat
    # lazy/rich vocabulary).
    scale_delta_points = [
        (10 ** 0.3, 0.0),
        (10 ** -1, 0.0),
        (1.0, 1.0),
        (10 ** -1, 1.0),
    ]
    # 0.0 is the unregularized control; 0.02 matches the "interior optimum near lambda~.02 at h"
    # called out in SpectralConditioner's own docstring (sslgap/methods/_common.py).
    w_regs = [0.0, 0.001, 0.005, 0.02, 0.1]
    train_seeds = np.arange(100, 105)

    m = 50
    n = 1_000
    d = 100
    n_iter = 20_000
    base_lr = 5e-3
    checkpoints_to_save = (0, 100, 1_000, 5_000, 10_000, 20_000)

    args = []
    for scale, delta in scale_delta_points:
        for w_reg in w_regs:
            for seed in train_seeds:
                args.append((scale, delta, w_reg, seed, base_lr / scale ** 2))

    n_machines_run = min(len(args), n_cpus)
    print(f"Total number of jobs is: {len(args)}. Running on {n_machines_run} cores")

    with multiprocessing.Pool(n_machines_run) as _p:
        out = _p.starmap(
            partial(get_results, n_iter=n_iter, n_samples=n, m=m, input_size=d,
                    checkpoints_to_save=checkpoints_to_save), args)

    print(f"Done with multiprocessing. Wrote {len(out)} run CSVs under {OUTPUT_DIR}")
