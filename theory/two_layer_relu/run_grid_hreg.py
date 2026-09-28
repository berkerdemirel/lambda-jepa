"""Two-layer ReLU teacher-student regression (Kunin et al. 2021 setup) with w_reg * SACReg(h) on the post-ReLU hidden layer: sweep over initialization scale, layer imbalance, w_reg and seeds; one CSV per run with test loss, kernel distance from initialization and the rank measures of h at each checkpoint (results/Relu/)."""
import multiprocessing
import os
import time
from functools import partial

os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import wandb

from sslgap.methods._common import SACReg
from sslgap.metrics.spectra import spectrum_row
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
    conditioner = SACReg(d_slice=m, d_draw=max(128, m)) if w_reg != 0 else None

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
    kernel_distance = kernel_distance_from_initial(K)

    torch.manual_seed(test_seed)
    test_inputs = torch.randn(n_test, input_size)
    test_inputs = test_inputs / torch.norm(test_inputs, dim=1, keepdim=True)
    X_test = np.float64(test_inputs.detach().numpy())

    rows = []
    for _ci, step in enumerate(checkpoints_to_save):
        test_loss = test_one_hidden_layer_relu(Ws[_ci], as_[_ci], teacher, n_test=n_test, test_seed=test_seed)
        h_test = relu(X_test @ Ws[_ci].T)
        spec, _eigs = spectrum_row(h_test)

        row = {
            "run_id": run_id, "scale": scale, "delta": delta, "w_reg": w_reg, "seed": seed,
            "step": step, "train_loss": losses[_ci], "reg_loss": reg_losses[_ci],
            "test_loss": float(test_loss), "kernel_distance": float(kernel_distance[_ci]),
        }
        row.update(spec)
        rows.append(row)

        wandb.log({
            "train/loss": row["train_loss"], "train/reg_loss": row["reg_loss"],
            "test/loss": row["test_loss"], "kernel/distance": row["kernel_distance"],
            **{f"diag/h/{k}": v for k, v in spec.items() if k not in ("n", "d")},
        }, step=step)

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    path = os.path.join(OUTPUT_DIR, f"{run_id}_diag.csv")
    pd.DataFrame(rows).to_csv(path, mode="a", header=not os.path.exists(path), index=False, na_rep="nan")
    try:
        wandb.finish()
    except OSError:
        print(f"[{run_id}] wandb.finish() hit an OSError (likely NFS stale-handle cleanup race); "
              f"result CSV was already written, continuing.")
    return run_id

if __name__ == "__main__":
    n_cpus = int(os.environ.get("SLURM_CPUS_PER_TASK", multiprocessing.cpu_count()))
    print(f"We have access to {n_cpus} cores")

    scale_delta_points = [
        (10 ** 0.3, 0.0),
        (10 ** -1, 0.0),
        (1.0, 1.0),
        (10 ** -1, 1.0),
    ]
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
