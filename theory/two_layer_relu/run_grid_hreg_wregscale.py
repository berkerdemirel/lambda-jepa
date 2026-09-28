"""The dense (scale, w_reg) grid at four imbalance values: the regularizer-strength axis of the appendix figure."""
import multiprocessing
import os
from functools import partial

import numpy as np

from run_grid_hreg import get_results, OUTPUT_DIR

if __name__ == "__main__":
    n_cpus = int(os.environ.get("SLURM_CPUS_PER_TASK", multiprocessing.cpu_count()))
    print(f"We have access to {n_cpus} cores")

    scales = np.logspace(-1, 0.3, 17)
    w_regs = [0.0005, 0.001, 0.002, 0.005, 0.01, 0.02, 0.05]
    deltas = [-1.0, -0.25, 0.0, 1.0]
    train_seeds = np.arange(100, 105)

    m = 50
    n = 1_000
    d = 100
    n_iter = 20_000
    base_lr = 5e-3
    checkpoints_to_save = (0, 100, 1_000, 5_000, 10_000, 20_000)

    args = [(scale, delta, w_reg, seed, base_lr / scale ** 2)
            for scale in scales for w_reg in w_regs for delta in deltas for seed in train_seeds]

    n_machines_run = min(len(args), n_cpus)
    print(f"Total number of jobs is: {len(args)}. Running on {n_machines_run} cores")

    with multiprocessing.Pool(n_machines_run) as _p:
        out = _p.starmap(
            partial(get_results, n_iter=n_iter, n_samples=n, m=m, input_size=d,
                    checkpoints_to_save=checkpoints_to_save), args)

    print(f"Done with multiprocessing. Wrote {len(out)} run CSVs under {OUTPUT_DIR}")
