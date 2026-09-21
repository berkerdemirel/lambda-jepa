"""Dense (scale, delta) grid, matching run_grid.py's own resolution exactly, x w_reg in {0, 0.1}
-- built to reproduce a Kunin-Fig-1(b)-style 2D phase portrait (test loss / effective rank /
kernel distance as a function of (scale, delta), faceted by checkpoint time), for both the
unregularised control and the regularised (SpectralConditioner, w_reg=0.1) case, side by side.

Grid: scales = logspace(-1, 0.3, 17), deltas = linspace(-1, 1, 17) (exactly run_grid.py's own
axes) x w_reg in [0.0, 0.1] x 5 seeds = 17*17*2*5 = 2890 runs. Same n_iter=20_000 and
checkpoints_to_save as the rest of the run_grid_hreg*.py family (already validated against the
paper's own 1,000,000-step horizon in run_grid_paper_validation.py -- 20k is not undertrained).

Reuses get_results/OUTPUT_DIR from run_grid_hreg.py unchanged. Each grid point gets its own
results/Relu/<run_id>_diag.csv (keyed by run_id), so this cannot collide with any previous sweep.
"""
import multiprocessing
import os
from functools import partial

import numpy as np

from run_grid_hreg import get_results, OUTPUT_DIR

if __name__ == "__main__":
    # See run_grid_hreg.py: honor SLURM_CPUS_PER_TASK, cpu_count() ignores the job's allocation.
    n_cpus = int(os.environ.get("SLURM_CPUS_PER_TASK", multiprocessing.cpu_count()))
    print(f"We have access to {n_cpus} cores")

    scales = np.logspace(-1, 0.3, 17)
    deltas = np.linspace(-1, 1, 17)
    w_regs = [0.0, 0.1]
    train_seeds = np.arange(100, 105)

    m = 50
    n = 1_000
    d = 100
    n_iter = 20_000
    base_lr = 5e-3
    checkpoints_to_save = (0, 100, 1_000, 5_000, 10_000, 20_000)

    args = [(scale, delta, w_reg, seed, base_lr / scale ** 2)
            for scale in scales for delta in deltas for w_reg in w_regs for seed in train_seeds]

    n_machines_run = min(len(args), n_cpus)
    print(f"Total number of jobs is: {len(args)}. Running on {n_machines_run} cores")

    with multiprocessing.Pool(n_machines_run) as _p:
        out = _p.starmap(
            partial(get_results, n_iter=n_iter, n_samples=n, m=m, input_size=d,
                    checkpoints_to_save=checkpoints_to_save), args)

    print(f"Done with multiprocessing. Wrote {len(out)} run CSVs under {OUTPUT_DIR}")
