"""Dense (scale, w_reg) grid at a few representative delta columns -- for a phase portrait with
w_reg as the y-axis (instead of delta) and scale on the x-axis, faceted by delta.

Deliberately EXCLUDES w_reg=0.0 and w_reg=0.1: run_grid_hreg_densegrid.py already swept those two
values fully densely across all 17 scales x all 17 deltas, so re-running them here would just
duplicate that job's work. The new w_reg values here (0.0005 to 0.05) fill in the gap between
that job's two w_reg rows, giving a genuinely denser w_reg axis when the two datasets are merged.

Grid: scales = logspace(-1, 0.3, 17) (same axis as run_grid_hreg_densegrid.py, for a consistent
x-axis across both sweeps) x w_reg in [0.0005, 0.001, 0.002, 0.005, 0.01, 0.02, 0.05] x delta in
[-1.0, -0.25, 0.0, 1.0] (the two interesting Rich-regime points -- -0.25 is the exact effrank
peak found in run_grid_hreg_densegrid.py's data -- plus the balanced and fully-upstream
endpoints) x 5 seeds = 17*7*4*5 = 2380 runs.

Some overlap with the ORIGINAL (pre-densegrid) sparse deltasign/deltapos sweeps is unavoidable
(they also tested w_reg in {0.001, 0.005, 0.02} at scale in {0.1, 10**0.3} for these same delta
values) -- append_csv will append a second, bit-identical copy of those specific rows; dedupe
with `df.drop_duplicates(subset=["run_id", "step"])` afterward, same as after the densegrid job.

Reuses get_results/OUTPUT_DIR from run_grid_hreg.py unchanged.
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
