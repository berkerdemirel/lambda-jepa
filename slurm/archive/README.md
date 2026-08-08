# slurm/archive — one-shot sbatch wrappers of closed experiments

Wrappers whose target scripts live in `experiments/archive/`. Generic parametric wrappers
(`train` · `extract` · `orbits` · `probe` · `audit` · `selftest` · `_env`) and current-program
wrappers stay in `slurm/`.

- **Wave 2 — 2026-08-07** (D-083 housekeeping): the E17/E19–E23 wrappers, 45 files, including
  the launch/landing shell drivers `e21_land.sh`, `e23_land_grid.sh`, `e23_wave_c2.sh` and the
  E22 one-shot `resume_vm4_1k.sh`. Two consolidated replacements now live in `slurm/`:
  **`pull.sbatch`** (was e12h/e21/e22/e23/e24 × pull) and **`grid_metrics.sbatch`** (was
  e23/e23c/e24 × grid_metrics). `e24_land.sh` stays live — the voas landing is still owed.
- **Wave 1 — 2026-07-20** (D-054 companion housekeeping): the ≤E18-era wrappers.

Nothing here is deleted. A wrapper in this directory still points at a real script; the
broken-reference sweep covers `archive/` too.
