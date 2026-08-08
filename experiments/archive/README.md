# experiments/archive — spent one-shot scripts of closed experiments

Nothing here is deleted, and nothing here is broken — these scripts ran, produced numbers that
are on the cards, and are kept so any result can be traced to the code that made it. Cards cite
the original `experiments/<name>.py` paths **as-of-their-date**; that is intentional.

## Wave 2 — 2026-08-07 (D-083 housekeeping)

Everything ≤E23-era whose card is closed: **E17** (COMPLETE), **E19**, **E20**, **E21**, **E22**,
**E23** (all CLOSED with approved takeaways). 44 scripts.

**Three families were consolidated rather than merely moved.** Where a retired script's job
still exists, this is the map:

| retired | replaced by | why |
|---|---|---|
| `e12h_pull` · `e21_pull` · `e21_vmpull` · `e21_vm3pull` · `e21_vm4pull` · `e22_pull` · `e22_vm3pull` · `e22_vm4pull_1k` · `e23_pull` · `e24_pull` | **`experiments/pull.py`** | ten copy-and-tweak bridges (their docstrings literally chain "Original header follows"). The survivor, `e24_pull.py`, was generic **but did not warm the conditioner ring** — a D-072 violation, since a loaded checkpoint starts with an empty ring (`extras()` is not overridden) and the cold read inflates the conditioner's share. That is the artifact E24-T2 retracted. `pull.py` warms across a batch sequence, records `qfill`/`warm` per row, and reads the weight map from the method's own `PULL_W` instead of hardcoding it. |
| `e23_grid_metrics` · `e23c_grid_metrics` · `e24_grid_metrics` | **`experiments/grid_metrics.py`** | byte-identical apart from a tag list and an output prefix. The three tag sets survive as named presets (`e23_grid`, `e23c_grid`, `e24_toy`), so each landed grid re-runs with one word. |
| `e17_guillotine{,_full}` · `e20f_guillotine` · `e20_guillotine_zoo` · `e21_guillotine{,_vm}` · `e23_guillotine_{1k,grid,grid_oas}` | *(nothing yet)* | the newer five share one figure grammar by copy-paste; extracting the shared readers is queued, not done. The live drivers `e24_guillotine.py` and `e26_guillotine.py` still carry their own copies. |

Also retired: `e2x_zpred.py` — the h→z linear R² metric was **cancelled** by D-060 (a contractive
many-to-little head reads high R² while doing heavy nonlinear work, so the statistic measures
linear reachability of the output, not head magnitude). Its CSV stays as record.

**Deliberately not ported:** `e12h_pull.py`'s Hydra path, which built a method from `train.py`
overrides with no checkpoint in order to dose a config that did not exist yet. The in-training
share logger does that job better — launch a short pilot and read its `[share] ep1` line at a
real formation state (the E27 dose procedure).

**Note on two files:** `e23c_grid_metrics.py` and `e23_guillotine_grid_oas.py` were never
committed before being archived, so `git log --follow` will not trace them past this move.

## Wave 1 — 2026-07-20 (D-054 companion housekeeping)

Everything ≤E18-era whose card was closed. Live instruments stayed in `experiments/`.

## What is still live in `experiments/`

Generic drivers (`train` · `extract` · `probe` · `audit` · `compare` · `extract_orbits` ·
`adapter_selftest`) · the consolidated instruments (`pull` · `grid_metrics`) · the reusable
depth-metrics pass (`e20f_depth_metrics.py` — e20f-named but frame-generic, four wrappers used
it) · cross-experiment instruments (`e2x_classcos` · `e2x_posneg`) · open-card work
(`e24_*` — the voas landing is owed; `e26_*`/`pubzoo_*` — the zoo read is pending; `e27_*`;
`omega_*` — the theory review is pending; `transfer_*`) · `defect_rank_validate.py` (cited by
PROTOCOL §6.9) · `e12_class_align.py` · `e13_pivot_rung0.py` (PIVOT — scheduled for removal
with the rest of that lineage under D-083 Wave B).
