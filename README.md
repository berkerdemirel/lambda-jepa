# sslgap — Loss Space ≠ Representation Space

A two-space audit of self-supervised learning: do SSL methods impose their stated desiderata
(invariance, variance/covariance control, isotropy, non-collapse, redundancy reduction, predictive
sufficiency) on the **backbone representation `h`** that is probed and deployed — or on a
**projector/predictor/decoder output `z`** that is thrown away after training? The literature survey
behind this project ([docs/report/](docs/report/ssl-projector-gap-report.html)) found **0 of 11
surveyed methods** apply their loss to the representation they evaluate.

The ambition: a unifying critique + evaluation + explanation of what current SSL methods actually do
and how they differ — then synthesize criteria that close the gap, or an in-depth account of the
loss-space↔representation-space relationship.

**Start here:** [docs/ROADMAP.md](docs/ROADMAP.md) → then the layer you need:

| file | role |
|---|---|
| [docs/ROADMAP.md](docs/ROADMAP.md) | hierarchical roadmap: phases → milestones → experiments, with decision gates |
| [docs/PROTOCOL.md](docs/PROTOCOL.md) | the fixed experimental frame (versioned; changes need a DECISIONS row) |
| [docs/DECISIONS.md](docs/DECISIONS.md) | the ledger — every locked/proposed decision and every agreed takeaway |
| [docs/MODELS.md](docs/MODELS.md) | model-instance matrix: every checkpoint with provenance and validation status |
| [docs/WORKFLOW.md](docs/WORKFLOW.md) | how results become conclusions; operational constraints |
| [docs/METRICS.md](docs/METRICS.md) | what each battery metric measures, bounds, and caveats |
| [docs/experiments/](docs/experiments/) | E01–E11 pre-registration cards |
| [docs/methods/](docs/methods/) | per-method dossiers (desideratum, loss space, recipe, port notes) |
| [docs/theory/THEORY_MAP.md](docs/theory/THEORY_MAP.md) | which theorems bind which space |
| [docs/literature/BIBLIOGRAPHY.md](docs/literature/BIBLIOGRAPHY.md) | 102 tagged references |

## The two spaces

```
x, T(x) ──► backbone f ──► h ──► head g ──► z ──► SSL loss        (z is discarded after training)
                           │
                           └──► linear / kNN / attentive probe     (h is what everyone uses)
```

Core deliverable: the **audit matrix** — every metric of the battery computed identically at `h` and
`z` (and per head layer) for every method under a matched frame, with transfer ratios
τ(metric, method) = value(h)/value(z).

## Layout

- `sslgap/` — package: `ckpt` (uniform head-preserving checkpoint schema + legacy adapters),
  `models` (backbones/heads/taps), `extract` (feature store), `metrics` (the battery), `probes`
  (frozen protocols), `audit` (matrix assembly), `methods` (uniform per-method trainers, from M1).
- `experiments/` — Hydra entry points (`extract.py`, `audit.py`, `probe.py`, `report.py`, later `train.py`) + `configs/`.
- `slurm/` — `_env.sh` + sbatch templates. All compute goes through SLURM.
- `third_party/` — read-only reviewed donor clones (see `third_party/DONORS.md`); never imported.
- `outputs/` (ckpts, logs) · `features/` (feature store) · `results/` (small CSVs + FINDINGS, committed).

## Setup

```bash
~/.local/bin/uv sync        # .venv: Python 3.11, torch 2.11.0+cu130 (matches ssl_explore/lejepa)
```

Sibling repos this project reads: `../ssl_explore` (harvested metric/probe code, DINO IN-100 control
checkpoints), `../lejepa` (official LeJEPA minimal + trained Imagenette checkpoints).
