# sslgap — Loss Space ≠ Representation Space

A two-space audit of self-supervised learning: do SSL methods impose their stated desiderata
(invariance, variance/covariance control, isotropy, non-collapse, redundancy reduction, predictive
sufficiency) on the **backbone representation `h`** that is probed and deployed — or on a
**projector/predictor/decoder output `z`** that is thrown away after training? The literature survey
behind this project ([docs/report/](docs/report/ssl-projector-gap-report.html)) found **0 of 11
surveyed methods** apply their loss to the representation they evaluate.

The ambition: a unifying critique + evaluation + explanation of what current SSL methods actually do
and how they differ — then synthesize criteria that close the gap.

## Where the program is

The audit is done; the work is now **constructive**. Measuring the gap produced two instruments —
a **dose law** (a loss term's effect tracks its realized share of trunk pull, not its nominal
weight) and a **cloud-thickness geometry** (Ω, calibrated by a touch law with one measured shape
constant c ≈ .82 that holds across every frame *and* across public backbones) — and those two
built a method: an invariance term at z plus a two-sided spectral conditioner at **both** z and h.
It is currently under test at ImageNet-1k across ViT-S/B/L. See
[docs/ROADMAP.md](docs/ROADMAP.md) §1 for the three eras and §3 for the full experiment index.

**Start here:** [docs/ROADMAP.md](docs/ROADMAP.md) → then the layer you need:

| file | role |
|---|---|
| [docs/ROADMAP.md](docs/ROADMAP.md) | question hierarchy → milestones → the E01–E27 index, with decision gates |
| [docs/PROTOCOL.md](docs/PROTOCOL.md) | the fixed experimental frame (versioned; changes need a DECISIONS row) |
| [docs/DECISIONS.md](docs/DECISIONS.md) | the ledger — every locked/proposed decision and every agreed takeaway |
| [docs/GLOSSARY.md](docs/GLOSSARY.md) | the project's vocabulary — what each term means and which words we use |
| [docs/METRICS.md](docs/METRICS.md) | what each battery metric measures, bounds, caveats — including the cloud calculus |
| [docs/WORKFLOW.md](docs/WORKFLOW.md) | how results become conclusions; operational constraints |
| [docs/HISTORY.md](docs/HISTORY.md) | append-only findings & lessons log (older sections in `HISTORY_ARCHIVE.md`) |
| [docs/MODELS.md](docs/MODELS.md) | the model-instance registry: tracks, provenance, validation status |
| [docs/experiments/](docs/experiments/) | E01–E27 experiment cards (pre-registration → numbers → agreed takeaway) |
| [docs/methods/](docs/methods/) | per-method dossiers (desideratum, loss space, recipe, port notes) |
| [docs/theory/OMEGA_CONNECTIVITY.md](docs/theory/OMEGA_CONNECTIVITY.md) | the live theory document (two cloud spaces, thickness, transport) |
| [docs/literature/BIBLIOGRAPHY.md](docs/literature/BIBLIOGRAPHY.md) | tagged references |

## The two spaces

```
x, T(x) ──► backbone f ──► h ──► head g ──► z ──► SSL loss        (z is discarded after training)
                           │
                           └──► linear / kNN / attentive probe     (h is what everyone uses)
```

`h` is **the input of the method's projection** (pre-MLP, D-036); `z` is the space the loss is
applied to. Everything between them is a measured *station*. The core audit deliverable is the
**audit matrix** — every battery metric computed identically at `h` and `z` (and per head layer)
for every method under a matched frame, with transfer ratios τ(metric, method) = value(h)/value(z).

## Layout

- `sslgap/` — the package: `methods` (uniform per-method trainers), `models` (backbones/heads/taps),
  `ckpt` (head-preserving checkpoint schema + legacy adapters), `extract` (feature store),
  `metrics` (the battery + the cloud calculus), `probes` (frozen protocols), `audit` (matrix assembly).
- `experiments/` — Hydra entry points: `train.py` · `extract.py` · `probe.py` · `audit.py` ·
  `compare.py`, plus per-experiment instruments, and `configs/`. Spent one-shots of closed
  experiments live in `experiments/archive/`.
- `slurm/` — `_env.sh` + sbatch wrappers. **All compute goes through SLURM**; the login node has no
  usable CUDA.
- `third_party/` — read-only reviewed donor clones (see `third_party/DONORS.md`); never imported.
- `outputs/` (checkpoints, logs) · `features/` (feature store) · `results/` (small CSVs, FINDINGS,
  figures — committed).

## Setup

```bash
~/.local/bin/uv sync        # .venv: Python 3.11, torch 2.11.0+cu130 (matches ssl_explore/lejepa)
```

Sibling repos this project reads: `../ssl_explore` (harvested metric/probe code, DINO IN-100 control
checkpoints), `../lejepa` (official LeJEPA minimal + trained Imagenette checkpoints).

## How work happens here

Pre-register predictions → run → numbers land **raw** on the card → discussion → an
`AGREED TAKEAWAY` written only jointly → a DECISIONS row. No conclusion is adopted, no protocol
changed, and no data-ladder rung advanced without a USER-APPROVED row. Failures are reported as
failures. See [docs/WORKFLOW.md](docs/WORKFLOW.md).
