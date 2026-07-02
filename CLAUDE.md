# CLAUDE.md

Guidance for Claude Code sessions in this repository.

## What this is

**sslgap** — a two-space audit of SSL methods: measure every method's stated desideratum at the
backbone representation `h` AND at the head/loss space `z` (plus intermediate head layers), under a
matched frame. Study design: `docs/report/ssl-projector-gap-report.html` (the founding literature
synthesis). **Read first:** `HANDOVER.md` (thin state) → `docs/ROADMAP.md` (phases/milestones) →
`PROTOCOL.md` (fixed frame) → `DECISIONS.md` (ledger).

Doc layering (keep separate when writing): concepts/justification → `docs/` dossiers & cards;
runnable protocol/params → `PROTOCOL.md` + configs; outcomes/lessons → `docs/HISTORY.md`
(append-only, never rewrite); `HANDOVER.md` stays thin (overwrite freely).

## The collaboration contract (NON-NEGOTIABLE)

The user (Berker) is a co-investigator, not an audience. SSL methods are heavily hand-tuned;
conclusions here need unusual care.

1. **Never fill an `AGREED TAKEAWAY` section unilaterally.** Experiment cards
   (`docs/experiments/E*.md`) receive numbers and plots; interpretation happens in discussion with
   the user, and only the jointly agreed version is written, then mirrored to `DECISIONS.md`.
2. Every experiment is **pre-registered**: directional predictions are committed to the card
   *before* the numbers exist. Predictions come from the report's §3.1 matrix or are discussed.
3. `PROTOCOL.md` is versioned; **any change to the frame, probes, or estimator settings requires a
   DECISIONS row** (status PROPOSED until user-approved).
4. Advancing a rung of the data ladder (toy → IN-100 → public IN-1k → IN-1k retrain) requires a
   USER-APPROVED gate row in `DECISIONS.md`.
5. Report numbers faithfully: failed runs are failures, parity misses are misses. Never smooth over
   a discrepancy to keep a milestone green.

## Commands

```bash
~/.local/bin/uv sync              # setup (.venv, torch 2.11.0+cu130; uv is not on PATH)
sbatch slurm/extract.sbatch ckpt=... frame=...      # feature extraction (GPU job)
sbatch slurm/audit.sbatch  run_id=...               # metric battery (CPU-heavy, small GPU ok)
sbatch slurm/probe.sbatch  run_id=... probe=...     # probes
squeue -u bdemirel                                  # job status; logs in outputs/<job>_<id>.out
```

**All compute goes through SLURM** — the login node has no usable CUDA and 2 overloaded CPUs. Even
"quick" feature extraction gets a GPU job. H100s: `--partition=gpu100 --constraint=H100`, and we cap
ourselves at **2 concurrent H100 jobs** (use the two singleton job names `h100-slotA`/`h100-slotB`
with `--dependency=singleton`). Everything else: `--partition=gpu` (A100/A40/L40S/3090).
Smoke-test with subset overrides before any sweep — in the predecessor project, setup was mistaken
for results four times.

## Hard-won constraints (carried from ssl_explore — do not relearn)

- **Training hygiene for any new end-to-end loss:** `grad_clip=1.0` from day one; cosine
  `eta_min ≤ lr/20`; warmup ≥ 1 epoch; per-audit checkpoints (`ep25/50/75/100`+best+last — an
  every-epoch overwrite once lost the best model); per-step `grad_norm` + per-term loss logging from
  birth. Kill-trigger: a single-step grad-norm > 100× running median is an INCIDENT — stop and read.
- **Sliced-Gaussianity (Epps–Pulley over random projections) is foolable** — decorrelation + CLT
  makes random slices look Gaussian. Isotropy claims lead with `kurt_topeig` / worst-direction
  stats; EP is never reported alone (encoded in the battery).
- **DataLoader hygiene:** `persistent_workers=True`; never create throwaway iterators
  (`next(iter(loader))` per checkpoint) — caused a CPU-side deadlock with GPU at 0%.
- **bs ≤ 128** for ViT-S/8@128 V=4 on a 24 GB card; H100 fits bs=128 @224 V=10.
- **wandb is always ONLINE** (entity `causal-learning-ai-ista`, project `sslgap`); log inspectable
  artifacts — the user watches live to catch bugs.
- HF datasets run **offline** on compute nodes (`HF_HUB_OFFLINE=1` in `slurm/_env.sh`); Imagenette
  is cached under `~/.cache/huggingface`.

## Project-specific constraints (new — equally binding)

- **Canonical ImageNet-100 = `~/data/imagenet100`** (CMC split). `~/data/imagenet-100` is a
  DIFFERENT 100-class subset (8% overlap) — never use it here (DECISIONS D-002).
- **`h` ≡ trunk `forward_features` output** (CLS and patch-GAP, per-layer taps). Anything trainable
  after the trunk is head: the lejepa-style `Linear(384→512)` "embedding" is tap `z.embed`, not `h`
  (D-003).
- **Never mix checkpoint provenance within a comparison** (controlled retrains vs public
  checkpoints are different tracks; public DINOv2 has no heads → no z-space claims for it).
- **Feature store budget** (D-005): fp16; nothing >8192-d stored (DINO prototype logits are
  recomputed from the bottleneck when needed); patch tokens only for final-layer/final-epoch/
  probed-branch; purge-after-table. `store.py` enforces a cap.
- Donor code (`third_party/`) is read-only reference. Ports go through review, get a `PORT_NOTES`
  section in the method dossier (donor commit, findings, deviations), and land in `sslgap/` in our
  style.

## Style

Compact, flat, research-grade. Module docstrings state the experimental *why*; comments only where
the reason isn't visible in code. No defensive try/except, no abstraction layers beyond what is
reused. New experiments follow: an `sslgap/` building block → a Hydra entry point in `experiments/`
→ a thin sbatch wrapper in `slurm/`. Metrics are pure functions over stored arrays (model never in
memory during audit). Every checkpoint carries `format/arch/provenance` per `sslgap/ckpt/schema.py`.
