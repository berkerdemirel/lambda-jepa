# GLOSSARY — the project's vocabulary

The single source for what this program's terms mean and which words it uses. Governed by
**D-083** (USER-APPROVED 2026-08-07: *"your proposed d083 is good. we also freeze the historical
run ids"*), which completes the clean break D-059 left open to Berker.

**The rule.** A term earns its place by being *literally true of the object* or by being
*standard in the literature*. Metaphors that were never cashed out are retired. Living
documents (this file, `PROTOCOL.md`, `METRICS.md`, `ROADMAP.md`, `MODELS.md`, `README.md`,
`HANDOVER.md`, `SESSION_OPENER.md`, method dossiers, `CLAUDE.md`), all code, and **all future
writing** use the current vocabulary. **Closed experiment cards and signed-off DECISIONS rows
are the record and stay verbatim** (D-059's own convention) — this file is their decoder.

---

## 1 · Renames

| retired | current | why |
|---|---|---|
| `floorssl` (the method) | **`spectral`** | nothing is floored: the loss is invariance at z plus a *two-sided* spectral conditioner at z and at h. The one-sided object is `HingeFloor`, and it is not the method. |
| `MomentFloor` (class) | **`SpectralConditioner`** | done under D-059 — the KL term taxes Σ deviations from I in **both** directions plus the cone. |
| `w_floor` · `h_lamb` | **`w_cond_z`** · **`w_cond_h`** | the loss then reads as its own formula: `w_inv·inv + w_cond_z·cond_z + w_cond_h·cond_h`. |
| `z_floor_batch` · `h_floor_batch` | **`cond_z_batch`** · **`cond_h_batch`** | same object, honest name. Values unchanged (`pooled` \| `view_mean`). |
| `moment_kl` · `h_moment_kl` (logged terms) | **`cond_z`** · **`cond_h`** | the wandb curve names should match the loss. |
| `orbit` (energies, dataset, stack, configs) | **cloud** | the object is the set of embeddings of one image's augmentations — a *cloud*, the theory doc's word since v2. Nothing is orbiting anything. |
| `orbit_energy.py` + `census.py` | **`sslgap/metrics/clouds.py`** | one subject, one module: cloud spread, center separation, touching. |
| `gauge` | *(plain wording)* | physics jargon for "unidentifiable up to a symmetry". Say that instead: "under BN, weight scale is partly unidentifiable". |
| `crater` · `conduit` · `firewall` · `knot` · `wall` | *(plain wording)* | narrative shorthand from closed experiments. They stay in the cards that coined them; new writing describes the thing. |

**Frozen strings (not vocabulary — data; RATIFIED by Berker 2026-08-07: "we also freeze the
historical run ids").** Run-ids minted before D-083 carry `floorssl`
(`toy.floorssl.s0.*`, `in1k.floorssl.s0.e27smc`, …), as do their checkpoints, feature stores,
result CSVs, and wandb runs; the augmentation-cloud store key is `o8`. These are identifiers
like a commit SHA — renaming them retroactively would make every card cite a path that never
existed. **The method is `spectral`; run-ids older than D-083 spell it `floorssl`.**

---

## 2 · The two spaces (normative definitions in `PROTOCOL.md` §1/§3)

- **h** — "the representation": the input of the method's projection/head (pre-MLP), per D-036.
  What is probed and deployed.
- **z** — "the proj": the space the training loss is applied to, plus every intermediate head
  tap `z.<role>.tapK`.
- **station** — one measured point along the depth axis, from trunk layers (`L03`, `L06`, `L09`)
  through `cls` into the head taps and `z.out`. A *guillotine curve* is a quantity read at every
  station.
- **guillotine** — cutting the head off at a station and reading the representation there.
  Literature-standard: Bordes, Balestriero & Vincent, *Guillotine Regularization* (2022).

## 3 · The cloud calculus (D-068; measured by `sslgap/metrics/`)

Embed one image's V augmentations → a **cloud** of V points; its **center** is their mean.

- **W** — within-cloud energy: mean squared distance between two views of the same image.
- **B** — between-center energy: mean squared distance between two images' centers, **debiased**
  by `W/V` for the sampling noise of the estimated center (exact; verified V-invariant to three
  digits over V ∈ {2, 8, 32}, which is why `o8` stores suffice).
- **Ω = W/B** — **thickness**: cloud size relative to image spacing. One number per space per
  checkpoint, label-free. The program's primary cross-space and cross-model quantity.
- **a, b, Λ** — transmission from a base station into a tap: `a` = how the within-cloud energy
  is scaled, `b` = how center separation is scaled, `Λ = √(Ω_h/Ω_z)`. Per D-068's usage rule,
  cross-space and cross-model reads ride on **Ω and Λ** (dimension-free); `a` and `b` are
  within-cell decompositions only.
- **touch** — two clouds touch when their radii sum exceeds their center distance.
  **ω(i,j)** = signed fractional overlap depth. **T** = touching fraction; **M** = median ω.
- **touch law** — `M = 1 − c/√Ω` with **c** the family's **shape constant** (measured c ≈ .82
  across 51 toy + 7 IN-100 + 3 IN-1k runs and the 6-model public zoo). Gives the
  **touching threshold** Ω\* = c², so a run's Ω has a calibrated position, not an arbitrary scale.
- **α-dial graph profile** — clouds linked when `α·(r_i + r_j) > d_ij`, α swept; the component
  census per α is the space's connectivity fingerprint.

## 4 · The force calculus (E24-T1; the dosing language)

- **pull** (`g`) — the trunk-only gradient norm a single loss term exerts, measured on a fixed
  batch at a held training state. Trunk-module-only by convention.
- **dose** (`w`) — a term's configured weight.
- **share** (`s`) — a term's *realized* fraction of total trunk pull, `w·g / Σ w·g`. The dose
  coordinate: E24-T1's recipe is stated in shares, s\* ≈ (.63, .34, .03) for
  (inv, conditioner-at-z, conditioner-at-h). Shares are **formation-phase targets** — mid-run
  equilibria drift, so a share is set at launch and read as a trajectory, never assumed held.
- **dose matching is init-only** — equal-pull transplants are certified at the *measured* state;
  a verbatim cross-frame transplant is the certified error class (E19-T1).
- **ring** — a detached ring buffer of the last *q* steps' conditioner inputs, widening the
  moment estimate without changing the slice width (cfg `queue_steps`). Gradient flows only
  through the current rows; the ring re-warms over *q* steps after a resume.
- **OAS** — Oracle Approximating Shrinkage of the slice scatter toward its own scalar mean,
  the ring-free estimator alternative (D-073).

## 5 · Frame words

- **frame** — the fixed experimental setting of a data-ladder rung (backbone, data, resolution,
  epoch budget, seeds). Enumerated in `PROTOCOL.md` §2.
- **lane** — one method's controlled line of runs within a frame.
- **cell** — a single run: one lane at one configuration.
- **landing / landing chain** — the post-training pipeline that turns a checkpoint into numbers:
  extract → probe → battery → instruments.
