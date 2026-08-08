# the house method — dossier

**Naming.** The method is **`spectral`** under D-083 (Berker 2026-08-07: *"the name floorssl is
misleading. our class is spectralconditioner right?"*). **Nothing in it is floored:** the
anti-collapse term is the **two-sided spectral conditioner** (`SpectralConditioner`, née
MomentFloor, renamed under D-059) — `KL(N(μ,Σ)‖N(0,I))/d′` taxes the sliced covariance spectrum
on BOTH sides of identity (variance above 1 exactly like below it; the E21 2048-d collapse lived
on the cap side) plus the mean/cone term. The only genuinely one-sided object in the codebase is
`HingeFloor` (Σ ⪰ I), and it is vetoed as method identity (D-049).

**Identifiers still spell the old name** — the method key `floorssl`, run-ids
`*.floorssl.s0.*`, cfg keys `w_floor`/`z_floor*`/`h_floor_batch`/`h_lamb`, logged terms
`moment_kl`/`h_moment_kl`. Those are frozen while the E27 chains run (renaming the method key
restarts every run at epoch 0; renaming a cfg key kills the next segment). The code-identifier
rename lands at D-083 Wave B; see [../GLOSSARY.md](../GLOSSARY.md).

---

## What it is

**Three terms, one estimator:** invariance at z + the two-sided spectral conditioner as the SOLE
anti-collapse at z + the same conditioner at declared h at a few-percent dose. No negatives, no
EMA, no stop-grad, no var/cov pair, no predictor.

    L = w_inv·inv(z) + w_cond_z·cond(z) + w_cond_h·cond(h)

Identity D-046; lineage E12-f2 → E19 → E20 → E21 (own class) → E22 (IN-1k) → E23 (capacity) →
E24 (dose) → E27 (multi-arch). Code: `sslgap/methods/floorssl.py` + `SpectralConditioner` in
`sslgap/methods/_common.py`.

**Doses are never nominal.** E19-T1: a term's effect tracks its *realized share* of trunk pull,
so every transplant across a frame, an aug family, a dataset, or an architecture is re-bridged by
measurement. E24-T1 fixed the target in share coordinates: **s\* ≈ (.63, .34, .03)** for
(inv, conditioner-at-z, conditioner-at-h), set at formation and read as a trajectory thereafter.

## The operating anatomy (E21-T2, D-064; what every current cell runs)

Both conditioners read **per-image view means**, not the pooled batch. That was the correction
D-051 forced (pooled conditioning lets augmentation spread pay the conditioner, and it competes
with inv over the same quantity — the channel the 2048-d collapse lived in). View-mean payment
drops the estimator's fresh `n` from `bs·V` to `bs`, which is a real wall: keeping `n/d′ ≥ 4`
would have forced the slice down to d′ = 32.

**The ring removes the wall.** A detached buffer of the last `q` steps' conditioner inputs
concatenates with the fresh rows before the covariance read (`n_eff = (q+1)·bs`; gradient flows
through the fresh rows only), so d′ = 128 is restored at bs = 128. Slice widths scale with trunk
width across architectures — PROTOCOL §6 item 10 carries the table. Two consequences that bite:
the ring **re-warms over q steps after every resume**, and any share/Ω measurement on a queued
run must warm it first (D-072) — cold-ring reads were the "z-dominance" artifact of E24-T2.

## One training step (the E27 Recipe v2 configuration)

Batch: N = 128 images × V = 10 views (2 globals @224 + 8 locals @96, one symmetric photometric
family; only the crop geometry differs between groups). Trunk f = ViT-S/16 with pos-embed
interpolation for the locals. Head g = the ladder at depth 2 ≡ the legacy expander byte-for-byte:
Linear 384→2048 · BN · ReLU · Linear 2048→2048 · BN · ReLU · Linear 2048→256.

    # 1 · ENCODE every view through the one trunk
    h[i,v] = CLS( f(view v of image i) )       # N×V×384 — the DECLARED representation (D-036)
    z[i,v] = g(h[i,v])                         # N×V×256 — the loss-terminal space

    # 2 · INVARIANCE at z (the only attraction term)
    inv = mean_{i,v} ‖ z[i,v] − mean_v z[i,v] ‖²        # view-to-mean form
    # ≡ all-pairs MSE × (V−1)/2V at any fixed view set — the constant is absorbed by the
    # dose procedure, so this is not a change of objective. At V=2 it reduces exactly to
    # the lineage's pairwise MSE.
    # REPORTING TRAP (E27 §(d.2)): the constant is absorbed by DOSING but not by PLOTTING.
    # The multicrop lane logs W·(V−1)/V, the uniform-V lane logs 2W — a 2.22x gap at V=10
    # for identical geometry. And inv is an absolute squared distance in z, so it also
    # scales with z²: a cell whose conditioner is losing the scale logs a SMALL inv while
    # its clouds are getting THICKER. Never compare raw inv across lanes — convert to one
    # functional and normalise by the z scale, or plot the scale-free W/B instead.

    # 3 · THE CONDITIONER STREAM — view means at BOTH taps, the inv fixed point
    zbar[i] = mean_v z[i,v]            # n = N = 128 fresh rows
    hbar[i] = mean_v h[i,v]
    rows_z  = cat(zbar, ring_z)        # ring: last q steps, detached → n_eff = (q+1)·128
    rows_h  = cat(hbar, ring_h)

    # 4 · THE CONDITIONER (identical function at both taps)
    Q = QR(randn(D, 128)) orthonormal              # FRESH frame every step, unseeded
    p = rows @ Q[:, :d′]                           # sub-frame discipline keeps RNG aligned
    μ, Σ = mean(p), cov(p) + 1e-4·I                # fp32, autocast off (Cholesky)
    cond = (tr Σ + ‖μ‖² − d′ − log det Σ) / (2·d′) # = KL(N(μ,Σ) ‖ N(0,I)) / d′
    # TWO-SIDED: tr Σ taxes variance above 1, −log det taxes it below (→ ∞ at collapse,
    # the anti-degeneracy barrier), ‖μ‖² taxes the cone. Moments-only ⇒ cluster-blind BY
    # CONSTRUCTION; fresh slices cover all directions across steps. At output dim ≤ d′ the
    # conditioner reads the EXACT full covariance and slice noise vanishes.

    # 5 · TOTAL
    loss = w_inv·inv + w_cond_z·cond(rows_z) + w_cond_h·cond(rows_h)

Frame-side (not in the loss): a detached LayerNorm→Linear monitor probe co-trains on h. Its
reading is arm-biased by ~+3 pts toward conditioner arms (E12-T9(i), OPEN) — converged offline
probes are the arbiter, never the monitor.

## Variants on the shelf

- **`floor_shrink=oas`** (D-073) — the ring-free estimator: OAS shrinkage of the slice scatter
  toward its own scalar mean, ρ detached, m live, target `mI` never `I` so trace is preserved and
  scale error stays fully supervised. Policy by scale (E24-T3/D-075): stabilizer at toy, ring by
  default at IN-100/IN-1k, sanctioned fallback at ViT-B/L if the longer rings go stale.
- **`z_floor=hinge`** — `HingeFloor` (Σ ⪰ I). Diagnostic arm only; **vetoed as method identity**
  (D-049, Berker: "vicreg with slicing").
- **`head_layers` / `head_width`** — the capacity ladder from E23; depth 2 at `expander_hidden`
  is byte-identical to the legacy expander, so the ladder grows out of a certified-healthy cell
  in both directions. E23-T4 read the map: smooth, with one interior optimum.
- **`head_norm=none`** — the BN-free head (the E19-T1 conduit hypothesis at full depth). Kept as
  a fork; the z-conditioner's unit-variance demand replaces BN's scale pin at z but hidden layers
  carry no pin — declared risk.

## Deviations / open flags

- The monitor-probe arm bias (E12-T9(i)) is **OPEN**; reminder re-arms at E27 close.
- Two rejected head builders (`floorssl_res_head`, `floorssl_stage_head` — the E23 rev1/rev2
  ladders, 6/6 and 9/9 collapsed) remain in the module for checkpoint assembly only and are
  scheduled for removal under D-083 Wave B.
