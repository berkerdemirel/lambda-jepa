# LDM identifiability × the head — problem note (OP-17)

**Status: working draft 2026-07-02 — discussion material, not results. Tags [P]/[O] per repo
convention; nothing here is proven.** Companion to THEORY_MAP.md §"Where the two anchors sit";
the open problem named there and in OPEN_PROBLEMS.md OP-17.

## Setup

LDM (Mikulasch & Zenke '26): minimizers of F_LDM recover the latent-model latents `s` up to an
affine map, *at the loss layer*: `z ≅ A·s + b`. Heads are not discussed (verified absence).
Practice trains `z = g(h)`, `h = f(x)`, probes `h`. Question: **what identifiability statement, in
which transformation class, does `h` inherit?**

## The composition problem, stated sharply

1. **Sufficiency survives trivially.** `z = g(h)` ⇒ `I(h;s) ≥ I(z;s)` (data processing). If z
   identifies s, h determines s. This is content, not geometry — the weak half, already [P] in the
   report.
2. **The affine class does not survive in general.** Identifiability-up-to-affine is a statement
   about *how* s is arranged. Any invertible nonlinear reparametrization of h preserves its
   information while destroying affine readability; nothing in F_LDM constrains f below g. So the
   naive conjecture "h ≅ (s, nuisance) up to affine" is FALSE without extra assumptions — the
   honest general claim is only "h ≅ some invertible function of (s, n)". The class gap
   [affine at z] vs [merely-invertible at h] IS the head's formal contribution. [P]
3. **The linear-head special case is exactly solvable and is the wedge.** If g is linear
   (`z = W·h`), then `A·s + b = W·h`: the projection of h onto row(W) remains **affinely**
   s-readable; only the null(W) component is unconstrained. So a linear head yields a structured
   decomposition h = [affine-s subspace] ⊕ [free complement carrying nuisance]. A nonlinear head
   destroys even this. Note the connection to OP-4 (the nonlinearity mystery): linear heads
   preserve partial affine identifiability yet lose ~3% downstream (SimCLR); if affine
   s-readability at h were what matters, linear heads should win. Either (a) it isn't what
   matters, or (b) trained nonlinear heads are *effectively linear on the data manifold*, keeping
   affine readability emergently. (a)/(b) are distinguishable by measurement — below. [O]

## Measurable shadows (all computable from features already in the store)

- **Head-linearity index (NEW metric candidate, PROPOSED — needs a DECISIONS row before entering
  the battery):** R² of ridge regression h → z on the m50k manifest (both spaces stored on the
  same images; zero new extraction). Near-1 for a trained nonlinear head = hypothesis (b): the
  head's used capacity is ~linear ⇒ LDM's affine class approximately composes through it.
  Complement: R² of z → h (how much of h the head discards — the null(W) analogue).
- **Linear-vs-MLP probe gap, per space (E11 grid, already planned):** affine identifiability at z
  predicts linear probes suffice there for latent-aligned readout; at h only invertible-class
  identifiability is guaranteed, so the MLP-over-linear gain should be strictly larger at h than
  at z, per method. A clean, pre-registrable sign prediction for E11. [P]
- **Nuisance side (E4 ledger):** "h ≅ (s, n)" needs the n inventory — augmentation-parameter
  decodability at h vs z quantifies which nuisances the head's null space actually carries.
- **E10 tie-in:** depth-0/linear-head arms make the special case (3) a *trained* condition rather
  than an assumption — the only place the affine-composition claim is directly testable end-to-end.

## What a theorem would need (proof-shape sketch, [O])

Claim shape: *if* z identifies s up to affine AND (i) g is L-Lipschitz with trained effective rank
r ≈ dim(z) on the data manifold, (ii) h's distribution satisfies a regularity condition (e.g. the
SIGReg-style isotropy E1 measures), *then* h identifies (s, n) up to a map whose deviation from
affine is bounded by the head-linearity deficit (1 − R²). The needed lemma is a quantitative
"approximate-affine inverse on a manifold" statement — plausibly adaptable from the
nonlinear-ICA/identifiability toolbox (LDM's own appendix machinery). Our E1/E4 numbers supply (i)
and (ii) empirically per method; if the head-linearity index is high across trained methods, the
theorem's premise is *typical*, which is what makes it worth proving rather than a vacuous
special case.

## Discussion queue for Berker

1. Adopt the head-linearity index into the battery (D-row needed; trivially cheap, pure function
   over stored arrays)?
2. Add the E11 sign prediction (MLP-over-linear gain larger at h than z) to the E11 PROPOSED
   list before any M2 number exists?
3. Is the theorem sketch worth a collaboration/outreach to the LDM authors post-M2, with our
   measurements as the empirical section?
