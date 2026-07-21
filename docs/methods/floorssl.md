# floorssl — the house method (dossier)

**Naming (D-059, Berker 2026-07-21: "right now there is nothing floored in there. it is two
sided spectral conditioner"):** the method's anti-collapse term is the **two-sided spectral
conditioner** (`SpectralConditioner`, née MomentFloor) — KL(N(μ,Σ)‖N(0,I))/d′ taxes the
sliced covariance spectrum on BOTH sides of identity (variance above 1 exactly like below
it; the E21 2048-d collapse lived on the cap side) plus the mean/cone. The only genuinely
one-sided object in the codebase is `HingeFloor` (Σ⪰I), vetoed as method identity (D-049).
The strings `floorssl` / `w_floor` / `z_floor*` / `moment_kl` survive as FROZEN identifiers
(run-id provenance, stored-ckpt cfg keys, wandb history keys) — identifiers, not claims.

Three terms, one estimator: MSE alignment at z + the two-sided spectral conditioner as the
SOLE anti-collapse at z + the same conditioner at declared h at a ~6%-share dose. No
negatives, no EMA/stop-grad, no var/cov pair. Identity D-046; lineage E12-f2 → E19 → E20 →
E21/E22 (cards); doses always equal-pull-bridged, never nominal (E19-T1). Live cells:
IN-100 d256 view-mean (32.8/38.7/.617) · vm3 symmetric-payment (32.8/38.7/.339, D-058) ·
IN-1k d256vm (33.6/33.6/.587, E22). Code: `sslgap/methods/floorssl.py` +
`SpectralConditioner` in `sslgap/methods/_common.py`.

## One training step (pseudocode; the d256 view-mean configuration)

Batch: N=128 images × V=4 views (lejepa aug family). Trunk f = ViT-S/16; expander
g = Linear 384→2048 · BN · ReLU · Linear 2048→2048 · BN · ReLU · Linear 2048→256.

    # 1 · ENCODE every view
    h[i,v] = CLS( f(view v of image i) )      # (N·V)×384 — the DECLARED representation (D-036)
    z[i,v] = g(h[i,v])                        # N×V×256  — the loss-terminal space

    # 2 · INVARIANCE at z (the only attraction term)
    inv = mean_{i, u<w} MSE(z[i,u], z[i,w])   # all-pairs; V=2 reduces to plain pairwise MSE

    # 3 · Z-CONDITIONER — sole anti-collapse, view-mean payment (D-051)
    zbar[i] = mean_v z[i,v]                   # per-image view means → n = N = 128
    Q = QR(randn(256,128)) orthonormal        # FRESH frame every step, unseeded
    p = zbar @ Q[:, :32]                      # first-32 sub-frame (n/d′ = 4, co-design)
    μ, Σ = mean(p), cov(p) + 1e-4·I           # fp32, autocast off
    reg_z = (tr Σ + ‖μ‖² − 32 − log det Σ) / (2·32)      # = KL(N(μ,Σ) ‖ N(0,I)) / d′
    # TWO-SIDED: tr Σ taxes variance above 1, −log det taxes it below (→ ∞ at collapse —
    # the anti-degeneracy side), ‖μ‖² taxes the cone. Moments-only ⇒ cluster-blind by
    # construction; fresh slices cover all directions across steps. At expander_dim ≤ 128
    # the conditioner reads the EXACT full covariance (no slicing).

    # 4 · H-CONDITIONER — calibrated ε-dose at the representation
    reg_h = SpectralConditioner(h-input, d′ per payment mode)   # ~6% trunk-pull share
    # pooled mode: input = all N·V view CLS, d′=128 (vm2). view_mean mode (vm3, D-058):
    # input = per-image mean CLS, d′=32 — the same co-design as z, symmetric payment.

    # 5 · TOTAL
    loss = w_inv·inv + w_floor·reg_z + h_lamb·reg_h      # vm2: 32.8 · 38.7 · 0.617; vm3 h: 0.339

Payment axis: POOLED conditions z.reshape(N·V, 256) — per direction it reads across-image +
within-image (aug) variance, so aug spread pays the conditioner and competes with inv (the
2048-d collapse's channel). VIEW-MEAN conditions per-image means — the aug channel shrinks
by V; within-view scatter control rests on inv alone (declared risk P-vm-B). Narrow slices
are the first-d′ columns of the canonical 128-draw, keeping per-step RNG streams aligned
across variants. Frame-side (not in the loss): a detached LayerNorm→Linear monitor probe
co-trains on h.
