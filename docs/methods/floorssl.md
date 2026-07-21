# floorssl — the house method (dossier)

Three terms, one estimator: MSE alignment at z + a Gaussian-moment KL floor as the SOLE
anti-collapse at z + the same floor at the declared h as a ~6%-share conditioner. No
negatives, no EMA/stop-grad, no var/cov pair. Identity decision D-046 (independent class);
lineage E12-f2 → E19 → E20 → E21 (cards); doses always equal-pull-bridged, never nominal
(E19-T1). Live cells: IN-100 d256 view-mean (32.8/38.7/.617) · IN-1k d256vm (33.6/33.6/.587,
E22). Code: `sslgap/methods/floorssl.py` + `MomentFloor` in `sslgap/methods/_common.py`.

## One training step (pseudocode; the d256 view-mean configuration)

Batch: N=128 images × V=4 views (lejepa aug family). Trunk f = ViT-S/16; expander
g = Linear 384→2048 · BN · ReLU · Linear 2048→2048 · BN · ReLU · Linear 2048→256.

    # 1 · ENCODE every view
    h[i,v] = CLS( f(view v of image i) )      # (N·V)×384 — the DECLARED representation (D-036)
    z[i,v] = g(h[i,v])                        # N×V×256  — the loss-terminal space

    # 2 · INVARIANCE at z (the only attraction term)
    inv = mean_{i, u<w} MSE(z[i,u], z[i,w])   # all-pairs; V=2 reduces to plain pairwise MSE

    # 3 · Z-FLOOR — sole anti-collapse, view-mean payment (D-051)
    zbar[i] = mean_v z[i,v]                   # per-image view means → n = N = 128
    Q = QR(randn(256,128)) orthonormal        # FRESH frame every step, unseeded
    p = zbar @ Q[:, :32]                      # first-32 sub-frame (n/d′ = 4, co-design)
    μ, Σ = mean(p), cov(p) + 1e-4·I           # fp32, autocast off
    reg_z = (tr Σ + ‖μ‖² − 32 − log det Σ) / (2·32)      # = KL(N(μ,Σ) ‖ N(0,I)) / d′
    # moments-only ⇒ cluster-blind by construction; −log det = the collapse barrier (→∞);
    # fresh slices cover all directions across steps. At expander_dim ≤ 128 the floor
    # reads the EXACT full covariance (no slicing).

    # 4 · H-FLOOR — calibrated conditioner at the representation
    reg_h = MomentFloor(pooled CLS batch, d′=128)        # ~6% trunk-pull share (E19-T1/E20)

    # 5 · TOTAL
    loss = w_inv·inv + w_floor·reg_z + h_lamb·reg_h      # IN-100 vm2: 32.8 · 38.7 · 0.617

Payment axis: the POOLED variant floors z.reshape(N·V, 256) — per direction it reads
across-image + within-image (aug) variance, so aug spread pays the floor and competes with
inv (the 2048-d collapse's channel). VIEW-MEAN floors per-image means — the aug channel
shrinks by V; within-view scatter control rests on inv alone (declared risk P-vm-B). The
32-slice is the first 32 columns of the canonical 128-draw, keeping per-step RNG streams
aligned across variants. Frame-side (not in the loss): a detached LayerNorm→Linear monitor
probe co-trains on h.
