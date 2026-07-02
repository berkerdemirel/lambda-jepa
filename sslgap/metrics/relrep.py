"""Relative representations (D-009; Moschella et al., ICLR 2023): represent each sample by its
cosine similarities to anchors, giving different models a common frame on shared data.

Semantics verified against the authors' library latentis (Flegyas/latentis @ 800699f,
src/latentis/transform/projection.py): the canonical cosine projection is
`l2normalize(x) @ l2normalize(anchors).T` with **NO centering by default** — latentis'
RelativeProjection takes optional `abs_transform` pre-transforms (Centering / STDScaling /
StandardScaling), default Identity. We mirror that: `abs_transform` param, default "none"; any
non-default choice is part of the reported protocol.

Two anchor modes with different semantics (PROTOCOL §9):
- dataset anchors — A anchor IMAGES (fixed seeded ids per manifest, A = feature dim by convention),
  shared across models: relreps of different models are comparable coordinate-wise. The primary
  cross-model tool.
- random-orthonormal anchors — Q [d, A] orthonormal (seeded): relrep = l2(X) @ Q, a rotation of the
  normalized space. Does NOT align different models; used as a rotation-invariance control on
  metrics.
"""
import numpy as np


def _abs_transform(X, mode):
    """latentis-style optional pre-transform of the absolute space (default: none)."""
    if mode == "none":
        return X
    if mode == "center":                                  # latentis Centering
        return X - X.mean(0, keepdims=True)
    if mode == "standardize":                             # latentis StandardScaling
        return (X - X.mean(0, keepdims=True)) / (X.std(0, keepdims=True) + 1e-12)
    raise ValueError(f"unknown abs_transform {mode!r}")


def anchor_indices(n, A, seed=0):
    """Fixed anchor image ids for a manifest of size n (shared across models by construction)."""
    rng = np.random.default_rng(seed)
    return np.sort(rng.choice(n, size=min(A, n), replace=False))


def relrep_dataset(X, anchor_idx, abs_transform="none"):
    """Cosine similarities to the anchor rows of the SAME feature matrix -> [N, A].
    Matches latentis cosine_proj (l2-normalize both sides, dot); abs_transform per module doc."""
    X = _abs_transform(X, abs_transform)
    Xn = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-12)
    return Xn @ Xn[anchor_idx].T


def relrep_orthonormal(X, A=None, seed=0):
    """Rotation control: l2-normalized X projected on A random orthonormal directions."""
    d = X.shape[1]
    A = A or d
    rng = np.random.default_rng(seed)
    Q, _ = np.linalg.qr(rng.standard_normal((d, min(A, d))))
    Xn = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-12)
    return Xn @ Q


def cross_model_relrep(Xa, Xb, A=None, seed=0, abs_transform="none"):
    """The D-009 comparison: relreps of two models' features over the SAME manifest with the SAME
    anchor ids (A defaults to min of the two dims, matching 'anchors = feature dimensionality' as
    closely as a two-model comparison allows). Returns (Ra, Rb) ready for the cross battery
    (cka/neighbor-jaccard/procrustes now operate in one frame)."""
    if Xa.shape[0] != Xb.shape[0]:
        raise ValueError("relrep comparison needs the same manifest/order")
    A = A or min(Xa.shape[1], Xb.shape[1])
    idx = anchor_indices(Xa.shape[0], A, seed=seed)
    return (relrep_dataset(Xa, idx, abs_transform=abs_transform),
            relrep_dataset(Xb, idx, abs_transform=abs_transform))
