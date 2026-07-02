"""Metrics on positive-pair feature matrices (A, B) [N, d] — two views of the same images,
extracted under a declared aug stack (own vs audit_v1; PROTOCOL §5)."""
import numpy as np


def alignment(A, B):
    """Wang–Isola alignment: E ||a - b||^2 on L2-normalized features. Lower = more view-invariant."""
    a = A / (np.linalg.norm(A, axis=1, keepdims=True) + 1e-12)
    b = B / (np.linalg.norm(B, axis=1, keepdims=True) + 1e-12)
    return float(((a - b) ** 2).sum(1).mean())


def cos_invariance(A, B):
    """Mean cosine similarity of raw positive pairs (the RCDM-style invariance readout)."""
    num = (A * B).sum(1)
    den = np.linalg.norm(A, axis=1) * np.linalg.norm(B, axis=1) + 1e-12
    return float((num / den).mean())
