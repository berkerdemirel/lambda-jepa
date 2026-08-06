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


def class_margin(X, y):
    """Label-conditioned cosine margin: mean same-class cos − mean diff-class cos on
    L2-normalized features, via the class-sum identity (no pairwise materialization).
    The zoo-guillotine 'class margin (same−diff)' column — promoted from the spent
    e20_guillotine_zoo/e23_guillotine_1k copies per D-054 (reused by e24_guillotine)."""
    X = X.astype(np.float32)
    X = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-12)
    same_n, same_s, S = 0.0, 0.0, np.zeros(X.shape[1], np.float64)
    tot_c2 = 0.0
    counts = []
    for c in np.unique(y):
        xc = X[y == c]
        sc = xc.sum(0, dtype=np.float64)
        n = len(xc)
        same_s += sc @ sc - n
        same_n += n * (n - 1)
        S += sc
        tot_c2 += sc @ sc
        counts.append(n)
    N = len(X)
    same = same_s / same_n
    diff = (S @ S - tot_c2) / (N * N - sum(n ** 2 for n in counts))
    return float(same - diff)


def pair_margin(A, B, seed=0):
    """Positive-pair vs random-pair contrast within the same space — the baseline that makes
    alignment/cos_invariance interpretable (METRICS.md coupling caveat: a cone-collapsed space has
    tiny alignment with zero invariance achievement; M1 dress rehearsal, Berker 2026-07-08).
    Random pairs are cross-view different-image pairs (a[p_i] vs b[p_{i+1}]), so the view pipeline
    is identical for both terms and only image identity differs."""
    a = A / (np.linalg.norm(A, axis=1, keepdims=True) + 1e-12)
    b = B / (np.linalg.norm(B, axis=1, keepdims=True) + 1e-12)
    p = np.random.default_rng(seed).permutation(len(a))
    q = np.roll(p, 1)
    pos_cos, rand_cos = (a * b).sum(1).mean(), (a[p] * b[q]).sum(1).mean()
    align_pos = ((a - b) ** 2).sum(1).mean()
    align_rand = ((a[p] - b[q]) ** 2).sum(1).mean()
    return {"pos_cos": float(pos_cos), "rand_cos": float(rand_cos),
            "cos_margin": float(pos_cos - rand_cos),
            "align_pos": float(align_pos), "align_rand": float(align_rand),
            "align_rel": float(align_pos / (align_rand + 1e-12))}
