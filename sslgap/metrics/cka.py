"""Feature/kernel drift similarity (E33 rich-vs-lazy diagnostic).

linear_cka: Kornblith et al. 2019 linear CKA between two feature matrices, computed from
D×D cross-moments (no N×N Gram needed). frob_alignment: ⟨K1,K2⟩_F/(‖K1‖_F‖K2‖_F), the
empirical-NTK alignment of the E33 spec; centered=True double-centers both kernels first
(Cortes et al. 2012 centered kernel alignment — reported alongside the plain cosine because
the uncentered mean component can dominate ViT kernels).
"""
import numpy as np


def linear_cka(X, Y):
    X = np.asarray(X, np.float64)
    Y = np.asarray(Y, np.float64)
    Xc = X - X.mean(0)
    Yc = Y - Y.mean(0)
    num = np.linalg.norm(Xc.T @ Yc) ** 2
    den = np.linalg.norm(Xc.T @ Xc) * np.linalg.norm(Yc.T @ Yc)
    return float(num / den)


def frob_alignment(K1, K2, centered=False):
    K1 = np.asarray(K1, np.float64)
    K2 = np.asarray(K2, np.float64)
    if centered:
        n = K1.shape[0]
        H = np.eye(n) - 1.0 / n
        K1 = H @ K1 @ H
        K2 = H @ K2 @ H
    return float((K1 * K2).sum() / (np.linalg.norm(K1) * np.linalg.norm(K2)))
