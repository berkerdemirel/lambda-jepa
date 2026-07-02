"""Cross-space metrics: how differently do two spaces (h vs z, same images) arrange the data?
CKA is CONTESTED for cross-representation correspondence (refuted in the report's adversarial
verification) — it is only ever reported as the triple (CKA, neighbor-Jaccard, Procrustes)."""
import numpy as np
from scipy.linalg import orthogonal_procrustes
from sklearn.neighbors import NearestNeighbors


def cka_linear(X, Y):
    """Biased linear CKA on centered features, via cross-gram (feasible at N=50k)."""
    Xc = X - X.mean(0)
    Yc = Y - Y.mean(0)
    xy = np.linalg.norm(Xc.T @ Yc) ** 2
    xx = np.linalg.norm(Xc.T @ Xc)
    yy = np.linalg.norm(Yc.T @ Yc)
    return float(xy / (xx * yy + 1e-12))


def neighbor_jaccard(X, Y, k=10, n_sub=5000, seed=0):
    """Mean Jaccard overlap of k-NN sets computed in each space on the SAME image subsample —
    the direct 'does the head reorder semantic neighborhoods' readout."""
    rng = np.random.default_rng(seed)
    idx = rng.choice(X.shape[0], min(n_sub, X.shape[0]), replace=False)
    Xs, Ys = X[idx], Y[idx]
    nx = NearestNeighbors(n_neighbors=k + 1).fit(Xs).kneighbors(Xs, return_distance=False)[:, 1:]
    ny = NearestNeighbors(n_neighbors=k + 1).fit(Ys).kneighbors(Ys, return_distance=False)[:, 1:]
    jac = [len(set(a) & set(b)) / len(set(a) | set(b)) for a, b in zip(nx, ny)]
    return float(np.mean(jac))


def knn_label_agreement(X, Y, labels, k=10, n_sub=5000, seed=0):
    """Fraction of images whose k-NN majority label matches between the two spaces."""
    rng = np.random.default_rng(seed)
    idx = rng.choice(X.shape[0], min(n_sub, X.shape[0]), replace=False)
    lab = np.asarray(labels)[idx]

    def votes(Z):
        nn = NearestNeighbors(n_neighbors=k + 1).fit(Z).kneighbors(Z, return_distance=False)[:, 1:]
        neigh = lab[nn]
        return np.array([np.bincount(r).argmax() for r in neigh])

    return float((votes(X[idx]) == votes(Y[idx])).mean())


def procrustes_distance(X, Y, k=64, seed=0):
    """Orthogonal-Procrustes residual between PCA-k projections (unit-scaled):
    min_R ||Xk R - Yk||_F / ||Yk||_F. Dim-matched by construction."""
    def pca_k(Z):
        Zc = Z - Z.mean(0)
        kk = min(k, Z.shape[1])
        w, V = np.linalg.eigh((Zc.T @ Zc) / (Zc.shape[0] - 1))
        P = Zc @ V[:, ::-1][:, :kk]
        return P / (np.linalg.norm(P) + 1e-12)

    Xk, Yk = pca_k(X), pca_k(Y)
    if Xk.shape[1] != Yk.shape[1]:
        kk = min(Xk.shape[1], Yk.shape[1])
        Xk, Yk = Xk[:, :kk], Yk[:, :kk]
    R, _ = orthogonal_procrustes(Xk, Yk)
    return float(np.linalg.norm(Xk @ R - Yk) / (np.linalg.norm(Yk) + 1e-12))
