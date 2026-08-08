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


def linear_map_fit(H_tr, Z_tr, H_va, Z_va, eps=1e-12):
    """Head-linearity index (D-015, approved 2026-07-08; its R² half was later CANCELLED as
    a standalone metric by D-060 and its map-spectrum half left unbuilt — this builds both,
    and the spectrum is the part that survives D-060's critique).

    Best linear fit h -> z by OLS on the TRAIN split (both sides centred by TRAIN means),
    scored on VAL. Returns:

      r2_total    1 - ||Z - HW||_F^2 / ||Z - Zbar||_F^2 on val. Invariant to any invertible
                  linear map of h and to rotation/global scale of z, so it is comparable
                  across methods. **D-060's caveat is REAL and rides with every use: a
                  contractive many-to-little head reads HIGH r2 while doing heavy nonlinear
                  work — r2 measures linear REACHABILITY of the output, not head magnitude.
                  Never quote it alone.**
      r2_meandim  per-dim R² averaged (equal-weight secondary; noisier when dims are dead).
      sigma_max/min, cond, effrank_map
                  singular spectrum of the fitted map W (d_h x d_z). These read contraction
                  DIRECTLY, which is what r2 cannot: sigma_min ~ 0 means the head annihilates
                  a direction of h no matter how high r2 is, and effrank_map (entropy of the
                  normalised spectrum, RankMe-style) counts how many directions the linear
                  part actually uses. Scale-free: effrank_map and cond are invariant to
                  global rescaling of either side.
      resid_frac, resid_effrank
                  residual covariance on val: unexplained share of z's variance, and the
                  effective rank of what the linear map misses = the nonlinear work.
    """
    H_tr, Z_tr = np.asarray(H_tr, np.float64), np.asarray(Z_tr, np.float64)
    H_va, Z_va = np.asarray(H_va, np.float64), np.asarray(Z_va, np.float64)
    hm, zm = H_tr.mean(0), Z_tr.mean(0)
    W, *_ = np.linalg.lstsq(H_tr - hm, Z_tr - zm, rcond=None)
    Zc = Z_va - zm
    R = Zc - (H_va - hm) @ W
    sse, sst = (R ** 2).sum(), (Zc ** 2).sum()
    per_dim = 1.0 - (R ** 2).sum(0) / np.maximum((Zc ** 2).sum(0), eps)
    s = np.linalg.svd(W, compute_uv=False)
    p = s / max(s.sum(), eps)
    effrank_map = float(np.exp(-(p * np.log(p + eps)).sum()))
    rc = np.linalg.eigvalsh(np.cov(R, rowvar=False))[::-1].clip(min=0)
    q = rc / max(rc.sum(), eps)
    return {"r2_total": float(1.0 - sse / max(sst, eps)),
            "r2_meandim": float(per_dim.mean()),
            "sigma_max": float(s[0]), "sigma_min": float(s[-1]),
            "cond": float(s[0] / max(s[-1], eps)),
            "effrank_map": effrank_map, "d_h": int(W.shape[0]), "d_z": int(W.shape[1]),
            "resid_frac": float(sse / max(sst, eps)),
            "resid_effrank": float(np.exp(-(q * np.log(q + eps)).sum()))}
