"""Defect-rank estimators k-hat (D-019 battery v2; TRD-pi R4 Beta-attenuation lemma).

Frame: shape defects in trained features concentrate in the HIGH-VARIANCE directions, and
uniform slicing of the whitened cloud is blind to low-rank defects at K>>r (overlap c^2 ~
Beta(r/2,(K-r)/2) => slice signal ~ r^2/K^2 — measured: undetectable at audit n for realistic
amplitudes). Both estimators therefore operate in the DECLARED frame: top-m PCA coordinates
(variance order, each standardized to unit variance) — the same frame kurt_topeig reads, m a
declared setting (default 128 or d, whichever is smaller).

Estimator A — Beta-spectrum moment fit (the lemma-consistent global estimator). Model: r
independent defect directions with excess kurtosis kappa each, in ambient dimension m. For a
uniform slice u (unit-variance coordinates), kappa_s = sum_i u_i^4 kappa_i (exact for
independent unit-variance components), so with equal kappas:

    E[kappa_s]   = kappa * 3r/(m(m+2))
    E[kappa_s^2] = kappa^2 * r(9r+96)/(m(m+2)(m+4)(m+6))
    ratio A := (E[kappa_s^2]/E[kappa_s]^2) * m(m+2)/((m+4)(m+6)) = 1 + 32/(3r)  =>  r-hat.

Noise correction: subtract the MEASURED null moments of a matched N(0,I_m) spectrum at the same
(n, m, n_slices) — which-null discipline. Limitation (stated, and why Estimator B exists):
mixed-SIGN defects cancel in E[kappa_s] and break the fit.

Estimator B — projection-pursuit direction count (sign-robust). FastICA with the kurtosis-family
contrast on the top-m standardized coordinates recovers candidate defect directions one by one;
k-hat_pp = number of recovered directions whose |excess kurtosis| clears the matched-Gaussian
null band (max |kappa| over the same procedure on a matched N(0, I_m) draw — the procedure's own
selection bias is inside the null, by construction). Returns the direction kurtosis profile so
class-alignment (eta^2) can be read against it downstream.
"""
import numpy as np


def top_pca_frame(X, m=128, lam_ratio_floor=1e-4):
    """Center, project to top-m PCA directions, standardize each (unit variance). Directions
    with eigenvalue < lam_ratio_floor x the top eigenvalue are excluded: below that ratio the
    stored-fp16 quantization floor manufactures spurious kurtosis when standardized (measured
    2026-07-10 on low-rank cells)."""
    X = np.asarray(X, dtype=np.float64)
    X = X - X.mean(0)
    C = np.cov(X, rowvar=False)
    w, V = np.linalg.eigh(C)
    idx = np.argsort(w)[::-1]
    idx = idx[: min(m, int((w[idx] > lam_ratio_floor * w[idx[0]]).sum()))]
    P = X @ V[:, idx]
    return P / P.std(0)


def slice_kurtosis_spectrum(P, n_slices=2048, seed=0):
    rng = np.random.default_rng(seed)
    U = rng.standard_normal((P.shape[1], n_slices))
    U /= np.linalg.norm(U, axis=0)
    S = P @ U
    S = (S - S.mean(0)) / S.std(0)
    return (S**4).mean(0) - 3.0


def khat_spectrum(X, m=128, n_slices=2048, seed=0, n_boot=200):
    """Estimator A. Returns khat, kappa_hat, CI, detect flag, and the raw moments."""
    rng = np.random.default_rng(seed)
    P = top_pca_frame(X, m=m, seed=seed)
    n, meff = P.shape
    ks = slice_kurtosis_spectrum(P, n_slices=n_slices, seed=seed)
    g = slice_kurtosis_spectrum(rng.standard_normal((n, meff)), n_slices=n_slices, seed=seed + 1)
    m1, m2 = ks.mean() - g.mean(), (ks**2).mean() - (g**2).mean()
    detect = bool(abs(m1) > 4 * g.std() / np.sqrt(len(g)) and m2 > 0)
    out = {"m_frame": meff, "m1": float(m1), "detect_A": detect,
           "khat_A": float("nan"), "kappa_A": float("nan"), "khat_A_ci": (float("nan"),) * 2}
    if not detect:
        return out
    def invert(mm1, mm2):
        A = (mm2 / mm1**2) * meff * (meff + 2) / ((meff + 4) * (meff + 6))
        return meff if A <= 1 else min(float(32 / (3 * (A - 1))), float(meff))
    r = invert(m1, m2)
    out.update(khat_A=r, kappa_A=float(m1 * meff * (meff + 2) / (3 * r)))
    boots = []
    for _ in range(n_boot):
        i, j = rng.integers(0, len(ks), len(ks)), rng.integers(0, len(g), len(g))
        b1, b2 = ks[i].mean() - g[j].mean(), (ks[i]**2).mean() - (g[j]**2).mean()
        if abs(b1) > 1e-12 and b2 > 0:
            boots.append(invert(b1, b2))
    if boots:
        out["khat_A_ci"] = (float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5)))
    return out


def khat_eig(X, m=128, band_sigmas=4.0):
    """Axis-aligned count: number of top-m PCA directions whose |excess kurtosis| exceeds the
    Gaussian sampling band (sd of kappa-hat under H0 = sqrt(24/n)). The cheapest k-hat; exact
    for defects diagonal in the eigenbasis (class structure empirically is, cf. E10 figures)."""
    P = top_pca_frame(X, m=m)
    kurts = (P**4).mean(0) - 3.0
    band = band_sigmas * np.sqrt(24.0 / P.shape[0])
    return {"khat_eig": int((np.abs(kurts) > band).sum()), "eig_band": float(band),
            "eig_kurts": kurts}


def _kurt_pursuit_dir(P, rng, restarts=8, iters=150, tol=1e-7):
    """Fixed-point kurtosis-extremum pursuit on whitened coordinates (u <- E[x (u^T x)^3] - 3u,
    normalized — the kurtosis-contrast FastICA update), restarts vectorized; returns the
    direction with the largest |excess kurtosis|."""
    n, d = P.shape
    U = rng.standard_normal((d, restarts))
    U /= np.linalg.norm(U, axis=0)
    for _ in range(iters):
        S = P @ U
        Un = (P.T @ (S**3)) / n - 3.0 * U
        Un /= np.linalg.norm(Un, axis=0)
        done = (1 - np.abs((U * Un).sum(0))).max() < tol
        U = Un
        if done:
            break
    S = P @ U
    S = (S - S.mean(0)) / S.std(0)
    K = (S**4).mean(0) - 3.0
    i = int(np.argmax(np.abs(K)))
    return U[:, i], float(K[i])


def _pursuit_profile(P, n_dirs, seed):
    """Deflated pursuit: find the extremal direction, restrict to its orthogonal complement
    (proper reparametrization — the complement of a whitened space stays whitened), repeat."""
    rng = np.random.default_rng(seed)
    comp = np.eye(P.shape[1])
    ks = []
    for _ in range(n_dirs):
        Q = P @ comp
        u_local, k = _kurt_pursuit_dir(Q, rng)
        ks.append(abs(k))
        M = np.eye(comp.shape[1]) - np.outer(u_local, u_local)
        q, r = np.linalg.qr(M)
        comp = comp @ q[:, np.abs(np.diag(r)) > 1e-9][:, : comp.shape[1] - 1]
        if comp.shape[1] == 0:
            break
    return np.array(ks)


def pursuit_null_band(n, meff, n_dirs, reps=6, seed=100, margin=1.15):
    """Calibrate the pursuit's matched-null band once per (n, meff, n_dirs): the same pursuit on
    `reps` N(0, I_meff) draws; band = margin x the max |excess kurtosis| the procedure ever
    finds on pure Gaussian data (its own selection bias, measured). Cache and share across
    same-shape cells."""
    rng = np.random.default_rng(seed)
    return margin * max(float(_pursuit_profile(rng.standard_normal((n, meff)), n_dirs,
                                               seed + 1 + i).max()) for i in range(reps))


def khat_pursuit(X, m=64, n_dirs=24, seed=0, band=None):
    """Estimator B (sign-robust, rotation-aware). k-hat_pp = length of the leading run of
    deflated-pursuit directions whose |excess kurtosis| clears the matched-null band (from
    pursuit_null_band; computed here with defaults when not supplied)."""
    P = top_pca_frame(X, m=m)
    n, meff = P.shape
    nd = min(n_dirs, meff)
    prof = _pursuit_profile(P, nd, seed)
    if band is None:
        band = pursuit_null_band(n, meff, nd)
    above = prof > band
    khat = int(np.argmin(above)) if not above.all() else len(above)
    return {"khat_pp": khat, "pp_profile": prof, "pp_null_band": float(band)}
