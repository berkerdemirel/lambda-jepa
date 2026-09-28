"""Cloud energies over V-view stores: within-image W, debiased between-image B, thickness Theta = W/B, transmission a, b, Lambda."""
import numpy as np

def orbit_energies(views, labels=None):
    """views: list of V arrays (N, D), same image order. Returns the per-space dict;
    with labels (N,) adds the class-split center energies (between/within, debiased)."""
    V = len(views)
    X = [v.astype(np.float64) for v in views]
    N, D = X[0].shape
    W = 0.0
    for u in range(V):
        for w in range(u + 1, V):
            W += ((X[u] - X[w]) ** 2).sum(1).mean()
    W /= V * (V - 1) / 2
    m = np.stack(X).mean(0)
    mc = m - m.mean(0)
    B_hat = 2.0 * (mc ** 2).sum(1).mean() * N / (N - 1)
    B = B_hat - W / V
    out = {"V": V, "N": N, "D": D, "W": W, "B_hat": B_hat, "B": B,
           "omega": W / B, "omega_hat": W / B_hat, "r_rms": np.sqrt(W / 2)}
    if labels is not None:
        y = np.asarray(labels)
        cls = np.unique(y)
        cm = np.stack([m[y == c].mean(0) for c in cls])
        nw = np.array([(y == c).sum() for c in cls], dtype=np.float64)
        Bw = np.mean([2.0 * ((m[y == c] - cm[k]) ** 2).sum(1).mean() * nw[k] / (nw[k] - 1)
                      for k, c in enumerate(cls) if nw[k] > 1]) - W / V
        cmc = cm - (cm * nw[:, None]).sum(0) / nw.sum()
        Bb = 2.0 * ((cmc ** 2).sum(1) * nw).sum() / nw.sum() * len(cls) / (len(cls) - 1)
        out.update({"B_within_cls": Bw, "B_between_cls": Bb, "BW_cls": Bb / Bw})
    return out

def transmission(eh, ez):
    """Cross-space gains from two orbit_energies() dicts (same store, same image order)."""
    a2, b2 = ez["W"] / eh["W"], ez["B"] / eh["B"]
    return {"a2": a2, "a": np.sqrt(a2), "b2": b2, "b": np.sqrt(b2),
            "lam": np.sqrt(b2 / a2), "omega_h_over_z": eh["omega"] / ez["omega"]}

def transmission_spectrum(h_views, z_views, ridge=1e-8):
    """The {a_k} spectrum: least-squares J minimizing E‖δz − Jδh‖² over view
    residuals δs_iv = s_iv − m_i, solved in the δh PC basis — a_k = ‖J u_k‖ is the head's
    gain on the k-th h-residual PC (eigenvalue s_k, descending). Selectivity = spread of
    {a_k}; the scalar a² = E‖δz‖²/E‖δh‖² (≡ W_z/W_h exactly — the V/(V−1) factors cancel)
    decomposes linearly as a2_lin = Σ a_k² s_k / Σ s_k with residual 1−R².
    Returns (spectrum_rows, summary): rows carry k, s_k, sk_share, a_k."""
    dh = np.concatenate([v.astype(np.float64) - np.mean(h_views, axis=0) for v in h_views])
    dz = np.concatenate([v.astype(np.float64) - np.mean(z_views, axis=0) for v in z_views])
    n = dh.shape[0]
    Shh = dh.T @ dh / n
    Czh = dz.T @ dh / n
    s, U = np.linalg.eigh(Shh)
    s, U = s[::-1], U[:, ::-1]
    eps = ridge * s.sum()
    G = Czh @ U
    a_k = np.linalg.norm(G, axis=0) / (s + eps)
    pred = (dh @ U) / (s + eps) @ G.T
    ez, eh = (dz ** 2).sum(1).mean(), (dh ** 2).sum(1).mean()
    r2 = 1.0 - ((dz - pred) ** 2).sum(1).mean() / ez
    a2_lin = float((a_k ** 2 * s).sum() / s.sum())
    share = s / s.sum()
    rows = [{"k": k, "s_k": float(s[k]), "sk_share": float(share[k]), "a_k": float(a_k[k])}
            for k in range(len(s))]
    w = a_k ** 2 * s
    summary = {"a2_total": float(ez / eh), "a2_lin": a2_lin, "r2_lin": float(r2),
               "a_max": float(a_k.max()), "a_med": float(np.median(a_k)),
               "cv_ak": float(a_k.std() / a_k.mean()),
               "top8_energy_share": float(np.sort(w)[-8:].sum() / w.sum())}
    return rows, summary
