"""Two-space estimators over V-view stores: whitening, the Theta spectrum, per-class organization vs view-sensitivity."""
import numpy as np
import pandas as pd

def cloud_moments(views):
    """views: list of V (N,D) same-source-order arrays. Returns mu, pooled centers m,
    Â (V-unbiased), B̂ = B̂pool − Â/V (debiased), B̂pool, B̂x (even/odd cross-group)."""
    X = np.stack([np.asarray(v, dtype=np.float64) for v in views])
    V, N, _ = X.shape
    m = X.mean(0)
    mu = m.mean(0)
    r = X - m
    A = np.einsum("vnd,vne->de", r, r) / (N * (V - 1))
    mc = m - mu
    Bpool = mc.T @ mc / (N - 1)
    ga, gb = X[0::2].mean(0), X[1::2].mean(0)
    C = (ga - ga.mean(0)).T @ (gb - gb.mean(0)) / (N - 1)
    return {"mu": mu, "m": m, "A": A, "B": Bpool - A / V, "Bpool": Bpool,
            "Bx": (C + C.T) / 2, "V": V, "N": N}

def whiten_map(B, floor_frac=1e-3):
    """Center-whitening on the truncated support: keep eigenvalues > floor_frac·top,
    descending. Returns R (r,D) with R B Rᵀ = I_r, the kept eigenvalues, and r."""
    w, U = np.linalg.eigh((B + B.T) / 2)
    keep = w > floor_frac * w.max()
    w, U = w[keep][::-1], U[:, keep][:, ::-1]
    return (U / np.sqrt(w)).T, w, int(w.size)

def theta_spectrum(A, B, floor_frac=1e-3):
    """Θ = R A Rᵀ and its eigen-decomposition (descending)."""
    R, b_eigs, r = whiten_map(B, floor_frac)
    Th = R @ A @ R.T
    lam, Uv = np.linalg.eigh((Th + Th.T) / 2)
    return {"R": R, "theta": Th, "lam": lam[::-1], "evecs": Uv[:, ::-1],
            "r": r, "b_eigs": b_eigs}

def _psd_clip(M):
    w, U = np.linalg.eigh((M + M.T) / 2)
    return (U * np.clip(w, 0, None)) @ U.T

def _split(N, holdout, seed):
    perm = np.random.default_rng(seed).permutation(N)
    ntr = int(round(N * (1 - holdout)))
    return perm[:ntr], perm[ntr:]

def _ls(X, Y, ridge=1e-8):
    """Ridge-stabilized least squares X→Y (both pre-centered); ridge is relative."""
    G = X.T @ X / X.shape[0]
    lam = ridge * np.trace(G) / G.shape[0]
    return np.linalg.solve(G + lam * np.eye(G.shape[0]), X.T @ Y / X.shape[0])

def accessibility(mh, mz, Az=None, V=None, holdout=0.5, floor_frac=1e-3, seed=0):
    """Held-out linear recovery of center-whitened z-centers from h-centers
    (Thm 4.1): I = W B_h Wᵀ + Σ_c. Whitening (from the pooled-center covariance of
    the targets) and the fit use train sources; Σ̂_c is reported on both splits.
    Az with V gives the finite-view noise floor tr(R Az Rᵀ)/(V·r_z) — an R² deficit
    at or below it is center-estimation noise, not inaccessibility.
    Returns (summary, sigma_eigs_by_split)."""
    tr, te = _split(mh.shape[0], holdout, seed)
    muh, muz = mh[tr].mean(0), mz[tr].mean(0)
    Rz, _, rz = whiten_map(np.cov(mz[tr], rowvar=False), floor_frac)
    Y, Xc = (mz - muz) @ Rz.T, mh - muh
    W = _ls(Xc[tr], Y[tr])
    sv = np.linalg.svd(W, compute_uv=False)
    p = sv ** 2 / (sv ** 2).sum()
    summary = {"r_z": rz, "d_h": mh.shape[1], "n_train": len(tr), "n_test": len(te),
               "floor_frac": floor_frac,
               "map_effrank": float(np.exp(-(p * np.log(p + 1e-300)).sum())),
               "map_rank_1pct": int((sv > 0.01 * sv[0]).sum())}
    if Az is not None and V:
        summary["noise_floor"] = float(np.trace(Rz @ Az @ Rz.T) / (V * rz))
    eigs = {}
    for split, idx in (("holdout", te), ("insample", tr)):
        E = Y[idx] - Xc[idx] @ W
        Sc = (E - E.mean(0)).T @ (E - E.mean(0)) / (len(idx) - 1)
        ev = np.linalg.eigvalsh((Sc + Sc.T) / 2)[::-1]
        eigs[split] = ev
        summary[f"r2_acc_{split}"] = float(1 - ev.sum() / rz)
        summary[f"r2_min_{split}"] = float(1 - ev[0])
        for tau in (0.5, 0.2, 0.1):
            summary[f"frac_resid_gt{tau}_{split}"] = float((ev > tau).mean())
    return summary, eigs

def center_fidelity(views, holdout=0.5, floor_frac=1e-3, seed=0):
    """Label-free test of the fidelity law (Thm 5.2ii): held-out residual spectrum of
    the best linear single-view→center map vs Θ(I+Θ)⁻¹ + Θ/V_c. Predictor views are
    the even group, target centers come from the disjoint odd group (App. E: center
    estimates and individual views use separate augmentation realizations); the Θ/V_c
    term is the known noise of that target. Moments, whitening, and the fit are
    train-source; observed residuals are held-out. Returns (summary, per-direction
    rows) and the whitened held-out residual covariance for gs_split."""
    V = len(views)
    tr, te = _split(views[0].shape[0], holdout, seed)
    mom = cloud_moments([v[tr] for v in views])
    ts = theta_spectrum(mom["A"], mom["B"], floor_frac)
    R, mu, lam, Uv = ts["R"], mom["mu"], ts["lam"], ts["evecs"]
    wh = lambda a: (np.asarray(a, dtype=np.float64) - mu) @ R.T
    pred_views, ctr_views = views[0::2], views[1::2]
    Vc = len(ctr_views)
    tgt = np.mean([np.asarray(v, dtype=np.float64) for v in ctr_views], axis=0)
    Xtr = np.concatenate([wh(v[tr]) for v in pred_views])
    Ytr = np.tile(wh(tgt[tr]), (len(pred_views), 1))
    W = _ls(Xtr, Ytr)
    Xte = np.concatenate([wh(v[te]) for v in pred_views])
    E = np.tile(wh(tgt[te]), (len(pred_views), 1)) - Xte @ W
    Sobs = (E - E.mean(0)).T @ (E - E.mean(0)) / (E.shape[0] - 1)
    obs = np.einsum("ij,jk,ki->i", Uv.T, Sobs, Uv)
    pred = lam / (1 + lam) + lam / Vc
    Wlaw = np.linalg.inv(np.eye(ts["r"]) + ts["theta"])
    rows = [{"j": j, "lam": float(lam[j]), "pred_resid": float(pred[j]),
             "obs_resid": float(obs[j]), "pred_resid_noiseless": float(lam[j] / (1 + lam[j]))}
            for j in range(ts["r"])]
    lp, lo = np.log(np.maximum(pred, 1e-12)), np.log(np.maximum(obs, 1e-12))
    summary = {"r": ts["r"], "V_c": Vc, "n_pred_views": len(pred_views),
               "theta_tr_over_r": float(lam.mean()), "theta_op": float(lam[0]),
               "resid_corr": float(np.corrcoef(pred, obs)[0, 1]),
               "resid_logcorr": float(np.corrcoef(lp, lo)[0, 1]),
               "resid_med_abs_gap": float(np.median(np.abs(obs - pred))),
               "resid_med_ratio": float(np.median(obs / np.maximum(pred, 1e-12))),
               "w_law_relerr": float(np.linalg.norm(W.T - Wlaw) / np.linalg.norm(Wlaw))}
    return summary, rows, {"Sobs": Sobs, "ts": ts, "Vc": Vc, "mu": mu,
                           "split": (tr, te), "wh": wh, "tgt": tgt,
                           "pred_views": pred_views}

def gs_split(ts, resid_cov, Vc):
    """Ĝ upper bound from ANY held-out view→center predictor's whitened residual
    covariance (App. E identity: that covariance = G + PSD extra + Θ/V_c target
    noise), so Ŝ = Θ − Ĝ_ub is a LOWER bound on excess thickness — conservative in
    the direction of claiming excess. With a linear predictor this carries no more
    information than the closed form Θ²(I+Θ)⁻¹; an MLP predictor
    (mlp_center_residuals) is what can tighten it."""
    G_ub = _psd_clip(resid_cov - ts["theta"] / Vc)
    S_lb = _psd_clip(ts["theta"] - G_ub)
    lin = ts["theta"] @ ts["theta"] @ np.linalg.inv(np.eye(ts["r"]) + ts["theta"])
    return {"G_ub_tr": float(np.trace(G_ub)), "G_ub_op": float(np.linalg.eigvalsh(G_ub)[-1]),
            "S_lb_tr": float(np.trace(S_lb)), "S_lb_op": float(np.linalg.eigvalsh(S_lb)[-1]),
            "theta_tr": float(np.trace(ts["theta"])),
            "S_lb_tr_linclosed": float(np.trace(lin)),
            "S_share_lb": float(np.trace(S_lb) / max(np.trace(ts["theta"]), 1e-12))}

def mlp_center_residuals(fid_ctx, width=1024, max_epochs=300, patience=20, bs=8192,
                         lr=1e-3, seed=0):
    """MLP q(view)→whitened center for the Ĝ tightening (same disjoint view groups as
    center_fidelity). Fit discipline (a fixed 20-epoch budget underfits below the linear closed
    form): the training
    sources are split fit/val BY SOURCE (all views of a source on one side — mixed
    views would leak the memorized center into val), early stopping on val MSE with
    best-val weight restore, and the held-out test sources are touched exactly once
    for the bound. Returns (residual covariance on test, diagnostics dict with
    train/val/test MSE + stopping epoch). Lazy torch; CUDA if available."""
    import torch
    tr, te = fid_ctx["split"]
    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(tr))
    nval = max(int(len(tr) * 0.1), 1)
    fit_s, val_s = tr[perm[nval:]], tr[perm[:nval]]
    wh, tgt = fid_ctx["wh"], fid_ctx["tgt"]

    def xy(src):
        X = np.concatenate([wh(v[src]) for v in fid_ctx["pred_views"]]).astype(np.float32)
        Y = np.tile(wh(tgt[src]), (len(fid_ctx["pred_views"]), 1)).astype(np.float32)
        return X, Y

    Xf, Yf = xy(fit_s)
    Xv, Yv = xy(val_s)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(seed)
    net = torch.nn.Sequential(
        torch.nn.Linear(Xf.shape[1], width), torch.nn.GELU(),
        torch.nn.Linear(width, width), torch.nn.GELU(),
        torch.nn.Linear(width, Yf.shape[1])).to(dev)
    opt = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=1e-5)
    Xt, Yt = torch.from_numpy(Xf).to(dev), torch.from_numpy(Yf).to(dev)
    Xvd, Yvd = torch.from_numpy(Xv).to(dev), torch.from_numpy(Yv).to(dev)

    def mse(X, Y):
        with torch.no_grad():
            s, n = 0.0, 0
            for k in range(0, X.shape[0], bs):
                p = net(X[k:k + bs])
                s += torch.nn.functional.mse_loss(p, Y[k:k + bs], reduction="sum").item()
                n += p.numel()
            return s / n

    best, best_ep, best_state, tr_at_best = np.inf, -1, None, np.nan
    for ep in range(max_epochs):
        pm = torch.randperm(Xt.shape[0], device=dev)
        for k in range(0, Xt.shape[0], bs):
            idx = pm[k:k + bs]
            loss = torch.nn.functional.mse_loss(net(Xt[idx]), Yt[idx])
            opt.zero_grad(); loss.backward(); opt.step()
        v = mse(Xvd, Yvd)
        if v < best:
            best, best_ep = v, ep
            tr_at_best = mse(Xt, Yt)
            best_state = {k: p.detach().clone() for k, p in net.state_dict().items()}
        elif ep - best_ep >= patience:
            break
    net.load_state_dict(best_state)
    Xe, Ye = xy(te)
    with torch.no_grad():
        P = np.concatenate([net(torch.from_numpy(Xe[k:k + bs]).to(dev)).cpu().numpy()
                            for k in range(0, Xe.shape[0], bs)])
    E = (Ye - P).astype(np.float64)
    diag = {"mlp_train_mse": float(tr_at_best), "mlp_val_mse": float(best),
            "mlp_test_mse": float((E ** 2).mean()), "mlp_best_ep": int(best_ep),
            "mlp_epochs_run": int(ep + 1)}
    return (E - E.mean(0)).T @ (E - E.mean(0)) / (E.shape[0] - 1), diag

def organization(h_views, mz, labels, holdout=0.5, floor_frac=1e-3, seed=0):
    """Per-class rows: d̂_h(y), d̂_z(y) (held-out center-organization errors of the
    unit-normalized centered class indicator), the transfer bound d_z + √(uᵀΣ_c u)
    (Thm 4.1iii), and the downstream decomposition check (Thm 5.2iii): held-out
    single-view probe error vs d̂_h² + âᵀΘ̂(I+Θ̂)⁻¹â. One consistent train/test
    source split across all fits; h whitening from B̂_h so the h-center coordinates
    are the theorem's orthonormal basis (up to the Θ/V estimation floor)."""
    y = np.asarray(labels)
    N = y.shape[0]
    tr, te = _split(N, holdout, seed)
    cls = np.unique(y)
    n_tr = np.array([(y[tr] == c).sum() for c in cls])
    n_te = np.array([(y[te] == c).sum() for c in cls])
    cls = cls[(n_tr >= 5) & (n_te >= 5)]
    p = np.array([(y[tr] == c).mean() for c in cls])
    T = (y[:, None] == cls[None, :]).astype(np.float64)
    T = (T - p) / np.sqrt(p * (1 - p))
    mom = cloud_moments([v[tr] for v in h_views])
    ts = theta_spectrum(mom["A"], mom["B"], floor_frac)
    mh_all = np.mean([np.asarray(v, dtype=np.float64) for v in h_views], axis=0)
    Mh = (mh_all - mom["mu"]) @ ts["R"].T
    muz = mz[tr].mean(0)
    Rz, _, rz = whiten_map(np.cov(mz[tr], rowvar=False), floor_frac)
    Mz = (mz - muz) @ Rz.T
    Wc = _ls(Mh[tr], Mz[tr])
    Ec = Mz[te] - Mh[te] @ Wc
    Sc = (Ec - Ec.mean(0)).T @ (Ec - Ec.mean(0)) / (len(te) - 1)
    Ah = _ls(Mh[tr], T[tr])
    Az_ = _ls(Mz[tr], T[tr])
    d_h = np.sqrt(np.maximum(((T[te] - Mh[te] @ Ah) ** 2).mean(0), 0))
    d_z = np.sqrt(np.maximum(((T[te] - Mz[te] @ Az_) ** 2).mean(0), 0))
    dir_resid = np.einsum("rc,rs,sc->c", Az_, Sc, Az_)
    F = ts["theta"] @ np.linalg.inv(np.eye(ts["r"]) + ts["theta"])
    view_loss_pred = np.einsum("rc,rs,sc->c", Ah, F, Ah)
    wh = lambda a: (np.asarray(a, dtype=np.float64) - mom["mu"]) @ ts["R"].T
    Xtr = np.concatenate([wh(v[tr]) for v in h_views])
    Wp = _ls(Xtr, np.tile(T[tr], (len(h_views), 1)))
    Xte = np.concatenate([wh(v[te]) for v in h_views])
    probe_err = ((np.tile(T[te], (len(h_views), 1)) - Xte @ Wp) ** 2).mean(0)
    rows = [{"cls": int(c), "d_h": float(d_h[k]), "d_z": float(d_z[k]),
             "dir_resid": float(dir_resid[k]),
             "transfer_bound": float(d_z[k] + np.sqrt(max(dir_resid[k], 0))),
             "probe_err": float(probe_err[k]),
             "decomp_pred": float(d_h[k] ** 2 + view_loss_pred[k])}
            for k, c in enumerate(cls)]
    dec_o = probe_err
    dec_p = d_h ** 2 + view_loss_pred
    summary = {"n_classes": len(cls), "d_h_med": float(np.median(d_h)),
               "d_z_med": float(np.median(d_z)),
               "transfer_viol_frac": float((d_h > d_z + np.sqrt(np.maximum(dir_resid, 0)) + 1e-6).mean()),
               "decomp_corr": float(np.corrcoef(dec_p, dec_o)[0, 1]),
               "decomp_med_ratio": float(np.median(dec_o / np.maximum(dec_p, 1e-12)))}
    return summary, rows
