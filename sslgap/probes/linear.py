"""Linear probes on frozen stored features.

linear_house_v1 — VERBATIM port of ssl_explore/sslx/meters.offline_probe (LayerNorm+Linear,
AdamW 1e-3/1e-7, 30 ep, bs 256, all four meters at the best-val-acc epoch). Kept for parity with
prior CAMPAIGN_LOG numbers (D-006); the M0 exit criterion checks THIS probe against those tables.

linear_raw_v1 — plain Linear on RAW (unnormalized) features, same optimizer/schedule/selection —
linear separability in its purest form; the original D-006v2 headline linear probe.

linear_l2_v1 — L2-normalize features, plain Linear — kept for the E11 probe-sensitivity study.

*_v2 (D-020) — SAME transform/optimizer/selection as the matching v1, convergence-guaranteed:
patience 120 on best-val (any improvement resets), hard cap 1000 epochs; best_ep + epochs_run
recorded so boundary-censoring is visible in every CSV. Why: at the fixed 30-ep budget a plain
Linear on unnormalized GAP features is boundary-censored (best_ep=29) and understates
separability by 4.7-6.8 pts, differentially across methods (results/diag/probe_conv.csv);
patience calibrated from the measured curves (max observed stale-gap 108 ep). v1 rows stay in
every CSV for continuity."""
import copy

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def _probe_loop(probe, Xtr, ytr, Xva, yva, epochs, lr, wd, bs, device, patience=None,
                return_probe=False):
    # seeding happens in the wrappers BEFORE probe construction (weights + shuffling share it)
    probe = probe.to(device)
    opt = torch.optim.AdamW(probe.parameters(), lr=lr, weight_decay=wd)

    @torch.no_grad()
    def meters(X, y):
        logits = probe(X)
        return ((logits.argmax(1) == y).float().mean().item(),
                F.cross_entropy(logits, y).item())

    n, best, ep = Xtr.shape[0], None, 0
    for ep in range(epochs):
        probe.train()
        perm = torch.randperm(n, device=device)
        for i in range(0, n, bs):
            idx = perm[i:i + bs]
            loss = F.cross_entropy(probe(Xtr[idx]), ytr[idx])
            opt.zero_grad()
            loss.backward()
            opt.step()
        probe.eval()
        tr_acc, tr_loss = meters(Xtr, ytr)
        va_acc, va_loss = meters(Xva, yva)
        if best is None or va_acc > best["val_acc"]:
            best = {"train_acc": tr_acc, "train_loss": tr_loss,
                    "val_acc": va_acc, "val_loss": va_loss, "best_ep": ep}
            if return_probe:                      # the selected epoch's weights, for callers that
                state = copy.deepcopy(probe.state_dict())   # need the map itself (E35 per-direction)
        if patience is not None and ep - best["best_ep"] >= patience:
            break
    best["epochs_run"] = ep + 1
    if return_probe:
        probe.load_state_dict(state)
        best["probe"] = probe.eval()
    return best


def _tensors(train_feats, train_y, val_feats, val_y, device, l2=False):
    Xtr = torch.as_tensor(np.asarray(train_feats), dtype=torch.float32, device=device)
    Xva = torch.as_tensor(np.asarray(val_feats), dtype=torch.float32, device=device)
    if l2:
        Xtr, Xva = F.normalize(Xtr, dim=1), F.normalize(Xva, dim=1)
    ytr = torch.as_tensor(np.asarray(train_y), dtype=torch.long, device=device)
    yva = torch.as_tensor(np.asarray(val_y), dtype=torch.long, device=device)
    return Xtr, ytr, Xva, yva


def _build(kind, d, num_classes, seed):
    torch.manual_seed(seed)
    if kind == "house":
        return nn.Sequential(nn.LayerNorm(d), nn.Linear(d, num_classes))
    return nn.Linear(d, num_classes)


def _linear(kind, train_feats, train_y, val_feats, val_y, num_classes,
            epochs, lr, wd, bs, device, seed, patience, return_probe=False):
    Xtr, ytr, Xva, yva = _tensors(train_feats, train_y, val_feats, val_y, device,
                                  l2=(kind == "l2"))
    probe = _build(kind, Xtr.shape[1], num_classes, seed)
    return _probe_loop(probe, Xtr, ytr, Xva, yva, epochs, lr, wd, bs, device, patience,
                       return_probe)


def linear_house_v1(train_feats, train_y, val_feats, val_y, num_classes,
                    epochs=30, lr=1e-3, wd=1e-7, bs=256, device="cuda", seed=0):
    return _linear("house", train_feats, train_y, val_feats, val_y, num_classes,
                   epochs, lr, wd, bs, device, seed, patience=None)


def linear_l2_v1(train_feats, train_y, val_feats, val_y, num_classes,
                 epochs=30, lr=1e-3, wd=1e-7, bs=256, device="cuda", seed=0):
    return _linear("l2", train_feats, train_y, val_feats, val_y, num_classes,
                   epochs, lr, wd, bs, device, seed, patience=None)


def linear_raw_v1(train_feats, train_y, val_feats, val_y, num_classes,
                  epochs=30, lr=1e-3, wd=1e-7, bs=256, device="cuda", seed=0):
    return _linear("raw", train_feats, train_y, val_feats, val_y, num_classes,
                   epochs, lr, wd, bs, device, seed, patience=None)


def linear_house_v2(train_feats, train_y, val_feats, val_y, num_classes,
                    epochs=1000, lr=1e-3, wd=1e-7, bs=256, device="cuda", seed=0, patience=120):
    return _linear("house", train_feats, train_y, val_feats, val_y, num_classes,
                   epochs, lr, wd, bs, device, seed, patience)


def linear_l2_v2(train_feats, train_y, val_feats, val_y, num_classes,
                 epochs=1000, lr=1e-3, wd=1e-7, bs=256, device="cuda", seed=0, patience=120):
    return _linear("l2", train_feats, train_y, val_feats, val_y, num_classes,
                   epochs, lr, wd, bs, device, seed, patience)


def linear_raw_v2(train_feats, train_y, val_feats, val_y, num_classes,
                  epochs=1000, lr=1e-3, wd=1e-7, bs=256, device="cuda", seed=0, patience=120,
                  return_probe=False):
    """return_probe: also hand back the selected epoch's fitted map (eval mode) — the caller can
    then read what the classifier does with a given feature direction without refitting."""
    return _linear("raw", train_feats, train_y, val_feats, val_y, num_classes,
                   epochs, lr, wd, bs, device, seed, patience, return_probe)


def linear_lbfgs_v1(train_feats, train_y, val_feats, val_y, num_classes, lam=1e-4,
                    max_iter=500, device="cuda", dtype=torch.float32):
    """Ridge multinomial logistic regression solved to the GLOBAL optimum by L-BFGS
    (strong-Wolfe line search). Returns the same meter dict as the *_v1/_v2 family plus `lam`.

    Why a second linear reader (E36): the *_v1/_v2 family is AdamW at a fixed lr with best-val
    early stopping, and neither the step geometry nor the stopping point is invariant to an
    invertible linear re-coordinatization of the input — so a representation can score lower purely
    for being ill-conditioned. On the E36 pair that is not academic: on the same task the treated
    cell's probe peaks at epoch 7 and the untreated cell's at epoch 312
    (`results/probes/in100.floorssl.s0.d256vm4{,zonly}.extL.csv`), the signature of a gradient
    reader fighting its input's conditioning. This objective is convex and is driven to
    convergence, so the only free knob is `lam`; composed with a whitening frame at alpha = 0 it
    gives a reader that conditioning cannot fool, which is what separates "the treated model holds
    extra content" from "the treated model is merely easier to read"."""
    Xtr = torch.as_tensor(np.asarray(train_feats), dtype=dtype, device=device)
    Xva = torch.as_tensor(np.asarray(val_feats), dtype=dtype, device=device)
    ytr = torch.as_tensor(np.asarray(train_y), dtype=torch.long, device=device)
    yva = torch.as_tensor(np.asarray(val_y), dtype=torch.long, device=device)
    # `lam` may be a scalar or a per-COLUMN vector. The vector form exists because an increment
    # test over a concatenation is only valid if the wider model nests the narrower one AFTER
    # regularization is selected: with one shared lam it does not (measured -- [centroid, residual
    # 2nd moment] scored 40.5 against the centroid alone at 60.0, because the two blocks want lam
    # 1e-4 and 1e-2 respectively). With a per-block lam, sending the second block's lam up recovers
    # the first block's own model exactly, so the increment is a genuine lower bound.
    lamv = torch.as_tensor(np.broadcast_to(np.asarray(lam, dtype=np.float64),
                                           (Xtr.shape[1],)).copy(),
                           dtype=dtype, device=device)[:, None]
    W = torch.zeros(Xtr.shape[1], num_classes, dtype=dtype, device=device, requires_grad=True)
    b = torch.zeros(num_classes, dtype=dtype, device=device, requires_grad=True)
    opt = torch.optim.LBFGS([W, b], max_iter=max_iter, history_size=20, tolerance_grad=1e-9,
                            tolerance_change=1e-12, line_search_fn="strong_wolfe")

    def closure():
        opt.zero_grad()
        loss = F.cross_entropy(Xtr @ W + b, ytr) + (lamv * W * W).sum()
        loss.backward()
        return loss
    opt.step(closure)
    with torch.no_grad():
        gnorm = float(torch.cat([W.grad.flatten(), b.grad.flatten()]).norm())
        tr_logits, va_logits = Xtr @ W + b, Xva @ W + b
        pred = va_logits.argmax(1)
        return {"train_acc": float((tr_logits.argmax(1) == ytr).float().mean()),
                "train_loss": float(F.cross_entropy(tr_logits, ytr)),
                "val_acc": float((pred == yva).float().mean()),
                "val_loss": float(F.cross_entropy(va_logits, yva)),
                "lam": lam if np.isscalar(lam) else "blockwise",
                "grad_norm": gnorm, "pred": pred.cpu().numpy()}


def lda_shrunk_v1(train_feats, train_y, val_feats, val_y, num_classes, shrink=0.1,
                  device="cuda"):
    """Regularized LDA: pooled within-class covariance shrunk toward its isotropic target.

    At shrink = 0 this reader is EXACTLY invariant to any invertible linear map of its input — it
    reads x through Sigma_within^-1, which absorbs the map — so it is the one member of the battery
    that conditioning cannot fool with no tuning at all. Shrinkage is needed at 384-d / 100 classes
    / 50k and travels with every number it produces (at shrink > 0 the exact invariance is only
    approximate, the same way E36's whitening frame loses equivariance off alpha = 0)."""
    X = torch.as_tensor(np.asarray(train_feats), dtype=torch.float64, device=device)
    y = torch.as_tensor(np.asarray(train_y), dtype=torch.long, device=device)
    Xv = torch.as_tensor(np.asarray(val_feats), dtype=torch.float64, device=device)
    yv = torch.as_tensor(np.asarray(val_y), dtype=torch.long, device=device)
    d = X.shape[1]
    mus, Sw = [], torch.zeros(d, d, dtype=torch.float64, device=device)
    for c in range(num_classes):
        xc = X[y == c]
        m = xc.mean(0)
        mus.append(m)
        xc = xc - m
        Sw += xc.T @ xc
    Sw /= (len(X) - num_classes)
    Sw = (1 - shrink) * Sw + shrink * (torch.trace(Sw) / d) * torch.eye(d, dtype=torch.float64,
                                                                       device=device)
    M = torch.stack(mus)                                          # [C, d]
    A = torch.linalg.solve(Sw, M.T)                               # Sigma^-1 mu_c as columns
    score = Xv @ A - 0.5 * (M * A.T).sum(1)                       # shared-covariance discriminant
    pred = score.argmax(1)
    return {"val_acc": float((pred == yv).float().mean()), "shrink": float(shrink),
            "pred": pred.cpu().numpy()}
