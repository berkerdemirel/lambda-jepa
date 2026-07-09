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
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def _probe_loop(probe, Xtr, ytr, Xva, yva, epochs, lr, wd, bs, device, patience=None):
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
        if patience is not None and ep - best["best_ep"] >= patience:
            break
    best["epochs_run"] = ep + 1
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
            epochs, lr, wd, bs, device, seed, patience):
    Xtr, ytr, Xva, yva = _tensors(train_feats, train_y, val_feats, val_y, device,
                                  l2=(kind == "l2"))
    probe = _build(kind, Xtr.shape[1], num_classes, seed)
    return _probe_loop(probe, Xtr, ytr, Xva, yva, epochs, lr, wd, bs, device, patience)


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
                  epochs=1000, lr=1e-3, wd=1e-7, bs=256, device="cuda", seed=0, patience=120):
    return _linear("raw", train_feats, train_y, val_feats, val_y, num_classes,
                   epochs, lr, wd, bs, device, seed, patience)
