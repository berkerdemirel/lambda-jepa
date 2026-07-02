"""Linear probes on frozen stored features.

linear_house_v1 — VERBATIM port of ssl_explore/sslx/meters.offline_probe (LayerNorm+Linear,
AdamW 1e-3/1e-7, 30 ep, bs 256, all four meters at the best-val-acc epoch). Kept for parity with
prior CAMPAIGN_LOG numbers (D-006); the M0 exit criterion checks THIS probe against those tables.

linear_raw_v1 — plain Linear on RAW (unnormalized) features, same optimizer/schedule/selection —
linear separability in its purest form; the D-006v2 headline linear probe.

linear_l2_v1 — L2-normalize features, plain Linear — kept for the E11 probe-sensitivity study."""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def _probe_loop(probe, Xtr, ytr, Xva, yva, epochs, lr, wd, bs, device, seed):
    torch.manual_seed(seed)
    probe = probe.to(device)
    opt = torch.optim.AdamW(probe.parameters(), lr=lr, weight_decay=wd)

    @torch.no_grad()
    def meters(X, y):
        logits = probe(X)
        return ((logits.argmax(1) == y).float().mean().item(),
                F.cross_entropy(logits, y).item())

    n, best = Xtr.shape[0], None
    for _ in range(epochs):
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
                    "val_acc": va_acc, "val_loss": va_loss}
    return best


def _tensors(train_feats, train_y, val_feats, val_y, device, l2=False):
    Xtr = torch.as_tensor(np.asarray(train_feats), dtype=torch.float32, device=device)
    Xva = torch.as_tensor(np.asarray(val_feats), dtype=torch.float32, device=device)
    if l2:
        Xtr, Xva = F.normalize(Xtr, dim=1), F.normalize(Xva, dim=1)
    ytr = torch.as_tensor(np.asarray(train_y), dtype=torch.long, device=device)
    yva = torch.as_tensor(np.asarray(val_y), dtype=torch.long, device=device)
    return Xtr, ytr, Xva, yva


def linear_house_v1(train_feats, train_y, val_feats, val_y, num_classes,
                    epochs=30, lr=1e-3, wd=1e-7, bs=256, device="cuda", seed=0):
    Xtr, ytr, Xva, yva = _tensors(train_feats, train_y, val_feats, val_y, device)
    probe = nn.Sequential(nn.LayerNorm(Xtr.shape[1]), nn.Linear(Xtr.shape[1], num_classes))
    return _probe_loop(probe, Xtr, ytr, Xva, yva, epochs, lr, wd, bs, device, seed)


def linear_l2_v1(train_feats, train_y, val_feats, val_y, num_classes,
                 epochs=30, lr=1e-3, wd=1e-7, bs=256, device="cuda", seed=0):
    Xtr, ytr, Xva, yva = _tensors(train_feats, train_y, val_feats, val_y, device, l2=True)
    probe = nn.Linear(Xtr.shape[1], num_classes)
    return _probe_loop(probe, Xtr, ytr, Xva, yva, epochs, lr, wd, bs, device, seed)


def linear_raw_v1(train_feats, train_y, val_feats, val_y, num_classes,
                  epochs=30, lr=1e-3, wd=1e-7, bs=256, device="cuda", seed=0):
    Xtr, ytr, Xva, yva = _tensors(train_feats, train_y, val_feats, val_y, device)
    probe = nn.Linear(Xtr.shape[1], num_classes)
    return _probe_loop(probe, Xtr, ytr, Xva, yva, epochs, lr, wd, bs, device, seed)
