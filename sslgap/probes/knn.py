"""Weighted-cosine kNN classifier."""
import torch
import torch.nn.functional as F

def knn_predict(feature, feature_bank, feature_labels, num_classes=10, knn_k=200, knn_t=0.1):
    """feature [B, D] L2-normed; feature_bank [D, N] L2-normed; feature_labels [N] long."""
    sim = torch.mm(feature, feature_bank)
    sim_weight, sim_idx = sim.topk(k=min(knn_k, feature_bank.shape[1]), dim=-1)
    sim_labels = torch.gather(feature_labels.expand(feature.shape[0], -1), -1, sim_idx)
    sim_weight = (sim_weight / knn_t).exp()
    one_hot = torch.zeros(sim_labels.numel(), num_classes, device=feature.device)
    one_hot.scatter_(-1, sim_labels.reshape(-1, 1), 1.0)
    scores = (one_hot.view(feature.shape[0], -1, num_classes) * sim_weight.unsqueeze(-1)).sum(1)
    return scores.argsort(dim=-1, descending=True)

@torch.no_grad()
def knn_topk_acc(train_feats, train_y, val_feats, val_y, num_classes=10,
                 knn_k=200, knn_t=0.1, bs=256, device="cpu", return_pred=False):
    """`return_pred` is additive and defaults off, so the ported metric is untouched: it exists
    because a kNN gap of 1.5 points on 5k queries is 75 images and needs a PAIRED test, which
    needs the per-query predictions."""
    train_feats = torch.as_tensor(train_feats).to(device)
    val_feats = torch.as_tensor(val_feats).to(device)
    train_y = torch.as_tensor(train_y).to(device)
    val_y = torch.as_tensor(val_y).to(device)
    bank = F.normalize(train_feats.float(), dim=1).t().contiguous()
    labels = train_y.long()
    correct, n, preds = 0, 0, []
    for i in range(0, val_feats.shape[0], bs):
        q = F.normalize(val_feats[i:i + bs].float(), dim=1)
        pred = knn_predict(q, bank, labels, num_classes, knn_k, knn_t)
        correct += (pred[:, 0] == val_y[i:i + bs].long()).sum().item()
        n += pred.shape[0]
        if return_pred:
            preds.append(pred[:, 0].cpu())
    if return_pred:
        return correct / n, torch.cat(preds).numpy()
    return correct / n

def knn_self_test():
    """Two separable clusters -> kNN must be perfect (port of sslx module_self_test)."""
    g = torch.Generator().manual_seed(0)
    X = torch.cat([torch.randn(50, 8, generator=g) + 5, torch.randn(50, 8, generator=g) - 5])
    y = torch.cat([torch.zeros(50), torch.ones(50)]).long()
    acc = knn_topk_acc(X, y, X, y, num_classes=2, knn_k=10)
    assert acc == 1.0, f"kNN self-test failed: {acc}"
    return True
