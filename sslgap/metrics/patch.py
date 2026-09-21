"""Patch-token diagnostics (Berker 2026-09-07: "it is very interesting that only on segmentation task we fail ... over regularizing
the backbone feats, therefore patch feats? please do a quick analysis on the patch features"). Pure functions over stored
arrays of LAST-LAYER patch tokens (the seg protocol's features: forward_intermediates(indices=[-1], norm=True) at 512^2) and the
CLS token of the same images. Measures, each answering one question about the dense representation:
  rank_pool / rank_img   RankMe of the pooled patch matrix and the mean per-image RankMe, both / d  — how many directions the
                         patches use across images and within one image.
  within_frac            tr(Sigma_within) / tr(Sigma_total): the share of patch variance that is spatial structure inside images
                         rather than image-to-image content — a dense task lives on the within part.
  cos_cls / cos_pair     mean cosine of a patch to its image's CLS, and between two random patches of the same image (raw and
                         after global centering) — CLS dominance and patch uniformity.
  locality               mean cosine of 4-neighbour patch pairs minus random pairs (globally centered) — spatial coherence.
  norm_out_frac          fraction of patches with norm > 2x the image's median norm (high-norm artifact tokens); cls_norm_ratio.
  ncm_*                  nearest-class-mean accuracy and patch-level mIoU with ADE20k majority patch labels: raw tokens,
                         globally centered + unit-normed (cosine), and per-image centered (the image's mean removed) — a
                         training-free proxy of the linear seg read and of where the label information sits."""
import numpy as np
from sslgap.metrics.spectra import rankme


def patch_labels(mask, patch=16, ignore=-1):
    """(H, W) int mask -> (H//patch * W//patch,) majority label per patch; a patch whose majority is `ignore` stays `ignore`."""
    H, W = mask.shape; h, w = H // patch, W // patch
    blocks = mask[:h * patch, :w * patch].reshape(h, patch, w, patch).transpose(0, 2, 1, 3).reshape(h * w, patch * patch)
    out = np.full(h * w, ignore, dtype=np.int64)
    for i, b in enumerate(blocks):
        vals, cnt = np.unique(b, return_counts=True)
        out[i] = vals[np.argmax(cnt)]
    return out


def _cos(a, b):
    return (a * b).sum(-1) / (np.linalg.norm(a, axis=-1) * np.linalg.norm(b, axis=-1) + 1e-8)


def patch_stats(P, cls, grid, rng=None, n_rank=20000, n_pairs=2000):
    """P: (N, L, d) patch tokens, cls: (N, d), grid: side length (L = grid^2)."""
    rng = rng or np.random.default_rng(0)
    N, L, d = P.shape
    flat = P.reshape(-1, d); mu = flat.mean(0)
    sub = flat[rng.choice(len(flat), min(n_rank, len(flat)), replace=False)]
    rank_pool = rankme(sub) / d
    rank_img = float(np.mean([rankme(P[i]) / d for i in range(min(N, 64))]))
    img_mean = P.mean(1)                                        # (N, d)
    within = float(np.mean(((P - img_mean[:, None]) ** 2).sum(-1)))
    total = float(np.mean(((flat - mu) ** 2).sum(-1)))
    cos_cls = float(np.mean(_cos(P, cls[:, None, :])))
    i = rng.integers(0, N, n_pairs); a = rng.integers(0, L, n_pairs); b = rng.integers(0, L, n_pairs)
    cos_pair = float(np.mean(_cos(P[i, a], P[i, b])))
    Pc = P - mu
    cos_pair_c = float(np.mean(_cos(Pc[i, a], Pc[i, b])))
    r, c = np.divmod(a, grid); right = np.where(c + 1 < grid, a + 1, a - 1); down = np.where(r + 1 < grid, a + grid, a - grid)
    coh_adj = float(np.mean(np.concatenate([_cos(Pc[i, a], Pc[i, right]), _cos(Pc[i, a], Pc[i, down])])))
    norms = np.linalg.norm(P, axis=-1); med = np.median(norms, axis=1, keepdims=True)
    return {"rank_pool": rank_pool, "rank_img": rank_img, "within_frac": within / total, "cos_cls": cos_cls,
            "cos_pair": cos_pair, "cos_pair_c": cos_pair_c, "locality": coh_adj - cos_pair_c,
            "norm_out_frac": float(np.mean(norms > 2 * med)), "cls_norm_ratio": float(np.mean(np.linalg.norm(cls, axis=-1) / med[:, 0]))}


def ncm(train_P, train_y, test_P, test_y, num_classes, ignore=-1):
    """Nearest-class-mean on patch tokens. Returns (top-1 over labeled test patches, patch mIoU over classes present in test)."""
    d = train_P.shape[-1]; tp, ty = train_P.reshape(-1, d), train_y.reshape(-1); keep = ty != ignore
    means = np.zeros((num_classes, d), dtype=np.float64); cnt = np.zeros(num_classes)
    np.add.at(means, ty[keep], tp[keep]); np.add.at(cnt, ty[keep], 1)
    present = cnt > 0; means[present] /= cnt[present, None]
    xp, xy = test_P.reshape(-1, d), test_y.reshape(-1); k = xy != ignore; xp, xy = xp[k], xy[k]
    d2 = (xp ** 2).sum(1)[:, None] - 2 * xp @ means.T + (means ** 2).sum(1)[None]
    d2[:, ~present] = np.inf; pred = d2.argmin(1)
    acc = float(np.mean(pred == xy)); ious = []
    for c in np.unique(xy):
        inter = np.sum((pred == c) & (xy == c)); union = np.sum((pred == c) | (xy == c))
        ious.append(inter / union if union else 0.0)
    return acc, float(np.mean(ious))


def ncm_views(train_P, train_y, test_P, test_y, num_classes, train_cls=None, test_cls=None):
    """The NCM readings: raw; globally centered + unit-normed (cosine); per-image centered; and, when the CLS tokens are given,
    `pluscls` = each patch concatenated with its image's CLS (both parts unit-normed after global centering) — the test of
    whether handing the head the image-level context closes the gap (Berker 2026-09-07)."""
    d = train_P.shape[-1]; mu = train_P.reshape(-1, d).mean(0)
    unit = lambda X: (X - mu) / (np.linalg.norm(X - mu, axis=-1, keepdims=True) + 1e-8)
    perimg = lambda X: X - X.mean(1, keepdims=True)
    views = [("raw", lambda X, C: X), ("cos", lambda X, C: unit(X)), ("perimg", lambda X, C: perimg(X))]
    if train_cls is not None:
        muc = train_cls.mean(0)
        unitc = lambda C: (C - muc) / (np.linalg.norm(C - muc, axis=-1, keepdims=True) + 1e-8)
        views.append(("pluscls", lambda X, C: np.concatenate([unit(X), np.repeat(unitc(C)[:, None, :], X.shape[1], axis=1)], axis=-1)))
    out = {}
    for name, f in views:
        acc, miou = ncm(f(train_P, train_cls), train_y, f(test_P, test_cls), test_y, num_classes)
        out[f"ncm_{name}_acc"], out[f"ncm_{name}_miou"] = acc, miou
    return out
