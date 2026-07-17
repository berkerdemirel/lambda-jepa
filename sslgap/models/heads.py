"""Head modules + tap wrappers. A "taps" module maps one trunk feature (cls or gap) to a dict of
named z-space tensors — the guillotine axis through the head (PROTOCOL §3).

DINOHead is ported verbatim from ssl_explore/sslx/dinov2.py (which follows Caron et al. 2021,
norm_last_layer semantics included) so the DINO-control checkpoints load unchanged."""
import torch.nn as nn
import torch.nn.functional as F


class DINOHead(nn.Module):
    """3-layer MLP -> L2-normalized bottleneck -> weight-normed prototypes. Port: sslx/dinov2.py."""

    def __init__(self, in_dim=384, hidden=2048, bottleneck=256, K=4096, norm_last_layer=True):
        super().__init__()
        self.mlp = nn.Sequential(nn.Linear(in_dim, hidden), nn.GELU(),
                                 nn.Linear(hidden, hidden), nn.GELU(),
                                 nn.Linear(hidden, bottleneck))
        self.last = nn.utils.parametrizations.weight_norm(nn.Linear(bottleneck, K, bias=False))
        g = self.last.parametrizations.weight.original0
        g.data.fill_(1.0)
        if norm_last_layer:
            g.requires_grad_(False)

    def forward(self, x):
        return self.last(F.normalize(self.mlp(x), dim=-1))


class DINOLinearHead(nn.Module):
    """E17 h-pull (D-039): the small LINEAR prototype head at the backbone h — DINO's final stage
    (L2-normalize then weight-normed prototypes) WITHOUT the MLP+bottleneck. "linear K_small"
    (Berker 2026-07-15). Same `last`/`original0` structure as DINOHead so the ep0-freeze +
    norm_last_layer gain-freeze logic (dino.on_epoch_start) applies unchanged."""

    def __init__(self, in_dim=384, K=512, norm_last_layer=True):
        super().__init__()
        self.last = nn.utils.parametrizations.weight_norm(nn.Linear(in_dim, K, bias=False))
        g = self.last.parametrizations.weight.original0
        g.data.fill_(1.0)
        if norm_last_layer:
            g.requires_grad_(False)

    def forward(self, x):
        return self.last(F.normalize(x, dim=-1))


class DinoHeadTaps(nn.Module):
    """z-taps through a DINOHead: post-GELU hiddens + the (raw) bottleneck. Prototype logits
    (K up to 65k) are NOT emitted — recompute when a metric needs them:
    `l2norm(bottleneck) @ head.last.weight.T` (D-005)."""

    def __init__(self, head: DINOHead):
        super().__init__()
        self.head = head

    def forward(self, x):
        m = self.head.mlp
        t1 = m[1](m[0](x))
        t2 = m[3](m[2](t1))
        return {"dino.tap1": t1, "dino.tap2": t2, "dino.bottleneck": m[4](t2)}


class TVMLPTaps(nn.Module):
    """z-taps through a torchvision.ops.MLP (the lejepa projector: Linear-BN-ReLU blocks
    + final Linear[+Dropout]). Taps = post-activation hidden states; "out" = final linear output."""

    def __init__(self, mlp: nn.Sequential, prefix="proj"):
        super().__init__()
        self.mlp = mlp
        self.prefix = prefix

    def forward(self, x):
        out, k = {}, 0
        # depth-0 projector (E10 D0) is a bare nn.Identity, not a Sequential -> proj.out = input
        layers = self.mlp if isinstance(self.mlp, nn.Sequential) else [self.mlp]
        for layer in layers:
            x = layer(x)
            if isinstance(x, tuple):  # no torchvision layer returns tuples; guard for exotic mlps
                x = x[0]
            if isinstance(layer, (nn.ReLU, nn.GELU)):
                k += 1
                out[f"{self.prefix}.tap{k}"] = x
        out[f"{self.prefix}.out"] = x
        return out


class ByolHeads(nn.Module):
    """BYOL student head stack: projector taps, then predictor taps chained on proj.out
    (z.proj.tap1, z.proj.out, z.pred.tap1, z.pred.out — PROTOCOL §3; loss space = pred.out)."""

    def __init__(self, proj: nn.Sequential, pred: nn.Sequential):
        super().__init__()
        self.proj = TVMLPTaps(proj, prefix="proj")
        self.pred = TVMLPTaps(pred, prefix="pred")

    def forward(self, cls):
        p = self.proj(cls)
        return {**p, **self.pred(p["proj.out"])}


class LejepaHeads(nn.Module):
    """The lejepa-minimal head stack: trunk CLS (384) -> timm classifier Linear 384->512
    (= the recipe's "emb"; our tap z.embed, D-003) -> torchvision MLP projector (z.proj.*)."""

    def __init__(self, embed_linear: nn.Linear, proj_mlp: nn.Sequential):
        super().__init__()
        self.embed = embed_linear
        self.proj = TVMLPTaps(proj_mlp, prefix="proj")

    def forward(self, cls):
        e = self.embed(cls)
        return {"embed": e, **self.proj(e)}


class LinearTap(nn.Module):
    """Single named linear tap — the deitlite classifier (z.logits): the supervised anchor's
    loss space is its 100-way logit layer, stored so the matrix has a z.final for it."""

    def __init__(self, linear: nn.Linear, name="logits"):
        super().__init__()
        self.linear = linear
        self.name = name

    def forward(self, x):
        return {self.name: self.linear(x)}
