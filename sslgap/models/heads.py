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
        for layer in self.mlp:
            x = layer(x)
            if isinstance(x, tuple):  # no torchvision layer returns tuples; guard for exotic mlps
                x = x[0]
            if isinstance(layer, (nn.ReLU, nn.GELU)):
                k += 1
                out[f"{self.prefix}.tap{k}"] = x
        out[f"{self.prefix}.out"] = x
        return out


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
