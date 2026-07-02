"""ViT trunks and the h-space readout.

h is the trunk's forward_features output (final LayerNorm applied by timm): h.cls = prefix token,
h.gap = mean over patch tokens (DECISIONS D-003 — anything trainable after this is head).
Per-layer readout for guillotine curves uses timm's get_intermediate_layers with norm=True
(the DINO convention: intermediate blocks passed through the final norm)."""
import timm
import torch


def build_vit_trunk(model_name, img_size, dynamic_img_size=False, drop_path_rate=0.0):
    """Headless ViT (num_classes=0) — h never contains a classifier layer."""
    return timm.create_model(model_name, pretrained=False, num_classes=0,
                             img_size=img_size, dynamic_img_size=dynamic_img_size,
                             drop_path_rate=drop_path_rate)


@torch.inference_mode()
def trunk_features(trunk, x, h_layers=()):
    """One frozen pass -> {"cls": [B,D], "gap": [B,D], "tokens": [B,N,D],
    and per l in h_layers: "cls.L<l>", "gap.L<l>"} (1-indexed block layers, norm=True)."""
    npre = getattr(trunk, "num_prefix_tokens", 1)
    feats = trunk.forward_features(x)                      # [B, npre+N, D], final norm applied
    out = {"cls": feats[:, 0], "gap": feats[:, npre:].mean(1), "tokens": feats[:, npre:]}
    if h_layers:
        inter = trunk.get_intermediate_layers(x, n=[l - 1 for l in h_layers],
                                              return_prefix_tokens=True, norm=True)
        for l, (patch, prefix) in zip(h_layers, inter):
            out[f"cls.L{l:02d}"] = prefix[:, 0]
            out[f"gap.L{l:02d}"] = patch.mean(1)
    return out
