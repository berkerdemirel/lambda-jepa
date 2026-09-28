"""ViT trunks (timm) and the backbone readout."""
import timm
import torch

def build_vit_trunk(model_name, img_size, dynamic_img_size=False, drop_path_rate=0.0, **kw):
    """Headless ViT (num_classes=0) — h never contains a classifier layer. kw overrides
    timm model args for public checkpoints whose architecture deviates from the timm
    default under the same tensor shapes (e.g. MoCo-v3 ViT-S: num_heads=12 vs timm's 6)."""
    return timm.create_model(model_name, pretrained=False, num_classes=0,
                             img_size=img_size, dynamic_img_size=dynamic_img_size,
                             drop_path_rate=drop_path_rate, **kw)

@torch.inference_mode()
def trunk_features(trunk, x, h_layers=()):
    """One frozen pass -> {"cls": [B,D], "gap": [B,D], "tokens": [B,N,D], "seq": [B,npre+N,D],
    and per l in h_layers: "cls.L<l>", "gap.L<l>"} (1-indexed block layers, norm=True).
    "seq" is the full normed sequence incl. prefix — the mask-ratio-0 decoder input for MAE
    (vit_tokens(keep=all) == forward_features, asserted by vitops_self_test)."""
    npre = getattr(trunk, "num_prefix_tokens", 1)
    feats = trunk.forward_features(x)
    out = {"cls": feats[:, 0], "gap": feats[:, npre:].mean(1), "tokens": feats[:, npre:],
           "seq": feats}
    if h_layers:
        inter = trunk.get_intermediate_layers(x, n=[l - 1 for l in h_layers],
                                              return_prefix_tokens=True, norm=True)
        for l, (patch, prefix) in zip(h_layers, inter):
            out[f"cls.L{l:02d}"] = prefix[:, 0]
            out[f"gap.L{l:02d}"] = patch.mean(1)
    return out
