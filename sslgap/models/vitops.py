"""Manual forward paths through a timm VisionTransformer + EMA update. Ported verbatim from
ssl_explore/sslx/vitops.py (see its docstring): I-JEPA needs context-only forwards, iBOT-style
mask-token substitution, DINO locals are below the training resolution. vitops_self_test asserts
exact equality against forward_features so a timm upgrade fails loudly at trainer start."""
import torch
from timm.layers import resample_abs_pos_embed


def vit_tokens(bb, x, keep=None, replace=None, mask_token=None):
    """x: [B,C,H,H] -> normed tokens [B,1+K,D] (cls first).
    replace: [B,N] bool — swap patch embeddings with mask_token BEFORE pos-add (iBOT).
    keep:    [B,K] long — retain only these patch indices AFTER pos-add (I-JEPA/MAE visible)."""
    z = bb.patch_embed(x)
    if replace is not None:
        z = torch.where(replace[..., None], mask_token.to(z.dtype), z)
    z = bb._pos_embed(z)
    if keep is not None:
        cls_tok, p = z[:, :1], z[:, 1:]
        p = torch.gather(p, 1, keep[..., None].expand(-1, -1, p.shape[-1]))
        z = torch.cat([cls_tok, p], 1)
    return bb.norm(bb.blocks(bb.norm_pre(bb.patch_drop(z))))


def vit_tokens_lowres(bb, x):
    """Forward crops smaller than the training resolution (DINO 64px locals)."""
    z = bb.patch_embed.proj(x).flatten(2).transpose(1, 2)
    g = x.shape[-1] // bb.patch_embed.patch_size[0]
    pos = resample_abs_pos_embed(bb.pos_embed, new_size=[g, g],
                                 old_size=bb.patch_embed.grid_size, num_prefix_tokens=1)
    z = torch.cat([bb.cls_token.expand(z.shape[0], -1, -1), z], 1) + pos
    return bb.norm(bb.blocks(bb.norm_pre(bb.patch_drop(bb.pos_drop(z)))))


@torch.no_grad()
def ema_update(teacher, student, m):
    for pt, ps in zip(teacher.parameters(), student.parameters()):
        pt.lerp_(ps, 1.0 - m)
    for bt, bs in zip(teacher.buffers(), student.buffers()):
        bt.copy_(bs)


@torch.no_grad()
def vitops_self_test(bb, device="cpu", img_size=128, local_size=64):
    was_training = bb.training
    bb.eval()
    if not isinstance(bb.patch_embed.norm, torch.nn.Identity):
        raise RuntimeError("vitops assumes no patch-embed norm — timm config changed?")
    if not (bb.num_prefix_tokens == 1 and not bb.no_embed_class and bb.reg_token is None):
        raise RuntimeError("vitops assumes 1 cls prefix token, no reg tokens, embedded cls")
    p = bb.patch_embed.patch_size[0]
    n_tok = (img_size // p) ** 2
    x = torch.randn(2, 3, img_size, img_size, device=device)
    keep = torch.arange(n_tok, device=device).expand(2, -1)
    if not torch.equal(vit_tokens(bb, x, keep=keep), bb.forward_features(x)):
        raise RuntimeError("vit_tokens diverged from forward_features — timm changed?")
    lo_tok = (local_size // p) ** 2
    lo = vit_tokens_lowres(bb, torch.randn(2, 3, local_size, local_size, device=device))
    if not (lo.shape[1] == 1 + lo_tok and torch.isfinite(lo).all()):
        raise RuntimeError("vit_tokens_lowres shape/finite check failed")
    bb.train(was_training)
