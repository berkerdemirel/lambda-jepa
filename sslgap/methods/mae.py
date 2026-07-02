"""MAE (He et al. 2022) — toy-rung instance per D-012: canonical pipeline (75% random masking,
encoder sees visible patches only, lightweight decoder, MSE on per-patch-normalized pixels of
masked patches), minimal aug (RRC 0.2-1.0 + flip), house AdamW. Toy adaptations (documented):
decoder canonical 512x8x16 with FIXED sincos pos incl. zero cls row and cls kept through the
decoder (verified vs models_mae.py@efb2a80 forward_decoder — scatter == the ids_restore unshuffle);
drop_path 0 (paper pretrain). Expected phenotype: WEAK linear probe (paper's own finding) — the
monitor floor for MAE is 'well above chance', not parity with the contrastive family."""
import torch
import torch.nn as nn
from timm.layers import trunc_normal_

from sslgap.models.posembed import get_2d_sincos_pos_embed
from timm.models.vision_transformer import Block

from sslgap.data import ViewsDataset, minaug_stack
from sslgap.methods._common import house_scheduler, trunk_arch
from sslgap.methods.base import SSLMethod
from sslgap.models.backbones import build_vit_trunk
from sslgap.models.vitops import vit_tokens


class MAEDecoder(nn.Module):
    """Canonical MAE decoder (models_mae.py@efb2a80 forward_decoder): decoder_embed(cls+visible) ->
    mask tokens at masked positions (scatter == the official ids_restore unshuffle) -> cls kept in
    the sequence -> + FIXED sincos pos (zero cls row) -> blocks -> norm -> pred -> cls stripped."""

    def __init__(self, patch=8, in_dim=384, dim=512, depth=8, heads=16, n_patches=256):
        super().__init__()
        self.embed = nn.Linear(in_dim, dim)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, dim))
        grid = int(n_patches ** 0.5)
        self.register_buffer("pos", get_2d_sincos_pos_embed(dim, grid, cls_token=True)[None])
        self.blocks = nn.Sequential(*[Block(dim, heads, qkv_bias=True) for _ in range(depth)])
        self.norm = nn.LayerNorm(dim)
        self.out = nn.Linear(dim, patch * patch * 3)
        trunc_normal_(self.mask_token, std=0.02)

    def forward(self, cls_vis_tokens, keep_idx, n_patches):
        """cls_vis_tokens [B,1+Kv,in_dim] (cls first), keep_idx [B,Kv] -> preds [B,N,p*p*3]."""
        z = self.embed(cls_vis_tokens)
        B, d = z.shape[0], z.shape[-1]
        cls_tok, vis = z[:, :1], z[:, 1:]
        full = self.mask_token.expand(B, n_patches, -1).clone()
        full.scatter_(1, keep_idx[..., None].expand(-1, -1, d), vis)
        x = torch.cat([cls_tok, full], 1) + self.pos
        x = self.norm(self.blocks(x))[:, 1:]
        return self.out(x)


def mae_decoder(patch=8, in_dim=384, dim=512, depth=8, heads=16, n_patches=256):
    return MAEDecoder(patch, in_dim, dim, depth, heads, n_patches)


def patchify(imgs, p):
    B, C, H, W = imgs.shape
    x = imgs.reshape(B, C, H // p, p, W // p, p)
    return x.permute(0, 2, 4, 3, 5, 1).reshape(B, (H // p) * (W // p), p * p * C)


class MAE(SSLMethod):
    name = "mae"

    def __init__(self, cfg, frame):
        super().__init__(cfg, frame)
        self.n_patches = (frame.img_size // self.cfg.patch) ** 2

    def build_modules(self):
        trunk = build_vit_trunk(self.frame.model_name, self.frame.img_size, drop_path_rate=0.0)
        dec = mae_decoder(self.cfg.patch, 384, self.cfg.dec_dim, self.cfg.dec_depth,
                          self.cfg.dec_heads, self.n_patches)
        return nn.ModuleDict({"backbone": trunk, "decoder": dec})

    def arch(self):
        return {"backbone": trunk_arch(self.frame, 0.0),
                "decoder": {"class": "sslgap.methods.mae.mae_decoder",
                            "kwargs": {"patch": self.cfg.patch, "in_dim": 384,
                                       "dim": self.cfg.dec_dim, "depth": self.cfg.dec_depth,
                                       "heads": self.cfg.dec_heads,
                                       "n_patches": self.n_patches}}}

    def build_train_dataset(self):
        return ViewsDataset(self.frame.dataset, "train", V=1, img_size=self.frame.img_size,
                            data_root=self.frame.data_root,
                            transforms=[minaug_stack(self.frame.img_size, scale=(0.2, 1.0))])

    def param_groups(self, modules):
        return [{"params": [p for m in modules.values() for p in m.parameters()],
                 "lr": self.cfg.lr, "weight_decay": self.cfg.wd}]

    def build_scheduler(self, optimizer, steps_per_epoch, total_steps):
        return house_scheduler(optimizer, steps_per_epoch, total_steps,
                               self.cfg.warmup_ep, self.cfg.eta_min)

    def training_step(self, modules, views, device):
        x = views[:, 0]                                            # [B,C,H,W]
        B = x.shape[0]
        n_keep = int(self.n_patches * (1 - self.cfg.mask_ratio))
        noise = torch.rand(B, self.n_patches, device=device)
        shuffle = noise.argsort(1)
        keep_idx, mask_idx = shuffle[:, :n_keep], shuffle[:, n_keep:]
        cls_vis = vit_tokens(modules["backbone"], x, keep=keep_idx)      # [B,1+Kv,384] cls first
        pred = modules["decoder"](cls_vis, keep_idx, self.n_patches)
        target = patchify(x, self.cfg.patch)
        if self.cfg.norm_pix_loss:
            mu, var = target.mean(-1, keepdim=True), target.var(-1, keepdim=True)
            target = (target - mu) / (var + 1e-6).sqrt()
        d = pred.shape[-1]
        idx = mask_idx[..., None].expand(-1, -1, d)
        loss = ((torch.gather(pred, 1, idx) - torch.gather(target, 1, idx)) ** 2).mean()
        # probe = full-image GAP so the monitor matches eval features; .clone() lifts the tensor
        # out of inference mode (Linear saves its input for backward -> probe grads need it).
        probe_feats = self.eval_features(modules, x, device).clone()
        return ({"loss": loss, "recon": loss}, probe_feats, 1)

    @torch.inference_mode()
    def eval_features(self, modules, x, device):
        feats = modules["backbone"].forward_features(x)
        return feats[:, 1:].mean(1)                                 # GAP (paper probes avgpool)

    def probe_dim(self):
        return 384
