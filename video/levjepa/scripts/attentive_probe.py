"""ImageNet-1k attentive probe on a video checkpoint (V-JEPA frozen-evaluation protocol)."""
import argparse
import csv
import math
import os
import time

import torch
import contextlib
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
from timm.data import create_transform
from torchvision.datasets import ImageFolder

import module as vit_models

NORM = ((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))

class CrossAttention(nn.Module):
    def __init__(self, dim, num_heads, qkv_bias=True):
        super().__init__()
        self.num_heads = num_heads
        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.kv = nn.Linear(dim, dim * 2, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim)

    def forward(self, q, x):
        B, n, C = q.shape
        q = self.q(q).reshape(B, n, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        N = x.shape[1]
        kv = self.kv(x).reshape(B, N, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q = F.scaled_dot_product_attention(q, kv[0], kv[1])
        return self.proj(q.transpose(1, 2).reshape(B, n, C))

class AttentiveClassifier(nn.Module):

    def __init__(self, embed_dim, num_heads, num_classes=1000, mlp_ratio=4.0, init_std=0.02):
        super().__init__()
        self.query_tokens = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.norm1 = nn.LayerNorm(embed_dim)
        self.xattn = CrossAttention(embed_dim, num_heads)
        self.norm2 = nn.LayerNorm(embed_dim)
        hidden = int(embed_dim * mlp_ratio)
        self.mlp = nn.Sequential(nn.Linear(embed_dim, hidden), nn.GELU(), nn.Linear(hidden, embed_dim))
        self.linear = nn.Linear(embed_dim, num_classes)
        nn.init.trunc_normal_(self.query_tokens, std=init_std)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=init_std)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LayerNorm):
                nn.init.ones_(m.weight); nn.init.zeros_(m.bias)
        with torch.no_grad():
            self.xattn.proj.weight.div_(math.sqrt(2.0))
            self.mlp[2].weight.div_(math.sqrt(2.0))

    def forward(self, x):
        q = self.query_tokens.expand(x.shape[0], -1, -1)
        q = q + self.xattn(q, self.norm1(x))
        q = q + self.mlp(self.norm2(q))
        return self.linear(q.squeeze(1))

class WarmupCosineSchedule:
    def __init__(self, optimizer, warmup_steps, start_lr, ref_lr, T_max, final_lr=0.0):
        self.optimizer, self.start_lr, self.ref_lr, self.final_lr = optimizer, start_lr, ref_lr, final_lr
        self.warmup_steps, self.T_max, self._step = warmup_steps, T_max - warmup_steps, 0.0

    def step(self):
        self._step += 1
        if self._step < self.warmup_steps:
            lr = self.start_lr + self._step / max(1, self.warmup_steps) * (self.ref_lr - self.start_lr)
        else:
            progress = (self._step - self.warmup_steps) / max(1, self.T_max)
            lr = max(self.final_lr, self.final_lr + (self.ref_lr - self.final_lr) * 0.5 * (1.0 + math.cos(math.pi * progress)))
        for g in self.optimizer.param_groups:
            g["lr"] = lr
        return lr

class CosineWDSchedule:
    def __init__(self, optimizer, ref_wd, T_max, final_wd=0.0):
        self.optimizer, self.ref_wd, self.final_wd, self.T_max, self._step = optimizer, ref_wd, final_wd, T_max, 0.0

    def step(self):
        self._step += 1
        wd = self.final_wd + (self.ref_wd - self.final_wd) * 0.5 * (1.0 + math.cos(math.pi * self._step / self.T_max))
        wd = max(self.final_wd, wd) if self.final_wd <= self.ref_wd else min(self.final_wd, wd)
        for g in self.optimizer.param_groups:
            if not g.get("WD_exclude", False):
                g["weight_decay"] = wd
        return wd

def load_encoder(ckpt_path, weights):
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    hp = ck["hyper_parameters"]
    m = hp["model"]
    enc = getattr(vit_models, m["name"])(img_size=m["img_size"], patch_size=m["patch_size"], num_frames=m["num_frames"],
                                         tubelet_size=m["tubelet_size"], use_rope=m.get("use_rope", True),
                                         token_drop_rate=0.0, attn_mode=m.get("attn_mode", "full"))
    sd = ck["state_dict_ema"] if weights == "ema" else ck["state_dict"]
    enc_sd = {k[len("encoder."):]: v for k, v in sd.items() if k.startswith("encoder.")}
    missing, unexpected = enc.load_state_dict(enc_sd, strict=False)
    assert not unexpected and all(k == "pos_embed" for k in missing), (missing, unexpected)
    return enc, hp

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--weights", default="ema", choices=["ema", "student"])
    ap.add_argument("--data", default=os.path.expanduser("~/data/imagenet"))
    ap.add_argument("--out", required=True, help="CSV of per-epoch train/val top-1")
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--bs", type=int, default=16, help="per GPU (V-JEPA: 16 x 64 GPUs)")
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--wd", type=float, default=1e-3)
    ap.add_argument("--final-wd", type=float, default=1e-6)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--limit", type=int, default=0, help="smoke: images per split")
    ap.add_argument("--tokens", default="all", choices=["all", "cls", "patches"])
    args = ap.parse_args()
    ddp = "RANK" in os.environ
    if ddp:
        dist.init_process_group("nccl")
        rank, world = dist.get_rank(), dist.get_world_size()
        torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    else:
        rank, world = 0, 1
    dev = torch.device("cuda")
    enc, hp = load_encoder(args.ckpt, args.weights)
    enc.to(dev).eval()
    for p in enc.parameters():
        p.requires_grad_(False)
    T_frames = enc.num_frames
    clf = AttentiveClassifier(enc.embed_dim, enc.num_heads).to(dev)
    if ddp:
        clf = nn.parallel.DistributedDataParallel(clf, device_ids=[dev.index])
    train_tf = create_transform(input_size=224, is_training=True, auto_augment="original", interpolation="bicubic",
                                re_prob=0.25, re_mode="pixel", re_count=1, mean=NORM[0], std=NORM[1])
    val_tf = T.Compose([T.Resize(int(224 * 256 / 224)), T.CenterCrop(224), T.ToTensor(), T.Normalize(*NORM)])
    train_ds = ImageFolder(os.path.join(args.data, "train"), train_tf)
    val_ds = ImageFolder(os.path.join(args.data, "val"), val_tf)
    if args.limit:
        g = torch.Generator().manual_seed(0)
        train_ds = torch.utils.data.Subset(train_ds, torch.randperm(len(train_ds), generator=g)[:args.limit].tolist())
        val_ds = torch.utils.data.Subset(val_ds, torch.randperm(len(val_ds), generator=g)[:args.limit].tolist())
    samp = lambda ds, shuffle: torch.utils.data.distributed.DistributedSampler(ds, shuffle=shuffle) if ddp else None
    train_loader = torch.utils.data.DataLoader(train_ds, batch_size=args.bs, shuffle=not ddp, sampler=samp(train_ds, True),
                                               num_workers=args.workers, pin_memory=True, persistent_workers=True, drop_last=False)
    val_loader = torch.utils.data.DataLoader(val_ds, batch_size=args.bs, shuffle=False, sampler=samp(val_ds, False),
                                             num_workers=args.workers, pin_memory=True, persistent_workers=True)
    params = list((clf.module if ddp else clf).named_parameters())
    groups = [{"params": [p for n, p in params if "bias" not in n and p.ndim != 1]},
              {"params": [p for n, p in params if "bias" in n or p.ndim == 1], "WD_exclude": True, "weight_decay": 0}]
    opt = torch.optim.AdamW(groups)
    ipe = len(train_loader)
    lr_s = WarmupCosineSchedule(opt, 0, args.lr, args.lr, args.epochs * ipe, final_lr=0.0)
    wd_s = CosineWDSchedule(opt, args.wd, args.epochs * ipe, final_wd=args.final_wd)
    if rank == 0:
        print(f"[attn-probe] {args.ckpt} weights={args.weights} enc={hp['model']['name']} dim={enc.embed_dim} heads={enc.num_heads} "
              f"frames={T_frames} | train {len(train_ds)} val {len(val_ds)} | bs {args.bs}x{world}={args.bs * world} ipe {ipe} epochs {args.epochs}", flush=True)
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)

    def encode(imgs):
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            tok = enc(imgs.unsqueeze(2).repeat(1, 1, T_frames, 1, 1))
            return tok if args.tokens == "all" else (tok[:, :1] if args.tokens == "cls" else tok[:, 1:])

    def run(loader, training, epoch):
        clf.train(training)
        if ddp and training:
            loader.sampler.set_epoch(epoch)
        correct = torch.zeros((), device=dev); seen = torch.zeros((), device=dev)
        for imgs, labels in loader:
            imgs, labels = imgs.to(dev, non_blocking=True), labels.to(dev, non_blocking=True)
            if training:
                lr_s.step(); wd_s.step()
            tokens = encode(imgs)
            with torch.autocast("cuda", dtype=torch.bfloat16), (contextlib.nullcontext() if training else torch.no_grad()):
                logits = clf(tokens)
            loss = F.cross_entropy(logits.float(), labels)
            if training:
                opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
            correct += (logits.argmax(1) == labels).sum(); seen += len(labels)
        if ddp:
            dist.all_reduce(correct); dist.all_reduce(seen)
        return 100.0 * correct.item() / seen.item()

    rows = []
    for ep in range(args.epochs):
        t0 = time.time()
        tr = run(train_loader, True, ep)
        va = run(val_loader, False, ep)
        if rank == 0:
            print(f"[attn-probe] ep{ep + 1}/{args.epochs} train_top1={tr:.2f} val_top1={va:.2f} lr={opt.param_groups[0]['lr']:.2e} ({(time.time() - t0) / 60:.1f} min)", flush=True)
            rows.append({"ckpt": args.ckpt, "weights": args.weights, "epoch": ep + 1, "train_top1": tr, "val_top1": va})
            with open(args.out, "w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    if rank == 0:
        print(f"[attn-probe] final val_top1={rows[-1]['val_top1']:.2f} best={max(r['val_top1'] for r in rows):.2f} -> {args.out}", flush=True)
    if ddp:
        dist.destroy_process_group()

if __name__ == "__main__":
    main()
