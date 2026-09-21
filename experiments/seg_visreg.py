"""ADE20k linear segmentation, the VISReg protocol (D-108.5; port of
third_party/visreg/downstream/segmentation/run_linear_seg.py, linear setup only).

Protocol preserved verbatim: frozen encoder, LAST-layer patch features via timm
forward_intermediates(norm=True); head = BN + 1x1 conv -> 150 classes, bilinear to
512^2; ADE20k labels shifted -1 with CE ignore_index=-1; train aug = RRC(0.5-1.0,
3/4..4/3) bicubic img / nearest mask @512 + hflip .5 + ImageNet norm; val = squash
Resize(512,512); AdamW lr 2e-3 bs 16, 40 ep, PolynomialLR(power .9, total_iters =
len(train)//bs*epochs — their single-GPU step-count quirk kept); bf16 autocast;
seed 42; mIoU = confusion matrix over classes present in val, best over epochs.

PORT_NOTES (declared deviations, all argued equivalence):
- single GPU, plain torch (donor used accelerate/DDP); nn.BatchNorm2d stands in for
  SyncBatchNorm (identical at world size 1).
- encoder.eval() + no_grad (donor left the frozen encoder in train mode; their ViTs
  carry no dropout/drop_path so modes coincide — ours may carry drop_path, so eval
  is the correct-and-equivalent choice).
- model loading through our checkpoint machinery: trunk weights land in a fresh
  dynamic-img-size timm ViT (strict state_dict load) so 512-input pos-embed
  interpolation is timm-native, same as the donor's create_backbone path.

Validation gate: timm vit_base_patch16_224.dino must land near the VISReg paper's
printed DINO-B/16 linear row (29.40 mIoU) BEFORE any house number is read.

  python experiments/seg_visreg.py --ckpt timm:vit_base_patch16_224.dino --tag in1k.pub.dinob400.seg
  python experiments/seg_visreg.py --ckpt outputs/<run>_ep100.pt --tag <run>.seg
CSV (per epoch, wall-safe): results/seg/<tag>.csv
"""
import argparse
import os

import numpy as np
import pandas as pd
import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ADE = os.path.expanduser("~/data/ade20k/ADEChallengeData2016")
IMG_SIZE, NUM_CLASSES, BS, EPOCHS, LR, SEED = 512, 150, 16, 40, 2e-3, 42


class ADE20K(Dataset):
    def __init__(self, split, train):
        d = "training" if split == "train" else "validation"
        self.img_dir, self.ann_dir = f"{ADE}/images/{d}", f"{ADE}/annotations/{d}"
        self.images = sorted(f for f in os.listdir(self.img_dir) if f.endswith(".jpg"))
        self.train = train
        self.norm = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])

    def __len__(self):
        return len(self.images)

    def __getitem__(self, i):
        img = Image.open(os.path.join(self.img_dir, self.images[i])).convert("RGB")
        mask = Image.open(os.path.join(self.ann_dir,
                                       self.images[i].replace(".jpg", ".png"))).convert("L")
        F_ = transforms.functional
        if self.train:
            y, x, h, w = transforms.RandomResizedCrop.get_params(
                img, scale=(0.5, 1.0), ratio=(3. / 4., 4. / 3.))
            img = F_.resized_crop(img, y, x, h, w, (IMG_SIZE, IMG_SIZE),
                                  interpolation=transforms.InterpolationMode.BICUBIC)
            mask = F_.resized_crop(mask, y, x, h, w, (IMG_SIZE, IMG_SIZE),
                                   interpolation=transforms.InterpolationMode.NEAREST)
            if torch.rand(1) > 0.5:
                img, mask = F_.hflip(img), F_.hflip(mask)
        else:
            img = F_.resize(img, (IMG_SIZE, IMG_SIZE),
                            interpolation=transforms.InterpolationMode.BICUBIC)
            mask = F_.resize(mask, (IMG_SIZE, IMG_SIZE),
                             interpolation=transforms.InterpolationMode.NEAREST)
        img = self.norm(F_.to_tensor(img))
        return img, torch.from_numpy(np.array(mask)).long() - 1   # 0=bg -> -1 ignored


def load_trunk(ckpt, device, adapter="native"):
    if ckpt.startswith("timm:"):
        trunk = timm.create_model(ckpt[5:], pretrained=True, num_classes=0,
                                  dynamic_img_size=True)
    else:
        from sslgap.ckpt import adapters
        import re
        tag = re.sub(r"_(ep\d+|last|best)\.pt$", "", os.path.basename(ckpt))
        loaded = adapters.load(adapter, ckpt, tag)
        src = loaded.branches[loaded.probed_branch].trunk
        trunk = timm.create_model(loaded.frame["model_name"], num_classes=0,
                                  dynamic_img_size=True)
        trunk.load_state_dict(src.state_dict(), strict=True)
    return trunk.eval().to(device).requires_grad_(False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--adapter", default="native")   # pubvit for OK-AI safetensors,
    args = ap.parse_args()                           # same lift the benches used
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    device = "cuda"
    torch.backends.cudnn.benchmark = True

    trunk = load_trunk(args.ckpt, device, args.adapter)
    dim = trunk.num_features
    head = nn.Sequential(nn.BatchNorm2d(dim),
                         nn.Conv2d(dim, NUM_CLASSES, kernel_size=1)).to(device)

    workers = int(os.environ.get("SLURM_CPUS_PER_TASK", 12))
    tr = DataLoader(ADE20K("train", True), batch_size=BS, shuffle=True,
                    num_workers=workers, pin_memory=True, persistent_workers=True)
    va = DataLoader(ADE20K("val", False), batch_size=BS, shuffle=False,
                    num_workers=workers, pin_memory=True, persistent_workers=True)

    opt = torch.optim.AdamW(head.parameters(), lr=LR)
    sched = torch.optim.lr_scheduler.PolynomialLR(
        opt, total_iters=len(tr.dataset) // BS * EPOCHS, power=0.9)
    crit = nn.CrossEntropyLoss(ignore_index=-1)

    def feats(x):
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            _, inter = trunk.forward_intermediates(
                x, indices=[-1], norm=True, return_prefix_tokens=False)
        p = inter[0]
        if p.ndim == 3:
            B, N, C = p.shape
            s = int(N ** 0.5)
            p = p.transpose(1, 2).reshape(B, C, s, s)
        return p.float()

    out_csv = f"{ROOT}/results/seg/{args.tag}.csv"
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    best = 0.0
    for ep in range(1, EPOCHS + 1):
        head.train()
        tot = 0.0
        for x, m in tr:
            x, m = x.to(device, non_blocking=True), m.to(device, non_blocking=True)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                logits = F.interpolate(head(feats(x)), size=(IMG_SIZE, IMG_SIZE),
                                       mode="bilinear", align_corners=False)
                loss = crit(logits, m)
            opt.zero_grad()
            loss.backward()
            opt.step()
            sched.step()
            tot += loss.item()

        head.eval()
        conf = torch.zeros(NUM_CLASSES, NUM_CLASSES, dtype=torch.long)
        with torch.no_grad():
            for x, m in va:
                x = x.to(device, non_blocking=True)
                logits = F.interpolate(head(feats(x)), size=(IMG_SIZE, IMG_SIZE),
                                       mode="bilinear", align_corners=False)
                pred = logits.argmax(1).cpu()
                valid = m != -1
                idx = m[valid] * NUM_CLASSES + pred[valid]
                conf += torch.bincount(idx, minlength=NUM_CLASSES ** 2) \
                    .reshape(NUM_CLASSES, NUM_CLASSES)
        inter_ = conf.diag()
        union = conf.sum(0) + conf.sum(1) - inter_
        iou = inter_ / union.float().clamp(min=1e-6)
        present = conf.sum(1) > 0
        miou = iou[present].mean().item()
        best = max(best, miou)
        pd.DataFrame([{"tag": args.tag, "epoch": ep, "train_loss": tot / len(tr),
                       "miou": miou, "best": best}]) \
            .to_csv(out_csv, mode="a", header=not os.path.exists(out_csv), index=False)
        print(f"[seg] {args.tag} ep{ep}/{EPOCHS} loss={tot / len(tr):.4f} "
              f"miou={miou:.4f} best={best:.4f}", flush=True)
    print(f"[seg] {args.tag} BEST mIoU {best:.4f}", flush=True)


if __name__ == "__main__":
    main()
