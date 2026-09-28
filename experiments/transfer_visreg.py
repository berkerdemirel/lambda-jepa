"""Transfer linear probes under the VISReg protocol (concatenated CLS of the last four blocks, 13 learning rates, 10 epochs, eight datasets)."""
import os
import random
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.amp import autocast
from torch.utils.data import DataLoader, TensorDataset
from torchvision import datasets as tvd
from torchvision import transforms as T

from sslgap.ckpt import adapters

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.expanduser("~/data/ssltransfer")
BICUBIC = T.InterpolationMode.BICUBIC
NORM = T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
TRAIN_TF = T.Compose([T.RandomResizedCrop(224, scale=(0.08, 1.0), interpolation=BICUBIC),
                      T.RandomHorizontalFlip(), T.ToTensor(), NORM])
TEST_TF = T.Compose([T.Resize(256, interpolation=BICUBIC), T.CenterCrop(224),
                     T.ToTensor(), NORM])

BASE_LRS = [1e-4, 2e-4, 5e-4, 1e-3, 2e-3, 5e-3, 1e-2, 2e-2, 5e-2, 0.1, 0.2, 0.3, 0.5]
BS = 32
SEED = 42

BENCH = {
    "dtd":      (dict(split="train", partition=1), dict(split="test", partition=1), 1880, 1880, 47),
    "aircraft": (dict(split="trainval"), dict(split="test"), 6667, 3333, 100),
    "cars":     (None, None, 8144, 8041, 196),
    "cifar10":  (dict(train=True), dict(train=False), 50000, 10000, 10),
    "cifar100": (dict(train=True), dict(train=False), 50000, 10000, 100),
    "flowers":  (dict(split="train"), dict(split="test"), 1020, 6149, 102),
    "food":     (dict(split="train"), dict(split="test"), 75750, 25250, 101),
    "pets":     (dict(split="trainval"), dict(split="test"), 3680, 3669, 37),
}
TVSETS = {"cifar10": tvd.CIFAR10, "cifar100": tvd.CIFAR100, "dtd": tvd.DTD,
          "aircraft": tvd.FGVCAircraft, "flowers": tvd.Flowers102,
          "food": tvd.Food101, "pets": tvd.OxfordIIITPet}

class HFCars(torch.utils.data.Dataset):

    def __init__(self, split, tf):
        from datasets import load_dataset
        self.ds = load_dataset("tanganke/stanford_cars", split=split)
        self.tf = tf

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, i):
        r = self.ds[i]
        return self.tf(r["image"].convert("RGB")), r["label"]

def build_split(name, spec, train):
    tf = TRAIN_TF if train else TEST_TF
    if name == "cars":
        return HFCars("train" if train else "test", tf)
    return TVSETS[name](root=DATA, download=True, transform=tf, **spec)

class ConcatCLS(nn.Module):

    def __init__(self, trunk):
        super().__init__()
        self.trunk = trunk
        self.dim = 4 * trunk.embed_dim

    def forward(self, x):
        _, inter = self.trunk.forward_intermediates(
            x, indices=[-4, -3, -2, -1], return_prefix_tokens=True, norm=True)
        return torch.cat([p[:, 0, :] for _, p in inter], dim=-1)

class MultiLR(nn.Module):
    def __init__(self, in_dim, n_cls, n_heads):
        super().__init__()
        self.heads = nn.ModuleList([
            nn.Sequential(nn.BatchNorm1d(in_dim, affine=True), nn.Linear(in_dim, n_cls))
            for _ in range(n_heads)])
        for h in self.heads:
            h[1].weight.data.normal_(0.0, 0.01)
            h[1].bias.data.zero_()

    def forward(self, x):
        return [h(x) for h in self.heads]

def build_encoder(src):
    kind, _, ref = src.partition(":")
    if kind == "timm":
        import timm
        trunk = timm.create_model(ref, pretrained=True, num_classes=0,
                                  dynamic_img_size=True, drop_path_rate=0.1)
    else:
        loaded = adapters.load(kind, os.path.expanduser(ref), run_id=os.path.basename(ref))
        trunk = loaded.branches[loaded.probed_branch].trunk
    enc = ConcatCLS(trunk).cuda().eval()
    for p in enc.parameters():
        p.requires_grad_(False)
    return enc

@torch.no_grad()
def extract_test(enc, ds):
    feats, ys = [], []
    loader = DataLoader(ds, batch_size=BS, num_workers=8, shuffle=False,
                        persistent_workers=False, pin_memory=True)
    for x, y in loader:
        with autocast("cuda", dtype=torch.bfloat16):
            feats.append(enc(x.cuda(non_blocking=True)))
        ys.append(torch.as_tensor(y))
    return torch.cat(feats), torch.cat(ys).cuda()

def run_dataset(enc, name, epochs, fh, tag):
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    np.random.seed(SEED)
    random.seed(SEED)
    tr_spec, te_spec, n_tr, n_te, n_cls = BENCH[name]
    train_ds = build_split(name, tr_spec, train=True)
    test_ds = build_split(name, te_spec, train=False)
    assert len(train_ds) == n_tr and len(test_ds) == n_te, \
        f"non-canonical {name} split: {len(train_ds)}/{len(test_ds)} vs {n_tr}/{n_te}"
    print(f"[tvlp] {name}: train={n_tr} test={n_te} classes={n_cls}", flush=True)

    test_feats, test_labels = extract_test(enc, test_ds)

    lrs = [lr * BS / 256.0 for lr in BASE_LRS]
    heads = MultiLR(enc.dim, n_cls, len(lrs)).cuda()
    opts = [torch.optim.SGD(h.parameters(), lr=lr, momentum=0.9, weight_decay=0)
            for h, lr in zip(heads.heads, lrs)]
    scheds = [torch.optim.lr_scheduler.CosineAnnealingLR(o, T_max=epochs, eta_min=0)
              for o in opts]

    train_loader = DataLoader(train_ds, batch_size=BS, shuffle=True, num_workers=8,
                              drop_last=True, persistent_workers=True, pin_memory=True)
    for ep in range(epochs):
        heads.train()
        for x, y in train_loader:
            y = y.cuda(non_blocking=True)
            with torch.no_grad(), autocast("cuda", dtype=torch.bfloat16):
                feat = enc(x.cuda(non_blocking=True))
            with autocast("cuda", dtype=torch.bfloat16):
                outs = heads(feat)
            for opt, out in zip(opts, outs):
                opt.zero_grad()
                F.cross_entropy(out, y).backward()
                opt.step()
        for s in scheds:
            s.step()
        print(f"[tvlp] {name} ep{ep + 1}/{epochs}", flush=True)

    heads.eval()
    accs = []
    loader = DataLoader(TensorDataset(test_feats, test_labels), batch_size=BS)
    with torch.no_grad():
        preds = [[] for _ in lrs]
        for feat, _ in loader:
            with autocast("cuda", dtype=torch.bfloat16):
                for i, out in enumerate(heads(feat)):
                    preds[i].append(out.float().argmax(1))
        y = test_labels.cpu().numpy()
        for i in range(len(lrs)):
            accs.append(float((torch.cat(preds[i]).cpu().numpy() == y).mean()))
    best = int(np.argmax(accs))
    print(f"[tvlp] {name}: best head lr={lrs[best]:.1e} acc={accs[best]:.4f}", flush=True)
    fh.write(f"{tag},{name},{epochs},{n_tr},{n_te},{lrs[best]:.6g},{accs[best]:.4f}\n")
    fh.flush()
    return accs[best]

def main():
    src, tag = sys.argv[1], sys.argv[2]
    epochs = int(sys.argv[3]) if len(sys.argv) > 3 else 10
    names = sys.argv[4].split(",") if len(sys.argv) > 4 else list(BENCH)
    out = os.path.join(ROOT, f"results/transfer/{tag}.visreg_lp.csv")
    os.makedirs(os.path.dirname(out), exist_ok=True)

    done = set()
    if os.path.exists(out):
        done = {ln.split(",")[1] for ln in open(out) if ln.count(",") >= 6}
    enc = build_encoder(src)
    print(f"[tvlp] {tag}: encoder {src} dim={enc.dim}; datasets={names}; done={sorted(done)}",
          flush=True)

    results = {}
    with open(out, "a") as fh:
        for name in names:
            if name in done:
                print(f"[tvlp] {name}: already in CSV, skipping", flush=True)
                continue
            results[name] = run_dataset(enc, name, epochs, fh, tag)

    order = [n for n in BENCH if n in set(names) | done]
    have = {ln.split(",")[1]: float(ln.strip().split(",")[6])
            for ln in open(out) if ln.count(",") >= 6}
    row = [have.get(n) for n in order]
    print("[tvlp] summary " + " ".join(f"{n}={v * 100:.1f}" if v is not None else f"{n}=--"
                                       for n, v in zip(order, row)), flush=True)
    vals = [v for v in row if v is not None]
    if vals:
        print(f"[tvlp] avg={100 * sum(vals) / len(vals):.1f} over {len(vals)} sets", flush=True)

if __name__ == "__main__":
    main()
