"""E25: ssl-transfer linear benchmark (Ericsson et al., CVPR 2021) on a project checkpoint.

Faithful port of the donor's linear protocol (github.com/linusericsson/ssl-transfer,
read 2026-08-05; donor code is reference-only): frozen features -> sklearn logistic
regression, lbfgs multinomial, C swept over logspace(-6, 5, 45) ascending with
warm_start, C selected on val, refit on train+val, scored on test. Metric = mean
per-class accuracy for aircraft/flowers/pets, top-1 for the rest. Preprocessing =
Resize(224, bicubic) + CenterCrop(224) + ImageNet normalization; features un-normalized.
Deviations from donor (recorded on the E25 card): h = our declared trunk CLS (D-003)
with the GAP tap as a non-primary rider (donor: R50 avgpool); datasets without an
official val split get a stratified 20% carve at seed 0 (donor ships custom split
files for some).

  python experiments/transfer_probe.py <dataset> [ckpt] [out_csv]
"""
import os
import sys

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from torch.amp import autocast
from torch.utils.data import DataLoader
from torchvision import datasets as tvd
from torchvision import transforms as T

from sslgap.ckpt import adapters
from sslgap.models.backbones import trunk_features

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.expanduser("~/data/ssltransfer")
CKPT = os.path.join(ROOT, "outputs/in1k.floorssl.s0.d256vm4_best.pt")
OUT = os.path.join(ROOT, "results/transfer/in1k.floorssl.s0.d256vm4.csv")

TF = T.Compose([T.Resize(224, interpolation=T.InterpolationMode.BICUBIC),
                T.CenterCrop(224), T.ToTensor(),
                T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

# name -> (splits dict, metric); splits: official torchvision split names, or
# "carve:<base>" = stratified 20% val carved from that extracted base split (seed 0)
BENCH = {
    "cifar10":  ({"train": dict(train=True), "test": dict(train=False), "val": "carve:train"}, "top1"),
    "cifar100": ({"train": dict(train=True), "test": dict(train=False), "val": "carve:train"}, "top1"),
    "dtd":      ({"train": dict(split="train"), "val": dict(split="val"), "test": dict(split="test")}, "top1"),
    "aircraft": ({"train": dict(split="train"), "val": dict(split="val"), "test": dict(split="test")}, "perclass"),
    "cars":     ({"train": dict(split="train"), "test": dict(split="test"), "val": "carve:train"}, "top1"),
    "flowers":  ({"train": dict(split="train"), "val": dict(split="val"), "test": dict(split="test")}, "perclass"),
    "food":     ({"train": dict(split="train"), "test": dict(split="test"), "val": "carve:train"}, "top1"),
    "pets":     ({"train": dict(split="trainval"), "test": dict(split="test"), "val": "carve:train"}, "perclass"),
}
TVSETS = {"cifar10": tvd.CIFAR10, "cifar100": tvd.CIFAR100, "dtd": tvd.DTD,
          "aircraft": tvd.FGVCAircraft, "flowers": tvd.Flowers102,
          "food": tvd.Food101, "pets": tvd.OxfordIIITPet}


class HFCars(torch.utils.data.Dataset):
    """StanfordCars via the tanganke/stanford_cars HF mirror — torchvision's upstream
    is dead (ValueError at download; E25 card deviation 3). Canonical split asserted;
    the Donghyun99 mirror was rejected by exactly this assert (8143/8040)."""

    def __init__(self, split):
        from datasets import load_dataset
        self.ds = load_dataset("tanganke/stanford_cars", split=split)
        assert (split, len(self.ds)) in (("train", 8144), ("test", 8041)), \
            f"non-canonical cars split {split}: {len(self.ds)}"

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, i):
        r = self.ds[i]
        return TF(r["image"].convert("RGB")), r["label"]


def build_split(name, spec):
    if name == "cars":
        return HFCars(spec["split"])
    return TVSETS[name](root=DATA, download=True, transform=TF, **spec)


@torch.inference_mode()
def extract(loaded, ds, device="cuda"):
    br = loaded.branches[loaded.probed_branch]
    feats, ys = {"cls": [], "gap": []}, []
    loader = DataLoader(ds, batch_size=256, num_workers=8, shuffle=False,
                        persistent_workers=True, pin_memory=True)
    for x, y in loader:
        with autocast(device, dtype=torch.bfloat16):
            f = trunk_features(br.trunk, x.to(device, non_blocking=True))
        for k in feats:
            feats[k].append(f[k].float().cpu())
        ys.append(torch.as_tensor(y))
    return ({k: torch.cat(v).numpy() for k, v in feats.items()},
            torch.cat(ys).numpy())


def carve(y, frac=0.2, seed=0):
    """Stratified val indices; returns (train_idx, val_idx)."""
    rng = np.random.default_rng(seed)
    tr, va = [], []
    for c in np.unique(y):
        idx = rng.permutation(np.flatnonzero(y == c))
        k = max(1, int(round(frac * len(idx))))
        va.append(idx[:k]); tr.append(idx[k:])
    return np.concatenate(tr), np.concatenate(va)


def score(clf, X, y, metric):
    p = clf.predict(X)
    if metric == "top1":
        return float((p == y).mean())
    return float(np.mean([(p[y == c] == c).mean() for c in np.unique(y)]))


def main():
    name = sys.argv[1]
    ckpt = sys.argv[2] if len(sys.argv) > 2 else CKPT
    out = sys.argv[3] if len(sys.argv) > 3 else OUT
    splits, metric = BENCH[name]
    os.makedirs(DATA, exist_ok=True)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    loaded = adapters.load("native", ckpt, run_id=os.path.basename(ckpt))
    loaded.eval_("cuda")

    X, Y = {}, {}
    for split, spec in splits.items():
        if isinstance(spec, str):
            continue
        ds = build_split(name, spec)
        X[split], Y[split] = extract(loaded, ds)
        print(f"[extract] {name}/{split}: n={len(Y[split])}", flush=True)
    for split, spec in splits.items():
        if isinstance(spec, str):
            base = spec.split(":")[1]
            tr_i, va_i = carve(Y[base])
            X[split] = {k: v[va_i] for k, v in X[base].items()}
            Y[split] = Y[base][va_i]
            X[base] = {k: v[tr_i] for k, v in X[base].items()}
            Y[base] = Y[base][tr_i]
            print(f"[carve] {name}: train={len(Y[base])} val={len(Y[split])}", flush=True)

    Cs = torch.logspace(-6, 5, 45).numpy()
    with open(out, "a") as fh:
        for space in ("cls", "gap"):
            Xtr, Xva, Xte = X["train"][space], X["val"][space], X["test"][space]
            clf = LogisticRegression(solver="lbfgs", warm_start=True)
            best = (-1.0, None)
            for C in Cs:
                clf.C = float(C)
                clf.fit(Xtr, Y["train"])
                v = score(clf, Xva, Y["val"], metric)
                if v > best[0]:
                    best = (v, float(C))
            Xtv = np.concatenate([Xtr, Xva])
            ytv = np.concatenate([Y["train"], Y["val"]])
            final = LogisticRegression(solver="lbfgs", C=best[1]).fit(Xtv, ytv)
            te = score(final, Xte, Y["test"], metric)
            row = (f"{name},{space},{metric},{len(ytv)},{len(Y['test'])},"
                   f"{best[1]:.6g},{best[0]:.4f},{te:.4f}")
            fh.write(row + "\n"); fh.flush()
            print(f"[result] {row}", flush=True)


if __name__ == "__main__":
    main()
