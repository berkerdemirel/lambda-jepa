"""Datasets, deterministic eval transform, aug stacks, and versioned feature manifests.

Transforms are verbatim ports of the house conventions (ssl_explore/sslx/data.py = lejepa minimal):
- eval = Resize(img)+CenterCrop(img) — matches the transform behind every prior CAMPAIGN_LOG
  number, so M0 parity checks are apples-to-apples.
- orbit = the shared lejepa aug stack; pinned as the FIXED AUDIT STACK v1 (PROTOCOL §5) for
  cross-method invariance/alignment comparisons.
- dino_global0/1 = Lightly/DINO-faithful global views (asymmetric blur/solarize) — the DINO
  control's OWN stack.

Manifests are CSV image lists (hf index or ImageFolder relpath + label), hashed; extraction and
probes consume only manifests, never raw splits (PROTOCOL §5)."""
import csv
import hashlib
import os

import numpy as np
import torch
from torchvision.datasets import ImageFolder
from torchvision.transforms import v2

_NORM = dict(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
_TAIL = [v2.ToImage(), v2.ToDtype(torch.float32, scale=True), v2.Normalize(**_NORM)]


def eval_transform(img_size):
    return v2.Compose([v2.Resize(img_size), v2.CenterCrop(img_size), *_TAIL])


def orbit_stack(img_size, rrc_scale=(0.08, 1.0)):
    """The house/lejepa orbit — FIXED AUDIT STACK v1."""
    return v2.Compose([
        v2.RandomResizedCrop(img_size, scale=rrc_scale),
        v2.RandomApply([v2.ColorJitter(0.8, 0.8, 0.8, 0.2)], p=0.8),
        v2.RandomGrayscale(p=0.2),
        v2.RandomApply([v2.GaussianBlur(kernel_size=7, sigma=(0.1, 2.0))]),
        v2.RandomApply([v2.RandomSolarize(threshold=128)], p=0.2),
        v2.RandomHorizontalFlip(), *_TAIL,
    ])


def _dino_view(size, scale, blur_p, solar_p, cj=0.5):
    """Port of sslx/dinov2.py _dino_view (Lightly DINOViewTransform)."""
    return v2.Compose([
        v2.RandomResizedCrop(size, scale=scale, interpolation=v2.InterpolationMode.BICUBIC,
                             antialias=True),
        v2.RandomHorizontalFlip(p=0.5),
        v2.RandomApply([v2.ColorJitter(0.8 * cj, 0.8 * cj, 0.4 * cj, 0.2 * cj)], p=0.8),
        v2.RandomGrayscale(p=0.2),
        v2.RandomApply([v2.GaussianBlur(kernel_size=9, sigma=(0.1, 2.0))], p=blur_p),
        v2.RandomApply([v2.RandomSolarize(threshold=128)], p=solar_p), *_TAIL,
    ])


STACKS = {
    "audit_v1": lambda s: (orbit_stack(s), orbit_stack(s)),
    "own_lejepa": lambda s: (orbit_stack(s), orbit_stack(s)),          # lejepa trains on the orbit
    "own_dino": lambda s: (_dino_view(s, (0.4, 1.0), 1.0, 0.0),        # global-0 / global-1 pair
                           _dino_view(s, (0.4, 1.0), 0.1, 0.2)),
}


# ---- manifests -------------------------------------------------------------------------------

def build_manifest_imagenette(split, out_csv):
    from datasets import load_dataset
    ds = load_dataset("frgfm/imagenette", "160px", split=split)
    rows = [(str(i), int(ds[i]["label"])) for i in range(len(ds))]
    return _write_manifest(out_csv, rows, source={"kind": "hf-imagenette", "split": split})


def build_manifest_imagefolder(root, out_csv, per_class=None, seed=0, linspace_n=None):
    """per_class: stratified random subset. linspace_n: evenly-spaced indices over the class-sorted
    ImageFolder order — the ssl_explore meters.make_eval_loaders selection rule, kept for parity
    checks against prior CAMPAIGN_LOG numbers."""
    ds = ImageFolder(root)
    if linspace_n is not None:
        idx = np.linspace(0, len(ds.samples) - 1, min(linspace_n, len(ds.samples))).astype(int)
        rows = [(os.path.relpath(ds.samples[i][0], root), ds.samples[i][1]) for i in idx]
        return _write_manifest(out_csv, rows, source={"kind": "imagefolder", "root": root,
                                                      "rule": f"linspace{linspace_n}"})
    by_cls = {}
    for path, y in ds.samples:
        by_cls.setdefault(y, []).append(os.path.relpath(path, root))
    rng = np.random.default_rng(seed)
    rows = []
    for y in sorted(by_cls):
        paths = sorted(by_cls[y])
        if per_class is not None and per_class < len(paths):
            paths = [paths[i] for i in rng.choice(len(paths), per_class, replace=False)]
        rows += [(p, y) for p in paths]
    return _write_manifest(out_csv, rows, source={"kind": "imagefolder", "root": root})


def _write_manifest(out_csv, rows, source):
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    with open(out_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["ref", "label"])
        w.writerows(rows)
    return {"path": out_csv, "n": len(rows), "sha256": file_sha256(out_csv), "source": source}


def file_sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def read_manifest(csv_path):
    with open(csv_path) as f:
        rows = list(csv.DictReader(f))
    return [(r["ref"], int(r["label"])) for r in rows]


# ---- datasets over manifests -----------------------------------------------------------------

class _Source:
    """Resolves manifest refs to PIL images. kind: hf-imagenette (ref = row index) or
    imagefolder (ref = relpath under root)."""

    def __init__(self, kind, root=None, split=None):
        self.kind = kind
        if kind == "hf-imagenette":
            from datasets import load_dataset
            self.ds = load_dataset("frgfm/imagenette", "160px", split=split)
        else:
            self.root = os.path.expanduser(root)

    def __call__(self, ref):
        if self.kind == "hf-imagenette":
            return self.ds[int(ref)]["image"].convert("RGB")
        from PIL import Image
        return Image.open(os.path.join(self.root, ref)).convert("RGB")


class EvalDataset(torch.utils.data.Dataset):
    def __init__(self, manifest_csv, source: _Source, img_size):
        self.items = read_manifest(manifest_csv)
        self.source, self.tfm = source, eval_transform(img_size)

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        ref, y = self.items[i]
        return self.tfm(self.source(ref)), y


class PairDataset(torch.utils.data.Dataset):
    """Two augmented views per image under a named stack (PROTOCOL §5: pairs are extracted under
    both the method's own stack and audit_v1). View draws are stochastic; alignment metrics
    average over the manifest and carry bootstrap CIs (estimator note in PROTOCOL §6)."""

    def __init__(self, manifest_csv, source: _Source, img_size, stack="audit_v1"):
        self.items = read_manifest(manifest_csv)
        self.source = source
        self.ta, self.tb = STACKS[stack](img_size)

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        ref, y = self.items[i]
        img = self.source(ref)
        return self.ta(img), self.tb(img), y


def make_source(frame):
    ds = frame.get("dataset", "imagenette")
    if ds == "imagenette":
        return lambda split: _Source("hf-imagenette", split=split)
    root = os.path.expanduser(frame["data_root"])
    return lambda split: _Source("imagefolder", root=os.path.join(root, split))
