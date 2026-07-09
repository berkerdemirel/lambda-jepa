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
import random

import numpy as np
import torch
from torchvision.datasets import ImageFolder
from torchvision.transforms import v2

_NORM = dict(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
_TAIL = [v2.ToImage(), v2.ToDtype(torch.float32, scale=True), v2.Normalize(**_NORM)]


def seed_everything(seed):
    """All RNG streams a job touches: python random (I-JEPA MaskSampler), numpy, torch (+cuda)."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def seed_worker(worker_id):
    """DataLoader worker_init_fn: derive python/numpy streams from the torch worker seed, so
    transform randomness in workers is reproducible given the loader's generator."""
    s = torch.initial_seed() % 2 ** 32
    random.seed(s)
    np.random.seed(s)


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
    "own_simclr": lambda s: (simclr_stack(s), simclr_stack(s)),
    "own_byol": lambda s: tuple(byol_pair(s)),                         # asymmetric blur/solarize
    "own_vicreg": lambda s: tuple(byol_pair(s)),                       # VICReg follows BYOL augs
    "own_mae": lambda s: (minaug_stack(s, (0.2, 1.0)),                 # masking is the method's z
                          minaug_stack(s, (0.2, 1.0))),                # machinery, not its stack
    "own_ijepa": lambda s: (minaug_stack(s, (0.3, 1.0)),
                            minaug_stack(s, (0.3, 1.0))),
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


class _FullSplit:
    """Raw PIL access to a full split (no manifest): imagenette via HF, else ImageFolder."""

    def __init__(self, dataset, split, data_root=None):
        if dataset == "imagenette":
            self.source = _Source("hf-imagenette", split=split)
            self.n = len(self.source.ds)
        else:
            sub = {"train": "train", "validation": "val"}[split]
            root = os.path.join(os.path.expanduser(data_root), sub)
            from torchvision.datasets import ImageFolder
            self._folder = ImageFolder(root)
            self.source = None
            self.n = len(self._folder)

    def __call__(self, i):
        if self.source is not None:
            row = self.source.ds[int(i)]
            return row["image"].convert("RGB"), int(row["label"])
        path, y = self._folder.samples[i]
        return self._folder.loader(path).convert("RGB"), y


class ViewsDataset(torch.utils.data.Dataset):
    """Training views over a FULL split: V>1 -> V draws of `aug` (default = the orbit stack, the
    lejepa recipe); V==1 -> the deterministic eval transform. Returns (views [V,C,H,W], label).
    `transforms`: optional per-view transform list (len V) for asymmetric-view methods (BYOL)."""

    def __init__(self, dataset, split, V, img_size, data_root=None, aug=None, transforms=None):
        self.V = V
        self.split_src = _FullSplit(dataset, split, data_root)
        if transforms is not None:
            assert len(transforms) == V
            self.tfms = transforms
        elif V > 1:
            self.tfms = [aug or orbit_stack(img_size)] * V
        else:
            self.tfms = [eval_transform(img_size)]

    def __getitem__(self, i):
        img, y = self.split_src(i)
        return torch.stack([t(img) for t in self.tfms]), y

    def __len__(self):
        return self.split_src.n


class MultiCropDataset(torch.utils.data.Dataset):
    """DINO multi-crop over a full split: 2 globals + n_local locals (Lightly/DINO-faithful views;
    port of sslx/dinov2.MultiCropDataset aug='dino'). Returns ((g [2,C,G,G], l [nl,C,L,L]), y)."""

    def __init__(self, dataset, split, img_size, local_size, n_local, data_root=None):
        self.split_src = _FullSplit(dataset, split, data_root)
        self.globs = [_dino_view(img_size, (0.4, 1.0), blur_p=1.0, solar_p=0.0),
                      _dino_view(img_size, (0.4, 1.0), blur_p=0.1, solar_p=0.2)]
        self.loc = _dino_view(local_size, (0.05, 0.4), blur_p=0.5, solar_p=0.0)
        self.n_local = n_local

    def __getitem__(self, i):
        img, y = self.split_src(i)
        g = torch.stack([t(img) for t in self.globs])
        l = torch.stack([self.loc(img) for _ in range(self.n_local)])
        return (g, l), y

    def __len__(self):
        return self.split_src.n


def supervised_stack(img_size):
    """Supervised-anchor aug (deitlite, D-022): plain RRC + flip — deliberately NO jitter/mixup/
    randaug so the anchor stays a minimal supervised reference, not a tuned competitor."""
    return v2.Compose([v2.RandomResizedCrop(img_size, scale=(0.08, 1.0)),
                       v2.RandomHorizontalFlip(), *_TAIL])


def simclr_stack(img_size):
    """SimCLR aug (paper Fig. 4 defaults): RRC + flip + jitter(0.8,.8,.8,.2)@0.8 + gray 0.2 + blur 0.5."""
    return v2.Compose([
        v2.RandomResizedCrop(img_size, scale=(0.08, 1.0)),
        v2.RandomHorizontalFlip(),
        v2.RandomApply([v2.ColorJitter(0.8, 0.8, 0.8, 0.2)], p=0.8),
        v2.RandomGrayscale(p=0.2),
        v2.RandomApply([v2.GaussianBlur(kernel_size=7, sigma=(0.1, 2.0))], p=0.5), *_TAIL,
    ])


def byol_pair(img_size):
    """BYOL/VICReg asymmetric view pair: view1 blur p=1.0/no solarize; view2 blur p=0.1/solarize 0.2."""
    def view(blur_p, solar_p):
        return v2.Compose([
            v2.RandomResizedCrop(img_size, scale=(0.08, 1.0)),
            v2.RandomHorizontalFlip(),
            v2.RandomApply([v2.ColorJitter(0.4, 0.4, 0.2, 0.1)], p=0.8),
            v2.RandomGrayscale(p=0.2),
            v2.RandomApply([v2.GaussianBlur(kernel_size=7, sigma=(0.1, 2.0))], p=blur_p),
            v2.RandomApply([v2.RandomSolarize(threshold=128)], p=solar_p), *_TAIL,
        ])
    return [view(1.0, 0.0), view(0.1, 0.2)]


def minaug_stack(img_size, scale=(0.3, 1.0)):
    """I-JEPA/MAE-style minimal aug: RRC + flip only (nothing photometric)."""
    return v2.Compose([v2.RandomResizedCrop(img_size, scale=scale),
                       v2.RandomHorizontalFlip(), *_TAIL])
