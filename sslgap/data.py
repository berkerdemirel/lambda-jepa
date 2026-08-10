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


def _bench_view(size, scale, blur_p, solar_p, cj=0.5):
    """D-092 replication view: _dino_view semantics with the solarize probability TRUE.
    The house wrap `RandomApply([RandomSolarize], p)` halves the effective probability —
    v2.RandomSolarize carries its own p=0.5 (measured 2026-08-10: wrapped .0988 vs bare
    .1983 at declared .2) — so external-faithful stacks use the bare transform. The frozen
    v1 stacks (orbit_stack/audit_v1, _dino_view) are NOT changed: their code is their
    definition; the declaration erratum is on the E27 card §(j.9)/D-092."""
    ops = [
        v2.RandomResizedCrop(size, scale=scale, interpolation=v2.InterpolationMode.BICUBIC,
                             antialias=True),
        v2.RandomHorizontalFlip(p=0.5),
        v2.RandomApply([v2.ColorJitter(0.8 * cj, 0.8 * cj, 0.4 * cj, 0.2 * cj)], p=0.8),
        v2.RandomGrayscale(p=0.2),
        v2.RandomApply([v2.GaussianBlur(kernel_size=9, sigma=(0.1, 2.0))], p=blur_p),
    ]
    if solar_p:
        ops.append(v2.RandomSolarize(threshold=128, p=solar_p))
    return v2.Compose(ops + _TAIL)


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


# ---- E14 foveal channel (PIVOT rung-1) --------------------------------------------------------
# Declared event channel per PIVOT §6.1: deterministic base scene (house eval geometry,
# Resize+CenterCrop 224), sharp fovea pasted into a ×4 down/up-sampled surround (all bilinear).
# Context = the sharp fovea-sized window at the partner's fovea, resized to 224 for the tokenizer.
# Everything is keyed by manifest ref: every extraction pass yields byte-identical views, so the
# target array is shared zoo-wide and no member has a same-pass draw advantage (E13 card §Constr).

FOVEAL_SIZES = {"foveal_v1": 96}   # fovea side px, 16-token-aligned @224 (f96-only per Berker)
_FOVEAL_DOWN = 4
_FOVEAL_GRID = 16

def _foveal_rng(ref, tag):
    h = hashlib.md5(f"{tag}:{ref}".encode()).digest()
    return np.random.default_rng(int.from_bytes(h[:8], "little"))


def foveal_iou(a, b, f):
    ox = max(0, f - abs(a[0] - b[0]))
    oy = max(0, f - abs(a[1] - b[1]))
    return ox * oy / (2 * f * f - ox * oy)


def foveal_boxes(ref, stack, img_size=224):
    """Deterministic (stratum, boxA, boxB) for one manifest ref; boxes are (x, y) top-lefts on the
    16-px grid. Stratum is keyed by ref ONLY (shared across fovea sizes → the f96/f64 contrast
    compares identical image splits); positions are keyed by (ref, stack). far = IoU 0 (A resampled
    until a disjoint partner exists — centered 96² foveas admit none on 224); near = IoU in (0,.5];
    copy = identical box (re-encoding contrast cell)."""
    f = FOVEAL_SIZES[stack]
    grid = list(range(0, img_size - f + 1, _FOVEAL_GRID))
    u = _foveal_rng(ref, "strata_v1").random()
    stratum = "far" if u < 0.5 else ("near" if u < 0.8 else "copy")
    rng = _foveal_rng(ref, stack)
    while True:
        a = (int(rng.choice(grid)), int(rng.choice(grid)))
        if stratum == "copy":
            return stratum, a, a
        lo, hi = (0.0, 0.0) if stratum == "far" else (1e-9, 0.5)
        cands = [(x, y) for x in grid for y in grid if lo <= foveal_iou(a, (x, y), f) <= hi]
        if cands:
            return stratum, a, cands[int(rng.integers(len(cands)))]


def foveal_event(base, box, f):
    """PIL 224 base scene -> foveal composite: ×4 bilinear down/up surround, sharp fovea.
    box=None -> no fovea (the E14-addendum no-fovea null: fully degraded event, `blur_v1`)."""
    from PIL import Image
    s = base.size[0]
    low = base.resize((s // _FOVEAL_DOWN,) * 2, Image.BILINEAR).resize((s,) * 2, Image.BILINEAR)
    if box is None:
        return low
    ev = low.copy()
    ev.paste(base.crop((box[0], box[1], box[0] + f, box[1] + f)), (box[0], box[1]))
    return ev


def foveal_ctx(base, box, f):
    """Sharp fovea-sized window, resized to the frame for the tokenizer (bilinear, declared)."""
    from PIL import Image
    return base.crop((box[0], box[1], box[0] + f, box[1] + f)).resize(base.size, Image.BILINEAR)


class PivotTrainDataset(torch.utils.data.Dataset):
    """PIVOT training items over the FULL train split. One fovea position per image per step
    (uniform over the 16-px grid), two light-photometric event views of the SAME event (§5.4
    equivalent-event condition), CLEAN sharp context crop(s). Shared hflip on the base BEFORE
    position sampling keeps views+ctx in one frame. Worker RNG per house seeding.
    k_queries=0 (E15/D-032, position-marginal, VOID per D-033): one c_B -> ((v1, v2, ctx), y).
    k_queries=K (E16/D-033 dense): K query positions, each supervising the event token block at
    q -> ((v1, v2, ctxs [K,C,H,W], qidx [K,2](ty,tx slot units)[, ctx_g lowres]), y)."""

    def __init__(self, dataset, split, img_size, data_root=None, f=96, k_queries=0,
                 global_ctx=False):
        assert img_size == 224
        self.split_src = _FullSplit(dataset, split, data_root)
        self.f, self.K, self.global_ctx = f, k_queries, global_ctx
        self.base = v2.Compose([v2.Resize(img_size), v2.CenterCrop(img_size)])
        self.photo = v2.Compose([
            v2.RandomApply([v2.ColorJitter(0.4, 0.4, 0.2, 0.1)], p=0.8),
            v2.RandomGrayscale(p=0.2)])
        self.tail = v2.Compose(_TAIL)
        self.slots = list(range(0, img_size - f + 1, _FOVEAL_GRID))

    def __len__(self):
        return self.split_src.n

    def __getitem__(self, i):
        img, y = self.split_src(i)
        base = self.base(img)
        if random.random() < 0.5:
            from PIL import Image
            base = base.transpose(Image.FLIP_LEFT_RIGHT)
        c = (random.choice(self.slots), random.choice(self.slots))
        v1 = self.tail(foveal_event(self.photo(base), c, self.f))
        v2_ = self.tail(foveal_event(self.photo(base), c, self.f))
        if not self.K:
            cb = (random.choice(self.slots), random.choice(self.slots))
            return (v1, v2_, self.tail(foveal_ctx(base, cb, self.f))), y
        qs = [(random.choice(self.slots), random.choice(self.slots)) for _ in range(self.K)]
        ctxs = torch.stack([self.tail(foveal_ctx(base, q, self.f)) for q in qs])
        qidx = torch.tensor([(q[1] // _FOVEAL_GRID, q[0] // _FOVEAL_GRID) for q in qs])
        out = (v1, v2_, ctxs, qidx)
        if self.global_ctx:
            out = out + (self.tail(foveal_event(base, None, None)),)
        return out, y


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


class OrbitDataset(torch.utils.data.Dataset):
    """V stochastic views per image under a named stack — the per-image augmentation-orbit sample
    for overlap/invariance estimators along the depth axis (E02; docs/theory/
    HEAD_OVERLAP_LIPSCHITZ.md). Asymmetric stacks (own_dino) alternate branches — view k draws
    ta if k is even else tb — so the store samples the method's actual positive-pair mixture;
    for symmetric stacks this is V iid draws."""

    def __init__(self, manifest_csv, source: _Source, img_size, stack, v):
        self.items = read_manifest(manifest_csv)
        self.source = source
        self.ta, self.tb = STACKS[stack](img_size)
        self.v = v

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        ref, y = self.items[i]
        img = self.source(ref)
        return tuple((self.ta if k % 2 == 0 else self.tb)(img) for k in range(self.v)), y


class FovealPairDataset(torch.utils.data.Dataset):
    """E14 pairs: mode='event' -> (event_A, event_B) for zoo members; mode='ctx' -> (ctx_A, ctx_B)
    for tokenizer runs. Slot convention matches PairDataset (viewA, viewB): D_read A→B pairs the
    member's slot-0 with the tokenizer's slot-1, as in E13. Scoring recomputes strata/boxes via
    foveal_boxes (single source of truth — no side-channel CSV)."""

    def __init__(self, manifest_csv, source: _Source, img_size, stack, mode):
        assert img_size == 224, "foveal_v1 geometry is declared on the 224 frame only"
        assert mode in ("event", "ctx")
        self.items = read_manifest(manifest_csv)
        self.source, self.stack, self.mode = source, stack, mode
        self.base = v2.Compose([v2.Resize(img_size), v2.CenterCrop(img_size)])
        self.tail = v2.Compose(_TAIL)
        self.f = FOVEAL_SIZES.get(stack)     # None for blur_v1 (no-fovea null, E14 addendum)

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        ref, y = self.items[i]
        base = self.base(self.source(ref))
        if self.stack == "blur_v1":          # location-free event; A == B by construction
            v = self.tail(foveal_event(base, None, None))
            return v, v.clone(), y
        _, a, b = foveal_boxes(ref, self.stack)
        make = foveal_event if self.mode == "event" else foveal_ctx
        return self.tail(make(base, a, self.f)), self.tail(make(base, b, self.f)), y


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


class LejepaMultiCropDataset(torch.utils.data.Dataset):
    """E27 Recipe v2 multicrop (D-079a/D-082): n_g globals @ img_size + n_l locals @ local_size,
    EVERY view the house symmetric lejepa photometric family (orbit_stack) — only the RRC
    geometry differs between groups. Defaults = the LeJEPA-repo-PUBLISHED geometry (README:
    globals 224 scale (0.3, 1.0), locals scale (0.05, 0.3) at 98 for /14 arches → 96 here, the
    /16 patch-divisible adaptation, declared). Their full trainer stays unpublished; view
    counts follow the paper recommendation (V_g=2, V_l=8 — outranks the README's 6-local
    example per the D-081 gap-fill order). DINO's asymmetric global1/global2 blur/solarize
    split is NOT copied. Returns ((g [n_g,C,G,G], l [n_l,C,L,L]), y)."""

    def __init__(self, dataset, split, img_size, data_root=None, n_g=2, n_l=8, local_size=96,
                 global_scale=(0.3, 1.0), local_scale=(0.05, 0.3)):
        self.split_src = _FullSplit(dataset, split, data_root)
        self.tg = orbit_stack(img_size, tuple(global_scale))
        self.tl = orbit_stack(local_size, tuple(local_scale))
        self.n_g, self.n_l = n_g, n_l

    def __getitem__(self, i):
        img, y = self.split_src(i)
        return (torch.stack([self.tg(img) for _ in range(self.n_g)]),
                torch.stack([self.tl(img) for _ in range(self.n_l)])), y

    def __len__(self):
        return self.split_src.n


class LightlyLejepaMultiCropDataset(torch.utils.data.Dataset):
    """E27 §(j)/D-092 Lightly-replication views: the exact transform behind Lightly's LeJEPA
    ViT-S/16 64.0 row — their benchmark constructs DINOTransform(global 224 (0.3,1.0), local 96
    (0.05,0.3), gaussian_blur=(0.5,0.5,0.5), n_local_views=6): per view bicubic RRC + flip .5 +
    jitter(0.4,.4,.2,.1)@p.8 + gray .2 + blur@p.5 (uniform — their override of DINO's 1.0/0.1/0.5
    asymmetry), solarize(128)@p.2 on the SECOND GLOBAL ONLY (the sole surviving per-view
    asymmetry). House k9 blur approximates their radius-sampled PIL blur (the standing DINO-port
    declaration, _dino_view). Returns ((g [2,C,G,G], l [n_l,C,L,L]), y)."""

    def __init__(self, dataset, split, img_size, data_root=None, n_g=2, n_l=6,
                 local_size=96, global_scale=(0.3, 1.0), local_scale=(0.05, 0.3)):
        self.split_src = _FullSplit(dataset, split, data_root)
        self.g1 = _bench_view(img_size, tuple(global_scale), blur_p=0.5, solar_p=0.0)
        self.g2 = _bench_view(img_size, tuple(global_scale), blur_p=0.5, solar_p=0.2)
        self.loc = _bench_view(local_size, tuple(local_scale), blur_p=0.5, solar_p=0.0)
        self.n_g, self.n_l = n_g, n_l

    def __getitem__(self, i):
        # n_g > 2 (D-097, the VISReg-geometry B mirrors): extra globals ride the g1
        # transform — Lightly's per-view asymmetry (solarize on the SECOND global only)
        # is preserved as exactly one solarize-eligible view at any n_g.
        img, y = self.split_src(i)
        gs = [self.g1(img), self.g2(img)] + [self.g1(img) for _ in range(self.n_g - 2)]
        return (torch.stack(gs),
                torch.stack([self.loc(img) for _ in range(self.n_l)])), y

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
