"""E14 pre-registration qualitative check: render foveal event/context pairs per stratum
(copy/near/far) for both declared fovea sizes, BEFORE any extraction launches — Berker signs off
on the channel visually (2026-07-12 request). Uses the exact FovealPairDataset code path
(pre-normalization). Output: results/figures/e14/preview_<stack>.png"""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sslgap.data import (FOVEAL_SIZES, _Source, foveal_boxes, foveal_ctx, foveal_event,
                         foveal_iou, read_manifest)
from torchvision.transforms import v2

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MANIFEST = os.path.join(ROOT, "features/manifests/in100.pairs100.v1.csv")
OUT = os.path.join(ROOT, "results/figures/e14")
PER_STRATUM = 3

import numpy as np

items = read_manifest(MANIFEST)
order = np.random.default_rng(14).permutation(len(items))  # manifest is class-ordered; mix classes
picked, want = [], {"far": PER_STRATUM, "near": PER_STRATUM, "copy": PER_STRATUM}
for ref, _ in (items[i] for i in order):  # stratum keyed by ref only -> same rows for both stacks
    s, _, _ = foveal_boxes(ref, "foveal_v1")
    if want.get(s, 0) > 0:
        want[s] -= 1
        picked.append((s, ref))
    if not any(want.values()):
        break
picked.sort(key=lambda t: ["copy", "near", "far"].index(t[0]))

source = _Source("imagefolder", root=os.path.expanduser("~/data/imagenet100/train"))
base_tfm = v2.Compose([v2.Resize(224), v2.CenterCrop(224)])
os.makedirs(OUT, exist_ok=True)

for stack, f in FOVEAL_SIZES.items():
    fig, axes = plt.subplots(len(picked), 4, figsize=(11, 2.6 * len(picked)))
    for r, (stratum, ref) in enumerate(picked):
        base = base_tfm(source(ref))
        s, a, b = foveal_boxes(ref, stack)
        assert s == stratum
        views = [foveal_event(base, a, f), foveal_event(base, b, f),
                 foveal_ctx(base, a, f), foveal_ctx(base, b, f)]
        for c, (im, title) in enumerate(zip(views, ["event_A", "event_B", "ctx_A", "ctx_B"])):
            ax = axes[r, c]
            ax.imshow(im)
            ax.set_xticks([]), ax.set_yticks([])
            if r == 0:
                ax.set_title(title, fontsize=10)
        axes[r, 0].set_ylabel(f"{stratum}\nIoU={foveal_iou(a, b, f):.2f}", fontsize=9)
    fig.suptitle(f"{stack}: fovea {f}px sharp, surround ×4 down/up (224 frame)", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    path = os.path.join(OUT, f"preview_{stack}.png")
    fig.savefig(path, dpi=110)
    print(path)
