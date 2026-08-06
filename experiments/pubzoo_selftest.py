"""D-076 pretrained-zoo wave 1: fetch every roster model + adapter selftest BEFORE any
extraction ("do the adapter make sure there's no mistakes").

Selftest kinds (recorded per model on the E26 card):
  zeroshot — open_clip end-to-end IN-1k val zero-shot top-1 vs the published number
             (exact preprocessing + weights check; tolerance ±0.8).
  hub      — timm CLS vs the official facebookresearch torch.hub implementation on 64
             identical val inputs: per-image cosine >= .999 (catches adapter/preproc
             divergence against the reference implementation).
  cfgonly  — fetch + resolved data-config + feature-stat smoke only; the published-number
             anchor lands at battery stage (weakest tier, marked).
dinov3 runs LAST: its HF weights are license-gated — a fetch failure there must not
cost the rest of the roster (no try/except by policy; order carries the risk).
"""
import csv
import os

import numpy as np
import torch
import timm
from torch.utils.data import DataLoader, Subset
from torchvision import datasets as tvd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
VAL = os.path.expanduser("~/data/imagenet/val")
OUT = os.path.join(ROOT, "results/diag/pubzoo_selftest.csv")

ROSTER = [
    ("clipb16",   "vit_base_patch16_clip_quickgelu_224.openai", "zeroshot", ("ViT-B-16-quickgelu", "openai"), 68.3),
    ("siglipb16", "vit_base_patch16_siglip_224.webli",   "zeroshot", ("ViT-B-16-SigLIP", "webli"), 76.0),
    ("dinov2b14", "vit_base_patch14_dinov2.lvd142m",     "hub", ("facebookresearch/dinov2", "dinov2_vitb14"), None),
    ("dinob16",   "vit_base_patch16_224.dino",           "hub", ("facebookresearch/dino:main", "dino_vitb16"), None),
    ("maeb16",    "vit_base_patch16_224.mae",            "cfgonly", None, None),
    ("dinov3b16", "vit_base_patch16_dinov3.lvd1689m",    "cfgonly", None, None),
]


def fixed_batch(transform, n=64):
    ds = tvd.ImageFolder(VAL, transform=transform)
    idx = np.linspace(0, len(ds) - 1, n).astype(int)      # deterministic class spread
    x = torch.stack([ds[i][0] for i in idx])
    return x


@torch.inference_mode()
def zeroshot(oc_name, oc_pretrained):
    import open_clip
    from open_clip import (IMAGENET_CLASSNAMES, OPENAI_IMAGENET_TEMPLATES,
                           build_zero_shot_classifier, get_tokenizer)
    model, _, preprocess = open_clip.create_model_and_transforms(
        oc_name, pretrained=oc_pretrained, device="cuda")
    model.eval()
    clf = build_zero_shot_classifier(model, get_tokenizer(oc_name),
                                     IMAGENET_CLASSNAMES, OPENAI_IMAGENET_TEMPLATES,
                                     device="cuda", use_tqdm=False)
    ds = tvd.ImageFolder(VAL, transform=preprocess)
    loader = DataLoader(ds, batch_size=256, num_workers=12, pin_memory=True)
    hit = n = 0
    for x, y in loader:
        f = model.encode_image(x.to("cuda", non_blocking=True))
        f = f / f.norm(dim=-1, keepdim=True)
        hit += (f @ clf).argmax(1).cpu().eq(y).sum().item()
        n += len(y)
    return 100.0 * hit / n


@torch.inference_mode()
def timm_cls(model, x):
    f = model.forward_features(x.to("cuda"))
    return f[:, 0] if f.ndim == 3 else f


@torch.inference_mode()
def run():
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    rows = []
    for key, tname, kind, extra, ref in ROSTER:
        print(f"=== {key} ({tname}) [{kind}] ===", flush=True)
        model = timm.create_model(tname, pretrained=True, num_classes=0).eval().cuda()
        cfg = timm.data.resolve_data_config({}, model=model)
        print(f"[cfg] {key}: {cfg}", flush=True)
        tf = timm.data.create_transform(**cfg)
        x = fixed_batch(tf)
        f = timm_cls(model, x)
        print(f"[smoke] {key}: cls {tuple(f.shape)} | mean-norm {f.norm(dim=1).mean():.3f}",
              flush=True)
        if kind == "zeroshot":
            acc = zeroshot(*extra)
            ok = abs(acc - ref) <= 0.8
            print(f"[zeroshot] {key}: {acc:.2f} vs published {ref} -> "
                  f"{'PASS' if ok else 'FAIL'}", flush=True)
            rows.append([key, tname, kind, f"{acc:.2f}", ref, "PASS" if ok else "FAIL"])
        elif kind == "hub":
            hub = torch.hub.load(*extra).eval().cuda()
            g = hub(x.to("cuda"))
            g = g if g.ndim == 2 else g[:, 0]
            cos = torch.nn.functional.cosine_similarity(f, g, dim=1)
            ok = bool(cos.min() >= 0.999)
            print(f"[hub] {key}: cos min {cos.min():.5f} mean {cos.mean():.5f} -> "
                  f"{'PASS' if ok else 'FAIL'}", flush=True)
            rows.append([key, tname, kind, f"{cos.min():.5f}", 0.999, "PASS" if ok else "FAIL"])
        else:
            rows.append([key, tname, kind, f"{f.norm(dim=1).mean():.3f}", "", "SMOKE"])
        del model
        torch.cuda.empty_cache()
    with open(OUT, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["model", "timm_name", "selftest", "measured", "reference", "verdict"])
        w.writerows(rows)
    print(f"[done] {OUT}", flush=True)


if __name__ == "__main__":
    run()
