"""E26 zoo wave 2: public-checkpoint extraction to the feature store (D-076 rules).

Per model (one job): eval features on the STANDARD in1k manifests (train.v1 full 1.28M,
val.v1 50k — same image lists as the retrain cells) + audit_v1 orbit views (pairs10
subset, V=8) — trunk stations ONLY (L03/L06/L09 via timm forward_intermediates
norm=True + final), no head taps. Spaces: pub.h.{cls,gap}[.Lkk] (+ pub.h.pool at the
final station where the model's native h is attention-pooled: siglip). Preprocessing:
the model's own timm-resolved eval transform (native size/interp/crop_pct/norm — the
selftest-verified path). Orbit views: the FIXED audit_v1 stack byte-identical at the
model's native size, with ONLY the normalization tail affinely re-based to the model's
constants (exact: Normalize(mean=(m_m−m_in)/s_in, std=s_m/s_in) appended).

  python experiments/pubzoo_extract.py <model_key>
"""
import os
import sys

import numpy as np
import timm
import torch
from torch.amp import autocast
from torch.utils.data import DataLoader
from torchvision.transforms import v2

from sslgap.data import _NORM, OrbitDataset, _Source, read_manifest, seed_everything
from sslgap.extract import FeatureStore

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MANI = os.path.join(ROOT, "features/manifests")
IN1K = os.path.expanduser("~/data/imagenet")

MODELS = {
    "clipb16": "vit_base_patch16_clip_quickgelu_224.openai",
    "siglipb16": "vit_base_patch16_siglip_224.webli",
    "dinov2b14": "vit_base_patch14_dinov2.lvd142m",
    "dinob16": "vit_base_patch16_224.dino",
    "maeb16": "vit_base_patch16_224.mae",
    "dinov3b16": "vit_base_patch16_dinov3.lvd1689m",
}
IDX = (2, 5, 8)                                    # blocks 3/6/9 -> L03/L06/L09
ORBIT_V, ORBIT_SEED = 8, 0


class ZooEval(torch.utils.data.Dataset):
    def __init__(self, manifest_csv, source, tfm):
        self.items = read_manifest(manifest_csv)
        self.source, self.tfm = source, tfm

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        ref, y = self.items[i]
        return self.tfm(self.source(ref)), y


def renorm_tail(mean, std):
    """Exact affine re-base appended AFTER the audit_v1 IN-normalized tail."""
    m_in, s_in = np.array(_NORM["mean"]), np.array(_NORM["std"])
    return v2.Normalize(mean=((np.array(mean) - m_in) / s_in).tolist(),
                        std=(np.array(std) / s_in).tolist())


def batch_spaces(model, x, npre, want_pool):
    """One forward -> {space: [B, d]} at the four trunk stations."""
    out = {}
    final, inter = model.forward_intermediates(x, indices=IDX, norm=True,
                                               output_fmt="NLC", return_prefix_tokens=True)
    for l, it in zip((3, 6, 9), inter):
        tok, pre = it if isinstance(it, tuple) else (it, None)
        out[f"pub.h.gap.L{l:02d}"] = tok.mean(1)
        if pre is not None:
            out[f"pub.h.cls.L{l:02d}"] = pre[:, 0]
    out["pub.h.gap"] = final[:, npre:].mean(1)
    if npre > 0:
        out["pub.h.cls"] = final[:, 0]
    if want_pool:
        out["pub.h.pool"] = model.forward_head(final, pre_logits=True)
    return {k: v.float().cpu() for k, v in out.items()}


@torch.inference_mode()
def run_eval(model, ds, store, run_id, mkey, info, npre, want_pool, bs):
    acc, ys = {}, []
    for x, y in DataLoader(ds, batch_size=bs, num_workers=12, shuffle=False,
                           persistent_workers=True, pin_memory=True):
        with autocast("cuda", dtype=torch.bfloat16):
            b = batch_spaces(model, x.to("cuda", non_blocking=True), npre, want_pool)
        for k, v in b.items():
            acc.setdefault(k, []).append(v)
        ys.append(torch.as_tensor(y))
    for space, chunks in acc.items():
        store.put(run_id, mkey, space, torch.cat(chunks).numpy())
    store.put_labels(run_id, mkey, torch.cat(ys).numpy())
    store.put_meta(run_id, mkey, {"kind": "eval", "manifest": info,
                                  "n": int(sum(len(y) for y in ys)),
                                  "spaces": sorted(acc), "h_layers": [3, 6, 9], "bs": bs,
                                  "method": "public", "frame": "in1k", "probed_branch": "pub"})
    print(f"[eval] {run_id} {mkey}: {len(acc)} spaces", flush=True)


@torch.inference_mode()
def run_orbits(model, ds, store, run_id, mkey, info, npre, want_pool, bs):
    acc, ys = {}, []
    g = torch.Generator().manual_seed(ORBIT_SEED)
    from sslgap.data import seed_worker
    # 4 workers: each prefetched batch holds V=8 full-res view tensors — 12 workers OOM'd
    # the 96G cgroup at 256px (63047450) and would at 518px; eval loaders are unaffected
    for views, y in DataLoader(ds, batch_size=bs, num_workers=4, shuffle=False,
                               persistent_workers=True, pin_memory=True, generator=g,
                               worker_init_fn=seed_worker):
        for k, xv in enumerate(views):
            with autocast("cuda", dtype=torch.bfloat16):
                b = batch_spaces(model, xv.to("cuda", non_blocking=True), npre, want_pool)
            for sp, v in b.items():
                acc.setdefault(f"{sp}.view{k}", []).append(v)
        ys.append(torch.as_tensor(y))
    for space, chunks in acc.items():
        store.put(run_id, mkey, space, torch.cat(chunks).numpy())
    store.put_labels(run_id, mkey, torch.cat(ys).numpy())
    store.put_meta(run_id, mkey, {"kind": "orbits", "manifest": info, "v": ORBIT_V,
                                  "n": int(sum(len(y) for y in ys)),
                                  "spaces": sorted(acc), "stack": "audit_v1(native-renorm)",
                                  "method": "public", "frame": "in1k", "probed_branch": "pub"})
    print(f"[orbits] {run_id} {mkey}: {len(acc)} spaces", flush=True)


def main():
    key = sys.argv[1]
    seed_everything(0)
    tname = MODELS[key]
    model = timm.create_model(tname, pretrained=True, num_classes=0).eval().cuda()
    cfg = timm.data.resolve_data_config({}, model=model)
    size = cfg["input_size"][1]
    npre = model.num_prefix_tokens
    want_pool = npre == 0                          # siglip: native h is the attn-pooled token
    bs = 128 if size > 300 else 256
    print(f"[model] {key} = {tname} | size {size} | npre {npre} | bs {bs}", flush=True)

    store = FeatureStore(os.path.join(ROOT, "features"))
    run_id = f"in1k.pub.{key}"
    eval_tf = timm.data.create_transform(**cfg, is_training=False)

    for mkey, csvname, sub in (("in1k.train.v1L", "in1k.train.v1", "train"),
                               ("in1k.val.v1L", "in1k.val.v1", "val")):
        csvp = os.path.join(MANI, csvname + ".csv")
        src = _Source("imagefolder", root=os.path.join(IN1K, sub))
        ds = ZooEval(csvp, src, eval_tf)
        run_eval(model, ds, store, run_id, mkey, {"path": csvp, "n": len(ds)},
                 npre, want_pool, bs)

    csvp = os.path.join(MANI, "in1k.pairs10.v1.csv")
    src = _Source("imagefolder", root=os.path.join(IN1K, "train"))
    ods = OrbitDataset(csvp, src, size, "audit_v1", ORBIT_V)
    tail = renorm_tail(cfg["mean"], cfg["std"])
    ods.ta = v2.Compose([*ods.ta.transforms, tail])
    ods.tb = v2.Compose([*ods.tb.transforms, tail])
    run_orbits(model, ods, store, run_id, "in1k.pairs10.v1@audit_v1.o8",
               {"path": csvp, "n": len(ods)}, npre, want_pool, bs)
    print(f"[done] {run_id}", flush=True)


if __name__ == "__main__":
    main()
