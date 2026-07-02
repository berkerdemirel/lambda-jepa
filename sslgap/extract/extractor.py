"""Frozen-feature extraction: one pass per (checkpoint, manifest[, stack]) writing every requested
space to the store. All spaces come from the SAME forward pass per batch, so h and z are computed
on identical inputs — the two-space comparison is paired by construction.

Space names: "<branch>.h.cls", "<branch>.h.gap" (+ ".L<k>" per-layer when h_layers set),
"<branch>.z.<tap>" for every tap the branch's head module emits (PROTOCOL §1, §3)."""
import torch
from torch.amp import autocast
from torch.utils.data import DataLoader

from sslgap.models.backbones import trunk_features


def _loader(ds, bs, num_workers):
    return DataLoader(ds, batch_size=bs, num_workers=num_workers, shuffle=False,
                      persistent_workers=num_workers > 0, pin_memory=True)


def _batch_spaces(loaded, x, h_layers):
    """All spaces for one input batch -> {space: [B, d] float32 cpu}."""
    out = {}
    for bname, br in loaded.branches.items():
        feats = trunk_features(br.trunk, x, h_layers=h_layers)
        feats["image"] = x               # heads needing their own trunk pass (I-JEPA context-only)
        for kind in ("cls", "gap"):
            out[f"{bname}.h.{kind}"] = feats[kind].float().cpu()
        for l in h_layers:
            for kind in ("cls", "gap"):
                out[f"{bname}.h.{kind}.L{l:02d}"] = feats[f"{kind}.L{l:02d}"].float().cpu()
        if br.heads is not None:
            taps = br.heads(feats[br.head_input])
            for tap, v in taps.items():
                out[f"{bname}.z.{tap}"] = v.float().cpu()
    return out


@torch.inference_mode()
def extract_eval(loaded, dataset, store, manifest_key, manifest_info, bs=256, num_workers=8,
                 device="cuda", h_layers=()):
    loaded.eval_(device)
    acc, ys = {}, []
    for x, y in _loader(dataset, bs, num_workers):
        x = x.to(device, non_blocking=True)
        with autocast(device, dtype=torch.bfloat16):
            batch = _batch_spaces(loaded, x, h_layers)
        for k, v in batch.items():
            acc.setdefault(k, []).append(v)
        ys.append(torch.as_tensor(y))
    for space, chunks in acc.items():
        store.put(loaded.run_id, manifest_key, space, torch.cat(chunks).numpy())
    store.put_labels(loaded.run_id, manifest_key, torch.cat(ys).numpy())
    store.put_meta(loaded.run_id, manifest_key, {
        "kind": "eval", "manifest": manifest_info, "n": int(sum(len(y) for y in ys)),
        "spaces": sorted(acc), "h_layers": list(h_layers), "bs": bs,
        "method": loaded.method, "frame": loaded.frame, "probed_branch": loaded.probed_branch,
        "ckpt_provenance": loaded.provenance})
    return sorted(acc)


@torch.inference_mode()
def extract_pairs(loaded, pair_dataset, store, manifest_key, manifest_info, stack, bs=256,
                  num_workers=8, device="cuda"):
    """Two views per image -> "<space>.viewA"/"<space>.viewB" (final-layer spaces only)."""
    loaded.eval_(device)
    acc, ys = {}, []
    for xa, xb, y in _loader(pair_dataset, bs, num_workers):
        with autocast(device, dtype=torch.bfloat16):
            a = _batch_spaces(loaded, xa.to(device, non_blocking=True), h_layers=())
            b = _batch_spaces(loaded, xb.to(device, non_blocking=True), h_layers=())
        for k in a:
            acc.setdefault(f"{k}.viewA", []).append(a[k])
            acc.setdefault(f"{k}.viewB", []).append(b[k])
        ys.append(torch.as_tensor(y))
    for space, chunks in acc.items():
        store.put(loaded.run_id, manifest_key, space, torch.cat(chunks).numpy())
    store.put_labels(loaded.run_id, manifest_key, torch.cat(ys).numpy())
    store.put_meta(loaded.run_id, manifest_key, {
        "kind": "pairs", "stack": stack, "manifest": manifest_info,
        "n": int(sum(len(y) for y in ys)), "spaces": sorted(acc),
        "method": loaded.method, "frame": loaded.frame,
        "ckpt_provenance": loaded.provenance})
    return sorted(acc)
