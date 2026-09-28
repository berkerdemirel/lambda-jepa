"""Checkpoint -> feature store: train/val features per station and the 8-view orbit stores."""
import os

import hydra
from omegaconf import DictConfig

from sslgap.ckpt import adapters
from sslgap.data import (EvalDataset, OrbitDataset, PairDataset, STACKS, _Source,
                         build_manifest_imagefolder, seed_everything)
from sslgap.extract import FeatureStore, extract_eval, extract_pairs, extract_views

def _manifests(cfg, frame, manifest_dir):
    ds = frame["dataset"]
    tag = {"imagenet100": "in100", "imagenet1k": "in1k"}[ds]
    root = os.path.expanduser(frame["data_root"])
    if cfg.get("linspace_n"):
        train_spec = (f"{tag}.linspace{cfg.linspace_n}.v1",
                      dict(sub="train", per_class=None, linspace_n=cfg.linspace_n))
    else:
        tpc = cfg.train_per_class
        if tpc is None:
            tpc = {"imagenet100": 500, "imagenet1k": 0}[ds]
        if ds == "imagenet1k" and tpc:
            raise ValueError(f"in1k train manifests are FULL-train only; "
                             f"got train_per_class={tpc}")
        train_spec = (f"{tag}.train{tpc}.v1" if tpc else f"{tag}.train.v1",
                      dict(sub="train", per_class=tpc or None, linspace_n=None))
    specs = {"train": train_spec,
             "val": (f"{tag}.val.v1", dict(sub="val", per_class=None, linspace_n=None)),
             "pairs": (f"{tag}.pairs{cfg.pairs_per_class}.v1",
                       dict(sub="train", per_class=cfg.pairs_per_class, linspace_n=None))}
    build = lambda name, kw: build_manifest_imagefolder(
        os.path.join(root, kw["sub"]), os.path.join(manifest_dir, name + ".csv"),
        per_class=kw["per_class"], seed=cfg.manifest_seed, linspace_n=kw["linspace_n"])
    source = lambda kw: _Source("imagefolder", root=os.path.join(root, kw["sub"]))
    out = {}
    for key, (name, kw) in specs.items():
        csv_path = os.path.join(manifest_dir, name + ".csv")
        info = build(name, kw) if not os.path.exists(csv_path) else {"path": csv_path, "n": None}
        out[key] = (name, csv_path, info, source(kw))
    return out

@hydra.main(version_base=None, config_path="configs", config_name="extract")
def main(cfg: DictConfig):
    seed_everything(cfg.seed)
    loaded = adapters.load(cfg.adapter, os.path.expanduser(cfg.ckpt), cfg.run_id,
                           random_init=cfg.random_init, seed=cfg.seed)
    store = FeatureStore(cfg.store_root, cap_gb=cfg.cap_gb)
    mans = _manifests(cfg, loaded.frame, os.path.expanduser(cfg.manifest_dir))
    img = loaded.frame["img_size"]

    if cfg.do_eval:
        for key in ("train", "val"):
            name, csv_path, info, source = mans[key]
            mkey = name + ("L" if cfg.h_layers else "")
            spaces = extract_eval(loaded, EvalDataset(csv_path, source, img), store,
                                  manifest_key=mkey, manifest_info=info, bs=cfg.bs,
                                  num_workers=cfg.num_workers, device=cfg.device,
                                  h_layers=tuple(cfg.h_layers), seed=cfg.seed)
            print(f"[extract] {cfg.run_id} {mkey}: {len(spaces)} spaces")

    if cfg.do_pairs:
        name, csv_path, info, source = mans["pairs"]
        own = f"own_{loaded.method}"
        stacks = ["audit_v1"] + ([own] if own in STACKS and own != "own_lejepa" else [])
        for stack in stacks:
            key = f"{name}@{stack}"
            spaces = extract_pairs(loaded, PairDataset(csv_path, source, img, stack=stack), store,
                                   manifest_key=key, manifest_info=info, stack=stack, bs=cfg.bs,
                                   num_workers=cfg.num_workers, device=cfg.device, seed=cfg.seed)
            print(f"[extract] {cfg.run_id} {key}: {len(spaces)} spaces")

    if cfg.get("orbit_v"):
        name, csv_path, info, source = mans["pairs"]
        own = f"own_{loaded.method}"
        stacks = list(cfg.get("orbit_stacks") or
                      ["audit_v1"] + ([own] if own in STACKS and own != "own_lejepa" else []))
        keep = set(cfg.orbit_spaces) if cfg.get("orbit_spaces") else None
        for stack in stacks:
            key = f"{name}@{stack}.o{cfg.orbit_v}"
            ds = OrbitDataset(csv_path, source, img, stack, cfg.orbit_v)
            spaces = extract_views(loaded, ds, store, manifest_key=key, manifest_info=info,
                                   stack=stack, bs=cfg.bs, num_workers=cfg.num_workers,
                                   device=cfg.device, h_layers=tuple(cfg.h_layers), seed=cfg.seed,
                                   keep=keep)
            print(f"[extract] {cfg.run_id} {key}: {len(spaces)} spaces")

    print(f"[extract] done: {cfg.run_id}")

if __name__ == "__main__":
    main()
