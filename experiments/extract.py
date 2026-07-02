"""Extraction driver: legacy checkpoint -> feature store (eval train+val, pairs under audit_v1 +
the method's own stack). Builds versioned manifests on first use.

  sbatch slurm/extract.sbatch ckpt=/nfs/.../lejepa/ckpt_lamb002.pt adapter=lejepa_minimal \
      run_id=toy.lejepa-lamb002.ext
  sbatch slurm/extract.sbatch ckpt=/nfs/.../ssl_explore/outputs/inv_dino-in100_ep100.pt \
      adapter=sslx_dino run_id=in100.dino-ctrl.ep100.ext
"""
import os

import hydra
from omegaconf import DictConfig

from sslgap.ckpt import adapters
from sslgap.data import (EvalDataset, PairDataset, STACKS, _Source, build_manifest_imagefolder,
                         build_manifest_imagenette)
from sslgap.extract import FeatureStore, extract_eval, extract_pairs


def _manifests(cfg, frame, manifest_dir):
    """(train, val, pairs) manifest specs for the checkpoint's dataset; built if absent."""
    ds = frame.get("dataset", "imagenette")
    if ds == "imagenette":
        specs = {"train": ("imagenette.train.v1", dict(split="train")),
                 "val": ("imagenette.val.v1", dict(split="validation")),
                 "pairs": ("imagenette.train.v1", dict(split="train"))}
        build = lambda name, kw: build_manifest_imagenette(kw["split"],
                                                           os.path.join(manifest_dir, name + ".csv"))
        source = lambda kw: _Source("hf-imagenette", split=kw["split"])
    else:
        root = os.path.expanduser(frame["data_root"])
        if cfg.get("linspace_n"):                       # parity manifests (CAMPAIGN_LOG rule)
            train_spec = (f"in100.linspace{cfg.linspace_n}.v1",
                          dict(sub="train", per_class=None, linspace_n=cfg.linspace_n))
        else:
            train_spec = (f"in100.train{cfg.train_per_class}.v1",
                          dict(sub="train", per_class=cfg.train_per_class, linspace_n=None))
        specs = {"train": train_spec,
                 "val": ("in100.val.v1", dict(sub="val", per_class=None, linspace_n=None)),
                 "pairs": (f"in100.pairs{cfg.pairs_per_class}.v1",
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
    loaded = adapters.load(cfg.adapter, os.path.expanduser(cfg.ckpt), cfg.run_id,
                           random_init=cfg.random_init, seed=cfg.seed)
    store = FeatureStore(cfg.store_root, cap_gb=cfg.cap_gb)
    mans = _manifests(cfg, loaded.frame, os.path.expanduser(cfg.manifest_dir))
    img = loaded.frame["img_size"]

    if cfg.do_eval:
        for key in ("train", "val"):
            name, csv_path, info, source = mans[key]
            spaces = extract_eval(loaded, EvalDataset(csv_path, source, img), store,
                                  manifest_key=name, manifest_info=info, bs=cfg.bs,
                                  num_workers=cfg.num_workers, device=cfg.device,
                                  h_layers=tuple(cfg.h_layers))
            print(f"[extract] {cfg.run_id} {name}: {len(spaces)} spaces")

    if cfg.do_pairs:
        name, csv_path, info, source = mans["pairs"]
        own = f"own_{loaded.method}"
        stacks = ["audit_v1"] + ([own] if own in STACKS and own != "own_lejepa" else [])
        for stack in stacks:
            key = f"{name}@{stack}"
            spaces = extract_pairs(loaded, PairDataset(csv_path, source, img, stack=stack), store,
                                   manifest_key=key, manifest_info=info, stack=stack, bs=cfg.bs,
                                   num_workers=cfg.num_workers, device=cfg.device)
            print(f"[extract] {cfg.run_id} {key}: {len(spaces)} spaces")
    print(f"[extract] done: {cfg.run_id}")


if __name__ == "__main__":
    main()
