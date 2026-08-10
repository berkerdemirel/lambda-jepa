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
from sslgap.data import (EvalDataset, FovealPairDataset, OrbitDataset, PairDataset, STACKS,
                         _Source, build_manifest_imagefolder, seed_everything,
                         build_manifest_imagenette)
from sslgap.extract import FeatureStore, extract_eval, extract_pairs, extract_views


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
        # manifest names are dataset-keyed (D-066): in100 names byte-identical to pre-D-066
        tag = {"imagenet100": "in100", "imagenet1k": "in1k"}[ds]
        root = os.path.expanduser(frame["data_root"])
        if cfg.get("linspace_n"):                       # parity manifests (CAMPAIGN_LOG rule)
            train_spec = (f"{tag}.linspace{cfg.linspace_n}.v1",
                          dict(sub="train", per_class=None, linspace_n=cfg.linspace_n))
        else:
            # train_per_class: null -> the PROTOCOL standard for the dataset (D-066: in1k =
            # FULL train; in100 keeps the m50k 500/class frame), 0 -> full split explicitly,
            # N -> N/class. Dataset-keyed guard added 2026-08-10: the 08-08/09 e24voas+e27lej
            # landings inherited a yaml default of 500 on in1k and landed on the SUPERSEDED
            # subsampled frame (train500) while their vm4 comparator was on the standard one.
            tpc = cfg.train_per_class
            if tpc is None:
                tpc = {"imagenet100": 500, "imagenet1k": 0}[ds]
            if ds == "imagenet1k" and tpc:
                # D-092 (Berker 2026-08-10: "remove 500k variant we do not need to use that
                # in any case"): the in1k eval frame is FULL train ONLY; the subsampled
                # variant is removed, not just non-default. linspace_n parity manifests
                # are a separate, still-legal object.
                raise ValueError(f"in1k train manifests are FULL-train only (D-066/D-092); "
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
            mkey = name + ("L" if cfg.h_layers else "")   # per-layer request -> own store dir;
            spaces = extract_eval(loaded, EvalDataset(csv_path, source, img), store,   # landed
                                  manifest_key=mkey, manifest_info=info, bs=cfg.bs,    # plain-eval
                                  num_workers=cfg.num_workers, device=cfg.device,      # dirs stay
                                  h_layers=tuple(cfg.h_layers), seed=cfg.seed)         # untouched
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

    if cfg.get("orbit_v"):       # V-view orbit stores for overlap/invariance along the E02 depth
        name, csv_path, info, source = mans["pairs"]     # axis (HEAD_OVERLAP_LIPSCHITZ.md)
        own = f"own_{loaded.method}"
        stacks = ["audit_v1"] + ([own] if own in STACKS and own != "own_lejepa" else [])
        for stack in stacks:
            key = f"{name}@{stack}.o{cfg.orbit_v}"
            ds = OrbitDataset(csv_path, source, img, stack, cfg.orbit_v)
            spaces = extract_views(loaded, ds, store, manifest_key=key, manifest_info=info,
                                   stack=stack, bs=cfg.bs, num_workers=cfg.num_workers,
                                   device=cfg.device, h_layers=tuple(cfg.h_layers), seed=cfg.seed)
            print(f"[extract] {cfg.run_id} {key}: {len(spaces)} spaces")

    if cfg.get("foveal"):        # E14 (D-031): event = zoo members, ctx = tokenizer runs,
        name, csv_path, info, source = mans["pairs"]   # blur = no-fovea null (addendum)
        for mode in (("event", "ctx") if cfg.foveal == "both" else (cfg.foveal,)):
            stack = {"event": "foveal_v1", "ctx": "foveal_v1_ctx", "blur": "blur_v1"}[mode]
            ds = FovealPairDataset(csv_path, source, img, stack, "ctx" if mode == "ctx" else "event")
            key = f"{name}@{stack}"
            spaces = extract_pairs(loaded, ds, store, manifest_key=key, manifest_info=info,
                                   stack=stack, bs=cfg.bs, num_workers=cfg.num_workers,
                                   device=cfg.device, seed=cfg.seed)
            print(f"[extract] {cfg.run_id} {key}: {len(spaces)} spaces")
    print(f"[extract] done: {cfg.run_id}")


if __name__ == "__main__":
    main()
