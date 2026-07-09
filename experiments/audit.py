"""Battery driver: stored features -> metric CSVs (numbers only; interpretation happens in card
discussions). Runs the single-space battery on every space of the train manifest, the pair battery
on every pairs manifest, and the cross-space similarity triple between the probed branch's h and
each of its z taps.

  sbatch slurm/audit.sbatch run_id=toy.lejepa-lamb002.ext

Wide-space runs can outlive the 8 h wall (byol's 8192-d taps ≈ 1.5 h/space at IN-100 scale) —
the pre-planned mitigation (HANDOVER 2026-07-09) is splitting by space:
  sbatch slurm/audit.sbatch run_id=... 'spaces=[student.z.proj.tap1]' out_suffix=part1 \
         do_pairs=false do_cross=false
then concat the {run_id}.{suffix}.csv pieces into {run_id}.csv.
"""
import os

import hydra
import pandas as pd
from omegaconf import DictConfig

from sslgap.extract import FeatureStore
from sslgap.metrics.battery import run_battery, run_cross_battery, run_pair_battery


@hydra.main(version_base=None, config_path="configs", config_name="audit")
def main(cfg: DictConfig):
    store = FeatureStore(cfg.store_root)
    run_dir = os.path.join(os.path.expanduser(cfg.store_root), cfg.run_id)
    manifests = sorted(os.listdir(run_dir))
    eval_mans = [m for m in manifests if "@" not in m and
                 (cfg.train_manifest is None or m == cfg.train_manifest)]
    pair_mans = [m for m in manifests if "@" in m]
    out_dir = os.path.join(cfg.results_root, "battery")
    os.makedirs(out_dir, exist_ok=True)

    frames = []
    man = cfg.train_manifest or next(m for m in eval_mans if "train" in m)
    meta = store.meta(cfg.run_id, man)
    only = set(cfg.spaces) if cfg.get("spaces") else None
    suffix = f".{cfg.out_suffix}" if cfg.get("out_suffix") else ""
    for space in store.spaces(cfg.run_id, man):
        if only and space not in only:
            continue
        print(f"[audit] battery {cfg.run_id} {man} {space}")
        frames.append(run_battery(store, cfg.run_id, man, space,
                                  n_boot=cfg.n_boot, max_n=cfg.max_n, seed=cfg.seed))
    if frames:
        battery = pd.concat(frames, ignore_index=True)
        battery.to_csv(os.path.join(out_dir, f"{cfg.run_id}{suffix}.csv"), index=False)

    pair_frames = []
    if not cfg.get("do_pairs", True):
        pair_mans = []
    for man_key in pair_mans:
        bases = sorted({s.rsplit(".view", 1)[0] for s in store.spaces(cfg.run_id, man_key)})
        print(f"[audit] pairs {cfg.run_id} {man_key} ({len(bases)} spaces)")
        pair_frames.append(run_pair_battery(store, cfg.run_id, man_key, bases, seed=cfg.seed))
    if pair_frames:
        pd.concat(pair_frames, ignore_index=True).to_csv(
            os.path.join(out_dir, f"{cfg.run_id}.pairs.csv"), index=False)

    if not cfg.get("do_cross", True):
        print(f"[audit] done (split part): {cfg.run_id}{suffix} -> {out_dir}")
        return
    branch = meta["probed_branch"]
    spaces = store.spaces(cfg.run_id, man)
    # PROTOCOL §3 h per method (D-003v2), recorded by the adapter; older M0 extractions
    # (lejepa_minimal / sslx_dino) predate h_space and keep the cls-first fallback.
    h_ref = (meta.get("ckpt_provenance") or {}).get("h_space")
    if not h_ref:
        h_ref = f"{branch}.h.cls" if f"{branch}.h.cls" in spaces else f"{branch}.h.gap"
    z_taps = [s for s in spaces if s.startswith(f"{branch}.z.") and s != h_ref]
    if not z_taps:      # I-JEPA: probed branch is the teacher, z lives on the student+predictor
        z_taps = [s for s in spaces if ".z." in s and s != h_ref]
    cross_frames = [run_cross_battery(store, cfg.run_id, man, h_ref, z, seed=cfg.seed)
                    for z in z_taps]
    if cross_frames:
        pd.concat(cross_frames, ignore_index=True).to_csv(
            os.path.join(out_dir, f"{cfg.run_id}.cross.csv"), index=False)
    print(f"[audit] done: {cfg.run_id} -> {out_dir}")


if __name__ == "__main__":
    main()
