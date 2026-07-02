"""Probe driver: frozen probes (linear_house_v1 for parity, linear_l2_v1 primary, knn_v1) on every
space present in both train and val manifests of a run.

  sbatch slurm/probe.sbatch run_id=in100.dino-ctrl.ep100.ext
"""
import os

import hydra
import numpy as np
import pandas as pd
import torch
from omegaconf import DictConfig

from sslgap.extract import FeatureStore
from sslgap.probes import knn_self_test, knn_topk_acc, linear_house_v1, linear_l2_v1, linear_raw_v1


@hydra.main(version_base=None, config_path="configs", config_name="probe")
def main(cfg: DictConfig):
    if not knn_self_test():
        raise RuntimeError("knn_self_test failed — kNN parity broken, refusing to probe")
    store = FeatureStore(cfg.store_root)
    run_dir = os.path.join(os.path.expanduser(cfg.store_root), cfg.run_id)
    manifests = [m for m in sorted(os.listdir(run_dir)) if "@" not in m]
    man_tr = cfg.train_manifest or next(m for m in manifests if "train" in m)
    man_va = cfg.val_manifest or next(m for m in manifests if "val" in m)
    ytr, yva = store.labels(cfg.run_id, man_tr), store.labels(cfg.run_id, man_va)
    num_classes = int(max(ytr.max(), yva.max())) + 1
    device = cfg.device if torch.cuda.is_available() else "cpu"

    rows = []
    spaces = sorted(set(store.spaces(cfg.run_id, man_tr)) & set(store.spaces(cfg.run_id, man_va)))
    for space in spaces:
        Xtr = np.asarray(store.get(cfg.run_id, man_tr, space), dtype=np.float32)
        Xva = np.asarray(store.get(cfg.run_id, man_va, space), dtype=np.float32)
        res = {}
        if "raw" in cfg.probes:
            res["linear_raw_v1"] = linear_raw_v1(Xtr, ytr, Xva, yva, num_classes,
                                                 device=device, seed=cfg.seed)["val_acc"]
        if "house" in cfg.probes:
            res["linear_house_v1"] = linear_house_v1(Xtr, ytr, Xva, yva, num_classes,
                                                     device=device, seed=cfg.seed)["val_acc"]
        if "l2" in cfg.probes:
            res["linear_l2_v1"] = linear_l2_v1(Xtr, ytr, Xva, yva, num_classes,
                                               device=device, seed=cfg.seed)["val_acc"]
        if "knn" in cfg.probes:
            res["knn_v1_k200"] = knn_topk_acc(Xtr, ytr, Xva, yva, num_classes,
                                              knn_k=200, knn_t=0.1, device=device)
            res["knn_v1_k20"] = knn_topk_acc(Xtr, ytr, Xva, yva, num_classes,
                                             knn_k=20, knn_t=0.1, device=device)
        print(f"[probe] {cfg.run_id} {space}: " +
              " ".join(f"{k}={v:.4f}" for k, v in res.items()))
        rows += [{"run_id": cfg.run_id, "train_manifest": man_tr, "val_manifest": man_va,
                  "space": space, "probe": k, "val_acc": v, "n_train": len(ytr),
                  "n_val": len(yva), "num_classes": num_classes} for k, v in res.items()]
    out_dir = os.path.join(cfg.results_root, "probes")
    os.makedirs(out_dir, exist_ok=True)
    pd.DataFrame(rows).to_csv(os.path.join(out_dir, f"{cfg.run_id}.csv"), index=False)
    print(f"[probe] done: {cfg.run_id} -> {out_dir}")


if __name__ == "__main__":
    main()
