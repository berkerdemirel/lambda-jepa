"""Frozen probes on a run's feature store: linear on raw features (AdamW, early stopping) and kNN (k = 200)."""
import os

import hydra
import numpy as np
import pandas as pd
import torch
from omegaconf import DictConfig

from sslgap.extract import FeatureStore
from sslgap.probes import (knn_self_test, knn_topk_acc, linear_house_v1, linear_house_v2,
                           linear_l2_v1, linear_l2_v2, linear_raw_v1, linear_raw_v2)

LINEAR = {"raw": [("linear_raw_v1", linear_raw_v1), ("linear_raw_v2", linear_raw_v2)],
          "house": [("linear_house_v1", linear_house_v1), ("linear_house_v2", linear_house_v2)],
          "l2": [("linear_l2_v1", linear_l2_v1), ("linear_l2_v2", linear_l2_v2)]}

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
    if cfg.get("space"):
        spaces = [s for s in spaces if s == cfg.space]
    for space in spaces:
        Xtr = np.asarray(store.get(cfg.run_id, man_tr, space), dtype=np.float32)
        Xva = np.asarray(store.get(cfg.run_id, man_va, space), dtype=np.float32)
        res = {}
        for family, versions in LINEAR.items():
            if family in cfg.probes:
                for name, fn in versions:
                    r = fn(Xtr, ytr, Xva, yva, num_classes, device=device, seed=cfg.seed)
                    res[name] = {"val_acc": r["val_acc"], "best_ep": r["best_ep"],
                                 "epochs_run": r["epochs_run"]}
        if "knn" in cfg.probes:
            res["knn_v1_k200"] = {"val_acc": knn_topk_acc(Xtr, ytr, Xva, yva, num_classes,
                                                          knn_k=200, knn_t=0.1, device=device)}
            res["knn_v1_k20"] = {"val_acc": knn_topk_acc(Xtr, ytr, Xva, yva, num_classes,
                                                         knn_k=20, knn_t=0.1, device=device)}
        print(f"[probe] {cfg.run_id} {space}: " +
              " ".join(f"{k}={v['val_acc']:.4f}" for k, v in res.items()), flush=True)
        rows += [{"run_id": cfg.run_id, "train_manifest": man_tr, "val_manifest": man_va,
                  "space": space, "probe": k, "val_acc": v["val_acc"], "n_train": len(ytr),
                  "n_val": len(yva), "num_classes": num_classes,
                  "best_ep": v.get("best_ep"), "epochs_run": v.get("epochs_run")}
                 for k, v in res.items()]
        out_dir = os.path.join(cfg.results_root, "probes")
        os.makedirs(out_dir, exist_ok=True)
        suffix = f".part_{cfg.space}" if cfg.get("space") else ""
        pd.DataFrame(rows).to_csv(os.path.join(out_dir, f"{cfg.run_id}{suffix}.csv"), index=False)
    print(f"[probe] done: {cfg.run_id} -> {os.path.join(cfg.results_root, 'probes')}")

if __name__ == "__main__":
    main()
