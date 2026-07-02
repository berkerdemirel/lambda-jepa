"""Cross-MODEL comparison on a shared manifest (D-009): for chosen spaces of two runs, compute the
similarity triple directly (dims permitting) and in the relative-representation frame (dataset
anchors, shared image ids), plus the battery on the relrep space.

  python experiments/compare.py run_a=toy.lejepa-lamb002.ext run_b=toy.infonce.ext \
      manifest=imagenette.train.v1 'spaces=[student.h.cls]'
"""
import os

import hydra
import numpy as np
import pandas as pd
from omegaconf import DictConfig

from sslgap.extract import FeatureStore
from sslgap.metrics import cross
from sslgap.metrics.relrep import cross_model_relrep


@hydra.main(version_base=None, config_path="configs", config_name="compare")
def main(cfg: DictConfig):
    store = FeatureStore(cfg.store_root)
    rows = []
    for space in cfg.spaces:
        Xa = np.asarray(store.get(cfg.run_a, cfg.manifest, space), dtype=np.float64)
        Xb = np.asarray(store.get(cfg.run_b, cfg.manifest, space), dtype=np.float64)
        lab = store.labels(cfg.run_a, cfg.manifest)
        if not (lab == store.labels(cfg.run_b, cfg.manifest)).all():
            raise ValueError("manifest order mismatch between run_a and run_b")
        Ra, Rb = cross_model_relrep(Xa, Xb, A=cfg.anchors or None, seed=cfg.seed, abs_transform=cfg.abs_transform)
        pairs = [("relrep", Ra, Rb)] + ([("direct", Xa, Xb)] if Xa.shape[1] == Xb.shape[1] else [])
        for frame, A, B in pairs:
            rows += [{"run_a": cfg.run_a, "run_b": cfg.run_b, "manifest": cfg.manifest,
                      "space": space, "frame": frame, "A": A.shape[1], "metric": name,
                      "value": fn(A, B)}
                     for name, fn in [("cka_linear", cross.cka_linear),
                                      ("neighbor_jaccard", cross.neighbor_jaccard),
                                      ("procrustes_distance", cross.procrustes_distance)]]
            rows.append({"run_a": cfg.run_a, "run_b": cfg.run_b, "manifest": cfg.manifest,
                         "space": space, "frame": frame, "A": A.shape[1],
                         "metric": "knn_label_agreement",
                         "value": cross.knn_label_agreement(A, B, lab)})
        print(f"[compare] {space}: done ({len(pairs)} frames)")
    out_dir = os.path.join(cfg.results_root, "compare")
    os.makedirs(out_dir, exist_ok=True)
    out = os.path.join(out_dir, f"{cfg.run_a}__vs__{cfg.run_b}.csv")
    pd.DataFrame(rows).to_csv(out, index=False)
    print(f"[compare] wrote {out}")


if __name__ == "__main__":
    main()
