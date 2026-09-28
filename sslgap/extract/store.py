"""Feature store: one fp16 .npy per (run_id, manifest, space) plus meta.json."""
import json
import os

import numpy as np

from sslgap.ckpt.schema import provenance_stamp, write_meta

class FeatureStore:
    def __init__(self, root, cap_gb=3000):
        self.root = os.path.expanduser(root)
        self.cap = cap_gb * (1 << 30)
        os.makedirs(self.root, exist_ok=True)

    def dir(self, run_id, manifest_key):
        d = os.path.join(self.root, run_id, manifest_key)
        os.makedirs(d, exist_ok=True)
        return d

    def _used_bytes(self):
        total = 0
        for dp, _, files in os.walk(self.root):
            total += sum(os.path.getsize(os.path.join(dp, f)) for f in files)
        return total

    def put(self, run_id, manifest_key, space, X):
        X = np.ascontiguousarray(X, dtype=np.float16)
        if X.shape[1] > 8192:
            raise ValueError(f"{space}: d={X.shape[1]} > 8192 — recompute, don't store")
        if self._used_bytes() + X.nbytes > self.cap:
            raise RuntimeError(
                f"feature store cap exceeded ({self.cap >> 30} GB) — purge or raise cap")
        path = os.path.join(self.dir(run_id, manifest_key), f"{space}.npy")
        np.save(path, X)
        return path

    def put_labels(self, run_id, manifest_key, y):
        np.save(os.path.join(self.dir(run_id, manifest_key), "labels.npy"),
                np.asarray(y, dtype=np.int64))

    def put_meta(self, run_id, manifest_key, meta):
        write_meta(os.path.join(self.dir(run_id, manifest_key), "meta.json"),
                   {**meta, "provenance": provenance_stamp()})

    def get(self, run_id, manifest_key, space, mmap=True):
        path = os.path.join(self.root, run_id, manifest_key, f"{space}.npy")
        return np.load(path, mmap_mode="r" if mmap else None)

    def labels(self, run_id, manifest_key):
        return np.load(os.path.join(self.root, run_id, manifest_key, "labels.npy"))

    def spaces(self, run_id, manifest_key):
        d = os.path.join(self.root, run_id, manifest_key)
        return sorted(f[:-4] for f in os.listdir(d)
                      if f.endswith(".npy") and f != "labels.npy")

    def meta(self, run_id, manifest_key):
        with open(os.path.join(self.root, run_id, manifest_key, "meta.json")) as f:
            return json.load(f)
