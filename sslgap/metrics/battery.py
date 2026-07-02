"""The metric battery: pure functions over stored feature arrays, computed identically on every
space (PROTOCOL §6 estimator discipline is enforced HERE — variants, PCA-k, bootstrap, nulls).

run_battery(store, run_id, manifest_key, space) -> tidy DataFrame:
  (space, metric, variant, value, ci_lo, ci_hi, n, d, null_gauss, seed)
variant ∈ {raw, l2} × {full, pca64}; metrics declare which apply. Bootstrap resamples images;
spectral metrics bootstrap on a row-subsample (cost class), noted in the `boot` column."""
from dataclasses import dataclass, field
from typing import Callable

import numpy as np
import pandas as pd

from sslgap.metrics import cross, isotropy, pairs, single, spectra
from sslgap.metrics.nulls import gaussian_match


@dataclass
class MetricSpec:
    name: str
    fn: Callable                      # fn(X, seed=...) -> float | dict[str, float]
    dim_sensitive: bool = False       # also computed on the PCA-64 projection
    l2_variant: bool = True           # also computed on row-L2-normalized features
    boot: str = "cheap"               # "cheap" (full-N resample) | "spectral" (10k subsample) | "none"
    gauss_null: bool = False          # also computed on the moment-matched Gaussian
    kwargs: dict = field(default_factory=dict)


def _spectrum_metric(agg):
    def fn(X, seed=0):
        eigs = spectra.covariance_eigs(X)
        return agg(eigs)
    return fn


DEFAULT_BATTERY = [
    MetricSpec("uniformity", lambda X, seed=0: single.uniformity(X, seed=seed),
               dim_sensitive=True, l2_variant=False, gauss_null=True),
    MetricSpec("variance_floor", lambda X, seed=0: single.variance_floor(X), l2_variant=False),
    MetricSpec("offdiag_redundancy", lambda X, seed=0: single.offdiag_redundancy(X),
               l2_variant=False, boot="spectral"),
    MetricSpec("collapse_margin", lambda X, seed=0: single.collapse_margin(X, seed=seed),
               l2_variant=False, boot="none"),
    MetricSpec("rankme", lambda X, seed=0: spectra.rankme(X), boot="spectral", gauss_null=True),
    MetricSpec("effective_rank", _spectrum_metric(spectra.effective_rank), boot="spectral",
               gauss_null=True),
    MetricSpec("participation_ratio", _spectrum_metric(spectra.participation_ratio),
               boot="spectral"),
    MetricSpec("alpha", _spectrum_metric(spectra.power_law_alpha), boot="spectral"),
    MetricSpec("epps_pulley", lambda X, seed=0: isotropy.epps_pulley(X, seed=seed),
               dim_sensitive=True, gauss_null=True, boot="spectral"),
    MetricSpec("kurt_topeig",
               lambda X, seed=0: {"mean": float(spectra.top_eigvec_excess_kurtosis(X).mean()),
                                  "worst": float(np.abs(spectra.top_eigvec_excess_kurtosis(X)).max())},
               l2_variant=False, gauss_null=True, boot="spectral"),
    MetricSpec("kurt_slices_mean_abs",
               lambda X, seed=0: float(np.abs(spectra.random_slice_excess_kurtosis(X, seed=seed)).mean()),
               l2_variant=False, gauss_null=True, boot="spectral"),
]

PAIR_BATTERY = [
    ("alignment", pairs.alignment),
    ("cos_invariance", pairs.cos_invariance),
]

CROSS_BATTERY = [  # (name, fn(X, Y, **kw)) — the similarity triple + neighborhood consistency
    ("cka_linear", cross.cka_linear),
    ("neighbor_jaccard", cross.neighbor_jaccard),
    ("procrustes_distance", cross.procrustes_distance),
]


def _l2(X):
    return X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-12)


def _pca_k(X, k=64):
    Xc = X - X.mean(0)
    kk = min(k, X.shape[1])
    _, V = np.linalg.eigh((Xc.T @ Xc) / (Xc.shape[0] - 1))
    return Xc @ V[:, ::-1][:, :kk]


def _rows_for(spec, X, variant, n_boot, seed):
    rows = []
    vals = spec.fn(X, seed=seed, **spec.kwargs)
    vals = vals if isinstance(vals, dict) else {"": vals}
    boots = {k: [] for k in vals}
    if spec.boot != "none" and n_boot > 0:
        rng = np.random.default_rng(seed)
        m = min(X.shape[0], 10000) if spec.boot == "spectral" else X.shape[0]
        nb = min(n_boot, 32) if spec.boot == "spectral" else n_boot
        for b in range(nb):
            idx = rng.choice(X.shape[0], m, replace=True)
            bv = spec.fn(X[idx], seed=seed + 1 + b, **spec.kwargs)
            bv = bv if isinstance(bv, dict) else {"": bv}
            for k in vals:
                boots[k].append(bv[k])
    for k, v in vals.items():
        name = spec.name if not k else f"{spec.name}.{k}"
        lo, hi = (np.percentile(boots[k], [2.5, 97.5]) if boots[k] else (np.nan, np.nan))
        rows.append({"metric": name, "variant": variant, "value": v,
                     "ci_lo": float(lo), "ci_hi": float(hi),
                     "n": X.shape[0], "d": X.shape[1], "boot": spec.boot})
    return rows


def run_battery(store, run_id, manifest_key, space, specs=None, n_boot=100, pca_k=64,
                max_n=None, seed=0):
    specs = specs or DEFAULT_BATTERY
    X = np.asarray(store.get(run_id, manifest_key, space), dtype=np.float64)
    if max_n and X.shape[0] > max_n:
        X = X[np.random.default_rng(seed).choice(X.shape[0], max_n, replace=False)]
    variants = {"raw|full": X}
    if any(s.l2_variant for s in specs):
        variants["l2|full"] = _l2(X)
    if any(s.dim_sensitive for s in specs) and X.shape[1] > pca_k:
        variants["raw|pca64"] = _pca_k(X, pca_k)
    gauss = {vk: gaussian_match(V, seed=seed) for vk, V in variants.items()
             if any(s.gauss_null for s in specs)}
    rows = []
    for spec in specs:
        for vk, V in variants.items():
            norm_v, dim_v = vk.split("|")
            if norm_v == "l2" and not spec.l2_variant:
                continue
            if dim_v == "pca64" and not spec.dim_sensitive:
                continue
            rs = _rows_for(spec, V, vk, n_boot, seed)
            if spec.gauss_null and vk in gauss:
                nulls = spec.fn(gauss[vk], seed=seed, **spec.kwargs)
                nulls = nulls if isinstance(nulls, dict) else {"": nulls}
                for r in rs:
                    sub = r["metric"][len(spec.name) + 1:] if r["metric"] != spec.name else ""
                    r["null_gauss"] = nulls.get(sub, np.nan)
            rows += rs
    df = pd.DataFrame(rows)
    df.insert(0, "space", space)
    df.insert(0, "manifest", manifest_key)
    df.insert(0, "run_id", run_id)
    return df


def run_pair_battery(store, run_id, manifest_key, base_spaces, seed=0, n_boot=200):
    """Pair metrics per space (needs <space>.viewA/.viewB in the store)."""
    rows = []
    for space in base_spaces:
        A = np.asarray(store.get(run_id, manifest_key, f"{space}.viewA"), dtype=np.float64)
        B = np.asarray(store.get(run_id, manifest_key, f"{space}.viewB"), dtype=np.float64)
        rng = np.random.default_rng(seed)
        for name, fn in PAIR_BATTERY:
            v = fn(A, B)
            boots = []
            for _ in range(n_boot):
                idx = rng.choice(A.shape[0], A.shape[0], replace=True)
                boots.append(fn(A[idx], B[idx]))
            lo, hi = np.percentile(boots, [2.5, 97.5])
            rows.append({"run_id": run_id, "manifest": manifest_key, "space": space,
                         "metric": name, "variant": "pairs", "value": v,
                         "ci_lo": float(lo), "ci_hi": float(hi),
                         "n": A.shape[0], "d": A.shape[1], "boot": "cheap"})
    return pd.DataFrame(rows)


def run_cross_battery(store, run_id, manifest_key, space_x, space_y, seed=0):
    """The similarity triple between two spaces on the same manifest (CKA never alone)."""
    X = np.asarray(store.get(run_id, manifest_key, space_x), dtype=np.float64)
    Y = np.asarray(store.get(run_id, manifest_key, space_y), dtype=np.float64)
    lab = store.labels(run_id, manifest_key)
    rows = [{"run_id": run_id, "manifest": manifest_key, "space": f"{space_x}~{space_y}",
             "metric": name, "variant": "cross", "value": fn(X, Y),
             "ci_lo": np.nan, "ci_hi": np.nan, "n": X.shape[0],
             "d": f"{X.shape[1]}x{Y.shape[1]}", "boot": "none"}
            for name, fn in CROSS_BATTERY]
    rows.append({"run_id": run_id, "manifest": manifest_key, "space": f"{space_x}~{space_y}",
                 "metric": "knn_label_agreement", "variant": "cross",
                 "value": cross.knn_label_agreement(X, Y, lab), "ci_lo": np.nan,
                 "ci_hi": np.nan, "n": X.shape[0], "d": f"{X.shape[1]}x{Y.shape[1]}",
                 "boot": "none"})
    return pd.DataFrame(rows)
