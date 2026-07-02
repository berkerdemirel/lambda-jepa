"""M1 results gallery: the numbers as pictures. Reads the feature store + results CSVs and writes
PNGs to results/figures/ (+ one wandb run, project sslgap, so the gallery is browsable live).
Numbers-only discipline applies to figures too: panels show measurements and nulls, no verdicts.

  sbatch slurm/viz.sbatch                          # everything available
  sbatch slurm/viz.sbatch 'panels=[pca,relrep]'    # subset

Palette: dataviz-validated. Methods = 7 reference categorical slots (fixed canonical order);
classes = 10-slot extension (worst adjacent CVD dE 24.2, PASS); sub-3:1 hues get direct labels.
"""
import os

import hydra
import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from omegaconf import DictConfig

from sslgap.data import _Source, read_manifest
from sslgap.extract import FeatureStore
from sslgap.metrics.relrep import anchor_indices, relrep_dataset

METHODS = ["simclr", "byol", "vicreg", "dino", "mae", "ijepa", "lejepa"]
MCOLOR = dict(zip(METHODS, ["#2a78d6", "#1baf7a", "#eda100", "#008300",
                            "#4a3aa7", "#e34948", "#e87ba4"]))
C10 = ["#2a78d6", "#1baf7a", "#eda100", "#008300", "#4a3aa7",
       "#e34948", "#e87ba4", "#eb6834", "#00a0c8", "#7a9b00"]
CLASSES = ["tench", "springer", "cassette", "chainsaw", "church",
           "horn", "truck", "gas pump", "golf", "chute"]      # frgfm/imagenette label order
INK, INK2, MUTED, GRID, SURF = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#fcfcfb"

# PROTOCOL §3 h / z.final (mirrors adapters._NATIVE_ASM; pinned by adapter_selftest EXPECT_H).
H = {"simclr": "student.h.gap", "byol": "student.h.gap", "vicreg": "student.h.gap",
     "dino": "teacher.h.cls", "mae": "student.h.gap", "ijepa": "teacher.h.gap",
     "lejepa": "student.z.embed"}
Z = {"simclr": "student.z.proj.out", "byol": "student.z.pred.out",
     "vicreg": "student.z.proj.out", "dino": "teacher.z.dino.bottleneck",
     "mae": "student.z.dec.tap8",                  # z proxy: deepest decoder tap (PROTOCOL: z = —)
     "ijepa": "student.z.pred.out", "lejepa": "student.z.proj.out"}
DEPTH = {  # guillotine axis per method: trunk taps (paper feature type) -> head taps in order
    "simclr": ["h.gap.L03", "h.gap.L06", "h.gap.L09", "h.gap.L12", "z.proj.tap1", "z.proj.out"],
    "byol": ["h.gap.L03", "h.gap.L06", "h.gap.L09", "h.gap.L12",
             "z.proj.tap1", "z.proj.out", "z.pred.tap1", "z.pred.out"],
    "vicreg": ["h.gap.L03", "h.gap.L06", "h.gap.L09", "h.gap.L12",
               "z.proj.tap1", "z.proj.tap2", "z.proj.out"],
    "dino": ["h.cls.L03", "h.cls.L06", "h.cls.L09", "h.cls.L12",
             "z.dino.tap1", "z.dino.tap2", "z.dino.bottleneck"],
    "mae": ["h.gap.L03", "h.gap.L06", "h.gap.L09", "h.gap.L12",
            "z.dec.tap2", "z.dec.tap5", "z.dec.tap8"],
    "ijepa": ["h.gap.L03", "h.gap.L06", "h.gap.L09", "h.gap.L12", "z.pred.out"],
    "lejepa": ["h.cls.L03", "h.cls.L06", "h.cls.L09", "h.cls.L12",
               "z.embed", "z.proj.tap1", "z.proj.tap2", "z.proj.out"],
}


def _style(ax, title=None):
    ax.set_facecolor(SURF)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
    ax.tick_params(colors=MUTED, labelsize=7)
    ax.grid(True, color=GRID, linewidth=0.5, alpha=0.7)
    if title:
        ax.set_title(title, color=INK, fontsize=9)


def _save(fig, out_dir, name, wb):
    """name may carry a subdir ("geometry/pca.png") — wandb keys keep the slash (section per dir)."""
    path = os.path.join(out_dir, name)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig.savefig(path, dpi=200, bbox_inches="tight", facecolor=SURF)
    plt.close(fig)
    if wb:
        import wandb
        wb.log({name.removesuffix(".png"): wandb.Image(path)})
    print(f"[viz] wrote {name}")
    return name


def _branch_space(method, tap):
    """DEPTH entry -> stored space name (branch prefix per PROTOCOL §3)."""
    br = "teacher" if (method in ("dino", "ijepa") and not tap.startswith("z.pred")) else "student"
    if method == "dino" and tap.startswith("z."):
        br = "teacher"
    return f"{br}.{tap}"


def _feats(store, rid, man, space, n=None, seed=0):
    X = np.asarray(store.get(rid, man, space), dtype=np.float32)
    if n and len(X) > n:
        idx = np.random.default_rng(seed).choice(len(X), n, replace=False)
        return X[idx], idx
    return X, np.arange(len(X))


def _lda2(X, y, reg=1e-4):
    """Two-component Fisher LDA (numpy; supervised — for class-geometry viz only)."""
    mu = X.mean(0)
    Sw = np.zeros((X.shape[1], X.shape[1]))
    Sb = np.zeros_like(Sw)
    for k in np.unique(y):
        Xk = X[y == k]
        d = Xk - Xk.mean(0)
        Sw += d.T @ d
        m = (Xk.mean(0) - mu)[:, None]
        Sb += len(Xk) * (m @ m.T)
    Sw += reg * np.trace(Sw) / len(Sw) * np.eye(len(Sw))
    evals, evecs = np.linalg.eig(np.linalg.solve(Sw, Sb))
    order = np.argsort(-evals.real)[:2]
    return evecs[:, order].real                                    # [d, 2] projection basis


def _class_scatter(ax, P, yy, label_classes=True):
    for k in range(10):
        sel = yy == k
        ax.scatter(P[sel, 0], P[sel, 1], s=4, c=C10[k], alpha=0.6, linewidths=0)
    if label_classes:
        for k in range(10):
            mu = P[yy == k].mean(0)
            ax.annotate(CLASSES[k], mu, fontsize=6, color=INK, ha="center", va="center",
                        bbox=dict(boxstyle="round,pad=0.12", fc="white", ec="none", alpha=0.75))
    ax.set_xticks([]), ax.set_yticks([])


def _class_legend(fig):
    fig.legend(handles=[plt.Line2D([], [], marker="o", ls="", color=C10[k], label=CLASSES[k])
                        for k in range(10)],
               loc="lower center", ncol=10, frameon=False, fontsize=7)


def _knn_idx(Xg, Xq, k=20, bs=512):
    """cosine top-k gallery indices per query (numpy, blocked)."""
    Gn = Xg / (np.linalg.norm(Xg, axis=1, keepdims=True) + 1e-12)
    Qn = Xq / (np.linalg.norm(Xq, axis=1, keepdims=True) + 1e-12)
    out = np.empty((len(Qn), k), dtype=np.int64)
    for i in range(0, len(Qn), bs):
        sim = Qn[i:i + bs] @ Gn.T
        out[i:i + bs] = np.argpartition(-sim, k, axis=1)[:, :k]
    return out


# ---- panels ------------------------------------------------------------------------------------

def panel_guillotine(cfg, run_ids, res, out, wb):
    fig, axes = plt.subplots(2, 4, figsize=(15, 6.5), sharey=True, layout="constrained")
    for ax, m in zip(axes.flat, METHODS):
        pb = pd.read_csv(os.path.join(res, "probes", f"{run_ids[m]}.csv"))
        pv = pb.pivot_table(index="space", columns="probe", values="val_acc")
        xs = [_branch_space(m, t) for t in DEPTH[m]]
        xs = [s for s in xs if s in pv.index]
        for probe, color, ls in (("linear_raw_v1", "#2a78d6", "-"), ("knn_v1_k200", "#e34948", "-")):
            ax.plot(range(len(xs)), [pv.loc[s, probe] for s in xs], ls, color=color,
                    linewidth=2, marker="o", markersize=4)
        ax.axvline(3.5, color=MUTED, linewidth=0.8, linestyle=":")     # trunk | head boundary
        ax.set_xticks(range(len(xs)))
        ax.set_xticklabels([t.split(".", 1)[1] if t.startswith("h.") else t for t in DEPTH[m]
                            if _branch_space(m, t) in pv.index], rotation=60, ha="right", fontsize=6)
        _style(ax, m)
    axes[1, 3].axis("off")
    axes[0, 0].set_ylabel("val acc", color=INK2, fontsize=8)
    fig.legend(handles=[plt.Line2D([], [], color="#2a78d6", lw=2, label="linear_raw_v1"),
                        plt.Line2D([], [], color="#e34948", lw=2, label="knn_v1_k200")],
               loc="lower right", frameon=False, fontsize=9)
    fig.suptitle("Guillotine curves: probes along trunk (L03–L12) → head taps (dotted line = trunk/head boundary)",
                 color=INK, fontsize=11)
    return _save(fig, out, "probes/probe_guillotine.png", wb)


def panel_pca(cfg, run_ids, store, man_val, out, wb):
    names = []
    for kind in ("pca", "lda"):
        fig, axes = plt.subplots(7, 2, figsize=(8, 24))
        for r, m in enumerate(METHODS):
            y = store.labels(run_ids[m], man_val)
            for c, (space, tag) in enumerate([(H[m], "h"),
                                              (Z[m], "z.final" if m != "mae" else "z proxy (dec.tap8)")]):
                ax = axes[r, c]
                X, idx = _feats(store, run_ids[m], man_val, space, n=cfg.pca_points)
                yy = y[idx]
                Xc = X - X.mean(0)
                if kind == "pca":
                    _, _, Vt = np.linalg.svd(Xc[:2000], full_matrices=False)
                    P = Xc @ Vt[:2].T
                else:
                    P = Xc @ _lda2(Xc, yy)
                _class_scatter(ax, P, yy)
                _style(ax, f"{m} · {tag} ({space}, d={X.shape[1]})")
        title = ("PCA(2) of val features, colored by class — paper-h vs loss space (unsupervised)"
                 if kind == "pca" else
                 "LDA(2) of val features — CLASS-SUPERVISED projection: class geometry, not intrinsic structure")
        fig.suptitle(title, color=INK, fontsize=11, y=1.001)
        fig.tight_layout()
        names.append(_save(fig, out, f"geometry/{kind}_h_vs_z.png", wb))
    return names


def panel_relrep(cfg, run_ids, store, man_val, out, wb):
    """All methods' h in ONE frame (D-009 dataset anchors, shared ids) -> shared PCA(2)."""
    Rs, ys = {}, None
    n_val = None
    for m in METHODS:
        X, _ = _feats(store, run_ids[m], man_val, H[m])
        n_val = len(X)
        idx = anchor_indices(n_val, cfg.relrep_anchors, seed=0)
        Rs[m] = relrep_dataset(X, idx)
        ys = store.labels(run_ids[m], man_val)
    # per-method centering removes each model's global anchor-similarity offset (otherwise one
    # shared "mean similarity" direction dominates PC1 and every panel is the same banana).
    Rs = {m: R - R.mean(0) for m, R in Rs.items()}
    allR = np.concatenate([Rs[m] for m in METHODS])
    names = []
    for kind in ("pca", "lda"):
        if kind == "pca":
            sub = allR[np.random.default_rng(0).choice(len(allR), 4000, replace=False)]
            _, _, Vt = np.linalg.svd(sub - sub.mean(0), full_matrices=False)
            W = Vt[:2].T
        else:
            W = _lda2(allR, np.tile(ys, len(METHODS)))
        proj = lambda R: R @ W
        fig, axes = plt.subplots(2, 4, figsize=(15, 8), layout="constrained")
        for ax, m in zip(axes.flat, METHODS):
            _class_scatter(ax, proj(Rs[m]), ys, label_classes=(kind == "lda"))
            _style(ax, f"{m} · relrep(h)")
        axes[1, 3].axis("off")
        _class_legend(fig)
        fig.suptitle(f"Relative representations of h (A={cfg.relrep_anchors} shared anchors, cosine, "
                     "per-method centered; D-009) — one common frame, shared "
                     + ("PCA(2) axes (unsupervised)" if kind == "pca"
                        else "LDA(2) axes (CLASS-SUPERVISED — class geometry only)"),
                     color=INK, fontsize=11)
        names.append(_save(fig, out, f"crossmodel/relrep_h_shared_{kind}.png", wb))
    return names


def panel_spectra(cfg, run_ids, null_ids, store, man_tr, out, wb):
    fig, axes = plt.subplots(2, 4, figsize=(15, 6.5), sharey=True, layout="constrained")
    for ax, m in zip(axes.flat, METHODS):
        for space, color, ls, lab in ((H[m], "#2a78d6", "-", "h"), (Z[m], "#e34948", "-", "z")):
            X, _ = _feats(store, run_ids[m], man_tr, space, n=8000)
            eig = np.linalg.svd((X - X.mean(0)) / np.sqrt(len(X)), compute_uv=False) ** 2
            ax.loglog(np.arange(1, len(eig) + 1), eig / eig[0], ls, color=color, lw=2)
        if null_ids.get(m):
            Xn, _ = _feats(store, null_ids[m], man_tr, H[m], n=8000)
            eig = np.linalg.svd((Xn - Xn.mean(0)) / np.sqrt(len(Xn)), compute_uv=False) ** 2
            ax.loglog(np.arange(1, len(eig) + 1), eig / eig[0], "--", color=MUTED, lw=1.5)
        _style(ax, m)
    axes[1, 3].axis("off")
    fig.legend(handles=[plt.Line2D([], [], color="#2a78d6", lw=2, label="h"),
                        plt.Line2D([], [], color="#e34948", lw=2, label="z.final"),
                        plt.Line2D([], [], color=MUTED, lw=1.5, ls="--", label="h, random-init null")],
               loc="lower right", frameon=False, fontsize=9)
    fig.suptitle("Covariance eigenspectra (normalized to top eig, log–log)", color=INK, fontsize=11)
    return _save(fig, out, "geometry/spectra_h_vs_z.png", wb)


def panel_probe_bars(cfg, run_ids, null_ids, res, out, wb):
    fig, axes = plt.subplots(1, 2, figsize=(13, 4), sharey=True)
    for ax, probe in zip(axes, ("linear_raw_v1", "knn_v1_k200")):
        xs = np.arange(len(METHODS))
        for off, (SP, color, lab) in enumerate([(H, "#2a78d6", "h"), (Z, "#e34948", "z.final")]):
            vals, nulls = [], []
            for m in METHODS:
                pb = pd.read_csv(os.path.join(res, "probes", f"{run_ids[m]}.csv"))
                r = pb[(pb.space == SP[m]) & (pb.probe == probe)]
                vals.append(r.val_acc.iloc[0] if len(r) else np.nan)
                if null_ids.get(m) and os.path.exists(os.path.join(res, "probes", f"{null_ids[m]}.csv")):
                    nb = pd.read_csv(os.path.join(res, "probes", f"{null_ids[m]}.csv"))
                    nr = nb[(nb.space == SP[m]) & (nb.probe == probe)]
                    nulls.append(nr.val_acc.iloc[0] if len(nr) else np.nan)
                else:
                    nulls.append(np.nan)
            b = ax.bar(xs + (off - 0.5) * 0.38, vals, width=0.34, color=color, label=lab)
            for rect, v in zip(b, vals):                           # selective direct labels
                ax.annotate(f"{v:.2f}", (rect.get_x() + rect.get_width() / 2, v), fontsize=6,
                            ha="center", va="bottom", color=INK2)
            ax.plot(xs + (off - 0.5) * 0.38, nulls, "_", color=INK, markersize=12, mew=1.5)
        ax.set_xticks(xs)
        ax.set_xticklabels(METHODS, fontsize=8)
        _style(ax, probe)
    axes[0].set_ylabel("val acc", color=INK2, fontsize=8)
    axes[0].legend(frameon=False, fontsize=9, loc="upper left")
    fig.suptitle("Headline probes at h vs z.final (black tick = random-init null)", color=INK, fontsize=11)
    return _save(fig, out, "probes/probe_bars.png", wb)


def panel_confusion(cfg, run_ids, store, man_tr, man_val, out, wb):
    from matplotlib.colors import LinearSegmentedColormap
    ramp = LinearSegmentedColormap.from_list("blue_seq",
        ["#fcfcfb", "#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"])
    fig, axes = plt.subplots(2, 7, figsize=(16, 5))
    for c, m in enumerate(METHODS):
        ytr, yva = store.labels(run_ids[m], man_tr), store.labels(run_ids[m], man_val)
        for r, SP in enumerate((H, Z)):
            Xg, _ = _feats(store, run_ids[m], man_tr, SP[m])
            Xq, _ = _feats(store, run_ids[m], man_val, SP[m])
            nn = _knn_idx(Xg, Xq, k=20)
            pred = np.array([np.bincount(ytr[row], minlength=10).argmax() for row in nn])
            Cm = np.zeros((10, 10))
            for t, p in zip(yva, pred):
                Cm[t, p] += 1
            Cm /= Cm.sum(1, keepdims=True)
            ax = axes[r, c]
            ax.imshow(Cm, cmap=ramp, vmin=0, vmax=1)
            ax.set_xticks([]), ax.set_yticks([])
            if c == 0:
                ax.set_ylabel(["h", "z"][r], color=INK, fontsize=10)
            if r == 0:
                ax.set_title(m, color=INK, fontsize=9)
    fig.suptitle("kNN(20) class confusion (rows = true, row-normalized) at h vs z — class order: "
                 + ", ".join(CLASSES), color=INK, fontsize=10)
    return _save(fig, out, "geometry/knn_confusion.png", wb)


def panel_jaccard(cfg, run_ids, store, man_val, out, wb):
    fig, ax = plt.subplots(figsize=(8, 4))
    data = []
    for m in METHODS:
        Xh, _ = _feats(store, run_ids[m], man_val, H[m])
        Xz, _ = _feats(store, run_ids[m], man_val, Z[m])
        nh, nz = _knn_idx(Xh, Xh, k=21), _knn_idx(Xz, Xz, k=21)    # self included; drop later
        jac = np.array([len(set(a[1:]) & set(b[1:])) / len(set(a[1:]) | set(b[1:]))
                        for a, b in zip(nh, nz)])
        data.append(jac)
    parts = ax.violinplot(data, positions=range(len(METHODS)), showmedians=True, widths=0.7)
    for pc, m in zip(parts["bodies"], METHODS):
        pc.set_facecolor(MCOLOR[m]), pc.set_alpha(0.7)
    for k in ("cmedians", "cmins", "cmaxes", "cbars"):
        parts[k].set_color(INK2), parts[k].set_linewidth(1)
    ax.set_xticks(range(len(METHODS)))
    ax.set_xticklabels(METHODS, fontsize=9)
    _style(ax, "Per-image neighborhood overlap between h and z (Jaccard@20, val) — what the head reorders (OP-7)")
    ax.set_ylabel("Jaccard@20(h, z)", color=INK2, fontsize=8)
    return _save(fig, out, "geometry/neighborhood_jaccard_h_z.png", wb)


def panel_invariance(cfg, run_ids, store, out, wb):
    fig, axes = plt.subplots(2, 4, figsize=(15, 6.5), sharex=True, sharey=True, layout="constrained")
    for ax, m in zip(axes.flat, METHODS):
        man = [k for k in os.listdir(os.path.join(os.path.expanduser(cfg.store_root), run_ids[m]))
               if "@audit_v1" in k]
        if not man:
            ax.axis("off")
            continue
        man = man[0]
        for space, color, lab in ((H[m], "#2a78d6", "h"), (Z[m], "#e34948", "z")):
            try:
                A = np.asarray(store.get(run_ids[m], man, f"{space}.viewA"), dtype=np.float32)
                B = np.asarray(store.get(run_ids[m], man, f"{space}.viewB"), dtype=np.float32)
            except Exception:
                continue
            cos = (A * B).sum(1) / (np.linalg.norm(A, axis=1) * np.linalg.norm(B, axis=1) + 1e-12)
            xs = np.sort(cos)
            ax.plot(xs, np.linspace(0, 1, len(xs)), color=color, lw=2)
        _style(ax, m)
    axes[1, 3].axis("off")
    fig.legend(handles=[plt.Line2D([], [], color="#2a78d6", lw=2, label="h"),
                        plt.Line2D([], [], color="#e34948", lw=2, label="z.final")],
               loc="lower right", frameon=False, fontsize=9)
    fig.suptitle("View-invariance shape: ECDF of cos(view A, view B) under the FIXED audit stack "
                 "(right = more invariant)", color=INK, fontsize=11)
    return _save(fig, out, "geometry/invariance_ecdf.png", wb)


def panel_nn_gallery(cfg, run_ids, store, man_tr, out, wb):
    src = _Source("hf-imagenette", split="train")
    csv = os.path.join(os.path.expanduser(cfg.manifest_dir), man_tr + ".csv")
    items = read_manifest(csv)
    q_idx = np.random.default_rng(1).choice(len(items), cfg.gallery_queries, replace=False)
    names = []
    for m in METHODS:
        Xh, _ = _feats(store, run_ids[m], man_tr, H[m])
        Xz, _ = _feats(store, run_ids[m], man_tr, Z[m])
        fig, axes = plt.subplots(len(q_idx), 11, figsize=(11, len(q_idx) * 1.05))
        for r, qi in enumerate(q_idx):
            nnh = _knn_idx(Xh, Xh[qi:qi + 1], k=6)[0]
            nnz = _knn_idx(Xz, Xz[qi:qi + 1], k=6)[0]
            cols = [qi] + [j for j in nnh if j != qi][:5] + [j for j in nnz if j != qi][:5]
            for c, j in enumerate(cols):
                ax = axes[r, c]
                ax.imshow(src(items[j][0]).convert("RGB").resize((64, 64)))
                ax.set_xticks([]), ax.set_yticks([])
                for s in ax.spines.values():
                    s.set_color("#2a78d6" if 1 <= c <= 5 else "#e34948" if c > 5 else INK)
                    s.set_linewidth(2 if c == 0 else 1)
                if r == 0:
                    ax.set_title((["query"] + [f"h NN{i}" for i in range(1, 6)]
                                  + [f"z NN{i}" for i in range(1, 6)])[c], fontsize=6, color=INK2)
        fig.suptitle(f"{m}: 5 nearest train neighbors at h (blue) vs z (red), cosine", color=INK, fontsize=11)
        fig.tight_layout()
        names.append(_save(fig, out, f"galleries/nn_gallery_{m}.png", wb))
    return names


def panel_tau_heatmap(cfg, run_ids, res, out, wb):
    mets = ["uniformity", "variance_floor.hinge", "offdiag_redundancy.mean_abs_corr",
            "rankme", "kurt_topeig.worst", "epps_pulley"]
    M = np.full((len(METHODS), len(mets)), np.nan)
    for i, m in enumerate(METHODS):
        p = os.path.join(res, "battery", f"{run_ids[m]}.csv")
        if not os.path.exists(p) or m == "mae":     # PROTOCOL §3: MAE z = — (dec.tap8 is a viz
            continue                                # proxy for geometry panels, not for τ)
        bat = pd.read_csv(p)
        bat = bat[bat.variant == "raw|full"]
        for j, met in enumerate(mets):
            vh = bat[(bat.space == H[m]) & (bat.metric == met)].value
            vz = bat[(bat.space == Z[m]) & (bat.metric == met)].value
            if len(vh) and len(vz) and vz.iloc[0] != 0:
                M[i, j] = vh.iloc[0] / vz.iloc[0]
    from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
    div = LinearSegmentedColormap.from_list("div", ["#2a78d6", "#f0efec", "#e34948"])
    fig, ax = plt.subplots(figsize=(8, 4.5))
    with np.errstate(all="ignore"):
        im = ax.imshow(np.log10(np.abs(M)), cmap=div, norm=TwoSlopeNorm(0, -2, 2))
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            if M[i, j] == M[i, j]:
                ax.annotate(f"{M[i, j]:.2g}", (j, i), ha="center", va="center", fontsize=7, color=INK)
    ax.set_xticks(range(len(mets)))
    ax.set_xticklabels(mets, rotation=40, ha="right", fontsize=7)
    ax.set_yticks(range(len(METHODS)))
    ax.set_yticklabels(METHODS, fontsize=8)
    ax.set_title("τ = value(h) / value(z.final), raw|full (color = log10|τ|: blue τ≪1, gray τ≈1, red τ≫1; "
                 "MAE row empty: z = —)", color=INK, fontsize=9)
    fig.colorbar(im, ax=ax, shrink=0.8).ax.tick_params(labelsize=7, colors=MUTED)
    return _save(fig, out, "matrix/tau_heatmap.png", wb)


def panel_orbits(cfg, out, wb):
    """Aug-orbit clouds: SAME Q images x K audit-stack draws for every method (extract_orbits.py).
    Color = image identity (not class); X = orbit centroid. Watch orbits contract h -> z."""
    orb_dir = os.path.expanduser(cfg.orbits_dir)
    fig, axes = plt.subplots(7, 2, figsize=(9, 26))
    for r, m in enumerate(METHODS):
        f = os.path.join(orb_dir, f"toy.{m}.s0.npz")
        if not os.path.exists(f):
            axes[r, 0].axis("off"), axes[r, 1].axis("off")
            continue
        d = np.load(f, allow_pickle=True)
        ids = d["img_id"]
        uids = list(dict.fromkeys(ids.tolist()))
        for c, key in enumerate(("h", "z")):
            X = d[key].astype(np.float32)
            Xc = X - X.mean(0)
            _, _, Vt = np.linalg.svd(Xc, full_matrices=False)
            P = Xc @ Vt[:2].T
            ax = axes[r, c]
            for j, u in enumerate(uids):
                sel = ids == u
                ax.scatter(P[sel, 0], P[sel, 1], s=10, c=C10[j], alpha=0.6, linewidths=0)
                mu = P[sel].mean(0)
                ax.scatter(*mu, marker="X", s=60, c=C10[j], edgecolors="white", linewidths=0.8)
                ax.annotate(CLASSES[int(d["y"][sel][0])], mu, fontsize=6, color=INK2,
                            xytext=(4, 4), textcoords="offset points")
            ax.set_xticks([]), ax.set_yticks([])
            _style(ax, f"{m} · {str(d['h_space']) if key == 'h' else str(d['z_space'])}")
    fig.suptitle("Augmentation orbits: 8 images x 24 audit-stack draws (color = image, X = orbit "
                 "centroid) — h (left) vs z.final (right)", color=INK, fontsize=11, y=1.001)
    fig.tight_layout()
    return _save(fig, out, "orbits/orbit_clouds.png", wb)


def panel_crossmodel(cfg, run_ids, store, man_tr, man_val, out, wb):
    """Cross-model comparisons IN the shared relrep coordinates (D-009: same anchor images ->
    coordinate-wise comparable spaces). No affinity proxies: direct per-image agreement,
    cross-model retrieval, and probe transfer."""
    from sklearn.linear_model import LogisticRegression
    A = cfg.relrep_anchors
    Rtr, Rva, ys = {}, {}, {}
    for m in METHODS:
        Xtr, _ = _feats(store, run_ids[m], man_tr, H[m])
        Xva, _ = _feats(store, run_ids[m], man_val, H[m])
        idx = anchor_indices(len(Xtr), A, seed=0)
        anc = Xtr[idx] / (np.linalg.norm(Xtr[idx], axis=1, keepdims=True) + 1e-12)
        Rtr[m] = (Xtr / (np.linalg.norm(Xtr, axis=1, keepdims=True) + 1e-12)) @ anc.T
        Rva[m] = (Xva / (np.linalg.norm(Xva, axis=1, keepdims=True) + 1e-12)) @ anc.T
        ys["tr"], ys["va"] = store.labels(run_ids[m], man_tr), store.labels(run_ids[m], man_val)
    n = len(METHODS)
    agree, retr, xfer = np.eye(n), np.eye(n), np.zeros((n, n))
    probes = {m: LogisticRegression(max_iter=500).fit(Rtr[m], ys["tr"]) for m in METHODS}
    for i, a in enumerate(METHODS):
        Ra = Rva[a] / (np.linalg.norm(Rva[a], axis=1, keepdims=True) + 1e-12)
        for j, b in enumerate(METHODS):
            Rb = Rva[b] / (np.linalg.norm(Rva[b], axis=1, keepdims=True) + 1e-12)
            if i != j:
                agree[i, j] = float((Ra * Rb).sum(1).mean())
                top1 = np.empty(len(Ra), dtype=np.int64)
                for s in range(0, len(Ra), 512):
                    top1[s:s + 512] = (Ra[s:s + 512] @ Rb.T).argmax(1)
                retr[i, j] = float((top1 == np.arange(len(Ra))).mean())
            xfer[i, j] = probes[a].score(Rva[b], ys["va"])
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.6), layout="constrained")
    from matplotlib.colors import LinearSegmentedColormap
    ramp = LinearSegmentedColormap.from_list("blue_seq",
        ["#fcfcfb", "#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"])
    for ax, M, title in ((axes[0], agree, "per-image agreement\nmean cos(relrep_A(x), relrep_B(x))"),
                         (axes[1], retr, "cross-model self-retrieval@1\nquery A -> gallery B (val)"),
                         (axes[2], xfer, "probe transfer acc\nlinear probe fit on A, eval on B")):
        im = ax.imshow(M, cmap=ramp, vmin=max(0.0, M.min() - 0.05), vmax=1.0)
        for i in range(n):
            for j in range(n):
                ax.annotate(f"{M[i, j]:.2f}", (j, i), ha="center", va="center", fontsize=7,
                            color="white" if M[i, j] > (M.max() + M.min()) / 2 else INK)
        ax.set_xticks(range(n)), ax.set_yticks(range(n))
        ax.set_xticklabels(METHODS, rotation=45, ha="right", fontsize=7)
        ax.set_yticklabels(METHODS, fontsize=7)
        ax.set_title(title, color=INK, fontsize=9)
    fig.suptitle(f"Direct cross-model comparison in shared relrep coordinates "
                 f"(A={A} anchor images, cosine, latentis-default no centering; rows = source A, "
                 "cols = target B)", color=INK, fontsize=11)
    return _save(fig, out, "crossmodel/relrep_direct_7x7.png", wb)


def panel_tsne(cfg, run_ids, store, man_val, out, wb):
    from sklearn.manifold import TSNE
    names = []
    for perp in cfg.tsne_perplexities:
        fig, axes = plt.subplots(2, 4, figsize=(15, 8), layout="constrained")
        for ax, m in zip(axes.flat, METHODS):
            X, idx = _feats(store, run_ids[m], man_val, H[m], n=cfg.tsne_points)
            yy = store.labels(run_ids[m], man_val)[idx]
            P = TSNE(2, perplexity=perp, init="pca", random_state=0).fit_transform(X)
            _class_scatter(ax, P, yy, label_classes=False)
            _style(ax, f"{m} · h")
        axes[1, 3].axis("off")
        _class_legend(fig)
        fig.suptitle(f"t-SNE of h (perplexity={perp}) — read NEIGHBORHOOD TOPOLOGY only: distances, "
                     "cluster sizes and inter-cluster gaps are not meaningful", color=INK, fontsize=11)
        names.append(_save(fig, out, f"geometry/tsne_h_perp{perp}.png", wb))
    return names


@hydra.main(version_base=None, config_path="configs", config_name="viz")
def main(cfg: DictConfig):
    store = FeatureStore(cfg.store_root)
    res = os.path.expanduser(cfg.results_root)
    out = os.path.join(res, "figures")
    os.makedirs(out, exist_ok=True)
    overrides = cfg.get("run_id_overrides") or {}
    run_ids = {m: overrides.get(m, f"toy.{m}.s0.ext") for m in METHODS}
    null_ids = {m: run_ids[m].removesuffix(".ext") + ".null.ext" for m in METHODS}
    man_tr, man_val = cfg.train_manifest, cfg.val_manifest

    wb = None
    if cfg.wandb:
        import wandb
        wb = wandb.init(entity="causal-learning-ai-ista", project="sslgap", name=cfg.wandb_name,
                        config={"panels": list(cfg.panels), "run_ids": run_ids})
    made = []
    P = set(cfg.panels)
    if "guillotine" in P:
        made.append(panel_guillotine(cfg, run_ids, res, out, wb))
    if "pca" in P:
        made += panel_pca(cfg, run_ids, store, man_val, out, wb)
    if "relrep" in P:
        made += panel_relrep(cfg, run_ids, store, man_val, out, wb)
    if "spectra" in P:
        made.append(panel_spectra(cfg, run_ids, null_ids, store, man_tr, out, wb))
    if "bars" in P:
        made.append(panel_probe_bars(cfg, run_ids, null_ids, res, out, wb))
    if "confusion" in P:
        made.append(panel_confusion(cfg, run_ids, store, man_tr, man_val, out, wb))
    if "jaccard" in P:
        made.append(panel_jaccard(cfg, run_ids, store, man_val, out, wb))
    if "invariance" in P:
        made.append(panel_invariance(cfg, run_ids, store, out, wb))
    if "gallery" in P:
        made += panel_nn_gallery(cfg, run_ids, store, man_tr, out, wb)
    if "tau" in P:
        made.append(panel_tau_heatmap(cfg, run_ids, res, out, wb))
    if "orbits" in P:
        made.append(panel_orbits(cfg, out, wb))
    if "crossmodel" in P:
        made.append(panel_crossmodel(cfg, run_ids, store, man_tr, man_val, out, wb))
    if "tsne" in P:
        made += panel_tsne(cfg, run_ids, store, man_val, out, wb)
    with open(os.path.join(out, "INDEX.md"), "w") as f:
        f.write("# M1 figures (generated by experiments/viz.py — numbers only)\n\n"
                + "\n".join(f"- {n}" for n in sorted(made)) + "\n")
    if wb:
        wb.finish()
    print(f"[viz] done: {len(made)} figures -> {out}")


if __name__ == "__main__":
    main()
