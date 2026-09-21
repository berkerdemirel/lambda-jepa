"""E36: does the treated model's extra content live in the part the invariance term deletes?

E35 closed against the brief's §6 obstacle — the invariance term "has no preference that anything
can violate". That is an artefact of localising to DIRECTIONS. It dissolves under a decomposition
into COMPONENTS, via an identity the objective supplies: (1/V^2) sum_vw ||z_v - z_w||^2 =
(2/V) sum_v ||z_v - z_bar||^2. `inv` IS the squared norm of the view residual, so the term has a
concrete violable target, and "does that component carry class information?" is answerable.

The cross-model half (Berker 2026-09-10): in a whitened shared-anchor frame the null map between
two models is the IDENTITY, so nothing is fitted — which is what sank the brief's §5(e) head fit.

FRAME DISCIPLINE, the one thing this script will not let you vary: the affine null (D = 0 under an
invertible-affine recoding) is exact ONLY at alpha = 0 and full numerical rank. Shrinkage and rank
truncation both break whitening's equivariance — measured, sham max|D| .85 at alpha = 1e-3 and .97
at 1e-2 on a scale where the whole signal lives in [-1, 1]. So the residual stages run at that one
corner and every number is read against the SHAM arm at the same setting, never against zero.
Sweeping alpha stays correct for the READER stage, which needs no cross-model cancellation.

  sbatch slurm/e36_relres.sbatch
  sbatch slurm/e36_relres.sbatch 'stages=[preflight]'
  sbatch slurm/e36_relres.sbatch 'stages=[views]' n_images=8192 'lam_grid=[1e-4]'
"""
import csv
import json

import hydra
import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf

from sslgap.extract import FeatureStore
from sslgap.metrics.pairs import class_margin
from sslgap.metrics.relrep import anchor_indices
from sslgap.metrics.spectra import (apply_whiten, effective_rank, participation_ratio,
                                    random_linear_map, whiten_frame)
from sslgap.paths import DIAG
from sslgap.probes import knn_topk_acc, lda_shrunk_v1, linear_lbfgs_v1, linear_raw_v2

DEV = "cuda" if torch.cuda.is_available() else "cpu"


def write_csv(name, rows):
    if not rows:
        return
    DIAG.mkdir(parents=True, exist_ok=True)
    keys = list(dict.fromkeys(k for r in rows for k in r))
    with open(DIAG / name, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)
    print(f"  -> results/diag/{name}  ({len(rows)} rows)", flush=True)


def relspace(X, Xa, mu, W):
    """Cosine of X's rows to the anchor rows, both carried through the SAME whitening frame.
    This is where the affine map cancels: at alpha = 0 / full rank the two cells' whitened clouds
    differ by an orthogonal transform, which a cosine does not see."""
    Z = torch.as_tensor(apply_whiten(X, mu, W), dtype=torch.float32, device=DEV)
    A = torch.as_tensor(apply_whiten(Xa, mu, W), dtype=torch.float32, device=DEV)
    Z = Z / (Z.norm(dim=1, keepdim=True) + 1e-12)
    A = A / (A.norm(dim=1, keepdim=True) + 1e-12)
    return Z @ A.T


def strat_split(y, fracs, seed=0):
    """Class-stratified split of image indices into len(fracs) parts."""
    rng = np.random.default_rng(seed)
    parts = [[] for _ in fracs]
    for c in np.unique(y):
        idx = np.flatnonzero(y == c)
        rng.shuffle(idx)
        cuts = np.cumsum([int(round(f * len(idx))) for f in fracs[:-1]])
        for p, chunk in zip(parts, np.split(idx, cuts)):
            p.append(chunk)
    return [np.sort(np.concatenate(p)) for p in parts]


def best_lam(Xf, yf, Xs, ys, Xt, yt, ncls, lams, tag):
    """Sweep the ridge constant on the SELECT split, report the winner on the untouched TEST split.
    Selection never touches the reported split — the stored protocol selects its epoch on the same
    5k it reports, which with a sweep on top would bias by several times the effect size."""
    sel = [(linear_lbfgs_v1(Xf, yf, Xs, ys, ncls, lam=l, device=DEV), l) for l in lams]
    r, lam = max(sel, key=lambda t: t[0]["val_acc"])
    out = linear_lbfgs_v1(Xf, yf, Xt, yt, ncls, lam=lam, device=DEV)
    print(f"    {tag:28s} lam={lam:<8g} sel={r['val_acc']:.4f} test={out['val_acc']:.4f} "
          f"train={out['train_acc']:.4f} |g|={out['grad_norm']:.2e}", flush=True)
    return out, lam, r["val_acc"]


def best_lam_blocks(Xf, yf, Xs, ys, Xt, yt, ncls, lams, split, tag):
    """Two-block ridge sweep: the block boundary is `split`, and lam is swept INDEPENDENTLY per
    block so the concatenation nests each block's own model (sending one block's lam up removes
    it). Without this the increment can come out negative and mean nothing."""
    best = None
    for l1 in lams:
        for l2 in lams:
            v = np.concatenate([np.full(split, l1), np.full(Xf.shape[1] - split, l2)])
            r = linear_lbfgs_v1(Xf, yf, Xs, ys, ncls, lam=v, device=DEV)
            if best is None or r["val_acc"] > best[0]:
                best = (r["val_acc"], l1, l2, v)
    s_, l1, l2, v = best
    out = linear_lbfgs_v1(Xf, yf, Xt, yt, ncls, lam=v, device=DEV)
    print(f"    {tag:28s} lam=({l1:g},{l2:g}) sel={s_:.4f} test={out['val_acc']:.4f} "
          f"train={out['train_acc']:.4f}", flush=True)
    return out, f"({l1:g},{l2:g})", s_


def mcnemar(pa, pb, y):
    """Discordant counts + two-sided p on the PAIRED predictions. The two accuracies are not
    independent draws — 1.78 points on 5k images is 89 images, and which images they are is the
    part worth reporting."""
    from math import comb, erfc, sqrt
    a, b = pa == y, pb == y
    n01, n10 = int((~a & b).sum()), int((a & ~b).sum())
    n = n01 + n10
    if n == 0:
        return n01, n10, 1.0
    if n <= 60:                                        # exact sign test
        p = min(1.0, 2.0 * sum(comb(n, i) for i in range(min(n01, n10) + 1)) / 2.0 ** n)
    else:                                              # continuity-corrected normal
        p = erfc(max(abs(n01 - n10) - 1.0, 0.0) / sqrt(2.0 * n))
    return n01, n10, p


def classpair(D, ycls, yanc, ncls):
    """[ncls, ncls] mean of D over (class(row), class(anchor)) cells. Exactly 0 under the affine
    null, so its diagonal-vs-offdiagonal contrast is a SIGNED read: two same-recipe models have the
    same expected class margin, so a seed difference is centred on zero here while a treatment that
    adds class structure is not."""
    D = D.cpu().numpy() if torch.is_tensor(D) else D
    code = (ycls[:, None] * ncls + yanc[None, :]).ravel()
    s = np.bincount(code, weights=D.ravel().astype(np.float64), minlength=ncls * ncls)
    c = np.bincount(code, minlength=ncls * ncls)
    return (s / np.maximum(c, 1)).reshape(ncls, ncls), c.reshape(ncls, ncls)


# ---------------------------------------------------------------- stages
def stage_preflight(cfg, store):
    """Is the fp16 store good enough for an alpha = 0 frame? alpha = 0 divides by sqrt(lam_min),
    so storage noise in the tail is amplified into the measured quantity. Three residuals on one
    scale: the store's own floor, the pipeline's floor, and the signal."""
    Xu = np.asarray(store.get(cfg.cell_u, cfg.fit_manifest, cfg.space)).astype(np.float32)
    Xt = np.asarray(store.get(cfg.cell_t, cfg.fit_manifest, cfg.space)).astype(np.float32)
    n, d = Xu.shape
    idx = anchor_indices(n, cfg.anchors, seed=cfg.anchor_seed)
    rng = np.random.default_rng(7)
    fp16 = lambda X: (rng.uniform(-.5, .5, X.shape) * np.abs(X) * 2.0 ** -10).astype(np.float32)

    srows = []
    for nm, X in [("untreated", Xu), ("treated", Xt)]:
        lam = np.clip(np.linalg.eigvalsh(np.cov(X, rowvar=False))[::-1], 0, None)
        nz = {f"nr_{t:g}": int((lam > t * lam[0]).sum()) for t in (1e-6, 1e-8, 1e-10, 1e-12)}
        srows.append(dict(cell=nm, lam1=lam[0], lam_383=lam[382], lam_384=lam[383],
                          cond=lam[0] / max(lam[383], 1e-30), effrank=effective_rank(lam),
                          pr=participation_ratio(lam), **nz))
        print(f"  {nm:10s} lam1={lam[0]:.4g} lam383={lam[382]:.4g} lam384={lam[383]:.4g} "
              f"effrank={effective_rank(lam):.1f} {nz}", flush=True)
    write_csv("e36_preflight_spectra.csv", srows)

    def resid(Xa, Xb, alpha, rank):
        D = (relspace(Xa, Xa[idx], *whiten_frame(Xa, alpha, rank))
             - relspace(Xb, Xb[idx], *whiten_frame(Xb, alpha, rank)))
        return float(D.pow(2).mean().sqrt()), float(D.abs().max())

    rows = []
    arms = [("noise_U", Xu, Xu + fp16(Xu)), ("noise_T", Xt, Xt + fp16(Xt)),
            ("sham_U_c1e4", Xu, Xu @ random_linear_map(d, 1e4, seed=1).T),
            ("sham_U_c10", Xu, Xu @ random_linear_map(d, 1e1, seed=2).T),
            ("signal_UT", Xu, Xt)]
    for nm, Xa, Xb in arms:
        for r in cfg.rank_grid:
            rms, mx = resid(Xa, Xb, 0.0, r)
            rows.append(dict(arm=nm, alpha=0.0, rank=r, rms=rms, max=mx))
            print(f"  {nm:12s} a=0      r={r:3d}  rms={rms:.5f} max={mx:.5f}", flush=True)
        for a in cfg.alpha_probe_grid[1:]:
            rms, mx = resid(Xa, Xb, a, cfg.rank_grid[1])
            rows.append(dict(arm=nm, alpha=a, rank=cfg.rank_grid[1], rms=rms, max=mx))
            print(f"  {nm:12s} a={a:<6g} r={cfg.rank_grid[1]:3d}  rms={rms:.5f} max={mx:.5f}",
                  flush=True)
    write_csv("e36_preflight.csv", rows)


def stage_reader(cfg, store):
    """The 1.78 itself, under readers that conditioning cannot fool. Selection on an inner split
    cut from train; reported on the untouched val. The sham arm is the reader's own selftest: a
    conditioning-invariant reader must score the SAME on h and on G h."""
    ytr = np.asarray(store.labels(cfg.cell_u, cfg.fit_manifest))
    yva = np.asarray(store.labels(cfg.cell_u, cfg.val_manifest))
    fit, sel = strat_split(ytr, [0.9, 0.1], seed=cfg.split_seed)
    ncls, rows, preds = int(ytr.max()) + 1, [], {}
    G = random_linear_map(384, 1e4, seed=1)

    for arm, cell in [("U", cfg.cell_u), ("T", cfg.cell_t), ("sham_U", cfg.cell_u)]:
        Xtr = np.asarray(store.get(cell, cfg.fit_manifest, cfg.space)).astype(np.float32)
        Xva = np.asarray(store.get(cell, cfg.val_manifest, cfg.space)).astype(np.float32)
        if arm == "sham_U":
            Xtr, Xva = Xtr @ G.T, Xva @ G.T
        # the stored protocol, replicated for parity with results/probes/*.extL.csv
        if cfg.parity:
            p = linear_raw_v2(Xtr, ytr, Xva, yva, ncls, device=DEV)
            rows.append(dict(arm=arm, reader="linear_raw_v2_parity", alpha="", lam="",
                             test_acc=p["val_acc"], best_ep=p["best_ep"]))
            print(f"  {arm:7s} linear_raw_v2 (stored protocol) = {p['val_acc']:.4f} "
                  f"@ep{p['best_ep']}", flush=True)
        best = None
        for a in cfg.alpha_probe_grid:
            mu, W = whiten_frame(Xtr[fit], a, cfg.rank)
            Zf, Zs = apply_whiten(Xtr[fit], mu, W), apply_whiten(Xtr[sel], mu, W)
            Zv = apply_whiten(Xva, mu, W)
            out, lam, s = best_lam(Zf, ytr[fit], Zs, ytr[sel], Zv, yva, ncls, cfg.lam_grid,
                                   f"{arm} whiten a={a:g}")
            rows.append(dict(arm=arm, reader="linear_lbfgs_v1", alpha=a, lam=lam,
                             sel_acc=s, test_acc=out["val_acc"], train_acc=out["train_acc"]))
            if best is None or s > best[0]:
                best = (s, out["pred"], a, lam)
            if a == 0.0:
                preds[f"{arm}@a0"] = out["pred"]      # the frame where invariance is CLAIMED
        preds[arm] = best[1]
        rows.append(dict(arm=arm, reader="linear_lbfgs_v1_BEST", alpha=best[2], lam=best[3],
                         test_acc=float((best[1] == yva).mean())))
        for sh in cfg.lda_shrink_grid:
            r = lda_shrunk_v1(Xtr[fit], ytr[fit], Xva, yva, ncls, shrink=sh, device=DEV)
            rows.append(dict(arm=arm, reader="lda_shrunk_v1", alpha="", lam=sh,
                             test_acc=r["val_acc"]))
            print(f"    {arm} lda shrink={sh:<6g} test={r['val_acc']:.4f}", flush=True)
        del Xtr, Xva

    n01, n10, p = mcnemar(preds["U"], preds["T"], yva)
    # the selftest is only meaningful at matched alpha: each arm's own best alpha can differ, and
    # comparing U at one frame against sham at another measures the frames, not the reader
    n01s, n10s, ps = mcnemar(preds["U@a0"], preds["sham_U@a0"], yva)
    rows.append(dict(arm="U_vs_T", reader="mcnemar", test_acc="", n01=n01, n10=n10, p=p))
    rows.append(dict(arm="U_vs_sham", reader="mcnemar", test_acc="", n01=n01s, n10=n10s, p=ps))
    print(f"  McNemar U vs T: T fixes {n01}, T breaks {n10}, p={p:.3g}", flush=True)
    print(f"  SELFTEST reader invariance (U vs sham): {n01s}/{n10s} discordant, p={ps:.3g}",
          flush=True)
    write_csv("e36_reader.csv", rows)


def stage_rel(cfg, store):
    """The residual between the two cells in the shared anchor frame, read structurally."""
    ytr = np.asarray(store.labels(cfg.cell_u, cfg.fit_manifest))
    yva = np.asarray(store.labels(cfg.cell_u, cfg.val_manifest))
    ncls = int(ytr.max()) + 1
    aidx = anchor_indices(len(ytr), cfg.anchors, seed=cfg.anchor_seed)
    G = random_linear_map(384, 1e4, seed=1)

    def R(cell, sham=False):
        Xt = np.asarray(store.get(cell, cfg.fit_manifest, cfg.space)).astype(np.float32)
        Xv = np.asarray(store.get(cell, cfg.val_manifest, cfg.space)).astype(np.float32)
        if sham:
            Xt, Xv = Xt @ G.T, Xv @ G.T
        mu, W = whiten_frame(Xt, 0.0, cfg.rank)
        return relspace(Xt, Xt[aidx], mu, W), relspace(Xv, Xt[aidx], mu, W)

    (Ru_t, Ru_v), (Rt_t, Rt_v), (Rs_t, Rs_v) = R(cfg.cell_u), R(cfg.cell_t), R(cfg.cell_u, True)
    rows, mats = [], {}
    for nm, Dt, Dv in [("signal_UT", Rt_t - Ru_t, Rt_v - Ru_v),
                       ("sham_U", Rs_t - Ru_t, Rs_v - Ru_v)]:
        M, cnt = classpair(Dv, yva, ytr[aidx], ncls)
        dg, off = float(np.trace(M) / ncls), float((M.sum() - np.trace(M)) / (ncls * ncls - ncls))
        anc = Dv.abs().mean(0).cpu().numpy()
        rows.append(dict(arm=nm, rank=cfg.rank, rms=float(Dv.pow(2).mean().sqrt()),
                         max=float(Dv.abs().max()), cp_diag=dg, cp_off=off, cp_signed=dg - off,
                         anchor_rms_max=float(anc.max()), anchor_rms_med=float(np.median(anc))))
        mats[nm] = M
        print(f"  {nm:10s} rms={rows[-1]['rms']:.5f} class-pair diag={dg:+.5f} off={off:+.5f} "
              f"SIGNED={dg - off:+.5f}", flush=True)
        # probe(D): a CONSTRAINED sub-probe (D = [I,-I][R_U;R_T]) -- sufficient, not necessary
        fit, sel = strat_split(ytr, [0.9, 0.1], seed=cfg.split_seed)
        out, lam, s = best_lam(Dt[fit].cpu().numpy(), ytr[fit], Dt[sel].cpu().numpy(), ytr[sel],
                               Dv.cpu().numpy(), yva, ncls, cfg.lam_grid, f"probe(D) {nm}")
        rows[-1]["probe_D"] = out["val_acc"]
    for nm, Rv in [("U", Ru_v), ("T", Rt_v)]:
        M, _ = classpair(Rv, yva, ytr[aidx], ncls)
        rows.append(dict(arm=f"level_{nm}", rank=cfg.rank, cp_diag=float(np.trace(M) / ncls),
                         cp_off=float((M.sum() - np.trace(M)) / (ncls * ncls - ncls))))
        rows[-1]["cp_signed"] = rows[-1]["cp_diag"] - rows[-1]["cp_off"]
    write_csv("e36_rel.csv", rows)
    np.savez(DIAG / "e36_rel_classpair.npz", **mats)
    print(f"  -> results/diag/e36_rel_classpair.npz", flush=True)


def stage_views(cfg, store):
    """The decomposition. R_iv = R_bar_i + (R_iv - R_bar_i); probe arms hold the invariant part
    FIXED and toggle only the residual's availability, so the contrast is content, not the sqrt(V)
    denoising that 'centroid vs single view' would measure."""
    y_all = np.asarray(store.labels(cfg.cell_u, cfg.view_manifest))
    rng = np.random.default_rng(cfg.split_seed)
    keep = np.sort(rng.choice(len(y_all), size=min(cfg.n_images, len(y_all)), replace=False))
    y = y_all[keep]
    ncls = int(y.max()) + 1
    fit, sel, test = strat_split(y, [0.6, 0.2, 0.2], seed=cfg.split_seed)
    aidx = fit[anchor_indices(len(fit), cfg.anchors, seed=cfg.anchor_seed)]
    V = cfg.V_store
    print(f"  n={len(keep)} fit/sel/test={len(fit)}/{len(sel)}/{len(test)} V={V}", flush=True)

    Rs, rows = {}, []
    arms = [("U", cfg.cell_u, False), ("T", cfg.cell_t, False)]
    if cfg.views_sham:                       # the null for every statistic below, same frame
        arms.append(("sham_U", cfg.cell_u, True))
    for arm, cell, sham in arms:
        X = np.stack([np.asarray(store.get(cell, cfg.view_manifest, f"{cfg.space}.view{v}"))[keep]
                      for v in range(V)], 1).astype(np.float32)          # [n, V, d]
        if sham:
            G = random_linear_map(X.shape[-1], 1e4, seed=1)
            X = (X.reshape(-1, X.shape[-1]) @ G.T).reshape(X.shape).astype(np.float32)
        mu, W = whiten_frame(X[fit].reshape(-1, X.shape[-1]), 0.0, cfg.rank)
        anc = X[aidx].mean(1)                                            # anchor = 8-view centroid
        R = torch.stack([relspace(X[:, v], anc, mu, W) for v in range(V)], 1)   # [n, V, A]
        Rs[arm] = R
        bar = R.mean(1)
        res = R - bar[:, None]
        # PROVEN degenerate as a per-image linear feature: sum_v res_iv = 0 by construction, so at
        # W = 0 the cross-entropy gradient on a residual block is sum_v (p - y) x res_iv =
        # (p - y) x 0 = 0 and those weights never leave zero (the smoke read C_both bit-identical
        # to A_mean and D_res bit-identical across two different models -- a proof, not a bug).
        # The residual's content is therefore SECOND order: s2_ij is how much image i's similarity
        # to anchor j wobbles under augmentation, per-image and anchor-indexed in the same shared
        # frame, and identical across cells under the affine null because R is.
        s2 = res.pow(2).mean(1)                                            # [n, A]
        w = float(res.pow(2).sum(-1).mean())                             # within-image (view)
        b_hat = float((bar - bar.mean(0)).pow(2).sum(-1).mean())
        rows.append(dict(arm=arm, stat="variance_split", within=w, between_debiased=b_hat - w / V,
                         between_raw=b_hat, theta=w / max(b_hat - w / V, 1e-12)))
        print(f"  {arm}: within={w:.4f} between(debiased)={b_hat - w / V:.4f} "
              f"ratio={w / max(b_hat - w / V, 1e-12):.4f}", flush=True)
        del X
        if sham:                             # residual statistics only; the probe arms are the
            continue                         # comparison, and a recoding of U is not one

        m, sd = s2[fit].mean(0), s2[fit].std(0) + 1e-8      # fit-split standardization, so the
        s2 = (s2 - m) / sd                                  # ridge sees both blocks on one scale
        cm = dict(bar=class_margin(bar[test].cpu().numpy(), y[test]),
                  res=class_margin(res[test].reshape(-1, R.shape[-1]).cpu().numpy(),
                                   np.repeat(y[test], V)),
                  s2=class_margin(s2[test].cpu().numpy(), y[test]))
        rows.append(dict(arm=arm, stat="class_margin", cm_bar=cm["bar"], cm_res=cm["res"],
                         cm_s2=cm["s2"]))
        print(f"  {arm}: class margin  centroid={cm['bar']:+.4f}  residual rows={cm['res']:+.4f} "
              f"  residual 2nd moment={cm['s2']:+.4f}", flush=True)

        def rows_of(idx, kind):
            """A_mean / C_both / D_res2 are PER IMAGE and share one row set, so the C - A contrast
            is exactly 'is the augmentation-sensitive structure available?' with nothing else moved.
            B_view and D_res1 are per (image, view)."""
            if kind in ("B_view", "D_res1"):
                f = (R if kind == "B_view" else res)[idx].reshape(len(idx) * V, -1)
                return f.cpu().numpy(), np.repeat(y[idx], V)
            f = {"A_mean": lambda: bar[idx],
                 "C_both": lambda: torch.cat([bar[idx], s2[idx]], -1),
                 "D_res2": lambda: s2[idx]}[kind]()
            return f.cpu().numpy(), y[idx]

        for kind in ["A_mean", "C_both", "D_res2", "B_view", "D_res1"]:
            Xf, yf = rows_of(fit, kind)
            Xs, ys = rows_of(sel, kind)
            Xt_, yt_ = rows_of(test, kind)
            if kind == "C_both":
                # the block grid must REACH the "second block switched off" limit, or the nesting
                # guarantee is not in the search space and the increment can read negative
                out, lam, s = best_lam_blocks(Xf, yf, Xs, ys, Xt_, yt_, ncls,
                                              list(cfg.lam_grid) + list(cfg.block_lam_extra),
                                              bar.shape[-1], f"{arm} {kind}")
            else:
                out, lam, s = best_lam(Xf, yf, Xs, ys, Xt_, yt_, ncls, cfg.lam_grid,
                                       f"{arm} {kind}")
            rows.append(dict(arm=arm, stat=kind, lam=lam, sel_acc=s, test_acc=out["val_acc"],
                             train_acc=out["train_acc"], d=Xf.shape[1], n_rows=len(Xf)))

    mats = {}
    for nm, other in [("signal_UT", "T")] + ([("sham_U", "sham_U")] if cfg.views_sham else []):
        D_inv = Rs[other].mean(1) - Rs["U"].mean(1)
        D_var = ((Rs[other] - Rs[other].mean(1, keepdim=True))
                 - (Rs["U"] - Rs["U"].mean(1, keepdim=True)))
        M, _ = classpair(D_inv[test], y[test], y[aidx], ncls)
        dg = float(np.trace(M) / ncls)
        off = float((M.sum() - np.trace(M)) / (ncls * ncls - ncls))
        mats[nm] = M
        rows.append(dict(arm=nm, stat="D_inv_classpair", cp_diag=dg, cp_off=off,
                         cp_signed=dg - off, rms=float(D_inv[test].pow(2).mean().sqrt())))
        # the residual sums to zero over views, so ANY view-averaged linear read of D_var vanishes
        # by construction: magnitude is the only available summary of it, and magnitude carries an
        # unknown seed component (no seed-1 cell exists) -- reported un-nulled, read against sham
        rows.append(dict(arm=nm, stat="D_var_rms_only",
                         rms=float(D_var[test].pow(2).mean().sqrt())))
        print(f"  {nm}: D_inv class-pair SIGNED = {dg - off:+.5f} | "
              f"D_var rms = {rows[-1]['rms']:.5f} (magnitude only, un-nulled)", flush=True)
    write_csv("e36_views.csv", rows)
    np.savez(DIAG / "e36_views_classpair.npz", **mats)


def stage_knn(cfg, store):
    """kNN is the most conditioning-SENSITIVE reader in the battery: a raw cosine metric is
    dominated by whatever few directions carry the most variance, and the untreated cell has
    effective rank 53.9 against the treated cell's 202.6. The stored raw-cosine rows read .5784 vs
    .6626 -- an 8.42-point gap against the linear reader's 1.78, which is exactly the asymmetry
    "the treatment mostly made h easier to read" predicts.

    On alpha = 0 whitened features the same cosine kNN becomes EXACTLY invariant to an invertible
    linear map of h (whitening leaves an orthogonal transform, which cosine does not see), so the
    whitened gap is the part of the 8.42 that conditioning cannot explain. The sham arm is the
    selftest: it must reproduce U exactly at alpha = 0 and need not anywhere else."""
    ytr = np.asarray(store.labels(cfg.cell_u, cfg.fit_manifest))
    yva = np.asarray(store.labels(cfg.cell_u, cfg.val_manifest))
    ncls = int(ytr.max()) + 1
    G = random_linear_map(384, 1e4, seed=1)
    rows, preds = [], {}
    for arm, cell in [("U", cfg.cell_u), ("T", cfg.cell_t), ("sham_U", cfg.cell_u)]:
        Xtr = np.asarray(store.get(cell, cfg.fit_manifest, cfg.space)).astype(np.float32)
        Xva = np.asarray(store.get(cell, cfg.val_manifest, cfg.space)).astype(np.float32)
        if arm == "sham_U":
            Xtr, Xva = Xtr @ G.T, Xva @ G.T
        views = [("raw", Xtr, Xva),                       # the stored protocol, for parity
                 ("center", Xtr - Xtr.mean(0), Xva - Xtr.mean(0))]
        for a in cfg.alpha_probe_grid:
            mu, W = whiten_frame(Xtr, a, cfg.rank)
            views.append((f"whiten_a{a:g}", apply_whiten(Xtr, mu, W), apply_whiten(Xva, mu, W)))
        for nm, A, B in views:
            r = {}
            for k in cfg.knn_k_grid:
                r[k], preds[(arm, nm, k)] = knn_topk_acc(A, ytr, B, yva, num_classes=ncls,
                                                         knn_k=k, device=DEV, return_pred=True)
            rows.append(dict(arm=arm, transform=nm, **{f"knn_k{k}": v for k, v in r.items()}))
            print(f"  {arm:7s} {nm:14s} " +
                  "  ".join(f"k={k}: {v:.4f}" for k, v in r.items()), flush=True)
        del Xtr, Xva
    for nm in [v["transform"] for v in rows if v["arm"] == "U"]:
        g = {a: next(v for v in rows if v["arm"] == a and v["transform"] == nm)
             for a in ("U", "T", "sham_U")}
        for k in cfg.knn_k_grid:
            c = f"knn_k{k}"
            n01, n10, pv = mcnemar(preds[("U", nm, k)], preds[("T", nm, k)], yva)
            s01, s10, ps = mcnemar(preds[("U", nm, k)], preds[("sham_U", nm, k)], yva)
            rows.append(dict(arm="GAP", transform=nm, k=k, gap=g["T"][c] - g["U"][c],
                             T_fixes=n01, T_breaks=n10, p=pv,
                             sham_dev=abs(g["sham_U"][c] - g["U"][c]), sham_p=ps))
            print(f"  GAP {nm:14s} k={k:<4d} T-U = {g['T'][c] - g['U'][c]:+.4f}  "
                  f"(T fixes {n01}, breaks {n10}, p={pv:.3g})   "
                  f"selftest |sham-U| = {abs(g['sham_U'][c] - g['U'][c]):.5f}", flush=True)
    write_csv("e36_knn.csv", rows)


def _refs(path):
    with open(path) as f:
        return [r["ref"] for r in csv.DictReader(f)]


def stage_mc(cfg, store):
    """Berker 2026-09-10: after averaging away the augmentations, does the CLEAN view carry
    anything extra that helps classification?

        m_i = (1/8) sum_v h_iv        the augmentation centroid
        c_i = h_i^clean - m_i         what the clean view adds on top of it

    Three probes per cell -- m, c, [m, c] -- and one comparison, Acc([m,c]) - Acc(m). Note
    span{m, c} = span{m, h_clean}, so that increment is exactly "what the clean feature adds to the
    augmentation average", read by a linear probe.

    This supersedes the second-moment arm of the views stage: the VIEW residual sums to zero over
    views and is provably unreadable by a linear probe, but c does not -- it is one specific vector
    per image -- so the first moment is available here and no second-moment detour is needed.

    Two things from this card's own numbers are kept and nothing else: an alpha = 0 whitening frame
    per block with a convergent L-BFGS reader (the stored AdamW reader moves 4.86 points under a
    CONTENT-FREE recoding, so it cannot referee an increment), and a per-block lam (with one shared
    lam a concatenation does not nest its own sub-model and the increment can read negative)."""
    rc, rv = _refs(cfg.clean_manifest_csv), _refs(cfg.view_manifest_csv)
    pos = {r: j for j, r in enumerate(rv)}
    missing = [r for r in rc if r not in pos]
    assert not missing, f"{len(missing)} clean images have no view rows"
    vmap = np.array([pos[r] for r in rc])
    y = np.asarray(store.labels(cfg.cell_u, cfg.fit_manifest))
    ncls = int(y.max()) + 1
    fit, sel, test = strat_split(y, [0.6, 0.2, 0.2], seed=cfg.split_seed)
    G = random_linear_map(384, 1e4, seed=1)
    print(f"  n={len(rc)} fit/sel/test={len(fit)}/{len(sel)}/{len(test)}", flush=True)

    ranks = {}

    def whiten_block(X, key):
        """alpha = 0 at the block's numerical rank, FIT on the fit split only.

        The rank is determined ONCE (on the first arm, the untreated cell) and reused for every
        other arm. It must not be re-derived per arm from a RELATIVE threshold: the sham is recoded
        at condition number 1e4, which spreads the spectrum by 1e8 in variance and pushes dozens of
        directions under any relative cut -- the first run of this stage probed the sham in 324
        dimensions against the untreated cell's 384 and the selftest then measured that rank
        mismatch (1.66 points on the c block) rather than the reader's conditioning-sensitivity,
        because rank truncation is not equivariant."""
        if key not in ranks:
            lam = np.clip(np.linalg.eigvalsh(np.cov(X[fit], rowvar=False))[::-1], 0, None)
            ranks[key] = int((lam > cfg.rank_rel_tol * lam[0]).sum())
        mu, W = whiten_frame(X[fit], 0.0, ranks[key])
        return apply_whiten(X, mu, W), ranks[key]

    rows, acc = [], {}
    for arm, cell in [("U", cfg.cell_u), ("T", cfg.cell_t), ("sham_U", cfg.cell_u)]:
        Hc = np.asarray(store.get(cell, cfg.fit_manifest, cfg.space)).astype(np.float32)
        m = np.zeros_like(Hc)
        for v in range(cfg.V_store):
            m += np.asarray(store.get(cell, cfg.view_manifest,
                                      f"{cfg.space}.view{v}"))[vmap].astype(np.float32)
        m /= cfg.V_store
        c = Hc - m
        if arm == "sham_U":                       # one recoding applied to BOTH, so m and c follow
            Hc, m, c = Hc @ G.T, m @ G.T, c @ G.T
        Zm, rm = whiten_block(m, "m")
        Zc, rck = whiten_block(c, "c")
        Zh, rh = whiten_block(Hc, "h_clean")
        preds = {}
        for nm, X, blocks in [("m", Zm, None), ("c", Zc, None),
                              ("h_clean", Zh, None),
                              ("mc", np.concatenate([Zm, Zc], 1), Zm.shape[1])]:
            if blocks is None:
                out, lam, s = best_lam(X[fit], y[fit], X[sel], y[sel], X[test], y[test], ncls,
                                       cfg.lam_grid, f"{arm} {nm}")
            else:
                out, lam, s = best_lam_blocks(X[fit], y[fit], X[sel], y[sel], X[test], y[test],
                                              ncls, list(cfg.lam_grid) + list(cfg.block_lam_extra),
                                              blocks, f"{arm} {nm}")
            preds[nm] = out["pred"]
            acc[(arm, nm)] = out["val_acc"]
            rows.append(dict(arm=arm, probe=nm, lam=lam, d=X.shape[1], rank_m=rm, rank_c=rck,
                             sel_acc=s, test_acc=out["val_acc"], train_acc=out["train_acc"]))
        n01, n10, pv = mcnemar(preds["m"], preds["mc"], y[test])
        inc = acc[(arm, "mc")] - acc[(arm, "m")]
        rows.append(dict(arm=arm, probe="INCREMENT_mc_minus_m", test_acc=inc,
                         T_fixes=n01, T_breaks=n10, p=pv))
        print(f"  {arm}: Acc(m)={acc[(arm, 'm')]:.4f}  Acc(c)={acc[(arm, 'c')]:.4f}  "
              f"Acc([m,c])={acc[(arm, 'mc')]:.4f}  Acc(h_clean)={acc[(arm, 'h_clean')]:.4f}",
              flush=True)
        print(f"  {arm}: INCREMENT Acc([m,c]) - Acc(m) = {inc:+.4f}  "
              f"(fixes {n01}, breaks {n10}, p={pv:.3g})", flush=True)
        del Hc, m, c, Zm, Zc, Zh

    iu, it = acc[("U", "mc")] - acc[("U", "m")], acc[("T", "mc")] - acc[("T", "m")]
    rows.append(dict(arm="T_minus_U", probe="centroid_gap", test_acc=acc[("T", "m")] - acc[("U", "m")]))
    rows.append(dict(arm="T_minus_U", probe="increment_gap", test_acc=it - iu))
    print(f"\n  centroid gap  Acc_T(m) - Acc_U(m)      = {acc[('T', 'm')] - acc[('U', 'm')]:+.4f}",
          flush=True)
    print(f"  increment gap  inc_T - inc_U            = {it - iu:+.4f}", flush=True)
    for nm in ("m", "c", "mc", "h_clean"):
        d = abs(acc[("sham_U", nm)] - acc[("U", nm)])
        print(f"  SELFTEST {nm:8s} |sham - U| = {d:.5f}", flush=True)
    write_csv("e36_mc.csv", rows)


def stage_theta(cfg, store):
    """Theta = spread across views / spread across images, decomposed in RAW h.

    Berker 2026-09-10: backbone invariance is a function of IMAGE spread too, so the h-conditioner
    -- which acts on the between-image distribution of per-image view MEANS (`h_floor_batch =
    view_mean`, read off both checkpoints) -- does move Theta, through its denominator. Theta rose
    .255 -> .373 with the treatment (brief §2). Which term moved is a different question from
    whether it moved, and it decides the story: a numerator rise means the backbone really did keep
    more view-to-view variation; a denominator FALL would mean image spread collapsed and Theta rose
    for the opposite reason. Both are reported raw and unnormalized, with the trace and the spectrum
    of the view-mean covariance, because Theta is a ratio and hides the scale of both terms.

    W = E_i (1/V) sum_v ||h_iv - m_i||^2      (within image, across views)
    B = E_i ||m_i - m_bar||^2 - W/V           (between images, debiased: the V-view mean still
                                               carries 1/V of the view variance)
    The sham arm rides along because Theta is a RAW-metric quantity: it survives isotropic rescaling
    but not a general invertible map, so it cannot referee anything on its own."""
    rc, rv = _refs(cfg.clean_manifest_csv), _refs(cfg.view_manifest_csv)
    pos = {r: j for j, r in enumerate(rv)}
    vmap = np.array([pos[r] for r in rc])
    G = random_linear_map(384, 1e4, seed=1)
    V = cfg.V_store
    rows = []
    for arm, cell in [("U", cfg.cell_u), ("T", cfg.cell_t), ("sham_U", cfg.cell_u)]:
        X = np.stack([np.asarray(store.get(cell, cfg.view_manifest, f"{cfg.space}.view{v}"))[vmap]
                      for v in range(V)], 1).astype(np.float32)
        if arm == "sham_U":
            X = (X.reshape(-1, X.shape[-1]) @ G.T).reshape(X.shape).astype(np.float32)
        m = X.mean(1)
        W = float(((X - m[:, None]) ** 2).sum(-1).mean())
        B_hat = float(((m - m.mean(0)) ** 2).sum(-1).mean())
        B = B_hat - W / V
        lam = np.clip(np.linalg.eigvalsh(np.cov(m, rowvar=False))[::-1], 0, None)
        rows.append(dict(arm=arm, W_view=W, B_image_debiased=B, B_image_raw=B_hat,
                         theta=W / B, trace_cov_m=float(lam.sum()), lam1_cov_m=float(lam[0]),
                         effrank_cov_m=effective_rank(lam)))
        print(f"  {arm:7s} W(view)={W:.4f}  B(image,debiased)={B:.4f}  Theta=W/B={W / B:.4f}   "
              f"| tr Cov(m)={lam.sum():.4f}  lam1={lam[0]:.4f}  effrank={effective_rank(lam):.1f}",
              flush=True)
        del X, m
    u, t = rows[0], rows[1]
    print(f"\n  T/U ratios:  W(view) x{t['W_view'] / u['W_view']:.3f}   "
          f"B(image) x{t['B_image_debiased'] / u['B_image_debiased']:.3f}   "
          f"Theta x{t['theta'] / u['theta']:.3f}", flush=True)
    print(f"  Theta under a CONTENT-FREE recoding of U: {u['theta']:.4f} -> "
          f"{rows[2]['theta']:.4f}  (Theta is a raw-metric quantity)", flush=True)
    write_csv("e36_theta.csv", rows)


def stage_thickness(cfg, store):
    """Augmentation thickness AS THE PAPER DEFINES IT (Berker's theory section, 2026-09-10):

        C_h = B_h + A_h,   B_h = Cov(E[H|Q]),   A_h = E[Cov(H|Q)]
        Theta_h = B_h^{dagger/2} A_h B_h^{dagger/2}          (a MATRIX)
        theta_h(a) = a' A_h a / a' B_h a                      (directional)

    Why this stage exists: none of the three "thickness" numbers this card has reported is
    Theta_h. `stage_theta` reported tr(A_h)/tr(B_h), a trace ratio that is neither an eigenvalue of
    Theta_h nor its operator norm; `stage_views` whitened by the POOLED covariance C_h, not by B_h.
    The theorem's two sides use ||Theta_h||_op (the WORST direction) and a' Theta_h (I+Theta_h)^-1 a
    (the TASK direction), and an averaged statistic can hide both -- the same failure mode as
    reporting sliced Gaussianity without a worst-direction read.

    Theorem 2 (too much thickness) is an EXACT identity for least squares:

        inf_u E[(y - u'H)^2] = d_h(y)^2 + a' Theta_h (I + Theta_h)^-1 a

    so it doubles as the selftest for this whole instrument: the left side is fit directly from
    single views, the right side is assembled from the centre fit and Theta_h, and they must agree.
    If they do not, A_h / B_h / Theta_h are mis-estimated and no thickness number here is readable.

    Estimators, stated: A_h uses the (V-1) correction; B_h = Cov(m) - A_h/V, debiased because the
    V-view mean still carries 1/V of the view variance. Targets are CENTRED one-hot (least squares,
    matching the theorem, not the logistic reader used elsewhere on this card)."""
    rc, rv = _refs(cfg.clean_manifest_csv), _refs(cfg.view_manifest_csv)
    pos = {r: j for j, r in enumerate(rv)}
    vmap = np.array([pos[r] for r in rc])
    y = np.asarray(store.labels(cfg.cell_u, cfg.fit_manifest))
    ncls, V = int(y.max()) + 1, cfg.V_store
    Y = np.eye(ncls, dtype=np.float64)[y]
    Y -= Y.mean(0)
    G = random_linear_map(384, 1e4, seed=1)
    rows = []
    for arm, cell in [("U", cfg.cell_u), ("T", cfg.cell_t), ("sham_U", cfg.cell_u)]:
        X = np.stack([np.asarray(store.get(cell, cfg.view_manifest, f"{cfg.space}.view{v}"))[vmap]
                      for v in range(V)], 1).astype(np.float32)
        if arm == "sham_U":
            X = (X.reshape(-1, X.shape[-1]) @ G.T).reshape(X.shape).astype(np.float32)
        n, _, d = X.shape
        m = X.mean(1).astype(np.float64)
        Xc = (X - X.mean(1, keepdims=True)).reshape(-1, d).astype(np.float64)
        A = Xc.T @ Xc / (n * (V - 1))                       # E[Cov(H|Q)], (V-1)-corrected
        del Xc
        mc = m - m.mean(0)
        B = mc.T @ mc / n - A / V                           # debiased Cov(E[H|Q])
        lam, U_ = np.linalg.eigh(B)
        lam, U_ = np.clip(lam[::-1], 0, None), np.ascontiguousarray(U_[:, ::-1])
        r = int((lam > cfg.rank_rel_tol * lam[0]).sum())
        P = U_[:, :r] / np.sqrt(lam[:r])                    # B_h^{dagger/2}
        Th = P.T @ A @ P                                    # Theta_h
        Th = (Th + Th.T) / 2
        ev = np.clip(np.linalg.eigvalsh(Th)[::-1], 0, None)

        Zm = mc @ P                                         # centres, Cov ~ I
        Smm = Zm.T @ Zm / n
        a = np.linalg.solve(Smm, Zm.T @ Y / n)              # [r, ncls] task directions
        d2 = (Y * Y).mean(0) - np.einsum("ic,ic->c", Zm @ a, Y) / n   # d_h(y)^2 per class
        IpT = np.linalg.inv(np.eye(r) + Th)
        pen = np.einsum("rc,rs,sc->c", a, Th @ IpT, a)      # the Eq (too-thick) penalty

        gm = X.reshape(-1, d).mean(0).astype(np.float64)    # single-view LS via moments only
        Shh = np.zeros((r, r)); Shy = np.zeros((r, ncls)); yy = (Y * Y).mean(0)
        for v in range(V):
            Zv = (X[:, v].astype(np.float64) - gm) @ P
            Shh += Zv.T @ Zv; Shy += Zv.T @ Y
        Shh /= n * V; Shy /= n * V
        lhs = yy - np.einsum("rc,rc->c", Shy, np.linalg.solve(Shh, Shy))
        rhs = d2 + pen
        err = float(np.abs(lhs - rhs).max() / np.abs(rhs).max())
        rows.append(dict(arm=arm, rank_B=r, theta_op=float(ev[0]), theta_mean=float(ev.mean()),
                         theta_median=float(np.median(ev)), theta_min=float(ev[-1]),
                         theta_p90=float(np.quantile(ev, .9)),
                         trace_ratio=float(np.trace(A) / np.trace(B)),
                         task_theta_mean=float((a * (Th @ a)).sum(0).mean()
                                               / (a * a).sum(0).mean()),
                         penalty_sum=float(pen.sum()), d2_sum=float(d2.sum()),
                         identity_rel_err=err))
        print(f"  {arm:7s} rank(B)={r}  ||Theta||_op={ev[0]:.4f}  mean={ev.mean():.4f}  "
              f"median={np.median(ev):.4f}  p90={np.quantile(ev, .9):.4f}  min={ev[-1]:.4f}",
              flush=True)
        print(f"          tr(A)/tr(B)={np.trace(A) / np.trace(B):.4f}   "
              f"task-direction theta={rows[-1]['task_theta_mean']:.4f}   "
              f"sum d_h^2={d2.sum():.4f}  sum penalty={pen.sum():.4f}", flush=True)
        print(f"          SELFTEST Thm-2 identity  max rel err = {err:.3e}", flush=True)
        np.savez(DIAG / f"e36_thickness_spectrum_{arm}.npz", eig=ev, d2=d2, pen=pen)
        del X, m, mc, Zm
    write_csv("e36_thickness.csv", rows)


def stage_thickprofile(cfg, store):
    """WHERE is each cell thick? Two checks the single Theta_h table cannot answer.

    (1) RANK STABILITY. ||Theta_h||_op divides by B_h's smallest RETAINED eigenvalue (~3e-9 for the
    untreated cell), so the reported 32.57 could be a tail estimation artifact rather than a real
    worst direction. Recomputed over a grid of retained ranks: if the op norm collapses as the tail
    is dropped it is noise, and only the bulk statistics are readable.

    (2) DIRECTIONAL PROFILE. theta_h(u_k) = u_k' A_h u_k / lambda_k(B_h) along B_h's OWN
    eigendirections, ordered by image spread. This is what adjudicates the disagreement between the
    trace ratio (which says the treatment made h thicker) and Theta_h's spectrum (which says
    thinner): the trace ratio weights directions by raw variance and so reports the top of the
    spectrum, while the spectrum of Theta_h treats all directions alike. Both can be true of
    different parts of the spectrum, and the profile shows which part is which.

    Also reported: the COUNT of directions with theta > 1, i.e. where view spread exceeds image
    spread outright -- a direction-counting read of "too thick" that no ratio of traces provides."""
    rc, rv = _refs(cfg.clean_manifest_csv), _refs(cfg.view_manifest_csv)
    pos = {r: j for j, r in enumerate(rv)}
    vmap = np.array([pos[r] for r in rc])
    V = cfg.V_store
    rows = []
    for arm, cell in [("U", cfg.cell_u), ("T", cfg.cell_t)]:
        X = np.stack([np.asarray(store.get(cell, cfg.view_manifest, f"{cfg.space}.view{v}"))[vmap]
                      for v in range(V)], 1).astype(np.float32)
        n, _, d = X.shape
        m = X.mean(1).astype(np.float64)
        Xc = (X - X.mean(1, keepdims=True)).reshape(-1, d).astype(np.float64)
        A = Xc.T @ Xc / (n * (V - 1))
        del Xc, X
        mc = m - m.mean(0)
        B = mc.T @ mc / n - A / V
        lam, U_ = np.linalg.eigh(B)
        lam, U_ = np.clip(lam[::-1], 0, None), np.ascontiguousarray(U_[:, ::-1])

        # (2) profile along B_h's own eigendirections, ordered by image spread
        a_diag = np.einsum("dk,de,ek->k", U_, A, U_)
        prof = a_diag / np.maximum(lam, 1e-300)
        np.savez(DIAG / f"e36_thickprofile_{arm}.npz", theta=prof, lam_B=lam, a_diag=a_diag)
        dec = [float(np.median(prof[i * 38:(i + 1) * 38])) for i in range(10)]
        print(f"  {arm}: theta along B_h eigendirections, median per decile of image spread "
              f"(strongest -> weakest):", flush=True)
        print("       " + "  ".join(f"{v:7.3f}" for v in dec), flush=True)
        print(f"       directions with theta > 1: {int((prof[:383] > 1).sum())} / 383   "
              f"theta at the strongest 38: median {dec[0]:.3f}", flush=True)

        # (1) rank stability of the operator norm and the bulk statistics
        for r in cfg.thick_rank_grid:
            P = U_[:, :r] / np.sqrt(lam[:r])
            Th = P.T @ A @ P
            ev = np.clip(np.linalg.eigvalsh((Th + Th.T) / 2)[::-1], 0, None)
            rows.append(dict(arm=arm, rank=r, theta_op=float(ev[0]), theta_mean=float(ev.mean()),
                             theta_median=float(np.median(ev)),
                             theta_p90=float(np.quantile(ev, .9)),
                             n_theta_gt1=int((prof[:r] > 1).sum()),
                             lam_B_min_kept=float(lam[r - 1]),
                             **{f"dec{i}": dec[i] for i in range(10)}))
            print(f"    rank={r:3d}  ||Theta||_op={ev[0]:8.3f}  mean={ev.mean():6.3f}  "
                  f"median={np.median(ev):6.3f}  p90={np.quantile(ev, .9):6.3f}  "
                  f"lam_B(min kept)={lam[r - 1]:.3e}", flush=True)
        del A, B, m, mc, U_
    write_csv("e36_thickprofile.csv", rows)


ZOO_PAIRS = [("byol", "in100.byol.s0.ep100.o8", "in100.byol.s0.e20f.ep100.o8"),
             ("dino", "in100.dino.s0.ep100.o8", "in100.dino.s0.e20f.ep100.o8"),
             ("lejepa", "in100.lejepa.s0.ep100.o8", "in100.lejepa.s0.e20f.ep100.o8"),
             ("simclr", "in100.simclr.s0.ep100.o8", "in100.simclr.s0.e20f.ep100.o8"),
             ("vicreg", "in100.vicreg.s0.ep100.o8", "in100.vicreg.s0.e20f.ep100.o8"),
             ("ours", "in100.floorssl.s0.d256vm4zonly.extL", "in100.floorssl.s0.d256vm4.extL")]


def _AB(store, run, manifest, space, V):
    X = np.stack([np.asarray(store.get(run, manifest, f"{space}.view{v}"))
                  for v in range(V)], 1).astype(np.float32)
    n, _, d = X.shape
    m = X.mean(1).astype(np.float64)
    Xc = (X - X.mean(1, keepdims=True)).reshape(-1, d).astype(np.float64)
    A = Xc.T @ Xc / (n * (V - 1))
    mc = m - m.mean(0)
    B = mc.T @ mc / n - A / V
    return A, B, n


def stage_zoo(cfg, store):
    """Does "treated cells ALWAYS have larger Theta at CLS" hold for the object Theorem 1 is
    proved about, or only for the trace ratio the appendix figure plots?

    The figure's column is `omega = W/B` from `orbit_energies` -- a ratio of two ENERGIES. The
    theory's Theta_h = B^{dagger/2} A B^{dagger/2} is a MATRIX whose spectrum is invariant to any
    invertible linear map of h. On our own pair the two disagree in SIGN (omega .248 -> .364,
    treated thicker; every functional of Theta_h, treated thinner). This runs both for all six
    control/treated pairs that have an 8-view store, so the question is settled on the zoo rather
    than on one cell.

    ESTIMATION LIMIT, stated because it decides how these are read: the zoo stores hold n = 10,000
    images for d = 384, so n/d = 26 and B_h's tail eigenvalues carry ~20% relative error
    (Marchenko-Pastur sqrt(d/n) = .196). ||Theta_h||_op divides by the SMALLEST retained eigenvalue
    and is therefore not estimable at full rank here -- the rank sweep on our own pair moved it from
    32.6 (rank 383) to 0.74 (rank 50). So every pair is compared at MATCHED, TRUNCATED rank, and
    the mean eigenvalue (robust) is reported next to the operator norm (fragile). Our own pair also
    rides along on its 50k store, where n/d = 130."""
    rows = []
    for name, ctrl, trt in ZOO_PAIRS:
        mani = cfg.zoo_manifest if name != "ours" else cfg.view_manifest
        try:
            Ac, Bc, nc = _AB(store, ctrl, mani, cfg.space, cfg.V_store)
            At, Bt, nt = _AB(store, trt, mani, cfg.space, cfg.V_store)
        except (FileNotFoundError, KeyError, OSError) as e:
            print(f"  {name:8s} SKIPPED ({type(e).__name__})", flush=True)
            continue
        out = {}
        for tag, A, B in [("control", Ac, Bc), ("treated", At, Bt)]:
            lam, U_ = np.linalg.eigh(B)
            lam, U_ = np.clip(lam[::-1], 0, None), np.ascontiguousarray(U_[:, ::-1])
            out[tag] = (A, lam, U_, np.trace(A) / np.trace(B))
        for r in cfg.zoo_rank_grid:
            rec = dict(method=name, rank=r, n=nc)
            for tag in ("control", "treated"):
                A, lam, U_, om = out[tag]
                P = U_[:, :r] / np.sqrt(np.maximum(lam[:r], 1e-300))
                Th = P.T @ A @ P
                ev = np.clip(np.linalg.eigvalsh((Th + Th.T) / 2)[::-1], 0, None)
                rec[f"{tag}_omega"] = float(om)
                rec[f"{tag}_theta_mean"] = float(ev.mean())
                rec[f"{tag}_theta_med"] = float(np.median(ev))
                rec[f"{tag}_theta_op"] = float(ev[0])
            rec["omega_treated_larger"] = rec["treated_omega"] > rec["control_omega"]
            rec["theta_mean_treated_larger"] = rec["treated_theta_mean"] > rec["control_theta_mean"]
            rows.append(rec)
            print(f"  {name:8s} rank={r:3d}  omega {rec['control_omega']:.4f} -> "
                  f"{rec['treated_omega']:.4f} {'UP  ' if rec['omega_treated_larger'] else 'DOWN'}"
                  f" | Theta mean {rec['control_theta_mean']:.4f} -> "
                  f"{rec['treated_theta_mean']:.4f} "
                  f"{'UP  ' if rec['theta_mean_treated_larger'] else 'DOWN'}"
                  f" | op {rec['control_theta_op']:8.3f} -> {rec['treated_theta_op']:8.3f}",
                  flush=True)
        del out
    n_om = sum(r["omega_treated_larger"] for r in rows if r["rank"] == cfg.zoo_rank_grid[0])
    n_th = sum(r["theta_mean_treated_larger"] for r in rows if r["rank"] == cfg.zoo_rank_grid[0])
    tot = len([r for r in rows if r["rank"] == cfg.zoo_rank_grid[0]])
    print(f"\n  at rank {cfg.zoo_rank_grid[0]}: omega treated-larger in {n_om}/{tot} pairs;  "
          f"Theta_h mean treated-larger in {n_th}/{tot}", flush=True)
    write_csv("e36_zoo_thickness.csv", rows)


@hydra.main(version_base=None, config_path="configs", config_name="e36_relres")
def main(cfg: DictConfig):
    print(OmegaConf.to_yaml(cfg), flush=True)
    store = FeatureStore(cfg.store_root)
    for s in cfg.stages:
        print(f"\n=== stage {s} ===", flush=True)
        {"preflight": stage_preflight, "reader": stage_reader, "rel": stage_rel,
         "views": stage_views, "knn": stage_knn, "mc": stage_mc,
         "theta": stage_theta, "thickness": stage_thickness,
         "thickprofile": stage_thickprofile, "zoo": stage_zoo}[s](cfg, store)
    print("\ndone", flush=True)


if __name__ == "__main__":
    main()
