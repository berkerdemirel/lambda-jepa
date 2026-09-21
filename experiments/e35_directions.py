"""E35: which directions of h does the objective price?

One trained model, one representation, two readers that disagree. Order h's directions by
variance on a clean fit set, delete them from the bottom, and at every truncation read (a) the
run's OWN training loss on augmented views — the projector is frozen and fed the truncated h, so
this is the loss the model would have incurred had the encoder never produced those directions —
and (b) the linear/kNN probes. The cell is the UNTREATED arm (`w_cond_h = 0`): everything the
objective knows about h reaches it through the head, which is the standard-SSL situation.

Controls (card §Arms): the same NUMBER of Haar-random directions (is this just dimensionality
reduction?) and the mirror cut from the top (can the instrument move the loss at all?).

Anatomy, stated where it is used: V = 4 audit-stack views (= this lane's training aug family and
its training V), the conditioner fed per-image view MEANS at n = 512 rows per batch — the lane's
own (queue_steps+1)*bs — on a fixed seeded 128-d slice, so the offline estimator has the training
estimator's n/d' = 4. Probes read CLEAN eval features, train manifest -> val manifest.

  sbatch slurm/e35_directions.sbatch
  sbatch slurm/e35_directions.sbatch n_images=4096 'keep_grid=[384,128,16]' probes=[linear]
"""
import csv
import os

import hydra
import numpy as np
import torch
from omegaconf import DictConfig

from sslgap.ckpt import adapters
from sslgap.extract import FeatureStore
from sslgap.metrics.isotropy import fixed_slice, moment_kl_slice
from sslgap.metrics.spectra import effective_rank, pca_frame, random_basis
from sslgap.paths import DIAG, figdir
from sslgap.probes import knn_topk_acc, linear_raw_v2

DIR_COLS = ["cell", "j", "lam", "view_share", "d_loss", "d_inv", "d_theta_z", "d_cond_z",
            "d_top1", "loss0", "inv0", "theta_z0", "top1_0"]
CUM_COLS = ["cell", "order", "n_removed", "frac_var_removed", "loss", "inv", "cond_z", "theta_z",
            "lin", "lin_train", "lin_bestep"]


def var_kept(B, lam, V):
    """Fraction of the fit set's variance inside span(B): sum_j b_j' Sigma b_j / tr(Sigma)."""
    c = V.T @ B
    return float((lam[:, None] * c * c).sum() / lam.sum())


def thickness(W, centers, V):
    """Theta = W/B in the house convention (sslgap.metrics.orbit_energy): the within-cloud pair
    energy over the DEBIASED between-center energy, B = B_hat - W/V. Scale-free, so it says
    whether views really moved closer together or z merely shrank — which a raw MSE cannot.
    The identity used to avoid a second pass over the views is asserted in the selftest."""
    m = centers - centers.mean(0)
    N = len(centers)
    B_hat = 2.0 * float((m * m).sum(1).mean()) * N / (N - 1)
    return float(W / (B_hat - W / V))


@torch.no_grad()
def loss_read(Hv, P, mu, head, dz, Qz, Qh, batches):
    """The lane's loss on the ablated views. Hv [V, N, d] fp16 (device); P [d, d] the ablation
    projector. Returns the loss terms in loss units (weights applied by the caller) plus the
    scale-free thickness of both clouds."""
    Vn, N, d = Hv.shape
    npair = Vn * (Vn - 1) / 2
    zbar = torch.empty(N, dz, device=Hv.device)
    hbar = torch.empty(N, d, device=Hv.device)
    ssz = torch.zeros((), device=Hv.device, dtype=torch.float64)
    ssh = torch.zeros((), device=Hv.device, dtype=torch.float64)
    for i in range(0, N, 8192):
        h = Hv[:, i:i + 8192].float()
        h = mu + (h - mu) @ P                                   # ambient coords, rank k
        z = head(h.reshape(-1, d))["proj.out"].reshape(Vn, h.shape[1], dz)
        for u in range(Vn):                                     # all-pairs MSE = the lane's inv
            for w in range(u + 1, Vn):
                ssz += (z[u] - z[w]).square().sum().double()
                ssh += (h[u] - h[w]).square().sum().double()
        zbar[i:i + 8192], hbar[i:i + 8192] = z.mean(0), h.mean(0)
    Wz, Wh = float(ssz) / (N * npair), float(ssh) / (N * npair)  # the house W (summed over dims)
    zb, hb = zbar.cpu().numpy(), hbar.cpu().numpy()
    return {"inv": Wz / dz,                                      # the loss term = W_z / dim(z)
            "cond_z": float(np.mean([moment_kl_slice(zb[b], Qz) for b in batches])),
            "cond_h": float(np.mean([moment_kl_slice(hb[b], Qh) for b in batches])),
            "theta_z": thickness(Wz, zb, Vn), "theta_h": thickness(Wh, hb, Vn)}


def ablate(X, P, mu):
    return (mu + (X - mu) @ P).cpu().numpy()


def write_rows(path, cols, rows):
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)


@hydra.main(version_base=None, config_path="configs", config_name="e35_directions")
def main(cfg: DictConfig):
    dev = cfg.device if torch.cuda.is_available() else "cpu"
    run_id = cfg.run_id or f"{cfg.cell}.extL"
    ckpt = cfg.ckpt or os.path.join(cfg.ckpt_dir, f"{cfg.cell}_ep{cfg.epoch}.pt")
    store = FeatureStore(cfg.store_root)
    f_dir = DIAG / f"e35_perdirection.{cfg.cell}.csv"
    f_cum = DIAG / f"e35_cumulative.{cfg.cell}.csv"
    if cfg.figure_only:
        return figure(f_dir, f_cum, cfg.cell)

    lc = adapters.load("native", ckpt, "e35")
    mcfg = lc.cfg["method"]
    # The score is ALWAYS w_inv*inv + w_cond_z*cond_z and never the conditioner at h — so on a
    # treated cell it asks "does the probe use directions the SSL objective never asked for?".
    # Scoring a treated cell with its own h term would report that the term cares about exactly
    # the directions it created (Berker 2026-09-10: "the prev experiment is self fulfilling").
    w_inv, w_cond_z = float(mcfg["w_inv"]), float(mcfg["w_floor"])
    head = lc.branches[lc.probed_branch].heads.to(dev).eval()

    Xtr = np.asarray(store.get(run_id, cfg.fit_manifest, cfg.space), dtype=np.float32)
    Xva = np.asarray(store.get(run_id, cfg.val_manifest, cfg.space), dtype=np.float32)
    ytr, yva = store.labels(run_id, cfg.fit_manifest), store.labels(run_id, cfg.val_manifest)
    ncls = int(max(ytr.max(), yva.max())) + 1
    # The basis is the eigenbasis of h — used ONLY as a complete orthonormal frame; nothing below
    # orders directions by variance (Berker 2026-09-09: the variance ordering was the flaw).
    mu_np, lam, U = pca_frame(Xtr)
    d = len(mu_np)
    dz = head(torch.zeros(2, d, device=dev))["proj.out"].shape[1]
    mu = torch.tensor(mu_np, dtype=torch.float32, device=dev)
    Xtr_t, Xva_t = torch.tensor(Xtr, device=dev), torch.tensor(Xva, device=dev)
    Ut = torch.tensor(U, dtype=torch.float32, device=dev)
    eye = torch.eye(d, device=dev)

    Hs = [store.get(run_id, cfg.cloud_manifest, f"{cfg.space}.view{v}") for v in range(cfg.V)]
    n_img = min(cfg.n_images or len(Hs[0]), len(Hs[0]))
    idx = np.random.default_rng(cfg.batch_seed).permutation(len(Hs[0]))[:n_img]
    idx.sort()                                   # memmap-friendly; the batch permutation is below
    Hv = torch.from_numpy(np.stack([np.asarray(H[idx]) for H in Hs])).to(dev)
    perm = np.random.default_rng(cfg.batch_seed).permutation(n_img)   # store order is class-grouped
    batches = [perm[i:i + cfg.cond_n] for i in range(0, n_img - cfg.cond_n + 1, cfg.cond_n)]
    Qz = fixed_slice(dz, cfg.d_slice, min(cfg.d_slice, dz), cfg.slice_seed)
    Qh = fixed_slice(d, cfg.d_slice, cfg.d_slice, cfg.slice_seed)
    print(f"[e35] {cfg.cell} d={d} dz={dz} scored by w_inv*inv + w_cond_z*cond_z "
          f"({w_inv}, {w_cond_z}); this cell's h term (w_cond_h={float(mcfg['h_lamb'])}) is NOT "
          f"in the score | loss on {n_img} "
          f"images x V={cfg.V}, {len(batches)} conditioner batches of {cfg.cond_n}", flush=True)
    if cfg.selftest:
        selftest(store, run_id, cfg, head, dev, dz)

    # view_share per direction: of the spread along u, the fraction that is disagreement between
    # views of ONE image rather than between images. 1 = pure view noise (the invariance loss
    # would like it gone), 0 = identical across views (invariance has nothing to gain from it).
    with torch.no_grad():
        C = torch.stack([(Hv[v].float() - mu) @ Ut for v in range(cfg.V)])       # [V, N, d]
        Wu = sum((C[u] - C[w]).square().mean(0) for u in range(cfg.V)
                 for w in range(u + 1, cfg.V)) / (cfg.V * (cfg.V - 1) / 2)
        m = C.mean(0)
        Bu = 2 * (m - m.mean(0)).square().mean(0) * len(m) / (len(m) - 1) - Wu / cfg.V
        share = (Wu / (Wu + Bu.clamp_min(1e-12))).cpu().numpy()
    print(f"[e35] view_share: median {np.median(share):.2f} min {share.min():.2f} "
          f"max {share.max():.2f}", flush=True)

    total = lambda L: w_inv * L["inv"] + w_cond_z * L["cond_z"]
    L0 = loss_read(Hv, eye, mu, head, dz, Qz, Qh, batches)
    p0 = linear_raw_v2(Xtr, ytr, Xva, yva, ncls, device=dev, seed=cfg.probe_seed,
                       return_probe=True)
    probe, top1_0, loss0 = p0["probe"], p0["val_acc"], total(L0)
    print(f"[e35] baseline: loss {loss0:.4f} (inv {L0['inv']:.5f} cond_z {L0['cond_z']:.4f}) "
          f"top-1 {top1_0:.4f}", flush=True)

    # (a) what the LOSS cares about, direction by direction: delete u_j alone and re-read the
    # objective. Same images, same fixed slice, frozen head -> the difference is exact.
    # (b) what the PROBE cares about: zero u_j for the already-fitted map -> exact too.
    rows = []
    with torch.no_grad():
        cv = (Xva_t - mu) @ Ut                             # val coordinates in the frame
        for j in range(d):
            u = Ut[:, j]
            Lj = loss_read(Hv, eye - torch.outer(u, u), mu, head, dz, Qz, Qh, batches)
            masked = Xva_t - torch.outer(cv[:, j], u)
            acc = (probe(masked).argmax(1).cpu().numpy() == yva).mean()
            rows.append({"cell": cfg.cell, "j": j, "lam": float(lam[j]),
                         "view_share": float(share[j]),
                         "d_loss": total(Lj) - loss0, "d_inv": Lj["inv"] - L0["inv"],
                         "d_theta_z": Lj["theta_z"] - L0["theta_z"],
                         "d_cond_z": Lj["cond_z"] - L0["cond_z"], "d_top1": top1_0 - float(acc),
                         "loss0": loss0, "inv0": L0["inv"], "theta_z0": L0["theta_z"],
                         "top1_0": top1_0})
            if j % 32 == 0 or j == d - 1:
                write_rows(f_dir, DIR_COLS, rows)
                print(f"[e35] direction {j:3d}: lam {lam[j]:.2e} d_theta_z "
                      f"{rows[-1]['d_theta_z']:+.5f} d_inv {rows[-1]['d_inv']:+.5f} "
                      f"d_top1 {rows[-1]['d_top1']:+.4f}", flush=True)
    write_rows(f_dir, DIR_COLS, rows)

    # (c) the consequence: delete cumulatively in the order the LOSS cares least about, and
    # retrain the probe. Control = a random order of the same directions.
    # Ordered by the change in THICKNESS, not in inv: inv is an absolute MSE at z, so deleting
    # any direction removes a non-negative contribution and inv falls by construction — that is
    # less signal, not better invariance (Berker 2026-09-10). Theta = W/B divides the scale out,
    # so a direction whose removal shrinks cloud and centers alike leaves it flat. Ascending =
    # the directions invariance is gladdest to lose go first.
    # Set B (Berker 2026-09-10, settled): directions whose removal leaves the invariance loss no
    # worse AND costs the probe more than the +-0.10-point floor — harmful-or-irrelevant to
    # invariance, useful for classification. The loss is already a mean over z's dimensions and z
    # keeps its 256 dims when an h direction goes, so no further normalisation enters. The whole
    # set is deleted in one go, because single-direction costs do not add.
    iv_tol, acc_tol = 0.01 * L0["inv"], 0.0010
    Bset = np.array([r["j"] for r in rows
                     if r["d_inv"] <= iv_tol and r["d_top1"] > acc_tol], dtype=int)
    orders = {"view_desc": np.argsort(-share),       # worst-for-invariance first
              "random": np.random.default_rng(cfg.order_seed).permutation(d),
              "set_B": np.concatenate([Bset, np.setdiff1d(np.arange(d), Bset)])}
    levels = {"set_B": [len(Bset)], "view_desc": [25, 50, 100, 192], "random": [25, 50, 100, 192]}
    print(f"[e35] set B: {len(Bset)} directions", flush=True)
    crows = []
    for name, order in orders.items():
        for m in levels.get(name, cfg.cum_levels):
            keep = np.setdiff1d(np.arange(d), order[:m], assume_unique=False)
            B = np.ascontiguousarray(U[:, keep])
            P = torch.tensor(B @ B.T, dtype=torch.float32, device=dev)
            L = loss_read(Hv, P, mu, head, dz, Qz, Qh, batches)
            pr = linear_raw_v2(ablate(Xtr_t, P, mu), ytr, ablate(Xva_t, P, mu), yva, ncls,
                               device=dev, seed=cfg.probe_seed)
            crows.append({"cell": cfg.cell, "order": name, "n_removed": m,
                          "frac_var_removed": 1 - var_kept(B, lam, U), "loss": total(L),
                          "inv": L["inv"], "cond_z": L["cond_z"], "theta_z": L["theta_z"],
                          "lin": pr["val_acc"], "lin_train": pr["train_acc"],
                          "lin_bestep": pr["best_ep"]})
            write_rows(f_cum, CUM_COLS, crows)
            print(f"[e35] {name:15} removed {m:3d}: loss {crows[-1]['loss']:.4f} "
                  f"top-1 {pr['val_acc']:.4f} ({crows[-1]['frac_var_removed']*100:.2f}% var)",
                  flush=True)
    print(f"[e35] -> {f_dir} | {f_cum}", flush=True)
    if cfg.figure:
        figure(f_dir, f_cum, cfg.cell)


@torch.no_grad()
def selftest(store, run_id, cfg, head_mod, dev, dz):
    """(1) the frozen head reproduces the stored z from the stored h (this whole script rests on
    it); (2) the readout conditioner IS the training term on a held slice."""
    from sslgap.methods._common import SACReg
    grab = lambda man, sp, n: torch.tensor(
        np.asarray(store.get(run_id, man, sp)[:n], dtype=np.float32), device=dev)
    h = grab(cfg.pairs_manifest, f"{cfg.space}.view0", 4096)
    z0 = grab(cfg.pairs_manifest, f"student.z.{cfg.z_tap}.view0", 4096)
    cos = torch.nn.functional.cosine_similarity(head_mod(h)["proj.out"], z0, dim=1)
    print(f"[e35][selftest] head parity vs stored z: median cos {cos.median():.6f} "
          f"min {cos.min():.6f}", flush=True)
    assert cos.median() > 0.999, "frozen-head recompute does not reproduce the stored z"
    x = torch.randn(512, dz) * torch.linspace(0.3, 2.0, dz) + 0.4
    torch.manual_seed(cfg.slice_seed)
    live = float(SACReg(d_slice=cfg.d_slice, d_draw=min(cfg.d_slice, dz))(x))
    mine = moment_kl_slice(x.numpy(), fixed_slice(dz, cfg.d_slice, min(cfg.d_slice, dz),
                                                  cfg.slice_seed))
    print(f"[e35][selftest] conditioner parity: live {live:.8f} readout {mine:.8f}", flush=True)
    assert abs(live - mine) / abs(live) < 1e-4, "readout conditioner != training term"
    # thickness identity: the in-loop W/B must equal sslgap.metrics.orbit_energy's, which owns
    # the definition (D-054). Checked on 2048 images, untruncated (P = I).
    from sslgap.metrics.orbit_energy import orbit_energies
    hv = [grab(cfg.cloud_manifest, f"{cfg.space}.view{v}", 2048) for v in range(cfg.V)]
    zv = [head_mod(hx)["proj.out"] for hx in hv]
    ref = orbit_energies([z.cpu().numpy() for z in zv])
    W = float(sum((zv[u] - zv[w]).square().sum().double()
                  for u in range(cfg.V) for w in range(u + 1, cfg.V))) / \
        (len(hv[0]) * cfg.V * (cfg.V - 1) / 2)
    mine_theta = thickness(W, torch.stack(zv).mean(0).cpu().numpy(), cfg.V)
    print(f"[e35][selftest] thickness parity at z: orbit_energies {ref['omega']:.6f} "
          f"in-loop {mine_theta:.6f}", flush=True)
    assert abs(ref["omega"] - mine_theta) / ref["omega"] < 1e-4, "thickness identity broken"


SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e6e5e1"
LOSSC, RANDC = "#16607d", "#c8792a"


def figure(f_dir, f_cum, cell):
    """Left: every direction of h as one point — how much the objective loses it (x) against how
    much the classifier loses it (y). Middle/right: delete cumulatively in the order the objective
    cares least about, and read both. No variance ordering anywhere."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    D = list(csv.DictReader(open(f_dir)))
    C = list(csv.DictReader(open(f_cum)))
    x = np.array([float(r["d_loss"]) for r in D])
    y = 100 * np.array([float(r["d_top1"]) for r in D])
    lam = np.array([float(r["lam"]) for r in D])

    fig, (a, b, c) = plt.subplots(1, 3, figsize=(13.2, 4.0), facecolor=SURFACE)
    sc = a.scatter(x, y, c=np.log10(np.clip(lam, 1e-12, None)), cmap="viridis", s=22,
                   edgecolor="none")
    a.set_xscale("symlog", linthresh=1e-3)     # loss changes span 1e-5..10: linear hides them
    a.axhline(0, color=GRID, lw=1)
    a.set_xlabel("change in the SSL loss when this direction is deleted", fontsize=9, color=INK2)
    a.set_ylabel("top-1 lost when this direction is deleted (points)", fontsize=9, color=INK2)
    a.set_title("one point per direction of h", fontsize=10.5, color=INK, loc="left")
    fig.colorbar(sc, ax=a, label="log10 variance of the direction")

    for name, col, lbl in (("view_desc", LOSSC, "deleted worst-for-invariance first"),
                           ("random", RANDC, "deleted in random order")):
        rs = sorted([r for r in C if r["order"] == name], key=lambda r: int(r["n_removed"]))
        xs = [int(r["n_removed"]) for r in rs]
        b.plot(xs, [float(r["loss"]) for r in rs], color=col, lw=2.2, marker="o", ms=4, label=lbl)
        c.plot(xs, [100 * float(r["lin"]) for r in rs], color=col, lw=2.2, marker="o", ms=4,
               label=lbl)
    b.set_title("SSL loss", fontsize=10.5, color=INK, loc="left")
    c.set_title("linear probe", fontsize=10.5, color=INK, loc="left")
    b.set_ylabel("loss", fontsize=9, color=INK2)
    c.set_ylabel("IN-100 top-1 (%)", fontsize=9, color=INK2)
    for ax in (b, c):
        ax.set_xlabel("directions removed (of 384)", fontsize=9, color=INK2)
        ax.legend(fontsize=8.5, frameon=False)
    for ax in (a, b, c):
        ax.set_facecolor(SURFACE)
        ax.tick_params(colors=INK2, labelsize=8.5)
        ax.grid(True, color=GRID, linewidth=0.7)
        ax.set_axisbelow(True)
        for sp in ax.spines.values():
            sp.set_visible(False)
    fig.tight_layout()
    out = figdir("e35") / "e35_directions.png"
    fig.savefig(out, dpi=170, facecolor=SURFACE)
    print(f"[e35] -> {out}", flush=True)


if __name__ == "__main__":
    main()
