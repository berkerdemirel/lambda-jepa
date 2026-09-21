"""E35 follow-up: split h into the part the projector passes to z and the part it kills, and
read the classifier on each half.

The objective only ever sees z. So whatever the projector does not transmit is content the loss
cannot ask for — no threshold of ours enters, the split is the head's own gain. Fit the head's
linear part on the clean train features (OLS h -> z), take its singular directions in h-space, and
probe the top-k transmitted subspace against its complement. The claim under test (Berker
2026-09-10): the treated representation keeps content the objective never required, and that
content is useful for classification — i.e. its killed half decodes well and the untreated one's
does not.

Caveat quoted with every row: the head is nonlinear, so this split is a linear approximation; the
unexplained share of z (resid_frac, D-060's own caveat) says how much of the head this misses.

  sbatch slurm/e35_transmit.sbatch cell=in100.floorssl.s0.d256vm4
"""
import csv
import os

import hydra
import numpy as np
import torch
from omegaconf import DictConfig

from sslgap.ckpt import adapters
from sslgap.extract import FeatureStore
from sslgap.metrics.cross import linear_map_fit
from sslgap.paths import DIAG
from sslgap.probes import linear_raw_v2

COLS = ["cell", "half", "k", "d_kept", "var_share", "lin", "lin_train", "top1_full",
        "r2_total", "resid_frac", "effrank_map", "n_train", "n_val"]


@hydra.main(version_base=None, config_path="configs", config_name="e35_transmit")
def main(cfg: DictConfig):
    dev = cfg.device if torch.cuda.is_available() else "cpu"
    run_id = cfg.run_id or f"{cfg.cell}.extL"
    ckpt = cfg.ckpt or os.path.join(cfg.ckpt_dir, f"{cfg.cell}_ep{cfg.epoch}.pt")
    store = FeatureStore(cfg.store_root)
    lc = adapters.load("native", ckpt, "e35t")
    head = lc.branches[lc.probed_branch].heads.to(dev).eval()

    Htr = np.asarray(store.get(run_id, cfg.fit_manifest, cfg.space), dtype=np.float32)
    Hva = np.asarray(store.get(run_id, cfg.val_manifest, cfg.space), dtype=np.float32)
    ytr, yva = store.labels(run_id, cfg.fit_manifest), store.labels(run_id, cfg.val_manifest)
    ncls = int(max(ytr.max(), yva.max())) + 1
    Htr_t, Hva_t = torch.tensor(Htr, device=dev), torch.tensor(Hva, device=dev)
    with torch.no_grad():                      # z of the SAME clean features the probe reads
        zt = lambda X: torch.cat([head(X[i:i + 8192])["proj.out"] for i in range(0, len(X), 8192)])
        Ztr, Zva = zt(Htr_t).cpu().numpy(), zt(Hva_t).cpu().numpy()

    # the head's linear part, fitted on train and scored on val (quality numbers from the house
    # metric; the map itself is refit here because linear_map_fit returns only its spectrum)
    q = linear_map_fit(Htr, Ztr, Hva, Zva)
    mu, zmu = Htr.mean(0), Ztr.mean(0)
    W = np.linalg.lstsq(Htr - mu, Ztr - zmu, rcond=None)[0]        # [384, 256]
    Uh, sv, _ = np.linalg.svd(W, full_matrices=True)               # Uh: h-space directions
    lam = np.linalg.eigvalsh(np.cov(Htr - mu, rowvar=False))[::-1]
    print(f"[e35t] {cfg.cell}: r2_total {q['r2_total']:.4f} resid_frac {q['resid_frac']:.4f} "
          f"effrank_map {q['effrank_map']:.1f} | gains: max {sv[0]:.3g} median {np.median(sv):.3g} "
          f"min {sv.min():.3g}", flush=True)

    base = linear_raw_v2(Htr, ytr, Hva, yva, ncls, device=dev, seed=cfg.probe_seed)["val_acc"]
    mu_t = torch.tensor(mu, device=dev)
    rows = []
    for k in cfg.k_grid:
        for half, cols in (("transmitted", Uh[:, :k]), ("killed", Uh[:, k:])):
            B = torch.tensor(np.ascontiguousarray(cols), dtype=torch.float32, device=dev)
            P = B @ B.T
            ab = lambda X: (mu_t + (X - mu_t) @ P).cpu().numpy()
            r = linear_raw_v2(ab(Htr_t), ytr, ab(Hva_t), yva, ncls, device=dev, seed=cfg.probe_seed)
            cov = np.cov(Htr - mu, rowvar=False)
            rows.append({"cell": cfg.cell, "half": half, "k": k, "d_kept": cols.shape[1],
                         "var_share": float(np.trace(cols.T @ cov @ cols) / lam.sum()),
                         "lin": r["val_acc"], "lin_train": r["train_acc"], "top1_full": base,
                         "r2_total": q["r2_total"], "resid_frac": q["resid_frac"],
                         "effrank_map": q["effrank_map"], "n_train": len(ytr), "n_val": len(yva)})
            print(f"[e35t] k={k:3d} {half:12} dims {cols.shape[1]:3d} "
                  f"var {rows[-1]['var_share']*100:5.1f}%  top-1 {r['val_acc']:.4f} "
                  f"(full {base:.4f})", flush=True)
            with open(DIAG / f"e35_transmit.{cfg.cell}.csv", "w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=COLS)
                w.writeheader()
                w.writerows(rows)
    print(f"[e35t] -> {DIAG / f'e35_transmit.{cfg.cell}.csv'}", flush=True)


if __name__ == "__main__":
    main()
