"""SIGReg value calibration (Berker, 2026-07-09): what does the training statistic read for
(a) true N(0,I) [analytic floor 1.053, MC-verified], (b) a COVARIANCE-MATCHED Gaussian twin of
the arm's constrained space, (c) the arm itself — under training conditions (aug views, V=4,
N=256/view, fresh slices)? The (c)−(b) gap is the shape/degeneracy residue the moment channel
cannot explain; (b)−floor is pure second-moment mismatch. Cells: A@proj16 (healthy), Blr@embed
(best pre-storm state), D0@embed (post-storm end), Dr@embed (variance-collapsed end).
Output: results/diag/sigreg_ref.csv.
"""
import csv
import os

import torch

from sslgap.ckpt.adapters import _resolve
from sslgap.data import ViewsDataset
from sslgap.methods.lejepa import SIGReg

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CELLS = [("A", "toy.lejepa.s0.e10A_ep150.pt", "proj"),
         # A's embed = the same trained net read at the UNCONSTRAINED 512-d space — the twin cell
         # to Dlr@embed for "what does free vs constrained embed geometry cost in sigreg terms"
         ("A_embed", "toy.lejepa.s0.e10A_ep150.pt", "embed"),
         ("Blr_best", "toy.lejepa.s0.e10Blr_best.pt", "embed"),
         ("D0", "toy.lejepa.s0.e10D0_ep150.pt", "embed"),
         ("Dr", "toy.lejepa.s0.e10Dr_ep150.pt", "embed"),
         # Dlr moment-part trajectory (the pre-registered discriminating quantity, card §e10Dlr)
         ("Dlr_ep38", "toy.lejepa.s0.e10Dlr_ep38.pt", "embed"),
         ("Dlr_ep75", "toy.lejepa.s0.e10Dlr_ep75.pt", "embed"),
         ("Dlr_ep112", "toy.lejepa.s0.e10Dlr_ep112.pt", "embed"),
         ("Dlr_ep150", "toy.lejepa.s0.e10Dlr_ep150.pt", "embed")]
BATCHES, BS, REPS = 8, 256, 5


def rebuild(ck):
    mods = {}
    for role, spec in ck["arch"].items():
        if role == "probe":
            continue
        m = _resolve(spec["class"])(**spec["kwargs"])
        m.load_state_dict(ck["modules"][role])
        mods[role] = m.cuda().train()          # train mode = training-condition statistics
    return mods


def main():
    torch.manual_seed(0)
    ds = ViewsDataset("imagenette", "train", V=4, img_size=128)
    loader = torch.utils.data.DataLoader(ds, batch_size=BS, shuffle=True, num_workers=8)
    batches = []
    for v, _ in loader:
        batches.append(v.cuda())
        if len(batches) == BATCHES:
            break
    sigreg = SIGReg().cuda()
    rows = []
    for tag, ckpt, space in CELLS:
        ck = torch.load(f"{ROOT}/outputs/{ckpt}", map_location="cpu", weights_only=False)
        mods = rebuild(ck)
        zs = []
        with torch.no_grad():
            for v in batches:
                emb = mods["encoder"](v.flatten(0, 1))
                z = mods["projector"](emb) if space == "proj" else emb
                zs.append(z.float())
        Z = torch.cat(zs)                                       # [BATCHES*BS*V, K]
        K = Z.shape[1]
        mu = Z.mean(0)
        cov = torch.cov(Z.T)
        # eigh sampling, not Cholesky: collapsed arms have genuinely rank-deficient covariance,
        # and their Gaussian twin must be equally degenerate (that IS the moment match)
        evals, evecs = torch.linalg.eigh(cov)
        A_half = evecs @ torch.diag(evals.clamp_min(0).sqrt())

        def T(x):     # per-view statistic at N=BS (matches training logs): reshape to [V,N,K]
            return float(sigreg(x.reshape(-1, BS, K)))

        t_act = sum(T(Z[i * BS * 4:(i + 1) * BS * 4]) for i in range(BATCHES)) / BATCHES
        t_gau = 0.0
        for _ in range(REPS):
            G = mu + torch.randn(BS * 4, K, device=Z.device) @ A_half.T
            t_gau += T(G) / REPS
        rows.append({"cell": tag, "space": space, "K": K,
                     "T_actual": round(t_act, 3), "T_cov_matched_gauss": round(t_gau, 3),
                     "floor_N01": 1.053,
                     "moment_part": round(t_gau - 1.053, 3),
                     "shape_degen_residue": round(t_act - t_gau, 3)})
        print("[sigref]", rows[-1], flush=True)
        del mods
        torch.cuda.empty_cache()
    out = os.path.join(ROOT, "results", "diag", "sigreg_ref.csv")
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print("[sigref] wrote", out)


if __name__ == "__main__":
    main()
