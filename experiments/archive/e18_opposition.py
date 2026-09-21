"""E18 deeper analysis (E18-T1's OWED mechanism: why do h-CF-shape terms impede inv?) —
the opposition read turned WITHIN the lejepa lane: at matched checkpoint states, the cosine
between each candidate h-term's trunk-gradient and the lane's inv trunk-gradient, for THREE
functionals through the method's own code paths (cfg.h_reg override, e20_opposition trick):
  moment   — the floor (the winning E20/f2 term)
  sigreg   — E17's Gaussian-CF shape term
  sigreg_t — E18's declared-prior t_8.2 CF shape term
States: the four arm lineages (sigreg_t, sigreg3, e20f, f2) + the control, at ep25
(formation) and ep100 (converged). Same seed-0 first batch everywhere; norms unweighted
(e12h_pull convention). Rows -> results/diag/e18_opposition.csv. RAW; no takeaway."""
import csv
import os

import torch
from omegaconf import OmegaConf
from torch.amp import autocast
from torch.utils.data import DataLoader

from sslgap.data import seed_everything
from sslgap.methods import METHODS
from sslgap.methods.base import Frame
from sslgap.methods.lejepa import H_KEYS

ROOT = "/nfs/scistore19/locatgrp/bdemirel/ssl_project"
OUT = f"{ROOT}/results/diag/e18_opposition.csv"
STATES = [(run, ep) for run in
          ["in100.lejepa.s0.hpull_sigreg_t", "in100.lejepa.s0.hpull_sigreg3",
           "in100.lejepa.s0.e20f", "in100.lejepa.s0.e12f2", "in100.lejepa.s0"]
          for ep in ["ep25", "ep100"]]
FUNCTIONALS = ["sacreg", "sigreg", "sigreg_t"]
NU = 8.2                       # D-041's declared prior (Varimax-recipe fit on the lane)


def flat(grads):
    return torch.cat([g.reshape(-1).float() for g in grads if g is not None])


def main():
    dev = "cuda"
    rows = []
    for run, ck in STATES:
        pay = torch.load(f"{ROOT}/outputs/{run}_{ck}.pt", map_location="cpu",
                         weights_only=False)
        for freg in FUNCTIONALS:
            cfg = OmegaConf.create({**pay["cfg"]["method"], "h_reg": freg, "h_lamb": 1.0,
                                    "sigreg_nu": NU})
            fr = pay["cfg"]["frame"]
            frame = Frame(name=fr["name"], model_name=fr["model_name"],
                          img_size=fr["img_size"], dataset=fr["dataset"],
                          data_root=fr["data_root"], epochs=fr["epochs"], seed=0,
                          grad_clip=1.0, num_workers=0, device=dev)
            seed_everything(0)
            method = METHODS["lejepa"](cfg, frame)
            modules = method.build_modules().to(dev)
            for role, sd in pay["modules"].items():
                if role != "probe":
                    modules[role].load_state_dict(sd)
            method.load_extras(pay.get("extras", {}))
            loader = DataLoader(method.build_train_dataset(), batch_size=pay["cfg"]["bs"],
                                shuffle=True, drop_last=True, num_workers=0,
                                generator=torch.Generator().manual_seed(0))
            views, y = next(iter(loader))
            params = [p for p in modules["encoder"].parameters() if p.requires_grad]
            with autocast(dev, dtype=torch.bfloat16):
                terms, _, _ = method.training_step(modules, views.to(dev), dev, y=y.to(dev))
            g = {}
            for key, keep in [("inv", True), ("sigreg", True), (H_KEYS[freg], False)]:
                g[key] = flat(torch.autograd.grad(terms[key], params, retain_graph=keep,
                                                  allow_unused=True))
            cs = lambda a, b: float((g[a] @ g[b]) / (g[a].norm() * g[b].norm()))
            hk = H_KEYS[freg]
            rows.append({"run": run, "ep": pay["epoch"] + 1, "functional": freg,
                         "cos_h_inv": round(cs(hk, "inv"), 5),
                         "cos_h_zreg": round(cs(hk, "sigreg"), 5),
                         "cos_inv_zreg": round(cs("inv", "sigreg"), 5),
                         "g_h": round(float(g[hk].norm()), 5),
                         "g_inv": round(float(g["inv"].norm()), 5),
                         "g_zreg": round(float(g["sigreg"].norm()), 5),
                         "slurm_job": os.environ.get("SLURM_JOB_ID", "")})
            print(f"{run:36s} ep{pay['epoch']+1:<4d} {freg:9s} cos(h,inv)={cs(hk,'inv'):+.4f} "
                  f"cos(h,zreg)={cs(hk,'sigreg'):+.4f} |g_h|={g[hk].norm():.3f} "
                  f"|g_inv|={g['inv'].norm():.3f}", flush=True)
    new = not os.path.exists(OUT)
    with open(OUT, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        if new:
            w.writeheader()
        w.writerows(rows)
    print("appended", OUT, flush=True)


if __name__ == "__main__":
    main()
