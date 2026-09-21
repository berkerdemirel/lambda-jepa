"""D-073 pre-certification — does the OAS-fresh force preserve the ring's at-scale
behavior? Same weights, same batch, CRN slice frames (the e21_vmcos discipline): on a
vm4 formation ckpt, measure the z-floor trunk gradient under (a) the cell's OWN ring
estimator with the ring WARMED by queue_steps prior batches (the operating force,
D-072) and (b) the OAS-fresh estimator (queue 0, floor_shrink=oas) on the identical
batch. cos(a, b) ≈ 1 at in100/in1k ⇒ OAS does not erase real large-frame anisotropy
(Berker's Frobenius-vs-Stein caveat, tested before any at-scale training); the rho
columns record the evidence level. Controls: inv/h are estimator-independent terms —
their cross-form cosines must be 1.000 (torch.manual_seed pins slice frames AND
drop_path draws per form). f64 dots. Rows -> results/diag/e24_oas_cmp.csv. RAW.
Usage: python e24_oas_cmp.py <label> <ckpt> [...]"""
import csv
import os
import sys

import torch
from omegaconf import OmegaConf
from torch.amp import autocast
from torch.utils.data import DataLoader

from sslgap.data import seed_everything
from sslgap.methods import METHODS
from sslgap.methods.base import Frame
from sslgap.ckpt.schema import load_payload

ROOT = "/nfs/scistore19/locatgrp/bdemirel/ssl_project"
OUT = f"{ROOT}/results/diag/e24_oas_cmp.csv"
REPEATS = 3


def cos(a, b):
    return float((a.double() @ b.double()) / (a.double().norm() * b.double().norm() + 1e-12))


def grads(method, modules, params, views, y, dev):
    torch.manual_seed(4242)
    with autocast(dev, dtype=torch.bfloat16):
        terms, _, _ = method.training_step(modules, views.to(dev), dev, y=y.to(dev))
    names = [k for k in terms if k != "loss"]
    G = {}
    for i, k in enumerate(names):
        g = torch.autograd.grad(terms[k], params, retain_graph=i < len(names) - 1,
                                allow_unused=True)
        G[k] = torch.cat([(x if x is not None else torch.zeros_like(pp)).reshape(-1)
                          for x, pp in zip(g, params)]).float()
    return G


def measure(label, ck_path, dev="cuda"):
    base = load_payload(ck_path, map_location="cpu")
    mcfg = dict(base["cfg"]["method"])
    fr = base["cfg"]["frame"]
    frame = Frame(name=fr["name"], model_name=fr["model_name"], img_size=fr["img_size"],
                  dataset=fr["dataset"], data_root=fr["data_root"], epochs=fr["epochs"],
                  seed=0, grad_clip=1.0, num_workers=0, device=dev)
    seed_everything(0)
    m_ring = METHODS["lambdajepa"](OmegaConf.create(mcfg), frame)
    modules = m_ring.build_modules().to(dev)
    m_oas = METHODS["lambdajepa"](OmegaConf.create(
        {**mcfg, "queue_steps": 0, "floor_shrink": "oas"}), frame)
    m_oas.build_modules()          # builds its own conditioners; steps run on `modules`
    for role, sd in base["modules"].items():
        if role != "probe":
            modules[role].load_state_dict(sd)
    m_ring.load_extras(base.get("extras", {}))
    loader = DataLoader(m_ring.build_train_dataset(), batch_size=128, shuffle=True,
                        drop_last=True, num_workers=0,
                        generator=torch.Generator().manual_seed(0))
    params = [p for p in modules["backbone"].parameters() if p.requires_grad]
    q = int(mcfg.get("queue_steps", 0) or 0)
    it = iter(loader)
    for _ in range(q):                      # warm the ring (D-072) before any read
        views, y = next(it)
        with torch.no_grad(), autocast(dev, dtype=torch.bfloat16):
            m_ring.training_step(modules, views.to(dev), dev, y=y.to(dev))
    ep, rows = base.get("epoch", -1), []
    for b in range(REPEATS):
        views, y = next(it)
        Gr = grads(m_ring, modules, params, views, y, dev)
        Go = grads(m_oas, modules, params, views, y, dev)
        ctrl = cos(Gr["inv"], Go["inv"])
        row = {"label": label, "ckpt": os.path.basename(ck_path), "ep": ep, "batch": b,
               "qfill": len(getattr(m_ring, "_zq", [])),
               "rho_z": round(m_oas.cond_z.rho_last, 4),
               "rho_h": round(m_oas.cond_h.rho_last, 4),
               "cos_z_ring_oas": round(cos(Gr["moment_kl"], Go["moment_kl"]), 4),
               "cos_h_ring_oas": round(cos(Gr["h_moment_kl"], Go["h_moment_kl"]), 4),
               "g_z_ring": round(Gr["moment_kl"].norm().item(), 6),
               "g_z_oas": round(Go["moment_kl"].norm().item(), 6),
               "g_inv": round(Go["inv"].norm().item(), 6),
               "g_h_ring": round(Gr["h_moment_kl"].norm().item(), 6),
               "g_h_oas": round(Go["h_moment_kl"].norm().item(), 6),
               "cos_inv_z_ring": round(cos(Gr["inv"], Gr["moment_kl"]), 4),
               "cos_inv_z_oas": round(cos(Go["inv"], Go["moment_kl"]), 4),
               "ctrl_inv": round(ctrl, 4),
               "slurm_job": os.environ.get("SLURM_JOB_ID", "")}
        rows.append(row)
        print(f"{label:10s} b{b} ctrl_inv={ctrl:+.4f} "
              f"cos_z(ring,oas)={row['cos_z_ring_oas']:+.4f} "
              f"g_z ring={row['g_z_ring']:.4f} oas={row['g_z_oas']:.4f} "
              f"rho_z={row['rho_z']:.3f} rho_h={row['rho_h']:.3f} | "
              f"cos(inv,z) ring={row['cos_inv_z_ring']:+.3f} "
              f"oas={row['cos_inv_z_oas']:+.3f}", flush=True)
    return rows


def main():
    args = sys.argv[1:]
    assert args and len(args) % 2 == 0, "pairs: <label> <ckpt>"
    rows = []
    for i in range(0, len(args), 2):
        rows += measure(args[i], args[i + 1])
    write_header = not os.path.exists(OUT)
    with open(OUT, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        if write_header:
            w.writeheader()
        w.writerows(rows)
    print("appended", OUT, flush=True)


if __name__ == "__main__":
    main()
