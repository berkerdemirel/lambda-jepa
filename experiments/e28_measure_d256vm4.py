"""Standalone, read-only measurement of the landed in100.floorssl.s0.d256vm4 checkpoints (the
h-regularized twin of d256vm4zonly: identical recipe, h_lamb=1.894 instead of 0.0) — the "with
regularizer on h" comparison point for the e28 scale sweep.

Does NOT modify experiments/e28_scaleinit.py or anything under sslgap/ (per e28's own
standalone convention) — it only imports spectrum_row/measure_spaces-equivalent logic. Uses
plain stock FloorSSL (not E28FloorSSL, which hard-asserts h_lamb==0.0) since this comparison
has nothing to do with the LN-relocation/scale machinery: h here is the ORDINARY (post-LN)
representation, comparable to h_ln in the e28x*_cdominetest diagnostics.

Output: results/e28/in100.floorssl.s0.d256vm4_cdominetest_diag.csv (own file, no collision
with anything landed or with any other run's outputs).
"""
import os

import torch
from omegaconf import OmegaConf
from torch.utils.data import DataLoader

from sslgap.data import ViewsDataset, seed_everything, seed_worker
from sslgap.methods.base import Frame
from sslgap.methods.floorssl import FloorSSL

from experiments.e28_scaleinit import append_csv, spectrum_row

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RUN = "d256vm4"
OUT_RUN_ID = "in100.floorssl.s0.d256vm4_cdominetest"
OUT = os.path.join(ROOT, "results/e28")


@torch.no_grad()
def measure(modules, loader, device):
    was = {k: m.training for k, m in modules.items()}
    for m in modules.values():
        m.eval()
    hs, zs = [], []
    for views, _ in loader:
        x = views.to(device, non_blocking=True).flatten(0, 1)
        h = modules["backbone"].forward_features(x)[:, 0].float()
        hs.append(h.cpu())
        zs.append(modules["projector"](h).float().cpu())
    for k, m in modules.items():
        m.train(was[k])
    return {"h": torch.cat(hs).numpy(), "z": torch.cat(zs).numpy()}


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # ep0 skipped: h_lamb doesn't affect init, so it's identical to d256vm4zonly's own ep0
    # (h_ln effrank=6.95, z effrank=14.61 — already measured).
    diag_path = f"{OUT}/{OUT_RUN_ID}_diag.csv"
    done = set()
    if os.path.exists(diag_path):
        import csv
        done = {int(r["epoch"]) for r in csv.DictReader(open(diag_path))}
    rows = []
    for ep in (25, 50, 75, 100):
        if ep in done:
            print(f"[d256vm4] ep{ep}: already in {diag_path} — skipped", flush=True)
            continue
        ckpt = os.path.join(ROOT, f"outputs/in100.floorssl.s0.{RUN}_ep{ep}.pt")
        if not os.path.exists(ckpt):
            print(f"[d256vm4] ep{ep}: {ckpt} absent — skipped", flush=True)
            continue
        pay = torch.load(ckpt, map_location="cpu", weights_only=False)
        cfg = OmegaConf.create(pay["cfg"])
        seed_everything(cfg.seed)
        frame = Frame(name=cfg.frame.name, model_name=cfg.frame.model_name,
                      img_size=cfg.frame.img_size, dataset=cfg.frame.dataset,
                      data_root=cfg.frame.get("data_root"), epochs=cfg.frame.epochs,
                      seed=cfg.seed, grad_clip=cfg.method.get("grad_clip", cfg.frame.grad_clip),
                      num_workers=8, device=device)
        method = FloorSSL(cfg.method, frame)
        modules = method.build_modules()
        modules["backbone"].load_state_dict(pay["modules"]["backbone"])
        modules["projector"].load_state_dict(pay["modules"]["projector"])
        modules = {k: v.to(device) for k, v in modules.items()}

        val_ds = ViewsDataset(frame.dataset, "validation", V=1, img_size=frame.img_size,
                              data_root=frame.data_root)
        val = DataLoader(val_ds, batch_size=256, num_workers=8, worker_init_fn=seed_worker,
                         pin_memory=True)
        feats = measure(modules, val, device)
        for space, X in feats.items():
            row, _ = spectrum_row(X)
            rows.append({"run_id": OUT_RUN_ID, "epoch": ep, "step": 0, "space": space, **row})
        print(f"[d256vm4] ep{ep}: h effrank={rows[-2]['effrank']:.2f} "
              f"z effrank={rows[-1]['effrank']:.2f} best_acc={pay.get('best_acc')}", flush=True)
        append_csv(f"{OUT}/{OUT_RUN_ID}_diag.csv", rows[-2:])  # write per-epoch: interruption-safe


if __name__ == "__main__":
    main()
