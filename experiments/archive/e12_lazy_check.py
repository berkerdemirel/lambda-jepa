"""E12 lazy-projector forensic (Berker's 2026-07-11 concern): in A2/A3 as pre-registered, nothing
at proj.out forbids the constant-output solution — inv can reach 0 without aligning anything.
This script measures, at ep2-smoke checkpoints vs fresh init: across-image std of the view-mean
vs within-image across-view std, at proj.out AND embed, plus the inv value. Raw numbers only.
"""
import glob

import torch
from omegaconf import OmegaConf
from torch.utils.data import DataLoader

from sslgap.data import seed_everything
from sslgap.methods import METHODS
from sslgap.methods.base import Frame

BASE = {"name": "lejepa", "lamb": 0.02, "V": 4, "proj_dim": 16, "emb_dim": 512, "drop_path": 0.1,
        "lr": 3e-4, "wd": 5e-2, "warmup_ep": 10, "eta_min": 1e-5, "grad_clip": 1.0}
ARMS = {  # amended arms (E12 §Amendment): additive h-term, shipped SIGReg@proj retained
    "e12a2smoke2": {"spec_norm": True, "embed_calib": True, "h_reg": "sigreg", "h_lamb": 0.0257},
    "e12a3smoke2": {"spec_norm": True, "embed_calib": True, "h_reg": "moment", "h_lamb": 0.4775},
}


@torch.no_grad()
def stats(modules, views):
    N, V = views.shape[:2]
    emb = modules["encoder"](views.flatten(0, 1))
    proj = modules["projector"](emb).reshape(N, V, -1)
    out = {}
    for name, t in (("proj", proj), ("embed", emb.reshape(N, V, -1))):
        vm = t.mean(1)                                   # view-mean per image [N, D]
        out[f"{name}_across_img_std"] = vm.std(0).mean().item()
        out[f"{name}_within_img_std"] = t.std(1).mean().item()
    pv = proj.transpose(0, 1)
    out["inv"] = (pv.mean(0) - pv).square().mean().item()
    return out


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    frame = Frame(name="in100", model_name="vit_small_patch16_224", img_size=224,
                  dataset="imagenet100", data_root="~/data/imagenet100", epochs=2, seed=0,
                  grad_clip=1.0, num_workers=0, device=device)
    for tag, over in ARMS.items():
        cfg = OmegaConf.create({**BASE, **over})
        seed_everything(0)
        method = METHODS["lejepa"](cfg, frame)
        modules = method.build_modules().to(device)
        ds = method.build_train_dataset()
        loader = DataLoader(ds, batch_size=32, shuffle=True, drop_last=True, num_workers=0,
                            generator=torch.Generator().manual_seed(0))
        views, _ = next(iter(loader))
        views = views.to(device)
        for m in modules.values():
            m.eval()
        print(f"[lazy_check] {tag} INIT   {stats(modules, views)}", flush=True)
        ck = glob.glob(f"/nfs/scistore19/locatgrp/bdemirel/ssl_project/outputs/in100.lejepa.s0.{tag}_last.pt")
        pay = torch.load(ck[0], map_location=device, weights_only=False)
        for role, sd in pay["modules"].items():
            if role != "probe":
                modules[role].load_state_dict(sd)
        print(f"[lazy_check] {tag} EP2    {stats(modules, views)}", flush=True)


if __name__ == "__main__":
    main()
