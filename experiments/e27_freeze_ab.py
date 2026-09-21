"""Does freeze_prebn change the forward pass? (Berker 2026-09-11.)

The freeze-now branch resumed the ep304 state and read inv ~15 against a running mean of 0.367 at
its second step, while the run that produced that checkpoint read ~0.34. requires_grad_(False)
cannot change a forward pass, so either the weights themselves produce 15 (and the old run's 0.34
came from state not in the checkpoint), or the live run's resume differs from a plain load.

One fixed minibatch, one checkpoint, no optimizer step:
  A  load, forward                                  (freeze off)
  B  same modules, requires_grad_(False) on W0/W3, forward again
  C  fresh load, freeze applied before the load, forward
inv is measured exactly as train_ddp does: against the EMA twin's view-mean, scaled 2V/(V-1).

Usage: python experiments/e27_freeze_ab.py <ckpt> [batches]
"""
import sys
import torch
from omegaconf import OmegaConf
from torch.utils.data import DataLoader
sys.path.insert(0, "/nfs/scistore19/locatgrp/bdemirel/ssl_project")
from sslgap.methods import METHODS
from sslgap.methods.base import Frame
from sslgap.ckpt.schema import load_payload

CKPT = sys.argv[1]; NB = int(sys.argv[2]) if len(sys.argv) > 2 else 3
dev = "cuda"
pay = load_payload(CKPT, map_location=dev)
cfg = OmegaConf.create(pay["cfg"])
frame = Frame(name=cfg.frame.name, model_name=cfg.frame.model_name, img_size=cfg.frame.img_size,
              dataset=cfg.frame.dataset, data_root=cfg.frame.get("data_root"), epochs=cfg.frame.epochs,
              seed=cfg.seed, grad_clip=cfg.method.get("grad_clip", cfg.frame.grad_clip),
              num_workers=8, device=dev)
print(f"[ab] {CKPT} epoch {pay['epoch']} | modules {sorted(pay['modules'])} | extras {pay.get('extras')}")

def build(freeze_before_load):
    torch.manual_seed(cfg.seed)
    m = METHODS[cfg.method.name](cfg.method, frame)
    mods = m.build_modules().to(dev)
    if freeze_before_load:
        for i in (0, 3):
            for par in mods["projector"][i].parameters():
                par.requires_grad_(False)
    for role, sd in pay["modules"].items():
        if role != "probe":
            mods[role].load_state_dict(sd)
    m.load_extras(pay.get("extras", {}))
    mods.train()
    return m, mods

m0, mods0 = build(False)
ds = m0.build_train_dataset()
dl = DataLoader(ds, batch_size=cfg.bs, shuffle=False, drop_last=True, num_workers=8)

def fwd(m, mods, views):
    with torch.no_grad():
        N, V = views.shape[:2]
        cls = mods["backbone"].forward_features(views.flatten(0, 1))[:, 0]
        z = mods["projector"](cls).reshape(N, V, -1)
        anchor = m._swa_mu(mods, views)
        inv = (z - anchor).square().mean() * (2 * V / (V - 1))
        return float(inv), z.float(), anchor.float()

it = iter(dl)
for b in range(NB):
    views = next(it); views = views.to(dev) if torch.is_tensor(views) else views[0].to(dev)
    invA, zA, aA = fwd(m0, mods0, views)                       # A: freeze off
    for i in (0, 3):                                           # B: toggle the freeze on the SAME modules
        for par in mods0["projector"][i].parameters():
            par.requires_grad_(False)
    invB, zB, aB = fwd(m0, mods0, views)
    m2, mods2 = build(True)                                    # C: freeze applied before the load
    invC, zC, aC = fwd(m2, mods2, views)
    print(f"\nbatch {b}: inv  A(no freeze) {invA:.6f}   B(toggled) {invB:.6f}   C(fresh+freeze) {invC:.6f}")
    print(f"   max|zA-zB| {float((zA - zB).abs().max()):.3e}   max|zA-zC| {float((zA - zC).abs().max()):.3e}")
    print(f"   anchor: max|aA-aB| {float((aA - aB).abs().max()):.3e}   max|aA-aC| {float((aA - aC).abs().max()):.3e}")
    print(f"   z per-dim var {float(zA.reshape(-1, zA.size(-1)).var(0).mean()):.4f} | "
          f"anchor per-dim var {float(aA.reshape(-1, aA.size(-1)).var(0).mean()):.4f}")
    for i in (0, 3):                                           # restore for the next batch
        for par in mods0["projector"][i].parameters():
            par.requires_grad_(True)
