"""A/B replay from the untouched ep304 state: does freezing W0/W3 cause the invariance blow-up?
(Berker 2026-09-11.)

Settled already: freezing is forward-function preserving at resume (e27_freeze_ab.py: identical z
and inv to six decimals), and the 14-hour lock of the live freeze-now run was a guard ratchet —
eviction cut the conditioner's ring from 512 rows to a permanent 384 while the trip threshold only
updates on accepted steps, so the trip condition could never clear. Everything after that loop
began is discarded as evidence.

What is NOT settled: the live run took 16 healthy accepted updates and then read inv 15.66. This
replays both systems from the same untouched checkpoint, with the same data order and the SAME
guard (the original four-step skip; NO eviction, estimator cardinality never changes), and records
every step:
    A  W0/W3 frozen
    B  W0/W3 trainable, exactly as before
If A blows up near the same point while B stays healthy, freezing the already-aligned ep304
projector is dynamically unstable despite being function-preserving. If neither does, the live
failure was the guard. If both do, the initiating event was not the freeze.

Usage: python experiments/e27_freeze_replay.py <ckpt> <out csv> [steps]
"""
import sys, copy
import torch, torch.nn as nn
from omegaconf import OmegaConf
from torch.utils.data import DataLoader
from torch.amp import GradScaler
sys.path.insert(0, "/nfs/scistore19/locatgrp/bdemirel/ssl_project")
from sslgap.methods import METHODS
from sslgap.methods.base import Frame
from sslgap.ckpt.schema import load_payload
import importlib.util
spec = importlib.util.spec_from_file_location("tddp", "/nfs/scistore19/locatgrp/bdemirel/ssl_project/experiments/train_ddp.py")
tddp = importlib.util.module_from_spec(spec); spec.loader.exec_module(tddp)

CKPT, OUT = sys.argv[1], sys.argv[2]
STEPS = int(sys.argv[3]) if len(sys.argv) > 3 else 1500
dev = "cuda"
pay = load_payload(CKPT, map_location=dev)
cfg = OmegaConf.create(pay["cfg"])
frame = Frame(name=cfg.frame.name, model_name=cfg.frame.model_name, img_size=cfg.frame.img_size,
              dataset=cfg.frame.dataset, data_root=cfg.frame.get("data_root"), epochs=cfg.frame.epochs,
              seed=cfg.seed, grad_clip=cfg.method.get("grad_clip", cfg.frame.grad_clip),
              num_workers=12, device=dev)
w_inv, w_floor, h_lamb = float(cfg.method.w_inv), float(cfg.method.w_floor), float(cfg.method.h_lamb)
print(f"[replay] {CKPT} epoch {pay['epoch']} | steps {STEPS} per branch | "
      f"guard inv {cfg.get('skip_inv_ratio')} zkl {cfg.get('skip_zkl_ratio')} | NO eviction", flush=True)

def branch(freeze, rows):
    torch.manual_seed(cfg.seed)
    m = METHODS[cfg.method.name](cfg.method, frame)
    mods = m.build_modules().to(dev)
    if freeze:
        for i in (0, 3):
            for par in mods["projector"][i].parameters():
                par.requires_grad_(False)
    probe = nn.Sequential(nn.LayerNorm(m.probe_dim()), nn.Linear(m.probe_dim(), cfg.num_classes)).to(dev)
    opt = torch.optim.AdamW(m.param_groups(mods) + [{"params": probe.parameters(),
                                                     "lr": cfg.probe_lr, "weight_decay": cfg.probe_wd}])
    for role, sd in pay["modules"].items():
        (probe if role == "probe" else mods[role]).load_state_dict(sd)
    opt.load_state_dict(pay["optim"]["opt"])
    for g, ref in zip(opt.param_groups, m.param_groups(mods) + [{"weight_decay": cfg.probe_wd}]):
        g["weight_decay"] = ref["weight_decay"]
    m.load_extras(pay.get("extras", {}))
    scaler = GradScaler(dev); scaler.load_state_dict(pay["optim"]["scaler"])
    mods.train()
    student = tddp._StudentFwd(mods["backbone"], mods["projector"])
    g = torch.Generator().manual_seed(cfg.seed)                       # identical data order per branch
    dl = DataLoader(m.build_train_dataset(), batch_size=cfg.bs, shuffle=True, drop_last=True,
                    num_workers=12, generator=g, pin_memory=True, persistent_workers=True)
    inv_med = zkl_med = None
    tag = "A_frozen" if freeze else "B_trainable"
    for s, batch in enumerate(dl):
        if s >= STEPS: break
        views = batch.to(dev) if torch.is_tensor(batch) else batch[0].to(dev)
        with torch.autocast(dev, dtype=torch.bfloat16):
            terms, _, _ = tddp._ddp_loss(m, student, mods, views, w_inv, w_floor, h_lamb, s)
        opt.zero_grad(set_to_none=True)
        scaler.scale(terms["loss"]).backward()
        scaler.unscale_(opt)
        gn = torch.nn.utils.clip_grad_norm_([p for gr in opt.param_groups for p in gr["params"]],
                                            frame.grad_clip or float("inf")).item()
        # the ORIGINAL guard: discard the update, never touch the ring, threshold from kept steps only
        skip = ((inv_med is not None and terms["inv"].item() > cfg.skip_inv_ratio * inv_med)
                or (zkl_med is not None and terms["moment_kl"].item() > cfg.skip_zkl_ratio * zkl_med))
        if not skip:
            scaler.step(opt)
        scaler.update()
        if not skip:
            inv_med = terms["inv"].item() if inv_med is None else 0.99 * inv_med + 0.01 * terms["inv"].item()
            zkl_med = terms["moment_kl"].item() if zkl_med is None else 0.99 * zkl_med + 0.01 * terms["moment_kl"].item()
        m.post_step(mods, s, STEPS)
        rows.append(dict(branch=tag, step=s, inv=terms["inv"].item(), moment_kl=terms["moment_kl"].item(),
                         h_moment_kl=terms["h_moment_kl"].item(), grad_norm=gn, skip=int(skip),
                         inv_med=inv_med or 0, zkl_med=zkl_med or 0,
                         W0=float(mods["projector"][0].weight.norm()),
                         W3=float(mods["projector"][3].weight.norm()),
                         W6=float(mods["projector"][6].weight.norm())))
        if s % 100 == 0 or (skip and s < 400):
            print(f"  {tag} step {s:5d} inv {terms['inv'].item():8.3f} (med {inv_med or 0:.3f}) "
                  f"zkl {terms['moment_kl'].item():7.3f} gn {gn:9.1f} {'SKIP' if skip else ''}", flush=True)
    n = sum(r["skip"] for r in rows if r["branch"] == tag)
    print(f"[replay] {tag}: {n}/{STEPS} skipped | first skip at "
          f"{next((r['step'] for r in rows if r['branch'] == tag and r['skip']), None)}", flush=True)

rows = []
branch(False, rows)     # B first: establishes the healthy reference
branch(True, rows)      # A: frozen
import pandas as pd
pd.DataFrame(rows).to_csv(OUT, index=False); print(f"[replay] wrote {OUT}")
