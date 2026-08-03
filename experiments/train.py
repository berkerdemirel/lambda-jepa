"""The frame loop (PROTOCOL §2): one trainer for every method. Methods supply recipe hooks
(sslgap/methods); this loop owns seeding, precision, the online-probe monitor, logging schema,
checkpoint cadence (schema v1, heads preserved), and resume.

  sbatch slurm/train.sbatch method=lejepa frame=toy_vits8            # frame-standard run
  sbatch slurm/train.sbatch method=lejepa frame=toy_vits8 frame.epochs=800 tag=portval
"""
import os
import random

import hydra
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import wandb
from omegaconf import DictConfig, OmegaConf
from torch.amp import GradScaler, autocast
from torch.utils.data import DataLoader

from sslgap.ckpt.schema import provenance_stamp, save_checkpoint
from sslgap.data import ViewsDataset, seed_everything, seed_worker
from sslgap.methods import METHODS
from sslgap.methods.base import Frame
from sslgap.metrics.orbit_energy import orbit_energies, transmission


@hydra.main(version_base=None, config_path="configs", config_name="train")
def main(cfg: DictConfig):
    # D-057: null num_workers -> one per allocated CPU (the input pipeline is the measured
    # bottleneck at fixed small worker counts; e22_bench{,2}).
    workers = (cfg.num_workers if cfg.num_workers is not None
               else int(os.environ.get("SLURM_CPUS_PER_TASK", 8)))
    frame = Frame(name=cfg.frame.name, model_name=cfg.frame.model_name,
                  img_size=cfg.frame.img_size, dataset=cfg.frame.dataset,
                  data_root=cfg.frame.get("data_root"), epochs=cfg.frame.epochs,
                  seed=cfg.seed, grad_clip=cfg.method.get("grad_clip", cfg.frame.grad_clip),
                  num_workers=workers, device=cfg.device)
    run_id = f"{frame.name}.{cfg.method.name}.s{cfg.seed}" + (f".{cfg.tag}" if cfg.tag else "")
    out_dir = os.path.expanduser(cfg.out_dir)
    os.makedirs(out_dir, exist_ok=True)
    ckpt_base = os.path.join(out_dir, run_id)
    last_path = f"{ckpt_base}_last.pt"

    seed_everything(cfg.seed)                  # python random (I-JEPA masks) + numpy + torch(+cuda)
    method = METHODS[cfg.method.name](cfg.method, frame)
    modules = method.build_modules().to(frame.device)
    probe = nn.Sequential(nn.LayerNorm(method.probe_dim()),
                          nn.Linear(method.probe_dim(), cfg.num_classes)).to(frame.device)

    train_ds = method.build_train_dataset()
    val_ds = ViewsDataset(frame.dataset, "validation", V=1, img_size=frame.img_size,
                          data_root=frame.data_root)
    g = torch.Generator().manual_seed(cfg.seed)
    nw = frame.num_workers
    train = DataLoader(train_ds, batch_size=cfg.bs, shuffle=True, drop_last=True,
                       num_workers=nw, generator=g, worker_init_fn=seed_worker,
                       pin_memory=cfg.pin_memory,
                       persistent_workers=cfg.persistent_workers and nw > 0,
                       prefetch_factor=cfg.prefetch_factor if nw > 0 else None)
    val = DataLoader(val_ds, batch_size=256, num_workers=nw, worker_init_fn=seed_worker,
                     pin_memory=cfg.pin_memory,
                     persistent_workers=cfg.persistent_workers and nw > 0)

    opt = torch.optim.AdamW(method.param_groups(modules)
                            + [{"params": probe.parameters(),
                                "lr": cfg.probe_lr, "weight_decay": cfg.probe_wd}])
    steps_per_epoch = len(train)
    scheduler = method.build_scheduler(opt, steps_per_epoch, steps_per_epoch * frame.epochs)
    scaler = GradScaler()

    start_ep, best_acc, wandb_id = 0, 0.0, None
    if cfg.resume and os.path.exists(last_path):
        pay = torch.load(last_path, map_location=frame.device, weights_only=False)
        saved_arch = {k: v for k, v in pay["arch"].items() if k != "probe"}
        assert saved_arch == method.arch(), (
            f"resume refused: {last_path} was trained with a different architecture "
            f"(saved arch != current method.arch()). Delete the stale run_id checkpoints "
            f"or change tag= to start a fresh run.")
        for role, sd in pay["modules"].items():
            (probe if role == "probe" else modules[role]).load_state_dict(sd)
        opt.load_state_dict(pay["optim"]["opt"])
        scheduler.load_state_dict(pay["optim"]["scheduler"])
        scaler.load_state_dict(pay["optim"]["scaler"])
        method.load_extras(pay.get("extras", {}))
        start_ep, best_acc = pay["epoch"] + 1, pay.get("best_acc") or 0.0
        wandb_id = pay["provenance"].get("wandb_id")
        print(f"[train] resumed {run_id} at epoch {start_ep}")

    run = wandb.init(project=cfg.wandb_project, name=run_id, id=wandb_id, resume="allow",
                     mode=cfg.wandb_mode, config=OmegaConf.to_container(cfg, resolve=True))

    def save(path, epoch):
        save_checkpoint(path, method=cfg.method.name, epoch=epoch,
                        step=(epoch + 1) * steps_per_epoch,
                        frame=OmegaConf.to_container(cfg.frame, resolve=True),
                        cfg=OmegaConf.to_container(cfg, resolve=True),
                        arch={**method.arch(),
                              "probe": {"class": "experiments.train.probe_arch",
                                        "kwargs": {"dim": method.probe_dim(),
                                                   "num_classes": cfg.num_classes}}},
                        modules={**{k: v for k, v in modules.items()}, "probe": probe},
                        extras=method.extras(),
                        optim={"opt": opt.state_dict(), "scheduler": scheduler.state_dict(),
                               "scaler": scaler.state_dict()},
                        best_acc=best_acc,
                        provenance=provenance_stamp(wandb_id=run.id, run_id=run_id,
                                                    seed=cfg.seed))

    def to_device(x):
        if torch.is_tensor(x):
            return x.to(frame.device, non_blocking=True)
        return type(x)(to_device(t) for t in x)

    # E24 realized-share logger (D-070): every share_log_every epochs, per-term trunk-only
    # w·g on ONE fixed probe batch. Trajectory-identical to logger-off by construction:
    # python/numpy/torch(+cuda) RNG forked-and-restored around batch draw and measurement
    # (the conditioners draw per-step slices), module buffers (BN stats) and the method's
    # queue rings snapshot-restored around the extra forward; autograd.grad touches no
    # .grad/optimizer state. null = off, code path untouched.
    share_batch = None
    if cfg.share_log_every:
        pull_w = getattr(method, "PULL_W", None)
        assert pull_w, f"share_log_every set but {cfg.method.name} declares no PULL_W"
        py_s, np_s = random.getstate(), np.random.get_state()
        with torch.random.fork_rng():
            torch.manual_seed(0)
            share_batch = next(iter(DataLoader(method.build_train_dataset(),
                                               batch_size=cfg.share_log_bs or 128,
                                               shuffle=True, num_workers=0,
                                               generator=torch.Generator().manual_seed(0))))
        random.setstate(py_s), np.random.set_state(np_s)

    def log_shares(epoch, step):
        py_s, np_s = random.getstate(), np.random.get_state()
        with torch.random.fork_rng():
            bufs = {r: {n: b.clone() for n, b in m.named_buffers()}
                    for r, m in modules.items()}
            rings = {a: list(getattr(method, a)) for a in ("_zq", "_hq")
                     if hasattr(method, a)}
            bx, by = share_batch
            with autocast(frame.device, dtype=torch.bfloat16):
                terms, _, _ = method.training_step(modules, to_device(bx), frame.device,
                                                   y=by.to(frame.device))
            params = [p for p in modules["backbone"].parameters() if p.requires_grad]
            names = [k for k in terms if k != "loss"]
            gs = {}
            for i, k in enumerate(names):
                g = torch.autograd.grad(terms[k], params, retain_graph=i < len(names) - 1,
                                        allow_unused=True)
                gs[k] = torch.cat([t.reshape(-1).float() for t in g
                                   if t is not None]).norm().item()
            # Ω diagnosis (Berker 2026-08-03: "measure Ω as well … for the diagnosis"): the
            # D-068 orbit calculus on the SAME fixed batch — eval-mode forward (the landing
            # extraction convention), label-free. Levels are train-aug-stack-local (the
            # landing o8/audit_v1 numbers stay canonical); the trajectory is the instrument.
            for m in modules.values():
                m.eval()
            N, V = bx.shape[:2]
            with torch.no_grad(), autocast(frame.device, dtype=torch.bfloat16):
                cls_e = modules["backbone"].forward_features(
                    to_device(bx).flatten(0, 1))[:, 0]
                z_e = modules["projector"](cls_e)
            orb = {}
            for nm_, feats in (("h", cls_e), ("z", z_e)):
                vs = feats.reshape(N, V, -1).float().cpu().numpy()
                orb[nm_] = orbit_energies([vs[:, v] for v in range(V)])
            tr = transmission(orb["h"], orb["z"])
            for r, m in modules.items():
                for n, b in m.named_buffers():
                    b.copy_(bufs[r][n])
            for a, v in rings.items():
                setattr(method, a, v)
            method.train_mode(modules)
        random.setstate(py_s), np.random.set_state(np_s)
        w = {k: float(cfg.method[pull_w[k]]) for k in names}
        tot = sum(w[k] * gs[k] for k in names) or 1.0
        # D-073 rider: OAS-shrunk conditioners stash their per-forward rho — the
        # evidence level is part of the trajectory record (absent = legacy estimator).
        rhos = {f"share/rho_{t}": c.rho_last for t, c in
                (("z", getattr(method, "cond_z", None)),
                 ("h", getattr(method, "cond_h", None)))
                if getattr(c, "rho_last", None) is not None}
        wandb.log({**{f"share/g_{k}": gs[k] for k in names},
                   **{f"share/{k}": w[k] * gs[k] / tot for k in names},
                   "share/wg_total": tot, **rhos,
                   **{f"orbit/{q}_{s}": orb[s][q] for s in ("h", "z")
                      for q in ("W", "B", "omega")},
                   **{f"orbit/{q}": tr[q] for q in ("a", "b", "lam")}}, step=step)
        print("[share] ep%d " % epoch +
              " ".join(f"{k}={w[k] * gs[k] / tot:.3f}(g{gs[k]:.3f})" for k in names) +
              f" | omega_h={orb['h']['omega']:.3f} omega_z={orb['z']['omega']:.3f} "
              f"lam={tr['lam']:.3f}" +
              ("".join(f" {k.split('/')[1]}={v:.3f}" for k, v in sorted(rhos.items()))
               if rhos else ""), flush=True)

    total_steps = steps_per_epoch * frame.epochs
    cadence = set(frame.cadence())
    gnorm_med, step = None, start_ep * steps_per_epoch
    for epoch in range(start_ep, frame.epochs):
        method.train_mode(modules)
        probe.train()
        method.on_epoch_start(modules, epoch)
        if cfg.share_log_every and epoch % cfg.share_log_every == 0:
            log_shares(epoch, step)
        for batch_x, y in train:
            batch_x = to_device(batch_x)
            y = y.to(frame.device, non_blocking=True)
            with autocast(frame.device, dtype=torch.bfloat16):
                terms, probe_feats, k = method.training_step(modules, batch_x, frame.device, y=y)
                y_rep = y.repeat_interleave(k) if k > 1 else y
                probe_loss = F.cross_entropy(probe(probe_feats), y_rep)
                loss = terms["loss"] + probe_loss
            opt.zero_grad()
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            gn = torch.nn.utils.clip_grad_norm_(
                [p for g in opt.param_groups for p in g["params"]],
                frame.grad_clip if frame.grad_clip else float("inf")).item()
            gnorm_med = gn if gnorm_med is None else 0.99 * gnorm_med + 0.01 * gn
            scaler.step(opt)
            scaler.update()
            scheduler.step()
            step += 1
            monitors = method.post_step(modules, step, total_steps)
            wandb.log({**{f"train/{k_}": v.item() for k_, v in terms.items()},
                       "train/probe": probe_loss.item(), "train/grad_norm": gn,
                       "lr": scheduler.get_last_lr()[0],
                       **{f"monitor/{k_}": v for k_, v in monitors.items()}}, step=step)
            if gnorm_med and gn > 100 * gnorm_med:
                print(f"[train] INCIDENT: grad_norm {gn:.1f} > 100x running median "
                      f"{gnorm_med:.3f} at step {step} (WORKFLOW.md kill-trigger)", flush=True)

        # D-056: monitor eval every eval_every epochs (final epoch always); ckpt saves are
        # OUTSIDE the gate — _last/cadence must land every epoch regardless of eval cadence.
        if (epoch + 1) % cfg.eval_every == 0 or epoch + 1 == frame.epochs:
            for m in modules.values():
                m.eval()
            probe.eval()
            correct, n = 0, 0
            with torch.inference_mode():
                for views, y in val:
                    x = views.to(frame.device, non_blocking=True).flatten(0, 1)
                    y = y.to(frame.device, non_blocking=True)
                    with autocast(frame.device, dtype=torch.bfloat16):
                        logits = probe(method.eval_features(modules, x, frame.device))
                    correct += (logits.argmax(1) == y).sum().item()
                    n += y.numel()
            acc = correct / n
            wandb.log({"test/acc": acc, "test/epoch": epoch}, step=step)
            print(f"[train] {run_id} ep{epoch + 1}/{frame.epochs} probe_acc={acc:.4f}",
                  flush=True)
            if acc > best_acc:
                best_acc = acc
                save(f"{ckpt_base}_best.pt", epoch)
        save(last_path, epoch)
        if (epoch + 1) in cadence:
            save(f"{ckpt_base}_ep{epoch + 1}.pt", epoch)
    wandb.finish()
    print(f"[train] done {run_id}: best={best_acc:.4f}")


def probe_arch(dim, num_classes):
    return nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, num_classes))


if __name__ == "__main__":
    main()
