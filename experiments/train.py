"""The frame loop (PROTOCOL §2): one trainer for every method. Methods supply recipe hooks
(sslgap/methods); this loop owns seeding, precision, the online-probe monitor, logging schema,
checkpoint cadence (schema v1, heads preserved), and resume.

  sbatch slurm/train.sbatch method=lejepa frame=toy_vits8            # frame-standard run
  sbatch slurm/train.sbatch method=lejepa frame=toy_vits8 frame.epochs=800 tag=portval
"""
import os

import hydra
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


@hydra.main(version_base=None, config_path="configs", config_name="train")
def main(cfg: DictConfig):
    frame = Frame(name=cfg.frame.name, model_name=cfg.frame.model_name,
                  img_size=cfg.frame.img_size, dataset=cfg.frame.dataset,
                  data_root=cfg.frame.get("data_root"), epochs=cfg.frame.epochs,
                  seed=cfg.seed, grad_clip=cfg.method.get("grad_clip", cfg.frame.grad_clip),
                  num_workers=cfg.num_workers, device=cfg.device)
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
    train = DataLoader(train_ds, batch_size=cfg.bs, shuffle=True, drop_last=True,
                       num_workers=frame.num_workers,          # persistent_workers=False: official
                       generator=g, worker_init_fn=seed_worker)
    val = DataLoader(val_ds, batch_size=256, num_workers=frame.num_workers,
                     worker_init_fn=seed_worker)

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

    total_steps = steps_per_epoch * frame.epochs
    cadence = set(frame.cadence())
    gnorm_med, step = None, start_ep * steps_per_epoch
    for epoch in range(start_ep, frame.epochs):
        method.train_mode(modules)
        probe.train()
        method.on_epoch_start(modules, epoch)
        for batch_x, y in train:
            batch_x = to_device(batch_x)
            y = y.to(frame.device, non_blocking=True)
            with autocast(frame.device, dtype=torch.bfloat16):
                terms, probe_feats, k = method.training_step(modules, batch_x, frame.device)
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
        print(f"[train] {run_id} ep{epoch + 1}/{frame.epochs} probe_acc={acc:.4f}", flush=True)

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
