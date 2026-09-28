"""Multi-GPU DDP trainer for lambda-JEPA: SACReg on the all-gathered view centers with rank-shared random slices; checkpoints in the schema of train.py."""
import math
import os
import random

import hydra
import numpy as np
import torch
import torch.distributed as dist
import torch.distributed.nn as dist_nn
import torch.nn as nn
import torch.nn.functional as F
import wandb
from omegaconf import DictConfig, OmegaConf
from torch.amp import GradScaler, autocast
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from sslgap.ckpt.schema import load_payload, provenance_stamp, save_checkpoint
from sslgap.data import ViewsDataset, orbit_stack, seed_everything, seed_worker
from sslgap.methods import METHODS
from sslgap.methods.base import Frame
from sslgap.metrics.orbit_energy import orbit_energies, transmission

class _StudentFwd(nn.Module):

    def __init__(self, backbone, projector):
        super().__init__()
        self.backbone, self.projector = backbone, projector

    def forward(self, views):
        if isinstance(views, (list, tuple)):
            g, l = views
            N, Vg = g.shape[:2]
            V = Vg + l.shape[1]
            cls = torch.cat(
                [self.backbone.forward_features(g.flatten(0, 1))[:, 0].reshape(N, Vg, -1),
                 self.backbone.forward_features(l.flatten(0, 1))[:, 0].reshape(N, V - Vg, -1)], 1)
        else:
            N, V = views.shape[:2]
            cls = self.backbone.forward_features(views.flatten(0, 1))[:, 0].reshape(N, V, -1)
        z = self.projector(cls.flatten(0, 1)).reshape(N, V, -1)
        return cls, z

def _all_gather(t):
    if not (dist.is_initialized() and dist.get_world_size() > 1):
        return t
    return torch.cat(dist_nn.all_gather(t.contiguous()), 0)

def _ddp_loss(method, ddp_student, modules, views, w_inv, w_floor, h_lamb, q_seed):
    cs = method.cfg.get("cond_stream")
    assert cs in (None, "all"), f"train_ddp supports cond_stream in (None,'all'); got {cs!r}"
    cls, z = ddp_student(views)
    N, V = z.shape[:2]
    if "teacher_backbone" in modules:
        anchor = method._swa_mu(modules, views)
        inv = (z - anchor).square().mean()
        if not isinstance(views, (list, tuple)):
            inv = inv * (2 * V / (V - 1))
    elif isinstance(views, (list, tuple)):
        inv = (z - z.mean(1, keepdim=True)).square().mean()
    else:
        inv = sum(F.mse_loss(z[:, u], z[:, w]) for u in range(V) for w in range(u + 1, V)) \
            / (V * (V - 1) / 2)
    zin = _all_gather(z[:, :V].mean(1))
    hin = _all_gather(cls[:, :V].mean(1))
    q = int(method.cfg.get("queue_steps", 0) or 0)
    qh = method.cfg.get("h_queue_steps")
    qh = q if qh is None else int(qh)
    if dist.is_initialized() and dist.get_world_size() > 1:
        with torch.random.fork_rng(devices=[z.device]):
            torch.manual_seed(q_seed)
            reg_z = method.cond_z(method._ring("_zq", zin, q))
            reg_h = method.cond_h(method._ring("_hq", hin, qh))
    else:
        reg_z = method.cond_z(method._ring("_zq", zin, q))
        reg_h = method.cond_h(method._ring("_hq", hin, qh))
    loss = w_inv * inv + w_floor * reg_z + h_lamb * reg_h
    terms = {"inv": inv, "moment_kl": reg_z, "h_moment_kl": reg_h}
    probe_feats = cls.flatten(0, 1).detach()
    return {"loss": loss, **terms}, probe_feats, V

@hydra.main(version_base=None, config_path="configs", config_name="train")
def main(cfg: DictConfig):
    rank = int(os.environ.get("RANK", 0))
    world = int(os.environ.get("WORLD_SIZE", 1))
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    is_main = rank == 0
    if world > 1:
        dist.init_process_group("nccl")
        torch.cuda.set_device(local_rank)
    device = f"cuda:{local_rank}"

    assert cfg.method.name == "lambdajepa", "train_ddp is the LambdaJEPA DDP add-on only"
    assert cfg.bs % world == 0, f"bs={cfg.bs} must divide world_size={world}"
    bs_rank = cfg.bs // world
    workers = (cfg.num_workers if cfg.num_workers is not None
               else int(os.environ.get("SLURM_CPUS_PER_TASK", 8)))
    frame = Frame(name=cfg.frame.name, model_name=cfg.frame.model_name,
                  img_size=cfg.frame.img_size, dataset=cfg.frame.dataset,
                  data_root=cfg.frame.get("data_root"), epochs=cfg.frame.epochs,
                  seed=cfg.seed, grad_clip=cfg.method.get("grad_clip", cfg.frame.grad_clip),
                  num_workers=workers, device=device)
    run_id = f"{frame.name}.{cfg.method.name}.s{cfg.seed}" + (f".{cfg.tag}" if cfg.tag else "")
    out_dir = os.path.expanduser(cfg.out_dir)
    if is_main:
        os.makedirs(out_dir, exist_ok=True)
    ckpt_base = os.path.join(out_dir, run_id)
    last_path = f"{ckpt_base}_last.pt"

    seed_everything(cfg.seed)
    method = METHODS[cfg.method.name](cfg.method, frame)
    modules = method.build_modules().to(device)
    if cfg.method.get("freeze_prebn"):
        for i in (0, 3):
            for par in modules["projector"][i].parameters():
                par.requires_grad_(False)
    if cfg.method.get("freeze_prebn_bn"):
        for i in (1, 4):
            for par in modules["projector"][i].parameters():
                par.requires_grad_(False)

    def train_mode():
        method.train_mode(modules)
        if cfg.method.get("freeze_prebn_bn"):
            for i in (1, 4):
                modules["projector"][i].eval()
    probe = nn.Sequential(nn.LayerNorm(method.probe_dim()),
                          nn.Linear(method.probe_dim(), cfg.num_classes)).to(device)
    student = _StudentFwd(modules["backbone"], modules["projector"])
    if world > 1:
        student = DDP(student, device_ids=[local_rank])
        probe = DDP(probe, device_ids=[local_rank])
    probe_core = probe.module if world > 1 else probe

    train_ds = method.build_train_dataset()
    val_ds = ViewsDataset(frame.dataset, "validation", V=1, img_size=frame.img_size,
                          data_root=frame.data_root)
    sampler = DistributedSampler(train_ds, num_replicas=world, rank=rank, shuffle=True,
                                 seed=cfg.seed, drop_last=True)
    g = torch.Generator().manual_seed(cfg.seed + rank)
    nw = workers
    train = DataLoader(train_ds, batch_size=bs_rank, sampler=sampler, drop_last=True,
                       num_workers=nw, generator=g, worker_init_fn=seed_worker,
                       pin_memory=cfg.pin_memory,
                       persistent_workers=cfg.persistent_workers and nw > 0,
                       prefetch_factor=cfg.prefetch_factor if nw > 0 else None)
    val = DataLoader(val_ds, batch_size=256, num_workers=nw, worker_init_fn=seed_worker,
                     pin_memory=cfg.pin_memory) if is_main else None

    opt = torch.optim.AdamW(method.param_groups(modules)
                            + [{"params": probe_core.parameters(),
                                "lr": cfg.probe_lr, "weight_decay": cfg.probe_wd}])
    steps_per_epoch = len(train)
    scheduler = method.build_scheduler(opt, steps_per_epoch, steps_per_epoch * frame.epochs)
    scaler = GradScaler()
    w_inv, w_floor, h_lamb = float(cfg.method.w_inv), float(cfg.method.w_floor), float(cfg.method.h_lamb)

    start_ep, best_acc, wandb_id, resume_step = 0, 0.0, None, None
    if cfg.resume and os.path.exists(last_path):
        pay = load_payload(last_path, map_location=device)
        saved_arch = {k: v for k, v in pay["arch"].items() if k != "probe"}
        assert saved_arch == method.arch(), f"resume refused: {last_path} arch mismatch"
        for role, sd in pay["modules"].items():
            (probe_core if role == "probe" else modules[role]).load_state_dict(sd)
        opt.load_state_dict(pay["optim"]["opt"])
        for g, ref in zip(opt.param_groups, method.param_groups(modules) + [{"weight_decay": cfg.probe_wd}]):
            g["weight_decay"] = ref["weight_decay"]
        if is_main:
            print(f"[train_ddp] resume: weight_decay per group from cfg = {[g['weight_decay'] for g in opt.param_groups]}", flush=True)
        scheduler.load_state_dict(pay["optim"]["scheduler"])
        scaler.load_state_dict(pay["optim"]["scaler"])
        method.load_extras(pay.get("extras", {}))
        start_ep, best_acc = pay["epoch"] + 1, pay.get("best_acc") or 0.0
        wandb_id, resume_step = pay["provenance"].get("wandb_id"), pay["step"]
        if is_main:
            print(f"[train_ddp] resumed {run_id} at epoch {start_ep}", flush=True)
            if cfg.method.get("freeze_prebn_bn"):
                bn = [modules["projector"][i] for i in (1, 4)]
                print("[train_ddp] freeze_prebn_bn: projector BN-1/BN-2 held (eval mode, affines fixed): running_var median "
                      f"{bn[0].running_var.median().item():.3e} / {bn[1].running_var.median().item():.3e}, min "
                      f"{bn[0].running_var.min().item():.3e} / {bn[1].running_var.min().item():.3e}", flush=True)
            if cfg.method.get("freeze_prebn"):
                print("[train_ddp] freeze_prebn held fixed at the RESUMED values: "
                      + ", ".join(f"projector.{i}.weight |w|={modules['projector'][i].weight.norm():.1f}"
                                  for i in (0, 3)), flush=True)

    run = None
    if is_main:
        suffix = cfg.get("wandb_new_run")
        wb = {"name": f"{run_id}.{suffix}"} if (suffix and wandb_id) else {"name": run_id, "id": wandb_id, "resume": "allow"}
        run = wandb.init(project=cfg.wandb_project, mode=cfg.wandb_mode, config=OmegaConf.to_container(cfg, resolve=True), **wb)
        if suffix and wandb_id:
            print(f"[train_ddp] wandb: new run {wb['name']} ({run.id}) from step {resume_step}; the old run {wandb_id} keeps its history", flush=True)

    def to_device(x):
        if torch.is_tensor(x):
            return x.to(device, non_blocking=True)
        return type(x)(to_device(t) for t in x)

    def save(path, epoch):
        save_checkpoint(path, method=cfg.method.name, epoch=epoch,
                        step=(epoch + 1) * steps_per_epoch,
                        frame=OmegaConf.to_container(cfg.frame, resolve=True),
                        cfg=OmegaConf.to_container(cfg, resolve=True),
                        arch={**method.arch(),
                              "probe": {"class": "experiments.train.probe_arch",
                                        "kwargs": {"dim": method.probe_dim(),
                                                   "num_classes": cfg.num_classes}}},
                        modules={**{k: v for k, v in modules.items()}, "probe": probe_core},
                        extras=method.extras(),
                        optim={"opt": opt.state_dict(), "scheduler": scheduler.state_dict(),
                               "scaler": scaler.state_dict()},
                        best_acc=best_acc,
                        provenance=provenance_stamp(wandb_id=run.id, run_id=run_id, seed=cfg.seed))

    share_batch = omega_aud = omega_gentle = None
    if is_main and cfg.share_log_every:
        pull_w = getattr(method, "PULL_W", None)
        assert pull_w, f"share_log_every set but {cfg.method.name} declares no PULL_W"
        py_s, np_s = random.getstate(), np.random.get_state()
        with torch.random.fork_rng():
            torch.manual_seed(0)
            share_batch = next(iter(DataLoader(method.build_train_dataset(),
                                               batch_size=cfg.share_log_bs or 128, shuffle=True,
                                               num_workers=0,
                                               generator=torch.Generator().manual_seed(0))))
            if isinstance(share_batch[0], (list, tuple)):
                def _fixed(aug):
                    return next(iter(DataLoader(
                        ViewsDataset(frame.dataset, "train", V=4, img_size=frame.img_size,
                                     data_root=frame.data_root, aug=aug),
                        batch_size=cfg.share_log_bs or 128, shuffle=True, num_workers=0,
                        generator=torch.Generator().manual_seed(0))))[0]
                omega_aud = _fixed(None)
                omega_gentle = _fixed(orbit_stack(frame.img_size, (0.7, 1.0)))
        random.setstate(py_s), np.random.set_state(np_s)

    def log_shares(epoch, step):
        py_s, np_s = random.getstate(), np.random.get_state()
        with torch.random.fork_rng():
            bufs = {r: {n: b.clone() for n, b in m.named_buffers()} for r, m in modules.items()}
            rings = {a: list(v) for a, v in vars(method).items() if a.startswith(("_zq", "_hq"))}
            bx, by = share_batch
            with autocast("cuda", dtype=torch.bfloat16):
                terms, _, _ = method.training_step(modules, to_device(bx), device,
                                                   y=by.to(device))
            params = [p for p in modules["backbone"].parameters() if p.requires_grad]
            names = [k for k in terms if k != "loss"]
            gs = {}
            for i, k in enumerate(names):
                grad = torch.autograd.grad(terms[k], params, retain_graph=i < len(names) - 1,
                                           allow_unused=True)
                gs[k] = torch.cat([t.reshape(-1).float() for t in grad if t is not None]).norm().item()
            for m in modules.values():
                m.eval()

            def orb_of(xb, with_z):
                nb, vb = xb.shape[:2]
                with torch.no_grad(), autocast("cuda", dtype=torch.bfloat16):
                    cls_e = modules["backbone"].forward_features(to_device(xb).flatten(0, 1))[:, 0]
                    z_e = (modules["projector"](modules["embed"](cls_e) if "embed" in modules
                                                else cls_e) if with_z else None)
                o = {}
                for nm_, feats in (("h", cls_e),) + ((("z", z_e),) if with_z else ()):
                    vs = feats.reshape(nb, vb, -1).float().cpu().numpy()
                    o[nm_] = orbit_energies([vs[:, v] for v in range(vb)])
                return o
            if omega_aud is not None:
                orb = orb_of(omega_aud, with_z=True)
                extra = {"orbit/h_own_omega": orb_of(bx[0], False)["h"]["omega"],
                         "orbit/h_gentle_omega": orb_of(omega_gentle, False)["h"]["omega"]}
            else:
                orb = orb_of(bx, with_z=True)
                extra = {}
            tr = transmission(orb["h"], orb["z"])
            for r, m in modules.items():
                for n, b in m.named_buffers():
                    b.copy_(bufs[r][n])
            for a, v in rings.items():
                setattr(method, a, v)
            train_mode()
        random.setstate(py_s), np.random.set_state(np_s)
        w = {k: float(cfg.method[pull_w[k]]) for k in names}
        tot = sum(w[k] * gs[k] for k in names) or 1.0
        rhos = {f"share/rho_{t}": c.rho_last for t, c in
                (("z", getattr(method, "cond_z", None)), ("h", getattr(method, "cond_h", None)))
                if getattr(c, "rho_last", None) is not None}
        wandb.log({**{f"share/g_{k}": gs[k] for k in names},
                   **{f"share/{k}": w[k] * gs[k] / tot for k in names},
                   "share/wg_total": tot, **rhos, **extra,
                   **{f"orbit/{q}_{s}": orb[s][q] for s in ("h", "z") for q in ("W", "B", "omega")},
                   **{f"orbit/{q}": tr[q] for q in ("a", "b", "lam")}}, step=step)
        print("[share] ep%d " % epoch
              + " ".join(f"{k}={w[k] * gs[k] / tot:.3f}(g{gs[k]:.3f})" for k in names)
              + f" | omega_h={orb['h']['omega']:.3f} omega_z={orb['z']['omega']:.3f} lam={tr['lam']:.3f}",
              flush=True)

    total_steps = steps_per_epoch * frame.epochs
    cadence = set(frame.cadence()) | {int(e) for e in (cfg.extra_cadence or [])}
    smoke = int(os.environ.get("DDP_MAX_STEPS", 0) or 0)
    gnorm_med, step, stop = None, start_ep * steps_per_epoch, False
    inv_med, zkl_med = None, None
    for epoch in range(start_ep, frame.epochs):
        train_mode()
        probe.train()
        method.on_epoch_start(modules, epoch)
        sampler.set_epoch(epoch)
        if is_main and cfg.share_log_every and epoch % cfg.share_log_every == 0:
            log_shares(epoch, step)
        for batch_x, y in train:
            batch_x = to_device(batch_x)
            y = y.to(device, non_blocking=True)
            q_seed = cfg.seed * 1_000_003 + step
            with autocast("cuda", dtype=torch.bfloat16):
                terms, probe_feats, k = _ddp_loss(method, student, modules, batch_x,
                                                  w_inv, w_floor, h_lamb, q_seed)
                y_rep = y.repeat_interleave(k) if k > 1 else y
                probe_loss = F.cross_entropy(probe(probe_feats), y_rep)
                loss = terms["loss"] + probe_loss
            opt.zero_grad()
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            gn = torch.nn.utils.clip_grad_norm_(
                [p for gr in opt.param_groups for p in gr["params"]],
                frame.grad_clip if frame.grad_clip else float("inf")).item()
            skip = False
            zkl_trip = False
            if cfg.get("skip_inv_ratio") or cfg.get("skip_zkl_ratio"):
                flag = torch.zeros(2, device=terms["loss"].device)
                if cfg.get("skip_inv_ratio") and inv_med is not None and terms["inv"].item() > cfg.skip_inv_ratio * inv_med:
                    flag[0] += 1
                if cfg.get("skip_zkl_ratio") and zkl_med is not None and terms["moment_kl"].item() > cfg.skip_zkl_ratio * zkl_med:
                    flag[1] += 1
                if dist.is_initialized():
                    dist.all_reduce(flag, op=dist.ReduceOp.MAX)
                skip = bool(flag.sum().item() > 0)
                zkl_trip = bool(flag[1].item() > 0)
            evicted = ""
            if skip:
                opt.zero_grad(set_to_none=True)
                if zkl_trip and cfg.get("skip_zkl_evict"):
                    buf = getattr(method, "_zq", [])
                    if len(buf) > 1:
                        v = [float(b.float().var(0, unbiased=True).sum()) for b in buf]
                        g = max(range(len(v)), key=lambda i: v[i])
                        evicted = f" | evicted ring block {g} of {len(buf)} (var {v[g]:.1f} vs median {sorted(v)[len(v) // 2]:.1f})"
                        del buf[g]
                if is_main:
                    print(f"[train_ddp] SKIPPED step {step + 1}: inv {terms['inv'].item():.3f} (mean {inv_med:.3f}) "
                          f"moment_kl {terms['moment_kl'].item():.3f} (mean {zkl_med:.3f}) grad_norm {gn:.1f}{evicted}", flush=True)
            else:
                scaler.step(opt)
            scaler.update()
            scheduler.step()
            step += 1
            if not skip:
                inv_med = terms["inv"].item() if inv_med is None else 0.99 * inv_med + 0.01 * terms["inv"].item()
                zkl_med = terms["moment_kl"].item() if zkl_med is None else 0.99 * zkl_med + 0.01 * terms["moment_kl"].item()
            monitors = method.post_step(modules, step, total_steps)
            if is_main:
                wandb.log({**{f"train/{k_}": v.item() for k_, v in terms.items()},
                           "train/probe": probe_loss.item(), "train/grad_norm": gn, "train/skipped": int(skip),
                           "lr": scheduler.get_last_lr()[0],
                           **{f"monitor/{k_}": v for k_, v in monitors.items()}}, step=step)
            if is_main and gnorm_med and math.isfinite(gn) and gn > 100 * gnorm_med:
                print(f"[train_ddp] INCIDENT: grad_norm {gn:.1f} > 100x running mean {gnorm_med:.3f} "
                      f"at step {step}", flush=True)
            if math.isfinite(gn):
                gnorm_med = gn if gnorm_med is None else 0.99 * gnorm_med + 0.01 * gn
            if smoke:
                if is_main:
                    print(f"[smoke] step {step} loss={terms['loss'].item():.4f} "
                          f"inv={terms['inv'].item():.4f} mkl={terms['moment_kl'].item():.4f} "
                          f"hkl={terms['h_moment_kl'].item():.4f} gn={gn:.2f}", flush=True)
                if step - start_ep * steps_per_epoch >= smoke:
                    stop = True
                    break

        if is_main and ((epoch + 1) % cfg.eval_every == 0 or epoch + 1 == frame.epochs):
            for m in modules.values():
                m.eval()
            probe.eval()
            correct, n = 0, 0
            with torch.inference_mode():
                for views, y in val:
                    x = views.to(device, non_blocking=True).flatten(0, 1)
                    y = y.to(device, non_blocking=True)
                    with autocast("cuda", dtype=torch.bfloat16):
                        logits = probe_core(method.eval_features(modules, x, device))
                    correct += (logits.argmax(1) == y).sum().item()
                    n += y.numel()
            acc = correct / n
            wandb.log({"test/acc": acc, "test/epoch": epoch}, step=step)
            print(f"[train_ddp] {run_id} ep{epoch + 1}/{frame.epochs} probe_acc={acc:.4f}", flush=True)
            if acc > best_acc:
                best_acc = acc
                save(f"{ckpt_base}_best.pt", epoch)
        if is_main:
            save(last_path, epoch)
            if (epoch + 1) in cadence:
                save(f"{ckpt_base}_ep{epoch + 1}.pt", epoch)
        if world > 1:
            dist.barrier()
        if stop:
            break
    if is_main:
        wandb.finish()
        print(f"[train_ddp] done {run_id}: best={best_acc:.4f}", flush=True)
    if world > 1:
        dist.destroy_process_group()

if __name__ == "__main__":
    main()
