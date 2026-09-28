"""lambda-JEPA pieces for the LeVJEPA trainer: SACReg on the all-gathered per-clip view centers, ring buffers, the share logger, the online K710 probe, named checkpoints."""
import math
import time

import lightning as pl
import numpy as np
import torch
import torch.distributed as dist
import torch.distributed.nn.functional as dist_nn
import torch.nn as nn

def _distributed():
    return dist.is_available() and dist.is_initialized() and dist.get_world_size() > 1

def all_gather_rows(t):
    if not _distributed():
        return t
    return torch.cat(dist_nn.all_gather(t.contiguous()), 0)

class SACReg(nn.Module):

    def __init__(self, d_slice=128, eps=1e-4):
        super().__init__()
        self.d_slice, self.eps, self.trace_last = d_slice, eps, None

    def forward(self, x, seed):
        with torch.autocast(x.device.type, enabled=False):
            x = x.reshape(-1, x.size(-1)).float()
            gen = torch.Generator(device=x.device).manual_seed(int(seed))
            Q, _ = torch.linalg.qr(torch.randn(x.size(1), self.d_slice, device=x.device, generator=gen))
            p = x @ Q
            mu = p.mean(0)
            pc = p - mu
            S = pc.T @ pc / (p.size(0) - 1)
            cov = S + self.eps * torch.eye(self.d_slice, device=x.device)
            self.trace_last = float(cov.diagonal().sum().detach() / self.d_slice)
            logdet = 2 * torch.linalg.cholesky(cov).diagonal().log().sum()
            return 0.5 * (cov.diagonal().sum() + mu.square().sum() - self.d_slice - logdet) / self.d_slice

class Ring:

    def __init__(self):
        self.bufs = {}

    def __call__(self, key, x, q):
        if not q:
            return x
        buf = self.bufs.get(key, [])
        out = torch.cat([x] + buf) if buf else x
        self.bufs[key] = [x.detach()] + buf[:q - 1]
        return out

    def snapshot(self):
        return {k: list(v) for k, v in self.bufs.items()}

    def restore(self, snap):
        self.bufs = snap

def term_pulls(terms, params):
    names = list(terms)
    out = {}
    for i, k in enumerate(names):
        grad = torch.autograd.grad(terms[k], params, retain_graph=i < len(names) - 1, allow_unused=True)
        out[k] = torch.cat([t.reshape(-1).float() for t in grad if t is not None]).norm().item()
    return out

def orbit_energies(views):
    V = len(views)
    X = [v.astype(np.float64) for v in views]
    N = X[0].shape[0]
    W = 0.0
    for u in range(V):
        for w in range(u + 1, V):
            W += ((X[u] - X[w]) ** 2).sum(1).mean()
    W /= V * (V - 1) / 2
    m = np.stack(X).mean(0)
    mc = m - m.mean(0)
    B = 2.0 * (mc ** 2).sum(1).mean() * N / (N - 1) - W / V
    return {"W": W, "B": B, "omega": W / B}

def transmission(eh, ez):
    a2, b2 = ez["W"] / eh["W"], ez["B"] / eh["B"]
    return {"a": math.sqrt(a2), "b": math.sqrt(b2), "lam": math.sqrt(b2 / a2)}

class ShareLogger(pl.Callback):

    def __init__(self, terms_fn, weights, dataset, collate_fn, bs=128, every=1, seed=0):
        self.terms_fn, self.weights, self.every, self.seed = terms_fn, weights, every, seed
        self.dataset, self.collate_fn, self.bs, self.batch = dataset, collate_fn, bs, None

    def on_train_start(self, trainer, pl_module):
        if not trainer.is_global_zero:
            return
        import multiprocessing as mp
        loader = torch.utils.data.DataLoader(
            self.dataset, batch_size=self.bs, shuffle=True, num_workers=8, collate_fn=self.collate_fn,
            multiprocessing_context=mp.get_context("spawn"), generator=torch.Generator().manual_seed(self.seed))
        t0 = time.time()
        self.batch = next(iter(loader))
        print(f"[share] fixed batch of {self.bs} drawn in {time.time() - t0:.0f}s", flush=True)

    def on_train_epoch_start(self, trainer, pl_module):
        if trainer.is_global_zero and trainer.current_epoch % self.every == 0:
            self.measure(trainer, pl_module)

    def measure(self, trainer, pl_module):
        t0 = time.time()
        dev = pl_module.device
        batch = {k: (v.to(dev) if torch.is_tensor(v) else v) for k, v in self.batch.items()}
        rings = pl_module.ring.snapshot()
        bufs = {n: b.clone() for n, b in pl_module.named_buffers()}
        with torch.random.fork_rng(devices=[dev] if dev.type == "cuda" else []):
            torch.manual_seed(self.seed)
            with torch.autocast(dev.type, dtype=torch.bfloat16, enabled=dev.type == "cuda"):
                terms, cls, z = self.terms_fn(pl_module, batch, seed=self.seed, distributed=False)
            params = [p for p in pl_module.encoder.parameters() if p.requires_grad]
            g = term_pulls(terms, params)
        for n, b in pl_module.named_buffers():
            b.copy_(bufs[n])
        pl_module.ring.restore(rings)
        pl_module.train()
        w = self.weights
        tot = sum(w[k] * g[k] for k in g) or 1.0
        orb = {}
        for name, feats in (("h", cls), ("z", z)):
            vs = feats.detach().float().cpu().numpy()
            orb[name] = orbit_energies([vs[:, v] for v in range(vs.shape[1])])
        tr = transmission(orb["h"], orb["z"])
        ep = trainer.current_epoch
        line = ("[share] ep%d " % ep
                + " ".join(f"{k}={w[k] * g[k] / tot:.3f}(g{g[k]:.3f})" for k in g)
                + f" | omega_h={orb['h']['omega']:.3f} omega_z={orb['z']['omega']:.3f} lam={tr['lam']:.3f}"
                + f" | kl_z={float(terms['moment_kl'].detach()):.3f} kl_h={float(terms['h_moment_kl'].detach()):.3f}"
                + f" trace_z={pl_module.cond_z.trace_last:.3f} trace_h={pl_module.cond_h.trace_last:.3f}")
        print(line + f" | measured in {time.time() - t0:.0f}s", flush=True)
        exp = getattr(trainer.logger, "experiment", None)
        if exp is not None and hasattr(exp, "log"):
            exp.log({**{f"share/g_{k}": g[k] for k in g}, **{f"share/{k}": w[k] * g[k] / tot for k in g},
                     "share/wg_total": tot,
                     **{f"orbit/{q}_{s}": orb[s][q] for s in ("h", "z") for q in ("W", "B", "omega")},
                     "orbit/lam": tr["lam"], "share/trace_z": pl_module.cond_z.trace_last,
                     "share/trace_h": pl_module.cond_h.trace_last}, step=trainer.global_step)

class K710Probe(pl.Callback):

    def __init__(self, dataset, collate_fn, num_classes, n_eval=1024, lr=1e-3, seed=0):
        self.dataset, self.collate_fn, self.C, self.n_eval, self.lr, self.seed = dataset, collate_fn, num_classes, n_eval, lr, seed
        self.head, self.opt, self.eval_batch, self.last_acc = None, None, None, None

    def on_train_start(self, trainer, pl_module):
        if not trainer.is_global_zero:
            return
        import multiprocessing as mp
        loader = torch.utils.data.DataLoader(self.dataset, batch_size=64, shuffle=True, num_workers=8, collate_fn=self.collate_fn,
                                             multiprocessing_context=mp.get_context("spawn"), generator=torch.Generator().manual_seed(self.seed + 1))
        t0 = time.time(); views, labels = [], []
        for b in loader:
            views.append(b["views"][:, :1].contiguous()); labels.append(b["label"])
            if sum(len(v) for v in views) >= self.n_eval:
                break
        self.eval_batch = {"views": torch.cat(views)[:self.n_eval], "label": torch.cat(labels)[:self.n_eval]}
        print(f"[probe] held-out batch of {self.n_eval} clips drawn in {time.time() - t0:.0f}s ({tuple(self.eval_batch['views'].shape)} uint8)", flush=True)
        dim = pl_module.encoder.embed_dim
        self.head = nn.Linear(dim, self.C).to(pl_module.device); self.opt = torch.optim.Adam(self.head.parameters(), lr=self.lr)

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        h, y = getattr(pl_module, "_probe_h", None), getattr(pl_module, "_probe_y", None)
        if self.head is None or h is None:
            return
        loss = nn.functional.cross_entropy(self.head(h.float()), y); self.opt.zero_grad(set_to_none=True); loss.backward(); self.opt.step()
        pl_module._probe_h = None

    def on_train_epoch_end(self, trainer, pl_module):
        if self.head is None:
            return
        dev = pl_module.device; correct = 0
        with torch.no_grad(), torch.autocast(dev.type, dtype=torch.bfloat16, enabled=dev.type == "cuda"):
            for i in range(0, self.n_eval, 128):
                v = pl_module.encoder(rearrange_views(self.eval_batch["views"][i:i + 128].to(dev)))[:, 0]
                correct += (self.head(v.float()).argmax(1).cpu() == self.eval_batch["label"][i:i + 128]).sum().item()
        acc = 100.0 * correct / self.n_eval; self.last_acc = acc
        print(f"[probe] ep{trainer.current_epoch + 1} k710_top1={acc:.2f} (held-out {self.n_eval}, one clean-ish view, linear on detached h)", flush=True)
        exp = getattr(trainer.logger, "experiment", None)
        if exp is not None and hasattr(exp, "log"):
            exp.log({"probe/k710_top1": acc}, step=trainer.global_step)

class CheckpointKeeper(pl.Callback):

    def __init__(self, probe=None, keep_epochs=()):
        self.probe, self.keep_epochs, self.best, self.best_epoch = probe, {int(e) for e in keep_epochs}, -1.0, -1

    def state_dict(self):
        return {"best": self.best, "best_epoch": self.best_epoch}

    def load_state_dict(self, state):
        self.best, self.best_epoch = state.get("best", -1.0), state.get("best_epoch", -1)

    def on_train_epoch_end(self, trainer, pl_module):
        import json, os
        d = trainer.checkpoint_callback.dirpath if trainer.checkpoint_callback is not None else trainer.default_root_dir
        e = trainer.current_epoch
        if e in self.keep_epochs:
            trainer.save_checkpoint(os.path.join(d, f"epoch-{e:04d}.ckpt"))
        if self.probe is not None:
            acc = trainer.strategy.broadcast(self.probe.last_acc, src=0)
            if acc is not None and acc > self.best:
                self.best, self.best_epoch = float(acc), e
                trainer.save_checkpoint(os.path.join(d, "best.ckpt"))
                if trainer.is_global_zero:
                    json.dump({"epoch": e, "epochs_completed": e + 1, "k710_top1": float(acc)}, open(os.path.join(d, "best.json"), "w"))
                    print(f"[keep] best.ckpt <- epoch {e + 1} (k710_top1 {acc:.2f})", flush=True)

def rearrange_views(v):
    x = v[:, 0].float().div_(255.0)
    mean = torch.tensor((0.485, 0.456, 0.406), device=x.device).view(1, 1, 3, 1, 1); std = torch.tensor((0.229, 0.224, 0.225), device=x.device).view(1, 1, 3, 1, 1)
    return ((x - mean) / std).permute(0, 2, 1, 3, 4)
