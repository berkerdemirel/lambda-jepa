"""The lambda-JEPA recipe inside LeVJEPA (ssl_project video side project, 2026-09-02).

Against the donor (MLO-lab/LeVJEPA @3ea0dda) exactly the lambda-JEPA recipe is transplanted, the one
running at ImageNet-1k (E27 cells e27v6s/e27v6b/e27v6L): V = 6 global views of the clip (the
lejepa view stack, no local crops), invariance = view-to-mean MSE at the projector output on the
all-pairs scale (no EMA twin: LeVJEPA has none, Berker 2026-09-02), the two-sided spectral conditioner on the PER-CLIP VIEW
CENTERS at the projector output (Z) and at the CLS token of the encoder output (H), each with a
detached ring of the last q steps' centers, doses (w_inv, w_z, w_h) taken from the ImageNet-1k
chains as they are. Token dropping, block-causal attention, projector, optimizer and schedule are
the donor's. Everything here is ported in math from ssl_project (sslgap/methods/_common.py,
sslgap/methods/lambdajepa.py, experiments/train_ddp.py, sslgap/metrics/orbit_energy.py); the code
is written for the donor's Lightning module.

DDP convention (house, train_ddp._all_gather): the per-clip centers are all-gathered with an
autograd-aware gather, every rank computes the conditioner on the GLOBAL batch, the backward
reduce-scatter returns each rank the gradient of its own rows, and DDP's parameter-gradient
averaging reconstructs the exact global-batch gradient. The random slice is drawn from a
step-seeded generator so all ranks share one basis.
"""
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
    """Autograd-aware all-gather along dim 0 (equal shapes: drop_last and a batch that divides
    the world size)."""
    if not _distributed():
        return t
    return torch.cat(dist_nn.all_gather(t.contiguous()), 0)


class SACReg(nn.Module):
    """KL(N(mu, S) || N(0, I_d')) / d' on a fresh random d'-dim orthonormal slice per step.
    Two-sided: variance above 1 is taxed like variance below it; the log-det is the
    anti-degeneracy barrier; a pure function of the batch mean and covariance. fp32 with
    autocast off (Cholesky). `trace_last` = whitened trace/d' of the slice, the stream-health
    read (a starved center stream shows trace/d' far below 1 at the fixed point)."""

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
    """Detached ring of the last q steps' conditioner inputs (D-064): widens the moment estimate
    from n = N_global to (q+1) N_global so the slice keeps its d'. Gradient flows only through
    the current rows; the ring re-warms over q steps after every resume (declared). One ring per
    tap ("z", "h"); snapshot/restore bracket the share measurement."""

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
    """The pull instrument: g_k = ||d term_k / d theta_encoder||; realized share of term k at
    weights w = w_k g_k / sum_j w_j g_j."""
    names = list(terms)
    out = {}
    for i, k in enumerate(names):
        grad = torch.autograd.grad(terms[k], params, retain_graph=i < len(names) - 1, allow_unused=True)
        out[k] = torch.cat([t.reshape(-1).float() for t in grad if t is not None]).norm().item()
    return out


def orbit_energies(views):
    """views: list of V arrays (N, D), same clip order. W = E_i E_{u<v} ||s_iu - s_iv||^2,
    B = E_{i!=j} ||m_i - m_j||^2 - W/V (debiased), thickness Theta = W/B (logged as omega)."""
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
    """The house share/dose logger, rank 0, every `every` epochs at epoch start: on a FIXED batch
    (drawn once from the training set with a seeded loader, no workers) the per-term encoder
    gradient norms g_k, the realized shares w_k g_k / sum, the moment-KL values, each stream's
    whitened trace/d', and the cloud energies at H and Z over the V views (omega_h, omega_z, lam).
    Single-process measurement (no gather), rings and BatchNorm buffers snapshotted and restored,
    RNG forked. Prints the `[share] epN ...` line the fleet watcher reads."""

    def __init__(self, terms_fn, weights, dataset, collate_fn, bs=128, every=1, seed=0):
        self.terms_fn, self.weights, self.every, self.seed = terms_fn, weights, every, seed
        self.dataset, self.collate_fn, self.bs, self.batch = dataset, collate_fn, bs, None

    def on_train_start(self, trainer, pl_module):
        if not trainer.is_global_zero:
            return
        # the fixed batch is decoded with workers: on video, bs clips x V views x T frames of augmentation
        # (12k frame-augs at bs 128) take many minutes single-threaded, and every other rank waits at the
        # first collective meanwhile (DDP timeout risk). Same seed -> same batch on every restart.
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
    """Online linear probe on the K710 labels (the video analog of train_ddp's per-epoch `probe_acc`; Berker
    2026-09-04: "some sort of a positive curve as we train"). A linear head on the DETACHED view-mean CLS (h) is
    trained on rank 0's own micro-batches with their labels (its own Adam; nothing flows back into the encoder);
    at every epoch END (2026-09-06: moved from the next epoch's start — the same head state — so CheckpointKeeper can
    save `best.ckpt` on it) it is scored on a FIXED held-out batch of pretraining clips drawn once with a seeded loader
    (the labels were never used by the SSL loss). Prints `[probe] epN k710_top1=..` (N = epochs completed) on rank 0,
    logs `probe/k710_top1`, keeps the value in `last_acc`. Attach with `+probe_k710=<held-out clips>` (see main.py);
    a resumed segment starts the head fresh (declared: its first epochs read low)."""

    def __init__(self, dataset, collate_fn, num_classes, n_eval=1024, lr=1e-3, seed=0):
        self.dataset, self.collate_fn, self.C, self.n_eval, self.lr, self.seed = dataset, collate_fn, num_classes, n_eval, lr, seed
        self.head, self.opt, self.eval_batch, self.last_acc = None, None, None, None

    def on_train_start(self, trainer, pl_module):
        if not trainer.is_global_zero:
            return
        import multiprocessing as mp
        # drawn in chunks of 64 clips and reduced to ONE view each on arrival: a single 1,024-clip batch materializes all
        # six views in one worker (~15 GB + the collate copy) and OOM-killed rank 0 of the 8 x 1 launch (job 64475095)
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
    """Named checkpoints beside ModelCheckpoint's rolling `last.ckpt` (Berker 2026-09-06: "instead of keeping every checkpoint
    we should specifically keep 239, last, best"): `epoch-{e:04d}.ckpt` at the end of every epoch e in keep_epochs (0-based:
    239 = the 240-epoch state) and `best.ckpt` (+ `best.json`) for the epoch whose online K710 probe read highest. The probe
    value lives on rank 0 and is broadcast so every rank takes the same decision (save_checkpoint is collective); the best
    value is part of the checkpoint (callback state), so a resumed segment continues the comparison."""

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
    """(b, 1, t, c, h, w) uint8 -> normalized (b, c, t, h, w) for the encoder (main.to_float_normalized semantics)."""
    x = v[:, 0].float().div_(255.0)
    mean = torch.tensor((0.485, 0.456, 0.406), device=x.device).view(1, 1, 3, 1, 1); std = torch.tensor((0.229, 0.224, 0.225), device=x.device).view(1, 1, 3, 1, 1)
    return ((x - mean) / std).permute(0, 2, 1, 3, 4)
