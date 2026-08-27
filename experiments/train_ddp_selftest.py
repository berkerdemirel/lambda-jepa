"""Faithfulness gate for the DDP add-on: at world_size=1, experiments/train_ddp._ddp_loss must
reproduce the canonical FloorSSL.training_step loss byte-for-byte (same RNG state in, same drop_path
masks, same conditioner slice). If this drifts, the DDP trainer is optimizing a different objective
than the paper's — so it gates every DDP launch. Runs on CPU or one GPU; no dataset (random views).

  python experiments/train_ddp_selftest.py            # PASS/FAIL + max abs term diff
"""
import os
import sys

import torch
from hydra import compose, initialize_config_dir

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from sslgap.methods import METHODS
from sslgap.methods.base import Frame
from experiments.train_ddp import _StudentFwd, _ddp_loss

CFG_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "configs"))


def _cfg(overrides):
    with initialize_config_dir(config_dir=CFG_DIR, version_base=None):
        return compose(config_name="train", overrides=overrides)


def _check(name, overrides, views_fn, device):
    cfg = _cfg(overrides)
    frame = Frame(name=cfg.frame.name, model_name=cfg.frame.model_name,
                  img_size=cfg.frame.img_size, dataset=cfg.frame.dataset,
                  data_root=cfg.frame.get("data_root"), epochs=cfg.frame.epochs, seed=0,
                  grad_clip=1.0, num_workers=0, device=device)
    torch.manual_seed(0)
    method = METHODS["floorssl"](cfg.method, frame)
    modules = method.build_modules().to(device)
    method.train_mode(modules)                              # drop_path active — tests the hard path
    student = _StudentFwd(modules["backbone"], modules["projector"])
    views = views_fn()
    w = (float(cfg.method.w_inv), float(cfg.method.w_floor), float(cfg.method.h_lamb))

    st = torch.random.get_rng_state()
    cst = torch.cuda.get_rng_state_all() if device.startswith("cuda") else None
    terms_c, pf_c, V_c = method.training_step(modules, views, device, y=None)
    torch.random.set_rng_state(st)
    if cst is not None:
        torch.cuda.set_rng_state_all(cst)
    terms_d, pf_d, V_d = _ddp_loss(method, student, modules, views, *w, q_seed=0)

    ok = V_c == V_d and torch.allclose(pf_c, pf_d, atol=1e-5, rtol=1e-4)
    worst = 0.0
    for k in ("loss", "inv", "moment_kl", "h_moment_kl"):
        d = (terms_c[k] - terms_d[k]).abs().item()
        worst = max(worst, d)
        ok = ok and torch.allclose(terms_c[k], terms_d[k], atol=1e-5, rtol=1e-4)
        print(f"    {k:14} canon={terms_c[k].item():.6f}  ddp={terms_d[k].item():.6f}  |Δ|={d:.2e}")
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}  (V={V_c}, worst |Δ|={worst:.2e})")
    return ok


def _check_ring(name, overrides, views_fn, device, steps=3):
    """Ring configs (queue_steps>0, D-103 all-global cells) need a MULTI-STEP check — the ring
    carries state across steps and both paths push to it, so each path runs `steps` steps on a
    fresh ring from the same RNG state and every step's terms must match byte-for-byte."""
    cfg = _cfg(overrides)
    frame = Frame(name=cfg.frame.name, model_name=cfg.frame.model_name,
                  img_size=cfg.frame.img_size, dataset=cfg.frame.dataset,
                  data_root=cfg.frame.get("data_root"), epochs=cfg.frame.epochs, seed=0,
                  grad_clip=1.0, num_workers=0, device=device)
    torch.manual_seed(0)
    method = METHODS["floorssl"](cfg.method, frame)
    modules = method.build_modules().to(device)
    method.train_mode(modules)
    student = _StudentFwd(modules["backbone"], modules["projector"])
    views = [views_fn() for _ in range(steps)]
    w = (float(cfg.method.w_inv), float(cfg.method.w_floor), float(cfg.method.h_lamb))

    def clear_rings():
        for a in [a for a in vars(method) if a.startswith(("_zq", "_hq"))]:
            delattr(method, a)

    st = torch.random.get_rng_state()
    cst = torch.cuda.get_rng_state_all() if device.startswith("cuda") else None
    canon = [method.training_step(modules, v, device, y=None)[0] for v in views]
    clear_rings()
    torch.random.set_rng_state(st)
    if cst is not None:
        torch.cuda.set_rng_state_all(cst)
    ddp = [_ddp_loss(method, student, modules, v, *w, q_seed=0)[0] for v in views]

    ok, worst = True, 0.0
    for s, (tc, td) in enumerate(zip(canon, ddp)):
        for k in ("loss", "inv", "moment_kl", "h_moment_kl"):
            d = (tc[k] - td[k]).abs().item()
            worst = max(worst, d)
            ok = ok and torch.allclose(tc[k], td[k], atol=1e-5, rtol=1e-4)
        print(f"    step {s}: " + "  ".join(
            f"{k}|Δ|={(tc[k] - td[k]).abs().item():.2e}"
            for k in ("loss", "inv", "moment_kl", "h_moment_kl")))
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}  ({steps} steps, worst |Δ|={worst:.2e})")
    return ok


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[train_ddp_selftest] device={device}")
    N = 4
    base = ["method=floorssl", "frame=in1k_vits16", "num_classes=1000",
            "method.head_layers=2", "method.expander_dim=256",
            "method.z_floor_batch=view_mean", "method.h_floor_batch=view_mean",
            "method.floor_shrink=oas", "method.queue_steps=0", "method.h_d_slice=256"]

    def mc_views():
        return (torch.randn(N, 4, 3, 224, 224, device=device),   # Vg=4 globals @224
                torch.randn(N, 6, 3, 96, 96, device=device))     # Vl=6 locals @96

    def v4_views():
        return torch.randn(N, 4, 3, 224, 224, device=device)

    ok = True
    # the lm4s5b path: multicrop (lightly_mc) + swa=ema + cond_stream=all (default) + q=0
    print("· lm4s5b path (multicrop + swa=ema):")
    ok &= _check("multicrop+swa", base + ["+method.aug=lightly_mc", "method.Vg=4", "method.Vl=6",
                                          "+method.swa=ema"], mc_views, device)
    # guards: multicrop WITHOUT swa, and the V-view path (no multicrop) with swa
    print("· multicrop, no swa:")
    ok &= _check("multicrop", base + ["+method.aug=lightly_mc", "method.Vg=4", "method.Vl=6"],
                 mc_views, device)
    print("· V-view + swa=ema:")
    ok &= _check("v4+swa", base + ["+method.aug=lejepa", "+method.V=4", "+method.swa=ema"],
                 v4_views, device)

    def v6_views():
        return torch.randn(N, 6, 3, 224, 224, device=device)

    # the D-103 all-global ring shape (B2'/L2': lejepa V=6 + ring z-q3/h-q7 + swa, no OAS)
    print("· V6 + ring (q=3, h-q=7) + swa=ema, 3 steps:")
    ok &= _check_ring("v6+ring+swa", ["method=floorssl", "frame=in1k_vits16",
                                      "num_classes=1000", "method.head_layers=2",
                                      "method.expander_dim=256", "method.z_floor_batch=view_mean",
                                      "method.h_floor_batch=view_mean", "method.queue_steps=3",
                                      "method.h_queue_steps=7", "method.h_d_slice=256",
                                      "+method.aug=lejepa", "+method.V=6", "+method.swa=ema"],
                      v6_views, device)
    print(f"\n[train_ddp_selftest] {'ALL PASS' if ok else 'FAILURES ABOVE'}")
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
