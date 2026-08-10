"""E27 Recipe v2 selftest (D-079a) — runs at the head of the smoke/pilot job, before any
training step: (1) dynamic_img_size @224 parity vs the static trunk (the S trunk must be
voas-equivalent at the landing resolution); (2) 96-px local forward shape; (3) the
view-to-mean inv == all-pairs x (V-1)/2V identity (the declared proportionality); (4) the
per-tap _ring refactor reproduces the D-064 inline semantics on fixed inputs; (5) one full
multicrop training_step: finite terms, ring lengths, backward, probe shapes."""
import sys

import torch
import torch.nn.functional as F
from omegaconf import OmegaConf

sys.path.insert(0, "/nfs/scistore19/locatgrp/bdemirel/ssl_project")
from sslgap.methods.base import Frame                       # noqa: E402
from sslgap.methods.floorssl import FloorSSL                # noqa: E402
from sslgap.models.backbones import build_vit_trunk         # noqa: E402

dev = "cuda" if torch.cuda.is_available() else "cpu"

# (1) + (2): dynamic pos-embed parity at the frame resolution, local shape at 96
torch.manual_seed(0)
t_static = build_vit_trunk("vit_small_patch16_224", 224).to(dev).eval()
t_dyn = build_vit_trunk("vit_small_patch16_224", 224, dynamic_img_size=True).to(dev).eval()
t_dyn.load_state_dict(t_static.state_dict())
x = torch.randn(4, 3, 224, 224, device=dev)
with torch.no_grad():
    d = (t_static.forward_features(x) - t_dyn.forward_features(x)).abs().max().item()
print(f"[selftest] dynamic@224 max|diff| = {d:.3e}")
assert d < 1e-4, "dynamic_img_size breaks @224 parity"
with torch.no_grad():
    s96 = tuple(t_dyn.forward_features(torch.randn(4, 3, 96, 96, device=dev)).shape)
print(f"[selftest] local@96 forward_features shape = {s96}")
assert s96[1] == 1 + 36 and s96[2] == 384

# (3): inv proportionality (V=10)
z = torch.randn(8, 10, 16, dtype=torch.float64)
ap = sum(F.mse_loss(z[:, u], z[:, w]) for u in range(10) for w in range(u + 1, 10)) / 45
vm = (z - z.mean(1, keepdim=True)).square().mean()
print(f"[selftest] all-pairs {ap:.8f} vs view-to-mean*2V/(V-1) {vm * 20 / 9:.8f}")
assert torch.allclose(ap, vm * 20 / 9, rtol=1e-9)

# (4) + (5): method-level
cfg = OmegaConf.create(dict(
    name="floorssl", w_inv=21.4, w_floor=49.6, h_lamb=1.89, z_floor="kl",
    z_floor_batch="view_mean", h_floor_batch="view_mean", z_d_slice=None, h_d_slice=None,
    queue_steps=3, h_queue_steps=None, floor_shrink=None, expander_hidden=2048,
    expander_dim=256, head_norm="none", head_layers=2, head_width=None, mlp_wd=0.05,
    grad_ckpt=False, drop_path=0.1, lr=1e-3, wd=5e-2, warmup_ep=10, eta_min=1e-5,
    aug="lejepa_mc", Vg=2, Vl=8, local_size=96, global_scale=[0.3, 1.0],
    local_scale=[0.05, 0.3], V=4))
frame = Frame(name="selftest", model_name="vit_small_patch16_224", img_size=224,
              dataset="imagenet1k", data_root=None, epochs=100, seed=0, grad_clip=1.0,
              num_workers=0, device=dev)
m = FloorSSL(cfg, frame)
mods = m.build_modules().to(dev)
assert m.probe_dim() == 384 and m.cond_h.d_slice == 128 and m.cond_h.d_draw == 128

xs = [torch.randn(128, 256) for _ in range(3)]
ref_q, outs_ref = [], []
for xi in xs:                       # the D-064 inline semantics, verbatim
    outs_ref.append(torch.cat([xi] + ref_q) if ref_q else xi)
    ref_q = [xi.detach()] + ref_q[:2]
outs_new = [m._ring("_tq", xi, 3) for xi in xs]
assert all(torch.equal(a, b) for a, b in zip(outs_ref, outs_new)), "ring refactor drifted"
del m._tq
print("[selftest] _ring == D-064 inline semantics over 3 fixed steps")

g = torch.randn(2, 2, 3, 224, 224, device=dev)
l = torch.randn(2, 8, 3, 96, 96, device=dev)
for _ in range(4):
    terms, pf, k = m.training_step(mods, (g, l), dev)
    assert all(torch.isfinite(v) for v in terms.values())
terms["loss"].backward()
assert k == 10 and tuple(pf.shape) == (20, 384)
assert len(m._zq) == 3 and len(m._hq) == 3
print(f"[selftest] multicrop step: loss {terms['loss'].item():.4f} "
      f"inv {terms['inv'].item():.4f} ring z/h = {len(m._zq)}/{len(m._hq)}")

# (6) grouped stream (D-087, wave 4): per-group rings fill; term = mean of two KLs;
# inv identical to the "all" path on the same weights/input (stream change only).
cfg_g = OmegaConf.merge(cfg, {"cond_stream": "grouped"})
mg = FloorSSL(cfg_g, frame)
mods_g = mg.build_modules().to(dev)
mods_g.load_state_dict(mods.state_dict())
for _ in range(4):
    terms_g, pf_g, k_g = mg.training_step(mods_g, (g, l), dev)
    assert all(torch.isfinite(v) for v in terms_g.values())
terms_g["loss"].backward()
assert k_g == 10 and tuple(pf_g.shape) == (20, 384)
assert all(len(getattr(mg, a)) == 3 for a in ("_zq0", "_zq1", "_hq0", "_hq1"))
assert not hasattr(mg, "_zq") and not hasattr(mg, "_hq")
print(f"[selftest] grouped step: loss {terms_g['loss'].item():.4f} "
      f"cond_z {terms_g['moment_kl'].item():.4f} cond_h {terms_g['h_moment_kl'].item():.4f} "
      f"rings z0/z1/h0/h1 = "
      f"{len(mg._zq0)}/{len(mg._zq1)}/{len(mg._hq0)}/{len(mg._hq1)}")
# inv parity across streams: seeded twin steps (drop_path draws must match) — the
# stream knob may only move the conditioner terms.
torch.manual_seed(123)
t_all, _, _ = m.training_step(mods, (g, l), dev)
torch.manual_seed(123)
t_grp, _, _ = mg.training_step(mods_g, (g, l), dev)
assert torch.allclose(t_grp["inv"], t_all["inv"], rtol=1e-5), \
    "grouped must not touch inv"

# (7) perview stream (D-087 sweep): 10 per-view KLs under OAS/no-ring; inv untouched.
cfg_p = OmegaConf.merge(cfg, {"cond_stream": "perview", "floor_shrink": "oas",
                              "queue_steps": 0, "h_queue_steps": None})
mp = FloorSSL(cfg_p, frame)
mods_p = mp.build_modules().to(dev)
mods_p.load_state_dict(mods.state_dict())
torch.manual_seed(123)
t_pv, pf_p, k_p = mp.training_step(mods_p, (g, l), dev)
t_pv["loss"].backward()
assert k_p == 10 and all(torch.isfinite(v) for v in t_pv.values())
assert torch.allclose(t_pv["inv"], t_all["inv"], rtol=1e-5), \
    "perview must not touch inv"
assert mp.cond_z.rho_last is not None, "perview cell must run OAS"
print(f"[selftest] perview step: loss {t_pv['loss'].item():.4f} "
      f"cond_z {t_pv['moment_kl'].item():.4f} rho {mp.cond_z.rho_last:.3f}")

# (8) perview + RING (D-088 e27pv cell): per-view rings fill to q at both taps.
cfg_pr = OmegaConf.merge(cfg, {"cond_stream": "perview"})   # base queue_steps=3, no shrink
mpr = FloorSSL(cfg_pr, frame)
mods_pr = mpr.build_modules().to(dev)
mods_pr.load_state_dict(mods.state_dict())
for _ in range(4):
    t_pr, _, _ = mpr.training_step(mods_pr, (g, l), dev)
    assert all(torch.isfinite(v) for v in t_pr.values())
t_pr["loss"].backward()
assert all(len(getattr(mpr, f"_zq{i}")) == 3 for i in range(10))
assert all(len(getattr(mpr, f"_hq{i}")) == 3 for i in range(10))
print(f"[selftest] perview+ring: loss {t_pr['loss'].item():.4f} rings 10x3 both taps")

# (9) SWA twin (D-095, the v10u+swa cell): teacher==student at build; at drop_path=0 and
# matched eval BN the SWA inv (x 2V/(V-1)) equals the legacy all-pairs inv EXACTLY (the
# anchor is then the batch's own mean); the running average matches hand math; the twin
# takes no grad; swa_k round-trips through extras.
cfg_s = OmegaConf.merge(cfg, {"aug": "lejepa", "V": 10, "drop_path": 0.0,
                              "swa": "uniform"})
ms = FloorSSL(cfg_s, frame)
mods_s = ms.build_modules().to(dev)
assert set(mods_s.keys()) == {"backbone", "projector", "teacher_backbone",
                              "teacher_projector"}
sd_s, sd_t = mods_s["backbone"].state_dict(), mods_s["teacher_backbone"].state_dict()
assert all(torch.equal(sd_s[k], sd_t[k]) for k in sd_s), "twin != student at build"

cfg_s0 = OmegaConf.merge(cfg, {"aug": "lejepa", "V": 10, "drop_path": 0.0})
ms0 = FloorSSL(cfg_s0, frame)
mods_s0 = ms0.build_modules().to(dev)
mods_s0.load_state_dict(
    {k: v for k, v in mods_s.state_dict().items() if not k.startswith("teacher_")})
u = torch.randn(2, 10, 3, 224, 224, device=dev)
for md in (mods_s, mods_s0):
    md.eval()                       # matched BN stats + no drop_path -> exact identity
t_swa, pf_s, k_s = ms.training_step(mods_s, u, dev)
t_leg, _, _ = ms0.training_step(mods_s0, u, dev)
assert k_s == 10 and tuple(pf_s.shape) == (20, 384)
assert torch.allclose(t_swa["inv"], t_leg["inv"], rtol=1e-4), \
    f"swa inv {t_swa['inv'].item():.6f} != legacy {t_leg['inv'].item():.6f} at init"
t_swa["loss"].backward()
assert all(p.grad is None for p in mods_s["teacher_backbone"].parameters())

ref = [p.detach().clone() for p in mods_s["backbone"].parameters()]
with torch.no_grad():
    for p in mods_s["backbone"].parameters():
        p.add_(1.0)
ms.post_step(mods_s, 0, 1)          # k=0: twin := student
with torch.no_grad():
    for p in mods_s["backbone"].parameters():
        p.add_(1.0)
ms.post_step(mods_s, 1, 1)          # k=1: twin = mean of the two states
pt = next(mods_s["teacher_backbone"].parameters())
exp = ref[0] + 1.5                  # mean(ref+1, ref+2)
assert torch.allclose(pt, exp, atol=1e-6), "SWA running average drifted from hand math"
assert ms.extras() == {"swa_k": 2}
ms.load_extras({"swa_k": 7})
assert ms._swa_k == 7
print(f"[selftest] swa: init-parity inv {t_swa['inv'].item():.4f} == legacy "
      f"{t_leg['inv'].item():.4f}; running avg exact; twin grad-free; k round-trips")

# (10) lightly_mc frame (D-095): the Lightly view geometry (2g+6l) under our loss —
# dynamic trunk engaged, mc branch consumes it, swa's mc path finite at V=8, OAS live.
cfg_lm = OmegaConf.merge(cfg, {"aug": "lightly_mc", "Vl": 6, "floor_shrink": "oas",
                               "queue_steps": 0, "h_queue_steps": None,
                               "swa": "uniform"})
mlm = FloorSSL(cfg_lm, frame)
mods_lm = mlm.build_modules().to(dev)
assert mlm._mc, "lightly_mc must build the dynamic trunk"
l6 = torch.randn(2, 6, 3, 96, 96, device=dev)
t_lm, pf_lm, k_lm = mlm.training_step(mods_lm, (g, l6), dev)
t_lm["loss"].backward()
assert k_lm == 8 and tuple(pf_lm.shape) == (16, 384)
assert all(torch.isfinite(v) for v in t_lm.values())
assert mlm.cond_z.rho_last is not None, "lightly_mc cell must run OAS"
assert all(p.grad is None for p in mods_lm["teacher_backbone"].parameters())
print(f"[selftest] lightly_mc+swa step: loss {t_lm['loss'].item():.4f} "
      f"inv {t_lm['inv'].item():.4f} V=8 rho {mlm.cond_z.rho_last:.3f}")

# (10b) 4-global geometry (D-097, the VISReg-matched B mirrors): the mc branch is
# Vg-agnostic — 4g+6l steps finitely with the right shapes.
cfg_l4 = OmegaConf.merge(cfg_lm, {"Vg": 4})
ml4 = FloorSSL(cfg_l4, frame)
mods_l4 = ml4.build_modules().to(dev)
g4 = torch.randn(2, 4, 3, 224, 224, device=dev)
t_l4, pf_l4, k_l4 = ml4.training_step(mods_l4, (g4, l6), dev)
t_l4["loss"].backward()
assert k_l4 == 10 and tuple(pf_l4.shape) == (20, 384)
assert all(torch.isfinite(v) for v in t_l4.values())
print(f"[selftest] 4g+6l step: loss {t_l4['loss'].item():.4f} V=10")
print("[selftest] ALL PASS")
