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
print("[selftest] ALL PASS")
