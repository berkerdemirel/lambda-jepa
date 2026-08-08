"""E27 lejepa multicrop-control selftest (D-082) — separate from e27_selftest.py so the
floorssl pilots' gate stays independent of control-lane code. Asserts: (1) encoder
dynamic_img_size @224 parity vs the static build; (2) two multicrop training_steps with
finite terms + backward; (3) probe conventions (emb tap, k=10); (4) the λ-convex loss
composition at λ=0.05 reproduces lamb*reg + (1-lamb)*inv on the step's own terms."""
import sys

import torch
from omegaconf import OmegaConf

sys.path.insert(0, "/nfs/scistore19/locatgrp/bdemirel/ssl_project")
from sslgap.methods.base import Frame                       # noqa: E402
from sslgap.methods.lejepa import LeJEPA, lejepa_encoder    # noqa: E402

dev = "cuda" if torch.cuda.is_available() else "cpu"

torch.manual_seed(0)
e_static = lejepa_encoder("vit_small_patch16_224", 224).to(dev).eval()
e_dyn = lejepa_encoder("vit_small_patch16_224", 224, dynamic_img_size=True).to(dev).eval()
e_dyn.load_state_dict(e_static.state_dict())
x = torch.randn(4, 3, 224, 224, device=dev)
with torch.no_grad():
    d = (e_static(x) - e_dyn(x)).abs().max().item()
print(f"[selftest-lej] dynamic@224 max|diff| = {d:.3e}")
assert d < 1e-4, "dynamic_img_size breaks the encoder @224 parity"

cfg = OmegaConf.create(dict(
    name="lejepa", lamb=0.05, V=4, proj_dim=16, emb_dim=512, drop_path=0.1, lr=5e-4,
    wd=5e-2, warmup_ep=1, eta_min=5e-7, grad_clip=None, aug="lejepa_mc", Vg=2, Vl=8,
    local_size=96, global_scale=[0.3, 1.0], local_scale=[0.05, 0.3]))
frame = Frame(name="selftest", model_name="vit_small_patch16_224", img_size=224,
              dataset="imagenet1k", data_root=None, epochs=100, seed=0, grad_clip=None,
              num_workers=0, device=dev)
m = LeJEPA(cfg, frame)
mods = m.build_modules().to(dev)
assert m.probe_dim() == 512

g = torch.randn(2, 2, 3, 224, 224, device=dev)
l = torch.randn(2, 8, 3, 96, 96, device=dev)
for _ in range(2):
    terms, pf, k = m.training_step(mods, (g, l), dev)
    assert all(torch.isfinite(v) for v in terms.values())
terms["loss"].backward()
assert k == 10 and tuple(pf.shape) == (20, 512)
recon = 0.05 * terms["sigreg"] + 0.95 * terms["inv"]
assert torch.allclose(terms["loss"], recon, rtol=1e-5), "λ-convex composition drifted"
print(f"[selftest-lej] mc step: loss {terms['loss'].item():.4f} "
      f"sigreg {terms['sigreg'].item():.4f} inv {terms['inv'].item():.4f}")
print("[selftest-lej] ALL PASS")
