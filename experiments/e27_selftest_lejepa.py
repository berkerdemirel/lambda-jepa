"""E27 lejepa multicrop-control selftest (D-082; D-092 Lightly-replication sections added
2026-08-10) — separate from e27_selftest.py so the floorssl pilots' gate stays independent
of control-lane code. Asserts: (1) encoder dynamic_img_size @224 parity vs the static
build; (2) two multicrop training_steps with finite terms + backward; (3) probe
conventions (emb tap, k=10); (4) the λ-convex loss composition at λ=0.05 reproduces
lamb*reg + (1-lamb)*inv on the step's own terms; (5) D-092: the lightly_mc view builders
carry the solarize asymmetry (global-2 only); (6) D-092: mc_form=lightly semantics —
bare-CLS anatomy (probe 384, V=8), inv == hand-computed locals-to-globals-mean MSE
(eval-mode determinism), λ-convex composition holds, backward finite; (7) the legacy
control path is untouched by the new keys (same cfg → same seeded step terms)."""
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

# --- (5) D-092: lightly_mc solarize asymmetry lives on global-2 only, at TRUE p --------
from sslgap.data import _bench_view  # noqa: E402
g1r, g2r = repr(_bench_view(224, (0.3, 1.0), 0.5, 0.0)), repr(_bench_view(224, (0.3, 1.0), 0.5, 0.2))
assert "RandomSolarize" not in g1r, "solarize leaked into global-1"
assert "RandomSolarize(p=0.2" in g2r.replace(" ", ""), "global-2 solarize missing or wrong p"
assert "RandomApply(RandomSolarize" not in g2r.replace(" ", ""), \
    "solarize is RandomApply-wrapped — effective p halves (the house-stack erratum)"

# --- (6) D-092: mc_form=lightly — bare-CLS anatomy + loss semantics --------------------
torch.manual_seed(0)
cfgL = OmegaConf.create(dict(
    name="lejepa", lamb=0.05, V=4, proj_dim=64, emb_dim=0, drop_path=0.1, lr=5e-4,
    wd=5e-2, warmup_ep=1, eta_min=5e-7, grad_clip=None, aug="lightly_mc",
    mc_form="lightly", n_slices=1024, Vg=2, Vl=6, local_size=96,
    global_scale=[0.3, 1.0], local_scale=[0.05, 0.3]))
mL = LeJEPA(cfgL, frame)
modsL = mL.build_modules().to(dev).eval()          # eval: drop_path off, BN on running stats
assert mL.probe_dim() == 384
gL = torch.randn(2, 2, 3, 224, 224, device=dev)
lL = torch.randn(2, 6, 3, 96, 96, device=dev)
termsL, pfL, kL = mL.training_step(modsL, (gL, lL), dev)
assert kL == 8 and tuple(pfL.shape) == (16, 384)
assert all(torch.isfinite(v) for v in termsL.values())
termsL["loss"].backward()
with torch.no_grad():
    embL = torch.cat([modsL["encoder"](gL.flatten(0, 1)).reshape(2, 2, -1),
                      modsL["encoder"](lL.flatten(0, 1)).reshape(2, 6, -1)], 1).flatten(0, 1)
    projL = modsL["projector"](embL).reshape(2, 8, -1).transpose(0, 1)
    hand = (projL[:2].mean(0) - projL[2:]).square().mean()
assert torch.allclose(termsL["inv"], hand, rtol=1e-4), "lightly inv semantics drifted"
reconL = 0.05 * termsL["sigreg"] + 0.95 * termsL["inv"]
assert torch.allclose(termsL["loss"], reconL, rtol=1e-5), "lightly λ-convex drifted"
print(f"[selftest-lej] lightly step: loss {termsL['loss'].item():.4f} "
      f"sigreg {termsL['sigreg'].item():.4f} inv {termsL['inv'].item():.4f}")

# --- (7) legacy-path semantics pinned: all-views-mean inv, unchanged by D-092 ----------
torch.manual_seed(0)
m2 = LeJEPA(cfg, frame)
mods2 = m2.build_modules().to(dev).eval()
t2, _, _ = m2.training_step(mods2, (g, l), dev)
with torch.no_grad():
    emb2 = torch.cat([mods2["encoder"](g.flatten(0, 1)).reshape(2, 2, -1),
                      mods2["encoder"](l.flatten(0, 1)).reshape(2, 8, -1)], 1).flatten(0, 1)
    proj2 = mods2["projector"](emb2).reshape(2, 10, -1).transpose(0, 1)
    hand2 = (proj2.mean(0) - proj2).square().mean()
assert torch.allclose(t2["inv"], hand2, rtol=1e-4), "legacy all-views inv drifted"
print("[selftest-lej] ALL PASS")
