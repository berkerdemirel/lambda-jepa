"""E12 (D-026) CPU dry-run — mechanics only, no data, no conclusions (D-022 precedent).

Per arm (A1/A2/A3/C1): build modules at the arm cfg, one training_step forward+backward on a
synthetic batch, assert finite loss/grads, then the arch() round-trip every extraction relies on:
rebuild each role from its dotted class+kwargs and load_state_dict(strict=True) — this is the test
that spec-norm parametrization keys (E12 p1) survive the ckpt schema unchanged.
"""
import importlib

import torch
from omegaconf import OmegaConf

from sslgap.methods import METHODS
from sslgap.methods.base import Frame

BASE = {"name": "lejepa", "lamb": 0.02, "V": 2, "proj_dim": 16, "emb_dim": 512, "drop_path": 0.1,
        "lr": 3e-4, "wd": 5e-2, "warmup_ep": 10, "eta_min": 1e-5, "grad_clip": 1.0}
ARMS = {  # amended 2026-07-11 (E12 card §Amendment): A2/A3 additive h-term, shipped z-side kept
    "e12a1": {"proj_depth": 0, "embed_calib": True},
    "e12a2": {"spec_norm": True, "embed_calib": True, "h_reg": "sigreg", "h_lamb": 0.0257},
    "e12a3": {"spec_norm": True, "embed_calib": True, "h_reg": "moment", "h_lamb": 0.4775},
    "e12c1": {"spec_norm": True, "embed_calib": True},
    # F-wave (D-027): f1/f2 = dose variants of a3 (mechanics identical, skipped here);
    # f3-f6 exercise the new terms + the h_start_ep gate
    "e12f3": {"spec_norm": True, "embed_calib": True, "h_reg": "spec_floor", "h_lamb": 0.4775},
    "e12f4": {"spec_norm": True, "embed_calib": True, "h_reg": "sigreg_std", "h_lamb": 0.1},
    "e12f5": {"spec_norm": True, "embed_calib": True, "h_reg": "moment_diag", "h_lamb": 0.4775},
    "e12f6": {"spec_norm": True, "embed_calib": True, "h_reg": "moment", "h_lamb": 0.4775,
              "h_start_ep": 10},
}


def rebuild(arch_role):
    mod, fn = arch_role["class"].rsplit(".", 1)
    return getattr(importlib.import_module(mod), fn)(**arch_role["kwargs"])


def main():
    frame = Frame(name="dry", model_name="vit_small_patch16_224", img_size=224,
                  dataset="imagenet100", data_root=None, epochs=1, seed=0, grad_clip=1.0,
                  num_workers=0, device="cpu")
    views = torch.randn(4, 2, 3, 224, 224)          # [N, V, C, H, W]
    y = torch.randint(0, 100, (4,))
    for tag, over in ARMS.items():
        cfg = OmegaConf.create({**BASE, **over})
        method = METHODS["lejepa"](cfg, frame)
        modules = method.build_modules()
        terms, feats, k = method.training_step(modules, views, "cpu", y=y)
        terms["loss"].backward()
        gn = torch.cat([p.grad.reshape(-1) for p in modules["encoder"].parameters()
                        if p.grad is not None]).norm()
        assert torch.isfinite(terms["loss"]) and torch.isfinite(gn), f"{tag}: non-finite"
        reg_key = "moment_kl" if over.get("floor") == "moment" else "sigreg"
        assert reg_key in terms, f"{tag}: missing term {reg_key}"
        if over.get("h_reg"):
            from sslgap.methods.lejepa import H_KEYS
            h_key = H_KEYS[over["h_reg"]]
            if over.get("h_start_ep", 0) > 0:      # f6: gate closed at epoch 0 — term must be absent
                assert h_key not in terms, f"{tag}: h_start_ep gate leaked {h_key}"
            else:
                assert h_key in terms and torch.isfinite(terms[h_key]), f"{tag}: bad h-term {h_key}"
        for role, spec in method.arch().items():
            twin = rebuild(spec)
            twin.load_state_dict(modules[role].state_dict(), strict=True)
        sd_keys = list(modules["projector"].state_dict())
        has_sn = any("parametrizations" in k_ for k_ in sd_keys)
        assert has_sn == bool(over.get("spec_norm")), f"{tag}: spec_norm keys mismatch"
        print(f"[dryrun] {tag} PASS  loss={terms['loss']:.4f} {reg_key}={terms[reg_key]:.4f} "
              f"inv={terms['inv']:.4f} enc_gnorm={gn:.3f} calib={getattr(method, '_calibrated', False)} "
              f"specnorm_keys={has_sn} feats={tuple(feats.shape)} k={k}", flush=True)
    print("[dryrun] ALL 4 ARMS PASS")


if __name__ == "__main__":
    main()
