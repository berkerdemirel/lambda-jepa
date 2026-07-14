"""Adapter↔trainer coherence self-test: for each native checkpoint, assemble branches through
from_native, run the extraction path, and assert the method's loss space recomputed from
independently rebuilt trainer modules matches the taps. Certifies the assemblers BEFORE any
extraction sweep (the predecessor-project lesson: setup mistaken for results). fp32, no autocast,
so mismatches are adapter bugs, not precision.

  sbatch slurm/selftest.sbatch outputs/toy.lejepa.s0_ep150.pt outputs/toy.vicreg.s0_ep150.pt
"""
import sys

import torch
import torch.nn.functional as F

from sslgap.ckpt import adapters
from sslgap.extract.extractor import _batch_spaces
from sslgap.models.vitops import vit_tokens

DEV = "cuda"

EXPECT_H = {"simclr": "student.h.cls", "vicreg": "student.h.cls", "byol": "student.h.cls",  # D-036: projector input (was GAP)
            "dino": "teacher.h.cls", "mae": "student.h.gap", "ijepa": "teacher.h.gap",
            "lejepa": "student.z.embed"}


def rebuild(ck):
    """Trainer modules straight from the arch blocks — the independent side of the comparison."""
    mods = {}
    for role, spec in ck["arch"].items():
        m = adapters._resolve(spec["class"])(**spec["kwargs"])
        m.load_state_dict(ck["modules"][role])
        mods[role] = m.to(DEV).eval().requires_grad_(False)
    return mods


def close(a, b, what):
    assert torch.allclose(a, b, atol=1e-4, rtol=1e-4), \
        f"{what}: max|delta|={(a - b).abs().max().item():.3e}"


def check_simclr(loaded, mods, x, sp):
    cls = mods["backbone"].forward_features(x)[:, 0]
    close(sp["student.z.proj.out"], mods["projector"](cls), "proj.out")


def check_byol(loaded, mods, x, sp):
    cls = mods["backbone"].forward_features(x)[:, 0]
    close(sp["student.z.pred.out"], mods["predictor"](mods["projector"](cls)), "pred.out")
    tcls = mods["teacher_backbone"].forward_features(x)[:, 0]
    close(sp["teacher.z.proj.out"], mods["teacher_projector"](tcls), "teacher proj.out")


def check_dino(loaded, mods, x, sp):
    for br, bb, hd in (("student", "backbone", "projector"),
                       ("teacher", "teacher_backbone", "teacher_projector")):
        cls = mods[bb].forward_features(x)[:, 0]
        close(mods[hd].last(F.normalize(sp[f"{br}.z.dino.bottleneck"], dim=-1)),
              mods[hd](cls), f"{br} prototype logits recomputed from bottleneck tap")


def check_mae(loaded, mods, x, sp):
    dec = mods["decoder"]
    seq = mods["backbone"].forward_features(x)
    B, n = x.shape[0], seq.shape[1] - 1
    direct = dec(seq, torch.arange(n, device=x.device).expand(B, -1), n)
    z = dec.embed(seq)
    z = dec.blocks(z + dec.pos.to(z.dtype))
    close(dec.out(dec.norm(z))[:, 1:], direct, "mask-ratio-0 sequence == decoder's own pass")
    close(sp["student.z.dec.tap8"], z[:, 1:].mean(1), "dec.tap8 == block-8 patch mean")


def check_ijepa(loaded, mods, x, sp):
    hd = loaded.branches["student"].heads
    B = x.shape[0]
    close(sp["student.z.pred.out"], hd(x)["pred.out"], "pred.out determinism (fixed mask)")
    ctx_idx = hd.ctx_idx.expand(B, -1)
    ctx = vit_tokens(mods["backbone"], x, keep=ctx_idx)[:, 1:]
    manual = mods["predictor"](ctx, ctx_idx, hd.tgt_idx.expand(B, -1, -1)).mean((1, 2))
    close(sp["student.z.pred.out"], manual, "pred.out == trainer modules under the same mask")
    assert not set(hd.ctx_idx.flatten().tolist()) & set(hd.tgt_idx.flatten().tolist()), \
        "context/target overlap"


def check_lejepa(loaded, mods, x, sp):
    emb = mods["encoder"](x)                      # timm classifier forward = Linear(cls) = z.embed
    close(sp["student.z.embed"], emb, "z.embed == encoder classifier output")
    close(sp["student.z.proj.out"], mods["projector"](emb), "proj.out")


CHECKS = {"simclr": check_simclr, "vicreg": check_simclr, "byol": check_byol,
          "dino": check_dino, "mae": check_mae, "ijepa": check_ijepa, "lejepa": check_lejepa}


def main(paths):
    for p in paths:
        ck = torch.load(p, map_location="cpu", weights_only=False)
        method = ck["method"]
        loaded = adapters.load("native", p, run_id="selftest").eval_(DEV)
        assert loaded.provenance["h_space"] == EXPECT_H[method], loaded.provenance["h_space"]
        assert loaded.provenance["h_space"].startswith(loaded.probed_branch + ".")
        torch.manual_seed(0)
        x = torch.randn(8, 3, ck["frame"]["img_size"], ck["frame"]["img_size"], device=DEV)
        with torch.inference_mode():
            sp = _batch_spaces(loaded, x, h_layers=(3, 6, 9, 12))
            sp = {k: v.to(DEV) for k, v in sp.items()}
            for k, v in sp.items():
                assert torch.isfinite(v).all(), f"{method} {k} non-finite"
            assert loaded.provenance["h_space"] in sp
            CHECKS[method](loaded, rebuild(ck), x, sp)
        null = adapters.load("native", p, run_id="selftest.null", random_init=True).eval_(DEV)
        with torch.inference_mode():
            spn = _batch_spaces(null, x, h_layers=())
        assert all(torch.isfinite(v).all() for v in spn.values()), f"{method} null non-finite"
        zs = sorted(k for k in sp if ".z." in k)
        print(f"[selftest] {method} OK ({p}): {len(sp)} spaces | z: {', '.join(zs)}")
    print(f"[selftest] ALL OK ({len(paths)} ckpts)")


if __name__ == "__main__":
    main(sys.argv[1:])
