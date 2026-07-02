"""Shared recipe utilities: the house toy-rung optimizer schedule (D-012) and arch helpers."""
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR


def house_scheduler(optimizer, steps_per_epoch, total_steps, warmup_ep, eta_min):
    """Toy-rung schedule: linear warmup (start 0.01x) then cosine to eta_min (house rule:
    eta_min <= lr/20)."""
    warmup = steps_per_epoch * warmup_ep
    s1 = LinearLR(optimizer, start_factor=0.01, total_iters=warmup)
    s2 = CosineAnnealingLR(optimizer, T_max=total_steps - warmup, eta_min=eta_min)
    return SequentialLR(optimizer, schedulers=[s1, s2], milestones=[warmup])


def trunk_arch(frame, drop_path):
    return {"class": "sslgap.models.backbones.build_vit_trunk",
            "kwargs": {"model_name": frame.model_name, "img_size": frame.img_size,
                       "dynamic_img_size": False, "drop_path_rate": drop_path}}


def ema_momentum(step, total_steps, base, end=1.0):
    """Cosine EMA momentum schedule base -> end (BYOL/DINO convention)."""
    import math
    return end - (end - base) * (math.cos(math.pi * step / total_steps) + 1) / 2
