"""The faithful-vs-comparable contract (PROTOCOL §2), in code.

Frame owns what must be IDENTICAL across methods within a rung: backbone topology, dataset, epoch
budget, seed, checkpoint cadence, online-probe monitor, logging schema, precision, resume. The
generic loop in experiments/train.py implements the frame once.

Recipe (everything method-specific) lives in the SSLMethod subclass: augs/views, optimizer groups,
schedules, heads, loss terms. Every deviation from the paper/donor recipe is documented in the
subclass docstring and the method dossier (PORT_NOTES)."""
from abc import ABC, abstractmethod
from dataclasses import dataclass

import torch.nn as nn


@dataclass
class Frame:
    name: str                 # e.g. "toy" — run_id prefix
    model_name: str           # timm trunk
    img_size: int
    dataset: str              # imagenette | imagenet
    data_root: str | None
    epochs: int
    seed: int
    grad_clip: float | None   # frame default 1.0; a recipe may override ONLY with a DECISIONS row
    num_workers: int
    device: str = "cuda"

    def cadence(self):
        """Quarter-point checkpoint epochs (1-indexed), always including the last epoch."""
        qs = sorted({max(1, round(self.epochs * q / 4)) for q in (1, 2, 3, 4)})
        return qs


class SSLMethod(ABC):
    """One SSL method's recipe. The trainer calls these hooks; the method never owns the loop."""

    name: str

    def __init__(self, cfg, frame: Frame):
        self.cfg = cfg
        self.frame = frame

    # --- construction -------------------------------------------------------------------------
    @abstractmethod
    def build_modules(self) -> nn.ModuleDict:
        """Roles -> modules (canonical role names: backbone, embed, projector, predictor, decoder,
        teacher_backbone, teacher_projector). Construction ORDER matters for seed-faithful ports."""

    @abstractmethod
    def build_train_dataset(self):
        """Faithful views/masking. Items: (views [V,C,H,W], y)."""

    @abstractmethod
    def arch(self) -> dict:
        """Role -> {"class": dotted callable, "kwargs": {...}} — rebuilds each module without the
        trainer (ckpt schema v1)."""

    # --- optimization -------------------------------------------------------------------------
    @abstractmethod
    def param_groups(self, modules) -> list:
        """Optimizer param groups for the method's modules (probe group is added by the trainer)."""

    @abstractmethod
    def build_scheduler(self, optimizer, steps_per_epoch, total_steps):
        ...

    # --- the step -----------------------------------------------------------------------------
    @abstractmethod
    def training_step(self, modules, views, device) -> tuple[dict, "torch.Tensor"]:
        """views [N,V,C,H,W] -> ({term: tensor incl. "loss"}, probe_feats [N*V, D] DETACHED)."""

    @abstractmethod
    def eval_features(self, modules, x, device):
        """Deterministic features for the online-probe monitor, x [B,C,H,W] -> [B, D]."""

    @abstractmethod
    def probe_dim(self) -> int:
        ...

    def post_step(self, modules) -> dict:
        """EMA updates / centering; returns monitor scalars. Default: nothing."""
        return {}

    def collapse_monitors(self, terms) -> dict:
        """Cheap per-step collapse signals. Default: none beyond the loss terms."""
        return {}

    def extras(self) -> dict:
        """Non-module checkpoint state (centers, queues). Default: none."""
        return {}
