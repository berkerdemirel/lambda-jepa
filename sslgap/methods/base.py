"""Frame (what is identical across methods within a comparison) and the SSLMethod recipe interface."""
from abc import ABC, abstractmethod
from dataclasses import dataclass

import torch.nn as nn

@dataclass
class Frame:
    name: str
    model_name: str
    img_size: int
    dataset: str
    data_root: str | None
    epochs: int
    seed: int
    grad_clip: float | None
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

    @abstractmethod
    def param_groups(self, modules) -> list:
        """Optimizer param groups for the method's modules (probe group is added by the trainer)."""

    @abstractmethod
    def build_scheduler(self, optimizer, steps_per_epoch, total_steps):
        ...

    @abstractmethod
    def training_step(self, modules, batch_x, device, y=None) -> tuple[dict, "torch.Tensor", int]:
        """batch_x = the dataset item's x part (tensor or nested tuple of tensors, on device) ->
        ({term: tensor incl. "loss"}, probe_feats [N*k, D] DETACHED, k = label repeats).
        probe_feats MUST be image-major (img0 x k, img1 x k, ...): the trainer aligns labels via
        y.repeat_interleave(k). View-major output silently trains the probe on wrong labels."""

    @abstractmethod
    def eval_features(self, modules, x, device):
        """Deterministic features for the online-probe monitor, x [B,C,H,W] -> [B, D]."""

    @abstractmethod
    def probe_dim(self) -> int:
        ...

    def post_step(self, modules, step, total_steps) -> dict:
        """EMA updates / centering; returns monitor scalars. Default: nothing."""
        return {}

    def on_epoch_start(self, modules, epoch):
        """Per-epoch recipe hooks (e.g. DINO freezes prototypes in epoch 0). Default: nothing."""

    def train_mode(self, modules):
        """Set train/eval per module at epoch start. Default: everything train. EMA-teacher
        methods with stochastic-depth students override to keep teachers eval (DINO/I-JEPA —
        the sslx control kept teachers eval; BYOL keeps its teacher in train mode: the target
        projector's BN uses batch statistics per the paper)."""
        for m in modules.values():
            m.train()

    def extras(self) -> dict:
        """Non-module checkpoint state (centers, queues). Default: none."""
        return {}

    def load_extras(self, extras):
        """Restore extras() state on resume. Default: nothing."""
