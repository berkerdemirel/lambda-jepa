"""DeiT-lite — the supervised anchor (D-007; recipe D-022, Berker-directed 2026-07-09).

"Lite" is deliberate: plain CE on the CLS token, RRC+flip augs only (no mixup/cutmix/randaug/
label smoothing/distillation), house optimizer per the D-018 uniformity logic — a minimal
supervised reference row for the audit matrix, not a tuned competitor. h = last-layer CLS (the
classifier input, D-003v2: what a supervised ViT's tables probe); z.final = the 100-way logits.
Uses the trainer's y pass-through (training_step y kwarg — added for this method; SSL methods
ignore it). The online probe (LN+Linear on detached CLS) stays as the uniform monitor even
though it ~duplicates the classifier — same contract as every other method."""
import torch
import torch.nn as nn
import torch.nn.functional as F

from sslgap.data import ViewsDataset, supervised_stack
from sslgap.methods._common import house_scheduler, trunk_arch
from sslgap.methods.base import SSLMethod
from sslgap.models.backbones import build_vit_trunk


class DeiTLite(SSLMethod):
    name = "deitlite"

    def build_modules(self):
        trunk = build_vit_trunk(self.frame.model_name, self.frame.img_size,
                                drop_path_rate=self.cfg.drop_path)
        classifier = nn.Linear(384, self.cfg.num_classes)
        return nn.ModuleDict({"backbone": trunk, "classifier": classifier})

    def arch(self):
        return {"backbone": trunk_arch(self.frame, self.cfg.drop_path),
                "classifier": {"class": "torch.nn.Linear",
                               "kwargs": {"in_features": 384,
                                          "out_features": self.cfg.num_classes}}}

    def build_train_dataset(self):
        return ViewsDataset(self.frame.dataset, "train", V=1, img_size=self.frame.img_size,
                            data_root=self.frame.data_root,
                            transforms=[supervised_stack(self.frame.img_size)])

    def param_groups(self, modules):
        return [{"params": [p for m in modules.values() for p in m.parameters()],
                 "lr": self.cfg.lr, "weight_decay": self.cfg.wd}]

    def build_scheduler(self, optimizer, steps_per_epoch, total_steps):
        return house_scheduler(optimizer, steps_per_epoch, total_steps,
                               self.cfg.warmup_ep, self.cfg.eta_min)

    def training_step(self, modules, views, device, y=None):
        h = modules["backbone"].forward_features(views.flatten(0, 1))     # V=1: [N, T, 384]
        cls = h[:, 0]
        logits = modules["classifier"](cls)
        loss = F.cross_entropy(logits, y)
        with torch.no_grad():
            acc = (logits.argmax(1) == y).float().mean()
        probe_feats = cls.detach()               # monitor = audited h (CLS, D-003v2); image-major
        return ({"loss": loss, "ce": loss, "cls_acc": acc}, probe_feats, 1)

    @torch.inference_mode()
    def eval_features(self, modules, x, device):
        return modules["backbone"].forward_features(x)[:, 0]              # audited h (CLS)

    def probe_dim(self):
        return 384
