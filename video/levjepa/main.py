import os

import hydra
import lightning as pl
import stable_pretraining as spt
import stable_pretraining.optim.utils as spt_optim_utils
import torch
from einops import rearrange
from hydra.core.hydra_config import HydraConfig
from hydra.utils import to_absolute_path
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import WandbLogger
from omegaconf import DictConfig, ListConfig, OmegaConf

from callbacks import WeightDecayUpdater, WeightEMA
import module as vit_models
from data.gpu_views import photometrics
from data.loader import build_loader
from module import SIGReg
from lambdajepa_reg import K710Probe, Ring, ShareLogger, SACReg, all_gather_rows

torch.set_float32_matmul_precision("high")


def _is_bias_or_norm_param(name: str, param: torch.nn.Parameter) -> bool:
    name_lower = name.lower()
    return "bias" in name_lower or "norm" in name_lower or param.ndim == 1


spt_optim_utils.is_bias_or_norm_param = _is_bias_or_norm_param


def build_model(cfg: DictConfig):
    factory = getattr(vit_models, cfg.model.name)
    return factory(
        img_size=cfg.model.img_size,
        patch_size=cfg.model.patch_size,
        num_frames=cfg.model.num_frames,
        tubelet_size=cfg.model.tubelet_size,
        use_rope=cfg.model.get("use_rope", True),
        token_drop_rate=cfg.model.get("token_drop_rate", 0.0),
        attn_mode=cfg.model.get("attn_mode", "full"),
    )


def build_projector(cfg: DictConfig, encoder: torch.nn.Module):
    return vit_models.Projector(
        input_dim=encoder.embed_dim,
        hidden_dim=cfg.projector.hidden_dim,
        output_dim=cfg.projector.output_dim,
        norm_layer=torch.nn.BatchNorm1d,
    )


def resolve_data_spec(spec):
    """Resolve `data.<split>` (a path, or a list of source mappings) to absolute paths."""
    if isinstance(spec, (DictConfig, ListConfig)):
        spec = OmegaConf.to_container(spec, resolve=True)
    if isinstance(spec, dict) and ("mp4_base" in spec or "clipfile_base" in spec):
        return spec                                   # mp4-backed dataset (data/mp4_loader.py)
    if isinstance(spec, str):
        return to_absolute_path(spec)
    if isinstance(spec, dict):
        spec = [spec]
    resolved = []
    for entry in spec:
        if isinstance(entry, str):
            entry = {"path": entry}
        else:
            entry = dict(entry)
        entry["path"] = to_absolute_path(entry["path"])
        resolved.append(entry)
    return resolved


def build_video_loader(cfg: DictConfig, split: str, shuffle: bool, hflip: bool):
    aug = cfg.augmentation
    return build_loader(
        resolve_data_spec(cfg.data[split]),
        batch_size=cfg.loader.batch_size,
        num_workers=cfg.loader.num_workers,
        num_frames=cfg.model.num_frames,
        frame_stride=cfg.frame_stride,
        crop_size=cfg.model.img_size,
        local_size=aug.local_size,
        local_crops_number=aug.local_crops_number,
        global_crops_scale=aug.global_crops_scale,
        local_crops_scale=aug.local_crops_scale,
        hflip=hflip,
        color_jitter_prob=aug.color_jitter_prob,
        grayscale_prob=aug.grayscale_prob,
        gaussian_blur_prob=aug.gaussian_blur_prob,
        color_jitter_hue=aug.get("color_jitter_hue", 0.1),
        normalize_on_gpu=cfg.loader.get("normalize_on_gpu", False),
        shuffle=shuffle,
        drop_last=split == "train",
        pin_memory=cfg.loader.pin_memory,
        persistent_workers=cfg.loader.persistent_workers,
        prefetch_factor=cfg.loader.prefetch_factor,
        random_crop=split == "train",
        views=cfg.augmentation.get("views"),
        photometrics_on_gpu=aug.get("photometrics_on_gpu", False),
        # Small datasets need many clips per video per epoch; val stays at 1.
        clips_per_video=(cfg.loader.get("clips_per_video", 1) if split == "train" else 1),
        stage_local=cfg.data.get("stage_local", ""),
        stage_limit=int(cfg.data.get("stage_limit", 0)),
    )


_IMAGENET_MEAN = (0.485, 0.456, 0.406)
_IMAGENET_STD = (0.229, 0.224, 0.225)


def to_float_normalized(x: torch.Tensor) -> torch.Tensor:
    """uint8 -> float32 in [0,1] -> ImageNet-normalized (loader.normalize_on_gpu).

    No-op on float32 input, so runs with normalize_on_gpu=false are unaffected.
    """
    if x.dtype != torch.uint8:
        return x
    x = x.to(torch.float32).div_(255.0)
    mean = torch.as_tensor(_IMAGENET_MEAN, device=x.device, dtype=x.dtype).view(-1, 1, 1)
    std = torch.as_tensor(_IMAGENET_STD, device=x.device, dtype=x.dtype).view(-1, 1, 1)
    return x.sub_(mean).div_(std)


def multiview_forward(self, batch, stage):
    global_frame = to_float_normalized(batch["global_frame"])
    local_frames = to_float_normalized(batch["local_frames"])
    batch_size = global_frame.shape[0]

    global_tokens = self.encoder(rearrange(global_frame, "b t c h w -> b c t h w"))
    global_cls = global_tokens[:, 0].unsqueeze(1)

    local_tokens = self.encoder(rearrange(local_frames, "b v t c h w -> (b v) c t h w"))
    local_cls = rearrange(local_tokens[:, 0], "(b v) d -> b v d", b=batch_size)

    embeddings = self.projector(torch.cat([global_cls, local_cls], dim=1))
    global_emb = embeddings[:, :1]

    output = {}
    output["pred_loss"] = (global_emb - embeddings).pow(2).mean()
    output["sigreg_loss"] = self.sigreg(rearrange(embeddings, "b v d -> v b d"))
    output["loss"] = output["pred_loss"] + self.sigreg_weight * output["sigreg_loss"]
    self.log_dict(
        {
            f"{stage}/pred_loss": output["pred_loss"].detach(),
            f"{stage}/sigreg_loss": output["sigreg_loss"].detach(),
            f"{stage}/loss": output["loss"].detach(),
        },
        on_step=True,
        on_epoch=True,
        sync_dist=True,
    )

    return output


def lambdajepa_terms(self, batch, seed, distributed=True):
    """The house v6 recipe on the donor's modules (see lambdajepa_reg docstring): V global views of a
    clip through the one encoder; inv = view-to-mean MSE at the projector output on the all-pairs
    scale (== all-pairs mean MSE exactly); the conditioner on the per-clip VIEW CENTERS at Z and
    at H (CLS at the encoder output), global batch under DDP, rings per tap. Returns the loss
    terms and the per-view CLS / projector outputs (the share logger's cloud energies). With
    augmentation.photometrics_on_gpu the workers ship crops and the photometric ops run here per (clip, view),
    seeded per rank and micro-batch (fixed in the share measurement, whose seed is fixed)."""
    views = batch["views"]                                            # (b, V, t, c, h, w) uint8
    if self.gpu_photometrics:
        views = photometrics(views, seed * max(self.trainer.world_size, 1) + self.global_rank)
    views = to_float_normalized(views)
    N, V = views.shape[:2]
    cls = self.encoder(rearrange(views, "b v t c h w -> (b v) c t h w"))[:, 0].reshape(N, V, -1)
    z = self.projector(cls)                                           # (b, V, d)
    inv = (z - z.mean(1, keepdim=True)).pow(2).mean() * (2 * V / (V - 1))
    if distributed and getattr(self, "probe_k710", False):            # the online K710 probe reads the detached centers
        self._probe_h, self._probe_y = cls.mean(1).detach(), batch["label"].to(cls.device)
    zin = all_gather_rows(z.mean(1)) if distributed else z.mean(1)    # per-clip centers
    hin = all_gather_rows(cls.mean(1)) if distributed else cls.mean(1)
    reg_z = self.cond_z(self.ring("z", zin, self.queue_steps), seed=seed)
    reg_h = self.cond_h(self.ring("h", hin, self.h_queue_steps), seed=seed + 10_007)
    return {"inv": inv, "moment_kl": reg_z, "h_moment_kl": reg_h}, cls, z


def lambdajepa_forward(self, batch, stage):
    # rank-shared slice seed per micro-batch (global_step and batch_idx agree across ranks)
    accum = max(int(getattr(self.trainer, "accumulate_grad_batches_", 1)), 1)
    seed = 1_000_003 * self.base_seed + self.trainer.global_step * accum + batch.get("batch_idx", 0) % accum
    terms, _, _ = lambdajepa_terms(self, batch, seed)
    output = dict(terms)
    output["loss"] = self.w_inv * terms["inv"] + self.w_z * terms["moment_kl"] + self.w_h * terms["h_moment_kl"]
    self.log_dict({f"{stage}/{k}": v.detach() for k, v in output.items()}
                  | {f"{stage}/trace_z": self.cond_z.trace_last, f"{stage}/trace_h": self.cond_h.trace_last},
                  on_step=True, on_epoch=True, sync_dist=True)
    return output


@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg: DictConfig):
    # checkpoints under the hydra run dir (the shared file system): stable_pretraining otherwise redirects every ModelCheckpoint to
    # ~/.cache/stable-pretraining/runs/... (hundreds of MB per epoch on the home directory)
    spt.set(cache_dir=os.path.join(HydraConfig.get().runtime.output_dir, "spt_cache"))
    train_loader = build_video_loader(
        cfg,
        split="train",
        shuffle=True,
        hflip=cfg.augmentation.hflip,
    )
    val_loader = build_video_loader(cfg, split="val", shuffle=False, hflip=False)
    data = spt.data.DataModule(train=train_loader, val=val_loader)

    encoder = build_model(cfg)
    projector = build_projector(cfg, encoder)

    world_size = cfg.trainer.devices * cfg.trainer.num_nodes
    # spt.Module does manual optimization, so Lightning rejects
    # Trainer(accumulate_grad_batches>1); spt reads the count from the trainer's
    # `accumulate_grad_batches_` attribute, set after the Trainer is built.
    accum = max(int(cfg.get("accumulate_grad_batches", 1)), 1)
    # scheduler.interval="step" advances per *optimizer* step, so total_steps must
    # be in optimizer-step units (batches / accum).
    batches_per_epoch = len(train_loader.dataset) // world_size // cfg.loader.batch_size
    steps_per_epoch = batches_per_epoch // accum
    total_steps = cfg.trainer.max_epochs * steps_per_epoch

    loss_type = cfg.loss.get("type", "sigreg")
    if loss_type == "lambdajepa":
        r = cfg.loss.lambdajepa
        parts = dict(
            forward=lambdajepa_forward,
            cond_z=SACReg(d_slice=r.z_d_slice, eps=r.eps),
            cond_h=SACReg(d_slice=r.h_d_slice, eps=r.eps),
            ring=Ring(), queue_steps=int(r.queue_steps), h_queue_steps=int(r.h_queue_steps),
            w_inv=float(r.w_inv), w_z=float(r.w_z), w_h=float(r.w_h), base_seed=int(cfg.get("seed", 0)),
            gpu_photometrics=bool(cfg.augmentation.get("photometrics_on_gpu", False)),
            probe_k710=bool(cfg.get("probe_k710")),
        )
    else:
        parts = dict(forward=multiview_forward, sigreg=SIGReg(**cfg.loss.sigreg.kwargs),
                     sigreg_weight=cfg.loss.sigreg.weight)
    module = spt.Module(
        encoder=encoder,
        projector=projector,
        **parts,
        optim={
            "optimizer": dict(cfg.optimizer),
            "scheduler": {
                "type": cfg.scheduler.name,
                "total_steps": total_steps,
                "peak_step": cfg.scheduler.peak_step,
                "start_factor": cfg.scheduler.start_lr / cfg.optimizer.lr,
                "end_lr": cfg.scheduler.end_lr,
            },
            "interval": cfg.scheduler.interval,
        },
        hparams=OmegaConf.to_container(cfg, resolve=True),
    )

    logger = None
    if cfg.wandb.enabled:
        logger = WandbLogger(**OmegaConf.to_container(cfg.wandb.config, resolve=True))
        # No logger access here: wandb.init fires at the first `.experiment` access, and the Manager injects the
        # resumed run id (from the working directory's wandb_resume.json) only inside its __call__ — the donor's
        # log_hyperparams at this point opened a fresh wandb run on every resume.
        # The config reaches wandb through the Manager (module.hparams) at fit start.

    callbacks = []
    if cfg.weight_decay_scheduler.enabled:
        callbacks.append(
            WeightDecayUpdater(
                schedule_type=cfg.weight_decay_scheduler.schedule_type,
                start_value=cfg.weight_decay_scheduler.start_value,
                end_value=cfg.weight_decay_scheduler.end_value,
                param_group_indices=list(
                    cfg.weight_decay_scheduler.param_group_indices
                ),
            )
        )
    if cfg.ema.enabled:
        ema_cfg = OmegaConf.to_container(cfg.ema, resolve=True)
        ema_cfg.pop("enabled")
        callbacks.append(WeightEMA(**ema_cfg))
    if loss_type == "lambdajepa" and cfg.loss.lambdajepa.get("share_log_every"):
        r = cfg.loss.lambdajepa
        callbacks.append(ShareLogger(
            terms_fn=lambdajepa_terms, weights={"inv": float(r.w_inv), "moment_kl": float(r.w_z), "h_moment_kl": float(r.w_h)},
            dataset=train_loader.dataset, collate_fn=train_loader.collate_fn, bs=int(r.get("share_log_bs") or 128),
            every=int(r.share_log_every), seed=int(cfg.get("seed", 0))))
    probe = None
    if loss_type == "lambdajepa" and cfg.get("probe_k710"):
        # +probe_k710=<n> : the online K710 linear probe on the detached h centers, scored on n held-out clips per epoch
        probe = K710Probe(dataset=train_loader.dataset, collate_fn=train_loader.collate_fn,
                          num_classes=len(train_loader.dataset.classes), n_eval=int(cfg.probe_k710), seed=int(cfg.get("seed", 0)))
        callbacks.append(probe)
    if cfg.get("keep"):
        # +keep.epochs=[239] +keep.best=true : named checkpoints next to the rolling last.ckpt (pair with checkpoint.save_top_k=0)
        from lambdajepa_reg import CheckpointKeeper
        callbacks.append(CheckpointKeeper(probe=probe if cfg.keep.get("best", True) else None, keep_epochs=cfg.keep.get("epochs", [])))
    callbacks.append(
        ModelCheckpoint(**OmegaConf.to_container(cfg.checkpoint, resolve=True))
    )
    trainer = pl.Trainer(**cfg.trainer, logger=logger, callbacks=callbacks)
    if cfg.get("grad_clip"):
        # spt.Module clips inside its manual optimization step and reads the trainer's
        # underscore attributes (Lightning rejects Trainer(gradient_clip_val) under manual opt).
        trainer.gradient_clip_val_ = float(cfg.grad_clip)
        trainer.gradient_clip_algorithm_ = "norm"
        print(f"[grad-clip] norm {cfg.grad_clip} at every optimizer step", flush=True)
    if accum > 1:
        trainer.accumulate_grad_batches_ = accum
        print(f"[grad-accum] spt manual accumulation over {accum} micro-batches "
              f"-> effective batch {cfg.loader.batch_size * world_size * accum}", flush=True)

    # Weights-only continuation
    resume_ckpt_path = cfg.resume.ckpt_path
    if resume_ckpt_path is not None and cfg.resume.weights_only:
        state = torch.load(
            to_absolute_path(resume_ckpt_path), map_location="cpu", weights_only=False
        )
        state_dict = state["state_dict"]
        if getattr(encoder, "pos_embed", None) is None:
            state_dict = dict(state_dict)
            state_dict.pop("encoder.pos_embed", None)
        module.load_state_dict(state_dict, strict=True)
        del state
        print(
            f"[resume] loaded weights from {resume_ckpt_path}; "
            "starting a fresh schedule (ckpt_path=None)",
            flush=True,
        )
        resume_ckpt_path = None

    manager = spt.Manager(
        trainer=trainer,
        module=module,
        data=data,
        ckpt_path=resume_ckpt_path,
        weights_only=cfg.resume.weights_only,
    )
    manager()


if __name__ == "__main__":
    main()
