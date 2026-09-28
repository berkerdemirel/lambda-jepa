"""Smoke test of the lambda-JEPA loss branch of main.py on random clips."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import lightning as pl
import stable_pretraining as spt
import torch
from omegaconf import OmegaConf
import module as vit_models
import main as M
from lambdajepa_reg import Ring, ShareLogger, SACReg

V, T, S, N = 6, 4, 32, 8

class Fake(torch.utils.data.Dataset):
    def __len__(self):
        return 32

    def __getitem__(self, i):
        g = torch.Generator().manual_seed(i)
        return {"views": (torch.rand(V, T, 3, S, S, generator=g) * 255).to(torch.uint8)}

loader = torch.utils.data.DataLoader(Fake(), batch_size=N, shuffle=True, drop_last=True, num_workers=0)
enc = vit_models.vit_tiny(img_size=S, patch_size=16, num_frames=T, tubelet_size=1, use_rope=True,
                          token_drop_rate=0.5, attn_mode="block_causal")
proj = vit_models.Projector(input_dim=enc.embed_dim, hidden_dim=64, output_dim=32, norm_layer=torch.nn.BatchNorm1d)
w = {"inv": 22.81, "moment_kl": 222.26, "w_h": 4.054}
module = spt.Module(
    encoder=enc, projector=proj, forward=M.lambdajepa_forward,
    cond_z=SACReg(d_slice=16), cond_h=SACReg(d_slice=32),
    ring=Ring(), queue_steps=3, h_queue_steps=7, w_inv=22.81, w_z=222.26, w_h=4.054, base_seed=0,
    optim={"optimizer": {"type": "AdamW", "lr": 1e-3, "weight_decay": 0.04},
           "scheduler": {"type": "LinearWarmupCosineAnnealing", "total_steps": 4, "peak_step": 1, "start_factor": .25, "end_lr": 1e-3},
           "interval": "step"},
    hparams={"smoke": True})
share = ShareLogger(terms_fn=M.lambdajepa_terms, weights={"inv": 22.81, "moment_kl": 222.26, "h_moment_kl": 4.054},
                    dataset=loader.dataset, collate_fn=loader.collate_fn, bs=N, every=1, seed=0)
trainer = pl.Trainer(accelerator="cpu", devices=1, max_epochs=1, limit_train_batches=3, logger=False,
                     enable_checkpointing=False, enable_progress_bar=False, callbacks=[share], num_sanity_val_steps=0)
trainer.gradient_clip_val_ = 1.0
trainer.gradient_clip_algorithm_ = "norm"
data = spt.data.DataModule(train=loader)
trainer.fit(module, datamodule=data)
print("rings after 3 steps: z=%d h=%d rows" % (sum(b.shape[0] for b in module.ring.bufs["z"]), sum(b.shape[0] for b in module.ring.bufs["h"])))
with torch.no_grad():
    module.eval(); batch = next(iter(loader))
    terms, cls, z = M.lambdajepa_terms(module, batch, seed=1, distributed=False)
print("terms:", {k: round(float(v), 4) for k, v in terms.items()}, "| cls", tuple(cls.shape), "z", tuple(z.shape))
print("SMOKE_OK")
