"""Per-term gradient share for the E10 SIGReg-placement arms (Berker, 2026-07-09).

The B/C collapse autopsy needs to know WHO owned the trunk's gradient, per phase: the loss
curves show λ·sigreg ≫ (1−λ)·inv in loss units at high-d placements, but losses are not
gradients. For each arm (A@proj16 / B@embed512 / C@cls384 / D0@embed512-no-projector) and each
cadence checkpoint (+ an arch-matched seed-0 random init labeled ep0), replay training-condition
batches (same aug stack, V, bs, bf16 autocast, train mode) and backprop each term separately:

  g_inv = ∇[(1−λ)·inv]   g_sig = ∇[λ·sigreg]   (exactly the two loss contributions, lejepa.py)

Recorded per (arm, epoch), averaged over BATCHES fixed batches: term losses; per-group grad
norms (trunk / embed Linear / projector) for both terms; sigreg share ‖g_sig‖/(‖g_sig‖+‖g_inv‖)
per group; cos(g_sig, g_inv) on the encoder; and slice-moment diagnostics of the constrained
space (mean/min/max slice variance, mean |slice mean| over 256 fresh unit directions) — these
decompose the sigreg value into moment-mismatch vs shape per the SIGREG_DIM_SCALING note.
Measurement deviations from training: no GradScaler (bf16 needs none for norms), identical
batches reused across arms/ckpts (paired comparison). Output: results/diag/e10_grad_share.csv.
"""
import csv
import os

import torch
from torch.amp import autocast

from sslgap.ckpt.adapters import _resolve
from sslgap.data import ViewsDataset
from sslgap.methods.lejepa import SIGReg

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ARMS = ["A", "B", "C", "D0"]
EPS = [0, 38, 75, 112, 150]
BATCHES, BS = 8, 256


def rebuild(ck, random_init):
    if random_init:
        torch.manual_seed(0)
    mods = {}
    for role, spec in ck["arch"].items():
        if role == "probe":
            continue
        m = _resolve(spec["class"])(**spec["kwargs"])
        if not random_init:
            m.load_state_dict(ck["modules"][role])
        mods[role] = m.cuda().train()
    return mods


def param_groups(mods):
    enc = list(mods["encoder"].named_parameters())
    return {"trunk": [p for n, p in enc if not n.startswith("head")],
            "embed": [p for n, p in enc if n.startswith("head")],
            "proj": list(mods["projector"].parameters())}


def grad_norm(ps):
    return sum(float(p.grad.pow(2).sum()) for p in ps if p.grad is not None) ** 0.5


def grad_vec(ps):
    # missing grads count as zeros: a term that doesn't touch a param (e.g. arm C's sigreg
    # never reaches the embed Linear) still compares on the same index set
    vs = [(p.grad if p.grad is not None else torch.zeros_like(p)).flatten().float() for p in ps]
    return torch.cat(vs) if vs else torch.zeros(1, device="cuda")


def zero(mods):
    for m in mods.values():
        m.zero_grad(set_to_none=True)


def step_terms(mods, sigreg, views, sig_at):
    N, V = views.shape[:2]
    x = views.flatten(0, 1)
    with autocast("cuda", dtype=torch.bfloat16):
        if sig_at == "cls":
            feats = mods["encoder"].forward_features(x)
            cls = feats[:, 0]
            emb = mods["encoder"].forward_head(feats)
        else:
            emb = mods["encoder"](x)
        proj = mods["projector"](emb).reshape(N, V, -1).transpose(0, 1)
        inv_loss = (proj.mean(0) - proj).square().mean()
        sig_in = {"proj": proj, "embed": emb.reshape(N, V, -1).transpose(0, 1),
                  "cls": cls.reshape(N, V, -1).transpose(0, 1) if sig_at == "cls" else None}[sig_at]
        sigreg_loss = sigreg(sig_in)
    return inv_loss, sigreg_loss, sig_in


@torch.no_grad()
def slice_moments(sig_in, m=256):
    z = sig_in.float()                                  # [V, N, K]
    A = torch.randn(z.size(-1), m, device=z.device)
    A = A.div_(A.norm(p=2, dim=0))
    s = z @ A                                           # [V, N, m]
    var, mean = s.var(dim=-2).mean(0), s.mean(dim=-2).mean(0)
    return {"slice_var_mean": float(var.mean()), "slice_var_min": float(var.min()),
            "slice_var_max": float(var.max()), "slice_absmean_mean": float(mean.abs().mean())}


def main():
    torch.manual_seed(0)
    ds = ViewsDataset("imagenette", "train", V=4, img_size=128)
    loader = torch.utils.data.DataLoader(ds, batch_size=BS, shuffle=True, num_workers=8,
                                         persistent_workers=False)
    batches = []
    for views, _ in loader:
        batches.append(views)
        if len(batches) == BATCHES:
            break

    sigreg = SIGReg().cuda()
    rows = []
    for arm in ARMS:
        ck = torch.load(f"{ROOT}/outputs/toy.lejepa.s0.e10{arm}_ep150.pt",
                        map_location="cpu", weights_only=False)
        lamb, sig_at = ck["cfg"]["method"]["lamb"], ck["cfg"]["method"].get("sigreg_at", "proj")
        for ep in EPS:
            if ep == 0:
                mods = rebuild(ck, random_init=True)
            else:
                cke = torch.load(f"{ROOT}/outputs/toy.lejepa.s0.e10{arm}_ep{ep}.pt",
                                 map_location="cpu", weights_only=False)
                mods = rebuild(cke, random_init=False)
            groups = param_groups(mods)
            acc = {k: [] for k in ["inv", "sigreg", "cos"]
                   + [f"{g}_{t}" for g in groups for t in ("inv", "sig")]}
            mom = []
            for views in batches:
                views = views.cuda(non_blocking=True)
                inv_loss, sigreg_loss, sig_in = step_terms(mods, sigreg, views, sig_at)
                mom.append(slice_moments(sig_in))
                zero(mods)
                ((1 - lamb) * inv_loss).backward(retain_graph=True)
                g_inv = {g: grad_norm(ps) for g, ps in groups.items()}
                v_inv = grad_vec(groups["trunk"] + groups["embed"])
                zero(mods)
                (lamb * sigreg_loss).backward()
                g_sig = {g: grad_norm(ps) for g, ps in groups.items()}
                v_sig = grad_vec(groups["trunk"] + groups["embed"])
                acc["inv"].append(inv_loss.detach().item())
                acc["sigreg"].append(sigreg_loss.detach().item())
                acc["cos"].append(float(torch.dot(v_inv, v_sig)
                                        / (v_inv.norm() * v_sig.norm() + 1e-12)))
                for g in groups:
                    acc[f"{g}_inv"].append(g_inv[g])
                    acc[f"{g}_sig"].append(g_sig[g])
            mean = lambda xs: sum(xs) / len(xs)  # noqa: E731
            row = {"arm": arm, "sigreg_at": sig_at, "ep": ep,
                   "loss_inv": mean(acc["inv"]), "loss_sigreg": mean(acc["sigreg"]),
                   "cos_inv_sig_enc": mean(acc["cos"])}
            for g in groups:
                gi, gs = mean(acc[f"{g}_inv"]), mean(acc[f"{g}_sig"])
                row[f"gnorm_inv_{g}"], row[f"gnorm_sig_{g}"] = gi, gs
                row[f"sig_share_{g}"] = gs / (gs + gi + 1e-12)
            for k in mom[0]:
                row[k] = mean([m[k] for m in mom])
            rows.append({k: (round(v, 6) if isinstance(v, float) else v) for k, v in row.items()})
            print("[gradshare] " + " ".join(f"{k}={v}" for k, v in rows[-1].items()), flush=True)
            # incremental write: a late-arm crash must not lose completed rows
            out = os.path.join(ROOT, "results", "diag", "e10_grad_share.csv")
            os.makedirs(os.path.dirname(out), exist_ok=True)
            with open(out, "w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=list(rows[0]))
                w.writeheader()
                w.writerows(rows)
            del mods
            torch.cuda.empty_cache()
    print(f"[gradshare] done -> {out}")


if __name__ == "__main__":
    main()
