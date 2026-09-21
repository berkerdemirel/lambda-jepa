"""Empirical NTK of the trunk at h.cls (E33 rich-vs-lazy diagnostic).

K[i,j] = (1/P) Σ_k ⟨∇_θ v_kᵀh(x_i), ∇_θ v_kᵀh(x_j)⟩ over TRUNK parameters only (the g_enc
trunk-module-only convention), h = forward_features CLS. Unit-norm random output projections
v_k (Hutchinson-style) instead of the exact D-output NTK: E[v vᵀ] = I/D makes E[K] the trace
NTK up to the constant 1/D, invisible to alignment. The v_k are FIXED by probe_seed across
checkpoints and cells, so alignment trajectories share their estimator noise (common random
numbers). fp32 forward/backward, no autocast; per-sample grads held bf16 on-GPU
([N, n_params] — ~44 GB for ViT-B at N=256, an A100-80 job), Gram accumulated through fp32
row chunks, K returned fp64 on CPU.
"""
import torch


def probe_vectors(dim, n_probes, seed):
    g = torch.Generator().manual_seed(seed)
    v = torch.randn(n_probes, dim, generator=g)
    return v / v.norm(dim=1, keepdim=True)


def empirical_ntk(trunk, images, n_probes=8, probe_seed=1009, device="cuda"):
    trunk = trunk.to(device).eval()
    params = list(trunk.parameters())
    for p in params:
        p.requires_grad_(True)
    n_par = sum(p.numel() for p in params)
    N = images.shape[0]
    K = torch.zeros(N, N, dtype=torch.float64)
    vs = None
    for k in range(n_probes):
        G = torch.empty(N, n_par, dtype=torch.bfloat16, device=device)
        for i in range(N):
            for p in params:
                p.grad = None
            h = trunk.forward_features(images[i:i + 1].to(device))[:, 0]
            if vs is None:
                vs = probe_vectors(h.shape[1], n_probes, probe_seed).to(device)
            (h[0] @ vs[k]).backward()
            G[i] = torch.cat([p.grad.reshape(-1) for p in params]).to(torch.bfloat16)
        for a in range(0, N, 32):
            Ga = G[a:a + 32].float()
            for b in range(0, N, 32):
                K[a:a + 32, b:b + 32] += (Ga @ G[b:b + 32].float().T).double().cpu() / n_probes
        del G
    return K
