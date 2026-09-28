"""Empirical NTK of the trunk at the CLS token (random output projections, trunk parameters only)."""
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
