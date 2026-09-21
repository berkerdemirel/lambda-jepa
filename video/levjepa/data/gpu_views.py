"""Photometric half of the lejepa view stack on the GPU (E34 second launch, 2026-09-03). The workers keep read,
decode and RandomResizedCrop and ship uint8 crops; this applies to each (clip, view) sample, with its own parameter
draws, exactly the ops the CPU Compose applied after the crop — ColorJitter(0.8, 0.8, 0.8, 0.2) at p .8 in a random
op order, RandomGrayscale p .2, GaussianBlur k7 sigma U(0.1, 2) p .5, RandomSolarize 128 p .2, RandomHorizontalFlip
p .5 — in float [0, 1], and returns uint8 so `main.to_float_normalized` stays the one normalization seam. Every op
reproduces torchvision's functional reference to 0.000/255 (scratch/video/e34_gpu_photometrics_bench.py, A100 and
H100); the one declared difference from the CPU path is the absence of the uint8 rounding between consecutive ops.
Why: on the CPU these ops cost 0.38 core-seconds per clip, 95 percent of a worker's CPU time, and bound the
loader; here they cost ~0.8 s per 576-sample micro-batch on an H100 while the workers keep ~0.04 core-s per clip."""
import torch
import torch.nn.functional as F


def _gray(x): return 0.2989 * x[..., 0:1, :, :] + 0.587 * x[..., 1:2, :, :] + 0.114 * x[..., 2:3, :, :]


def _rgb2hsv(img):
    r, g, b = img.unbind(dim=-3)
    maxc, _ = img.max(dim=-3); minc, _ = img.min(dim=-3)
    eqc = maxc == minc; cr = maxc - minc; ones = torch.ones_like(maxc)
    s = cr / torch.where(eqc, ones, maxc); crd = torch.where(eqc, ones, cr)
    rc, gc, bc = (maxc - r) / crd, (maxc - g) / crd, (maxc - b) / crd
    hr = (maxc == r) * (bc - gc); hg = ((maxc == g) & (maxc != r)) * (2.0 + rc - bc); hb = ((maxc != g) & (maxc != r)) * (4.0 + gc - rc)
    return torch.stack((torch.fmod(((hr + hg + hb) / 6.0 + 1.0), 1.0), s, maxc), dim=-3)


def _hsv2rgb(img):
    h, s, v = img.unbind(dim=-3)
    h6 = h * 6.0; i = torch.floor(h6); f = h6 - i; i = i.to(torch.int32) % 6
    p = (v * (1.0 - s)).clamp_(0.0, 1.0); q = (v * (1.0 - s * f)).clamp_(0.0, 1.0); t = (v * (1.0 - s * (1.0 - f))).clamp_(0.0, 1.0)
    def sel(a): return torch.where(i == 0, a[0], torch.where(i == 1, a[1], torch.where(i == 2, a[2], torch.where(i == 3, a[3], torch.where(i == 4, a[4], a[5])))))
    return torch.stack((sel((v, q, p, p, t, v)), sel((t, v, v, q, p, p)), sel((p, p, t, v, v, q))), dim=-3)


def _brightness(x, f): return (x * f).clamp_(0, 1)
def _contrast(x, f): m = _gray(x).mean(dim=(-3, -2, -1), keepdim=True); return (f * x + (1 - f) * m).clamp_(0, 1)
def _saturation(x, f): return (f * x + (1 - f) * _gray(x)).clamp_(0, 1)
def _hue(x, f): hsv = _rgb2hsv(x); return _hsv2rgb(torch.cat([(hsv[..., 0:1, :, :] + f) % 1.0, hsv[..., 1:, :, :]], -3))
_JITTER = (_brightness, _contrast, _saturation, _hue)


def _blur(x, sigma, apply):
    """Separable k7 gaussian with reflect padding, one kernel per sample (the identity where not applied)."""
    B, T, C, H, W = x.shape
    t = torch.linspace(-3, 3, 7, device=x.device)
    k = torch.exp(-0.5 * (t / sigma[:, None]) ** 2); k = k / k.sum(1, keepdim=True)
    delta = torch.zeros(7, device=x.device); delta[3] = 1
    k = torch.where(apply[:, None], k, delta).repeat_interleave(T * C, 0)
    y = F.pad(x.reshape(1, B * T * C, H, W), (3, 3, 3, 3), mode="reflect")
    y = F.conv2d(y, k.view(-1, 1, 1, 7), groups=B * T * C)
    y = F.conv2d(y, k.view(-1, 1, 7, 1), groups=B * T * C)
    return y.reshape(B, T, C, H, W)


def _stack(x, gen):
    """uint8 (B, T, 3, H, W) -> uint8, the stack per sample; the op order of ColorJitter is drawn per sample."""
    B, dev = x.shape[0], x.device
    U = lambda *s: torch.rand(*s, device=dev, generator=gen)
    x = x.float().div_(255)
    jit = U(B) < 0.8
    fac = [0.2 + 1.6 * U(B), 0.2 + 1.6 * U(B), 0.2 + 1.6 * U(B), -0.2 + 0.4 * U(B)]
    order = U(B, 4).argsort(1)
    for k in range(4):
        for op in range(4):
            m = jit & (order[:, k] == op)
            if m.any(): x[m] = _JITTER[op](x[m], fac[op][m].view(-1, 1, 1, 1, 1))
    m = U(B) < 0.2
    if m.any(): x[m] = _gray(x[m]).expand(-1, -1, 3, -1, -1)
    x = _blur(x, 0.1 + 1.9 * U(B), U(B) < 0.5)
    m = (U(B) < 0.2).view(B, 1, 1, 1, 1); x = torch.where(m & (x >= 128 / 255), 1 - x, x)
    m = (U(B) < 0.5).view(B, 1, 1, 1, 1); x = torch.where(m, x.flip(-1), x)
    return x.mul_(255).round_().to(torch.uint8)


def photometrics(views, seed, chunk=144):
    """views: uint8 (N, V, T, C, H, W) crops on the GPU -> uint8 of the same shape, every (clip, view) sample through
    the stack with its own draws from a generator seeded with `seed` (per rank and micro-batch in training, fixed in
    the share measurement). Chunked over samples to cap the float working set (~18 GiB at chunk 144)."""
    flat = views.reshape(-1, *views.shape[2:])
    gen = torch.Generator(device=views.device).manual_seed(int(seed))
    return torch.cat([_stack(c, gen) for c in flat.split(chunk)]).reshape(views.shape)
