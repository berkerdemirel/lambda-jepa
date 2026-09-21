"""RAM workaround benchmark (E34): (1) parallel copy of the K400 mp4 set (7,312 files, 11 GB) from BeeGFS
to /dev/shm — copy rate, extrapolated to the 155 GB of all three sets; (2) decord decode of random
16-frame clips (stride matching the 15 fps store x training stride 2) from /dev/shm and from BeeGFS,
1 process and 8 processes; (3) same with the short-edge-384 resize applied. Prints clips/s.
Usage: ram_stage_bench.py [--nproc 8]"""
import argparse, os, random, shutil, subprocess, time
from concurrent.futures import ProcessPoolExecutor
import numpy as np

SRC = "/mnt/beegfs/locatgrp/shared/datasets/kinetics/k710_20pct_videos_k400"; DST = "/dev/shm/k710_bench/k400"

def clip_from(path, T=16):
    from decord import VideoReader, cpu
    import torch
    vr = VideoReader(path, ctx=cpu(0), num_threads=1); n = len(vr); fps = vr.get_avg_fps() or 30.0
    stride = 2 * max(1, round(fps / 15.0)); span = stride * (T - 1) + 1
    start = random.randrange(0, max(1, n - span)); idx = [start + i * stride for i in range(T)]
    idx = [min(i, n - 1) for i in idx]
    arr = vr.get_batch(idx).asnumpy()                       # (T, H, W, 3) uint8
    x = torch.from_numpy(arr).permute(0, 3, 1, 2)            # (T, C, H, W)
    h, w = x.shape[-2:]; s = 384 / min(h, w)
    x = torch.nn.functional.interpolate(x.float(), size=(round(h * s), round(w * s)), mode="bilinear", antialias=True).to(torch.uint8)
    return x.shape

def decode_many(args):
    files, seed = args; random.seed(seed); t0 = time.time(); shapes = [clip_from(f) for f in files]; return time.time() - t0, len(files), shapes[0]

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--nproc", type=int, default=8); a = ap.parse_args()
    files = sorted(os.path.join(d, f) for d, _, fs in os.walk(SRC) for f in fs if f.endswith(".mp4"))
    total = sum(os.path.getsize(f) for f in files); print(f"K400 mp4 set: {len(files)} files, {total/1e9:.1f} GB", flush=True)
    random.seed(0); sample = random.sample(files, 64)
    # 1) decode straight from BeeGFS first (no copy needed for this number)
    dt, n, shp = decode_many((sample[:16], 1)); print(f"decode from BeeGFS, 1 process: {n} clips in {dt:.1f}s = {n/dt:.2f} clips/s, frame {tuple(shp)}", flush=True)
    chunks = [(sample[i::a.nproc], 100 + i) for i in range(a.nproc)]; t0 = time.time()
    with ProcessPoolExecutor(a.nproc) as ex: res = list(ex.map(decode_many, chunks))
    dt = time.time() - t0; n = sum(r[1] for r in res); print(f"decode from BeeGFS, {a.nproc} processes: {n} clips in {dt:.1f}s = {n/dt:.2f} clips/s = {n/dt/a.nproc:.2f} per process", flush=True)
    # 2) copy to RAM, 16 parallel streams, one cp per file (names carry spaces and parentheses)
    shutil.rmtree("/dev/shm/k710_bench", ignore_errors=True); os.makedirs(DST, exist_ok=True)
    t0 = time.time()
    subprocess.run(f"cd '{SRC}' && find . -type d -exec mkdir -p '{DST}/{{}}' \\; && find . -name '*.mp4' -print0 | xargs -0 -P 16 -I{{}} cp '{{}}' '{DST}/{{}}'", shell=True, check=True)
    n_copied = sum(len(fs) for _, _, fs in os.walk(DST))
    dt = time.time() - t0; print(f"COPY BeeGFS->/dev/shm: {n_copied}/{len(files)} files, {total/1e9:.1f} GB in {dt:.0f}s = {total/dt/1e6:.0f} MB/s -> 155 GB would take {155e9/(total/dt)/60:.1f} min", flush=True)
    random.seed(0); sample = random.sample(files, 64)
    for label, root in (("RAM", DST),):
        paths = [f.replace(SRC, root) for f in sample]
        dt, n, shp = decode_many((paths[:16], 1)); print(f"decode from {label}, 1 process: {n} clips in {dt:.1f}s = {n/dt:.2f} clips/s, frame {tuple(shp)}", flush=True)
        chunks = [(paths[i::a.nproc], 100 + i) for i in range(a.nproc)]; t0 = time.time()
        with ProcessPoolExecutor(a.nproc) as ex: res = list(ex.map(decode_many, chunks))
        dt = time.time() - t0; n = sum(r[1] for r in res); print(f"decode from {label}, {a.nproc} processes: {n} clips in {dt:.1f}s = {n/dt:.2f} clips/s = {n/dt/a.nproc:.2f} per process", flush=True)
    shutil.rmtree("/dev/shm/k710_bench", ignore_errors=True); print("RAM_STAGE_BENCH_DONE", flush=True)

if __name__ == "__main__":
    main()
