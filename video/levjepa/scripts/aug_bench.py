"""Loader throughput benchmark for the six-view recipe (E34): frame decode alone vs decode + the lejepa
view stack, single process and with workers, on real K700 clips. Prints frame-augmentations per second per
worker. Usage: python scripts/aug_bench.py [--clips 16] [--workers 8]"""
import argparse, os, sys, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch, multiprocessing as mp
from data.loader import VJEPAClipDataset, LejepaViewsTransform

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--clips", type=int, default=16); ap.add_argument("--workers", type=int, default=8)
    a = ap.parse_args()
    path = "/mnt/beegfs/locatgrp/shared/datasets/kinetics/k710_20pct_k700_2020.lance"
    V, T = 6, 16
    print(f"cpus visible {os.cpu_count()}, torch threads {torch.get_num_threads()}", flush=True)
    # 1) decode only (transform=None) in-process
    ds = VJEPAClipDataset(path=path, num_frames=T, frame_stride=2, clips_per_video=1, random_crop=True, transform=None, pad_short=False, name="k700")
    idx = list(range(a.clips)); t0 = time.time(); items = ds.__getitems__(idx); dt = time.time() - t0
    print(f"decode only: {a.clips} clips x {T} frames in {dt:.1f}s = {a.clips*T/dt:.0f} frames/s (one process); frame {tuple(items[0]['frame'].shape)}", flush=True)
    # 2) the six-view stack on the decoded clips, in-process
    tf = LejepaViewsTransform(views=V, size=224, normalize_on_gpu=True)
    t0 = time.time(); [tf(it) for it in items]; dt = time.time() - t0
    print(f"augment only: {a.clips} clips x {V} views x {T} frames = {a.clips*V*T} frame-augs in {dt:.1f}s = {a.clips*V*T/dt:.0f} frame-augs/s (one process)", flush=True)
    # 3) the real path with spawned workers (decode + augment), first batch excluded from timing
    ds2 = VJEPAClipDataset(path=path, num_frames=T, frame_stride=2, clips_per_video=1, random_crop=True, transform=tf, pad_short=False, name="k700")
    bs = 8
    loader = torch.utils.data.DataLoader(ds2, batch_size=bs, shuffle=True, num_workers=a.workers, multiprocessing_context=mp.get_context("spawn"), persistent_workers=True, prefetch_factor=2)
    it = iter(loader); t0 = time.time(); next(it); print(f"workers={a.workers}: first batch (spawn + index) {time.time()-t0:.0f}s", flush=True)
    n = 6; t0 = time.time(); [next(it) for _ in range(n)]; dt = time.time() - t0
    print(f"workers={a.workers}: {n} batches x {bs} clips = {n*bs*V*T} frame-augs in {dt:.1f}s = {n*bs*V*T/dt:.0f} frame-augs/s total = {n*bs*V*T/dt/a.workers:.0f} per worker", flush=True)
    print("AUG_BENCH_DONE", flush=True)

if __name__ == "__main__":
    main()
