"""Throughput of the mp4-backed training loader (six views, 8 workers) reading from LEVJEPA_MP4_ROOT
(BeeGFS or RAM): first batch time, then 8 batches. Usage: mp4_loader_smoke.py"""
import os, sys, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from hydra import compose, initialize
import main as M

def run():
    with initialize(version_base=None, config_path="../conf"):
        cfg = compose(config_name="lambdajepa_vits", overrides=["data=k710_clipfiles", "loader.num_workers=8", "loader.batch_size=16"])
    loader = M.build_video_loader(cfg, split="train", shuffle=True, hflip=True)
    it = iter(loader); t0 = time.time(); b = next(it); print(f"first batch {time.time() - t0:.0f}s, views {tuple(b['views'].shape)} {b['views'].dtype}", flush=True)
    n = 8; t0 = time.time()
    for _ in range(n): b = next(it)
    dt = time.time() - t0; clips = n * cfg.loader.batch_size
    print(f"{n} batches x {cfg.loader.batch_size} clips in {dt:.1f}s = {clips/dt:.2f} clips/s with 8 workers = {clips/dt/8:.2f} clips/s/worker ({os.environ.get('LEVJEPA_CLIPFILE_ROOT', 'BeeGFS')})", flush=True)
    print("MP4_LOADER_SMOKE_DONE", flush=True)

if __name__ == "__main__":
    run()
