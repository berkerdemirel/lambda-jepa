"""Data-path smoke for E34: open the three K710-20% stores through the donor loader exactly as training
will (conf lambdajepa_vits + data=k710_20pct, six lejepa views), report the mixture, and pull two batches.
Usage: python scripts/k710_data_smoke.py   (CPU job; no model)."""
import os, sys, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from hydra import compose, initialize
import lance, torch
import main as M


def main():
    with initialize(version_base=None, config_path="../conf"):
        cfg = compose(config_name="lambdajepa_vits", overrides=["data=k710_20pct", "loader.num_workers=4", "loader.batch_size=8"])
    for src in M.resolve_data_spec(cfg.data.train):
        ds = lance.dataset(src["path"]); t = ds.take([0]).to_pylist()[0]
        print(f"{src['name']}: {ds.count_rows()} rows, label of row 0 = {t['label']}, frame {t['h']}x{t['w']}", flush=True)
    t0 = time.time()
    loader = M.build_video_loader(cfg, split="train", shuffle=True, hflip=cfg.augmentation.hflip)
    print(f"loader built in {time.time() - t0:.0f}s: {len(loader.dataset)} clips/epoch, {len(loader)} batches of {cfg.loader.batch_size}", flush=True)
    it = iter(loader)
    for i in range(2):
        t0 = time.time(); b = next(it)
        v = b["views"]; print(f"batch {i}: views {tuple(v.shape)} {v.dtype} min {int(v.min())} max {int(v.max())}, other keys {[k for k in b if k != 'views']}, {time.time() - t0:.1f}s", flush=True)
    print("K710_DATA_SMOKE_OK", flush=True)


if __name__ == "__main__":
    main()
