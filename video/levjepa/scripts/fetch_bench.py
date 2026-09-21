"""Where do the seconds go in a clip read? For each store: (i) Lance take of 16 contiguous rows of one
random episode (the training access pattern), 8 episodes, wall time and bytes; (ii) JPEG decode of the
fetched frames with PIL; (iii) a full-row scan speed (sequential read) for reference. Usage: fetch_bench.py"""
import io, time, random, sys
import lance, numpy as np
from PIL import Image
B = "/mnt/beegfs/locatgrp/shared/datasets"
stores = {"k700": f"{B}/kinetics/k710_20pct_k700_2020.lance", "k400": f"{B}/kinetics/k710_20pct_k400.lance", "wt": f"{B}/walking_tours/train.lance"}
for name, path in stores.items():
    ds = lance.dataset(path); n = ds.count_rows()
    frags = ds.get_fragments(); print(f"== {name}: {n} rows, {len(frags)} fragments, versions {len(ds.versions())}", flush=True)
    random.seed(0); starts = [random.randrange(0, n - 32) for _ in range(8)]
    t0 = time.time(); tot = 0; nrows = 0
    for s in starts:
        rows = list(range(s, s + 16)); t1 = time.time()
        tbl = ds.take(rows, columns=["frame"]); vals = tbl["frame"].to_pylist(); dt = time.time() - t1
        tot += sum(len(v) for v in vals); nrows += len(vals)
        print(f"   take 16 rows @ {s}: {dt:.2f}s, {sum(len(v) for v in vals)/1e6:.1f} MB", flush=True)
    dt = time.time() - t0; print(f"   TAKE: {nrows} rows in {dt:.1f}s = {nrows/dt:.1f} rows/s, {tot/dt/1e6:.1f} MB/s", flush=True)
    t0 = time.time(); imgs = [np.asarray(Image.open(io.BytesIO(v)).convert("RGB")) for v in vals]; dt = time.time() - t0
    print(f"   DECODE: {len(vals)} frames in {dt:.2f}s = {len(vals)/dt:.0f} frames/s, shape {imgs[0].shape}", flush=True)
    t0 = time.time(); k = 0
    for batch in ds.to_batches(columns=["frame"], batch_size=256):
        k += batch.num_rows
        if k >= 2048: break
    dt = time.time() - t0; print(f"   SCAN: {k} rows sequential in {dt:.1f}s = {k/dt:.0f} rows/s", flush=True)
print("FETCH_BENCH_DONE", flush=True)
