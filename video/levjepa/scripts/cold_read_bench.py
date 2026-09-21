"""COLD read-path comparison on the K700 store (fresh random clips per method so page cache cannot help):
take(rows) vs to_table(offset, limit) vs scanner(offset, limit); then the same range read from 8 threads
concurrently (the loader's parallelism). 16 contiguous rows per clip. Usage: cold_read_bench.py"""
import time, random, lance
from concurrent.futures import ThreadPoolExecutor
path = "/mnt/scistore19_placeholder"
path = "/mnt/beegfs/locatgrp/shared/datasets/kinetics/k710_20pct_k700_2020.lance"
ds = lance.dataset(path); n = ds.count_rows()
def starts(seed, k=8): r = random.Random(seed); return [r.randrange(0, n - 32) for _ in range(k)]
def timeit(label, fn, ss):
    t0 = time.time(); nb = 0
    for s in ss: nb += sum(len(v) for v in fn(s)["frame"].to_pylist())
    dt = time.time() - t0; print(f"{label}: {len(ss)} clips x 16 rows in {dt:.2f}s = {len(ss)*16/dt:.0f} rows/s, {nb/dt/1e6:.1f} MB/s", flush=True)
timeit("cold take(rows)", lambda s: ds.take(list(range(s, s + 16)), columns=["frame"]), starts(11))
timeit("cold to_table(offset,limit)", lambda s: ds.to_table(columns=["frame"], offset=s, limit=16), starts(22))
timeit("cold scanner(offset,limit,bs16)", lambda s: ds.scanner(columns=["frame"], offset=s, limit=16, batch_size=16).to_table(), starts(33))
ss = starts(44, 32); t0 = time.time()
with ThreadPoolExecutor(8) as ex: nb = sum(sum(len(v) for v in t["frame"].to_pylist()) for t in ex.map(lambda s: ds.to_table(columns=["frame"], offset=s, limit=16), ss))
dt = time.time() - t0; print(f"cold to_table x8 threads: 32 clips in {dt:.2f}s = {32*16/dt:.0f} rows/s, {nb/dt/1e6:.1f} MB/s", flush=True)
ss = starts(55, 32); t0 = time.time()
with ThreadPoolExecutor(8) as ex: nb = sum(sum(len(v) for v in t["frame"].to_pylist()) for t in ex.map(lambda s: ds.take(list(range(s, s + 16)), columns=["frame"]), ss))
dt = time.time() - t0; print(f"cold take x8 threads: 32 clips in {dt:.2f}s = {32*16/dt:.0f} rows/s, {nb/dt/1e6:.1f} MB/s", flush=True)
print("COLD_READ_BENCH_DONE", flush=True)
