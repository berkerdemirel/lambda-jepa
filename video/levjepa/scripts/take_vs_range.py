"""Lance read-path comparison on the K700 store for the training access pattern (16 contiguous rows of a
random episode): ds.take(rows) [donor loader] vs ds.to_table(offset, limit) [range read] vs scanner with
an episode filter. 8 random clips each, cold-ish. Usage: take_vs_range.py"""
import time, random, lance
path = "/mnt/beegfs/locatgrp/shared/datasets/kinetics/k710_20pct_k700_2020.lance"
ds = lance.dataset(path); n = ds.count_rows(); random.seed(1)
starts = [random.randrange(0, n - 32) for _ in range(8)]
def run(label, fn):
    t0 = time.time(); nb = 0
    for s in starts:
        t1 = time.time(); tbl = fn(s); dt = time.time() - t1; nb += sum(len(v) for v in tbl["frame"].to_pylist())
        print(f"   {label} @ {s}: {dt:.2f}s", flush=True)
    dt = time.time() - t0; print(f"{label}: 8 clips x 16 rows in {dt:.1f}s = {128/dt:.1f} rows/s, {nb/dt/1e6:.1f} MB/s", flush=True)
run("take(rows)", lambda s: ds.take(list(range(s, s + 16)), columns=["frame"]))
run("to_table(offset,limit)", lambda s: ds.to_table(columns=["frame"], offset=s, limit=16))
run("scanner(batch_size=16).to_table offset", lambda s: ds.scanner(columns=["frame"], offset=s, limit=16, batch_size=16).to_table())
ep = ds.take([starts[0]], columns=["episode_idx"])["episode_idx"][0].as_py()
run("scanner(filter episode)", lambda s: ds.scanner(columns=["frame"], filter=f"episode_idx == {ds.take([s], columns=['episode_idx'])['episode_idx'][0].as_py()}").to_table())
print("TAKE_VS_RANGE_DONE", flush=True)
