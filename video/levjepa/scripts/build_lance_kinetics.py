"""Build the K710-20 % training store from the subset list."""
import argparse
import io
import os
import sys
import time
from multiprocessing import get_context
from pathlib import Path

SHORT_EDGE = 384
JPEG_Q = 90
TARGET_FPS = 15.0
MAX_FRAMES = 192
DECODE_CHUNK = 32

def stride_for(fps):
    return max(1, round(fps / TARGET_FPS))

def encode_video(task):
    from PIL import Image
    from decord import VideoReader, cpu

    idx, path, min_frames = task
    try:
        vr = VideoReader(str(path), ctx=cpu(0), num_threads=1)
        n_raw, fps = len(vr), vr.get_avg_fps()
        stride = stride_for(fps if fps and fps > 0 else 30.0)
        n_out = min(n_raw // stride, MAX_FRAMES)
        if n_out < min_frames:
            return idx, None
        out = []
        for base in range(0, n_out, DECODE_CHUNK):
            n = min(DECODE_CHUNK, n_out - base)
            arrs = vr.get_batch([(base + i) * stride for i in range(n)]).asnumpy()
            for j, arr in enumerate(arrs):
                img = Image.fromarray(arr)
                w, h = img.size
                if h < w:
                    nh, nw = SHORT_EDGE, round(w * SHORT_EDGE / h)
                else:
                    nh, nw = round(h * SHORT_EDGE / w), SHORT_EDGE
                img = img.resize((nw, nh), Image.BILINEAR)
                buf = io.BytesIO()
                img.save(buf, format="JPEG", quality=JPEG_Q)
                out.append((base + j, buf.getvalue(), img.size[1], img.size[0]))
        return idx, out
    except Exception as exc:
        print(f"  clip {path} failed: {exc}", flush=True)
        return idx, None

def main():
    import lance
    import pyarrow as pa

    ap = argparse.ArgumentParser()
    ap.add_argument("--videos", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--workers", type=int, default=int(os.environ.get("SLURM_CPUS_PER_TASK", 16)))
    ap.add_argument("--min-frames", type=int, default=32)
    ap.add_argument("--limit", type=int, default=0, help="smoke: only the first N videos")
    ap.add_argument("--classes", default="/path/to/kinetics/k710_20pct.csv",
                    help="the draw CSV: its sorted unique labels are the label index space, shared by every per-set store")
    args = ap.parse_args()
    out_path = Path(args.out)
    if out_path.exists():
        raise SystemExit(f"{out_path} already exists; remove it first")
    root = Path(args.videos)
    import csv
    with open(args.classes) as f:
        classes = sorted({r["label"] for r in csv.DictReader(f)})
    index = {c: i for i, c in enumerate(classes)}
    present = sorted(p.name for p in root.iterdir() if p.is_dir())
    unknown = [c for c in present if c not in index]
    if unknown:
        raise SystemExit(f"class dirs not in {args.classes}: {unknown[:5]}")
    videos = [(v, index[c]) for c in present for v in sorted((root / c).glob("*.mp4"))]
    if args.limit:
        videos = videos[:args.limit]
    if not videos:
        raise SystemExit(f"no {{class}}/*.mp4 under {root}")
    with open(str(out_path) + ".classes.txt", "w") as f:
        f.write("\n".join(classes) + "\n")
    print(f"{len(videos)} videos over {len(classes)} classes; {args.workers} workers", flush=True)
    schema = pa.schema([
        pa.field("episode_idx", pa.int32()), pa.field("step_idx", pa.int32()),
        pa.field("frame", pa.binary()), pa.field("h", pa.int16()), pa.field("w", pa.int16()),
        pa.field("label", pa.int16()),
    ])
    t0 = time.time()
    written = {"eps": 0, "rows": 0, "failed": 0}

    def batches():
        ep = 0
        ctx = get_context("spawn")
        tasks = [(i, str(v), args.min_frames) for i, (v, _) in enumerate(videos)]
        with ctx.Pool(args.workers) as pool:
            for i, frames in pool.imap(encode_video, tasks, chunksize=4):
                if frames is None:
                    written["failed"] += 1
                    continue
                lab = videos[i][1]
                written["eps"] += 1
                written["rows"] += len(frames)
                yield pa.RecordBatch.from_arrays([
                    pa.array([ep] * len(frames), type=pa.int32()),
                    pa.array([s for s, _, _, _ in frames], type=pa.int32()),
                    pa.array([b for _, b, _, _ in frames], type=pa.binary()),
                    pa.array([h for _, _, h, _ in frames], type=pa.int16()),
                    pa.array([w for _, _, _, w in frames], type=pa.int16()),
                    pa.array([lab] * len(frames), type=pa.int16()),
                ], schema=schema)
                ep += 1
                if written["eps"] % 1000 == 0:
                    el = time.time() - t0
                    rate = written["eps"] / el
                    print(f"  {written['eps']}/{len(videos)} clips, {written['rows']/1e6:.2f}M rows, "
                          f"{rate:.1f} clips/s, eta {(len(videos) - written['eps']) / rate / 60:.0f} min", flush=True)

    lance.write_dataset(batches(), str(out_path), schema=schema, mode="create")
    print(f"wrote {out_path}: {written['eps']} episodes, {written['rows']} rows, "
          f"{written['failed']} dropped/failed, in {(time.time() - t0) / 60:.1f} min", flush=True)
    ds = lance.dataset(str(out_path))
    print(f"verify: {ds.count_rows()} rows in store", flush=True)
    assert ds.count_rows() == written["rows"], "row count mismatch"
    print("LANCE_BUILD_DONE", flush=True)

if __name__ == "__main__":
    sys.exit(main())
