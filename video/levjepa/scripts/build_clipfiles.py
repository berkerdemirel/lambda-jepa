"""Clip-file stores for the video EVALUATION sets (E34, Berker 2026-09-06: "we need to do ssv2 and k400 as they did").
Same file format as the K710 training stores (data/clipfile_loader.py: header n_frames + int64 offsets + JPEG q90 frames,
one file per clip, `{out}/{split}/{class}/{stem}.clip`, `{out}/classes.txt` = the label index space), built straight from
the sources — no Lance store, no extracted-mp4 intermediate:
  --ssv2      the Qualcomm webm files + the label JSONs; frames kept at NATIVE rate (12 fps) and NATIVE size (240p): V-JEPA's
              SSv2 probe samples `frame_step 4` at the native rate and resizes the short side itself.
  --kinetics  the CVDF tarballs streamed once each (members read into memory, decoded by decord from bytes); labels from the
              K400 annotation CSV keyed by `{youtube_id}_{start:06d}_{end:06d}`; stored at 15 fps (integer stride off the
              native rate, the K710 builder's convention) and short edge 256 (V-JEPA's probe resizes the short side to 224;
              its training crops are resized from whatever the source gives) — half the bytes of the 384 training stores.
Resumable (existing non-empty clip files are skipped); one SLURM array task per shard (--shard i/n).
Usage: build_clipfiles.py --ssv2 --split train|validation --out <root> [--shard i/n] [--limit N]
       build_clipfiles.py --kinetics --split train|val --tars <tar.gz ...> --annotations <csv> --out <root> [--limit N]"""
import argparse, io, json, os, struct, sys, tarfile, time
from multiprocessing import get_context

import numpy as np

JPEG_Q = 90


def encode_video(task):
    """(key, source, short_edge, target_fps, max_frames, min_frames) -> (key, [jpeg bytes] | None). source = path or bytes."""
    from PIL import Image
    from decord import VideoReader, cpu
    key, src, short_edge, target_fps, max_frames, min_frames = task
    try:
        vr = VideoReader(io.BytesIO(src) if isinstance(src, bytes) else src, ctx=cpu(0), num_threads=1)
        fps = vr.get_avg_fps() or 30.0
        stride = max(1, round(fps / target_fps)) if target_fps else 1
        n = min(len(vr) // stride, max_frames)
        if n < min_frames:
            return key, None
        out = []
        for base in range(0, n, 32):
            for arr in vr.get_batch([(base + i) * stride for i in range(min(32, n - base))]).asnumpy():
                img = Image.fromarray(arr)
                if short_edge:
                    w, h = img.size
                    nw, nh = (round(w * short_edge / h), short_edge) if h < w else (short_edge, round(h * short_edge / w))
                    img = img.resize((nw, nh), Image.BILINEAR)
                buf = io.BytesIO(); img.save(buf, format="JPEG", quality=JPEG_Q); out.append(buf.getvalue())
        return key, out
    except Exception as exc:  # one bad video must not kill the build; counted by the caller
        print(f"  {key} failed: {type(exc).__name__}: {str(exc)[:80]}", flush=True)
        return key, None


def write_clip(path, jpegs):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    offs = np.cumsum([0] + [len(b) for b in jpegs]).astype(np.int64)
    tmp = path + ".part"
    with open(tmp, "wb") as f:
        f.write(struct.pack("<I", len(jpegs))); f.write(offs.tobytes()); [f.write(b) for b in jpegs]
    os.replace(tmp, path)


def done(path):
    return os.path.exists(path) and os.path.getsize(path) > 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ssv2", action="store_true"); ap.add_argument("--kinetics", action="store_true")
    ap.add_argument("--split", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--videos", default="/mnt/beegfs/locatgrp/shared/datasets/ssv2/videos/20bn-something-something-v2")
    ap.add_argument("--labels", default="/mnt/beegfs/locatgrp/shared/datasets/ssv2/labels/labels")
    ap.add_argument("--tars", nargs="*", default=[]); ap.add_argument("--annotations", default=None)
    ap.add_argument("--short-edge", type=int, default=None); ap.add_argument("--target-fps", type=float, default=None)
    ap.add_argument("--max-frames", type=int, default=192); ap.add_argument("--min-frames", type=int, default=None)
    ap.add_argument("--workers", type=int, default=int(os.environ.get("SLURM_CPUS_PER_TASK", 8)))
    ap.add_argument("--shard", default="0/1"); ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()
    si, sn = (int(x) for x in a.shard.split("/"))
    t0 = time.time(); stats = {"written": 0, "skipped": 0, "dropped": 0, "unlabeled": 0}
    ctx = get_context("spawn"); pool = ctx.Pool(a.workers)
    out_split = os.path.join(a.out, a.split)

    def sink(key, jpegs, label):
        if jpegs is None:
            stats["dropped"] += 1; return
        write_clip(os.path.join(out_split, label.replace("/", "_"), key + ".clip"), jpegs); stats["written"] += 1
        if stats["written"] % 500 == 0:
            el = time.time() - t0
            print(f"  {stats['written']} written ({stats['written'] / el:.1f}/s), {stats['dropped']} dropped, {stats['skipped']} skipped, {el / 60:.0f} min", flush=True)

    if a.ssv2:
        short_edge, target_fps, min_frames = a.short_edge or 0, a.target_fps or 0, a.min_frames or 1
        classes = sorted(json.load(open(os.path.join(a.labels, "labels.json"))).items(), key=lambda kv: int(kv[1]))
        names = [k for k, _ in classes]; index = {k: int(v) for k, v in classes}
        if si == 0:
            os.makedirs(a.out, exist_ok=True); open(os.path.join(a.out, "classes.txt"), "w").write("\n".join(names) + "\n")
        items = json.load(open(os.path.join(a.labels, f"{a.split}.json")))
        items = [(it["id"], names[index[it["template"].replace("[", "").replace("]", "")]]) for it in items]
        items = sorted(items)[si::sn]
        if a.limit: items = items[:a.limit]
        print(f"ssv2 {a.split} shard {si}/{sn}: {len(items)} videos, native rate/size, {a.workers} workers", flush=True)
        todo = [(vid, lab) for vid, lab in items if not done(os.path.join(out_split, lab.replace("/", "_"), vid + ".clip"))]
        stats["skipped"] = len(items) - len(todo); labels = dict(todo)
        tasks = [(vid, os.path.join(a.videos, vid + ".webm"), short_edge, target_fps, a.max_frames, min_frames) for vid, _ in todo]
        for key, jpegs in pool.imap_unordered(encode_video, tasks, chunksize=8):
            sink(key, jpegs, labels[key])
    elif a.kinetics:
        import csv
        short_edge, target_fps, min_frames = (256 if a.short_edge is None else a.short_edge), (15.0 if a.target_fps is None else a.target_fps), a.min_frames or 32
        rows = list(csv.DictReader(open(a.annotations)))
        label_of = {f"{r['youtube_id']}_{int(r['time_start']):06d}_{int(r['time_end']):06d}": r["label"] for r in rows}
        names = sorted({r["label"] for r in rows})
        if si == 0:
            os.makedirs(a.out, exist_ok=True); open(os.path.join(a.out, "classes.txt"), "w").write("\n".join(names) + "\n")
        tars = sorted(a.tars)[si::sn]
        print(f"kinetics {a.split} shard {si}/{sn}: {len(tars)} tarballs, {len(label_of)} annotated clips, 15 fps / short edge {short_edge}, {a.workers} workers", flush=True)

        def members():
            n = 0
            for tb in tars:
                with tarfile.open(tb, "r|gz") as tf:
                    for m in tf:
                        if not m.isfile() or not m.name.endswith(".mp4"): continue
                        stem = os.path.basename(m.name)[:-4]; lab = label_of.get(stem)
                        if lab is None:
                            stats["unlabeled"] += 1; continue
                        if done(os.path.join(out_split, lab.replace("/", "_"), stem + ".clip")):
                            stats["skipped"] += 1; continue
                        yield (stem, tf.extractfile(m).read(), short_edge, target_fps, a.max_frames, min_frames)
                        n += 1
                        if a.limit and n >= a.limit: return
                print(f"  tarball {os.path.basename(tb)} streamed ({stats['written']} written so far)", flush=True)
        for key, jpegs in pool.imap(encode_video, members(), chunksize=4):
            sink(key, jpegs, label_of[key])
    pool.close(); pool.join()
    print(f"CLIPFILES_DONE {a.split} shard {si}/{sn}: {stats} in {(time.time() - t0) / 60:.1f} min", flush=True)


if __name__ == "__main__":
    sys.exit(main())
