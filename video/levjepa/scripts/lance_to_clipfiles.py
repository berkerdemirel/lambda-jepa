"""Fix B (E34, 2026-09-03): re-pack a Lance frame store into ONE SMALL FILE PER CLIP so training reads are
one contiguous read per clip (the ImageNet access pattern; BeeGFS handles ~9 MB files well) and the frames
stay exactly the JPEGs the donor's builder produced (15 fps, short edge 384, JPEG q90). Sequential scan of
one fragment (fast on BeeGFS: ~1,000 rows/s) -> for every episode a file `{out}/{label}/{episode_idx}.clip`
= header (n_frames, offsets) + concatenated JPEG bytes. Usage: lance_to_clipfiles.py <store> <out_dir> <fragment_idx>
(one SLURM array task per fragment; the label names come from <store>.classes.txt)."""
import os, sys, struct, time
import lance, numpy as np

store, out, frag_idx = sys.argv[1], sys.argv[2], int(sys.argv[3])
classes = open(store + ".classes.txt").read().split("\n")
ds = lance.dataset(store); frags = ds.get_fragments(); frag = frags[frag_idx]
os.makedirs(out, exist_ok=True)
t0 = time.time(); n_clips = n_rows = 0; cur_ep = None; buf = []; cur_label = None

def flush():
    global n_clips
    if cur_ep is None or not buf: return
    d = os.path.join(out, classes[cur_label].replace("/", "_")); os.makedirs(d, exist_ok=True)
    p = os.path.join(d, f"{cur_ep}.clip"); tmp = p + ".part"
    offs = np.cumsum([0] + [len(b) for b in buf]).astype(np.int64)
    with open(tmp, "wb") as f:
        f.write(struct.pack("<I", len(buf))); f.write(offs.tobytes()); [f.write(b) for b in buf]
    os.replace(tmp, p); n_clips += 1

for batch in frag.scanner(columns=["episode_idx", "step_idx", "frame", "label"], batch_size=256).to_batches():
    eps = batch["episode_idx"].to_pylist(); frames = batch["frame"].to_pylist(); labels = batch["label"].to_pylist()
    for ep, fr, lab in zip(eps, frames, labels):
        if ep != cur_ep:
            flush(); cur_ep, cur_label, buf = ep, lab, []
        buf.append(fr); n_rows += 1
flush()
print(f"fragment {frag_idx}: {n_clips} clips, {n_rows} rows in {time.time() - t0:.0f}s = {n_rows/(time.time()-t0):.0f} rows/s", flush=True)
print("CLIPFILES_DONE", flush=True)
