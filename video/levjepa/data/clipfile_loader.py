"""Clip-file dataset (fix B, E34): one `{label}/{episode}.clip` file per clip = the donor builder's JPEG
frames at 15 fps, short edge 384 (see scripts/lance_to_clipfiles.py). A sample = 16 frames at the training
stride from a random start, decoded with PIL — identical frames to the Lance path; the header (frame count and
offsets) then ONE window read of the 31-frame span (~1.2 of the ~5 MB file; 2026-09-03: BeeGFS is bound per
request and per client node, ~35 whole-file vs ~60-80 span reads per second per node, not per byte).
Output: {"frame": uint8 (T, C, H, W), "label": int}."""
import io, os, random, struct
import numpy as np, torch
from PIL import Image


class ClipFileDataset(torch.utils.data.Dataset):
    def __init__(self, roots, classes_file, num_frames=16, frame_stride=2, clips_per_video=1, random_crop=True, transform=None, name="clipfiles",
                 shard=None, stage_dir=None, stage_limit=0):
        """roots: clip-file directories ({label}/{episode}.clip); classes_file: the shared label list written by
        the store builder (<store>.classes.txt, identical for every set) = the global label index space.
        shard=(rank, world): this rank keeps a FIXED slice of the sorted file list (rank::world, cut to equal length) instead of
        Lightning's per-epoch DistributedSampler split — the same expectation, and it lets stage_dir copy the slice ONCE to the
        node's local disk (2026-09-06: BeeGFS serves ~25-35 clip reads per second per node, which bounds a 120k-clip epoch at
        ~10 min on eight nodes; the copy costs ~7 min per rank, the epochs then read local NVMe). stage_limit caps the slice (smokes)."""
        self.classes = [c for c in open(classes_file).read().split("\n") if c]
        index = {c.replace("/", "_"): i for i, c in enumerate(self.classes)}
        self.files, self.labels = [], []
        for root in roots:
            for c in sorted(d for d in os.listdir(root) if os.path.isdir(os.path.join(root, d))):
                for f in sorted(os.listdir(os.path.join(root, c))):
                    if f.endswith(".clip"): self.files.append(os.path.join(root, c, f)); self.labels.append(index[c])
        if shard is not None:
            rank, world = shard; n = len(self.files) // world
            self.files, self.labels = self.files[rank::world][:n], self.labels[rank::world][:n]
            if stage_limit: self.files, self.labels = self.files[:stage_limit], self.labels[:stage_limit]
            if stage_dir: self.files = self._stage(self.files, stage_dir, rank)
        self.num_frames, self.frame_stride, self.clips_per_video = num_frames, frame_stride, clips_per_video
        self.random_crop, self.transform, self.name = random_crop, transform, name

    @staticmethod
    def _stage(files, stage_dir, rank):
        """Copy files to stage_dir (16 threads), keeping the last three path parts (set/class/file); files already present
        with the same size are skipped (a resumed job on the same node reuses them)."""
        import shutil, time
        from concurrent.futures import ThreadPoolExecutor
        t0 = time.time(); dst = [os.path.join(stage_dir, *f.split(os.sep)[-3:]) for f in files]
        for d in {os.path.dirname(x) for x in dst}: os.makedirs(d, exist_ok=True)
        def cp(pair):
            if not (os.path.exists(pair[1]) and os.path.getsize(pair[1]) == os.path.getsize(pair[0])): shutil.copyfile(pair[0], pair[1])
            return os.path.getsize(pair[1])
        with ThreadPoolExecutor(16) as ex: nbytes = sum(ex.map(cp, zip(files, dst)))
        print(f"[stage] rank {rank}: {len(files)} clip files, {nbytes / 1e9:.1f} GB -> {stage_dir} in {(time.time() - t0) / 60:.1f} min", flush=True)
        return dst

    def __len__(self): return len(self.files) * self.clips_per_video
    def describe(self): return f"ClipFileDataset {self.name}: {len(self.files)} clips x{self.clips_per_video}, {len(self.classes)} classes"

    def _read(self, path):
        with open(path, "rb") as f:
            n = struct.unpack("<I", f.read(4))[0]
            offs = np.frombuffer(f.read(8 * (n + 1)), dtype=np.int64); base = 4 + 8 * (n + 1)
            span = self.frame_stride * (self.num_frames - 1) + 1
            start = random.randrange(0, n - span + 1) if (self.random_crop and n > span) else 0
            last = min(start + span - 1, n - 1)
            f.seek(base + int(offs[start])); blob = f.read(int(offs[last + 1] - offs[start]))   # the span only
        idx = [min(start + i * self.frame_stride, n - 1) for i in range(self.num_frames)]
        frames = [np.asarray(Image.open(io.BytesIO(blob[int(offs[i] - offs[start]): int(offs[i + 1] - offs[start])])).convert("RGB")) for i in idx]
        return torch.from_numpy(np.stack(frames)).permute(0, 3, 1, 2)

    def __getitem__(self, idx):
        i = idx // self.clips_per_video
        s = {"frame": self._read(self.files[i]), "label": self.labels[i]}
        return self.transform(s) if self.transform else s
