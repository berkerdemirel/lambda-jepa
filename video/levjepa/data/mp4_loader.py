"""mp4-backed clip dataset for E34 (ssl_project, 2026-09-03): the ImageNet access pattern for video.

Why: the donor's Lance frame stores are unusable on BeeGFS for random clip reads (a cold row take costs
24-210 s per clip; range reads and 8-way concurrency did not rescue it — measured, E34 card). Kinetics
is 120k short clips, so each clip is read as ONE small file (the extracted mp4, ~2 MB) and decoded with
decord in the worker, exactly the frames the Lance builder would have stored: native fps / round(fps/15)
= the 15 fps store rate, times the training stride 2 -> 16 frames ~2.1 s apart at 7.5 fps, short edge
resized to 384. Files come from BeeGFS or, when staged, from the node's RAM disk (LEVJEPA_MP4_ROOT).
Output per sample: {"frame": uint8 (T, C, H, W), "label": int} -> the same transforms as the Lance path.
"""
import csv
import os
import random

import numpy as np
import torch
import torch.utils.data

SHORT_EDGE = 384


class Mp4ClipDataset(torch.utils.data.Dataset):
    def __init__(self, roots, draw_csv, num_frames=16, frame_stride=2, clips_per_video=1, random_crop=True,
                 transform=None, name="mp4"):
        """roots: list of directories holding {label}/{id}_{start}_{end}.mp4 (one per Kinetics version);
        draw_csv: the K710-20% draw (youtube_id,time_start,time_end,label,source) — defines the file
        list and the shared label index; files missing on disk (mirror-unavailable) are skipped."""
        with open(draw_csv) as f:
            rows = list(csv.DictReader(f))
        classes = sorted({r["label"] for r in rows})
        index = {c: i for i, c in enumerate(classes)}
        self.files, self.labels = [], []
        by_source = {}
        for root in roots:
            by_source[os.path.basename(root.rstrip("/")).split("k710_20pct_videos_")[-1]] = root
        for r in rows:
            root = by_source.get(r["source"])
            if root is None:
                continue
            p = os.path.join(root, r["label"], f'{r["youtube_id"]}_{int(r["time_start"]):06d}_{int(r["time_end"]):06d}.mp4')
            if os.path.exists(p):
                self.files.append(p); self.labels.append(index[r["label"]])
        self.num_frames, self.frame_stride, self.clips_per_video = num_frames, frame_stride, clips_per_video
        self.random_crop, self.transform, self.name, self.classes = random_crop, transform, name, classes

    def __len__(self):
        return len(self.files) * self.clips_per_video

    def describe(self):
        return f"Mp4ClipDataset {self.name}: {len(self.files)} clips x{self.clips_per_video}, {len(self.classes)} classes"

    def _decode(self, path):
        from decord import VideoReader, cpu
        vr = VideoReader(path, ctx=cpu(0), num_threads=1)
        n, fps = len(vr), (vr.get_avg_fps() or 30.0)
        stride = self.frame_stride * max(1, round(fps / 15.0))      # store rate 15 fps x training stride
        span = stride * (self.num_frames - 1) + 1
        start = random.randrange(0, n - span + 1) if (self.random_crop and n > span) else 0
        idx = [min(start + i * stride, n - 1) for i in range(self.num_frames)]
        arr = vr.get_batch(idx).asnumpy()                                  # (T, H, W, 3)
        x = torch.from_numpy(arr).permute(0, 3, 1, 2)                       # (T, C, H, W) uint8
        h, w = x.shape[-2:]
        if min(h, w) != SHORT_EDGE:
            s = SHORT_EDGE / min(h, w)
            x = torch.nn.functional.interpolate(x.float(), size=(round(h * s), round(w * s)), mode="bilinear",
                                                antialias=True).round().clamp(0, 255).to(torch.uint8)
        return x

    def __getitem__(self, idx):
        i = idx // self.clips_per_video
        sample = {"frame": self._decode(self.files[i]), "label": self.labels[i]}
        return self.transform(sample) if self.transform else sample


def mp4_roots(base):
    return [os.path.join(base, f"k710_20pct_videos_{s}") for s in ("k700_2020", "k600", "k400")]
