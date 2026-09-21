# E34 — The house recipe on video: LeVJEPA's trainer with the two-space conditioner (K710-20%, ViT-S first)

## Why

Berker's agenda (2026-09-02, HANDOVER §7): take the LeVJEPA video pretraining recipe — one encoder,
token dropping, block-causal attention, LeJEPA's invariance + SIGReg at Z — and run OUR recipe in
it: six global views, invariance at Z, the two-sided spectral conditioner on per-clip view centers
at Z and at the retained backbone H, doses as calibrated at ImageNet-1k. The question is the
project's question on a new modality: does conditioning the centers at H, next to the invariance
objective at Z, improve the frozen representation of a video encoder as it did for images
(E20/E27), under a recipe that never touches H otherwise. Not for the ICLR paper (D-113 clause 5:
exempt from the 16-H100 cap).

## Frame (anatomy stated loudly)

- **Data:** K710-20% — the union of the Kinetics-400/600/700-2020 TRAIN sets (CVDF mirror tarballs
  on BeeGFS), duplicates removed by YouTube id, validation overlap removed (any id in a K400/K600/
  K700 validation list), then 20 percent PER CLASS, seed 0 (Berker 2026-09-02). Their own 20 percent
  list is not released, so the subsample is ours and differs from theirs by construction. RAW draw
  (job 64174352, `scratch/video/k710_subsample.py`): union 649,798 ids → 623,048 after removing
  26,750 validation-overlap ids, 724 normalized label strings (K710 proper is 710: naming variants
  across versions split 14 classes; a per-class 20 percent draw is unaffected by a split), draw
  124,615 clips (K700-2020 101,772 / K600 14,473 / K400 8,370; per class 9 to 363, median 164);
  lists `kinetics/k710_union.csv`, `k710_20pct.csv` on BeeGFS. RAW extraction (2026-09-03 ~02:00,
  `scratch/video/kinetics_extract.py`, one streaming pass per tarball, members taken only from the
  version the draw assigned): K700-2020 101,772 / 101,772; K600 13,552 / 14,473 — the 921 missing
  ids are absent from their class tarballs on the CVDF mirror under any window (mirror availability,
  181 classes touched, at most 39 in one class); K400 7,342 / 8,370 after its duplicate cleanup
  (the K400 and K600 extraction arrays ran before the source filter existed and were cleaned
  afterwards; 1,028 ids absent from the mirror), of which 30 members were the mirror's known
  truncated-at-4-MiB videos and were replaced from CVDF's `replacement_for_corrupted_k400.tgz`
  (all 30 recovered). Stores (RAW): K600 13,520 episodes / 1,881,447 frame rows, 32 clips dropped
  (2 undecodable, 30 shorter than 32 stored frames); K400 7,312 episodes / 1,008,618 rows, 30 dropped
  (4 undecodable — truncated at 8 MiB in the mirror, no moov atom, not in the replacement archive — and
  26 too short); K700 99,268 episodes / 12,833,864 rows, 2,504 dropped (3 undecodable, 2,501 shorter
  than 32 stored frames — K700-2020 carries more sub-2 s clips). **Trained set = 120,100 clips,
  15.72 M frame rows over the three stores (96.4 percent of the 124,615 drawn); the rest is
  mirror-unavailable or too short.** Stores complete 2026-09-03 05:15; data-path smoke PASSED 06:20
  (`scripts/k710_data_smoke.py`, job 64275757): the three stores compose to one 120,100-clip epoch
  (82.7 / 11.3 / 6.1 percent by version), batches of six 16-frame 224 views as uint8, first batch
  91 s (episode index build), then 0.1 s per batch of 8 with 4 workers. The trained set
  is therefore the draw minus mirror-unavailable and too-short clips, to be stated with the final count.
  Store = three per-version Lance
  stores (`k710_20pct_{k700_2020,k600,k400}.lance`, one episode per clip) read as one dataset
  through their mixture loader (`conf/data/k710_20pct.yaml`). Download fact: the CVDF mirror
  from this cluster tops out near 90 MB/s aggregate whatever the stream count (measured 19:55),
  so the 2.6 TB of tarballs take ~8 h; per-set chains start each store as its set lands. Frames
  encoded with their Walking Tours convention: 15 fps, short edge 384, JPEG q90, one episode per
  video, Lance store. Evaluation sets: ImageNet-1k (frames repeated), SSv2 (Qualcomm), K400 val.
- **Model:** their ViT-S/16, 16 frames at stride 2 (7.5 fps), tubelet 1, RoPE, token drop .95,
  block-causal attention; projector 2048 → 256 with BatchNorm (theirs). Runs on `video/levjepa`
  (donor MLO-lab/LeVJEPA @3ea0dda; PORT_NOTES `docs/methods/levjepa.md`).
- **Recipe (ours, the v6 form; `conf/sslgap_vits.yaml`):** V = 6 global views @224 of the same
  16-frame clip through the lejepa view stack; inv = view-to-mean MSE at Z on the all-pairs scale;
  conditioner on per-clip view centers, Z slice 128 ring q3, H (CLS at the encoder output) slice
  128 ring q3; doses w = (31.07 / 225.50 / 1.671) = the e27v6s chain, as they are; no twin, no OAS;
  grad clip 1.0. Optimizer, schedule, batch and precision = their released config (AdamW 4e-4,
  wd .04, betas (.9,.95), 1,200 warmup steps then flat, effective batch 3,072, bf16-mixed);
  epochs 240 as in their grid; one random temporal crop per clip per epoch (`clips_per_video: 1` per
  store — the donor's loader default of 200 is a Walking Tours setting and would have made an epoch
  24 M clips; caught by the data-path smoke 2026-09-03), so an epoch is 120,100 clips = 39 optimizer
  steps at batch 3,072 and 240 epochs = 9,360 steps. Estimator anatomy: view-mean centers at both taps, global batch
  under DDP (n = 3,072 rows per step before the ring), n/d′ ≥ 24 at both taps.
- **Instruments:** their per-step loss terms; the house `[share]` line each epoch (realized shares,
  g per term, omega_h / omega_z / lam over the six views, moment-KL values, whitened trace/d′ of
  both center streams); per-epoch checkpoints (their ModelCheckpoint, every_n_epochs 1 for this
  card); wandb online.
- **Evaluation (landing = bench only, in1k rule extended):** frozen attentive probe on ImageNet-1k,
  the V-JEPA protocol ported from facebookresearch/jepa (`video/levjepa/scripts/attentive_probe.py`,
  wrapper `slurm/attentive_probe_in1k.slurm`; unit-checked 2026-09-02, not yet run on a real
  checkpoint): one learnable query + one cross-attention block over the encoder's full token set
  (no dropping), images repeated over the 16 frames, timm 'original' AutoAugment + random erasing
  .25 for training, resize 256 + center crop 224 for validation, AdamW lr 1e-3 cosine to 0 without
  warmup, wd 1e-3 cosine to 1e-6 (biases and norms excluded), 20 epochs, bf16. Protocol notes to
  carry with every number: V-JEPA's global batch is 1,024 (64 GPUs x 16), ours 128 (8 x 16) at the
  same lr; the weights evaluated are the donor's EMA encoder by default (`--weights ema`), the
  student is reported next to it. K400 linear probe and SSv2 attentive probe: later.

## Cells

| cell | loss | views | arch | epochs | status |
|---|---|---|---|---|---|
| `vid.floorssl.s0.k710s` (FIRST) | sslgap: inv + cond_z + cond_h, doses (31.07/225.50/1.671) | 6 global | ViT-S/16 | 240 | LAUNCHED 2026-09-03 13:05 (after a morning of driver-broken nodes: three earlier starts died at initialization, one healthy start was cancelled by a watcher false positive, and two duplicate starts were cancelled), job 64289395, 2 nodes × 4 H100 (gpu268, gpu272), batch 96 × 8 cards × accum 4 = 3,072, 22 loader workers per card; wandb run id q8l2osu5 (three dead same-name wandb runs f0655tm7, uveuxb6m, 29g572zu are not the run); run dir `/mnt/beegfs/locatgrp/shared/bdemirel/levjepa_runs/vid.floorssl.s0.k710s`, wandb `sslgap/vid.floorssl.s0.k710s`; the four other submitted shapes were cancelled by the first-wins watcher |
| `vid.floorssl.s0.k710s2` (SECOND LAUNCH of S: GPU photometrics + span reads, batch 3,072 = accum 4) | sslgap, doses (31.07/225.50/1.671), rings q3/q3 (n/d′ 24 within-step) | 6 global | ViT-S/16 | 240 | ran 2026-09-03 20:02 → 09-04 13:39 (job 64377149, epochs 0–~100, 39 optimizer steps per epoch); CANCELLED by Berker's decision to relaunch without accumulation; kept as the THEIR-BATCH CONTRAST — checkpoints under its run dir, K710 probe curve and share lines on this card |
| `vid.floorssl.s0.k710s3` (THIRD LAUNCH of S, Berker 2026-09-04 ~13:35 "more faithful to our imagenet chains as it wont be underutilizing the epochs") | sslgap, doses as above; NO accumulation: batch 768 per optimizer step, 156 steps/epoch (37,440 in 240); rings OFF (768 fresh rows / 128 = n/d′ 6, the ImageNet chains' certified 4); warm-up 12 % of the run (peak_step 4,493); lr 4e-4 | 6 global | ViT-S/16 | 240 | LAUNCHED 13:41 as job 64475095 (rank 0 OOM-killed in the probe draw, see the incident note); RELAUNCHED 14:05 as job 64479565 on 8 nodes × 1 H100 (eight BeeGFS clients), 2 × 4 resumer 64479566 chained; online K710 probe from step 0 |
| `vid.floorssl.s0.k710b` (B CELL, Berker 2026-09-04 "overlap is fine just to be ready to the best case"; third-launch settings: batch 512 per step = 234 steps/epoch, z ring off (n/d′ 4), h ring q1 (n/d′ 4), warm-up peak_step 6,739) | sslgap, doses = the e27v6b400 chain (22.81/222.26/4.054), rings z-q3 / h-q7, slices 128 / 256 | 6 global | ViT-B/16 | 240 | smoke 64450968 passed; queued 13:41 as job 64475097, which started at 14:01 in the slot freed for S and was cancelled 2 min in; requeued as job 64480238 (`slurm/e34cf3_shape_h100x4x2_vitb.slurm`), pends on the 31-GPU cap until v6Llr100 lands tonight; GPU photometrics + span reads + the online K710 probe from step 0; `share_log_bs` 64 |
| their recipe on our pipeline (control) | LeJEPA: view-to-global + SIGReg .02 | 1 global + 10 local | ViT-S/16 | 240 | NOT launched — Berker's call ("im not sure how necessary"); the only clean external reference for the data pipeline |
| z-only twin (w_h = 0) | inv + cond_z | 6 global | ViT-S/16 | 240 | NOT launched — the E20-style within-recipe control for the H term; proposed, not approved |

## Pre-registered predictions (PROPOSED 2026-09-02 evening; launch authorized by Berker 2026-09-03
morning, verbatim: "that sounds good. if verified we can start cooking!" — read as: the single-GPU
smoke first, then the first cell on the predictions below as written)

- **P1 (formation):** the run passes the ep1–5 formation gate on the doses as transplanted: no
  collapse signature, moment-KL VALUE at Z not below .1 by ep2 (the q2 quench pattern = kill),
  realized shares inv-heavy at ep1 as in every ImageNet chain (inv ≥ .6), the H share a few
  percent. A trip is a finding about the transplant, not a reason to re-dose silently.
- **P2 (the moment channel on video):** the center streams stay full — whitened trace/d′ of both
  streams above .5 through training (the D-084 starvation read was .17 on a malformed stream) —
  and omega_h settles in the band of the ImageNet v6 runs (.3 to .6) by mid-run.
- **P3 (the claim, directional):** the H term improves the frozen ImageNet-1k read of the video
  encoder over the same recipe without it (the z-only twin, if launched), by at least +1 point
  under the same evaluator; against their published ViT-S number the comparison is confounded by
  the subsample and the evaluator and is reported with that caveat, never as a win.
- **P4 (thickness):** at landing the H thickness of the video encoder is measurable and
  moderate (Θ_h within the touch-law band of the image runs), i.e. the six-view recipe does not
  drive the backbone to the invariant end.

## ViT-B smoke (RAW, 2026-09-04 12:0x–12:2x, job 64450968, one H100, K710 clip files)

bs 64 × 6 views × 16 frames with the GPU photometrics, share logger at bs 64, the online K710 probe on 256 held-out
clips, 2 × 60 micro-batches: peak GPU memory 48.2 GB; `[share] ep0` .064 / .909 / .026 (g 2.51 / 3.64 / 5.74),
kl_z 3.15, kl_h 3.77, traces .034 / .223, omega_h 3.95; ep1 .082 / .903 / .015, kl_z 2.13; the probe callback drew its
batch (254 s) and scored (0.00 after 3,840 clips of head training, as expected for a 724-way head); step-50 rate 1.1
it/s on the loader-bound first epoch. Passed → the 2 × 4 shape (`slurm/e34cf2_shape_h100x4x2_vitb.slurm`, bs 64 ×
accum 6) queued 12:30 under the QOS cap (starts after v6Llr100 lands).

## Third launch of S and the B settings — declared deviations (2026-09-04 13:41; predictions P1–P4 and P-B1–B4 carried as written)

Berker: "i would cancel the s cell and b cell runs. i think this is more faithful to our imagenet chains as it wont be
underutilizing the epochs" / "you can btw submit it 8x1h100s if it will make us gain time". The second launch made 39
optimizer steps per epoch at the donor's batch 3,072 (the ImageNet S chain makes 10,009); the third makes 156 at batch
768 (B: 234 at 512) with the same epochs and data — matched epochs and data, 4× (B: 6×) the updates. Declared: (1) batch
768 / 512 at the donor's lr 4e-4 (per sample 4×/6× hotter than their 3,072; under clip 1.0, as the ImageNet chains at
batch 128, lr 1e-3); (2) the conditioner's rows carried as the RATIO n/d′ (S: rings off → 6 fresh; B: z off → 4, h q1 →
4), not as the ring length (q3 literally would give 24 within-step); (3) warm-up kept at the donor's 12 percent of the
run (peak_step 4,493 / 6,739 instead of 1,200); (4) the 240-epoch read against LeVJEPA's ViT-S 39.4 / ViT-B 50.7 is now
"matched epochs and data, more updates", reported as such. Read (not agreed) that motivated it: the second launch's
in-domain probe rose but decelerated (h.cls linear 9.3 → 11.2 → 12.5 at epochs 30/60/94) and its kl_h peaked at
epochs 60–70; the h dosing question (trace H .05) is unchanged by this relaunch.

**Third-launch incident (RAW, 13:41–13:59):** the first 8 × 1 job (64475095) lost rank 0 to a host-memory kill at 13:56
(`Detected 1 oom_kill event`) while the online probe drew its 1,024-clip held-out batch as ONE loader batch (six views
per clip materialized in a worker, ~15 GB plus the collate copy, on top of the rank's ~180 GB of training prefetch, with
200 GB per node). Fix: the draw runs in chunks of 64 and keeps one view per clip on arrival (`K710Probe.on_train_start`),
and the 8 × 1 shape asks 300 GB per node. Resubmitted 13:59; the run dir and wandb name are unchanged (fresh start).

**Third-launch pace (RAW, 14:05–15:00):** epochs 0/1/2 took 454 / 631 / 828 s at 22 loader workers per rank; sstat showed
2.3 of 24 CPUs busy per rank and the per-node BeeGFS rate ~18 clips/s — with one rank per node the shape ran only 22 concurrent
reads per node, on the rising part of the client's concurrency curve (the 2 × 4 shape had 88 per node and reached 55–85
files/s). Variant `e34cf3_shape_h100x8x1_w48.slurm` (48 workers on 48 CPUs per rank, same run, resume clause) chained to take
over at the epoch-3 checkpoint (job 64492719; 2 × 4 resumer 64492720 behind it). Same recipe, same run; an efficiency change.

## Pre-registered predictions for the ViT-B cell (PROPOSED 2026-09-04 midday, before any B number exists; written with the S cell at epoch 95 in view)

Reference points (docs/methods/levjepa.md, read from their Figure 2): matched-epoch IN-1k attentive probe, 240 epochs on
the 20 % K710 subsample — LeVJEPA ViT-B 50.7, V-JEPA 2 ViT-B 51.6, VideoMAEv2 ViT-B 47.1; ViT-S: LeVJEPA 39.4, V-JEPA 2 38.7.

- **P-B1 (formation):** as P1 — the ep1–5 gate passes on the transplanted doses; inv-heavy shares from ep1; no quench.
- **P-B2 (the moment channel, revised by what S showed):** the Z stream reaches whitened trace/d′ > .5 by epoch 60 as
  in S; the H stream at the transplanted h dose (4.054, ×2.4 the S dose in nominal terms) THINS below .2 by epoch 60 (S
  reads .05 at that point) — the starvation recurs; if the H trace holds above .3 the higher B dose is what differs.
- **P-B3 (the claim against the published points, directional):** the epoch-240 IN-1k attentive read (their protocol,
  our port; caveats: our 20 % draw, batch 128 vs 1,024) lands INSIDE the published ViT-B band, between VideoMAEv2's
  47.1 and LeVJEPA's 50.7 — the honest bet; above 50.7 is the upside surprise; below 47.1 = the recipe does not
  transfer at B.
- **P-B4 (the in-domain curve):** the online K710 linear probe rises monotonically over the first 60 epochs and
  exceeds the S cell's value at matched epochs (the offline probe of 2026-09-04 gives S's curve).
- **S cell, P3 target now known:** LeVJEPA ViT-S 39.4 (the epoch-240 read of the S cell is compared to it with the
  same caveats). **P2 status at S epoch 95 (RAW):** Z passes (.60), H is REFUTED so far (.048 and flat since epoch 45).

## Smoke (RAW, 2026-09-03 10:38–10:53, job 64281756, one A100, Walking Tours store)

200 optimizer steps of the ViT-S recipe at batch 32 with 6 loader workers: exit 0, 805 s for the epoch
(4.0 s per step — decode-and-augment bound: 32 clips × 6 views × 16 frames = 3,072 frame augmentations
per step, ≈ 128 per second per worker), rings filling, clipping at norm 1.0. Share line at initialization
(fixed batch of 64 clips): inv=.084 (g 2.83), moment_kl=.904 (g 4.19), h_moment_kl=.011 (g 7.11);
omega_h 4.96, omega_z 6.34, lam .885; KL values 3.24 / 3.38; whitened trace/d′ .030 (Z) / .186 (H) —
an initialization read, not a formation read. Consequence for the run: at batch 3,072 the step costs
≈ 295k frame augmentations, so the launch shapes carry 22 loader workers per card (24 CPUs); expected
wall time ≈ 12–30 h depending on the shape's total worker count.

## Launch incidents (RAW, 2026-09-03)

Two starts on healthy nodes (13:05 job 64289395, 13:38 job 64292886) never reached their first optimizer
step: ranks 1–7 hit the 30-min DDP timeout while rank 0 was still drawing the share logger's fixed batch.
Root cause measured on a CPU node (`scripts/aug_bench.py`, `take_vs_range.py`): decode of a clip from the
K700 store ran at 2 frames/s per process while the six-view augmentation ran at 256 frame-augs/s — the
loader's Lance `take` costs 24–210 s per clip cold on the 13-fragment store; a range read costs 0.2 s.
Range reads did not rescue it (patched loader 0.2 clips/s with 8 workers; cold `to_table(offset, limit)`
47 s per clip). The cold benchmark's last line explains the shape: within ONE process, the first take on each
fragment costs 100–200 s (fragment metadata read over BeeGFS) and later takes on that fragment cost 0.05 s per
clip (8 threads: 32 clips in 1.4 s) — a 13-fragment store means ~30–40 min of warm-up per loader worker, paid
by every worker on every start. Decision (2026-09-03 15:10): abandon Lance for training reads; **fix B** —
the stores re-packed into one file per clip holding the builder's JPEG frames (`scripts/lance_to_clipfiles.py`,
`data/clipfile_loader.py`, `data=k710_clipfiles`; same frames, one contiguous read per clip). The mp4-decode
path (`data=k710_mp4`, 0.53 clips/s/worker, H.264-decode-bound) stays as the fallback. Measured on the
converted K400 clip files (CPU node, 8 workers, `scripts/clipfile_loader_smoke.py`): **0.79 clips/s per worker**
(first batch 23 s), i.e. ≈ 22 s per 3,072-clip step with 176 workers → ≈ 2.4 days for 240 epochs; the
six-view augmentation is about 40 percent of the per-clip cost. On the K700 files (8.7 MB per clip, read from
444 GB) the same smoke gives 0.58 clips/s per worker → ≈ 30 s per step, ≈ 3.3 days for 240 epochs. Conversion done 15:35 for K700 (99,268 files) and K400 (7,312); K600's second store fragment was missed by
the first array and is converting (job 64311697). The five launch shapes (`slurm/e34cf_shape_*.slurm`,
`data=k710_clipfiles`) are queued behind it (jobs 64312322–26) with the first-healthy-shape-wins watcher.

**Second launch (2026-09-03 20:00; Berker: "okay queue 8x1 h100s", "you can do that cancel").** The first cell
(job 64312322) ran at 53 s per optimizer step, 34 min per 39-step epoch, ≈ 5.7 days. Measured 2026-09-03 evening
(`scratch/video/e34_clip_profile.py`, `e34_read_concurrency.py`, `e34_gpu_bench.py`, `e34_gpu_photometrics_bench.py`,
`e34_nfs_window_probe.py`; outputs `outputs/e34*_<job>.out`): BeeGFS delivers ~35 whole-file (5 MB) or ~60–80 span
(1.2 MB) reads per second PER CLIENT NODE at ~100 ms per request, and that rate scales with nodes (three nodes at
once: ~80 span reads/s each) — the first cell's two nodes sat at 29 clips/s each. On one core a clip costs 0.30 s
of read latency, 0.02 s of JPEG decode, 0.017 s for six RandomResizedCrops and 0.38 s of photometric ops (color
jitter 0.36 of it); an H100 needs 0.28 s per 96-clip micro-batch (forward + backward, bf16). NFS is no faster per
node (cold 1.2 MB windows: 84 vs 52–69 files/s at 64 readers). Changes, all in `video/levjepa/`: (1)
`data/gpu_views.py` — the photometric half of the lejepa stack applied per (clip, view) on the GPU, batched with
per-sample draws, every op 0.000/255 against torchvision's functional reference (A100 and H100); the workers ship
uint8 RandomResizedCrops. DECLARED DEVIATION from the first cell: float math without the uint8 rounding between
consecutive ops (≤ 1/255 per op). Cost 0.81 s per micro-batch on H100. (2) `data/clipfile_loader.py` — header + one
window read of the 31-frame span instead of the whole file. (3) `augmentation.photometrics_on_gpu` switch through
the transform, `build_loader`, `conf/config.yaml` and `main.sslgap_terms` (seeded per rank and micro-batch; the share
measurement's seed is fixed, so its fixed batch keeps fixed views). (4) `main.py`: `spt.set(cache_dir=<run dir>/
spt_cache)` — checkpoints land under the BeeGFS run dir; the first cell's had been redirected by stable-pretraining
to `~/.cache/stable-pretraining/runs/20260903/153311/839b6d934098/checkpoints` (epochs 0–6 kept there, 361 MB each).
Recipe, doses, views, schedule, effective batch and seed unchanged. Shapes `slurm/e34cf2_shape_h100x4x2.slurm` and
`e34cf2_shape_h100x8x1.slurm` (run `vid.floorssl.s0.k710s2`, run dir `levjepa_runs/vid.floorssl.s0.k710s2`, wandb
name the same): both 8 ranks × 96 × accum 4 = 3,072; each shape starts the run if no checkpoint exists and otherwise
CONTINUES it from the newest `last.ckpt` with the same wandb run (rings re-warm over q steps after a resume,
declared). Job 64377149 (2 × 4, the first cell's freed nodes gpu273/gpu277) and job 64377150 (8 × 1 = eight BeeGFS
client nodes, gpu267/269/274 excluded; SLURM's worst-case start estimate 09-10, a start on 09-04 is plausible). Job
64312322 cancelled 20:00 after epoch 6 (`[share] ep7`, step 273). Projection (per-node read caps × nodes): 2 × 4
≈ 18–28 s per step (2–3 days), 8 × 1 ≈ 5–7 s per step (13–19 h). The path was checked piecewise (transform, span
read, config, `sslgap_terms` with the flag on the login CPU); the one-card SLURM smokes never got a card before the
launch, so the 2 × 4 start is the smoke (RAW; read its `[share] ep0` and step-50 rate line first).

## Numbers (RAW)

**Live run (the first cell): job 64312322, started 2026-09-03 15:32 on gpu273 + gpu277 (2 × 4 H100), clip-file
data path, 120,100 clips / 724 classes, batch 96 × 8 × accum 4 = 3,072, 156 micro-batches = 39 optimizer steps per
epoch; epoch wall time 2,053 s (34 min) → 240 epochs ≈ 5.7 days (≈ 09-09).** A second shape (h100x2) started
in the same minute and was cancelled after 3 min; its wandb entry is a dead duplicate.

Share lines (rank 0, fixed 128-clip batch, measurement 1–8 s), read in OPTIMIZER STEPS — a video epoch is 39
steps, so the card's 'ep1–5 formation gate' written in ImageNet epochs (~5,000 steps each) corresponds to most of
this run; the quench gate (kl_z value < .1) is the kill criterion and is far from tripping:

| epoch (step) | inv share (g) | Z share (g) | H share (g) | kl_z / kl_h | trace_z / trace_h | omega_h / omega_z / lam |
|---|---|---|---|---|---|---|
| 0 (init) | .055 (2.41) | .934 (5.58) | .011 (8.76) | 2.90 / 3.01 | .039 / .252 | 3.25 / 4.28 / .87 |
| 1 (39) | .317 (1.07) | .665 (.309) | .018 (1.15) | 1.88 / 2.22 | .056 / .230 | 3.87 / 4.92 / .89 |
| 2 (78) | .455 (.833) | .517 (.130) | .028 (.962) | 1.52 / 2.42 | .059 / .098 | 4.51 / 5.98 / .87 |
| 3 (117) | .473 (.768) | .500 (.112) | .028 (.840) | 1.32 / 2.25 | .070 / .114 | 3.93 / 5.67 / .83 |

Read (not agreed): the transplanted doses put the trunk pull half on the Z conditioner at step 117 (ImageNet
chains: ~.65 inv / .34 Z / .01 H once formed), the shares are moving in that direction each epoch, both center
streams are still far from their fixed point (whitened trace/d′ .07 at Z, .11 at H; the D-084 starvation read on
images was .17), and thickness is at initialization values (omega_h ≈ 4). No collapse signature.

**In-domain K710 probe of the second launch's checkpoints (RAW, 2026-09-04 12:10, job 64445781, `scripts/k710_probe.py`,
`results/e34/k710_probe.csv`):** one clean view per clip (center span, resize 224 + center crop), fixed class-balanced
split of the pretraining clips (20 train + 8 eval per class, seed 0; 14,468 / 5,766 clips, 724 classes, chance 0.14),
k-NN (k 20, cosine) and a linear classifier on the frozen features. Epoch 0 = the initialization.

| epoch | student h kNN / linear | student z kNN / linear | EMA h kNN / linear |
|---|---|---|---|
| 0 | 1.89 / 3.10 | 1.87 / 3.02 | 0.73 / 1.42 |
| 10 | 2.46 / 4.70 | 2.34 / 4.06 | 0.94 / 1.68 |
| 30 | 3.50 / 6.82 | 3.31 / 5.67 | 1.04 / 2.13 |
| 60 | 4.30 / 8.79 | 4.08 / 6.73 | 1.14 / 3.04 |
| 93 | 5.01 / 9.07 | 3.87 / 6.14 | 1.53 / 3.57 |

Read (not agreed): the representation organizes the K710 classes as it trains (h: ×2.7 kNN, ×2.9 linear over the
random-init level by epoch 93), so the training works; the h slope flattens between epochs 60 and 93 (linear +0.3,
kNN +0.7) and z DECLINES after epoch 60 while the inv share rises past .84 — the head's output is losing class
information to invariance while the backbone keeps it (the two-space signature). The EMA lags (25 percent init at
epoch 89, no decay warm-up). The linear probe is data-starved (train 40–50 vs eval 9), so the absolute level is not
a representation quality number; the epoch-120 full attentive read is the first number in the paper's unit.
Subset attentive-probe peeks on the epoch-89 checkpoint (64k ImageNet images, 5 epochs, not the protocol): EMA 5.57,
student 8.91 val top-1 (`results/e34/peek_attnprobe_ep89_{ema,student}.csv`).

**Same probe at 50 train clips per class with the CLS and patch-GAP taps, plus LeVJEPA's released ViT-L VideoMix as the
instrument's reference (RAW, 2026-09-04 12:5x, jobs 64455530 / 64456461, `results/e34/k710_probe50.csv`; 36,031 train /
5,708 eval clips, 724 classes; the reference loads into our `vit_large` unchanged, keys `encoder.*`):**

| model / epoch | h.cls kNN / linear | h.gap kNN / linear | z kNN / linear |
|---|---|---|---|
| ours ViT-S, epoch 0 (student) | 2.28 / 3.84 | 1.28 / 3.03 | 2.10 / 3.45 |
| ours ViT-S, epoch 30 (student) | 4.50 / 9.29 | 3.66 / 9.37 | 4.10 / 8.20 |
| ours ViT-S, epoch 60 (student) | 5.80 / 11.21 | 3.85 / 10.23 | 5.12 / 8.48 |
| ours ViT-S, epoch 93 (student) | 5.43 / 12.35 | 3.89 / 10.81 | 5.34 / 8.88 |
| ours ViT-S, epoch 94 (student) | 5.54 / 12.49 | 3.80 / 10.58 | 4.92 / 8.62 |
| LeVJEPA ViT-L VideoMix (released; reference) | 13.37 / 27.38 | 5.78 / 21.34 | – |

Read (not agreed): CLS decodes better than the patch mean for both models (the loss supervises CLS in both recipes); with
more probe data the S curve is steeper and still rising at 94 (h.cls linear 9.3 → 11.2 → 12.4 over epochs 30/60/94;
kNN plateaus 5.8 → 5.5), z stays below h and flat after 60; the released ViT-L (5× the parameters, the full video mixture)
reads 27.4 linear / 13.4 kNN on CLS, so our ViT-S at 40 percent of its schedule sits at 45 percent of that model's linear
read on this instrument. Not the landing protocol.

## AGREED TAKEAWAY

*(empty — filled only with Berker)*

**Third launch, 48-worker takeover (RAW, 2026-09-04 15:07–15:24).** The 22-worker 8 × 1 (64479565) was cancelled at 15:07 after
its epoch-3 checkpoint; its epoch-4 lines had already printed: `[share] ep4 inv=0.099 moment_kl=0.889 h_moment_kl=0.012 |
kl_z=0.832 kl_h=1.254 trace_z=0.460 trace_h=0.297`, `[probe] ep4 k710_top1=2.83` (624 optimizer steps; chance 0.14 %). The 48-worker
copy (64492719) and its chained 2 × 4 resumer (64492720) crashed at start: the resume clause passed `wandb.config.id=<id>`, a key the
config does not declare (Hydra rejects it without `+`); the trainer's own mechanism is a `wandb_resume.json` sidecar in the working
directory, which the shared `video/levjepa` directory made common to the S and B cells. Fix in all four `e34cf3_shape_*.slurm`: run
from the cell's run directory, pass only `resume.ckpt_path`; the S sidecar (run un990fbr) moved into its run directory. In the gap
the B cell (64480238) took the freed cards for 3 min (fresh start, no checkpoint) and was cancelled; it is resubmitted once S runs.
Relaunch: 64498300 (resumer 64498301). Lost: ~17 min of S wall-clock.
Relaunch 64498300 started 15:26 on gpu265,266,268,271,272,273,275,277 and resumed the epoch-3 checkpoint, but still opened a
fresh wandb run (tsrjm4mc): the donor's `main.py` logged the hyper-parameters at logger creation, which fires `wandb.init` before
the trainer injects the resumed id; the call is removed (the config reaches wandb through the trainer at fit start). Third-launch
wandb runs: un990fbr epochs 0–3, tsrjm4mc epoch 4 onward; trainer step axis is continuous (624 optimizer steps at the seam).
B cell resubmitted as 64499984 at 15:29 (pends on the 31-GPU cap).
15:45 — the 48-worker relaunch (64498300, mem 300G) lost rank 0 to a host-memory kill during start-up (after the share's fixed-batch
draw, 457 s; before the first logged step): 48 workers × prefetch 2 × ~1.4 GB micro-batches in shared memory plus ~3 GB per
worker process and rank 0's share/probe draws exceed 300 GB. Resubmitted at 600G per node (nodes have 2 TB): 64507327, resumer
64507328. Its wandb run tsrjm4mc stays empty; the relaunch re-injects un990fbr from the unchanged sidecar.
Relaunch 64507327 (600G) trains since 16:20 in wandb run un990fbr. Epoch times under 48 workers: epoch 4 300 s (the ranks' timer;
rank 0 reads 875.7 s with the start-up inside — the prefetch burst of 48 ready batches makes this epoch unrepresentative), epoch 5
680–686 s. Against the 22-worker epochs 2–3 (828 / 834 s) that is ~18 % faster, not the 2× the loader-concurrency argument
promised: the steady state is bounded by BeeGFS throughput, not by the worker count. Epoch-5/6 reads: kl_z .792 → .773, kl_h
1.174 → 1.129, trace_h .343 → .368, inv .103 → .146; online K710 probe 1.56 → 2.93 % (head re-learned after the resume reset —
the callback's linear head is not in the checkpoint; fix pending Berker's word). At 680 s per epoch the remaining 234 epochs take
~44 h (landing ≈ 09-06 midday).

**B cell launch record (RAW, 2026-09-04 22:3x).** Job 64499984 started on gpu272,273 (2 × 4 H100), fresh start in its own run
directory (`wandb_resume.json` none), wandb run qyx0dhse. Launch check = the epoch-0 share line against the S cell's:
S `inv=0.057(g2.376) moment_kl=0.932(g5.365) h_moment_kl=0.011(g8.622) | omega_h=4.143 omega_z=5.451 | kl_z=2.926 kl_h=3.106 trace_z=0.032 trace_h=0.214`;
B `inv=0.069(g2.571) moment_kl=0.904(g3.461) h_moment_kl=0.027(g5.695) | omega_h=4.010 omega_z=5.454 | kl_z=3.149 kl_h=3.746 trace_z=0.034 trace_h=0.216`.
Init geometry matches (omega, trace within noise); the B doses read g 2.57 / 3.46 / 5.70 at its batch 512 (S: 2.38 / 5.37 / 8.62 at
768). No performance number of B is read before P-B1..B4 are approved.
**BeeGFS contention between cells (RAW, 2026-09-04 22:35–23:25).** With the B cell running on two more nodes (64499984, 2 × 4, 96
readers per node) the S epochs went 681, 719, 756, 825, 1,148, 1,790 s (epochs 28–33) and B did not finish its first epoch in 50 min:
both cells read the same 120k clips per epoch and their combined demand exceeds the BeeGFS servers' delivery. Berker: "cancel b and
requeue it behind s" — B cancelled 23:25 (fresh start, no checkpoint) and requeued as 64582327 behind the S resumer. Rule: one
video cell at a time on BeeGFS.

**Third launch, offline in-domain probe at matched epochs (RAW, 2026-09-05 11:20; job 64662464 on a 3090, `results/e34/k710_probe_s3c.csv`;
20 train / 8 eval clips per class = 14,468 / 5,766, seed 0, one clean center view; student weights, backbone CLS unless noted; linear / kNN k20).**

| epoch | killed second launch (batch 3,072, 39 steps/epoch) | third launch (batch 768, 156 steps/epoch) |
|---|---|---|
| 0 | 3.10 / 1.89 | 3.02 / 1.98 |
| 10 | 4.70 / 2.46 | 6.04 / 3.09 |
| 30 | 6.82 / 3.50 | 9.07 / 5.31 |
| 60 | 8.79 / 4.30 | 11.39 / 6.17 |
| 85 (second: 93) | 9.07 / 5.01 | 11.67 / 7.30 |

LeVJEPA's released ViT-L (VideoMix) on the same split reads 20.12 / 10.91 (CLS; 14.45 / 4.61 on patch-GAP). NOTE the split: the earlier
reference of 27.4 / 13.4 on the card came from the 50-per-class split (`k710_probe50.csv`, 36,031 train clips); numbers are comparable only
within one split. Other taps of the third launch at epoch 85: patch-GAP 8.79 / 3.83, z 9.43 / 6.00; EMA CLS 8.97 / 4.96 (the .9999 EMA lags
the student by ~2.7 linear points at this stage). At equal epochs the third launch leads the killed run by 1.3–2.6 linear and 0.6–1.9
kNN points from epoch 10 on; at 35 % of its schedule its ViT-S sits at 58 % (linear) / 67 % (kNN) of the reference ViT-L.

**Landing evaluation chained (2026-09-05 21:10, Berker: "before the b cell we should do the eval so that we can compare with levjepa and
vjepa").** S lands ≈ 09-06 07:45 (epoch 159 at 20:40, 491 s/epoch over the last 20). Chained `afterany` S on the same 8 × 1 H100 shape
(`slurm/attentive_probe_in1k_8x1.slurm`, new: one process per node via srun, env:// rendezvous): the full-protocol IN-1k attentive
probe (V-JEPA port: 20 epochs, full 1.28M train, 50k val, global batch 8 × 16 = 128 at lr 1e-3 — the batch note stays) on the
landing `last.ckpt`, **student weights first (64669070) then EMA (64669071)**; outputs `results/e34/attnprobe_k710s3_ep240_{student,ema}.csv`.
The B cell (64582327) now waits `afterany` the EMA probe. Evaluator fixes applied: `PYTHONPATH` exported in both slurm files; the
validation forward under DDP wrapped in `no_grad` (the peek's DDP crash). The peek (second launch ep89, 64k subset, 5 epochs) read
5.57 EMA / 8.91 student; the reference points are LeVJEPA ViT-S 39.4 and V-JEPA 2 ViT-S 38.7 (`docs/methods/levjepa.md`).

**S third launch LANDED (RAW, 2026-09-06 06:20; job 64507327, 1 d 13:43 h; 240 epochs = 37,440 optimizer steps at batch 768).** Final
online K710 probe 24.80 % at epoch 239 (epochs 200–239: 21–25 %, noisy on 1,024 clips); last share reads kl_z .57, kl_h .70, trace_z
.93, trace_h 1.02 (the H stream thickened from .30 at epoch 4). Landing checkpoint: the `last.ckpt` (= `epoch-0239.ckpt`) under
`vid.floorssl.s0.k710s3/spt_cache/runs/20260904/160312/a5a3c3bd0aeb/checkpoints/` (the chained 2 × 4 resumer 64507328 ran 5 min,
found the run complete and left an EMPTY run dir 20260906/054030 — harmless). The attentive-probe chain (student 64669070, EMA
64669071, 8 × 1) pends on Resources: other users took S's cards within minutes; 4 × 1 copies were added (student then EMA; global batch
64 instead of 128 — recorded here as the protocol note if the 4 × 1 copies are the ones that run); first to start wins, B waits for both
EMA copies.
**Attentive-probe port fix (2026-09-06 07:45; PORT_NOTES).** The port's `CrossAttention.forward` had dropped V-JEPA's output projection
(`self.proj`, whose weight the init rescales by 1/√2): under DDP the unused proj parameters made the reducer hang ("expected to have
finished reduction" — the crash of both 4 × 1 copies at their first training step, and the peek's DDP crash of 09-04, which I had
misread as a validation no_grad issue). Fixed: `proj` applied after the attention. The 09-04 peek numbers (second launch ep89: 5.57 EMA /
8.91 student) were computed WITHOUT the projection — sanity level only, not the protocol. 4 × 1 copies resubmitted (student then EMA);
the 8 × 1 chain pends on Resources and picks the fixed script up at start; B waits for both EMA copies.
**Landing attentive probe RUNNING (RAW, 2026-09-06 08:00).** The 4 × 1 student copy (64683974, gpu266/268/272/273, global batch 4 × 16 = 64
at lr 1e-3 — the protocol note: V-JEPA used 1,024) started 07:35 with the fixed port; epoch 1/20: train 14.36, **val top-1 27.04**, 21 min
per epoch → lands ≈ 14:40; the EMA copy follows (≈ 7 h), then B. The 8 × 1 chain was cancelled. Reference points: LeVJEPA ViT-S 39.4,
V-JEPA 2 ViT-S 38.7 at their protocol; ours is read at epoch 20.

**S third launch — IN-1k attentive probe at landing, STUDENT weights (RAW, 2026-09-06 14:25; job 64683974, 6 h 50 min on 4 × 1 H100).**
Epoch-20 val top-1 **47.23** (train 37.20; curve 27.0 / 31.2 / 33.7 / 35.1 / 36.8 / 38.0 / 39.3 / 39.8 / 41.3 / 41.7 / 42.6 / 43.5 / 44.3 /
45.2 / 45.9 / 46.3 / 46.8 / 47.0 / 47.2 / 47.2 over epochs 1–20; `results/e34/attnprobe_k710s3_ep240_student_4x1.csv`). Protocol notes
carried with the number: global batch 64 (4 × 16) at lr 1e-3 against V-JEPA's 1,024 (16× more head updates over the same 20 epochs — a
matched 4 × 256 rerun is the cheap closer); the port's output projection restored on 09-06 (the 09-04 peek lacked it). Reference
points at their protocol: LeVJEPA ViT-S 39.4, V-JEPA 2 ViT-S 38.7, LeVJEPA ViT-B 50.7, V-JEPA 2 ViT-B 51.6, VideoMAEv2 ViT-B 47.1
(pixel reads of their Figure 2). **P3 as pre-registered:** the claim is the +1-point improvement over the same recipe WITHOUT the H term
under the same evaluator — the z-only twin is NOT launched, so P3 proper is not yet readable; the comparison to their published ViT-S is
"reported with the caveat, never as a win" by the card's own wording. The EMA copy (64683975) runs on the same cards since 14:25.

**IN-1k attentive-probe FAIRNESS AUDIT (RAW, 2026-09-06 15:30; Berker: "verify this video run is a fair eval" — HANDOVER §0.1 item by
item, against V-JEPA's evaluator `evals/image_classification_frozen/eval.py` + `configs/evals/vitl16_in1k.yaml` from facebookresearch/jepa,
the copies fetched 2026-09-02).**
(a) **Probe batch — NOT matched.** V-JEPA: `batch_size 16` per GPU × 8 nodes × 8 tasks = 1,024 global at lr 1e-3 (`start_lr = lr`, `warmup 0`,
`final_lr 0`), wd 1e-3 → 1e-6, 20 epochs → 1,252 iterations per epoch, ≈25,040 head updates. Ours (4 × 1 H100): 4 × 16 = 64 global at the same
lr → 20,019 iterations per epoch, 400,380 updates = 16× theirs (the 8 × 1 launcher would give 128 = 8×).
(b) **Port.** Matches the donor: `AttentiveClassifier` (one complete cross-attention block, heads = encoder heads, MLP ratio 4, query
trunc-normal .02, `proj` and `fc2` rescaled by 1/√2 = their `_rescale_blocks` at layer 1; the output projection restored 09-06), AdamW groups
(bias / 1-d parameters without wd), `WarmupCosineSchedule` and `CosineWDSchedule` verbatim, schedulers stepped before the optimizer, the DDP
validation under `no_grad`. **Deviations found today, undeclared until now:** (1) V-JEPA clips the classifier's gradient to norm 1.0 at every
step (`clip_grad_norm_(classifier.parameters(), 1.0)`, eval.py lines 303 / 308); the port does not clip. (2) V-JEPA's final `Linear(embed_dim,
1000)` keeps PyTorch's default init (it sits outside the pooler's `_init_weights`); the port trunc-normal-initializes it (.02, zero bias).
(3) V-JEPA runs float16 autocast with a GradScaler (the flag is named `use_bfloat16`); the port runs bfloat16 without a scaler. (4) Their top-1
is the running mean of per-batch accuracies, ours correct / seen — equal up to the last partial batch.
(c) **Input pipeline — matched.** Train = timm `create_transform(input_size 224, is_training, auto_augment 'original', bicubic, re_prob .25
'pixel' × 1)`; val = Resize(int(224 · 256 / 224)) → CenterCrop 224 → ToTensor → ImageNet mean / std; each image repeated over the encoder's 16
frames (their forward pre-hook `unsqueeze(2).repeat(1, 1, frames, 1, 1)`); no token dropping; the encoder's full token set (CLS + 16 × 196
patches at tubelet 1) is the key / value set. V-JEPA's ViT-L/16 has tubelet 2 (8 × 196) and no CLS — architecture-side, not evaluator-side.
(d) **Weights — open.** V-JEPA evaluates `checkpoint_key: target_encoder` (its EMA target). LeVJEPA keeps an EMA "for evaluation only"
(`WeightEMA`: decay .9999 per optimizer step, applied every 32 steps); which weights their Figure 2 reports is not stated. Ours: the same
callback; with 37,440 optimizer steps the EMA's memory of the initialization is .9999^37,440 = 2.4 % and its time constant 10,000 steps
(the last ≈64 epochs) — a well-formed average. At their released batch (3,072 → ≈9,400 steps on 120k clips) the same rule would leave ≈39 %
of the initialization in the EMA, which argues their grid numbers are online-encoder reads — unverifiable. Both of ours are read; the EMA copy
runs (ep1 26.09 / ep2 30.77 vs the student's 27.04 / 31.21 at the same epochs).
(e) **References.** LeVJEPA ViT-S 39.4 and V-JEPA 2 ViT-S 38.7 (and the B / L points) are pixel reads of their Figure 2, calibrated on the
text's ViT-B 50.7 to ±0.2, read-off uncertainty ±0.3 (`docs/methods/levjepa.md`); no table of theirs prints the ViT-S value.
(f) **Pretraining.** Matched: arch (ViT-S, per-frame tokenization, block-causal attention, token drop .95), 240 epochs, the class-balanced 20 %
K710 draw (ours, seed 0, 120,100 clips after the extraction losses; their list is unreleased), their trainer and data pipeline. NOT matched, by
Berker's decisions (§third launch): batch 768 without accumulation = 37,440 optimizer steps (their released config 3,072 → ≈9,400 steps at this
data size; the grid batch is not stated), 6 global views (≈940 tokens per clip vs ≈450 for 1 global + 10 locals), our objective, gradient
clipping at 1.0, no locals.
**Verdict (RAW, Fable's read):** the evaluator is protocol-faithful in the classifier, augmentations, schedule and input; the probe's
OPTIMIZATION is not (16× the updates, unclipped, a differently initialized head) and the weights convention is open. 47.23 is not a
like-for-like number against 39.4 and stays RAW with the caveat; P3 proper (the z-only twin) is untouched by this audit.
**Proposed closer (D-118 PROPOSED; no launch, no evaluator edit without Berker's word):** rerun the student probe at their protocol — 4 × 256 =
1,024 global (`--bs 256`; the launcher hard-codes `--bs 16`, a one-line change), gradient clip 1.0 and the default head init added to the port
(two lines), `--workers 20` (the batch-64 runs are loader-bound at 8 workers per node: 254 images / s per node = 21 min per epoch; 24 CPUs per
task are allocated) → 1,252 iterations per epoch, ≈25,040 updates, estimated 3–4 h on the same 4 × 1 H100 shape (≈7 h at the current worker
count); then the EMA under the same protocol if its batch-64 read lands within ≈1 point of the student's. The batch-64 CSVs stay as what they are.

**Berker's ruling on the audit (2026-09-06 ~15:50, verbatim): "no one cares about how much batch size one used for eval. i think this is not
a confound. i was asking basically apart from that if we have a recipe issue. and for video there are multiple tasks right? why are we only
seeing in1k?"** → D-118 WITHDRAWN; the batch-64 student read 47.23 stands as the protocol read (still RAW: joint interpretation pending).
Recipe differences that remain on the PRETRAINING side against LeVJEPA's ViT-S point (all his own choices, listed for his call): (1) optimizer
steps 37,440 at batch 768 vs ≈9,360 at their released batch 3,072 (39 steps per epoch, the count our killed second launch ran) — 4×;
(2) tokens per clip per step ≈941 (6 global views × 157 retained) vs ≈447 (1 global + 10 locals at 96²: 157 + 10 × 29) — ≈2.1× the encoder
FLOPs per epoch [CORRECTED 17:20 below: the grid uses V = 4 locals → 273 tokens, ≈3.5×], i.e. ≈2.7 EFLOP against their ≈1.3 for ViT-S on their Figure 2 axis [corrected: ≈4.5 EFLOP]; (3) gradient clipping at 1.0 (they clip nothing);
(4) our own 20 % draw (120,100 clips after the extraction losses; their list is unreleased); (5) the objective itself = the treatment.
Unchanged: arch, token drop .95, 240 epochs, lr 4e-4 with warm-up at 12 % of the steps then flat, wd .04, EMA rule, their loader and trainer.
Eval-side nothing else differs but trivia (the port's missing probe clip and head init — declared in PORT_NOTES, left as is). Weights: the EMA
read runs (ep2 30.77 vs the student's 31.21). **Tasks:** LeVJEPA's epoch-matched grid (their Figure 2) reports ImageNet-1k only; SSv2
appears for ViT-B (30.4 at 240 epochs, Tables 1 / 4) and, with K400 (linear probe on mean-pooled tokens), only in the FLOP-matched ViT-B row
(1,085 epochs: 61.0 / 40.4 / 44.6). Our side has only the IN-1k evaluator; SSv2 (220,847 webm + labels) and K400 (train / val tarballs +
annotations) sit raw on BeeGFS, neither encoded into a clip store nor evaluated — the K400 linear probe and the SSv2 attentive probe are
the owed evaluators (HANDOVER §6).

## SSv2 and K400 evaluators — the V-JEPA video protocol ported to clip files (PROPOSED readings; Berker 2026-09-06: "we need to do ssv2 and k400 as they did. it is fine if you dont share the performance (check if they shared checkpoints), we should be reporting those numbers anyway, then we go to the base model training similarly to our s.")

**What they did.** LeVJEPA App. C: the attentive probe "follows Bardes et al. without modification"; Kinetics-400 = "the output tokens of the
frozen encoder are averaged into a single vector and a linear classifier is trained on the pooled representation", hyperparameters from
V-JEPA. V-JEPA's `evals/video_classification_frozen/eval.py` + `configs/evals/vitl16_{ssv2_16x2x3,k400_16x8x3}.yaml` (commit 51c59d5, fetched
2026-09-06, copies in the session scratchpad): 16 frames at frame_step 4 (native rate), SSv2 2 segments / K400 8 segments, 3 spatial views at
eval (short side 224, crops along the long side), `attend_across_segments` (all segments' tokens concatenated before the probe), training on
every segment with one spatial view; train augmentation = video RandAugment `rand-m7-n4-mstd0.5-inc1` (same ops on all frames of a clip),
random-resized-crop scale (.08, 1) ratio (3/4, 4/3), no flip, RandomErasing p .25 as a cube; AdamW lr 1e-3 → 0 cosine, wd .01 → 1e-6,
grad clip 1.0, 20 epochs, batch 4 × 64 GPUs = 256, fp16 + GradScaler; loss = CE averaged over views, prediction = mean softmax over views.
**Shared checkpoints:** LeVJEPA released ONE (galilai-group/LeVJEPA-VideoMix-Large: ViT-L/16, 100 epochs on VideoMix 1.8 M clips) with published
frozen-probe reads 69.5 IN-1k / 55.0 SSv2 — the validation targets for our IN-1k and SSv2 ports; no S / B checkpoint, no K400 number for it.
V-JEPA 2 and VideoMAEv2 publish no 20 %-K710 S / B checkpoints either. Their grid (Figure 2) reports IN-1k only; SSv2 exists at ViT-B (30.4,
240 epochs) and SSv2 + K400 in the FLOP-matched ViT-B row (1,085 epochs: 40.4 / 44.6) — so the S reads are reported without an external S
reference, as Berker said.

**Port (`video/levjepa/scripts/video_probe.py`, docstring = the protocol; `slurm/video_probe_nx1.slurm`).** Verbatim: the frame-index logic
(`loadvideo_decord`, allow_clip_overlap, random windows at train AND val — for our stores every case falls in the overlap branch, so val is
deterministic), the eval resize + 3-crop (cv2 bilinear), the train transform (timm's RandAugment op primitives with V-JEPA's level functions,
one draw of ops and magnitudes per clip; one crop box per segment; cube erasing), the aggregation, the heads, the schedules, the clip, the
loss and the view-averaged prediction. **Declared readings (mine, for veto):** (1) frame_step in STORED frames — SSv2 store at the native 12 fps →
4; K400 store at 15 fps (their 4 at 30 fps ≈ our 2) → 2; (2) our encoder's CLS token stays in every segment's token set (V-JEPA has none);
(3) the K400 head = mean over ALL concatenated tokens of the 8 segments → Linear (their App. C sentence read literally; init = PyTorch default);
(4) bf16 without a scaler; top-1 = correct / seen; (5) batch per GPU = what fits (Berker: not a confound), reported with every number.
**Stores (`scripts/build_clipfiles.py`, launchers `slurm/build_{ssv2,k400}_clipfiles.slurm`):** the K710 clip-file format; SSv2 = the 220,847
Qualcomm webm (train 168,913 / validation 24,777; 174 classes) at native 12 fps / 240p, all frames (their sampler pads short clips);
K400 = the CVDF tarballs (train 485 parts, val 21; annotations 246,534 / 19,906 rows) streamed once, members decoded from memory, stored at
15 fps / short edge 256 (V-JEPA's probe resizes the short side to 224 anyway; half the bytes of a 384 store; ≈2.6 MB per 10-s clip ≈ 0.65 TB).
Class dirs = the label text; `classes.txt` = the index space. **Estimates (RAW, before the smokes):** SSv2 build ≈ 1 h on 8 CPU shards; K400 build
≈ 2–3 h on 20 shards; SSv2 attentive probe on ViT-S ≈ 7 h on 4 × 1 H100 (BeeGFS whole-file reads, ~35 / s per node, bound it), K400 linear probe
≈ 10 h on 4 × 1 (5 h on 8 × 1) — the K400 probe reads ≈ 0.5 TB per epoch, the same load as a training cell on BeeGFS.
**Sequencing (Berker's call):** the B cell (64582327) starts by itself after the EMA probe (~21:30) and would share BeeGFS with these probes;
the S reads first and then B (his sentence order) means holding B ≈ 2 days; letting B start means running the S probes alongside it (SSv2 light,
K400 heavy) or staging each rank's shard of the eval store to the H100 nodes' local NVMe (SSv2 ≈ 140 GB, K400 ≈ 650 GB / 8 ≈ 80 GB per node).
**Owed before any number:** the CPU smoke (synthetic store, vit_tiny) and the GPU smoke (the S checkpoint on a 72-clip SSv2 subset), then the
full stores, then the runs on Berker's word; port validation on the released ViT-L (SSv2 → 55.0; IN-1k → 69.5) is the one use of their checkpoint.

**Rulings on the plan (Berker 2026-09-06 ~17:00 verbatim: "two day hold is too much. i was expecting this eval runs to be faster. lets do vit b after ema probe. (also i do not want ema separate than student from now on, i expect ema to be superior) ... we wont be running their checkpoints unless necessary (for now it is not)").** Applied: (1) the B cell keeps its chained start after the EMA probe; the S probes run alongside it — the
probe's data path now stages each rank's fixed shard of the store to the node's local disk once at start (`--stage-local`; BeeGFS's ~35
whole-file reads per second per node had bounded the SSv2 epoch at 20 min on 4 nodes; local reads bound it by decode + augmentation instead,
est. SSv2 ≈ 25 min copy + 20 × 4 min ≈ 1.7 h on 4 × 1 H100, K400 ≈ 15 min copy + 20 × 12 min ≈ 4.3 h on 8 × 1), and the probes never touch
BeeGFS while B trains; (2) EMA weights only from now on — every probe reads the EMA encoder (V-JEPA's own convention), the video table shows
one Ours row per cell (EMA) with IN-1k / SSv2 / K400 columns, the student IN-1k read (47.23) stays on this card as a record; (3) no run of
LeVJEPA's released ViT-L. Answer to "69.5 vs 57.5": 57.5 is their ViT-L trained 240 epochs on the 20 % K710 subsample (Figure 2, the epoch-
matched grid the table reports); 69.5 is their released ViT-L VideoMix checkpoint, 100 epochs on the combined 1.8 M-clip corpus (K710 + SSv2 +
Walking Tours + PE-Video, their §5.4) — a different pretraining set, not a grid row.

**Builder + probe smokes PASSED; the full stores are building (RAW, 2026-09-06 16:20–16:35).** CPU smoke 64696448: the synthetic-store probe ran both
heads (vit_tiny, chance-level as expected); the SSv2 builder wrote 72 clips (e.g. 36 frames / 850 KB at 427 × 240, 47 frames / 487 KB at 320 × 240 —
SSv2 sources come in both widths, native kept) and the K400 builder 12 clips from the tar stream (150 frames each, 2.6–4.5 MB at 341–452 × 256 →
the K400 store is ≈ 0.7–1.1 TB, not the 0.65 TB estimated). GPU smoke 64696449 (the S checkpoint on the 72-clip subset) running. Full builds
launched 16:33 on defaultp: SSv2 train 64697450 / validation 64697451 (8 shards each), K400 train 64697452 (20 shards) / val 64697453 (7 shards).
**GPU smoke PASSED (RAW, 2026-09-06 16:37; job 64696449, one A40, the S landing checkpoint, student weights on the 72-clip SSv2 subset):** SSv2
attentive probe 2 epochs (48 train / 24 val, 174 classes, 12 s per epoch with 6 workers on BeeGFS), K400-style sampling with the linear-mean head
(8 segments × 16 frames, 3 views at val) 1 epoch — both heads run end to end on CUDA; accuracies at chance on 24 clips as expected. The S probes
are CHAINED behind the store builds (EMA weights, per-rank shards staged to node-local disk): SSv2 attentive = job 64697512 on 4 × 1 H100
(bs 32 per GPU, `results/e34/videoprobe_ssv2_attentive_k710s3_ep240_ema_4x1.csv`), K400 linear-mean = job 64697513 on 8 × 1 H100 (bs 8 per
GPU, `results/e34/videoprobe_k400_linear_mean_k710s3_ep240_ema_8x1.csv`); both `afterok` their stores' arrays. Predictions: none registered against
external numbers (no S reference exists); the table rows fill from the CSVs (epoch-20 finals).

**CORRECTION + FLOP matching (2026-09-06 17:20, Berker: "1085 epochs is what make levjepa useful with 10 local crops on the table with vitb. what would
the nof epochs correspond if we try to match flops? with our 6 global views").** The paper "retain[s] V = 4 for all comparisons" (its §4 ablation; the
FLOP-matched Table 3 row is the exception at V = 10, 1,085 epochs), so the epoch-matched grid costs 157 + 4 × 29 = 273 retained tokens per clip per
step, not the 447 I used above. Encoder FLOPs scale with retained tokens (attention adds ≈7 % per global view at ViT-S, ≈1 % per local; forward +
backward the same factor for both recipes): ours = 6 × 157 = 942 tokens → **≈3.5× the grid's FLOPs per epoch** (not 2.1×). Equivalences for our
6-global-view recipe (S or B, same token count): **≈70 epochs** match their 240-epoch grid budget; **≈515 epochs** match the FLOP-matched budget
(= the baselines' 240 ViT-B epochs = LeVJEPA's 1,085 epochs at V = 10: 1,085 × 447 / 942). Our S-240 (and the B cell at 240) therefore sits at
≈3.5× the grid's cost (≈4.5 EFLOP on their Figure 2 axis for S, about their ViT-B grid point's 4.6–4.8) and at ≈47 % of the FLOP-matched budget.

**B cell → 515 epochs, chained (Berker 2026-09-06 17:30: "to build their table 3 we need 515 epochs right?" → "yes lets do that update (and do a chain
in case it is necessary)").** The queued 2 × 4 job 64582327 keeps its captured 240-epoch script (a pending job's script cannot be edited); the schedule
is warm-up then flat, so extension = resume with a larger max_epochs. Chain submitted: 64697841 → 64697842 (`slurm/e34cf3_shape_h100x8x1_vitb.slurm`,
`afterany` the previous; MAX_EPOCHS 515; every epoch's checkpoint is kept, so epoch-0239.ckpt remains the 240-epoch landing state for the grid rows).
The extensions run on the 8 × 1 H100 shape: BeeGFS serves ~25–35 clip reads per second PER NODE (S measured 200 clips/s over eight nodes = 600 s per
epoch), so the 2 × 4 shape would take ~30 min per epoch (12 days for 515) against ~10 min on eight nodes (3.6 days). Berker: "i am aware of beegfs
issue but is there anything we can do about it? like to move it in group shared etc?" — the data already lives in the group's BeeGFS dir; NFS is slower;
the lever is the per-node request cap, so: (a) more BeeGFS clients (8 × 1 instead of 2 × 4, done for the chain); (b) node-local staging — each rank
copies its fixed shard of the clip files (≈15k files, ≈69 GB) to the node's NVMe once (~7 min), the epochs then read local disk and are bound by
decode + augmentation (est. 2–3 min per epoch → 515 epochs ≈ 1 day), and the training never touches BeeGFS while the probes run; implemented
2026-09-06 in `data/clipfile_loader.py` (`shard`, `stage_dir`, `stage_limit`) + `data/loader.py` + `main.py` (`+data.stage_local=<dir>
+trainer.use_distributed_sampler=false`, off by default; a fixed per-rank slice of the sorted file list replaces the per-epoch DistributedSampler
split — same expectation); smoke job = stage_smoke (two ranks × 96 clips × 3 steps on one node); (c) the H100 nodes' 21 TB local disks need
permissions (admin). The B extensions take the staging flag (STAGE_LOCAL=1) once the smoke passes.

**B relaunched on 8 × 1 with node-local staging and the new checkpoint retention (Berker 2026-09-06 ~18:00: "so we should do 8x1 launch and also
instead of keeping every checkpoint we should specifically keep 239, last, best. node local staging making 2-3 min per epoch is amazing. why dont
we do that?").** Cancelled: the 2 × 4 base 64582327 and the first extension chain 64697841 / 64697842. New chain 64698365 → 64698366 → 64698367
(`slurm/e34cf3_shape_h100x8x1_vitb.slurm`, STAGE_LOCAL=1, MAX_EPOCHS=515; the base waits `afterok` the combined smoke 64698364 AND `afterany` the
EMA IN-1k probe 64683975). Retention (`checkpoint.save_top_k=0 checkpoint.save_last=true +keep.epochs=[239] +keep.best=true`): ModelCheckpoint keeps
the rolling `last.ckpt` only; the new `CheckpointKeeper` (sslgap_reg.py) writes `epoch-0239.ckpt` at the end of epoch index 239 (= the 240-epoch
state for the grid rows) and `best.ckpt` + `best.json` whenever the online K710 probe improves (its best value travels in the checkpoint state,
so resumed segments continue the comparison); the K710Probe now scores at epoch END (the same head state as the next epoch's start) so `best`
is the state that produced the read, and its line reads `[probe] epN` with N = epochs completed. Node-local hygiene in the launcher: stale
`lev_*` / `vp_*` staging dirs of jobs no longer in the queue are removed before staging. The 2–3 min per epoch is an ESTIMATE (decode + GPU
bound; 48 workers per node); the first staged epoch is the measurement. Smoke 64698364 (one node, two ranks × 96 staged clips, 2 epochs × 3
steps): staging, the epoch-end probe, `epoch-0000.ckpt`, `best.ckpt` / `best.json` and `last.ckpt`.
**Combined smoke PASSED (RAW, 2026-09-06 16:51; job 64698364, one gpu-partition node, two ranks):** each rank staged its 96-clip shard (0.6 GB) to
`/tmp/lev_smoke_<job>` (the copy ran once per loader — train and val both build the dataset; the second pass found the files present),
trained 2 × 3 steps at batch 16 × 2, the epoch-end probe printed `[probe] ep1` / `ep2` (0.00 / 3.12 on 32 held-out clips, chance-level as
expected), and the checkpoints dir holds exactly `epoch-0000.ckpt`, `best.ckpt` + `best.json` ({"epoch": 1, "epochs_completed": 2,
"k710_top1": 3.125}) and `last.ckpt` — plus a `last-v1.ckpt` written at fit end by the donor's checkpoint manager (a versioned duplicate of
last; the launcher resumes the exact name `last.ckpt`, unaffected). The B base 64698365 is therefore gated only on the EMA probe's end.
**Evaluation stores COMPLETE (RAW, 2026-09-06 18:2x; arrays 64697450–53, all 43 shards COMPLETED).** SSv2 `datasets/ssv2/ssv2_clipfiles`: train
168,913 / validation 24,777 clips (every labeled video; 174 classes; native 12 fps and size), built in ≈6 min per train shard. K400
`datasets/kinetics/k400_eval_clipfiles`: train 238,680 of the 246,534 annotated clips (2,578 dropped by the builder — undecodable or fewer than 32
stored frames — and 5,276 absent from the CVDF mirror's tarballs), val 19,791 of 19,906 (90 dropped, 25 absent); 400 classes; 15 fps, short edge 256;
≈24 min per train shard. The S probes 64697512 (SSv2) / 64697513 (K400) now pend on the H100 pool (Priority).

**S third launch — IN-1k attentive probe at landing, EMA weights (RAW, 2026-09-06 21:14; job 64683975, 6 h 49 min on 4 × 1 H100, global batch 64).**
Epoch-20 val top-1 **46.76** (train 37.07; curve 26.1 / 30.8 / 33.0 / 34.7 / 36.1 / 37.3 / 38.3 / 39.7 / 40.7 / 41.4 / 42.2 / 43.2 / 43.9 / 44.6 /
45.4 / 45.8 / 46.3 / 46.6 / 46.8 / 46.8 over epochs 1–20; `results/e34/attnprobe_k710s3_ep240_ema_4x1.csv`). Against the student's 47.23 under the
identical protocol the EMA reads 0.47 lower, and it trailed the student at every epoch (by 0.4–1.0). Berker's expectation (17:00: "i expect ema
to be superior") is NOT met on this read — RAW, joint read pending; the table carries the EMA row per his ruling (tab:video Ours ViT-S 46.8).
The EMA rule here (.9999 per optimizer step applied every 32; 37,440 steps; the last ≈64 epochs weighted) is the donor's; the probe's own noise
across runs is unknown (one run each).

**B cell RUNNING (RAW, 2026-09-07 00:03; the 8 × 1 shape 64747615 won on gpu[265-266,268-270,273,275-276]; the 4 × 2 / 2 × 4 shapes cancelled by the
first-wins job; successors 64768378 → 64768379 chained; wandb qyx0dhse continued).** The launcher resumed the cell's own `last.ckpt` of the 2026-09-04
attempt (2 × 4 shape, BeeGFS reads; that job completed epoch 0 before its cancel — `epoch-0000.ckpt` of 23:35), so epoch 1 onward runs on the staged
shape: staging 15,012 clips = 78 GB per rank in 9–10 min (two loaders build the dataset; the second pass finds the files), epoch 1 in ≈11 min
including the probe's held-out draw, then **0.7 optimizer steps per second = 234 steps in ≈5.6 min per epoch** (my estimate was 2–3 min; the bound
is JPEG decode + augmentation with 24 workers per rank, 358 clips/s over eight ranks) — against ≈10 min on 8 × 1 over BeeGFS and ≈33 min on 2 × 4.
513 epochs remain → ≈48 h of training; the 2-day wall puts the boundary near epoch 250–300, the successor re-stages (10 min) and continues. New
checkpoint layout confirmed on disk: `best.ckpt` + `best.json` + `last.ckpt` only. Epoch-2 share line: inv .242 (g 4.5), moment_kl .750, kl_z 2.22,
kl_h 1.22, trace_z .48, trace_h .28, omega_h 3.39 — the ep1–5 formation gate is read on the P-B1 terms (no quench: kl_z ≫ .1).
**The two S probes failed at their first start (00:06) and were resubmitted at 00:3x:** the SSv2 probe (64697512, 20 workers, 120G per node) was
host-OOM-killed after staging (persistent train AND val workers each hold prefetched multi-view batches); the K400 probe (64697513) lost rank 0 to a
port collision (both probes started together with gpu265 as master and the same fixed port) and its other ranks hung — cancelled. Fixes in
`slurm/video_probe_nx1.slurm`: MASTER_PORT = 29000 + job id mod 1000, 12 workers (NUM_WORKERS), mem 400G default (250G on this resubmission), stale
`vp_*` / `lev_*` staging dirs removed before staging. Staging measured: SSv2 42k files / 37 GB per rank in 9.7 min; K400 29.8k files / 92 GB per
rank in 8.5 min.
**B cell, epochs 1–15 (RAW, 2026-09-07 01:30; job 64747615; per-epoch time ≈5.3 min from the checkpoint stamps).** Online K710 probe (held-out 1,024 clips,
linear on detached h, scored at epoch end): ep2 4.10 · ep3 6.35 · ep4 7.81 · ep5 9.96 · ep10 16.31 · ep15 21.29 (the S cell read 12.11 at ep30 and 24.80
at ep239 on the same instrument). Share lines: inv share .22 → .40 → .31 (ep1 / 10 / 15), moment_kl .75 → .60 → .69, h_moment_kl .029 → .003 → .004;
kl_z 2.25 → 2.21 flat (no quench), kl_h 3.33 → .69 (ep10), trace_z .37 → .71, trace_h .21 → .52 (ep10), omega_h 6.2 → 1.18 (ep10). Recorded for the
P-B1..B4 read with Berker; nothing interpreted.

**S third launch — SSv2 attentive probe at landing, EMA weights (RAW, 2026-09-07 13:07; job 64769203, 5 h 38 min on 4 × 1 H100 with staged
shards, per-GPU batch 32 = 128 global, 12 workers, 16.3 min per epoch).** Epoch-20 val top-1 **37.31** (best 37.39 at ep18; train 57.11; curve
ep3 29.31 · ep7 32.15 · ep11 35.50 · ep14 36.56 · ep18 37.39; `results/e34/videoprobe_ssv2_attentive_k710s3_ep240_ema_4x1.csv`). Protocol: V-JEPA's
SSv2 frozen attentive probe as ported (16 × 2 × 3, attend across segments, RandAugment / RRC / cube erasing, AdamW 1e-3 cosine, wd .01 → 1e-6, clip 1.0,
20 epochs). References at their protocol: no ViT-S point published; LeVJEPA ViT-B 30.4 (240 epochs, V = 4); FLOP-matched ViT-B rows LeVJEPA 40.4 /
V-JEPA 2 42.5 / VideoMAEv2 43.6 (1,085-epoch-equivalent budgets). In tab:video (Ours ViT-S SSv2 = 37.3). Joint read pending.
**B cell reached epoch 240 (RAW, 2026-09-07 18:21).** `epoch-0239.ckpt` written by the keeper (the 240-epoch state for the grid rows; the run continues to 515 —
epoch 245 at 18:43, ≈4.5 min per epoch). Online K710 probe at ep240 64.06 (ep245 65.33; best.ckpt = ep242 at 66.80); share line ep240: inv .286 (g .65),
moment_kl .712, h_moment_kl .002, kl_z 2.22, kl_h .36, trace_z .90, trace_h 1.00, omega_h .32. The three probes for the tab:video B row (IN-1k attentive,
SSv2 attentive, K400 linear-mean; EMA weights) run on this checkpoint on Berker's word; P-B1..B4 unread until then.

**S third launch — K400 linear probe at landing, EMA weights (RAW, 2026-09-07 23:53; job 64769204, 16 h 31 min on 8 × 1 H100 with staged shards, per-GPU
batch 8 = 64 global, 12 workers, ≈45–50 min per epoch).** Epoch-20 val top-1 **29.53** (best 29.58 at ep18; train 32.36; curve ep1 18.20 · ep5 25.70 ·
ep10 28.26 · ep15 29.34; `results/e34/videoprobe_k400_linear_mean_k710s3_ep240_ema_8x1.csv`). Protocol: LeVJEPA's K400 read (App. C) = a linear classifier on
the mean of all output tokens of the 8 segments, V-JEPA's K400 frozen-eval sampling (16 × 8 × 3, frame step 2 on the 15-fps store) and optimization (AdamW 1e-3
cosine, wd .01 → 1e-6, clip 1.0, 20 epochs); train set = 238,680 of the 246,534 annotated clips (the rest absent from the mirror or too short), val 19,791 of
19,906. References: no ViT-S or 240-epoch point published; the FLOP-matched ViT-B row reads LeVJEPA 44.6 / V-JEPA 2 40.7 / VideoMAEv2 37.4. The tab:video S row
is complete: IN-1k 46.8 / SSv2 37.3 / K400 29.5 (EMA). Joint read pending.

### B-row evaluation chain (2026-09-08 07:48; Berker: "you can launch video evals (3 evals) for our video job as chains so that they can immediately start after landing")
The three frozen probes of the B cell's landing checkpoint (`last.ckpt` at epoch 515, EMA weights) are queued behind the training chain:
`afterok:64747615?afterok:64768378?afterok:64768379` (any segment completing the run satisfies it; a failed segment hands over to the next).
The chain's insurance segments 64768378 → 64768379 were re-wired from `singleton` to `afternotok` (each starts only if its predecessor fails),
so a clean landing does not spend 8 cards on two no-op resumes (each would stage 78 GB per node before exiting) ahead of the probes.
| job | probe | shape | per-GPU batch (global) | wall-time ask | out |
|---|---|---|---|---|---|
| 64926335 | IN-1k attentive (V-JEPA protocol) | 8×1 H100 | 16 (128) | 36 h | `results/e34/attnprobe_k710b_ep515_ema_8x1.csv` |
| 64926336 | SSv2 attentive (16×2×3, D-119) | 8×1 H100 | 32 (256) | 48 h | `results/e34/videoprobe_ssv2_attentive_k710b_ep515_ema_8x1.csv` |
| 64926337 | K400 linear on mean-pooled tokens (16×8×3, D-119) | 8×1 H100 | 8 (64) | 72 h | `results/e34/videoprobe_k400_linear_mean_k710b_ep515_ema_8x1.csv` |
Protocol note: the S row ran the IN-1k probe at 4×1 (global 64) and SSv2 at 4×1 (global 128); the B row uses 8×1 (global 128 / 256) for speed
(Berker 2026-09-06: eval batch size is not a confound; lr 1e-3 kept in both). S wall-times: IN-1k 6 h 49 (4×1), SSv2 5 h 38 (4×1), K400 16 h 31 (8×1);
the B backbone is ≈3.8× the FLOPs, hence the wall-time asks. `attentive_probe_in1k_8x1.slurm` now takes its rendezvous port from the job id
(as `video_probe_nx1.slurm` does). The table's Ours-B row carries epoch 515 (`paper_exhibits.video_table`); it stays `---` until the csvs land, RAW after.

### B cell landed (2026-09-08 14:38): 515 epochs, job 64747615 COMPLETED rc 0 (1 d 14 h 36 min for epochs 2–515 at 4.4 min/epoch, 8×1 H100, node-local staging)
Final online K710 probe (held-out 1024, one clean-ish view, linear on detached h) 73.24 at ep515; kl_z 2.22 / kl_h .33 flat through the last 100 epochs (no quench-gate event).
Checkpoints under `…/vid.floorssl.s0.k710b/spt_cache/runs/20260907/001351/2192b1a0560d/checkpoints/`: `epoch-0239.ckpt` (09-07 18:21), `best.ckpt` (+ `best.json`, by the
online probe, 13:44), `last.ckpt` and `last-v1.ckpt` (both 14:38; the landing checkpoint verified below before the evals read it). The insurance segments
64768378 / 64768379 never started (DependencyNeverSatisfied after the clean landing) and hold no resources. The three evals (64926335 / 36 / 37) became
eligible at 14:38 and are queued on H100 resources; numbers RAW on landing.
Landing checkpoint verified 14:45: `last.ckpt` (Lightning epoch counter 514, global step 120,510 = 515 × 234) and `last-v1.ckpt` (counter 515, same global step; the fit-end save) hold identical student and EMA tensors (0 of 158 / 149 differ), so the evals' `last.ckpt` glob reads the epoch-515 weights. `best.ckpt` = epoch 502 by the online probe.

**Eval shape re-derived (2026-09-08 15:0x; Berker: "gpu100 says there're 23 free h100s are we sure? maybe a shape constraint (that might be unnecessary
as we solved the reading issue?) is the reason we are not getting the resources").** He was right on both counts. At 14:58 the partition had 28 free
cards, but among the nodes our asks may use (13 minus the excluded gpu271/274/277 and the draining gpu268) the free cards sat on five nodes
(gpu265 1, gpu269 3, gpu270 1, gpu272 1, gpu273 4), and the evals asked for one card on each of EIGHT distinct nodes — unsatisfiable, whatever the
total. The one-card-per-node shape only ever served the BeeGFS per-node read cap during the epochs; with per-rank node-local staging the epochs
read local disk, so the layout is free. Both launchers (`video_probe_nx1.slurm`, `attentive_probe_in1k_8x1.slurm`, names kept) now ask for
8 ranks on any 1–8 nodes (`--ntasks=8 --nodes=1-8 --gpus-per-task=1 --gpu-bind=none`, memory per GPU: 80 G attentive IN-1k, 250 G SSv2, 300 G K400 —
the S probes' measured peak host RSS per rank were 55 / 195 / 248 GB; staging directory per job per node, removed per node after every rank on it
finishes; the master port from the job id). Resubmitted 15:03 — 65002280 IN-1k attentive, 65002281 SSv2, 65002282 K400 — and the 8-distinct-node
asks 64926335 / 36 / 37 cancelled (never started). 65002280 started within a minute on gpu[269-270,273] (3 nodes); the other two pend on
priority / topology and start as cards free up. Costs of the free layout: staging time scales with 1/nodes (the per-node read cap; K400's 258k
files ≈ 15 min on 8 nodes, ≈ 40 min on 3) and local disk per node grows with ranks per node (K400 ≈ 81 GB per rank). Out files renamed
`*_k710b_ep515_ema_8gpu.csv` (8 ranks, layout-free) — the table glob is unchanged.
**The layout-free ask, second wall (2026-09-08 15:04–15:20): per-task card masking breaks NCCL.** The first layout-free launches used `--ntasks=8
--gpus-per-task=1`. On the H100 nodes that masks one card per task (even with `--gpu-bind=none`), and NCCL 2.28 then fails between ranks of one
node — `transport/p2p.cc:275` / `transport/shm.cc:590 Cuda failure 101 'invalid device ordinal'` — because a rank cannot open its peers' cards
(65002280: 'invalid device ordinal' on LOCAL_RANK ≥ 1; 65003541 / 65003542 / 65004812: NCCL init failure with LOCAL_RANK = 0). Two-rank
smokes on one node (`smoke_videoprobe/ddp_smoke.py`, general GPU partition, 15:12):
| ask | cards visible per task | NCCL all-reduce |
|---|---|---|
| `--ntasks=2 --gpus-per-task=1 --gpu-bind=none` | 2 (unmasked there) | OK |
| `--ntasks=2 --gpus-per-task=1 --gpu-bind=per_task:1` | 1 | FAIL (shm.cc:590) |
| `--gpus=2 --ntasks-per-gpu=1` | 1 (implicit binding) | FAIL (shm.cc:590) |
| `--gpus=2 --ntasks-per-gpu=1 --gpu-bind=none` | 2 | OK |
| `--gpus=2 --ntasks=2` | 2 | OK |
| `--ntasks=2 --gres=gpu:2` (the training shapes) | 2 | OK |
The H100 nodes behave differently from the general partition (smokes 15:16–15:18, `smkH*` logs): `--ntasks-per-gpu=1 --gpu-bind=none` gave both
tasks of a node the SAME single card (NCCL 'Duplicate GPU detected'; the v5 evals 65008316–18 died on it), and `--gpus-per-task=1 --gpu-bind=none`
across gpu[269-270,273] exposed the wrong card set per node (gpu273: one card for four tasks; gpu269: two for three). The plain job-level ask
`--gpus=8 --ntasks=8 --nodes=1-8` (no per-task / per-gpu task option, no bind option) placed 3 / 1 / 4 tasks on 3 / 1 / 4 cards with every card of
the node visible to its tasks, and all eight ranks all-reduced (65009354). Chosen and written into both launchers (LOCAL_RANK = SLURM_LOCALID);
the evals resubmitted 15:2x as v6 (ids in the HANDOVER; v2–v5 cancelled or dead, see `build_ids.txt`).

**Third wall, and the re-derivation that removes all three (2026-09-08 15:19–15:55): SLURM task placement does not follow card placement.**
The v6 ask (`--gpus=8 --ntasks=8 --nodes=1-8`) failed on its first real launch, 8 s in. SLURM spreads tasks by CPU availability and cards by card
availability, and the two need not agree: on gpu[269-270,273] the eval got 4 / 1 / 3 tasks against 2 / 1 / 4 cards, so LOCAL_RANK 2 and 3 on gpu269
indexed cards that were not there — `invalid device ordinal` at `set_device` (65009767 / 65009768 / 65009769, all three; logs in `outputs/`). The
2-CPU smoke 65009354 had aligned 3 / 1 / 4 on 3 / 1 / 4 by luck, which is why the ask looked good. No SLURM option makes a flexible-node task layout
follow the cards except `--ntasks-per-gpu` / `--gpus-per-task`, and those are the second wall (per-task masking, above).
**Re-derivation.** The flexible node count was introduced this afternoon to dodge a scheduling problem, and it brought both walls with it. A DDP job
is a uniform shape; asking for it as one removes both walls at once: `--nodes=N --ntasks-per-node=k --gres=gpu:k` makes the task count equal the card
count on EVERY node by construction, and a job-level per-node gres leaves all of a node's cards visible to all of its tasks, so
`LOCAL_RANK = SLURM_LOCALID` indexes a real device and NCCL 2.28 keeps its intra-node transports. It is the shape every training cell in this project
already runs (the 8×1 B cell, the 2×4 resumer) and the only ask that passed on both node classes (`--ntasks=2 --gres=gpu:2`, smoke 65006463). Both
launchers now carry `--nodes=4 --ntasks-per-node=2 --gres=gpu:2` = 8 ranks (names kept; any uniform N×k with N·k = 8 may be given on the sbatch line).
Memory is per node again — 200 G attentive / 500 G SSv2 / 620 G K400, against the S runs' measured per-rank peaks 55 / 195 / 248 GB at 2 ranks per
node; K400 stages 2 × 81 GB per node of the ~675 GB free local disk. 4×2 was chosen over 2×4 and 8×1 because it needs neither eight distinct nodes
(the 14:58 blocker) nor ~1 TB of memory on a single node.
| job | probe | shape | per-GPU batch (global) | wall-time ask | out |
|---|---|---|---|---|---|
| 65024351 | IN-1k attentive (V-JEPA protocol) | 4×2 H100 | 16 (128) | 36 h | `results/e34/attnprobe_k710b_ep515_ema_8gpu.csv` |
| 65024352 | SSv2 attentive (16×2×3, D-119) | 4×2 H100 | 32 (256) | 48 h | `results/e34/videoprobe_ssv2_attentive_k710b_ep515_ema_8gpu.csv` |
| 65024353 | K400 linear on mean-pooled tokens (16×8×3, D-119) | 4×2 H100 | 8 (64) | 72 h | `results/e34/videoprobe_k400_linear_mean_k710b_ep515_ema_8gpu.csv` |
Submitted 15:52. The IN-1k probe holds the backfill reservation (`Resources`); the other two pend behind it on priority. Free cards on the nine nodes
our asks may use were 7 at submit time (gpu266 1, gpu270 1, gpu272 1, gpu273 4), so no 8-rank shape of any layout places immediately — the remaining
wait is capacity, not layout. Numbers RAW on landing; the video table's Ours-B row auto-fills from the csvs.
**gpu277 is broken, and its idle cards are not capacity (2026-09-08 16:01).** SLURM reports gpu277 IDLE with all eight H100s free and no drain
reason, which is why the partition looks emptier than it is. Eight ranks on it (smoke 65024570, the uniform `--nodes=1 --ntasks-per-node=8
--gres=gpu:8` ask) all died in CUDA init: `Error 802: system not yet initialized` on every rank — the node's GPU stack, not our ask. It stays on the
`--exclude` line of both eval launchers together with gpu274 (3–9× slow reads, 09-03) and gpu271 (prolog failures, 09-04); gpu268 is drained by SLURM
itself (`Reason=Prolog error`, 09-07 19:50). Usable nodes for an 8-rank ask are therefore nine, not thirteen.

## B ladder — protocol reads and landing records (RAW; 2026-09-08 → 09-12, appended 2026-09-12 evening from the CSVs, the job accounting and the logs — the card had no entry after 09-08 16:01)

**Cell `vid.floorssl.s0.k710b`, ViT-B/16, batch 512 = 234 optimizer steps per epoch.** Checkpoint states (student + EMA in each): 240 epochs
= 56,160 steps (`epoch-0239.ckpt`) and 515 epochs = 120,510 steps (`epoch-0515.ckpt`, the landing `last.ckpt` of job 64747615) under
`…/spt_cache/runs/20260907/001351/2192b1a0560d/checkpoints/`. **Continuation to 1,085 epochs** (= LeVJEPA's Table 3 epoch count): job
65158904 started 2026-09-09 20:06 on gpu[266,268,270,272] (8 ranks, node-local staging), resumed that `last.ckpt` with `max_epochs 1085`,
landed 2026-09-11 08:08:43 (1 d 12 h 03 min for epochs 516–1085 = 3.8 min per epoch; the successor 65158905 found the run complete after
18 min). Landing state: `last.ckpt` = `last-v1.ckpt` = 1,085 epochs = 253,890 steps under `…/runs/20260909/201413/35e1c1e77378/checkpoints/`;
`best.ckpt` = epoch 980 by the online K710 probe (78.22 on the 1,024 held-out clips, `best.json`). Landing lines: `[probe] ep1085
k710_top1=74.51` (ep1084 75.39; linear on the detached CLS, one clean-ish view; the same instrument read 64.06 at ep240 and 73.24 at ep515);
`[share] ep1084 inv=0.290(g0.774) moment_kl=0.708(g0.194) h_moment_kl=0.002(g0.024) | omega_h=0.269 omega_z=0.153 lam=1.323 | kl_z=2.225
kl_h=0.294 trace_z=0.927 trace_h=0.993` (no quench-gate event on the whole ladder: kl_z 2.22 flat from ep240 to ep1085).

**Protocol reads (EMA weights; the D-119 evaluators as ported; 20 probe epochs at lr 1e-3; 4 nodes × 2 H100 for every B read; the number is
the val top-1 at probe epoch 20, the probe's own train top-1 in parentheses):**

| checkpoint | IN-1k attentive (global batch 128) | SSv2 attentive (16×2×3, global 256) | K400 linear on mean-pooled tokens (16×8×3, global 64) |
|---|---|---|---|
| B 240 ep (56,160 steps) | 52.91 (43.79) — job 65179605, 6 h 42 | 43.86 (63.68) — 65179606, 3 h 51 | 35.31 (38.18) — 65179607, 20 h 26 |
| B 515 ep (120,510 steps) | 55.89 (46.92) — 65024351, 6 h 48 | 47.85 (68.31) — 65024352, 3 h 18 | NOT COMPLETED: 36.82 (39.86) at probe epoch 16 when job 65024353 was cancelled after 14 h 21 (`diag/videoprobe_k400_linear_mean_k710b_ep515_ema_8gpu.partial_ep16.csv`) |
| B 1085 ep (253,890 steps) | 57.09 (48.20) — 65419464, 6 h 41 | 48.27 (69.48) — 65419465, 5 h 11 | 35.51 (39.24) — 65419466, 20 h 09 |
| S 240 ep (37,440 steps at batch 768; for reference) | 46.76 (37.07) — 64683975, 4 × 1, global 64 | 37.31 (57.11) — 64769203, 4 × 1, global 128 | 29.53 (32.36) — 64769204, 8 × 1, global 64 |

CSVs: `results/e34/attnprobe_k710b_ep{240,515,1085}_ema_8gpu.csv`, `videoprobe_ssv2_attentive_k710b_ep{240,515,1085}_ema_8gpu.csv`,
`videoprobe_k400_linear_mean_k710b_ep{240,1085}_ema_8gpu.csv`; `tab:video` reads them (B rows 240 / 515 / 1085; K400 at 515 prints ---).
Column deltas B 240 → 1085: IN-1k +4.18, SSv2 +4.41, K400 +0.20. At matched probe epoch 16 the K400 linear read orders 515 (36.82) > 1085
(35.38) > 240 (34.97). The K400 linear probe's TRAIN top-1 ends at 38.2 / 39.9 (epoch 16) / 39.2 for 240 / 515 / 1085; its val curve at 1085
moves 35.10 → 35.51 over probe epochs 14–20 under the cosine tail. One probe run per cell; the probe's run-to-run spread is not measured.

**Off-protocol reads (all under `results/e34/diag/`, never table candidates — `paper_exhibits.final()` takes `sorted(glob(...))[-1]`):**
- IN-1k attentive at lr 1.25e-4 on the 515 state (job 65158682, 6 h 41; `attnprobe_k710b_ep515_ema_8gpu_lr1.25e-4.csv`): 54.96 at epoch 20
  against 55.89 at the protocol lr (−0.93).
- K400 linear at lr 2.5e-4 on the 515 state (job 65158683, cancelled after 6 h 43 at probe epoch 6; `…_lr2.5e-4.csv`): 26.69 at epoch 6
  against 32.18 at epoch 6 under the protocol lr.
- K400 with the ATTENTIVE head on the 1085 state (job 65489558, 23 h 54; same store, weights, batch, schedule — only the head differs;
  `videoprobe_k400_ATTENTIVE_k710b_ep1085_ema_8gpu.csv`): **61.97** at epoch 20 (train 74.86; 41.53 at probe epoch 1, above the linear
  head's epoch-20 value). Not the protocol number: LeVJEPA's 44.6 is a linear read (their Table 3 caption, App. C).
- IN-1k attentive over the CLS token only, 1085 state (job 65474361, `--tokens cls`, cancelled at probe epoch 2 on Berker's word;
  `attnprobe_k710b_ep1085_ema_8gpu_CLSONLY.partial_ep2.csv`): 21.64 / 25.06 at epochs 1 / 2 against 35.34 / 39.33 for the full token set.

**Reference facts re-checked 2026-09-12 (arXiv 2608.27395 text):** Table 3 caption — "ViT-B encoders pretrained on the 20% subsample of
K710 at equal total pretraining FLOPs, evaluated frozen; IN1K and SSv2 report attentive-probing top-1 accuracy, K400 reports linear-probing
top-1 accuracy"; App. C — "the output tokens of the frozen encoder are averaged into a single vector and a linear classifier is trained on
the pooled representation"; §3 — "mean pooling followed by a linear classifier is a strictly weaker adaptation than the attentive probe,
the reported Kinetics-400 accuracies constitute a conservative estimate". Their K400 linear-probe optimization is not printed; the port
applies V-JEPA's attentive-probe optimization (AdamW 1e-3 cosine, wd .01 → 1e-6, clip 1.0, 20 epochs) to the linear head — the D-119
declared reading. The port's pooled vector = the mean over all 8 × 3,137 concatenated tokens of a view (8 CLS tokens among them); the
encoder output is LayerNorm-ed (`module.py` `self.norm`).

Nothing above is interpreted. The K400 column is the open question (HANDOVER §0, Berker: investigate without assuming a cause); every
number stays RAW pending the joint read; P3 (the z-only twin) remains unlaunched and unreadable.

### K400 feature read — built and launched (2026-09-12 evening; Berker: "cache the feats and do your thing anyway, (no k240 attentive yet because i dont see a clear benefit of that)")
What it is: the clip-level features the K400 protocol reads, cached for the three B EMA states in one pass over the FULL K400 eval store
(train 238,680 / val 19,791 clips), then optimizer-free readers and geometry on the same clips. `video/levjepa/scripts/k400_feature_cache.py`
(sampling = the protocol's val sampling, 16 × 8 at frame step 2, ONE centre view per segment, no augmentation; per clip averaged over the
8 segments: `pool` = mean over all 3,137 tokens = the protocol's pooled vector, `gap` = patch mean, `cls`; scalars tok_norm, cls_norm,
within_var, seg_var; fp16 under `features/e34_k400/<split>_rank<r>.pt`, ≈3.6 GB) and `experiments/e34_k400_features.py` (kNN k 20 t .07
from `sslgap.probes.knn`; ridge multinomial logistic regression solved by L-BFGS on standardized features, λ ∈ {1e-5, 1e-4, 1e-3}, from
`sslgap.probes.linear.linear_lbfgs_v1`; RankMe/d, effective rank/d, top-eigenvalue share from `sslgap.metrics.spectra`; cos(tap, CLS);
the token-variance split within a segment / across segments / across clips; the Fisher ratio on val; out
`results/e34/diag/k400_features_read.csv`). What it can say: whether the pooled vector's linearly decodable K400 class content moves
across 240 / 515 / 1085 under converged readers (against the 20-epoch AdamW head's 35.3 / — / 35.5), whether the CLS and patch-mean
reads of the same encoder output move, and what the pooled vector is made of. What it cannot say: why the encoder does what it does.
Smoke PASSED 19:51 (job 65720764, one A100, 48 clips per split, the 240 state: both rank files written, the read ran on them on CPU).
Launched 19:5x: cache job **65721269** (`video/levjepa/slurm/k400_feature_cache.slurm`, uniform 4 × 2 H100, node-local staging, bs 8 per
GPU, 20 workers, 8 h limit) → read job **65721370** chained `afterok` (`slurm/e34_k400_features.sbatch`, one general-partition GPU).
Not launched (Berker's word): the attentive K400 read on the 240 state. Numbers RAW on landing.

**K400 feature read LANDED (RAW, 2026-09-12 22:0x; cache job 65721269 COMPLETED 1 h 48 min on gpu[268,270,272,273], read job 65721370 6.5 min
on one A100; `results/e34/k400_features_read_k710b.csv` (moved out of diag/ 2026-09-12 23:0x when Berker made the converged CLS read the table's K400 number); train 238,680 / val 19,791 clips, one centre view per segment, EMA weights; L-BFGS
final gradient norms ≤ 1.4e-3 = converged.)** Readers on the val split (top-1, %); the L-BFGS cell is the best of λ ∈ {1e-5, 1e-4, 1e-3}
with the train top-1 at that λ in parentheses; the protocol row is tab:video's 20-epoch AdamW head (3 spatial views, RandAugment training):

| tap | reader | B 240 | B 515 | B 1085 |
|---|---|---|---|---|
| pool (= the protocol's pooled vector) | protocol head, 20 ep AdamW | 35.31 | 36.82 at probe ep16 (cancelled) | 35.51 |
| pool | L-BFGS ridge logistic, converged | 41.82 (62.8) | 43.31 (65.7) | 43.39 (57.0, λ 1e-4) |
| pool | kNN k 20 cosine | 18.99 | 19.07 | 18.25 |
| gap (patch mean) | L-BFGS / kNN | 41.80 / 18.99 | 43.34 / 19.05 | 43.41 / 18.26 |
| cls | L-BFGS ridge logistic, converged | 42.21 (56.4) | 44.75 (61.2) | 45.68 (62.9) |
| cls | kNN k 20 cosine | 26.03 | 27.71 | 28.63 |
| Fisher ratio tr(B)/tr(W), val | pool / cls | .232 / .176 | .256 / .188 | .265 / .201 |

Geometry on the train split: pool norm 19.1 → 18.4 → 16.8, cls norm 25.7 → 25.7 → 25.4, patch-token norm 31.1 → 31.3 → 29.1; pool effective
rank / d .074 → .061 → .047 (RankMe / d .373 → .376 → .339), its top eigenvalue's share of the between-clip variance .165 → .192 → .248; cls
effective rank / d .344 → .391 → .399 (RankMe / d .68 → .77 → .81), top share .03 → .05; cos(pool, cls) .252 → .235 → .184. Token-variance
split: within a segment .795 → .812 → .834, across the 8 segments .074 → .066 → .059, across clips .131 → .122 → .107. pool and gap agree
to the second decimal on every reader (the CLS is one token in 3,137). Not interpreted; joint read pending.

**K400 readout RULED (2026-09-12 22:5x; D-124, USER-DIRECTED).** Berker: "i think it is a fair game for us to report our CLS instead of mean pool and compare
with them claiming equal compute?" — Fable's answer: not as a like-for-like cell (their readout is the mean of the output tokens applied to baselines without a
CLS; their ViT-L's CLS also decodes better than its patch mean on our K710 instrument; equal compute = the 515 row by retained-token arithmetic), fair as a
marked readout of ours. Berker: "i am pretty sure they use a converged probe for themselves. and it is a linear game where we have almost a closed form
performance? it is completely fine. do the same for vit small ep240 and vit base ep240." → tab:video's K400 cell for every Ours row = the converged
ridge-logistic fit on the CLS (best of three ridge strengths, one centre view per segment, EMA, full K400 train / val), marked $^\ast$ with the readout in
the caption; B 240 / 515 / 1085 = 42.2 / 44.8 / 45.7 from `results/e34/k400_features_read_k710b.csv`; the S-240 state (`k710s3` `last.ckpt` = epoch 239,
37,440 steps, vit_small) cache launched as job 65736112 (4 × 2 H100, `features/e34_k400_s/`) → read 65736113 → `results/e34/k400_features_read_k710s3.csv`.
The protocol's mean-pooled 20-epoch head (S 29.53; B 35.31 / — / 35.51) stays here as the record. RAW until the S read lands and the row is jointly read.
**D-124 AMENDED (2026-09-12 23:1x; Berker: "reporting 35.5 for mean pooled is obnoxious i thought we are doing the same thing as they do for linear. we can just
report the converged ones for mean pool and then claim in text cls is even better? so our numbers become 41.8 and 43.4?").** Basis: LeVJEPA's public release
(commit 3ea0dda) carries no evaluation code at all and App. C gives the K400 linear probe no hyperparameters; V-JEPA's frozen video evaluator has no linear
head; the 20-epoch AdamW head was the D-119 reading and stalls 8 points under the optimum. Applied: tab:video's K400 cell for Ours = the converged
ridge-logistic fit on THEIR readout token, the mean of all output tokens (B 240 / 515 / 1085 = 41.8 / 43.3 / 43.4; S 240 pending job 65736112 → 65736113),
marked $^\ast$ with the reader named in the caption; the CLS read (42.2 / 44.8 / 45.7) stated in the caption text as the higher read. The mean-pooled
20-epoch-head numbers (29.53 / 35.31 / 35.51) remain on this card as the record of the D-119 reading. RAW; joint read pending.
**ViT-S ep240 K400 feature read LANDED (RAW, 2026-09-12 23:4x; cache job 65736112 COMPLETED 44 min 42 s on gpu[269-270,272-273], read 65736113;
`results/e34/k400_features_read_k710s3.csv`; the `k710s3` landing state = epoch 239, 37,440 steps at batch 768, EMA; same clips, view and readers as the B
ladder; L-BFGS gradient norms ≤ 1.2e-3.)** Val top-1: pool (= their readout token) converged fit **38.27** (λ 1e-5; 35.67 at 1e-4) against the 20-epoch
mean-pooled head's 29.53; CLS converged fit **39.96** (λ 1e-5); kNN k 20: pool 17.52, cls 26.09; Fisher ratio pool .246 / cls .264. Geometry (train): pool
norm 13.5, cls norm 18.5, patch-token norm 22.7 (cls / token .99); pool effective rank / d .100 (RankMe / d .410), top-eigenvalue share .213, cos(pool, cls)
.326; cls effective rank / d .438 (RankMe / d .746); token-variance split within a segment .744 / across segments .092 / across clips .163 (B 240: .795 /
.074 / .131). tab:video S row now reads 46.8 / 37.3 / 38.3$^\ast$ with the CLS 40.0 in the caption text. Not interpreted.
