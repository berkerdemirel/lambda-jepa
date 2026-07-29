# E22 — floorssl d256 view-mean at ImageNet-1k (single-cell scaling run)

**Status: PRE-REGISTERED (Berker 2026-07-20: "could you run an imagenet1k experiment using
d256 view mean for 100 epochs (it'll take a lot of time so calculate it carefully, when
submitting the job). you can use 1 h100."). Ledger row D-055. Predictions and the dose rule
locked in this file BEFORE the bridge numbers and before any training number exists.**

Views follow-up (Berker same evening: "verify how many multicrops dino training uses and how
many views lejepa imagenet1k training uses and we can also use that many views to match") —
VERIFIED from the donors: lejepa's own ImageNet-1k recipe uses **V=4** (its
`scripts/minimal_inet1k.py` launches `+V=4 +lamb=0.02 +proj_dim=16`; the paper-scale ViT-L
IN-1k config sets `++n_views=4`), so the lane's V=4 already matches. dino's donor default is
**2 global (224²) + 8 local (96²) = 10 crops** (`main_dino.py --local_crops_number` default 8;
the README ViT-S/16 command keeps it) ≈ 3.47 full-view pixel equivalents — nearly the same
budget as V=4. Multi-crop for floorssl is NOT built here; it would be new machinery and a new
aug/dose frame — a separate arm if ever wanted.

## Question

Does the current operating point (d256, view-mean floor, the vm2 doses) train healthily and
keep its structure at the IN-1k rung? This is a single-cell METHOD-SCALING run under Berker's
direct directive — NOT the data-ladder audit advancement: L-004's binding order stands,
G-M2/G-M4 remain pending, and no audit-matrix claim is made from this cell (D-055).

## Frame + arm

New frame `in1k_vits16` (PROTOCOL §2, D-055): ViT-S/16@224, 100 ep, canonical
`~/data/imagenet` — counts verified 2026-07-20: train 1,281,167 = 10.113× IN-100's 126,689;
val 50,000. Everything else = the M2 conventions: bs 128, house optimizer (AdamW 1e-3 /
wd 5e-2 / warmup 10 ep / cosine 1e-5 — warmup is epochs-defined, so absolute warmup steps
scale ×10 with the dataset; declared frame-family property, same 10% fraction), grad_clip
1.0, cadence ckpts ep25/50/75/100 + best + last, wandb online.

Arm **`in1k.floorssl.s0.d256vm`** = the vm2 method config VERBATIM: aug=lejepa V=4,
expander 2048→256, head_norm=bn, z_floor=kl, **z_floor_batch=view_mean, z_d_slice=32** —
the estimator co-design is intact at IN-1k because the view-mean floor's n IS bs (=128,
n/d′=4, independent of dataset size); num_classes=1000 (monitor head only). bs stays 128:
D-051's rejected alternative ("bs↑ changes the optimization frame for every term") stands.

## Doses — the dataset-axis bridge (rule locked BEFORE its numbers)

Nominal = vm2's **w_inv 32.8 · w_floor 38.7 · h_lamb 0.617**. E19-T1: nominal cross-frame
transplants are the certified error class; D-047 measured the aug axis alone shifting g_inv
−24%. The dataset axis is therefore measured the same way before launch
(`experiments/e22_pull.py`, job 62448807): hold the vm2 ep25 formation state, swap ONLY the
data source (IN-100 vs IN-1k, same lejepa V=4 pipeline, same seeded loader construction,
`torch.manual_seed(4242)` before the step = common slice-frame Q and drop_path draws), two
batches per side, g_enc trunk-module-only (e12h_pull convention). **RULE: per-term
w′ = w · (g_in100/g_in1k) is applied only if any term's mean ratio leaves [0.90, 1.10];
otherwise the nominal doses launch verbatim** — a <10% single-state correction chases batch
noise and costs comparability. Measured values are recorded below either way.

## Pre-registered directional predictions (Claude's pick: P-1k-A)

- **P-1k-A** — healthy end-to-end: the early-window fork analog clears (monitor rising
  through ep3–6 at the 1000-way scale), no kill trigger, monitor curve qualitatively mirrors
  vm2 (steep early rise, no collapse); ep25 anatomy shows the vm signature (z compliant in
  its own view-mean frame, content parked at h).
- **P-1k-B** — early-window collapse at bridged doses ⇒ the dose law has a dataset axis
  beyond per-term pull matching (sharpens E19-T1).
- **P-1k-C** — healthy but early monitor plateau (information-starved analog at 1000
  classes).

## Kill criteria

House incident rule (single-step grad-norm >100× running median → stop and read). Monitor
≤2×chance (0.002 for 1000-way) at ep≥3 with a falling trend = the E21 kill, adapted. z-var
scale-implosion watch (laug_eps lesson: a flat floor value can BE the crashed state — read
z-var, not the loss value).

## Compute plan + the time calculation

Measured on this exact config at IN-100 (vm2, H100, job 62436144): 33 ep in 5h11m31s =
**9.44 min/ep** including the per-epoch val eval. Scaling: ×10.113 train images → 95.5
min/ep; + val eval 50k vs 5k (forward-only, ~+1–2 min/ep); + per-link ImageFolder scan over
1.28M NFS files (~2–4 min per restart). **Estimate ≈ 97 min/ep → 100 ep ≈ 162 h ≈ 6.8 days
of pure H100 compute; budgeted 192 h (24 × 8h links, ~18% margin).**

Sequence: 2-ep smoke on the gpu partition (A100-constrained, ~4.5 h) tonight → **24 × 8h
`h100-slotA` singleton links, every link `afterok`-gated on the smoke** (a failed smoke
leaves the whole chain pending = safe-by-construction; surplus links no-op since train.py's
epoch loop exits when start_ep ≥ epochs). The links queue behind vm2's two remaining slotA
links and start when the slot frees (~Tue 09:00), so 1 H100 total and the ≤2-cap is
untouched. **ETA ≈ Jul 27–28** (inter-link queue gaps on gpu100 are the uncontrolled term).

## Numbers land below this line as they arrive; AGREED TAKEAWAY only after joint discussion.

### Bridge record (job 62448807, A40, 2026-07-20 ~23:15; results/diag/e22_pull.csv; RAW)

Per-term g_enc at the vm2 ep25 held state, mean over 2 common-random-number batch pairs
(per-pair ratios in brackets):

| term | g in100 | g in1k | ratio g100/g1k | per-pair |
|---|---|---|---|---|
| inv | .1998 | .1950 | **1.025** | [1.039, 1.014] |
| moment_kl (z-floor) | .1247 | .1437 | **0.868** | [0.989, 0.761] |
| h_moment_kl | .1982 | .2085 | **0.951** | [0.930, 0.972] |

The z-floor mean ratio leaves the pre-declared [0.90, 1.10] band → the rule FIRES; all terms
take their measured correction: **launch doses w_inv 33.6 · w_floor 33.6 · h_lamb 0.587**
(from 32.8/38.7/.617). Honest flag: the z-floor's two pairs disagree (0.99 vs 0.76) — the
2-batch mean is what the locked rule pinned; the correction direction (z-floor demand reads
harder on IN-1k batches) is the E19-T1-consistent read, magnitude noisy at n=2.

### Input-pipeline benchmark + chain revision (2026-07-20 ~23:55; job 62448902; results/diag/e22_bench.csv; RAW)

Trigger (Berker): "speed benchmarking playing with IO speed … if IO bound there's hope …
workers, persistent workers, pin memory, async loading or sharded dataset. btw batch size =
256 could work (if we do 2 h100s right?)".

**Live probe first (dmon on the RUNNING jobs):** GPU duty ~20–35% on both — smoke/A100 sm%
0/0/8/53/51/0/0/0/0/0, vm2/H100 0/0/36/84/75/0/0/0 — with 4 of 8 workers in D-state (NFS
wait). INPUT-BOUND everywhere; even the IN-100 H100 pace (9.44 min/ep) was starvation.

**Stage decomposition** (300 random IN-1k samples): NFS cold read 10.2 ms/img (cached 0.12)
· JPEG decode ≈2.3 ms · **4× lejepa augs 18.2 ms — the dominant CPU term** · mean file 110 KB.

**Loader grid** (bs 128 V=4, persistent workers, shuffle, 24-CPU node): 8w 2.2–2.3 b/s
(= the shipped setting) · 12w 3.67 · 16w 3.3–3.4 · **24w 4.46 b/s (571 imgs/s)** · prefetch 6
HURTS (−10–15%) · pin_memory ≈ neutral for loader throughput (its payoff is H2D overlap —
transfers already run non_blocking).

**GPU step** (A40): bs128 0.406 s/step (315 imgs/s, 16.8 GB) · bs256 0.798 s (321 imgs/s,
**33.4 GB**). H100 demand ≈ 0.15–0.20 s/step (vm2's 0.572 end-to-end × the ~30% observed
duty; A40/2.5 device factor agrees) ≈ 5–6.7 b/s = the rate the loader must beat.

**REVISION (perf-only; loss math byte-untouched):** `train.py`'s DataLoaders gained cfg knobs
`pin_memory` / `persistent_workers` / `prefetch_factor` with defaults = the old behavior for
every existing lane (the old inline "persistent_workers=False: official" was D-011 port-exact
lineage that had leaked into the shared loop — flag: CLAUDE.md hygiene says True; default
ruling deferred to Berker). The 24 armed links were cancelled and the chain resubmitted as
**16×8h links 62448964-79** with `num_workers=24 pin_memory=true persistent_workers=true`,
`--cpus-per-task=28` (H100 nodes are 224c/8 GPUs = 28/GPU fair share), still afterok-gated on
smoke 62448846 (the smoke validates data path + doses under the OLD loader knobs — declared
acceptable: loader knobs cannot touch the loss). **Expected 4.5–6 b/s → ~30–40 min/ep →
100 ep ≈ 50–67 h ≈ 2.1–2.8 days; revised ETA ≈ Thu evening–Fri** (was ~7 days). Conservative
floor (no scaling win, bench 8w rate) = 127 h, inside the 128 h chain capacity.

**Sharding verdict:** file-IO (10 ms, hideable by parallel workers — proven by the worker
scan) is not the binding term; the aug CPU (18 ms) is — tar/webdataset shards would not
reduce it → NOT built. The lever beyond worker scaling is GPU-side augmentation (uint8
transfer ≈ ×4 smaller H2D + augs off the CPU) — a real build, only if >6 b/s is ever needed.

**bs 256 / 2×H100 (the question answered):** bs 256 V=4 peaks at 33.4 GB → **fits ONE H100**;
per-image GPU rate is identical (321 vs 316 imgs/s), so bs alone buys ~nothing, and while
input-bound it buys exactly nothing. A ×2 wall-clock from two H100s means DDP, which the
trainer does not have — new machinery with semantic forks (per-rank vs Sync BN in the
expander; global-batch-256 lr scaling; the floor estimator's per-rank n) + its own validation
discipline, and it would occupy the whole H100 cap. With the pipeline fix the single card is
expected near-saturated; DDP is a separate decision if a future run needs it.

### Bench round 2 + final chain (2026-07-21 ~00:50; job 62448992; results/diag/e22_bench2.csv; Berker's picks applied)

Worker scaling at the H100-node CPU share: 24w 4.20 · **28w 5.30 b/s (679 imgs/s)** · 32w 5.05
(oversubscription dip). Pre-resize lever measured in memory (shorter-side 256, q87): decode
1.86→0.54 ms · aug4 19.2→14.6 ms · file 123→**22.8 KB** — would clear the H100 compute ceiling
(~6.2 b/s ≈ 27 min/ep). Realized "before" anchor: the smoke runs 1.23 steps/s (157 imgs/s) on
the old 8-worker settings (wandb 399j3nzo).

**Berker's rulings:** E22 stays FULL-RES (pre-resized copy NOT built — certified-pipeline
comparability outranks the last ~15%); e200's 5 pending links retro-tuned (62449372-76,
perf-only, its eval cadence untouched); eval_every=2 adopted for E22 (D-056; the knob gates
ONLY the monitor eval — ckpt saves sit outside it after a first-cut `continue` was caught
pre-commit that would have skipped `_last` and the odd cadence epochs ep25/75; early-window
kill now reads ep2/4/6). **FINAL chain: 16×8h links 62449427-42** at
`num_workers=28 pin_memory=true persistent_workers=true eval_every=2`, 28 CPUs, afterok on
smoke 62448846. Expected ~30 min/ep → **100 ep ≈ 50 h; ETA ≈ Thu**.

### PARKED builds (design notes; no build without a directive)

- **GPU-side augmentation** (the >6 b/s lever): loader returns uint8 HWC (H2D 77 MB/step vs
  308), RRC/flip/jitter/blur/solarize as torchvision-v2 or DALI device ops. Removes the
  ~15–19 ms/img CPU term → loader ceiling >10 b/s at 8 workers. Forks to resolve: aug RNG
  moves to a device stream (new stream semantics, pinned-seed reproducibility); blur/jitter
  kernel parity vs the PIL implementations (distribution-level check before any certified
  use). ~1–2 days build + smoke discipline.
- **DDP 2×H100** (~×2 → ~1-day IN-1k): torchrun, per-rank bs 128 (global 256),
  DistributedSampler; recommended per-rank BN (each rank's expander BN still sees 512 rows =
  the certified statistics; SyncBN would change them); lr-scaling question at global-256 is a
  recipe fork to rule on; per-rank view-mean floor keeps the certified n=128; rank-0
  checkpointing/wandb. Occupies the full H100 cap while running. ~2–3 days build+validation.

### Smoke un-gated on live curves (Berker 2026-07-21 ~01:10: "i dont wanna waste so much time
### on the smoke … check the current curves to see if they look healthy, then launch")

Read at step 3602 (~0.36 ep, wandb 399j3nzo), against the kill criteria: grad_norm max/median
2.6× (bar 100×) · z-floor moment_kl 1.54→0.65 monotone DESCENDING (an implosion would spike
the −logdet barrier, not descend) · probe CE 7.05→6.48, below the 1000-way chance level 6.91
· inv plateau .39 ≈ the healthy vm2 formation band (the collapse signature — inv racing up
WITH the monitor dying — absent) · h_moment_kl mild upward transient 1.01→1.25 (conditioner-
dose pattern; an ep25-anatomy item). VERDICT healthy → all 16 links' afterok dropped
(`scontrol update Dependency=singleton`); the chain now starts at slot handover
unconditionally. The smoke keeps running to ~04:00 as a free A100 eval-path/ckpt record
(gates nothing; scancel 62448846 to drop it). SLURM label note: `gap-train` on the smoke is
just train.sbatch's job name (gap = sslgap), not a run identity — the run is
`in1k.floorssl.s0.d256vm.smoke`.

### Third-H100 authorization (Berker 2026-07-21 ~01:20: "ah you dont need to wait for vm2 to
### hand over, this is just a large scale peek so you can use 1 additional h100.")

The 16 links were moved to their own singleton lane `h100-slotC` (scontrol rename, job IDs
unchanged) — a per-cell exception to the standing ≤2-H100 cap on the D-021 precedent; the cap
itself is unchanged as a rule. Three H100s now run concurrently: slotA vm2 (ends ~09:00),
slotB e200, slotC E22. E22 training starts immediately; ETA ≈ Wed night–Thu.

### Realized tuned pace (first link, gpu274, 2026-07-21 ~01:45; RAW)

Marginal rate over three 100 s windows at steps 967→2469: **4.80 / 4.79 / 4.71 steps/s**
(cumulative 4.40 incl. startup) = 90% of the bench loader ceiling, ~3.9× the pre-tuning
realized pace. **≈35–36 min/ep (eval every 2nd ep included) → 100 ep ≈ 60 h → ETA ≈ Wed
afternoon/evening.** Residual gap to the ~6.2 b/s compute ceiling = main-process contention
at 28w/28c — micro-tuning between 8h links is possible from realized data if ever worth it.
Note: renamed links keep submission-time log paths (`outputs/h100-slotA_624494xx.out`).

## vm3 at IN-1k (D-061; Berker 2026-07-21: "also launch another imagenet1k with 256vm3
## setting. (you can use one h100)")

Cell `in1k.floorssl.s0.d256vm3` = the E22 cell + the D-058 symmetric payment
(`h_floor_batch=view_mean h_d_slice=32`) — both conditioners on per-image view means at
d′=32. Lane `h100-slotB` (free since e200 finished; fleet back to three H100s: vm3-in100 on
slotA until ~18:30 · E22 on slotC · this). Fast loader + eval_every=2 (the E22 conventions).

**Doses (pre-declared):** w_inv/w_floor transplant E22's measured dataset bridge (the terms
are byte-identical between vm2/vm3 configs): 33.6/33.6 expected, re-measured in the same job.
The vm-h term's dataset ratio is UNMEASURED (E22 bridged the pooled h term) →
`e22_vm3pull` (job 62473849): full vm3 config at the vm2 ep25 held state, IN-100 vs IN-1k,
2 common-RNG batch pairs; rule as E22 — per-term w(in1k) = w_vm3-in100 · g_in100/g_in1k,
applied only if a mean ratio leaves [0.90, 1.10]; vm3-in100 doses = 32.8/38.7/0.339. The
bridge job doubles as the mechanics gate (full training_step fwd+bwd on real IN-1k batches
under the exact config; both component paths — vm3 code, in1k frame — already smoked) →
chain launches UNGATED with the early-curve watch (the E22 pattern Berker set).

**Pre-registered predictions (pick: A):**
- **P-1kvm3-A** — healthy; early window tracks E22 within noise (the h payment axis is
  second-order at conditioner share, at scale too).
- **P-1kvm3-B** — the symmetric h payment shows at scale: h_kl (own frame) lower than E22's,
  clean-frame h effrank lower (the vm spectral signature extending to h), probes ≥ E22.
- **P-1kvm3-C** — instability the IN-100 vm3 didn't show: the n=128 h estimator under
  IN-1k's diversity (kill criteria standing, E22's set).

Time: the E22 measured pace applies (~36 min/ep) → ~60 h → ETA ≈ Fri morning. 16×8h links.

### vm3-1k bridge record (job 62473849; e22_vm3pull.csv; RAW)

Mean g_in100/g_in1k over 2 common-RNG pairs: inv **1.025** · z-conditioner **0.868** ·
h-conditioner(view-mean@32) **1.014** — the two byte-identical terms REPRODUCE E22's ratios
to 3 decimals (in-job consistency check), and in100-b0's g_h 0.3585 reproduces the morning
vm3pull exactly. The z ratio again leaves the band → rule fires on all terms from the
vm3-in100 doses (32.8/38.7/0.339): **launch w_inv 33.6 · w_floor 33.6 · h_lamb 0.344**.

## vm4 at IN-1k (D-065; Berker 2026-07-22: "launch in1k for vm4" — the E21-T2 operating
## recipe at scale)

Cell `in1k.floorssl.s0.d256vm4` = the E22 frame + the full vm4 config (view-mean payment
both taps, d′=128 both, queue_steps=3). Lane h100-slotA (free post-vm3-in100), 16×8h
singleton links, **--exclude=gpu277 everywhere** (the black-hole node), fast loader,
eval_every=2. Doses by the composed bridge (`e22_vm4pull_1k`, job 62593375): per-term
w(in1k) = w_vm4-in100 (32.8/157.8/1.894) × g_in100/g_in1k measured with the queue warmed at
the vm2 ep25 held state; band rule [0.90, 1.10] as E22. No smoke (config-only composition of
two validated paths — the vm4 code smoked at IN-100 this morning, the in1k frame long
validated; gate = the bridge dry-run + the early-curve watch, the E22 pattern).

**Pre-registered predictions (pick: A):**
- **P-1kvm4-A** — the E21-T2 recipe scales: healthy end-to-end; probes ≥ the E22 vm2 cell at
  matched epochs; both spaces high-rank at the ep-cadence anatomy (the vm4 signature).
- **P-1kvm4-B** — partial: queue staleness interacts with the 10× data diversity (each
  image reappears in the queue window never — pure cross-image staleness) and conditioning
  work lands between vm2-1k and the IN-100 vm4 pattern.
- **P-1kvm4-C** — instability the IN-100 cell didn't show; kill criteria standing (E22 set).

### vm4-1k bridge record (job 62593375; e22_vm4pull_1k; RAW) + launch

Warmed-queue held-state pulls, mean of 2 common-RNG batches (per-batch values in the .out):
inv g_in100/g_in1k = .17725/.21610 = **.820** · moment_kl .03115/.03805 = **.819** ·
h_moment_kl .06765/.07630 = **.887** — all three OUTSIDE the [0.90, 1.10] band (both
batches agree in direction; IN-1k pulls harder per unit weight on every term — unlike the
vm2/vm3 bridges, where ratios sat near parity). The locked rule fired on all terms:
**w(in1k) = 32.8×.820 / 157.8×.819 / 1.894×.887 = 26.9 / 129.2 / 1.679**.

Chain LAUNCHED: h100-slotA links **62593427–62593442** (16×8h singleton, --exclude=gpu277,
28 CPUs, fast loader explicit, eval_every=2), ARGS = the vm3-1k pattern + queue_steps=3 +
d_slice 128 both taps + the doses above. Verified 16/16 in queue at submit. ETA at the vm2
pace (~36 min/ep): ~2.5 days of lane time.

### vm2-1k LANDED (2026-07-23 20:04) + the in1k eval frame (D-066 PROPOSED)

Training finished clean: **best online probe .6254** (ep92 read .6170, monotone to the end;
100/100 epochs, chain drained with links to spare). Cadence ckpts ep25/50/75/100+best+last
on disk.

**Landing audit fired** (extract 62626403 → probe 62626404 + battery 62626405) under a
newly DECLARED in1k eval frame — none existed for this rung, and `extract.py` hardcoded
`in100.*` manifest names for every imagefolder dataset (latent trap: the existing IN-100
CSVs would have been silently reused — IN-100 images audited under an in1k run_id). Fix:
manifest names are now dataset-keyed (`in100.*` byte-identical, `in1k.*` new; loud KeyError
on unknown datasets). Frame parameters (PROPOSED, D-066): `in1k.train500.v1` (500/class =
the in100 frame's per-class density; 500k rows), `in1k.val.v1` (full 50k),
`in1k.pairs10.v1` (10/class → 10k pairs images = the in100 pairs frame's TOTAL n — matched
global-estimator size; per-class pair claims not made at this rung). Numbers land RAW under
this frame; if the parameters are amended, re-extraction is one job. vm3-1k/vm4-1k reuse
the same frame + machinery when they land.

### vm2-1k landing numbers (jobs 62626403-05, 2026-07-24; RAW, frame = D-066 PROPOSED)

Probes (val 50k, 1000-way; full table results/probes/): **h.cls linear .6211** (l2_v2; raw
.6088, house .6020) · h.cls kNN k20 **.5399** · h.gap linear .506/kNN .327 · z.proj.tap1
linear .573 · z.proj.out linear .474/kNN .385. Online-monitor best was .6254 (trains with
the model on aug batches; the frozen-feature probe reading ~.62 is the family-comparable
number). Battery (train500 store): h.cls (d384) rankme **345.5** effrank **282.6**
(/d: .900/.736) · z.proj.tap1 (d2048) rankme 1291.8 effrank 277.7 · z.proj.out (d256)
rankme **164.3** effrank **133.0** (/d: .642/.520). Full CSVs in results/battery/. No
cross-cell reads until vm3/vm4-1k land (matched frame, joint discussion).

### vm4-1k PAUSED at ep74 (2026-07-24 14:08)

Cancelled mid-run (all 11 slotA links) to free H100s for Berker's rebuttal jobs. NOT a
result — a compute yield. State: `outputs/in1k.floorssl.s0.d256vm4_last.pt` epoch 74, step
750675, best_acc **.5847** (online monitor; ep62 frozen probe read .5520), modules+optim
intact, moment queue re-warms on resume (declared, `extras` empty by design). Resume:
`bash slurm/resume_vm4_1k.sh` — same args → same run_id → auto-resume from ep74; ~26 epochs
(~16h) left. No landing audit until it finishes.

### vm3-1k LANDED (2026-07-24) — the d′=32 wall arm at scale (landing 62629948-50; RAW, D-066 frame)

Training clean, best online **.6176** (100/100 ep). Frozen probes (val 50k, 1000-way):
h.cls **lin .6211** (l2_v2; raw .6089) · h.cls **kNN k20 .5035** / k200 .4567 · z.proj.out
lin .466 / kNN .387. Battery (train500): h.cls (d384) rankme 278.3 (.725) **effrank 153.4
(.400)** · z.proj.out (d256) rankme 162.7 (.636) effrank 132.3 (.517). Full CSVs in
results/. RAW — the 3-cell joint read (vm2/vm3/vm4) waits on vm4-1k finishing its last 26 ep
and on joint discussion.

**Side-by-side so far (RAW, NO takeaway; frozen v2 probes + train500 battery):**
| cell | online best | h.cls lin | h.cls kNN | h effrank/384 | z.out lin | z.out effrank/256 |
|---|---|---|---|---|---|---|
| vm2-1k (relaxed z, strict h) | .6254 | .6211 | **.5399** | **.736** | .474 | .520 |
| vm3-1k (symmetric, d′=32 wall) | .6176 | .6211 | .5035 | .400 | .466 | .517 |
| vm4-1k (symmetric, queue d′=128) | .5847@ep74 | — | — | — | — | — |

Flags for the joint read: h.cls LINEAR is identical (.6211) across vm2/vm3 — the linear
probe does not separate them; the separation lives in **kNN (.5399 vs .5035)** and **h
effrank (.736 vs .400/d)** — the same label-aware / geometry axis E21-T2 identified at
IN-100. vm4's numbers complete the picture when it lands.

## IN-1k eval = the E20/E21 guillotine-zoo, 3-cell (Berker 2026-07-29; PLAN, taps TBD-jointly)

Repeat the guillotine (metrics × depth, one row per cell, zoo2 rules) for the IN-1k family.
Machinery already exists — `experiments/e21_guillotine_vm.py` encodes every rule Berker set
(0–1 axes for bounded quantities, ranks ÷ per-station d with z-out d-annotation, per-row
gauss_kl ylims, class-cos-as-margin, no gap station). Only a new FAMILIES list + the depth
stores are needed. **No new figure code.**

- **Rows = vm2 / vm3 / vm4** (the estimator-payment axis at scale). **Control gap flagged:**
  at IN-100 the grey control was zonly/d256-pooled; neither was run at IN-1k, so either no
  grey control (vm2 as the visual reference) or a new d256-pooled-1k run (a full cell — likely
  not worth it). Recommend: no control, vm2 reference.
- **Depth stations (ViT-S/16 @ 1k):** backbone L03/L06/L09 + cls, then head cls→tap1(2048)→
  z.out(256) — same fractional depth as the IN-100 `[3,6,9]`+cls. **Requires L-tap
  re-extraction:** vm2/vm3 landed as plain `.ext` (no h_layers); the guillotine reads `.extL`
  stores. That re-extraction is a **gpu-partition** job (A40/L40S) — it does NOT touch the
  rebuttal H100s. Ready to fire for vm2/vm3 now once the tap set is agreed; vm4 rides its
  landing. Then run the depth-metrics instrument → an `in1k` depth-metrics CSV → render.
- **Two synergies with E23:** (a) the **pos-invariance column across depth IS the orbit-radius
  trajectory** (pos-cos = orbit tightness) — this guillotine doubles as the first
  orbit-leakage instrument at scale, feeding the E23 study directly; (b) the measured
  linear-ties-but-kNN-separates gap (vm2/vm3 both .6211 linear; kNN .5399 vs .5035) becomes a
  DEPTH TRAJECTORY — the figure shows WHERE along h→z the separation emerges = the leakage
  gradient made visible. RAW; the 3-cell takeaway waits for vm4-1k + joint discussion.

### D-066 REVISED → the STANDARD full eval (Berker 2026-07-29: "why not full 1k eval on 50k?")

Correction: the val eval was ALWAYS the full 50k (the reported val_acc is on all 50k). What
the first-draft frame subsampled was (a) the probe TRAIN set → 500/class = 500k of 1.28M, and
(b) the structural battery, which `audit.py` ran on that 500k train sample rather than the
clean 50k val. Rationale (match in100 per-class density) was weak; dropped. **Standard frame:
probe FIT on full 1.28M train → eval on 50k val; battery on full 50k val; pairs bumped.**
Feasible (GPU-SGD probe fits a 24GB card even at 2048-d; store << budget; only a probe
walltime bump). **⇒ the vm2/vm3-1k probe+battery numbers above are under the SUPERSEDED
subsampled frame** and are re-run pending (gpu partition; vm4 on landing). Kept: the
dataset-keyed manifest fix. Re-run is one extract (train_per_class=null → full) + battery with
`train_manifest=in1k.val.v1` + a probe walltime bump.
