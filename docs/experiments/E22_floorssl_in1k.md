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
