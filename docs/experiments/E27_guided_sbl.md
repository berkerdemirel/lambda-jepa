# E27 — the guided multi-ViT in1k program (S/B/L)

**Status: S PHASE RUNNING (launched 2026-08-06 evening under D-079a/D-081/D-082 — see
§S-wave launch log). Five chains on IN-1k: `e27smc`/`e27smcb`, `e27slg`/`e27slgb`, and
the `e27lej` control; ep17–22 of 100 as of 2026-08-07, every spectral cell above the
[.35, .50] band, first steering read due at ep25. B and L stay gated (D-079b/c).
Successor row: E25-T1 (Berker verbatim: "we will look for training of multiple vits on
in1k next. we shouldnt do it blindly, this is where our target recipe for joint loss
forces come into the play while our R6 read provides an additional free guide").**
*(The design-draft header this replaced — "NOTHING TRAINS until Berker approves" — was
true when written on 2026-08-06 morning and was overtaken by the same day's approvals;
corrected 2026-08-07.)*

## Question

Does the guided recipe — realized-share dosing (E24-T1/T5) + the threshold-normalized
geometric band (theory §7, R6) — transport across ViT capacity at IN-1k, and does
floorssl beat the LeJEPA references at matched arch/epochs beyond ViT-S? Secondary:
does the R6 band's placement claim hold at scale (is the incumbent vm4's Ω_h(cls) = .211
genuinely over-removal — i.e., does a band-steered cell beat the anchor-dosed cell)?

## The anchor-attraction diagnosis (measured, 2026-08-06 — the input this design answers)

Live voas trajectory (target (.55, .42, .03), w = 21.4/49.6/1.89 derived on the vm4-ep25
formation state; per-epoch `[share]` logger):

| ep | s_inv | s_z | s_h | g_inv | g_z | g_h | Ω_h(train) | Λ | ρ̂_z |
|---|---|---|---|---|---|---|---|---|---|
| 1 | .514 | .419 | .067 | 1.140 | .401 | 1.670 | .674 | 1.12 | .27 |
| 2 | .628 | .334 | .039 | .544 | .125 | .380 | .596 | 1.31 | .36 |
| 10 | .712 | .257 | .031 | .259 | .040 | .128 | .467 | 1.52 | .42 |
| 25 | .693 | .277 | .030 | .239 | .041 | .118 | .375 | 1.59 | .47 |
| 40 | .739 | .233 | .028 | .287 | .039 | .122 | .357 | 1.62 | .49 |
| 54 | .734 | .238 | .028 | .284 | .040 | .121 | .336 | 1.55 | .52 |

Reading (raw, for the joint discussion): at ep1 the run sits essentially ON target — the
dose was right for the state it was derived on. By ep10, g_z has fallen 10× (.401→.040;
the satisfying conditioner's force recedes, ρ̂ climbing .27→.52 = the evidence meter)
while g_inv fell only 4.4× — the share migrates inv-up and PARKS at the ≈(.70–.74, .23–.28,
.03) fixed point for 45 straight epochs. The "anchor" is the equilibrium share profile of
the loss family; static weights cannot hold realized shares off it (E24's bidirectional
attraction, now measured on a full in1k trajectory). Same signature in the vm4 lineage
(in100 v-cells end at inv .72–.84 by ep99).

**Design consequence:** share targets are FORMATION-phase targets — they steer ep0–10 and
select the equilibrium's basin; past formation, shares belong to the equilibrium and the
meaningful mid-run channel is the GEOMETRY (Ω_h against the frame's band), not the share
vector. Mid-run steering, where used, acts on Ω_h.

## Design

**Arches:** ViT-S/16 (22M, the incumbent frame) · ViT-B/16 (86M, new frame
`in1k_vitb16`) · ViT-L/16 (304M, new frame `in1k_vitl16`) — each needs a PROTOCOL §2
frame row (D-079). 224px, 100 ep, bs 128, lejepa V=4 lane, house hygiene (grad_clip 1.0,
warmup 10, cosine, fast loader, per-epoch `[share]`+orbit logging from birth).

**Anatomy (invariant across S/B/L — the arch is the ONLY axis; config CONFIRMED by
Berker 2026-08-06). SUPERSEDED on the aug/V axis and the slice-width axis by Recipe v2
below (Berker 2026-08-06 second round — LeJEPA-recommended frame):** view-mean h+z · d′ = 128 both taps · head = the family ladder
K2/hidden-2048/out-256 (D/d′ = 2) — head byte-identical across arches except the input
width (384/768/1024) · bs 128 · lejepa V=4 · 100 ep. **Estimator RULED: the running-cov
ring (queue 3, n_eff/d′ = 4 — d256vm4 style), Berker 2026-08-06: "we already observed
running cov works better so we will do our design on top of that (i am thinking it wont
effect the forces)."** The e24voas ep100 A/B still lands on this card as information
(deficit −.030 @ ep25 → −.024 @ ep54), but the design does not wait on it. **Dose
adjustment at ep10 (Berker, same message): "we will adjust the dose at ep10 checking
the metrics + forces"** — the standing two-pass rule: ep10 confirm read of realized
shares (warm-ring per D-072) + the geometry channels (Ω_h/Ω_z, Λ), one correction from
the cell's own ep10 g's. **Anchor-attraction diagnosis AGREED (Berker, same message).**

**Dose procedure per arch (E24-T1 verbatim):** healthy pilot past warmup (~1 GPU-h) →
measure per-term trunk g + T at the formation state (warm ring per D-072 if ring) →
w_i = s*_i·T/g_i at **s\* = (.63, .34, .03)** (the anchor/basin coordinate — dosing ON
the equilibrium, not off it; the voas lesson) → ep10 confirm ± one correction → run.
T = the certified in1k family total (9.0) scaled per-arch by the measured pilot.

**The R6 guide (the "free geometric target") — MEASURED 2026-08-06 (native law census,
`results/diag/omega_lawcensus_*.csv`):** c(in1k) = **.8414** (n=3, R² .993) → threshold
c² = **.708**; the constant is frame-invariant (toy .8193 / in100 .8083 / zoo .8258 /
all-25-runs .8211 ± .019 — R5 holds). Registered band [0.5, 0.7]·c² → **Ω_h(cls) ∈
[.35, .50], center .42**. Evidence around it: toy top cells sit at .54–.78 of threshold
(C′ max K3 = .78), in100 top at .48–.56, the zoo's strong aug-invariance stratum
(dinov2/dinov3/dino) at .47–.57 — while our in1k winner vm4 sits at **.30** (and the
low side degrades everywhere it is sampled: toy K0 .32 / W64h0 .34, in1k vm3 .26). An
alternative reading (positions of per-frame winners drift down with scale: .65 → .48 →
?) would center the target lower, ~[.45, .55]·c² ≈ Ω_h [.32, .39] — **the steering
center is an approval-time choice; the S-band cell discriminates either way.** Channel
calibration: in-training Ω rides the train-aug stack (trajectory instrument); the o8
landing number is canonical — voas ep100 landing measures the offset for this frame.

**The S discriminating pair (the "not blindly" cell):** ViT-S runs TWO cells:
- **S-anchor:** dosed at s\*, untouched to ep100 (the guided-recipe baseline; direct A/B
  against vm4 .6392 and voas).
- **S-band:** dosed at s\*, plus a PRE-REGISTERED steering rule — at ep25 and ep50, if
  in-training Ω_h sits below the band floor (0.5·c², channel-calibrated), re-dose the
  two conditioners down / inv up by the share law to lift the equilibrium Ω toward
  0.6·c²; at most two corrections, both logged as D-rows on this card. This is the
  moved voas steering question, run as its own arm instead of contaminating the A/B.
B and L inherit the S winner's placement (single cell each). If S-band ≤ S-anchor, R6's
actionable claim is refuted at in1k and B/L run anchor-dosed — that refutation is a
result.

**RE-SCOPED AT LAUNCH (D-081) THEN COMPLETED (D-082, Berker: "launch band steering
for both versions"):** the S wave = the full 2×2 — **e27smc / e27smcb** (LeJEPA-
matched small locals, anchor / band-steered) and **e27slg / e27slgb** (large locals
(0.7,1.0), anchor / band-steered) — all four dosed at s\* from their version's own
pilot; band cells launch byte-identical to their anchors, the steering rule above
(ep25/ep50, audit-channel Ω_h vs [.35,.50], ≤2 corrections as D-rows) is the only
difference. The ep10 health check (metrics+forces on the saved ep10 ckpt) applies
to every cell. The aug axis (mc vs lg) and the steering axis (anchor vs band) are
now both live — R6's placement question discriminates on each independently.

**Budget + compute reality (REWRITTEN per D-080, Berker 2026-08-06):** H100 nodes are
FULLY DRAINED → the launches start on **A100-80GB** (the only other 80 GB card); when
H100s return, the in1k program may use **up to 8 concurrent H100s** (the ≤2-slot cap
stays for everything else). **Efficiency is mandatory pre-launch** ("it is important
that we run the jobs efficiently, dataloader persistent worker etc as well as multi
gpu"): loader tuning verified per node class (persistent workers, pin_memory,
workers = allocated CPUs — D-057 defaults) and **multi-GPU DDP built + smoke-tested**
(D-056's parked item unparked by directive; bs and estimator anatomy must be
preserved under DDP — global batch semantics = a design point of the resource plan,
NOT a silent change). Measured single-GPU anchors: S ≈ 48 min/ep on H100 incl.
every-epoch probe (voas); A100 expected ~1.3–1.6× slower; B ~near the input wall;
L ~1.5–2× S (grad_ckpt available). **The concrete resource proposal (GPUs per run,
workers, DDP layout — H100 case AND A100 case) OPENS the next session, before any
launch (Berker's sequencing).** Cells: S-anchor + S-band + B + B-lejepa-control + L.

**Eval:** standard full in1k frame (D-066): linear+kNN on full train→val 50k; landing
chains (extract o8+L → probe + battery + D-068 instruments) per cell; E25 transfer
battery re-run on the S/B/L bests (arch-confound killer for the ssl-transfer band);
guillotine per arch on the standard stations.

**References at matched frame (verify-at-read):** Lightly LeJEPA ViT-S/16 100ep = 64.0
(E25-T1's beaten bar); LeJEPA paper ViT-L in1k 100ep (Table 2 lineage — exact in1k
linear number to be transcribed before launch); ViT-B reference TBD (lejepa retrain at B
= a possible control cell, NOT in this budget — Berker's call).

## Pre-registered predictions (DRAFT — lock at approval, before any number)

- **P-E27-1 (share transport):** formation-state shares land on s\* after ≤1 correction
  at every arch; the post-formation drift parks at the same (.70–.75, .22–.28, ~.03)
  equilibrium at all three widths.
- **P-E27-2 (R6, the discriminator):** S-band ≥ S-anchor + .003 online — the incumbent
  .211 is over-removal. (Refutation = R6 not actionable at in1k; recorded either way.)
  **[D-081/D-082 operational form, LOCKED AT LAUNCH (before any number, veto open):
  the prediction now reads on BOTH live axes. Aug axis: large locals demand less
  invariance → Ω_h(lg) > Ω_h(mc) at matched epoch; if R6's actionable claim holds
  (the incumbent .30-of-threshold is over-removal), the version sitting nearer the
  band wins: e27slg ≥ e27smc + .003 online. Steering axis (the original form, per
  version): band ≥ anchor + .003 online wherever the anchor's Ω_h sits below the
  band floor and steering actually fires; if the anchor is already in-band the
  steering cell should be a null (≤ noise band). Refutations recorded either
  way.]**
- **P-E27-3 (capacity):** probe rises S→B→L at matched frame; Ω_h(cls) at land falls
  with capacity toward the public-zoo strong stratum (.32–.39) — or, if P-E27-2 holds,
  stays pinned in-band while accuracy rises.
- **P-E27-4 (vs references):** beats the matched-frame LeJEPA reference wherever one
  exists (S: 64.0; L: paper number), same-protocol linear.
- **P-E27-5 (transfer):** the E25 combined-table gap to LeJEPA ViT-L closes
  monotonically with our arch size at matched pretraining.
- **P-E27-6 (theory tie):** every healthy cell lands below its frame threshold
  (M < 0 at h) with Λ ≥ 1; unhealthy cells (if any) show the wall signature in ρ̂/g_z
  before the probe shows it.
- **Kills:** house rules; ep-gate max(25, warmup+5) full-curve; noise stat vs the voas
  band (mean|Δacc| ≤ .027, inv_sd ≤ .019); single-step gnorm >100× median = INCIDENT.

## Design rulings (all Berker 2026-08-06 — design COMPLETE, gate rows remain)

1. **The S pair** ("we do pair with mid adjustments if necessary at ep10") — ViT-S
   runs S-anchor + S-band; both get the ep10 metrics+forces check; the band cell's
   steering applies at ep10, further mid-run adjustment only if necessary (Ω_h
   leaving the band).
2. **The band = R6-as-registered [0.5, 0.7]·c²(in1k) → Ω_h(cls) ∈ [.35, .50]**
   (answered via the question round).
3. **lejepa control cell at ViT-B: YES** ("yes i will extend the budget") — lejepa
   V=4 ViT-B under the identical frame (100 ep, bs 128), the matched-frame reference
   no publication provides; adds ~4–5 d of one slot to the program budget (S refs =
   Lightly 64.0; L ref = the LeJEPA paper number, to transcribe before launch).
4. **The frame = LeJEPA's recommended starting point (second round, Berker
   2026-08-06):** "lejepa says 'We thus recommend to use λ = 0.05, Vg = 2, Vl = 8,
   and batch size ≥ 128 as starting points' we will keep the augmentations same …
   same architecture and nof epochs. batch size will be 128." → BOTH lanes train
   under identical multicrop aug (2 global + 8 local), matched arch, 100 ep,
   bs = 128 exactly (the estimator co-design pin). **λ configures the lejepa
   CONTROL cells only** — "our own runs wont include anything with lambda tho we
   have our own recipe" (the house three-term loss, doses by the pilot procedure).
5. **Slice ratio held across arches (same message):** "as the dimensionality of the
   trunk increases we should consider keeping the ratio same (for exploring the
   dimensionalities) which would increase our running cov step size, with a fixed
   bs=128" → h-tap d′ = D/3 per arch, ring length grows to keep n_eff/d′ ≥ 4 at
   fresh n = bs = 128 (table in Recipe v2).
6. **Compute stance (same message):** single-GPU per cell wherever bs=128 V=10
   fits — "im not sure if we really will need ddp if we are gonna fit bs=128 with
   this much views … we wont gain much as long as we fit v=10 (2 global, 8 local)
   with bs=128 + architecture"; DDP is the MITIGATION where fit or wall-clock
   demands ("if not ofc we should mitigate accordingly"), not the default.
7. **LeJEPA's multicrop mechanics are unpublished** (their full trainer je.py is
   absent from the public drop; only the minimal single-size scripts exist — the
   E25 precedent) → the multicrop construction below is a DECLARED RECONSTRUCTION;
   each component tagged RULED (Berker's words) or PROPOSED (awaiting his word).

## Recipe v2 — the run recipe, self-sufficient (2026-08-06 second round; supersedes the Anatomy bullets it names)

**A. Frame (S wave per D-082 = FIVE launched cells: e27smc, e27smcb, e27slg, e27slgb,
e27lej; then B, B-lejepa-ctrl, L on the gates)**
1. IN-1k train, 100 ep, bs = 128 exactly. [RULED]
2. Aug, identical in both lanes: V = 2 global @224 + 8 local @96 per image.
   **TWO S versions [RULED, D-081 — "i want two versions"]: (i) e27smc(b) = locals
   RRC (0.05, 0.3), the LeJEPA-matched family; (ii) e27slg(b) = locals RRC
   (0.7, 1.0) — large-crop locals, the only changed knob (V, resolutions, compute
   byte-matched; the large-crop question as a training arm).** Scales CORRECTED at
   D-082 from the DINO-canonical guess to the **LeJEPA-repo-PUBLISHED geometry**
   (README: globals (0.3, 1.0) @224; locals (0.05, 0.3) @98 — 96 here, the declared
   /16-patch adaptation; V_l=8 per the paper quote outranks the README's 6-local
   example). Family continuity (the delegated arrangement): the certified V=4 stack
   (RRC (0.08,1.0)@224) is a scale-mixed family — (i) splits the mixture into
   explicit strata ("we already had small crops"); (ii) removes the small stratum
   (the contrast cell). B/L inherit the S winner's version.
   [RULED: 2g+8l, same aug both lanes] Crop geometry [DECLARED, DINO-canonical —
   LeJEPA's own is unpublished; presented in the launch round and the S launch
   rides it; the B-phase paper/repo recheck may still transcribe an override]:
   global RRC scale (0.4, 1.0), local RRC scale (0.05, 0.4), bicubic; locals =
   6×6 tokens at patch16. Photometric = the house lejepa symmetric family on EVERY
   view (jitter .8/.8/.8/.2 p.8, grayscale p.2, blur k7 σ(.1,2), solarize p.2,
   hflip) — DINO's asymmetric global1/global2 blur/solarize split is NOT copied
   [DECLARED: symmetric, our lane's family].
3. Trunks: timm vit_{small,base,large}_patch16_224 @224, dynamic_img_size=True
   (pos-embed interpolation for 96-px locals; @224 forward unchanged — smoke
   asserts), drop_path 0.1. Globals and locals forward through the same trunk (two
   forwards per step, embeddings concatenated).
4. Eval unchanged: online probe every epoch; landing chains (extract o8+L → probe +
   battery + instruments) per cell; E25 transfer battery on bests; guillotine per
   arch; refs Lightly 64.0 (S), paper number (L, transcribe-before-launch).

**B. Our lane (floorssl cells — λ appears NOWHERE here)**
5. Loss = w_inv·inv + w_floor·cond_z + h_lamb·cond_h (unchanged three-term).
6. inv [RULED — Berker: "proposed inv is agreed"]: view-to-mean over all 10 views,
   mean_v‖z_v − z̄‖² — the LeJEPA structure; exactly ∝ the lineage all-pairs at
   fixed view set (all-pairs = 2V/(V−1) · view-to-mean), constant absorbed by the
   dose procedure; O(V) at V=10; locals symmetric.
7. Conditioner stream [RULED — Berker: "conditioner stream should use mean of
   all"]: per-image view-mean over ALL 10 views at both taps (the fixed-point
   object inv contracts toward; the 8-local average knocks per-view crop noise
   down ~1/√8, and the noise floor anneals as inv contracts). Fallback ON RECORD
   (his "if you think it is too much" delegation, held in reserve): globals-only
   mean, to invoke if the ep10 h-forces read crop-noise-dominated. Fresh n = bs =
   128 rows/step.
8. Estimator table [RULED — Berker: "for L make the slice ratio 384 for L
   (cosmetics)"]; z side fixed by the byte-identical head (out=256 at every arch).
   q = the ring length: how many PAST steps' detached view-mean batches concatenate
   with the fresh 128 rows before the covariance read (n_eff = (q+1)·128; gradient
   through the fresh rows only):

   | arch | D | h d′ | h ring q | h n_eff (n/d′) | h d_draw | z d′ / q |
   |---|---|---|---|---|---|---|
   | S | 384 | 128 | 3 | 512 (4.00) | 128 | 128 / 3 |
   | B | 768 | 256 | 7 | 1024 (4.00) | 256 | 128 / 3 |
   | L | 1024 | 384 | 11 | 1536 (4.00) | 384 | 128 / 3 |

   Ring knob splits per-tap (h_queue_steps; null = shared legacy queue_steps,
   byte-identical default). S row ≡ the certified vm4 anatomy unchanged.
9. Ring staleness → **OAS fallback SANCTIONED at B/L [RULED — Berker: "with B and L
   we will be careful because ratios change so we can fallback to oas in case
   features move too fast for stabilization"]:** the longer rings hold rows ≤7/≤11
   steps old (E24 staleness lesson); watched at the ep10 metrics+forces check + the
   ρ̂ meter; fallback = `floor_shrink=oas`, no ring (the D-075 pattern at scale).
   voas OAS-vs-ring A/B (~08-08) is the standing information point.
10. Doses per the card procedure, measured under THIS aug: pilot past warmup
    (~1 GPU-h) → per-term trunk g + T (warm ring) → w_i = s*_i·T/g_i at
    s* = (.63, .34, .03) → ep10 confirm ± one correction, **read on the SAVED ep10
    state — ep10 added to the checkpoint cadence (extra_cadence=[10]; Berker: "be
    mindful of ep10 checkpoint")**. S-band steering unchanged: Ω_h(cls) ∈
    [.35, .50], ep25/ep50 rule, channel offset calibrated at voas landing.
11. Head/opt (unchanged): ladder K2/hidden-2048/out-256, in_dim = trunk width
    (384/768/1024), mlp_wd .05; lr 1e-3, wd 5e-2, warmup 10 ep, cosine eta_min
    1e-5, grad_clip 1.0, bf16; share_log_every=1, eval_every=1.
12. In-training Ω instrument [RESOLVED at implementation — supersedes the earlier
    globals-only proposal on a measured fact]: **audit_v1 = orbit_stack (0.08, 1.0)
    @frame size = EXACTLY the voas train/instrument stack AND the o8 landing
    stack** — so the steering channel keeps the voas-IDENTICAL construction (fixed
    batch, V=4, audit_v1, eval-mode) and the R6 band + the voas offset transfer
    directly. Berker's large-crop question ("maybe we should deviate … only
    consider large crops like 0.7 and above? … or we launch for both") is resolved
    MEASUREMENT-side, no aug fork and no second launch: three channels logged every
    epoch in every multicrop run — `omega_h/omega_z` (audit_v1, the steering +
    continuity channel), `orbit/h_own_omega` (the run's 2 training globals),
    `orbit/h_gentle_omega` (fixed V=4 @ RRC (0.7, 1.0)). Same 128 images in every
    channel (shared seed-0 loader). Landing numbers stay canonical. Online probe:
    cls of all 10 views, labels ×10 (both trainers' existing convention).

**C. Control lane (lejepa cells)**
13. Loss = λ·SIGReg + (1−λ)·inv with λ = 0.05 [RULED — the paper recommendation;
    the only place λ exists]. inv = view-to-mean over all 10 (their minimal-script
    structure verbatim); SIGReg per-view over the batch, averaged over views+slices
    (port structure verbatim).
14. Gap-fill order [RULED — "recheck the paper and the repo to fill the missing
    gaps otherwise match ours"]: paper > repo > house. **Recheck EXECUTED at D-082;
    final control dials:** λ = .05 (paper recommendation); lr 5e-4 + final lr =
    lr/1000 (eta_min 5e-7) + wd 5e-2 + bf16 (repo README large-scale values);
    warmup 1 ep, proj MLP [2048, 2048, 16] BN, embed 512, NO grad clip, drop_path
    0.1 (repo minimal / port lineage). Head-shape note (his question, confirmed
    fine): our K2/2048 ladder ≡ their MLP in depth + hidden width; residues =
    their 512-embed stage (the z.embed tap, D-003 — "we dont insert linear",
    correct: our lane never had it) and out-dim (16 vs 256, "can differ it is
    fine"). No share logger on the control (lejepa declares no PULL_W); its Ω
    comes from landing chains at cadence ckpts incl. ep10.
15. **Control cell LAUNCHED at ViT-S (D-082)** — "same architecture" read: the
    lejepa control matches the launching runs' arch, and the Lightly 64.0 S
    reference is not at our multicrop frame; chain e27lej (8×8h, selftest-gated).
    The RULED ViT-B control (ruling 3) stays owed at the B phase.

**D. Compute (stance = ruling 6, operationalized)**
16. V=10 multicrop ≈ 0.88× the V=4-global per-image cost (690 vs 788 token-units):
    S fits a single 80 GB card like voas; B expected to fit flat; L expected to fit
    WITH grad_ckpt — smoke asserts each before launch.
17. Wall-clock (single GPU, H100, 28 workers — pilot pins): S ~25–35 min/ep →
    2–2.5 d; B ~1.5–2 h/ep → 6.5–8.5 d; L (ckpt) ~7–9 h/ep → ~30 d. **S pair
    LAUNCHED single-GPU per the ruling ("we launch s and then you start building
    b/l with smokes")**; B layout = Berker's call at D-079b (1 GPU ~1 wk vs 4-GPU
    DDP ~2 d); L single-GPU is ~a month → the mitigation clause points at DDP-8
    (~4–5 d) unless the month is acceptable. B/L machinery (frames, lejepa
    multicrop path, DDP option — global-128 sharding, gathered conditioner stream,
    shared slice draw, SyncBN, parity gate — and the OAS fallback wiring) is built
    + smoked DURING the S days; DDP is NOT used at S.
18. Workers: 28/GPU on H100 (224c/8g), 32 on gpu238 (A100 fallback ≈ 2× wall).
    The 8-H100 in1k budget is loose under this stance (S phase: 2 cells + voas =
    3 GPUs).

**E. Build before the S launch (~half day incl. smoke)**
19. ViewsDataset multicrop mode (Vg/Vl/local_size/scales; batch = globals
    [N,2,3,224,224] + locals [N,8,3,96,96]; trainer to_device is tuple-safe).
20. floorssl: view-to-mean inv; per-tap queue knob; d_draw = d′_h; in_dim/probe_dim
    from trunk.num_features (byte-identical at 384).
21. lejepa method: multicrop path + the λ-convex form (port pieces exist).
22. Frames in1k_vitb16/in1k_vitl16 + PROTOCOL §2 rows; D-079a row.
23. Smokes: 2-ep subset S smoke (multicrop path, share logger, resume/ring
    re-warm); @224 dynamic_img_size parity assert; then the S pilot → doses.

## S-wave launch log (2026-08-06, evening — RAW record)

Both pilots: selftest ALL PASS (dynamic pos-embed @224 max|diff| = 0.000e+00; ring
refactor == D-064 inline; inv identity exact). Doses by the pre-registered formula
w_i = s\*_i·9.0/g_i on each pilot's ep1 [share] g's (fresh formation state under the
cell's own aug, incumbent-weight measurement):

| cell(s) | pilot ep1 g (inv / z / h) | w (inv / floor / h_lamb) | w·g check | chains |
|---|---|---|---|---|
| e27smc + e27smcb | .505 / .277 / .458 | 11.23 / 11.05 / 0.590 | 5.67/3.06/0.27 = 9.0 | 63127172–87 |
| e27slg + e27slgb | .248 / .159 / .450 | 22.86 / 19.25 / 0.600 | 5.67/3.06/0.27 = 9.0 | 63127547–62 |
| e27lej (control) | — (λ = .05 fixed) | — | — | 63118686–93 |

Pilot ep1 instrument lines (RAW; three channels live from birth): mc — Ω_h aud 1.063,
own-globals .698, gentle .597, Λ 1.065; lg — aud 1.214, own .554, gentle .286, Λ .938.
Raw observation for the joint read: g_inv(lg) = .248 ≈ half of g_inv(mc) = .505 — large
locals are easier to align per unit weight, so the formula doses lg's inv ~2× higher.
ep10 confirm watchers armed on all four floorssl cells; band cells byte-identical to
anchors until/unless ep25 steering fires.

## S vs the incumbent IN-1k cells — RAW comparison (2026-08-07, Berker's question)

Asked: *"in theory d256vm4 and e27smc should be the same except the view count am i wrong?
maybe the dosing was the issue."* Numbers only; no reading attached.

**(a) What actually differs between `d256vm4` and `e27smc`** (from the stored ckpt cfgs).
IDENTICAL: head architecture (`head_layers=2, width=None` ≡ `floorssl_head(norm="bn")`
byte-for-byte), expander 2048→256, `z/h_floor_batch=view_mean`, d′=128 both taps,
`queue_steps=3`, `head_norm=bn`, lr 1e-3, wd .05, warmup 10, eta_min 1e-5, drop_path .1,
bs 128, 100 ep. `mlp_wd` reads `None` vs `0.05` but resolves to the same 0.05 on both groups.
DIFFERENT: **(i) the aug family**, not merely the view count — vm4 is V=4 uniform
RRC (0.08, 1.0) @224; e27smc is 2 globals @224 RRC (0.3, 1.0) + 8 locals @**96** RRC
(0.05, 0.3), so 8 of 10 views are small low-res crops; **(ii) the inv functional** —
all-pairs MSE vs view-to-mean (related by (V−1)/2V: .375 at V=4, .450 at V=10, absorbed by
dosing); **(iii) the doses** — 26.9/129.2/1.679 vs 11.23/11.05/0.590.

**(b) Probe accuracy at matched epochs** (in-training monitor; E12-T9(i) bias caveat rides).

| ep | d256vm4 | e24voas | e27smc | e27smcb | e27slg | e27slgb | e27lej |
|---|---|---|---|---|---|---|---|
| 16 | .4188 | .3880 | .3558 | .3340 | .3445 | .3437 | .3002 |
| 18 | .4329 | .3991 | .3754 | .3375 | .3577 | .3554 | .3163 |
| 22 | .4438 | .4123 | .3249 | .3569 | — | — | .3344 |
| 24 | .4484 | .4180 | .3465 | — | — | — | .3477 |

Δ vs vm4 at ep18: voas −3.4 · smc −5.8 · slg −7.5 · slgb −7.8 · smcb −9.5 · lej −11.7 pts.
`e27smc` peaked .3754 (ep19) and gave back to .3249 by ep22 while its byte-identical twin
`e27smcb` rose .3365 → .3632; the pair's ep0 share lines match to the digit, so that spread is
run-to-run nondeterminism at one seed (the only replicate information this program has).

**(c) Realized forces on ONE warm-ring instrument** (`experiments/pull.py`, 6 batches, warm
rows b3–b5 only; `results/diag/pull.csv`, job 63181443). vm4/voas at ep25, S cells at ep10 —
the S cells have no ep25 checkpoint yet, so this is state-matched only within each pair.

| run | s_inv | s_cond_z | s_cond_h | Σ w·g | cos(g_inv, g_cond_z) |
|---|---|---|---|---|---|
| `d256vm4` @ep25 | .628 | .363 | **.009** | **9.98** | +0.139 |
| `e24voas` @ep25 | .719 | .251 | .030 | **7.10** | +0.366 |
| `e27slg` @ep10 | .675 | .308 | .017 | **1.29** | +0.112 |
| `e27smc` @ep10 | **.453** | **.521** | .026 | **1.24** | **−0.239** |

s\* = (.63, .34, .03). In-training `[share]` logs agree at matched epochs (voas ep15–25 holds
.68–.72/.26–.29/.03 with Σ w·g ≈ 7–8; slg ep10–18 holds .63–.75/.24–.35/.015–.019 with
Σ w·g ≈ 1.2–1.4; smc ep16–21 sits at .41–.51/.46–.56 with Σ w·g ≈ 0.9–1.1).

Three raw observations for the joint read: **(1)** vm4's realized profile matches s\* on the
first two coordinates but its h-share is .009, a third of the target; **(2)** the four S cells
run at Σ w·g ≈ 1.2–1.3 against vm4's 9.98 and voas's 7.10 — the dose formula pinned Σ w·g = 9.0
at the *pilot's ep1* g's, and the g's then fell ~10× by ep5 (voas: 168 → 42 → 8.8 → ~7.5 plateau;
smc: 54 → 7.2 → 1.6 → ~1.0 plateau); **(3)** `cos(g_inv, g_cond_z)` is **negative for the mc
cell alone** (−0.24 warm) while vm4, voas and lg are all positive (+0.11 … +0.37). Warm rows —
this is not the stale-ring artifact E24-T2-CLOSE retracted. **Caveat that must ride any dosing
reading: AdamW normalizes by √v̂, so a uniform rescale of the loss largely cancels in the
update; what a smaller Σ w·g does change is the strength of decoupled weight decay relative to
the gradient, and the share profile, which does not cancel.**

## (d) Why the mc pair is unstable — the term curves and the conditioner anatomy (RAW, 2026-08-07)

Berker: *"analyze the h_moment_kl logs from wandb curves. those two particular runs are very
unstable."*

**Per-step term levels by epoch window** (wandb history, median over the window):

| run | h_cond ep2→ep23 | h_cond p99/med | z_cond ep2→ep23 | inv (med) | grad_norm (med) |
|---|---|---|---|---|---|
| `d256vm4` | .485 → **.320** monotone | 1.06 | .324 → **.183** | .237 | 10.1 |
| `e24voas` | .199 → **.125** monotone | 1.20 | .216 → **.130** | .258 | 7.2 |
| `e27slg` | .273 → **.190** monotone | 1.05 | .300 → **.181** | .050 | 1.49 |
| `e27slgb` | .273 → **.190** monotone | 1.05 | .304 → **.181** | .050 | 1.48 |
| `e27smc` | .288 → .242 → .294 → .239 → **.276** *(oscillating, no descent)* | **1.6–1.9** | .786 → **.635** | .100 | 1.52 |
| `e27smcb` | .285 → .253 → .270 → .259 → **.258** *(same)* | **1.6–1.9** | .792 → **.628** | .100 | 1.51 |

The mc pair is the only cell whose h-conditioner never converges — it wanders in a .24–.29 band
with within-epoch spread 60–90% above its median, where every other cell (including its lg
sibling, identical except local crop scale) descends monotonically with a 5% spread. Its
z-conditioner sits at **.63, 3.5× its lg twin's .18**, flat from ep2. Its `inv` is also 2×
worse (.100 vs .050). All three terms are simultaneously unsatisfied.

**Conditioner anatomy at ep10** (one batch, fresh 128 rows, d′=128 — no ring, so absolute KL
is inflated for both cells equally; the estimator-robust column is trace/d, which no rank
deficiency touches):

| cell | stream | trace/d | cone/d | −logdet/d | KL |
|---|---|---|---|---|---|
| smc | z pooled | .269 | .015 | +1.80 | .542 |
| smc | **z mean(all) — what it trains on** | **.170** | .015 | +3.20 | 1.193 |
| smc | z mean(globals) | .302 | .017 | +2.55 | .934 |
| slg | z pooled | .585 | .006 | +1.23 | .409 |
| slg | **z mean(all) — what it trains on** | **.531** | .007 | +1.86 | .700 |
| slg | z mean(globals) | .488 | .006 | +1.99 | .741 |
| smc | h pooled | **1.509** | .059 | −0.087 | .240 |
| smc | h mean(all) | **.670** | .065 | +1.75 | .744 |
| slg | h pooled | .952 | .037 | +0.64 | .312 |
| slg | h mean(all) | .808 | .037 | +1.51 | .677 |

Three raw readings. **(1)** In both cells the KL is dominated by **−logdet**: the conditioner is
fighting *contraction*, not over-variance. **(2)** The mc cell's training stream carries
**trace/d = .17 against a target of 1.0** — the view-mean z is contracted 6× — versus .53 for
lg. Two compounding sources: its pooled z is already contracted (.269 vs .585), and averaging
over the 10 views costs it a further 37% (lg loses 9%). **(3)** At h the mc cell swings from
**over-dispersed pooled (1.51) to contracted view-mean (.67)** — a 2.25× swing with the choice
of stream, against 1.18× for lg. Switching mc to a globals-only stream would move its z stream
from .170 to .302 and would move lg's the wrong way (.531 → .488).

This is the evidence the card's pre-registered reserve names: *"globals-only mean, to invoke if
the ep10 h-forces read crop-noise-dominated"* (§Recipe v2 item 7).

### (d.2) READING TRAP — the raw `inv` panel is not comparable across the two lanes

Berker (2026-08-07): *"if you just compare the inv panels from wandb, they start higher, then
reduce to around 0.35 after 50-100k steps. but all new runs start super aligned. why?"*
Three effects stack, and **none of them is better alignment**:

1. **Different functional.** The multicrop branch logs the view-to-mean form `W·(V−1)/V`; the
   V=4 lane logs all-pairs `2W`. Identical geometry ⇒ the multicrop number is **2.22× smaller
   at V=10**. (The dose procedure is immune — it reads gradient norms, not loss values — but
   the CURVES are not.)
2. **`inv` is not scale-free**: it is an absolute squared distance in z, so it scales with z².
   The mc cell's pooled z variance is trace/d = **.269 against vm4's .651**.
3. **Remove both and the ordering inverts.** Decomposing `pooled = B+W`, `viewmean = B+W/V`:

| run | W (within-image) | B (between-image) | **W/B** | logged inv (200k) | predicted |
|---|---|---|---|---|---|
| `d256vm4` | .139 | .512 | **0.27** | .272 | 2W = .278 |
| `e24voas` | .136 | .712 | **0.19** | .284 | 2W = .272 |
| `e27smc` | .110 | .159 | **0.69** | .098 | .9W = .099 |
| `e27slg` | .060 | .525 | **0.114** | .046 | .9W = .054 |

Predicted-vs-logged agrees to ~2–5% on all four, so the decomposition is validated by an
independent route. **The mc cell's z clouds are 2.5× THICKER than vm4's, not more aligned** —
its small `inv` is a symptom of the conditioner losing the scale (z contracting toward the
origin), consistent with its Ω_h .62–.65, its −logdet-dominated KL, and its stream at trace
.170. `e27slg` is the only cell genuinely more aligned (W/B .114), its near-global locals
agreeing by construction. **Convention for any future cross-lane `inv` panel: convert to one
functional AND normalise by the z scale, or plot W/B instead.**

## (e) S-wave RELAUNCH log — D-084 (2026-08-07)

Wave 1 cancelled at ep19–25 (`e27smc` .3564 · `e27smcb` .3483 · `e27slg` .3576 ·
`e27slgb` .3641). Two changes, both measured, everything else byte-matched:

1. **`cond_stream=globals`** — the conditioner reads the mean over the 2 globals; `inv`
   keeps every view. Additive knob, byte-gated pre-vs-post on three legacy paths
   (uniform-V voas cfg · multicrop key-absent · multicrop `cond_stream=all`): all PASS.
   The new path is real — mc @ep10 `moment_kl` .717 → .459 at unchanged weights, `inv`
   bit-identical.
2. **Doses re-derived at the ep10 held state under that stream** (warm ring, D-072),
   `w_i = s*_i·T/g_i`, s\* = (.63, .34, .03), **T = 10.0** (= vm4's realized total at ep25,
   9.98) instead of wave 1's T = 9.0 measured at a 2-epoch pilot's ep1.

| cell | g_inv | g_cond_z | g_cond_h | w_inv | w_floor | h_lamb | (wave 1) |
|---|---|---|---|---|---|---|---|
| mc | .0506 | .0501 | .0615 | **124.50** | **67.83** | **4.878** | 11.23/11.05/0.590 |
| lg | .0382 | .0396 | .0427 | **164.87** | **85.92** | **7.032** | 22.86/19.25/0.600 |

Cells `e27smc2` · `e27smcb2` · `e27slg2` · `e27slgb2`, jobs 63181847–78, 8×8 h singleton
segments each on gpu100 (the H100 fleet came back up — 13 nodes — so D-080's ≤8 in1k budget
applies, currently 4 in use). `e27lej` kept running (control is method-independent and the
comparison is at matched epochs); `e24voas` untouched at ep75.

**Standing caveat:** globals-only fixes the stream, not the opposition —
`cos(g_inv, g_cond_z)` for mc stays **−0.27 warm** under the new stream where lg, vm4 and
voas are all positive (+.11 … +.37). If the mc pair misbehaves again, the next lever is its
local crop scale (0.05, 0.3), not the dose.

## (f) Wave 2 FAILED — the shape diagnostic, and wave 3 (D-085, 2026-08-07)

Wave 2 was cancelled at ~0.8 ep. **The diagnostic is `inv`'s SHAPE, not any term's level.**

| run | w_floor/w_inv | inv @200 | @500 | @1k | @2k | @4k | @7k | h_kl @10k |
|---|---|---|---|---|---|---|---|---|
| `d256vm4` | **4.80** | — | .276 | .372 | .457 | .479 | .456 | .949 |
| `e27s_pilot` (mc) | **2.32** | .106 | .130 | .175 | .223 | .248 | .242 | .659 |
| `e27slg_pilot` | **2.32** | .102 | .124 | .159 | .201 | .203 | .177 | .443 |
| `e27smc` (wave 1) | 0.98 | — | .093 | .099 | .108 | .114 | .113 | .581 |
| **`e27smc2` (wave 2)** | **0.545** | **.062** | **.060** | **.058** | **.058** | **.058** | **.057** | .545 |

Every healthy run's `inv` **rises** through formation — the conditioner opens the space,
invariance fights back, they settle. Wave 2's never rose: pinned from step 200, with
`moment_kl` (.62) and `h_moment_kl` (.57) flattening with it from ~2k. Nothing was moving.
The ordering is monotone in `w_floor/w_inv`: 4.80 → 2.32 → 0.98 → 0.545.

**Two errors on the record.** (i) Wave 2 changed the conditioner stream AND the doses in one
launch — not separable. (ii) The doses were derived from forces at **wave-1's ep10 state, which
the same analysis had just shown to be pathological**: a contracted z gives a small `g_cond_z`,
so share-targeting to .34 handed inv 1.8× the conditioner's nominal weight. Dosing to a share
target off a sick state reproduces the sickness. Also corrected: an aggregate median comparison
(wave 2 .58 vs wave 1 .27) was unfair — first epoch against a whole-run distribution; **at
matched steps wave 2 was lower on all three terms**, which is why level is not the diagnostic.

**Wave 3 (running):** doses **21.4 / 49.6 / 1.89** (the incumbent voas/pilot weights — the
ratio with the demonstrated healthy shape on BOTH aug versions) with **`cond_stream=globals`
as the ONLY change from the pilots**, isolating the stream question at last. Cells
`e27smc3` · `e27smcb3` · `e27slg3` · `e27slgb3`, all four on **A100-80GB** (`gpu` partition,
node gpu238) rather than gpu100: 22 H100s were free but STRANDED — another user held 26
single-GPU jobs at 56 CPU + 500 GB each, so four of them saturate a node's CPU and RAM while
leaving 4 H100s idle and unschedulable (0 free CPU, ~2 GB free RAM on gpu265/268/270/275/276;
24 h wall limit, ~1 h elapsed). D-080 sanctions A100 for the in1k program, and SLURM's native
partition OR is disabled here, so placement is one partition per submit (`PART`/`CONSTR` env on
`slurm/e27_wave_sN.sh`). **Read at ~2–4k steps: if `inv` rises off its
start, the shape is healthy; if it pins again, the stream change is implicated and the next
lever is mc's local crop scale.**

## (g) Wave 3 read — the aug axis is DEAD (D-086, 2026-08-07). DECISION OWED.

**mc-vs-lg separation at matched steps** — this is the S wave's whole question (D-081):

| term | wave 1 (all-views stream) | wave 3 (`cond_stream=globals`) | ratio |
|---|---|---|---|
| `inv` | .0142 → .0313 | **.00005 → .00046** | ~60× smaller |
| `moment_kl` | .160 → .521 | **.00041 → .00138** | ~400× smaller |
| `h_moment_kl` | .073 → .194 | .0011 → .0205 | ~20× smaller |

All four wave-3 cells agree to three decimals on every term. **The two aug versions have
become the same experiment.**

**Mechanism (hypothesis, consistent with the data):** the conditioner now reads only the 2
globals, whose `global_scale` is identical across versions, so both conditioner terms are
aug-blind *by construction*. Nothing then constrains the 8 locals except `inv` — and `inv` can
satisfy itself by making the encoder INSENSITIVE to local content rather than by aligning it.
The locals become decorative: full compute cost, no signal. This is the f5 lesson's
"absorbable = information-free compliance" (cited in D-043).

**Secondary symptom — the spiking Berker asked about:** `h_moment_kl` descends to .576 at 2k
then **turns and rises to .690 by 10k**, where every reference keeps descending through 15k
(vm4 1.34 → .67, voas .79 → .27, wave-1 smc 1.48 → .40). Spike ratio p99/med is 2.1–2.3
(better than wave 1's 5.4–7.9) but the trend has inverted. Consistent with the conditioner
pinning a stream it only partly observes while `inv` drags the representation toward a mean
dominated by 8 unconstrained views.

**Note the shape criterion did NOT catch this.** Wave 3's `inv` rises (.102 → .160), which is
what §(f) said to look for. The criterion was necessary and not sufficient: a run can have the
healthy *shape* and still be measuring nothing.

**Options for the joint call (nothing decided; wave 3 is burning ~16 A100-h/night and my
recommendation is to STOP it):**

1. **Revert to wave-1 config verbatim** and treat the mc instability as a dose/lr question —
   Berker's original framing, which the evidence no longer contradicts. Wave 1 is the only
   configuration in which the aug axis was measurably alive.
2. **Keep the all-views stream, change the aug** — soften mc's locals from (0.05, 0.3) toward
   lg's (0.7, 1.0). Attacks the thick-cloud cause directly rather than the conditioner's view
   of it, and keeps the axis alive because the conditioner still sees the locals.
3. **Conditioner on all views, `inv` on globals-only** — the mirror of what was tried, and the
   one combination not yet examined.
4. **[PROPOSED — Fable 2026-08-08, the takeover recommendation] `cond_stream=grouped` + doses
   at the certified LEVEL:** the conditioner reads TWO scale-homogeneous streams — per-image
   mean over the 2 globals AND per-image mean over the 8 locals — one KL each, averaged, own
   ring per group (fresh n = bs each, q=3, d′=128). Why this and not 1–3: the §(d) anatomy
   shows the all-views mean is a malformed conditioning object (locals are deliberately
   partial-content views, NOT exchangeable identity estimates — mixing them into one mean gave
   mc trace/d .17 and the unique inv↔cond opposition cos −.27), while D-086 shows blinding the
   conditioner to locals kills the axis; grouped restores exchangeability WITHIN each stream
   while keeping locals conditioner-constrained. It is the structural analogue of per-view
   SIGReg — under which the e27lej control is HEALTHY on this exact aug (ep60 .4378) — carried
   into our view-mean anatomy. Option 1's dose-level fix is subsumed (wave 4 doses target
   Σw·g ≈ 10 at the warm plateau, not a transient ep1 — the wave-1 Σ≈1.2 error corrected at
   its root); option 2 stays the named next lever if mc still fails (it changes the aug family
   E27 exists to test — last resort); option 3 keeps the malformed mean object. Procedure:
   3-ep pilots per version at the incumbent healthy-shape weights (21.4/49.6/1.89) under the
   grouped stream — the pilots ARE the smoke — derive w = s\*·10.0/g at the ep3 warm-plateau
   g's, then the 2×2 relaunches ONLY if the gate below passes. Code staged
   (`cond_stream=grouped` in floorssl.py + per-group ring snapshots in the share logger +
   selftest §6; legacy/all/globals paths untouched).

**Whatever is chosen: smoke it for 2 epochs and check the mc-vs-lg separation BEFORE launching
chains.** That check costs ~1 GPU-hour and would have caught this twice. **Wave-4 gate,
pre-registered (all four must hold on the pilots before any chain): (a) mc-vs-lg separation on
inv + both cond terms at wave-1 order of magnitude (≥10× above wave 3's); (b) inv RISES
through formation on both versions; (c) mc's h-cond spread p99/med < 1.5 (vs wave 1's
1.6–1.9); (d) both group streams' trace/d > .30 for mc at ep3 (vs the .17 pathology). Any
fail → STOP and report; the pre-named next lever is option 2 (mc local scale).**

**STOPPED 2026-08-07 on Berker's word** ("cancel them"), all four cells at ~10k steps / ~1.5 h
on gpu238. Final `h_moment_kl` (last-5% median): **`e27smc3` .826 · `e27slg3` .754** — the
upward trend held to the end (.576 at 2k → .690 at 10k → .78–.83 at stop), against every
reference still descending. Nothing else was touched: `e24voas` ep78 and the `e27lej` control
ep29 continue. Design decision deferred to the joint session 2026-08-08.

## (h) The v4→v10 diagnostic sweep (D-087, 2026-08-08 — Berker's breadth directive; supersedes the §(g) single-option question)

Berker: two successful anchors (vm4, voas); V=10 exists to buy performance via compute +
aug diversity; large-crop cells failed too in wave 3, so harsh locals are NOT established
as the cause; "my initial attempt was to scale the thing from v=4 to v=10"; breadth, no
H100 cap, 4–5 ep cells, joint takeaway, survivors continue. 5 ep CONFIRMED sufficient for
the health/separation read (every wave-1/2/3 discriminating signal appeared by ep5);
probe ordering at ep5 is indicative only.

**12 cells, 5 ep each, single H100, selftest-gated, fixed known-good doses** (uniform-V
lane = vm4's w 26.9/129.2/1.679; multicrop lane = voas's w 21.4/49.6/1.89 — no derived
doses this round). Jobs 63200567–78, `slurm/e27_diag.sbatch`:

| cell | axis isolated | config delta |
|---|---|---|
| e27dv4 | re-anchor (vm4 on today's code + Ω channels) | V=4 uniform (0.08,1)@224 |
| e27dv6 | pure view count, step 1 | V=6 uniform |
| e27dv10u | pure view count → 10 (the original scaling intent) | V=10 uniform |
| e27dmc | wave-1 mc at the HEALTHY dose level (the dose framing) | mc aug, all-views stream |
| e27dlg | its large-locals twin | lg aug, all-views stream |
| e27dmcg | grouped stream on mc (§(g) opt 4) | cond_stream=grouped |
| e27dlgg | grouped on lg | cond_stream=grouped |
| e27dmid | resolution WITHOUT harshness | locals @96, scale (0.3,1.0) |
| e27dhf | harshness WITHOUT resolution | locals @224, scale (0.05,0.3) |
| e27dpv | the healthy control's anatomy in our loss | cond_stream=perview + OAS, no ring |
| e27dmco | is the ring implicated on mc's stream | mc + OAS, no ring |
| e27dmc48 | dose-ratio probe (D-085 monotone table) | mc, w_floor 102.7 (ratio 4.8) |

**Read protocol per cell (the survivors' criteria, pre-registered):** inv SHAPE (rises
through formation) · cond_z/cond_h descent + within-epoch spread p99/med · realized Σw·g
and shares · mc-vs-lg separation where paired · Ω channels · stream trace/d at ep5 ·
probe@5 (indicative only, E12-T9 caveat). Takeaway = JOINT on the assembled table;
survivors continue to full runs, then B/L + controls per the program.

## Gates

- **D-079a: USER-APPROVED 2026-08-06 (Berker: "we launch s and then you start
  building b/l with smokes") — the S pair is OPEN on Recipe v2.** The vitb16/vitl16
  PROTOCOL rows land with the B/L build (owed pre-D-079b).
- **D-079b:** B opens on the S joint read (winner placement + health) + the B
  compute-layout call (single vs DDP-4).
- **D-079c:** L opens on the B joint read (+ grad_ckpt/DDP-8 decision — single-GPU
  L is ~a month).

## AGREED TAKEAWAY

*(joint only — empty until discussed)*
