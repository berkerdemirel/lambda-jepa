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

### (h.1) Sweep results — RAW (2026-08-08 night; all 12 landed, dv10u to ep4 of 5 on its 8h wall)

| cell | shares ep4 (inv/z/h) | Σw·g | Ω_h aud | probe@5 | inv .5k→2k→10k→40k | z 2k→40k | h 2k→40k | h p99/med |
|---|---|---|---|---|---|---|---|---|
| e27dv4 | .588/.402/.010 | 15.2 | .548 | .2989 | .288→.467→.424→.348 ↑ | .955→.242 | 1.25→.421 | 1.24 |
| e27dv6 | .535/.455/.010 | 13.4 | .492 | .3327 | .269→.467→.424→.318 ↑ | 1.07→.220 | 1.50→.384 | 1.28 |
| e27dv10u | .593/.398/.010 | 14.7 | .439 | **.3340**@ep4 | .266→.454→.434→.304 ↑ | 1.18→.210 | 1.87→.362 | 1.33 |
| e27dmc | .437/.548/.015 | 9.5 | .655 | .3067 | .132→.222→.230→.211 ↑ | 1.34→.379 | 1.41→.229 | 1.36 |
| e27dlg | .681/.292/.027 | 3.8 | .732 | .2350 | .125→.205→.158→.098 ↑↓ | .937→.124 | 1.40→.168 | 1.23 |
| e27dmcg | .442/.548/.010 | 13.2 | .699 | .3004 | .146→.224→.199→.184 ↑ | .916→.343 | .952→.226 | 1.27 |
| e27dlgg | .639/.336/.025 | 3.3 | .779 | .2491 | .142→.206→.149→.094 ↑↓ | .751→.123 | 1.09→.163 | 1.25 |
| e27dmid | .590/.385/.026 | 4.4 | .591 | .3145 | .126→.215→.203→.136 ↑ | 1.13→.182 | 1.47→.197 | 1.27 |
| e27dhf | .458/.522/.020 | 7.3 | .586 | .3315 | .125→.222→.223→.174 ↑ | 1.27→.301 | 1.53→.223 | 1.36 |
| e27dpv | .654/.319/.027 | 5.3 | .591 | .3251 | .466→.445→.343→.258 ↓ (OAS-immediate) | .140→.083 | .176→.060 | 1.27 |
| e27dmco | .576/.403/.021 | 18.2 | .527 | .3231 | .489→.604→.460→.331 ↑ | .628→.165 | .954→.125 | 1.37 |
| e27dmc48 | .489/.501/.010 | 15.2 | .615 | .2957 | .164→.392→.409→.330 ↑ | 1.07→.244 | 1.54→.296 | 1.44 |

**Separations (mc vs lg), matched steps:** all-views @10k inv .072 / moment .41 / h .25;
@40k .113/.25/.062. Grouped @10k .050/.32/.18. Wave-1 range .014–.031 / .16–.52; wave-3
.00005–.0005 / .0004–.0014 — **the axis is alive in BOTH streams at proper doses,
~1000× wave 3.** Ω direction: Ω(lg) > Ω(mc) in both streams (.732/.779 vs .655/.699) —
the P-E27-2 aug-axis direction; the nearer-band version (mc) also wins the probe.
h-spread: EVERY cell 1.23–1.44 vs wave-1 mc's 1.6–1.9 — the oscillation is gone at
healthy dose levels, including on wave-1-mc's exact stream (e27dmc). e27dv4 re-anchor
reproduces vm4's signature (inv .467@2k vs .457; spread 1.24) — code path validated.
Same-day context: the voas landing (OAS, V=4) lost the A/B to vm4's ring by −4.2 linear /
−7.6 kNN at land — ep5 probe leads of the OAS cells (dpv/dmco) must be read against that.

## (i) The FINAL S wave — 5×100 ep on the adopted frame (D-088, 2026-08-08 night)

Berker's constraints after the sweep read: V=10 constant (the adopted in1k convention —
CHECKED: DINO repo default 2g+8l, DINOv2 2+8, LeJEPA V_g=2/V_l=8; SwAV 2+6 the
exception); aug family constant (changing it risks less diversity + unmatched compute);
≤1 OAS variant ("dont get fixated on oas … vm4 is the clear winner from the curves");
plus the uniform lane restored → five runs. Everything byte-matched except the delta:

| run | delta | aug | stream / estimator | w | segs |
|---|---|---|---|---|---|
| e27mc | — (main line) | mc 2g+8l | all-views mean / ring q3 | 21.4/49.6/1.89 | 12 |
| e27oas | estimator | mc 2g+8l | all-views / OAS no-ring | same | 12 |
| e27grp | stream | mc 2g+8l | grouped / ring q3 | same | 12 |
| e27pv | stream | mc 2g+8l | per-view / ring q3 | same | 12 |
| e27v10u | the lane | V=10 uniform (0.08,1)@224 | all-views (legacy) / ring q3 | 26.9/129.2/1.679 | 21 |

Jobs 63202191–63202259; ep10 ckpt + health check on all five (combined watcher); landing
chains at ep100; band-steering cells RETIRED from this wave (steering dissolved into the
diagnostic program; the R6/Ω read is observational — uniform lane in-band at ep5, mc
family above). e27lej control continues (ep60+ .4378). Read at land: probe/kNN vs vm4
.6416/.6554-l2 + Lightly 64.0 + e27lej; stream/estimator verdicts feed the B/L design
(D-079b). 5 + control = 6 of the 8 in1k budget.

**Interruption note (2026-08-09):** a cluster NFS outage 02:32–03:25 killed all running
segments and drained the singleton chains (follow-ups failed at 0 s, no output files);
every `_last.pt` verified intact and all six lanes resumed (chains 63209099–70). wandb
runs show a "crashed"/frozen window while each cell re-trains its unlogged stretch past
the step high-water mark — cosmetic, self-healing.

**ep10 HEALTH CHECK — PASSED on all cells, NO corrections (rule fires only on
pathology; 2026-08-09):**

| cell | ep10 shares (inv/z/h) | Σw·g | Ω_h aud | Λ | probe |
|---|---|---|---|---|---|
| e27mc | (at ep13–14 by read time) | — | — | — | **.4152@ep14** (vm4's online ep16 = .4188; wave-1 smc ep16 = .3558) |
| e27oas | .584/.384/.032 | 6.4 | **.426 (in-band)** | 1.44 | .3889@ep10 |
| e27grp | .500/.482/.018 | 3.8 | .525 | 1.30 | .3945@ep10 |
| e27pv | .667/.321/.012 | 3.0 | .644 | 1.41 | .3879@ep11 |
| e27v10u | ep4: matches its diag twin (.3332 vs .3340) | — | — | — | on-pace |

Tight four-way race in the mc family (.388–.395 at ep10, all shapes healthy); e27mc is
pacing AT the vm4 online curve — the dose-level fix visible at scale. e27lej at ep88
.4851, finishing today.

**§(i) ADDENDUM — mc + grp cancelled on the ring-staleness diagnosis (D-089,
2026-08-09 evening).** Berker flagged both lanes; the trajectories confirm: e27mc probe
non-monotone past the noise band (.4228→.4069 ep22; .4292→**.3999** ep24, g_h ×2.4 at
the dip; chronic z-share .46–.52) and e27grp with episodic conditioner explosions
(ep16 measurement batch: g_cond_z .034→**.575**→.032, shares .945/.053; probe
.4334→**.3961** ep19). The healthy contrasts pin the mechanism — oas (no ring, same
stream as mc) monotone to .4402; pv (per-view rings, scale-homogeneous within each)
smooth .4262; v10u/vm4 (rings on homogeneous means) smooth — **the ring is toxic
specifically on heterogeneous-mean streams** (mc's all-10 mean, grp's locals-group
mean): stale rows of a fast-moving noisy mean → episodic −logdet blowups → violent
conditioner steps. The E24 staleness lesson at the D-084-caveat's predicted site; the
sanctioned OAS fallback invoked. Cancelled at ep24/ep19 (curves = the recorded
ring-vs-OAS A/B on these streams; ckpts trimmed to _last). Replacements fresh:
**e27grpo** (grouped+OAS; first-run combo, early-watched) and **e27pvo** (per-view+OAS
= the diag leader config; completes the stream×estimator grid with pv-ring live).
Wave now: **oas · pv · grpo · pvo · v10u** + lej. mc's LeJEPA-matched main-line story
transfers to e27oas verbatim (identical aug + stream; the estimator is an internal
dial, not part of the matched-frame claim).

**e27lej LANDED (2026-08-10, RAW):** ep100 online best .4929 → landed at
`student.h.cls`: **linear_raw_v2 .5718 · house_v2 .5551 · l2_v2 .5722 · kNN k200
.3373 / k20 .3795** (landed ≫ online here — their online monitor rides the 512-embed,
not cls). Depth: L09 .488, L06 .29, L03 .15; gap .500. References beside it: vm4
(V=4 ring incumbent) .6416 raw / .6554 l2 / .5335 kNN → **+7.0 linear / +19.6 kNN
over the matched-frame lejepa control**; Lightly external 64.0 is not at this frame
(their protocol/schedule) — the control exists precisely to replace that comparison.
The wave cells' target stands: beat vm4 on lej's own V=10 frame. Numbers:
`results/probes/in1k.lejepa.s0.e27lej.extL.csv`; battery/instruments landing.
**Landed Ω_h(cls) from the o8 store (2026-08-10): lej = .1446** — .20 of the c²=.708
threshold, BELOW vm4's .30-of-threshold placement, with the wide linear-vs-kNN gap
(.57/.34) riding it (raw; R6/zoo-placement data point for the joint read).

**D-087 SWEEP TAKEAWAY — E27-T1, USER-APPROVED (Berker 2026-08-10: "d087 agreed";
wording Fable, veto stays open; scope ViT-S/in1k, 5-ep health reads + the running
wave):**
*Scaling V=4→V=10 succeeds or fails on dose level and stream homogeneity — not on
view count or crop harshness.* (a) The share recipe must be driven at the family's
realized total (Σw·g ≈ 7–10): deriving w at a transient early state starved wave 1
(Σ≈1.2) and produced every "instability"; at proper level all 12 sweep configs are
shape-healthy. (b) The aug axis is real and favors small/harsh locals — large locals
collapse the diversity V=10 exists to buy (−7..−10 pts at ep5, Σ≈3.5). (c) Pure
view-count scaling pays monotonically at fixed weights (v4 .2989 → v6 .3327 → v10
.3340 at ep5). (d) Estimator follows stream homogeneity: ring on scale-homogeneous
streams, OAS on heterogeneous-mean streams (D-089 spike evidence + voas's
homogeneous-V=4 landed loss to the ring close the law both ways). Probe orderings at
ep5 indicative only; the 100-ep wave is the confirming instrument.

## (j) The lejepa faithfulness cross-match — Lightly 64.0 vs e27lej .5718 (2026-08-10, Berker priority (i); every external fact below fetched live from the named sources this day)

**(j.0) Provenance of the 64.0 — it is Lightly's own reproduction, not a paper number.**
Full row (docs.lightly.ai benchmarks, LightlySSL 1.5.25; run artifacts on their S3):
LeJEPA | ViT-S/16 | bs 512 | 100 ep | **linear top1 64.0** / top5 85.8 | finetune 78.7/94.5 |
kNN top1 47.1 / top5 74.3. The run's own `hparams.yaml`: 128/device (×4 devices ⇒ 512),
lr 5e-4, wd .05; its checkpoint filename carries **`val_online_cls_top1=0.5623`** at ep99.
The LeJEPA paper (arXiv 2511.08544v3) contains **no ViT-S in1k number at all** — its in1k
runs are ViT-L/14 (online probe **77.1**, 100 ep) and ConvNeXtV2-H (78.5); ViT-H/14 "79%"
abstract-only. **The owed L transcription (§refs): paper Table 2, LeJEPA ViT-L 304M IN-1k
100 ep frozen-backbone few-shot — 1-shot avg 29.55 / 10-shot avg 60.95 / all-shots avg
79.48.** Added to the program's reference sheet: Lightly PR #1932 reports an unpublished
ViT-L/14 run (linear 71.6, kNN 53.4) that never landed in their docs table.

**(j.1) Axis table — our control vs the Lightly-64.0 implementation vs the official at-scale
recipe** (Lightly: `benchmarks/imagenet/vitb16/lejepa.py` + `lightly/loss/lejepa_loss.py` +
`lightly/transforms/dino_transform.py`, master; official: galilai-group/lejepa README +
`scripts/launch_*_ablation.md` + paper v3; ✔ = matches ours):

| axis | ours (e27lej) | Lightly 64.0 run | official at-scale |
|---|---|---|---|
| views | 2g@224 (0.3,1) + **8l@96** (0.05,0.3) | 2g@224 (0.3,1) + **6l@96** (0.05,0.3) | README 2g@224+**6l@98**; paper appendix default 6l@96; §6.1 rec V_l=8; in1k ablation cmds g@238/l@98 (/14-aligned). Paper Table 1b: V=8 (2g+6l) 74.24 **>** V=10 (2g+8l) 74.06 — the §6.1 V_l=8 rec sits below their own table optimum |
| photometrics | jitter **(0.8,.8,.8,.2)** p.8 · gray .2 · blur k7 p.5 · solarize p.2 **all views** · flip | jitter **(0.4,.4,.2,.1)** p.8 · gray .2 · blur p.5 all views · **solarize p.2 on global-2 only** · flip · bicubic | README publishes the table: **(0.4,.4,.2,.1)** p.8, gray .2, blur .5, solarize .2, flip — "applied identically to both view types" (symmetric). Our 0.8-family rode MINIMAL.md's single-res script |
| inv anchor | mean over **ALL 10 views**, every view pulled | **globals-only mean**; the 6 locals pulled to it (globals enter via centroid gradient only) | paper Eq. 6–8/Alg. 2: μ = mean of the **V_g globals**; **all V views** regress to it (all-views mean = the ResNet/single-res special case) — **ours is the wrong functional for multicrop** |
| SIGReg coverage | all 10 views | **locals only** (globals get no SIGReg — Lightly's own deviation from paper Eq. 9) | paper Eq. 9: (λ/V)·Σ over **all** views |
| SIGReg estimator | **256 slices**, 17 knots, t_max 3 | **1024 slices**, 17 knots, t_max 3, DDP all-reduced (sees the global 512 batch) | §6.1 rec: 17 pts, 1024 slices (domain [−5,5]); Table 1a best [−3,3]/2048/5pts. Ours 256 rode MINIMAL |
| projector | 512-embed → MLP [2048,2048,**16**] BN | **384 → [2048,2048,64]** BN (no embed layer at all) | in1k cmds `projector_dim=512`, `embedding_dim=512`; Table 1d best **64** (75.3–75.65) > 512 (73.9–74.8) > 1024; 16 = imagenette-minimal, untested at scale |
| λ / form | .05, convex ✔ | .05, convex ✔ | §6.1 rec .05 ✔ (stable-pretraining sibling uses inv+λ·sigreg non-convex, sigreg pooled — a third official variant) |
| optimizer | AdamW 5e-4 flat, wd 5e-2 ✔ | AdamW 5e-4 **flat at bs 512** (no scaling rule), wd 5e-2 | README: AdamW 5e-4 "good starting point", wd 5e-2 ViT ✔ |
| bs | **128** | **512** (SIGReg + head-BN see 512) | paper: "competitive with batch sizes as small as 128"; Table 1c optimum **512** (74.72 vs 128's 72.20, +2.5); ablation cmds scale bs with V (V=10 → 640) |
| schedule | warmup 1 ep (.01×), cosine → lr/1000 ✔ | identical (warmup 1 ep from .01×, cosine → .001×) ✔ | README final = lr/1000 ✔; at-scale warmup duration unpublished (MINIMAL 1 ep = our source; stable-pretraining 10; VISReg 5) |
| clip / EMA / masking | none / none / none | none / none / none ✔ | README quotes no-clip; **but every in1k ablation cmd carries `teacher_student=true` (SWA teacher; paper Table 4: +3.4 on in100 vit_s8) and `patch_mask_ratio=0.3`** — the paper's Table 1 numbers were produced WITH both |
| precision | bf16 | 16-mixed (fp16) | bf16 ✔ |
| epochs / arch | 100, ViT-S/16, drop_path .1, dynamic_img_size ✔ | identical ✔ | 100 ✔, drop_path .1 ✔ |
| **linear eval** | house `raw_v2`: frozen STORED feats (Resize+CC224), plain Linear, AdamW 1e-3/wd 1e-7, no augs, best-val | **MAE recipe**: frozen backbone, CLS; BN(affine=False)+Linear; "LARS" 0.1·bs/256 m.9 wd 0 (≡ plain SGD-m at wd 0 — their LARS skips trust scaling for wd=0 params); **90 ep, RRC+flip train augs**, max val_top1 over epochs | README: concat CLS of **last two layers** + LayerNorm, AdamW wd 1e-6, lr sched as pretraining; paper: "online linear probe" (details unspecified) |
| kNN | k200 **t=.1**, cosine, L2, bank **500/class (train500 — see (j.5))** | k200 **t=.07**, cosine, L2, **full-train bank** | not reported |
| online monitor | Linear on the **512-embed**, aug'd views → .4929 | Linear on **CLS**, global view 1 → **.5623** | — |

**(j.2) Protocol share, measured on THEIR side:** the same Lightly weights read **56.23
online-CLS vs 64.0 offline-MAE-recipe — +7.8 pts is protocol alone**. Our whole headline
gap is 64.0 − 57.18 = 6.8. Direct decomposition on OUR weights LAUNCHED: the Lightly
protocol ported as `experiments/bench_probe.py` (port notes in-file; declared deviations:
single GPU bs 1024 with their self-scaling lr rule, bf16, ImageFolder) running on
e27lej_ep100 (job 63222504) and d256vm4_ep100 (63222505) →
`results/probes/<run_id>.bench.csv` (+ kNN at their t=.07 beside our t=.1 as
`.bench_knn.csv`). **kNN cross-check says the gap is NOT protocol-only:** near-matched
functional (same InstDisc form; deltas only t .1→.07 and our 500k bank vs their full) reads
.3373 vs .471 — a −13.4 that linear-protocol differences cannot carry; training-side
mismatches are real contributors.

**(j.3) Port-fidelity misses vs sources AVAILABLE at the D-082 recheck (honest record):**
(1) **inv anchor** — the paper's multicrop prediction target (Eq. 6–8: globals-mean) was in
the paper we cited; we ported MINIMAL's all-views form onto multicrop. Weight (1−λ)=.95
rides this term, and the object it anchors to (the locals-dominated all-10 mean) is the
same malformed mean §(d.2)/D-086 diagnosed in our own lane. (2) **photometric family** —
the README's augmentation table (0.4-family) was present in the README whose geometry
paragraph D-082 transcribed; we kept MINIMAL's 0.8-family. (3) **SIGReg slices** — 256
(MINIMAL) vs the README quick-start/ablation-cmd 1000–1024. (4) **proj_dim 16** — the
in1k ablation files (`projector_dim=512`, Table 1d best 64) were in `scripts/`; 16 is
imagenette-minimal. Newly learned, not knowable then: their in1k Table-1 runs used an SWA
teacher + 30% patch masking (ablation cmd flags) — the "no heuristics" branding does not
describe their own at-scale commands; our no-EMA/no-mask control matches the paper's
*recommendation*, not their Table-1 practice. `scripts/je.py` + configs remain unpublished
(inventory re-checked today; repo unchanged since 2026-01-25, now at galilai-group/lejepa).

**(j.4) Independent corroboration — VISReg (HaiyuWu/visreg, arXiv 2606.02572, Wu +
Balestriero + Levine, code Apr/Jun 2026), the lineage's only PUBLISHED full multicrop in1k
ViT trainer** (the session's JOB-2 reference; see §(j.8)): inv anchor =
`proj[:n_global].mean(0)` — globals-only, credited by their paper to LeJEPA (confirms
axis 3 of (j.1)); photometrics 0.4-family jitter (0.4/0.4/0.2/0.05) p.8 + gray .2 +
blur .5 + solarize .2; locals **96 at ViT-B/16 and 98 at ViT-L/14** — independently
validates our D-082 "96 is the /16 adaptation" call; AdamW wd 5e-2 bf16 cosine→lr/1000;
their deltas from LeJEPA: 4 globals + 6 locals, eff. bs 512 (16×32 H100), warmup 5,
**grad_clip 1.0**, λ=.9(B)/.8(L) on an SWD-sketch regularizer (Epps–Pulley replaced),
proj [2048,2048,256/384] BN+**GELU**, probe = DINOv2 protocol (concat CLS last-4, SGD
lr-grid, SyncBN head). Their in1k LP: B/16 400ep 75.7, L/14 400ep 77.0 (no ViT-S rows).

**(j.5) EVAL-FRAME INCIDENT found during the cross-match (D-090): the 08-08/09 landings
rode the SUPERSEDED train500 frame.** `in1k.floorssl.s0.e24voas.extL` and
`in1k.lejepa.s0.e27lej.extL` extracted their train manifest at the yaml default
`train_per_class=500` → `in1k.train500.v1L` (500k rows) — the exact subsampled frame
**D-066 (REVISED, USER-APPROVED 07-29) superseded** — while every vm-era comparator
(d256vm/vm3/vm4) rode the standard FULL train (1,281,167; their 07-30 extracts passed
`train_per_class=null` explicitly). Consequences: (a) the E24-T3 landed A/B magnitudes
(voas −4.2/−4.2/−7.6 vs vm4) are confounded in vm4's favor — probe fit on 2.56× the rows
+ 2.56× kNN bank; (b) e27lej's landed .5718/.3373 and the "+7.0/+19.6" margins carry the
same confound direction; (c) the wave landings would have inherited it. **Corrections all
launched 2026-08-10:** dataset-keyed guard in `experiments/extract.py::_manifests`
(`train_per_class: null` now resolves to the PROTOCOL standard per dataset — in1k FULL,
in100 m50k 500/class; yaml default flipped 500→null); full-train backfill extracts
e27lej_ep100 (63222500) + e24voas_ep100 (63222502) with standard-frame 12-space probe
reruns chained (63222501/63222503, 24 h walls, `probe.py` now writes its CSV cumulatively
per space — the 08-09 lej probe died at its 4 h wall with 8/12 spaces probed, no CSV, no
z-taps; numbers had been stdout-harvested); wave landing chains RE-ARMED corrected for all
five lanes (extract singleton-gated on each lane's drain + probe + audit: 63222506–20; the
previously armed chains had drained in the 08-09 NFS outage). Until the reruns land, the
canonical comparison set (vm4/vm3/vm on full-train) excludes voas+lej; corrected numbers
replace the landed ones on this card and E24's when they arrive, with the deltas reported.

**(j.6) Comparable-protocol readings available already** (raw, for the joint read):
our landed offline `raw_v2` on CLS .5718 sits next to their ONLINE CLS monitor .5623
(different protocols, both weaker-than-MAE-recipe reads); their offline-vs-online delta
is +7.8 on identical weights; our own online (512-embed, aug'd) reads .4929 vs landed
.5718 — the embed-riding monitor's −7.9 underread noted at landing. vm4's raw .6416 /
l2 .6554 (full-frame, no-aug probe) ≥ the 64.0 bar measured under their aug-assisted
MAE recipe — the P-E27-4 comparison as currently claimable rides a HARSHER protocol on
our side; the bench-probe pass upgrades it to same-protocol.

**(j.7) Control-fix options (PROPOSED — nothing launched; the S wave and its e27oas
main-line A/B are UNAFFECTED — λ never enters our lane):**
- **P (paper-faithful control at the MATCHED frame) — my recommendation:** keep the frame
  identical to our lanes (2g@224+8l@96, bs 128 — paper-blessed floor, single GPU), fix the
  loss to paper Eq. 6–9 (inv anchor = globals-mean, all views pulled; SIGReg over all
  views), README photometrics (0.4-family, symmetric), 1024 slices, proj_dim 64
  (Table 1d best; embed 512 stays), leave no-EMA/no-mask (the paper's recommended,
  heuristic-free configuration — declared, since their Table-1 practice differs). One
  ViT-S slot ~2.5 d. Decide AFTER the bench-probe lands (if protocol explains most of the
  linear gap, the rerun's case rests on the kNN gap + loss-form correctness).
- **L (Lightly replication):** their exact lejepa.py (6 locals, locals-only inv+SIGReg,
  solarize asymmetry, proj 64 no-embed, bs 512) — reproduces the 64.0 bar itself; needs
  DDP-4 or waits for the DDP build (their SIGReg all-reduces over the global batch;
  grad-accum is NOT equivalent for SIGReg/BN). A replication exercise, not our matched
  control — only if Berker wants the bar validated in-house.
- **E (eval-only):** no rerun; the bench-probe + corrected-frame numbers become the
  official comparison basis. Cheapest; leaves the control's loss-form deviation standing.

**(j.8) JOB 2 — the vicreg/visreg reference check (session priority (ii), closed):**
`facebookresearch/vicreg` is ARCHIVED (Nov 2024; last commit Dec 2022): full in1k
pretraining + eval code but **ResNet-only, LARS, no ViT ever**; linear protocol =
SGD m.9 wd 1e-6, head-only lr .3 (README .02), 100 ep, RRC+flip, trunk 2048-d, frozen
backbone in eval mode. solo-learn lists ViT backbones but has **no ViT-VICReg result/
checkpoint anywhere** — a ViT-S VICReg cell would be an adaptation with no published
reference. **The useful second reference is VISReg** (§(j.4)) — same author lineage,
full published multicrop ViT in1k trainer with configs + weights (ViT-B/L; CC BY-NC),
DINOv2-protocol eval; closest existing public analogue of LeJEPA's unpublished je.py,
and it even ships a `vicreg.py` baseline inside its ViT multicrop trainer.

## (j.9) D-092 execution log (2026-08-10, Berker's cross-match-round directives)

**e27lejl — the Lightly-replication control, LAUNCHED** (chain + landing armed; see the
S-wave §(i) conventions). Config = Lightly's benchmark lejepa.py verbatim, single H100:

| axis | e27lejl (replication) | (e27lej, for contrast) |
|---|---|---|
| views | 2g@224 (0.3,1) + **6l**@96, jitter **(0.4,.4,.2,.1)** p.8, solarize p.2 **global-2 only**, blur p.5 all, bicubic | 8l, jitter (0.8,.8,.8,.2), solarize all views, bilinear |
| loss | .05·SIGReg(**locals**, **1024** slices) + .95·MSE(**locals → globals-mean**) | SIGReg(all 10, 256) + MSE(all → all-10 mean) |
| head | 384 → [2048,2048,**64**] BN — **no embed stage**; h = trunk CLS | 512-embed → [2048,2048,16] |
| opt | AdamW 5e-4 flat, wd 5e-2, warmup 1 ep, cosine→lr/1000, no clip, no EMA, **bs 512 global, one GPU** (≡ their 4×128: SIGReg/BN see the global batch either way), grad_ckpt | same opt dials at bs 128 |
| declared deviations | bf16 (theirs fp16-mixed); house k9 blur approximating their PIL radius blur; seed 0 (theirs unseeded); house online monitor (CLS, all views) | — |

Targets at land (full-frame + bench column): Lightly linear 64.0 / kNN 47.1 / online-CLS
56.23. e27lej stays on the card as the declared matched-frame variant (its corrected-frame
reprobe pending, D-090). New guarded code paths (all legacy paths byte-identical, selftest
§5–7 incl. a hand-pinned legacy inv): `lightly_mc` views (`LightlyLejepaMultiCropDataset`),
`mc_form=lightly`, `n_slices`, `grad_ckpt`, `emb_dim=0` bare-CLS anatomy (+ adapter branch:
h = `student.h.cls`, no z.embed space).

**AUG-DECLARATION ERRATUM (found by the D-092 selftest, measured 2026-08-10):** the house
pattern `RandomApply([v2.RandomSolarize(...)], p=.2)` HALVES the effective probability —
torchvision-v2 RandomSolarize carries its own p=.5 (measured: wrapped .0988 vs bare .1983
over 20k trials). Scope: `orbit_stack` (audit_v1 + the lejepa/floorssl training family)
and `_dino_view` — **every house run's effective solarize is ≈.1 where docs said .2.**
Frozen stacks are NOT changed (v1 semantics = the code; internal comparisons all rode the
same effective value); the §(j.1) photometric row corrects to "solarize eff .1". The
replication's `_bench_view` uses the bare transform (true .2 = Lightly's own
RandomSolarization(prob=.2)). Same trap found IN VISReg's shipped multicrop
(`RandomApply([RandomSolarize], p=.2)`, multicrop.py:118) — their published numbers ran
effective ≈.1 and our faithful repro will match them as shipped; by coincidence the house
effective .1 equals VISReg's.

**in1k eval standard (PROTOCOL v1-draft.8):** full-train only — the train500 variant is
REMOVED (stores + manifest deleted, 8.1 G; extract.py raises on in1k per-class; the
confounded voas CSV preserved as `*.train500-superseded.csv`) — and every in1k landing now
carries `bench_linear_v1` (+ t=.07 kNN rider) beside the house probes; bench jobs armed on
all five wave lanes and the replication.

**VISReg reproduction staged:** donor pinned `third_party/visreg` @ 47b1cf4; runnable copy
`~/visreg_repro` on their pins (py 3.12, torch 2.8 — their 07-22 SyncBN commit makes pins
load-bearing) with ONE declared patch (HF hub → local imagefolder; `PATCHES.md`). Runs the
shipped ViT-B config as-is: 4g@224+6l@96, λ=.9 on the SWD sketch, proj [2048,2048,256]
BN+GELU, K=4096 projections, AdamW 9e-4, wd 5e-2, warmup 5, clip 1.0, bf16, 100 ep,
effective bs 512. Upstream ambiguities flagged (paper says K=2048 + 400-ep results; config
ships 4096/100 — we run the config). Their co-trained probe is gradient-isolated
(`probe(emb.detach())` — a monitor, no supervised leakage). Launch = accelerate multi-GPU
(N × 512/N) as wave landings free H100 slots; fit smoke first.

## (k) D-094 execution log (2026-08-10 late — the S-read round)

**Berker's S read (verbatim in D-094): v10u = the winner; keep only e27v10u + e27oas;
lejepa → "the lightly numbers, i want no diff"; variant program (v10u+swa, v10u+bs512,
+ the three ViT-B mirrors) conditional on attributing the gain to uniform views.**

**Kills executed:** e27pv (ep44) · e27grpo (ep25) · e27pvo (ep25) — segments, landing
chains, bench jobs cancelled; ckpts → `_last` only (D-089 policy). Curve record at kill:
pv plateaued .45–.46 from ep34 (last reads .4587/.4521/.4475); grpo .4527, pvo .4649 —
all below oas's matched-epoch curve. Stream verdict material (raw): all-views-mean ≥
grouped ≈ per-view at S; estimator question inside per-view (pv vs pvo) dies half-answered.

**The attribution table (raw, mid-training — the (h)-style read behind Berker's
conditional):** e27oas and e27v10u differ on FOUR axes, not one — geometry, estimator
(ring is D-089-illegal on mc's heterogeneous mean, so each lane rides its legal-best),
dose weights (per-lane share-parity derivations), and **per-step compute: 1970 vs 690
tokens/sample = ×2.86** (wall ×1.75: ≈101 vs ≈58 min/ep).

| comparison | v10u | oas | Δ |
|---|---|---|---|
| epoch-matched (ep19) | .5130 | .4377 | **+7.5** |
| FLOP-matched (≈46 mc-ep-eq: v10u ep16 vs oas ep46) | .4990 | .4984 | **+0.1 ≈ tie** |

**lejepa zero-diff vehicle = Lightly's own code (D-094(2)):** e27lejl stood down while
queued (deviations bf16/blur/seed/monitor = the "no diff" violations; drop_path 0.1
verified matched, a diff-candidate closed). `third_party/lightly` @ f444cf36 pristine;
`~/lightly_repro` runnable, zero patches (REPRO.md); `slurm/lightly_lejepa.sbatch` =
segmented 4×H100 wrapper around THEIR `main.py --methods lejepa` defaults (4×128 = bs
512, 16-mixed, unseeded, their eval chain; finetune-eval skipped, declared); smoke
63229333. Their-code details confirmed at source: backbone carries `drop_path_rate=0.1`,
online monitor = first-global-view CLS (detached), scheduler = per-step cosine from
1-ep warmup (.01×→lr→.001×), current HEAD transforms are BICUBIC (#2018).

**Variant riders staged for the ruling:** SWA-teacher = declared reconstruction
(`je.py`/configs unpublished; only post-hoc weight-averaging is fully defined);
v10u-B ≈ 2.9× the measured mc-B pin → ≈6.5–7.3 d/100 ep single-GPU (mc-B 2.6 d);
bs-512 v10u fit unproven (lejl's proven bs-512 path was 616 tokens/sample; v10u is
1970 — fit smoke required before promising single-GPU).

**(k.1) D-095 — the ruling landed: FLOP-matched re-base + SWA implemented (same
evening).** Berker, convinced by the FLOP-collapse table: the new cells ride the exact
Lightly view stack (2g@224+**6l**@96, 0.4-family jitter, true-p solarize global-2-only,
bicubic) under OUR loss — frame- AND FLOP-matched to the lejepa reproduction bar.
`aug=lightly_mc` in floorssl (selftest §10); estimator OAS no-ring (heterogeneous mean,
D-089 law). SWA per the paper's one-line spec (*"SWA on the encoder producing μ in
Eq. (6)"*, Izmailov equal-weight): grad-free eval twin deepcopied post-init (student
byte-identical to parent), per-step equal average, anchor = twin's all-view z-mean at
the lane's own anchor set, ×2V/(V−1) on the uniform branch pins the init pull to the
calibrated w_inv; swa_k in extras; twin in ckpts (post-hoc SWA-eval free). Selftest
§(9) PASSED on GPU (in-job 63232096: init-parity exact, hand-math average, grad-free,
round-trip). Program: **e27lmc** (pilot 63239252, dose from the [share] ep1 g's at
s*=(.63,.34,.03), T=9.0) → **e27lmc_swa** (base doses, ~+⅓ step) + **e27lmc_b512**
(flat lr; the lejl bs-512 single-GPU path IS this geometry) → three ViT-B mirrors.
v10u+swa superseded pre-launch; v10u/oas land unchanged (scaling story + house-aug
control; FLOP-overlay read pre-registered). Same-evening greens: Lightly repro smoke
TRAINING on 4×H100 (63229333, epoch 0 at their exact 1.28M/512 = 2502 steps);
b512fit past construction no-OOM; T1(b) harsh-locals tension of the mild 0.4-family
flagged in-conversation.

## (l) The estimator A/B — e27lmcs5q (Berker directive 2026-08-15; pre-registered BEFORE pilot numbers)

**Context (the S-cell diagnosis, `results/figures/e27/s_cells_online_traj.png`):** the landed
S cells split into two curve families — {vm4 .6392, v10u .6581@92} (homogeneous @224 views,
E24 grid doses, ring q=3) vs {oas .6063, lmc .6026, lmcb5 .5982, lmcse .6150} (multicrop,
pilot-law doses, OAS q=0) — with the vm4-family advantage forming in the LAST THIRD of
training (vm4 mid-pack until ~ep60). Within-family axes measured small: photometrics −0.4
(oas↔lmc), bs512 a crossover (+4 mid, −0.4 end, ep99 share state shifted), swa +1.24.
Between-family +3.3–5, carried by three entangled axes: view geometry / dose regime /
estimator. **This section = the estimator axis, isolated at the bs512+swa point.**

**Directive:** lmcs5 (S · lightly_mc 2g+6l · bs512 · swa=ema · OAS q=0 · w 3.53/4.46/0.180)
already exists (mid-flight ep39) → launch its **ring-queue twin `e27lmcs5q`**: identical
config except `floor_shrink=null queue_steps=3` (the vm4-family estimator, both taps —
h_queue_steps unset falls through to queue_steps).

**Dose procedure (the standing per-config pilot law, T=9.0):** pilot `e27lmcs5qp`
(63571049, 4h H100) runs the q-config at lmcs5's incumbent w; chain doses
w_i = 9.0·τ_i/g_i^ep1 with **τ = lmcs5's own realized ep1 shares (.492/.474/.035)** —
the twin targets its partner's formation, the L↔B precedent (D-097 arc). Chain launches
on the pilot ep1 line (watcher armed); one-shot transfer, negative-feedback miss expected
and reported as at B/L.

**Relation to E27-T1/D-089 (NOT a silent protocol change):** T1's estimator clause
(ring↔homogeneous, OAS↔heterogeneous) was agreed on 5-ep health reads with "the 100-ep
wave is the confirming instrument" — this A/B is that instrument for the clause: ring
deliberately run on the heterogeneous mc stream (view_mean anatomy: the per-step ring
objects are view-means, homogeneous ACROSS steps; the heterogeneity lives inside each
mean).

**Pre-registered predictions (committed 2026-08-15, no pilot numbers exist):**
- **P-l-A (estimator carries the family gap):** lmcs5q develops the vm4-style late slope —
  tracks lmcs5 through mid-training, separates in the last third, endpoint ≥ +1 online.
- **P-l-B (null):** twins track within ±0.5 throughout → estimator not the driver at this
  point; the family gap falls to dose regime / view geometry (separator cells:
  lmc-at-heavy-doses, v4-at-pilot-doses — not launched, Berker's call).
- **P-l-C (T1-clause vindication):** ring on the mc stream is actively worse — instability
  or endpoint ≤ −1 vs lmcs5.

**Read protocol:** A/B at ep100 ONLY (both arms) + bench columns; NO mid-run reads — the
bs512 mid-phase inflation is measured (lmcb5 +4 at ep25/50, endpoint −0.4). lmcs5's own
A-side endpoint is pending (ep39) — the A/B completes when BOTH land.

**Pilot record + launch (2026-08-15 night, RAW):** pilot 63571049 ep1 under ring q=3 at
incumbent w: shares (.728/.232/.040), **g = (0.158 / 0.040 / 0.172)** — an order of
magnitude below lmcs5's OAS ep1 g's (1.008/0.768/1.388): the estimator swap alone
collapses the formation gradients (ring's n=bs×4 + no OAS inflation). ep1 probe .0434
(lmcs5's own ep1 .0565 — same band at mis-calibrated doses); no instability. Doses by the
law: **w_q = (28.03 / 106.65 / 1.831)**. FLAG (raw, un-interpreted): this share-matched
point lands nearly ON vm4's E24 grid optimum (26.9/129.2/1.679) — the "light pilot-law
doses vs heavy grid doses" distinction between the two curve families may be largely the
OAS-vs-ring g-scale difference expressed through the same law; if so the entangled
dose-regime and estimator axes partially merge, and this run tests them jointly rather
than estimator-alone. Chain: 4×24h singleton 63571153–56; pilot cancelled after ep1
(read-and-cancel). ep1 recalibration check owed (realized vs τ; one-shot miss expected
per B/L).

**Chain ep1 recalibration check (63571153, RAW):** realized shares **(.678/.283/.038)**
vs τ (.492/.474/.035) — the B/L-style negative-feedback miss, inv-heavy/moment-light:
raising w_inv 3.53→28.03 doubled g_inv (0.158→0.314); w_floor ×24 barely moved g_m
(0.040→0.034). Same direction as L's accepted miss (.656/.323/.021 vs .576/.397/.027),
larger split; note lmcs5 itself missed the trio τ the OTHER way (.474 moment vs .34).
**Dose level Σw·g(ep1) = 12.9 — ABOVE the E27-T1 healthy band 7–10** (the one borderline
number; T1's band was a 5-ep health heuristic). Live health all normal: ep1 probe .0421
(band: lmcs5 .0565), omega_z 6.61→3.02 declining, lam .865, no NaN. Per the §(l)
pre-registration the one-shot procedure stands: run continues, ep2–5 settling + the
Σ-level watched (health watcher armed); accept-vs-redose = Berker's call, as at L.

**ep2–5 settling record (RAW, watcher 2026-08-16):** shares settle at **≈(.80/.19/.01)**
(ep2 transient .875 inv, then .795/.791/.807) — the inv-heavy miss PERSISTS; the twin's
realized share state is materially off both τ and the A-side's own settled state
(lmc/lmcb5 ep25 ≈ .60–.61/.36–.38/.02–.03), so the A/B carries a dose-realization
confound alongside the estimator swap. **Σw·g descended INTO the T1 band: 12.9 → 22.3
(ep2 g_inv transient) → 10.1 → 7.95 → 7.0 by ep5** — the level concern resolved. Probe
.0421→.0992→.1527→.1959→.2332 (trails lmcs5's ep1–3 by ~.02–.03 — consistent with the
P-l-A shape, judged at ep100 only). omega_h 2.26→1.18, omega_z 3.02→0.88, lam .865→1.16
— healthy throughout. Options if the share miss is ruled unacceptable: second-shot
re-dose (w_inv↓/w_floor↑ from the settled g's — would be NEW procedure, the program is
one-shot to date) vs accept-with-confound-noted (the L precedent).

**Second shot — e27lmcs5q2 (Berker directives 2026-08-23):** first directive "compute and
launch another variant … applying the correct ratios sounds important" → a twin-target
secant (τ = lmcs5 ep1, T=9.0, g = q1's chain-ep1 → w = 14.10/125.47/1.158) was launched
(63583136–40) and **cancelled pre-ep1, no ckpt written**, on Berker's mid-session
correction: *"apply and calculate your doses from the winners (without watching what has
happened in our existing runs, adjustments and tunings wont be as strong). compare
multiple runs' forces along with how they performed (in a matched compute frame)."*

**The winner-derived dose (the launched one).** Cross-run force census (per-run `[share]`
ep1 lines re-read from first-segment logs; share = w·g/Σ verified exact on every cell;
resume-first-line transients excluded — v10u ep75/ep93 carry cold-resume artifact lines):
ring family — **v10u ep1 shares (.441/.542/.017), g (.935/.239/.576), Σw·g = 57.0** at the
E24 grid w (26.9/129.2/1.679); vm4 ep1 pre-dates the share logger, its E22 warm-pull read
(held state) gives shares (.535/.452/.012) at Σ≈10.9 — corroborates the moment-rich shape,
not the ep1 level. OAS family ep1: oas (.462/.505/.033) Σ=76.8 (w 21.4/49.6/1.89) · lmc
(.476/.488/.036) Σ=8.98 (trio w 11.43/13.85/0.451, the pilot-law T=9 exactly) · lmcb5
(.388/.581/.031) Σ=20.4 (same trio w, bs512-shifted g's) · lmcse (.567/.395/.038, own
pilot w — trio-w algebra fails on it alone) · lmcs5 (.492/.474/.035) Σ=7.24 · q1
(.678/.283/.038) Σ=12.9. Finals against these: 68.41/66.25/65.69/65.12/64.49/62.77 —
within the 616–690 tok/s frame the pattern is NOT monotone in any single number (lmcb5 is
the most moment-rich ep1 AND worst; oas ran Σ77 mid-pack), flagged raw for the joint read;
the clean winner fact is v10u's formation point.

Target = the top cell's clean ep1: **τ_win = (.441/.542/.017), Σ_win = 57.0** (ring↔ring
comparable with q2; geometry differs by design — that IS the §(l) axis). Doses solved
through the measured ring-internal response exponents rather than the frozen-g secant
(pilot→chain pair: g_inv ∝ w^.33, g_m ∝ w^−.051, g_h ∝ w^.197): solve w·g(w) = Σ_win·τ_i
→ **w2 = (61.75 / 1020.0 / 3.19)** (frozen-g secant would give (80.1/908.7/3.56) and
forecast an inv-heavy miss (.554/.429); the exponent solve forecasts landing ≈ τ_win at
Σ≈57 IF the power laws extrapolate — w_floor extrapolates ×9.6 past the measured range,
declared). Config identical to q1 otherwise (submit-line diff = the three w's + tag).
Chain 5×24h singleton **63583418–22** (`--exclude=gpu269,gpu273`, post-maintenance
sick-GPU incident). **ep1 recalibration check OWED on q2's first `[share]` line** —
judged against τ_win and Σ≈57, grad-norm kill-trigger discipline applies (q1 at Σ12.9 was
"above the T1 band"; the winner's own measured 57 supersedes the band as the reference
point for this lane, the §(l) partial-merge flag playing out). Context for the read: q1's
drift continued past the ep5 settle (ep60 = .898/.100/.002, g (0.354/0.010/0.014)); the
dose targets ep1 FORMATION per the standing law.

**q2 chain ep1 recalibration (63583418, RAW):** realized shares **(.688/.301/.011)**,
g = (0.775/0.021/0.248), **Σw·g = 70.1** — level lands in the winner regime (×1.23 over
the 57 target; q1 sat at 12.9), shape misses inv-heavy AGAIN and lands ≈ q1's own ep1
(.678/.283/.038). The fitted exponents did NOT extrapolate: realized local exponents on
the 28→61.75 / 106.65→1020 / 1.831→3.19 legs are g_inv ∝ w^1.14 (near-linear, vs .33
fitted), g_m ∝ w^−.21 (w_floor ×9.6 DROPPED g_m .034→.021 — cumulative: w_floor ×229
since lmcs5 moved g_m only .040→.021), g_h ∝ w^−.17 (sign flip). RAW FLAG for the joint
read: on the ring/mc-stream lane the moment share appears w_floor-UNREACHABLE beyond
~.30 at ep1 — the winner profile (.54 moment) may not be expressible through w in this
configuration; whether that is itself the §(l) answer (estimator/stream sets the share
ceiling, doses only pick the level) is an interpretation and WAITS. Health at ep1:
probe .0370 (q1 .0421, lmcs5 .0565 — lower, watched), omega_z 3.94, lam .753, epoch
completed with no kill-trigger. Per the pre-registration the one-shot stands: run
continues, ep2–5 settle to be recorded, ep100 + bench = the read.

**q2 vs d256vm4 matched-epoch loss read (RAW, Berker request 2026-08-25, q2 at ep43):**
`results/figures/e27/q2_vs_vm4_losses.png` + `results/compare/q2_vs_vm4_losses_perep.csv`
(wandb per-step means bucketed per epoch; vm4=wezwcmvp, q2=9ri419ko; term scales differ by
geometry/doses — shapes and slopes are the comparable part). Headline values (q2/vm4):
moment_kl ep10 **.037/.211**, ep43 **.029/.166** — q2 quenches the z-moment residual to
~1/6 of vm4's level by ep10 and holds flat, BELOW lmcs5's .090 (the previous lowest),
despite w_floor=1020 (the w-unreachable flag holding in the loss values, not just the g's).
inv slope ep5–10: q2 −.0227/ep vs vm4 −.0074/ep (fast-falling; the winner v10u's was
rising). h_moment_kl ep43: q2 .173 vs vm4 .281. Online acc ep43: q2 .546 vs vm4 .499 —
q2 +4.7 at matched epoch, the exact magnitude of the measured bs512 mid-phase inflation
(lmcb5 +4@25/50 → −0.4 endpoint), overlay = band context only. Moment SHARE (q2 own logs):
ep25 .152 → ep40 .144 → ep43 .191 (noisy tick-up; the pre-registered ep25→50 drift knot
lands ~ep50). Instrument state at ep43: mkl-residual + early-inv-slope carry the
saturator profile, share-drift unresolved until ep50; per pre-registration these kill,
never crown — the §(l) read stays ep100+bench.

**Matched-STEP amendment (Berker 08-25: "epoch nums can be confounded" — bs512 vs bs128
= 4× fewer steps/epoch):** `results/figures/e27/q2_vs_vm4_losses_steps.png` +
`results/compare/q2_vs_vm4_losses_perstep.csv` (window = q2's 110k steps = vm4's ep11;
schedule phase declared 4× misaligned: warmup ends q2 25k / vm4 100k). The moment-quench
is AXIS-ROBUST: at every common step q2's mkl is 6–10× below vm4's (10k: .070/.499;
110k: .029/.205), the quench completes by ~10k steps at lr ≤ .4·peak inside q2's own
warmup, and it holds on the samples axis too (5.12M samples: q2 .070 vs vm4 ≈.26); vm4
reaches mkl .095 only at its OWN 1M-step endpoint — still 3× q2's level. The inv
contrast does NOT survive the axis change: the epoch-axis level gap was mostly the 4×
step deficit (values near-converge at the window edge, .311/.296), and slopes converge
(50–100k: q2 −.00098/1k vs vm4 −.00070/1k). METHOD NOTE for the 400-round watch pair:
the early-inv-slope instrument is axis-confounded across batch sizes — read slope signs
on the STEP axis or within-bs cohorts; mkl-residual + moment-share drift are the
axis-robust pair.

**q2 ep2–8 settle (RAW, read 08-23 20:16):** shares (.765/.226/.010) → (.840/.154/.006)
→ (.829/.166/.005) → (.779/.216/.006) → (.758/.236/.006) → (.799/.198/.004) →
(.786/.210/.003). **Σw·g: 70.1 (ep1) → ≈20 by ep5, HOLDING 19–20 through ep8** — the
second shot sustains ~2.7× q1's settled level (7.0), while the share shape settles onto
q1's own settled shape (≈.79/.21/.005 vs q1's .80/.19/.01): with both shots now settled,
the lane's realized SHAPE is dose-invariant and only the LEVEL moved — the ep1 raw flag
holding at settle. Probe .0370 → .1966 (ep5) → .2748 (ep8); ep5 vs q1's ep5 .2332 =
−.037 (band-adjacent, trailing). omega_h/z declining (0.97/0.76 at ep8), lam 1.13 —
healthy, no kill-trigger. Read at ep100 + bench per §(l).

## (m) The 400-epoch round (D-103, launched 2026-08-25; deadline 24 Sep)

**Cells (all six + twins, Berker directive):** B1 `e27lm4sbe400` · B3 `e27lm4sbetl400`
(sbe + eta_min 5e-5 = lr/20, one delta) · L1 `e27lm4Ls5b400` · L3 `e27lm4Ls5btl400` ·
B2′ `e27v6b400` / L2′ `e27v6L400` (all-global: aug=lejepa V=6@224, ring z-q3/h-q7(B)/q11(L),
h_d_slice 256/384, swa, 1182 tok = ×1.17 anchor) · twins `e27lm4sbetl100`,
`e27lm4Ls5btl100`, `e27v6b100`, `e27v6L100`. DDP-2/-4, 120h singleton segments,
exclusions gpu269/273/267. Pilots `e27v6b_pilot`/`e27v6L_pilot` at grid w; chain doses
w_i = 57.0·τ_win,i/g_i^ep1 (D-102 winner procedure, τ_win = v10u).

**Pre-registered predictions (committed BEFORE any numbers; 2026-08-25):**
- **P-m-1 (epoch scaling):** B1/L1 land above their 100-ep selves on bench by an amount
  in the VISReg band (+3…+5; VISReg-B measured +4.3 on our yardstick). Below +2 =
  our-recipe-specific saturation, the round's central negative result.
- **P-m-2 (the moment hypothesis, main):** B2′/L2′ hold the moment channel (no q2-style
  quench: pilot/early mkl NOT <.1 by ep2 — that pattern = kill) and show the vm4-family
  shape: track the mc incumbents mid-run at matched epochs, separate UPWARD in the last
  third. Directional: v6-400 ≥ sbe-400 at matched FLOPs (×1.17 declared).
- **P-m-3 (tail cells):** per the agreed hypothesis the mc saturation is moment-vacuity,
  NOT lr schedule → B3 ≈ B1 and L3 ≈ L1 (Δ ≤ +0.5 bench); the 100-ep twins referee the
  same direction cheaply by ~09-04. A large B3−B1 gap would REFUTE the hypothesis's
  sufficiency and revive the schedule arm.
- **Watch protocol per cell:** ep1–5 formation gate (kill authority) · watch pair =
  mkl-residual + moment-share drift, both on the STEP axis for cross-bs reads (the §(l)
  amendment; kill only when both agree, never crown) · matched-epoch online overlay vs
  100-ep incumbents as a BAND (declared confound: 400-cosine lr phase) · endpoint reads =
  ep400 + bench only.

**B2′ pilot ep1 + chain launch (RAW, 2026-08-25 ~12:45):** pilot 63665081 at grid w
(26.9/129.2/1.679): ep1 shares **(.618/.373/.008)**, g = **(1.102/0.139/0.239)**, ep1
probe .0949, omega_h .720 / omega_z .816 / lam .940 — healthy, no kill-trigger;
moment SHARE .373, moment-KL VALUE ≈ .44 at ep1-end (wandb; the P-m-2 quench gate is on
the VALUE, <.1 by ep2 — far clear; CORRECTED 08-25 14:20, the first write quoted the
share as the value). Law applied:
**w = 57.0·τ_win/g^ep1 = (22.81 / 222.26 / 4.054)** — extrapolation ×0.85/×1.72/×2.41
from pilot w (modest; the q2 lesson's ×9.6 territory avoided). Chains launched on the
line per D-103: `e27v6b400` 3×120h (63668566–68) + twin `e27v6b100` 2×120h (63668569–70),
DDP-2. Chain-ep1 recalibration check OWED (shape expected to land between pilot's
inv-heavy (.618/.373) and τ_win (.441/.542) per the standing negative-feedback miss).

**Chain ep1 recalibration (RAW, 63668566 ~13:55):** realized shares **(.654/.339/.007)**
— inv-heavy again, slightly PAST the pilot's own shape (the negative-feedback miss,
stronger than forecast); g = (4.089/0.218/0.255) → **Σw·g = 142.8, ×2.5 over the
Σ=57 target** (q2's own ep1 overshot ×1.23 then settled 70→20 by ep5 — level settles
down as g's shrink; watched). ep1 probe .0918 (pilot band ✓), moment SHARE .339 / moment-KL VALUE ≈ .29 easing to
.24 into ep2 (wandb; the vm4 early band, nowhere near q2's .09-by-ep3 collapse — the
quench gate is on the value; share≠value correction as in the pilot block), omega_h
.813 / omega_z .943 / lam .929, in-job selftests ALL PASS. No kill trigger; per the
one-shot procedure the run continues, ep2–5 settle = the formation-gate record.

**ENOSPC incident (14:04–14:08, during the disk-full window):** the group fs hit 100%
(38G free) mid-launch; `e27v6b400` seg1 died on its ep2 `_last` checkpoint write (file
truncated to 0 bytes), its two singleton spares then started into the full disk and died
at the gate within 90 s — the whole chain burned; the mocov3-S re-bench (63662294) died
mid-write too. The B2′ TWIN on the same node survived (its ep2 ckpt landed complete).
Repair after the purge freed 391G: corrupt 0-byte ckpt removed, `e27v6b400` relaunched
fresh 3×120h (63671801–03; ~2 epochs lost, same doses — the ep1 recalibration numbers
above remain the measured formation state), mocov3-S bench relaunched (63671804,
RESUMES from its intact ep-20 head state — the resume machinery's first live save).
Lesson for the round: chain spares burn fast when the failure is environmental —
restock after any multi-segment failure event.

**L2′ pilot ep1 + chain launch (RAW, 63665136 ~14:15):** ep1 shares **(.828/.169/.003)**,
g = **(1.776/0.075/0.095)**, ep1 probe .0554, moment-KL VALUE ≈ .53, omega_h .802 /
omega_z 1.135 / lam .841 — healthy, no kill-trigger. Law applied: **w = 57.0·τ_win/g =
(14.15 / 411.9 / 10.20)** — extrapolation ×0.53/×3.19/×6.08 from grid w; the h_lamb ×6.08
is the program's second-largest single-knob transplant (after q2's failed ×9.6),
DECLARED. Chains launched per D-103: `e27v6L400` 5×120h (63669105–09) + twin `e27v6L100`
2×120h (63669110–11), DDP-4. **THE 12-RUN GRID IS FULLY QUEUED as of 14:20.** Chain-ep1
recalibration owed on e27v6L400's first line.

**Mean-washing readout (RAW, `e27_meanwash.py` → results/compare/e27_meanwash.csv,
2026-08-25 ~12:00; 7 ckpts × own training geometry, N=512, fixed 128-slice, exact
scatter):** the hypothesis's PREMISE holds — pairwise view correlations split global
(r_gg .87–.97) vs local (r_ll .40–.69) on every mc cell, q2 the extreme (.454 z /
.396 h); view-mean effective-n: all-global cells 1.11 vs mc 1.23–1.68 (q2 max). The
naive SHAPE mechanism does NOT: centered-covariance KL of the view-mean ≈ per-view
(wash_ratio 1.03–1.09 at z, >1 at h) — the mc mean is not more isotropic in raw shape,
so "the mean Gaussianizes" fails as stated for raw z. Refined candidate (raw, awaits
joint read): the quench lives in the ESTIMATOR-relative residual — higher n_eff shrinks
the stream's step-to-step moment fluctuations, the ring estimate tracks the stream
tightly, and the CONDITIONED KL collapses without the raw shape being "done"
(functional conclusion unchanged: the constraint loses grip on the mc stream; mechanism
relocated to estimator-tracking). Sharp test available: cross-anatomy conditioned
residuals on one ckpt (q2's z under a 4-global stream vs its own mc stream). P-m-2's
chain health gate is unaffected (defined on the training instrument itself).

**RAW instrument note (Berker pilot-watch question 2026-08-25, "grad norms ~100, clipped
at 1.0"):** wandb census run on the concern — the B2′ pilot's regime (ep1 grad_norm
median 90, p90 135, 100% of steps clipped, inv rel-jitter .017) is the FAMILY NORM, not
a pathology: vm4 ep1 median 71 / v10u 95 (both 100% clipped, inv jitter .036/.038 —
the pilot is 2× smoother), q2 ep1 median 269. Full-run fact, never previously inspected:
**vm4 and v10u spent their ENTIRE 100 epochs at 100% clipping** (median raw norm 71–96
@ep1 → 7–10 mid-run → RISING to 22–27 by ep100), q2 likewise (304→15–17). The family
trains as normalized-gradient SGD throughout: the clip sets step LENGTH (= lr), the w's
set only the direction MIX — consistent with the share formalism being the operative
dose instrument and dose LEVEL Σw·g reading non-monotone in past cells; corollary
(raw): under permanent clipping the eta_min tail cells act directly on late step length.
No kill-trigger condition (that clause = >100× running-median spikes). Pilot proceeds
to its ep1 line unchanged.

**B2′/L2′ ep2–5 settles + the formation gate reading (RAW, 23:57–00:15 08-25/26; owed
since the chain-ep1 blocks):**
- **B2′ (e27v6b400) shares ep2–5:** (.726/.267) → (.655/.337) → (.648/.345) →
  (.660/.334) — settled ~(.66/.33/.007). Σw·g ep5 = 18.7 (down from the ×2.5
  chain-ep1 overshoot 142.8, the q2-pattern 70→20 settling ✓). **mkl VALUE ≈ .168 at
  ep2** (wandb y0e74bet; .106 by ep~6) — **P-m-2 quench gate PASS.**
- **L2′ (e27v6L400) chain-ep1 recalibration + settles:** ep1 shares (.755/.238/.007),
  g = (3.163/.034/.039) → **Σw·g = 59.2 = ×1.04 of the Σ=57 target — ON TARGET** (the
  round's cleanest one-shot; B2′ was ×2.5). Shares ep2–5: (.873/.121) → (.856/.139) →
  (.842/.155) → (.862/.134) — share HOLDS ~.13–.15. **But mkl VALUE (wandb qd0gilvf):
  ≈ .095 at ep2 → .081 (ep2.5) → .051 (ep4) → .047 (ep5) — at/below the P-m-2 .1
  line by ep2 and continuing down, FASTER than q2's .09-by-ep3.** Gate adjudication
  per the §(l) amendment ("kill only when both agree"): the two instruments DISAGREE
  (value quenches, share holds — q2 itself showed this split late) → **NO KILL under
  the registered rule; RAISED to Berker as the L2′ value-quench reading.** Watch pair
  continues on the STEP axis; if the share follows the value down, both-agree is met.
- CORRECTION of the 08-25 ~19:15 grid sweep note (HANDOVER): that sweep read the
  gate against the WRONG cells (tl/Ls5b lanes) and quoted SHARES for the B cells;
  the gate's subjects are B2′/L2′ = the v6 cells and the quantity is the VALUE —
  fixed here; the tl-lane wandb values quoted there (Ls5b400 ~.11–.14, Ls5btl400
  ~.13–.17 at ep2) stand as recorded but carry no gate.

**q2 ep25→50 share-drift knot (RAW, pre-registered read; warm lines only, 00:57
08-26; segment-head artifacts excluded per rule — joint read owed):**
| entering ep | inv share (g) | mkl share (g) | ω_h | ω_z | lam | probe after ep |
|---|---|---|---|---|---|---|
| 25 | .846 (.300) | .152 (.003) | .639 | .330 | 1.391 | .4872 |
| 51 | .831 (.321) | .167 (.004) | .522 | .236 | 1.487 | .5592 |
| 52 | .816 (.350) | .182 (.005) | .536 | .251 | 1.460 | .5654 |
| 53 | .871 (.375) | .128 (.003) | .515 | .220 | 1.530 | .5671 |
**The shares do NOT drift**: inv .82–.87, mkl .13–.18, h ~.002 at BOTH ends of the
span (neighbors ep23/24/26 in the same band) — the share composition is stationary
across ep25→50. What moves: ω_h −18% (.639→.52), ω_z −30% (.330→.22–.25), lam
+7–10% (1.39→1.46–1.53), own/gentle channels down in step; probe +7.4 pts
(.487→.562), the steep late climb continuing (.5671 @ep53). Restart note: ep51
probe .5592 dips below the pre-kill ep50 .5615 (the gpu274-discarded steps
re-trained), ep52+ resumes the climb — recovery clean.

**L2′ TWIN (e27v6L100, 63669110) chain-ep1 recalibration + early value read (RAW,
2026-08-26 ~17:45; started 15:42 on gpu268 = the grid's 12/12 completion):** ep0
shares (.008/.980/.013), ep1 shares **(.792/.199/.009)**, g = (2.470/0.021/0.041) →
**Σw·g = 44.0 = ×0.77 of the Σ=57 target** at the shared L2′ doses w =
(14.15/411.9/10.20) — undershoot where the main overshot ×1.04; ep1 probe .0520
(main's pilot .0554 band ✓). omega_h 1.176 / omega_z 1.456 / lam .899 — no
kill-trigger. **mkl VALUE (wandb nrkcs741, train/moment_kl):** .92 (~step 500) → .44
(~1.3k) → .159 (~ep1-end 2.5k) → **.088 by ~ep2-end (step ~5k)** — the L2′-sibling
quench signature PRESENT (at/below the P-m-2 .1 line by ep2, the main read
≈.095@ep2→.047@ep5), while the ep1 SHARE holds .199. Adjudication per the §(l)
both-agree amendment, same as the main: instruments disagree → NO KILL; the twin
joins the main's share tripwire watch (kill only if the share follows the value
below .1). ep2–5 settle lines owed on print (~48–60 min/ep).
**Twin ep2–5 settle (RAW, completed ~20:50 08-26):** shares (.856/.140/.004) →
(.808/.187/.005) → (.882/.116/.002) → (.802/.193/.005) — mkl share OSCILLATES
.116–.193 about ~.16, holding above the line (runs slightly higher than the main's
.13–.15). Σw·g 44.0 → 48 → 16 → 30 → **8.8 at ep5** (the family's settled level;
q1 7.0, B2′ 18.7). Probe .0520 → .1003 → .1681 → .2369 → **.3034 @ep5** (Ls5b's
matched-epoch ep5 = .4087; −10.5 pts, the all-global formation trailing the mc
incumbent early — RAW, the registered overlay read is mid-run+late). mkl VALUE
(nrkcs741) .159@ep1-end → .088@~ep2 → **.041–.048 at ~ep5** — the sibling
signature through formation (main: .095→.047@ep5). Formation-gate window CLOSES:
value-quench present, share holds → instruments disagree → NO KILL; per the
2026-08-26 standing, any future trigger is REPORTED to Berker (alert-only). Watch
continues via the twin share tripwire + the staleness sweep.
**Twin ep14 share excursion (RAW, 08-27 ~07:0x, alert-only report):** tripwire fired
on ep14 mkl share **.086** — first sub-.1 sample; series ep6–13 oscillates .130–.203,
and the ep14 line rides an inv-g jump (.537→.744 denominator) with mkl g pinned at
the one-digit .002. VALUE (nrkcs741) .029–.031 at ~ep14 — sustained quench, the
main's sibling (.026@ep23). Same shape as the main's ep23 event: single-sample
excursion, share not established as "following the value down". REPORTED, no action
(Berker decides).
**Main ep38 second excursion + pattern statement (RAW, 08-27 ~09:0x):** tripwire
fired again on ep38 share **.083**; the intervening ep24–37 series oscillated
.101–.146 (median ~.12, vs ~.13 over ep11–22 — the band's centre compressing slowly
downward), and both sub-.1 samples (ep23 .096, ep38 .083) coincide with inv-g
denominator spikes at pinned mkl g=.002. Standing characterization: recurring
ISOLATED dips on denominator spikes, no consecutive or sustained sub-.1 run yet.
Alert-only reports continue; the card updates on pattern CHANGE (consecutive
samples, or the band centre itself crossing .1).
**TWIN PATTERN CHANGE — two CONSECUTIVE sub-.1 samples (RAW, 08-27 ~10:0x):** twin
ep17 .098 → **ep18 .068** (after ep14 .086 / ep15 .165 / ep16 .124). Under the
ORIGINAL §(l) both-agree rule this would have met the kill condition (value
quenched .03 + share following); under the 2026-08-26 alert-only standing it is
REPORTED, no action. Context: ep18's inv-g spiked to 1.394 (largest since
formation; typical .4–.7) — the dip is again denominator-heavy — and the twin's
probe is HEALTHY and closing on its frame-mate: .5871@ep18 vs Ls5btl100's .6003
(−1.3 pts, from −10.5 at ep5). Berker's call owed on the pair's standing.
**RULED (Berker 2026-08-27, verbatim: "loss curves for e27v6l100 looks pretty
healthy. im not sure about relaunching it. we can wait a bit more to view how
healthily it improves the losses."):** the pair's standing = WAIT-AND-WATCH — both
cells keep running, no kill, no relaunch; the read stays on loss/probe health.
Tripwires remain armed as alert-only reporters.

**L2′ MAIN ep23 share excursion + tripwire adjudication (RAW, 2026-08-26 ~18:0x):**
the share tripwire fired on ep23 **mkl share .096** — the series' first sub-.1 sample
(ep11–23: .184 .131 .118 .136 .146 .142 .125 .128 .135 .122 **.100 .152 .096** —
oscillating ±.03 about ~.13, no segment restart in play). Cross-check: mkl VALUE
(qd0gilvf) is **sustained deep-quenched, median .0266 over the last 500 steps**
(.095@ep2 → .047@ep5 → .026@ep23) — the value leg of both-agree holds. Ruling under
the §(l) both-agree amendment: ONE sub-.1 share sample inside the series' own
oscillation does not establish "the share follows the value down" → **NO KILL on a
single sample.** **KILL AUTHORITY WITHDRAWN (Berker 2026-08-26 eve, verbatim: "dont
kill without my permission just report and i will decide"):** all watch tripwires in
this round are ALERT-ONLY — on any trigger (incl. two-consecutive sub-.1 shares) the
reading is REPORTED and Berker decides. Tripwires stay armed on both L2′ cells.
Mechanism note from the same exchange (Berker: "if h_moment loss is too small maybe
its share being small is understandable?"): substantially right — at value .026 the
mkl gradient is g ≈ .002–.004 (vs inv .5–.6), so share ≈ 411.9·g/(Σ≈9.4) rides a
one-significant-digit gradient print and the .10↔.15 wobble is noise-sized; the
share's design value (q2 held .13–.18 STATIONARY at value .03, so shares do NOT
automatically fall when a loss is satisfied) requires a SUSTAINED decline to mean
anything. Probe check at the event: v6L400 ep16→23 .5690→.6161 climbing, −1.1 pts vs
Ls5b's matched-epoch .6275 under the declared 400-cosine lr confound — no
degeneration signature.

**100-ep twin finals + the owed-bench ruling (RAW, 2026-08-31):** all four D-103 twins
are done (online probe, final ep): `e27v6b100` .7239 (benched 08-28: 74.15/64.28 — the
B winner) · `e27v6L100` .7290 (the L winner) · `e27lm4sbetl100` .6826 vs B1 incumbent
.6858 (Δ −0.3) · `e27lm4Ls5btl100` .7127 vs L1 .7130 (Δ −0.03). The P-m-3 online
referee reads B3≈B1 and L3≈L1 — the eta_min tail does nothing at 100 ep; the bench
referee = the same read at the ep400 endpoints. Berker compute ruling (2026-08-31,
verbatim): "if we dont think they are gonna be the winner (close online gap to the
winner of that arch) then we dont waste compute and we take a note what we tried and
what the online acc is." Applied: `e27v6L100` landing chain launched (extract 63937543
→ house probe 63937544 + bench 63937545); NO bench for `e27lm4sbetl100` (−4.1 online vs
v6b100), `e27lm4Ls5btl100` (−1.6 vs v6L100), `e27lmcs5q2` .6225 (−1.7 vs the S comparator
`d256vm4` .6392 — Berker 2026-08-31: the v4 line is the S yardstick, `e27v10u` .6662
is a significantly-more-compute lane and not the comparator; NOTE `e27lm4s5b` .6774 is
ViT-B despite the tag — checkpoint `model_name`, not tag, per the standing trap). The q2 ring-vs-OAS estimator A/B read
rides the online curves + training instruments; this note is the record of what was
tried.

**The 400-round trim (D-106, Berker 2026-08-31 "lets kill sbetl400, Ls5btl400 and
Ls5b400"):** killed mid-run with `_last`+ep100 ckpts intact — `sbetl400` @ep188 online
.6263 · `Ls5btl400` @ep161 .6957 · `Ls5b400` @ep160 .6963. Survivors to ep400:
`sbe400` (P-m-1 + P-m-2's matched-recipe endpoint comparator at B) · `v6b400` ·
`v6L400`. P-m-3's 400-tail read is FORFEITED at both arches — its referee is the
100-ep twins' null (B3−B1 −0.3, L3−L1 −0.03) + mid-400 tracking ±0.4; P-m-1-L
forfeited (B carries the read). Endpoint comparisons against the killed cells cite
their last matched epochs, never extrapolations.

**The v6s pair (D-107, Berker 2026-08-31 "lets do v6s!" — pre-registered BEFORE any
number exists):** `e27v6s400` + twin `e27v6s100`, the S member of the v6 family.
Frame `in1k_vits16`, bs128 DDP-2, aug=lejepa V=6@224 (1182 tok/sample; full ledger
×2.15 an OK-AI-S run per the D-106 accounting — the declared cost), ring z-q3/h-q3
with `h_d_slice=128` (n_eff/d′ = 4, the S anatomy), expander 256, swa=ema,
head_layers 2, mlp_wd .05, view_mean both floors, extra_cadence [10]. Procedure =
D-103 verbatim: pilot `e27v6s_pilot` at grid w (26.9/129.2/1.679, frame.epochs=3 —
warmup_ep=10 is fixed-count so pilot ep1 ≡ chain ep1 LR state) → chain doses
w_i = 57.0·τ_win,i/g_i^ep1, τ_win = (.441/.542/.017); doses shared 400/100;
120h singleton segments, exclusions gpu269/273/267.
**Predictions (committed 2026-08-31, pre-launch):**
- **P-v6s-1:** v6s100 bench_linear_v1 = vm4 + [2.5, 4.0] → [68.8, 70.3] (the
  v6b100−sbe100 analog band applied to vm4's 66.31).
- **P-v6s-2 (directional, the honest bet):** v6s100 lands BELOW OK-AI DINO-S-100
  (70.05) — at ×2.15 the winner recipe does not close the S block at 100 ep;
  beating it is the upside surprise.
- **P-v6s-3:** the 400-endpoint slope prediction is set when `e27v6b400` lands,
  before v6s400's endpoint is read.
- **Watch protocol:** ep1–5 formation gate (the only kill authority; mkl VALUE < .1
  by ep2 = the quench kill, P-m-2 convention); everything else ALERT-ONLY per the
  2026-08-26 ruling.

## Gates

- **D-079a: USER-APPROVED 2026-08-06 (Berker: "we launch s and then you start
  building b/l with smokes") — the S pair is OPEN on Recipe v2.** The vitb16/vitl16
  PROTOCOL rows land with the B/L build (owed pre-D-079b).
- **D-079b:** B opens on the S joint read (winner placement + health) + the B
  compute-layout call (single vs DDP-4).
- **D-079c:** L opens on the B joint read (+ grad_ckpt/DDP-8 decision — single-GPU
  L is ~a month).

## AGREED TAKEAWAY

**E27-T1 (USER-APPROVED, Berker 2026-08-10: "d087 agreed"):** *Scaling V=4→V=10
succeeds or fails on dose level and stream homogeneity — not on view count or crop
harshness.* Full four-clause wording in §(h.1) (dose level Σw·g ≈ 7–10 · aug axis
favors small/harsh locals, large locals collapse diversity · pure view-count scaling
pays monotonically · estimator follows stream homogeneity: ring↔homogeneous,
OAS↔heterogeneous). Scope ViT-S/in1k 5-ep health reads; the 100-ep wave is the
confirming instrument. Mirrored to DECISIONS.

*(further rows joint, as the wave lands)*
