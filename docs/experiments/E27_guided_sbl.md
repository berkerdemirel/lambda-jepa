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
**Pilot ep1 + chain launch (RAW, 63940159, 2026-08-31 ~13:05):** ep1 shares
**(.538/.438/.024)**, g = **(0.809/0.137/0.580)**, probe .0743, omega_h .804 /
omega_z .952 / lam .919 — family-band formation, no kill pattern. Law applied:
**w = 57.0·τ_win/g = (31.07 / 225.50 / 1.671)** — extrapolation ×1.16/×1.74/×1.00
from grid w (modest). S is LOADER-BOUND (~50 min/ep like B, the 6×224² crop
pipeline dominates, not the ViT) → chains sized 3×120h + 2×120h: `e27v6s400`
63941606–08, `e27v6s100` 63941609–10 (launched ~13:15). Chain-ep1 recalibration +
ep2–5 formation-gate read OWED on their first lines.

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

**v6s100 tail storm and roll-back (RAW, 2026-09-04; D-115):** the landing segment 64286559 (gpu271) printed its first
clipped gradient burst at step 849,724 (epoch 85) and 291 by 10:55 (pre-clip norms to 6e7); online probe ep84-93 =
.6512 .6532 .6518 .6518 .6515 .6529 .6532 .6557 .6538 .6560 (no drop, the tail gain resumed), but the per-step loss
median drifted 34.5 -> 39.5 and burst steps show the forward exploding (inv to 7e5). Cause read from the ep75 vs ep92
checkpoints: weight decay on the projector's pre-BN layers (norms 44.7 -> 32.0, 104.5 -> 78.4), the second BN's input
running variance collapsed 11x (1.9e-2 -> 1.7e-3). Berker: roll back to a checkpoint (urgent, health issue) -> segment
cancelled 10:55, `_last.pt` <- `_ep75.pt` (epoch index 74, step 750,675, best .6351), storm-era last/best kept as
`*_storm_ep92.pt`; first rerun segment 64290100 (unchanged recipe, gpu266) cancelled 11:09 before its first checkpoint;
segment 64435570 resubmitted from the same ep75 state with `method.mlp_wd=0` (Berker: "ok remove wd there"), then
replaced 11:26 by 64439249 with `+wandb_new_run=wd0` (fresh wandb run lx9tnd73 from step 750,675; the old run kept the
storm and dropped every lower step),
`train_ddp.py` now re-applies the config's weight decay after the optimizer state load. Landing moves to ~09-05 midday;
P-v6s-1/2 unchanged; the bench reads the landing checkpoint as before.

**The v6Llr pair (D-109, Berker 2026-08-31 "lets do adjusted lr on L. even if we only
have ep100 it is good. but we will try ep400 too." — pre-registered BEFORE any number
exists):** `e27v6Llr100` + `e27v6Llr400`, single delta vs the v6L pair = **method.lr
4e-3** (linear scaling from OUR B baseline 1e-3·bs512/bs128; under the family's
permanent clipping, step length = lr, so 4e-3 × 2,502 steps/ep = the B cells'
per-epoch path exactly). Everything else = the v6L line verbatim (bs512 DDP-4,
V=6@224, ring z-q3/h-q11, h_d_slice 384, swa=ema, grad_ckpt). Procedure: pilot
`e27v6Llr_pilot` at grid w (26.9/129.2/1.679, epochs=3) → w = 57.0·τ_win/g^ep1 →
chains. **Cap sequencing (D-093 16-H100 budget): v6Llr100 launches on the pilot line;
v6Llr400 queues behind the twin's completion (~09-05 start, endpoint ~09-22) — the
paper's L read = v6Llr100 + the running v6L400; the lr-400 is the follow-up arm.**
**Pilot ep1 + chain launch (RAW, 63940396, 2026-08-31 ~14:30):** ep1 shares
**(.796/.201/.004)** — the L-family signature; g = **(0.549/0.029/0.041)**; probe
**.0672 vs the v6L pilot's .0554** (+1.2pp at ep1, the hotter steps already visible);
omega_h .860 / omega_z .975 / lam .939 — healthy, gate passing. Law applied verbatim:
**w = 57.0·τ_win/g = (45.79 / 1065.3 / 23.63)** — extrapolation **×1.70/×8.25/×14.1**
from grid w. **DECLARED LOUDLY: the h_lamb ×14.1 and w_floor ×8.25 are the program's
largest transplants, PAST q2's failed ×9.6** — the lr=4e-3 ep1 state simply carries
much weaker moment gradients, and the pre-registered system (law verbatim + the
ep1–5 formation gate with quench kill authority, mkl VALUE < .1 by ep2) is the
referee; the L2′ precedent (×6.08, landed ×1.04 on target) is the hopeful prior, q2
the cautionary one. Chains launched IN PARALLEL per the disabled-cap ruling:
`e27v6Llr400` 5×120h (63942171–78) + `e27v6Llr100` 2×120h (63942180/82), DDP-4.
Chain-ep1 recalibration + the formation window are the next reads; Berker veto
window: kills are cheap in the first hours.

**RE-DOSED 25 min later (D-111, Berker: "i think vit 6b dose ratios are the good
ones in terms of shares and forces" + Fable concurrence):** the verbatim-law chains
(63942171–82) killed at ~5 min (NO ckpts written under the tags — resume-trap
checked clean) and relaunched at the WINNER-share dose: τ = v6b400's settled
realized shares **(.655/.338/.007)** in place of v10u's τ_win → **w = 57.0·τ/g^ep1
= (68.01 / 664.3 / 9.73)**, extrapolations ×2.53/×5.14/×5.80 — all inside validated
territory (≤ L2′'s ×6.08). Wandb value check before the call: pilot ep1-end
moment-KL VALUE ≈ .45–.48 (family band; v6b .44, v6L .53) and h-moment ≈ 1.8 — both
terms carry real residuals, so the boosts demand work, not force a satisfied
constraint. `e27v6Llr400` 5×120h (63942217–21) + `e27v6Llr100` 2×120h (63942222/23).
**v6s intentionally NOT re-dosed**: its v10u-τ provenance matches how v6b/v6L
themselves were dosed — re-dosing would add a family confound; its transplants are
mild (≤×1.74) and its own gate is armed.

**Predictions (committed 2026-08-31, pre-launch):**
- **P-v6Llr-1 (formation):** lr=4e-3 passes the ep1–5 gate (clipping absorbs the
  scale — the step is longer, not less stable). Declared fallback: lr=2e-3 (sqrt) if
  the gate trips; a trip is itself a finding (L instability at B-matched step length).
- **P-v6Llr-2 (the step-starvation bet):** v6Llr100 final online ≥ v6L100 + 0.5
  (.7290 → ≥.734), equivalently bench(v6Llr100) − bench(v6b100) ≥ +0.8 — recovering
  the field's B→L band. A null (≤ +0.2) REFUTES step-starvation as the main cause of
  the small B→L delta.

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

**v6Llr100 landing (RAW, 2026-09-04 18:50; segment 63942223, 1 d 07:45 h, zero INCIDENT lines):** final online probe .6976
(ep98 .6970, ep99 .6970, ep100 .6976; best .6976). **P-v6Llr-2 read against its pre-registered thresholds: REFUTED** — the bet
was ≥ .734 (v6L100 .7290 + 0.5); the landing sits 3.1 points BELOW v6L100's final online, a null far past the ≤ +0.2 mark.
Berker (22:30): "i will conclude v6 large lr experiment as a failure" — the 400-epoch twin `e27v6Llr400` (segment 64290102, ep98/400
online .5550 at the cut; successor 64290103) was cancelled at 18:35 on his word. No bench chain was run (landing = bench-only on
request; none requested). Checkpoints stay under `outputs/` (`in1k.floorssl.s0.e27v6Llr100_*`, `_e27v6Llr400_*`) for the purge
list. Interpretation (D-111 / the L-program fate) waits for the joint read.

**v6b400 tail storm caught at onset — same mechanism, same fix (RAW, 2026-09-04 23:57, Berker: "do it").** Kill-trigger bursts: single
ones at epochs 230/235/239/245, then 25 in epoch 264 and 8 in the first third of epoch 265 (norms up to 62k, the trigger's running
mean inflated to 100–320); online probe stalled (.6909 at ep261 → .6874/.6899/.6884); Berker saw the z moment-KL spike on wandb.
Projector BatchNorm-2 running variance median/min: ep261 best 3.2e-2 / 6.2e-3 (1 of 2048 channels < 1e-2), ep264 last 2.3e-2 / 3.7e-3
(1 channel), no dead z dimension yet — the state is still clean by the zeroing criterion (the v6s100 storm reached 93 % channels < 1e-2
after seven storm epochs). Pre-BN weight norms had shrunk 116/192/105 (ep100) → 69/116/82 (ep264): the D-115 driver. Action:
segment 64286597 and its queued successor 64286598 cancelled; ep264 state kept as `_last_storm_ep264.pt`; the epoch-261 best
state placed as the resume file; new segment **64588207** submitted with `method.mlp_wd=0 +wandb_new_run=wd0` (restore recipe
otherwise verbatim; fresh wandb run `in1k.floorssl.s0.e27v6b400.wd0`). Cost: 3 epochs of compute. The new segment pends on
priority (every allowed H100 node full; the cancelled successor had carried the chain's age priority — lesson: for a mid-chain fix,
submit the fixed segment BEFORE cancelling the old successor only if the singleton lets it; otherwise accept the queue).

**v6s100 LANDED (RAW, 2026-09-05 10:45; wd0 segment 64439249, epochs 76–100, wandb lx9tnd73).** Final online probe .6734 (best .6735 at
epochs 98–99); the storm-era run had reached .6560 at epoch 93. The segment logged 236 kill-trigger bursts (1–19 per epoch, 4k–20k,
all clipped) with the z moment-KL never above 0.16 at any step (Berker's breakage indicator; the storm read p99 1.5 → 43), the typical
gradient norm flat at 33–35 through epoch 82 then rising to 43 at epoch 99 (the cosine tail), and the projector's BatchNorm-2 running
variance recovering from 1.9e-2 / 2.0e-3 (ep75) to 8.4e-2 / 4.7e-3 at ep100 with 2 of 2048 channels below 1e-2 and no dead z
direction. Per-step tables: `results/diag/v6s100_wd0_gradnorm_tail.txt`, `_burst_terms.txt`. Landing checkpoint
`outputs/in1k.floorssl.s0.e27v6s100_ep100.pt` (clean); the rolling copy removed. Evaluation chain launched on Berker's word
("kick off the eval for v6s100 on ssltransfer, bench etc (filling the tables)"), the v6b100 landing recipe: extract 64662974
(`.extL`, h_layers 3/6/9/12), bench 64662975, transfer (VISReg linear-probe block) 64662976, ADE20k seg 64662977 — all on the
`gpu` partition, pending on priority at 10:55. Numbers land in `results/probes/in1k.floorssl.s0.e27v6s100.extL.bench*.csv`,
`results/transfer/in1k.floorssl.s0.e27v6s100.visreg_lp.csv`, `results/seg/in1k.floorssl.s0.e27v6s100.seg.csv`; the paper tables
auto-fill from them (`experiments/paper_exhibits.py`).

**v6s100 transfer block landed (RAW, 2026-09-05 12:55; job 64662976, `results/transfer/in1k.floorssl.s0.e27v6s100.visreg_lp.csv`, the
VISReg linear-probe protocol, 8 datasets).** Mean 77.60 (v6b100: 80.90): dtd 71.1, aircraft 55.5, cars 62.5, cifar10 95.3, cifar100 81.6, flowers 90.4, food 75.3, pets 89.1. Bench and seg still queued.
**v6s100 kNN (RAW, 2026-09-05 13:58; store-side rider, job 64663134 on gpu242):** k200 / t0.07 on the extracted student CLS features =
**59.24** (t0.1: 58.56). References: v6b100 64.28, v6L100 64.44. The linear bench (64662975) is still queued on the gpu partition
(estimated start 09-06 14:00); the main-table row fills from the complete linear CSV + this kNN CSV.

**v6b400 storm RE-OPENED under wd=0 (RAW, 2026-09-06 05:30; segment 64588207, wandb z1o9jpmv).** Epoch 283: z moment-KL max .350,
5 steps > 0.3 (first excursions since the fix). Epoch 284 (half done at 05:45): 157 steps > 0.3 in RUNS OF EXACTLY FOUR (= the z ring's
`queue_steps=3` + the current batch), each opened by an outlier batch whose invariance loss reads 3 → 96 (typical .29) with a
gradient spike of 70 → 15,000 (clipped); the excursions grow through the epoch (.31 → 1.77) while the h moment-KL stays flat (.21)
and the typical step is unchanged (grad median 21, z-KL median .093). The projector's BatchNorm-2 running variances are LARGE here
(median 1.83, min .13, no channel near zero; pre-BN norms 168 / 291 / 139, growing without wd) — NOT the low-variance amplification
of the v6s100 storm: the B storm is an outlier-batch divergence at z that the z ring propagates for three steps. wd=0 delayed the
onset by ~20 epochs (ep264 → ep284), it did not remove it. Protected roll-back point: `outputs/in1k.floorssl.s0.e27v6b400_wd0_ep283_preonset.pt`
(end of epoch 283, best probe .7084; the epoch's last 700 steps carried the 5 mild excursions). Options prepared for Berker: (a) roll
back to ep283 + an outlier-batch skip guard (skip the update when the rank-max inv or z moment-KL exceeds N× its running mean;
draft `scratchpad/skip_outlier_guard_patch.py`; ~0.5 % of steps at the observed rate; the grad-norm-triggered guard would miss most
trigger steps, whose norms are 70–500); (b) let it run (the storm wastes ~1 h per epoch; the roll-back point is fixed). The segment
is left running pending his word (no cancel without confirmation).
**v6s100 bench landed (RAW, 2026-09-06 07:15; H100 job 64665288, 11 h 05 min, Lightly linear protocol, 90 epochs):** final 69.73, max
69.73 (epoch 86), top-5 89.0; kNN k200/t0.07 59.24. Main-table row `Ours ViT-S/16 100 = 69.7 / 59.2` written by the generator into
main.tex (main_lean.tex's stale 66.3 / 54.5 replaced by hand). Same shape and epochs: house lm4sbe 71.2 / 56.0, lm4s5b 68.6 / 58.3,
DINO (OK-AI) 70.0 / 64.7, iBOT (OK-AI) 70.9 / 65.7, LeJEPA (OK-AI) 62.5 / 45.5; the B cell v6b100 74.15 / 64.28. The generator also
filled the L-100 row (72.8 / 64.4) from the resumed v6L100 bench — reverted to '---' pending Berker's open L-100 row call. Seg still queued.
**v6b400 rolled back again with the outlier-batch skip guard (D-117; 2026-09-06 07:53, Berker: "diagnose then fix and replace").**
Epoch 285 of the storm read z-KL max 19.3 with 289 steps > 0.3, invariance outliers to 3,588, 60 gradient spikes > 4,000 (probe .7097,
hiding it). Segment 64588207 cancelled at 07:53; its state kept as `_last_storm2_ep286.pt`; the pre-onset epoch-283 copy placed as the
resume file; new segment **64684405** (successor 64684406) with `+skip_inv_ratio=10 +skip_zkl_ratio=3 +wandb_new_run=wd0g`
(`method.mlp_wd=0` kept; everything else verbatim) started at once on gpu266. Guard code: `experiments/train_ddp.py` (config-gated;
`train/skipped` logged per step; `[train_ddp] SKIPPED step …` lines in the log). Cost: three epochs of compute (283 → 286).

**v6b400 guarded segment, epochs 284–292 (RAW, 2026-09-06 15:30; segment 64684405 on gpu266, wandb mcryqln2 `.wd0g`; per-step table
`results/diag/v6b400_wd0g_epoch_tail.txt`).** Per log epoch — probe · skipped steps · INCIDENT lines · z moment-KL max / steps > 0.3 (rank-0 rows) ·
invariance-outlier steps (> 2) · grad-norm spikes > 4,000: ep284 .7083 · 80 · 2 · .423 / 12 · 45 · 1; ep285 .7084 · 54 · 6 · .271 / 0 · 44 · 5;
ep286 .7068 · 37 · 7 · .225 / 0 · 18 · 4; ep287 .7087 · 8 · 10 · .133 / 0 · 9 · 5; ep288 .7118 · 3 · 8 · .129 / 0 · 5 · 6; ep289 .7124 · 4 · 6 ·
.171 / 0 · 5 · 4; ep290 .7126 · 8 · 8 · .166 / 0 · 12 · 7; ep291 .7125 · 8 · 9 · .153 / 0 · 6 · 8. The guard's running means settled within the first
epoch (its 12 steps > 0.3 are the first 200 steps of the segment); the typical step is unchanged (grad-norm median 21.0 → 21.8, z-KL median
.092–.093, inv median .293 → .282); the INCIDENT spikes (4k–11k pre-clip norms, 4–8 per epoch, with inv and z-KL at their typical values) are
the v6s100-type backward spikes — clipped, not skipped, and not the storm's trigger. Projector (Frobenius norms |W0| / |W3| pre-BN, |W6| output;
BN-2 running variance median / min): ep283 preonset 168.3 / 290.7 / 138.8, 1.83 / .132 → ep291 184.7 / 321.1 / 148.9, 2.70 / .168 = +1.2 % per
epoch on W0 and W3 (the wd-0 growth continues under the guard); no BN channel below 1e-2, no dead z dimension. Clean copy `_wd0_ep291.pt`
(verified: BN read + no z-KL step > 0.3 in its epoch); `_wd0_ep283_preonset.pt` kept; the ep290 copy removed. Hourly watch continues.

**lm4sbe400 STORM ONSET (RAW, 2026-09-06 21:30; segment 63665085 on gpu270, wandb tpn49ue1; the LeJEPA-family control at B, `mlp_wd=0.05` still
on).** Kill-trigger bursts per log epoch: 0–3 through ep276 (133 over the segment), then **20 in ep277** and 25 more in the first part of ep278 —
pre-clip norms 1,045 → 98,813 with the running mean itself inflating 8 → 423; the online probe still rose (.6702 → .6709). Projector read
(`results/diag` convention): pre-BN norms |W0| / |W3| / |W6| 91.0 / 143.7 / 77.8 at ep200 → **64.2 / 101.0 / 67.4** at ep277 (−30 %, the D-115
driver: wd on the scale-invariant pre-BN layers at the cosine tail); BN-2 running variance median 3.7e-2 → **4.5e-3 with 2,048 / 2,048 channels
below 1e-2** (the v6s100 storm state read 93.5 %; here the zeroing criterion is already complete); BN-1 carries 55 zero-variance channels (the
quiet dead channels known since ep100); no dead z dimension. The per-step z-KL read is not available for this run (the wandb crawl of a 2.77 M-step
run times out — the D-115/116 limitation). Storm-era state kept: `outputs/in1k.floorssl.s0.e27lm4sbe400_last_storm_ep277.pt`. Roll-back points on
disk: `_ep200.pt` (ep199 state, 77 epochs back; BN-2 already thinning, 19 channels < 1e-2), `_ep100.pt`. Berker's word needed (the control lane is
untouched by every ruling so far): (a) leave it running (its comparator role is the matched-recipe endpoint; the state is past the zeroing);
(b) D-115 fix from ep200 with `mlp_wd=0` (+ the D-117 guard), 77 epochs ≈ 3.2 days; (c) continue from `_last.pt` with `mlp_wd=0` + the guard
(stops the driver, keeps the thinned state). No action taken; the hourly watcher reports its bursts per epoch.
**lm4sbe400 KILLED (2026-09-06 21:5x, Berker verbatim: "kill lm4sbe400. we can consider fixing it later. for me b start is the priority right now").**
Segment 63665085 cancelled in its storm epoch 278 (online probe .6709 at ep277 = its best); successor 64285285 cancelled. Kept on disk under
`outputs/`: `_last.pt` / `_best.pt` (the storm state), `_last_storm_ep277.pt`, `_ep200.pt`, `_ep100.pt`, `_ep10.pt`. No bench. The fix (D-115 from
ep200, or wd 0 + the D-117 guard from the storm state) is deferred; the two H100 cards go to the video B cell.

**v6s400 chain RE-SHAPED for speed (2026-09-07 20:38; Berker: "8.6 days is unacceptable. fix it.").** Diagnosis: both IN-1k chains are loader-bound — 12
workers per rank on a 28-CPU allocation deliver ≈375 images/s (six 224² views per image ≈ 63 ms of CPU each) against ≈8 min of ViT-S GPU time per
epoch, hence 57 min per epoch (S) and 55 (B) alike. Fix = the same recipe verbatim (`method.mlp_wd=0.05` and every other override unchanged) on
the same 2 GPUs with `--cpus-per-task=96 --mem=300G num_workers=44` (88 workers). Old segment 63941607 cancelled at 20:38 right after its ep183
checkpoint (`_last.pt` 20:36), old successors 64285269 / 63941608 cancelled (28-CPU scripts); new singleton segments 64912673 → 64912674 resume
`_last.pt`; exclusion reduced to gpu274. Expected ≈15 min per epoch (loader ≈1,500 images/s; GPU ≈8–13 min) → the remaining 217 epochs ≈ 2.3 days
instead of 8.6 — to be measured on the first epoch. The wd decision (D-115 clock, ≈ep200) is untouched by this change.
**Re-shaped S chain: first two segments died on gpu277 (RAW, 2026-09-07 21:34–21:43).** 64912673 and 64912674 both landed on gpu277 (the node that came
back from its "Prolog error" drain into MIXED) and failed in 4 min each: `CUDA initialization: Unexpected error from cudaGetDeviceCount() ... Error 802:
system not yet initialized` → `cuda_available False` → `ProcessGroupNCCL is only supported with GPUs`. The node is broken for CUDA work while SLURM
schedules onto it. Resubmitted 21:50 as 64924620 → 64924621 with `--exclude=gpu274,gpu277,gpu271` (same 96-CPU / 44-worker shape, recipe verbatim);
the pending B successors, the v6b400 successor and the video launchers now exclude gpu277 as well. `_last.pt` = ep183 (20:36) untouched. Lost: ≈1.3 h
of chain time so far.

**Why only segmentation trails — patch-token diagnostic (RAW, 2026-09-07 22:0x; Berker: "it is very interesting that only on segmentation task we fail
... over regularizing the backbone feats, therefore patch feats? please do a quick analysis"). Instrument `sslgap/metrics/patch.py` + `experiments/patch_diag.py`
(the seg protocol's own features: last-layer tokens, `forward_intermediates(norm=True)` at 512², ADE20k val, 200 images for the structure reads; a
training-free nearest-class-mean (NCM) on ADE20k majority patch labels, 300 train / 200 test images, as the dense proxy); tables `results/diag/patch_diag.csv`
(512), `patch_diag_224.csv` (224), `patch_diag_ablate.csv`.** (1) The deficit is in the frozen patch tokens: NCM top-1 orders the eight S/B-100 checkpoints
exactly as tab:seg does (S: ours .387 < LeJEPA .420 < DINO .458 < iBOT .471; B: .444 < .478 < .511 < .577). (2) Not a rank/collapse effect: per-image
effective rank .55 (S) / .43 (B) and pooled rank in the public band; no high-norm artifact tokens; neighbour-vs-random coherence the highest of all (.60 vs
.46–.54). (3) The missing piece is the image-level component inside the patch tokens: ours have 95–96 % of their variance within images and are orthogonal to
the CLS (cos .00 / −.01; the public models .11–.44, with 13–32 % of the patch variance image-level). With each image's mean removed the local content is
competitive (NCM S .358 vs LeJEPA .330 / DINO .383; B .404 vs .400 / .449), so the raw gap to DINO (S .071, B .067) is mostly the global channel
(centered gap .025 / .045). Same ordering at 224 — not a resolution effect. (4) Attribution (`patch_diag_ablate.csv`): our recipe WITH 2g+8l locals
(`e27v10u`, h on) keeps the signature (cos_cls −.01, within .96) and lm4sbe (B, 4g+6l, h on) too — local crops do not put the global component back; the
LeJEPA repro in our codebase (locals, no h) has cos_cls .69; the IN-100 twin pair is the clean test: z-only `d256vm4zonly` cos_cls .59 / within .83 → with
the h term `d256vm4` cos_cls .13 / within .90 (NCM equal at that scale, .277 / .276). Read (mine, not agreed): the h conditioner on the CLS centers makes the CLS
the sole, isotropized carrier of image-level content and decouples the patch tokens from it; a patch-only linear seg head then has no scene context, which
is what DINO / LeJEPA / iBOT patches carry. The obvious confirmation is a seg read whose head also sees the CLS (a protocol variant, not VISReg's) — Berker's
call. Joint read pending.
**Re-shaped S segment RUNNING; B re-shape launched (RAW, 2026-09-08 01:35).** 64924620 started 01:02 on gpu270 (CUDA fine; resume line weight_decay [0.05, 0.05,
1e-07], ep183 → ep184); the first epoch's checkpoint landed 01:32 — ≈27 min including the selftests and worker warm-up, against 57 min before; the
steady-state number is read at the ep185 stamp. On that ≥2× read and Berker's "in case it works, do the same for vit base (but first queue then cancel)":
fast guarded B segments 64925357 → 64925358 queued (96 CPUs / 44 workers per rank, 300G, `mlp_wd=0 +skip_inv_ratio=10 +skip_zkl_ratio=3` kept, no
`+wandb_new_run` so run mcryqln2 continues, exclude gpu274/277/271), the old 28-CPU successor 64684406 cancelled, and the swap helper 64925359
(`slurm/e27_swap_at_ckpt.sh`) cancels the running segment 64684405 within a minute of its next `_last.pt` write (ep329 ≈ 02:13). The guard's running
means restart at the resume (the first ~200 steps skip a little more, as after D-117) — the only effect beyond speed.
**Patch + CLS proxy (RAW, 2026-09-08 02:05; `results/diag/patch_diag_cls.csv`; Berker: "if we include cls to the segmentation task along with patches
then our miou is good?").** Nearest-class-mean on [patch ⊕ CLS] (both parts globally centered and unit-normed) vs on the patch alone (cosine): ours S .381 →
.419 (+.038), ours B .437 → .426 (−.011); LeJEPA S .417 → .366, B .473 → .398 (the CLS hurts); DINO S .451 → .520, B .502 → .566 (+.06–.07); iBOT S .470 →
.481, B .575 → .568. Read: INCONCLUSIVE for the question — a nearest-mean classifier weights the two parts equally and cannot learn how to combine them,
so it says only that a trained head is needed to test the context hypothesis; the measured facts (our patches carry no image-level component; the h term
removes it; our local content is competitive) stand, the claim that handing the head the CLS recovers the mIoU does NOT follow from this proxy (it helps
ours a little at S and not at B). The proper test = the VISReg seg head on [patch, CLS] (2d input) for ours and the comparators, ≈4 h per checkpoint on the
gpu partition — Berker's call; no run launched.
**Speeds measured (RAW, 2026-09-08 02:56).** v6s400 fast segment (96 logical CPUs = 48 physical cores on gpu270, 44 workers per rank): checkpoints ep184 01:32:17 →
ep185 01:59:42 → ep186 02:27:13 → ep187 02:54:52 = **27.5 min per epoch steady** (57 before, ×2.1); 213 epochs remain → ≈4.1 days (≈09-12 05:00). The bound is now
CPU cores: 63 ms of view pipeline per image × 1.28 M images / 48 cores ≈ 28 min — the nodes have 112 cores (2 × 56, 2 threads), so a 192-CPU ask would halve
it again if it could be scheduled; GPU-side photometrics would remove it. v6b400: the swap helper cancelled 64684405 at 02:13 (one minute after its ep330
checkpoint); the fast guarded segment 64925357 started 02:14 on gpu272 (resume line `[0.05, 0, 1e-07]`, guard on), first checkpoint (ep331) at 02:55:43 = 41.5 min
including selftests and warm-up (55 before), probe .7317, 15 skips / 20 INCIDENT lines in that epoch (the guard's means restarting); steady state at the next stamp.

**v6s400: wd 0 + guard applied in the fast shape (D-120; 2026-09-08 07:19; Berker: "do the wd=0 fix with the guard. make sure we stay in high speed config.").**
State at the switch (ep196, `_last.pt`): pre-BN norms 66.0 / 153.5 / 107.3 (ep100: 82.1 / 191.9 / 117.7), BN-2 running variance median .096 (ep100 .277, ep182
.122), min .044, no channel below 1e-2, no dead z dimension, zero bursts in the fast segment (ep184–196 probes .5936 → .6012). Queued: 64926249 → 64926250
(`method.mlp_wd=0 +skip_inv_ratio=10 +skip_zkl_ratio=3 +wandb_new_run=wd0g`, 96 CPUs / 44 workers per rank, exclude gpu274/277/271, everything else verbatim);
the wd-.05 successor 64924621 cancelled; swap helper 64926251 cancels the running wd-.05 segment 64924620 within a minute of its next `_last.pt` write (ep197 or
ep198). Verification from then on = the hourly per-epoch read (skips, z-KL excursions, projector norms and BN-2 variance, clean copies `_wd0_ep<N>.pt`) as for v6b400.
Switch executed 07:29–07:31: the wd-.05 segment wrote `_last.pt` (log ep197, probe .6060) and was cancelled by the helper; 64926249 started on gpu270 at 07:31, `resume: weight_decay per group from cfg = [0.05, 0, 1e-07]`, resumed at epoch index 197, new wandb run `in1k.floorssl.s0.e27v6s400.wd0g` (1ay07kmz) from step 1,971,773 (= 197 × 10,009; the old run 9ibtanir keeps its history); wandb config verified: `skip_inv_ratio` 10, `skip_zkl_ratio` 3, `mlp_wd` 0, `num_workers` 44. Per-epoch record from here: `results/diag/v6s400_wd0g_epoch_tail.txt` (appended by the hourly watcher).
First guarded epoch (log ep198 = bin 197, 07:31–07:59, 30 min including the segment start; RAW): probe .6062 (ep197 .6060); guard skips 0 (rank-0 view and log), AMP-skipped steps 4, INCIDENT lines 2 (grad norm 3652 / 4249 vs running mean ≈33 — below the guard's inv/z-KL triggers: inv max 2.16, z moment-KL max .275, no step > .3, median .124); grad-norm median 26.9, p99 293. Projector after one wd-0 epoch: pre-BN norms 80.9 / 188.5 / 123.8 (ep196: 66.0 / 153.5 / 107.3), BN-2 running variance median / min 0.276 / 0.102 (ep196: .096 / .043), no channel below 1e-2.
**wd-0 segment, epochs 198–208 (bins 197–207; 2026-09-08 13:10 read; RAW, report only).** Probe .6062 → .6156 (best .6174 at ep207; ep197 was .6060), 28.2 min/epoch.
The guard's skips and the grad-norm tail rise monotonically while the z moment-KL stays flat (the breakage indicator has not moved) and the
grad-norm median is flat; the projector norms grow 23 % / 16 % / ≈10 % / … / ≈4 % per epoch (65.6/152.9/107.2 at ep196 → 167.2/394.2/191.9 at ep207),
BN-2 running variance median .096 → 7.1, min .043 → 1.1, no channel below 1e-2. Per epoch (bin = log epoch − 1; rank-0 rows of run 1ay07kmz):
| bin | skips | z-KL max | inv max | inv > 2 | gn median | gn p99 | gn max | gn > 1000 |
|---|---|---|---|---|---|---|---|---|
| 197 | 0 | .275 | 2.16 | 1 | 26.9 | 293 | 4249 | 16 |
| 199 | 0 | .155 | 2.44 | 10 | 25.8 | 643 | 4061 | 57 |
| 201 | 5 | .189 | 4.88 | 17 | 25.5 | 998 | 11246 | 100 |
| 203 | 5 | .163 | 4.36 | 23 | 25.3 | 956 | 8626 | 98 |
| 205 | 10 | .162 | 5.80 | 32 | 25.4 | 1119 | 10886 | 116 |
| 207 | 22 | .176 | 6.12 | 33 | 25.4 | 1553 | 18377 | 141 |
(full table: `results/diag/v6s400_wd0g_epoch_tail.txt`). For reference, v6b400's guarded epochs 327–344 sit at 11–28 skips per epoch with a stable tail.

**v6s400rb STORM ONSET (RAW, 2026-09-12 22:0x; segment 65416997 on gpu272, wandb ioy1ojnu; the roll-back branch: resumed at ep200 with W0/W3 frozen at
their checkpoint values 104.7 / 244.9 and `mlp_wd` .05 on W6/BN, D-117 guard WITH ring-block eviction `skip_zkl_evict`).** After 100 epochs with zero
guard trips (bins ep200–299: skips 0, z moment-KL max ≤ .24, no step > 0.3), bin ep300 (log ep301) opens the D-117 signature: first trip at step
3,007,900; per epoch (bins 300 / 301 / 302 / 303-partial) skips 28 / 217 / 221 / 181, z-KL steps > 0.3 106 / 257 / 290 / 279 (max .61; runs of 1–4
steps = the q3 ring + the current batch), invariance max 5.5 / 20.2 / 16.0 / 17.0 against a typical .37, grad-norm p99 596 / 814 / 1,426 / 1,301 (max
to 9,012; two kill-trigger incidents at steps 3,013,899 and 3,016,978), while the typical step is unchanged (z-KL median .110, inv median .37,
grad-norm median 52); h moment-KL median .315 → .322. Online probe ep299–303: .6558 / .6570 / .6526 / .6547 / .6560. Projector: W0/W3 frozen,
|W6| 100.9 → 100.8; BN-2 running variance median .085 (ep297) → .067 (ep302), min .035 → .025 (the trend since ep203: .34 / .16 / .085 / .067), one
dead BN-1 channel inherited from the parent. Eviction is LIVE on this branch (105 evictions in 664 skips; the fz2 launch has it off, 0 in 50,294) —
the variant HANDOVER §2 records as the cause of the first freeze attempt's 10,009 / 10,009 lock-up. **Clean states:** rolling verified copies
`_wd0_ep203 … _wd0_ep298` (each BN-2-clean and z-KL-clean at write time; `_wd0_ep298.pt` = epoch index 297, step 2,982,682); the newest pre-onset
state, epoch index 299 (end of log ep300, step 3,002,700, best online .65696; its epoch: 0 skips, 0 steps > 0.3, 17 > 0.2, 2 grad-norm steps > 4,000),
copied 22:06 from `_best.pt` to **`outputs/in1k.floorssl.s0.e27v6s400rb_wd0_ep300_preonset.pt`** (byte-identical, verified epoch 299, BN-2 median
.080 / min .031) because `_best.pt` is overwritten on any later probe improvement. **v6s400fz2 has no z-KL-clean epoch** (74 bins since ep304, every
one with 470–1,000 z-KL steps > 0.3 and 500–1,000 skips = the parent storm carried in and discarded by the guard; BN-2 median 127 / min 1.2, far from
the zeroing state); its `_best.pt` (epoch 377, .67982) and `_last.pt` are storm-era by that rule. The parent wd0g chain's last z-KL-clean epochs were
log ep225 / 232 / 237 (copies on disk `e27v6s400_wd0_ep217/219/224/225/232.pt`; from ep239 excursions rise 21 → 143 at ep277, 593 at ep282, 2,572 at
ep303). Nothing touched; the run continues under the guard; the decision is Berker's ("it is essential that we have clean checkpoints for both").

**v6s400rb ROLLED BACK + the block hold (2026-09-12 22:2x; Berker verbatim: "i want rb to be rolled back. and what is the treatment you think we should be
employing?" → recommendation given → "do it i trust you. on fz2 we will keep it as is. that model is training fine i think still (linear probe has improved
over the course of ~80eps) so these will be two variants where one has a full clean training and the other has some movements and guards.").** Executed:
segment 65416997 + successor 65416998 + the chained evals cancelled 22:1x (the storm-era state kept as `outputs/in1k.floorssl.s0.e27v6s400rb_last_storm_ep304.pt`,
epoch index 303); the pre-onset epoch-299 state (`…rb_wd0_ep300_preonset.pt`, step 3,002,700, best online .65696) copied to `…rb_last.pt` (verified epoch=299);
relaunched 22:2x as **65736100** (started at once on gpu267; spare singleton 65736101; evals ex 65736102 → bench 65736103, tvlp 65736104 on `_ep400.pt`) with rb's
own submit line minus `+skip_zkl_evict=true` (the eviction variant of the guard, the first freeze attempt's lock-up), plus **`+method.freeze_prebn_bn=true`**
(new, `experiments/train_ddp.py`: the projector's BN-1/BN-2 held in eval mode with their affines fixed, re-applied after every train-mode switch — the forward
uses the resume-point running statistics, so the block's gain 1/√(running var) is pinned at the resume values, median ≈3.5, worst channel ≈5.7, and the
cross-image batch coupling inside the block is gone; the block Linear→BN→ReLU→Linear→BN→ReLU is a fixed map for epochs 300–400, W6 and the trunk train under
the unchanged losses and doses; the function at the resume point changes only by batch-vs-running normalisation) and `+wandb_new_run=frzW3bn`. Why the hold
(the measurement): with W0/W3 frozen the block still drifted — BN-2 running variance .34 (ep203) → .16 (ep250) → .085 (ep298) → .067 (ep302), min .034 → .025 —
i.e. the amplification rose ≈2× over the clean 100 epochs and the D-117 outlier storm opened one epoch after the roll-back point; at that rate the gain would
double again by ep400. Prediction (pre-registered here): skips in the tens per epoch, z-KL excursions confined to the ring's 4-step runs, no roll-back needed;
the alternative (pure roll-back) was expected to re-form the storm within epochs. Declared deviation for the S-400 rb row: epochs 300–400 with the projector's
first two blocks fully held (weights + normalisation) and the D-117 guard. **fz2 kept as is** (Berker): its online probe .6550 at ep305 → .6809 at ep379 on the
branch; the two branches are two variants of the S-400 tail — rb "a full clean training", fz2 "some movements and guards" (his words) — and which is Ours S-400
is his call at landing. Landing ≈ Tue 20:00 (100 epochs at 27.5 min) if gpu267 holds the pace.

**v6s400rb under the block hold — first 4 epochs (RAW, 2026-09-13 00:1x; segment 65736100 on gpu267, wandb 11qwdmkx, 26.9 min/epoch).** Guard: skips
0 / 0 / 0 / 0 (bins ep300–303), z moment-KL steps > 0.3: 0, max .26 / .13 / .13 / .13, median .109 → .103; grad-norm median 41.5 → 39.2, p99 63 → 53 (was
300–600 before the onset), max 633 / 85 / 74 / 110, no step > 1,000 (was 10–130 per epoch). The pre-registered expectation (skips in the tens, excursions
confined to the ring's runs) is met with room. **What the hold changed beyond the guard (facts, not read):** online probe .6570 (ep300, pre-onset) → .6604 /
.6650 / .6662 (ep301–303; +.003 per epoch against +.0005 per epoch over ep250–300); per-step medians inv .37 → .295 / .276 / .272 / .270, h moment-KL .315 →
.295 / .273 / .256 / .244 (falling ≈ .015 per epoch; flat at .31–.32 over the previous 20 epochs); |W6| 100.7 (ep299) → 92.0 (ep301) → 82.1 (ep303), i.e.
≈ 5 % per epoch against 0.2 % per epoch under wd .05 before the hold (the unopposed AdamW decay at lr ≈ 1.6e-4 × wd .05 over 10,009 steps is ≈ 8 % per
epoch, so the gradient now opposes the decay only weakly; at 5 % per epoch |W6| would reach ≈ 0.6 by ep400 if nothing settles); omega_h .211 → .210 and
omega_z .135 → .117 on the share batch; lam 1.25 → 1.34. The share line at the resume step (ep300, fixed batch) read moment_kl share .809 (g .966 against
.056 before) and relaxed to .215 (g .049) by ep301 — the batch-vs-running normalisation change showed as a one-epoch z-KL transient on the fixed batch,
not on the training steps. Clean copies resume under the rule (`_wd0_ep303` written). Watch items for the hourly line: |W6| per epoch, h moment-KL, the
probe. Nothing touched.

**v6s400fz2 LANDED (RAW, 2026-09-13 07:35; segment 65440875 COMPLETED after 1 d 19 h 20 min on gpu269, wandb 81obfvyw; the spare 65440876 found the run
complete in 5 min).** Final online probe .6812 at ep400 (best .6826 at ep397; ep305 .6550 at the branch start); `outputs/in1k.floorssl.s0.e27v6s400fz2_ep400.pt`
(epoch index 399, step 4,003,600) = `_last.pt`; `_best.pt` = ep397. Projector at landing: W0 / W3 frozen at 336.7 / 784.6 (the grown ep304 values), |W6| 317.3;
BN-1 running-var median 22.5 (one dead channel, inherited), BN-2 median 130 / min 1.03, no dead z dim. **The branch ran in the guarded storm state
throughout** (bins ep304–399 from the per-step rows): skips per epoch mean 675 (498–988), z moment-KL steps > 0.3 per epoch mean 641 (441–1,280), z-KL max 3.28,
grad-norm p99 ≈ 3e5 and max to 5.9e6 on the skipped steps against a median of 86 → 113; 64,777 of 960,864 steps skipped = 6.7 %; no z-KL-clean epoch, so no
verified clean copy exists on the branch (Berker 2026-09-12: kept as is — "some movements and guards"). Landing chain running: extract 65719628 → bench
65719629 (pends), transfer 65719630 (running) on `_ep400.pt`; the bench-only rule applies (in1k landing). Numbers RAW until the bench lands and the row is jointly
read against rb's.
**fz2 extract INCIDENT (2026-09-13 09:5x): `ex-v6s400fz2` 65719628 on gpu241 (RTX 3090 node) finished its GPU pass and then wrote the train-split store to
BeeGFS at ≈1.5 MB/s (10 of 13 files in 51 min, the 5.2 GB `z.proj.tap1` at 43 % after 16 min; GPU 0 %, the process at its usual ≈100 GB RSS — the same
footprint as the S-100 and B-400 extracts, which took 51 / 66 min); the projection put the finish ≈10 min past the 4 h wall limit, `scontrol` refused a
limit raise, and a rerun starts from scratch (the store `put` overwrites). Cancelled at 2 h 18 min together with its bench 65719629, the partial store
removed, resubmitted as extract **65737727** (`--constraint=L40S`, gpu278) → bench **65737728** (`afterok`). A 256 MB direct write to BeeGFS from the login
node measured 10.9 MB/s at 10:0x — the file system was slow cluster-wide this morning, not only from gpu241. The transfer read (65719630, 78.55) is
unaffected (it reads the checkpoint, not the store).**
**fz2 extract, second rerun with the store on NFS (2026-09-13 11:56).** The L40S rerun 65737727 finished its train pass in 76 min and then wrote the store to
BeeGFS at ≈2 MB/s again (4.3 GB in 37 min); direct-write tests from the login node at 11:53: BeeGFS 2.8 MB/s, NFS 2.1 GB/s. Cancelled at 1 h 59 min with its
bench; this run's store directory on BeeGFS replaced by a symlink `features/in1k.floorssl.s0.e27v6s400fz2.extL → features_nfs/<same>` (project NFS, 1.1 TB
free; ≈23 GB; the extractor and the bench resolve it transparently; to be moved back to BeeGFS after the table per D-005's purge-after-table); resubmitted
as extract **65738215** (gpu283, L40S) → bench **65738216**. rb's chain (65736102/03/04) is untouched — re-check BeeGFS before it lands Tuesday.

**v6s400fz2 BENCH LANDED (RAW, 2026-09-14 ~05:0x; bench 65738216 COMPLETED in 15 h 29 min on the store written to NFS; bench_linear_v1, 90 epochs on the
stored CLS features):** max val top-1 **71.13** at bench epoch 85 (last 71.12; the curve flat 71.10–71.13 over epochs 85–90); kNN rider k 200: t .07 **62.30**,
t .1 61.98. Transfer (VISReg protocol, from 09-13) **78.55** mean. Against the landed cells: S-100 69.73 / 59.24 / 77.60, B-400 75.99 / 68.57 / 81.97. Not
entered in tab:video's S-400 rows: which branch is "Ours S-400" is Berker's call once rb's bench lands (rb at ep377 online .6962 vs fz2's .6812 at ep400).

**v6s400rb LANDED (RAW, 2026-09-14 19:47; segment 65736100 COMPLETED after 1 d 21 h 32 min on gpu267, wandb 11qwdmkx; the spare 65736101 found the run
complete in 6 min).** Final online probe .6978 at ep400 (best .6981; .6570 at the ep300 roll-back point); `outputs/in1k.floorssl.s0.e27v6s400rb_ep400.pt`
(= `_last.pt`). **Zero guard skips and zero z moment-KL steps > 0.3 over the 100 epochs under the block hold** (per-step bins ep300–399); the projector's
W0 / W3 / BN-1 / BN-2 held throughout, |W6| 100.7 → ≈50. Separate verified copy at ep348 (`…rb_hold_ep348.pt`). Landing chain running on the ep400
checkpoint: extract 65736102 (gpu241, store → NFS via the pre-created symlink) → bench 65736103; transfer 65736104 running. Against fz2 (online .6812 at
ep400, bench 71.13, kNN 62.30, transfer 78.55): the online probe reads +1.7; the bench decides. Numbers RAW until the bench lands and the row is jointly read.
**v6s400rb TRANSFER LANDED (RAW, 2026-09-14 20:5x; job 65736104, 1 h 06 min; VISReg protocol):** mean **79.25** — DTD 71.8, Aircraft 57.9, Cars 67.9, CIFAR-10 95.5,
CIFAR-100 81.8, Flowers 90.3, Food 78.1, Pets 90.7 — against fz2 78.55 (rb ahead on 7 of 8 sets, Flowers −0.1), S-100 77.60, B-400 81.97. Extract 65736102 in
its GPU pass on gpu241 (store → NFS), bench 65736103 behind it.

**v6s400rb BENCH LANDED (RAW, 2026-09-15 13:5x; bench 65736103 COMPLETED in 16 h 14 min on the NFS store; bench_linear_v1):** max val top-1 **72.24** at bench
epoch 87 (last 72.24); kNN k 200: t .07 **62.90**, t .1 62.36; transfer 79.25 (09-14). Against fz2 71.13 / 62.30 / 78.55, S-100 69.73 / 59.24 / 77.60, B-400
75.99 / 68.57 / 81.97. **Converged unaugmented linear read on the same stored CLS features (job 66005309, `experiments/lbfgs_linear.py`, ridge logistic
by L-BFGS on standardized features, λ ∈ {1e-5, 1e-4, 1e-3}, `results/diag/lbfgs_linear.csv`; Berker's question "how much we gain from augmentations"):**
best λ = 1e-5 for every cell — rb 71.73 (train 77.40), fz2 70.61 (75.72), S-100 69.14 (75.39), B-400 75.38 (83.66); bench − L-BFGS = +0.51 / +0.52 / +0.59 /
+0.61, i.e. the Lightly recipe's augmentation + BatchNorm head + LARS schedule over 90 epochs read ≈0.5–0.6 above the optimum of the unaugmented linear
problem, the same offset in every cell and the same ranking (the K400 20-epoch head, by contrast, read 6–8 under its optimum). Which branch is Ours S-400
is Berker's call: rb leads fz2 on the bench (+1.11), kNN (+0.60), transfer (+0.70) and the converged read (+1.12), with zero guard skips over its tail.
