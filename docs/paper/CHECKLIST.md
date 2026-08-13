# Paper checklist — one figure per question (D-100; Berker 2026-08-12)

Working rule: every item is a question the paper must answer, answered by exactly ONE
figure. Regenerate the deck with `python experiments/paper_checklist_figs.py`
(reads `results/twospace/`, `results/compare/`; figures → `results/figures/paper/`).
Statuses move only in conversation. Palette semantics fixed across the deck:
z = blue, h = orange, treatment(ours) = green family, control(lejepa) = violet.

| # | Question | Figure | Status | Notes / caveats |
|---|---|---|---|---|
| C1 | Does each method's stated desideratum live in z and fail to transfer to h? (Thm 2.1) | `C1_desideratum_z_vs_h.png` | **RENDERED — CORRECTED**: first render rode the e20f (treated!) lanes; now the untreated controls (`in100.<m>.s0.ext`). Control numbers are STRONGER: lejepa moment-KL .005@z vs 3.74@h (800×); vicreg floor .99@z vs .46@h; simclr = **cos margin per D-013** .84@z vs .40@h (raw alignment inverts on the control because h.cls is cone-y, rand-cos .48 — the margin is the honest cell) | EP is N-scaled → only N-matched manifests comparable |
| C2 | Does the h moment term protect stable center capacity? (Prop 3.2) | `C2_capacity.png` | **REWORKED** (Berker: rank-only, ±treatment) — E12 single-factor pairs (gvc/gv, gdc/gd, c1/f2) + ours; populates as e12 twospace jobs land | Θ dropped from this figure (lives in C6/C7) |
| C3 | Does the treatment improve accessibility? (Thm 4.1) | `C3_accessibility.png` | **REWORKED** (Berker: within-method ±treatment) — r_z is identical within a pair, so R²_acc compares fairly there and only there | cross-method bare R²_acc comparisons dropped (the r_z=16-vs-256 confound); held-out was already on — matching r_z is the actual fix |
| C4 | Does the view→center fidelity law Θ(I+Θ)⁻¹ hold? (Thm 5.2ii) | `C4_fidelity_law.png` | **RENDERED** — IN-100 only (Berker: one-dataset discipline; in1k version → appendix) | **Reading guard (Berker 2026-08-12): validates the meter, NOT "smaller Θ is better" — Thm 5.2(i) is the too-thin bound and G_h is required thickness.** small-λ z-directions sit slightly above the law (fit-noise floor) — quantify before paper text |
| C5 | How much retained thickness is excess (S) vs structured (G)? | `C5_gs_split.png` | **RENDERED (first pass)** — excess-share lower bounds .45–.87 at declared h; floor arms show higher excess share than their controls | **MLP caveat before quoting:** the 20-epoch MLP UNDERFITS on most runs (its bound lands BELOW the linear closed form — backwards; only lejepa-ctrl tightened, .45 vs .39). Fix: longer/wider MLP + report per-arm max(MLP, linear) — both are valid lower bounds |
| C6 | Does the moment treatment move thickness + accessibility over training? | `C6_trajectory.png` | **REWORKED** (Berker: IN-100 only, no dataset mixing; methods ± treatment + ours) — per-method small multiples; populates as the 21 early-ckpt `.o8` re-extract chains (63401591–63401677) drain (~1–2h) | early e12 stores were pairs-only → fresh o8 extracts under NEW run_ids `.ep<k>.o8` (nothing overwritten); in1k trajectories (e27oas/e27lej) stay a separate appendix figure |
| C7 | Where along depth does the bridge form (guillotine)? | `C7_depth_profile.png` | **REWORK PENDING** same treatment as C6 (Berker: "for that as well" — IN-100 only, methods ± treatment + ours); depends on whether e12 ep100 o8 stores carry layer taps — else ep100 `.o8` re-extracts queue too | current render mixes datasets — superseded |
| C8 | WHERE does the treatment's downstream change come from — organization or the view-fidelity term? (Thm 5.2iii as an explanatory tool) | `C8_term_shift.png` (+ `C8b_decomposition_identity.png` supporting) | **RETHOUGHT** (Berker: the identity scatter invites a "small Θ good" misreading; too-thin hurts too — Thm 5.2(i), and the treatment may raise Θ while helping probes). v2: per E12 pair, mean per-class Δ of each term (treated−control) + Δprobe marker; renders when e12 class rows land | the identity scatter (C8b) stays as the validation that the two terms are the exact split |
| C9 | Is objective (6.1) competitive at matched frame+FLOPs? | `C9_external_anchors.png` | **PENDING** — lightly offline linear/kNN lands today (bar 64.0/47.1); bench finals ~08-13; combos running; VISReg-B training since 08-12 ~12:20 | the E27 scoreboard figure: probe-vs-epoch, ours vs external anchor, per rung |
| C10 | (optional) Does B̂pool = B̂ + Â/V hold empirically? | — | **OPTIONAL** per Berker ("if we are sensitive and clean") | one afternoon on existing o8 stores if wanted |
| C11 | Too-thin exhibit (Thm 5.2i): do view-difficult distinctions require thickness — per class, view difficulty vs achieved d across spaces of different thickness? | `C11_too_thin.png` | **ADDED** (Berker 2026-08-12); design to agree BEFORE building: E_A(y) has no model-free estimator — proxy = best single-view predictor across available spaces (upper bound on recoverability → lower bound on E_A); natural thin extreme = lejepa's 16-dim z | the "floor" side of the thickness-budget story, symmetric to C4's "price" side |

## Lane definitions (pinned 2026-08-12 — "we should be on the same page")
- **Standing rule (Berker 2026-08-13): every treated/untreated measurement includes the
  ours pair** — control `in100.floorssl.s0.d256vm4zonly` (h_lamb=0, whole objective
  behind the MLP; D-100 twin) vs treated `in100.floorssl.s0.d256vm4` — in the deck
  (C2/C3/C5/C6/C7/C8), the E20 guillotine (8th row), and any ad-hoc ±treatment table.
- **Treatment pairs = the E20F wave** (`results/figures/e20/e20_guillotine_zoo.png`):
  control `in100.<method>.s0` (grey) vs floor arm `in100.<method>.s0.e20f` (colored);
  methods vicreg/simclr/byol/dino/lejepa (dino e20f = overshoot dose — flag on its cells;
  mae/ijepa have no o8 stores → battery-only). "ours" = `in100.floorssl.s0.d256vm4`.
- **e12 pairs (gvc/gv, gdc/gdc, c1/f2) RETIRED from the deck** — kept only as possible
  robustness rows; their early-ckpt re-extract chains were cancelled.
- **C1 exhibit lanes = the CONTROLS**, never the e20f arms.

## Caveats that must ride any cross-run reading
- **r_z comparability** (C3): R²_acc averages over the target's own whitened directions.
- **EP N-scaling** (C1): the Epps–Pulley statistic scales with N; match manifests.
- **declared h per run** (all): lejepa e20f stores declare h = z.embed (D-036); e27lej
  declares h.cls — the figure code resolves each run's declared h, never assumes.
- **pairs10 stores** (C8): 10 sources/class → per-class fits are noisy; pairs100 is the standard.
