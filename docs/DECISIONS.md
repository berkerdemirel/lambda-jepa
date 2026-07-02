# DECISIONS.md — the ledger

Append-only. Every project-shaping decision and every agreed experimental takeaway gets a row.
**Rule (see WORKFLOW.md): nothing here reaches status USER-APPROVED without Berker's explicit sign-off
in conversation; no conclusion is adopted, no protocol changed, no data-ladder rung advanced without
a USER-APPROVED row.** Superseded rows are struck through, never deleted.

Status ∈ { PROPOSED, USER-APPROVED, SUPERSEDED }.

## Project decisions

| ID | date | decision | alternatives considered | evidence / why | status |
|---|---|---|---|---|---|
| L-001 | 2026-07-02 | Engine: own unified codebase; solo-learn forked as *reviewed donor* (port after review, never a library dependency); lightly + official repos same donor role | adopt solo-learn wholesale; lightly-based trainer; from-scratch | user: "we will build our unified codebase as i do not want to fully trust their code. you can use their code + integrate it into our codebase but after careful review" | USER-APPROVED |
| L-002 | 2026-07-02 | First controlled grid roster = core-7, one per family: SimCLR, BYOL, VICReg, DINO, MAE, I-JEPA, LeJEPA. Later: Barlow Twins, SwAV, MoCo v3, iBOT, DINOv2-toy | full 12-method zoo day 1; checkpoint-first | pipeline must be proven before multiplying recipe-validation risk | USER-APPROVED |
| L-003 | 2026-07-02 | Pacing: understanding-first, no deadline | ICLR/CVPR/ICML 2027 targets | user choice | USER-APPROVED |
| L-004 | 2026-07-02 | Compute: ≤2 H100 concurrent; `gpu` partition unrestricted. **Data ladder (binding order): toy (Imagenette) + IN-100 → consistent conclusions → public IN-1k ckpts as validation → IN-1k retrains last** | fixed GPU-hour budget | user: draw consistent conclusions at small scale first, then validate against what papers report, only then train IN-1k | USER-APPROVED |
| L-005 | 2026-07-02 | Conclusions protocol: pre-registered predictions; results land as numbers; interpretation only after joint discussion; this ledger is the record | — | user: "unless we agree on the takeaway, do not assume any conclusion" | USER-APPROVED |
| D-001 | 2026-07-02 | `ssl_project/` is a git repo; package `sslgap`; wandb project `sslgap`. **Amendment (Berker): the repo is a clean project — assistant/session files (CLAUDE.md, HANDOVER.md) untracked; governance docs under `docs/`** | governance in root | user directive 2026-07-02 | USER-APPROVED |
| D-002 | 2026-07-02 | Canonical IN-100 = `~/data/imagenet100` (CMC split); `~/data/imagenet-100` deprecated for this project | reconcile the two dirs | verified: the two share only **8/100 classes** — different datasets. CMC split matches the prior DINO control (`inv_dino-in100`) and solo-learn's published numbers | USER-APPROVED |
| ~~D-003~~ | 2026-07-02 | ~~h ≡ trunk forward_features output; anything trainable after the trunk counts as head~~ | — | superseded by D-003v2 (user's definition) | SUPERSEDED |
| **D-003v2** | 2026-07-02 | **`h` = the representation the method's paper uses for its linear-probe tables ("the representation"); `z` = the space the loss is applied to ("the proj"). If the loss is applied after ONLY a linear projection of h, the method counts as loss-on-representation ("we respect the method") — no core-7 case (nearest: MoCo v1, later roster).** Intermediate taps (trunk layers, lejepa `z.embed`, head layers) remain measured spaces for guillotine curves — they just aren't "h". Per-method h column in PROTOCOL §3; consistency flags F1–F4 there | trunk-only h (old D-003) | Berker's definition 2026-07-02; **F1–F4 resolved 2026-07-02: trunk-GAP for ResNet-native; NO concat/best-of readouts (last-layer feature type only; paper-exact variants = E11 arms); probe protocol fixed as ours; linear maps add no capacity → CLS→Linear counts as h (LeJEPA h = z.embed)** | USER-APPROVED |
| D-004 | 2026-07-02 | Budget = epochs-matched per rung; pixels-seen/epoch recorded per method as a covariate in MODELS.md | FLOP-matched budgets | FLOP-matching would force recipe changes (multi-crop DINO sees ~1.7× pixels/ep); recipe fidelity wins, the covariate keeps us honest | USER-APPROVED |
| D-005 | 2026-07-02 | Feature-store policy: fp16; nothing >8192-d stored (DINO prototype logits recomputed from bottleneck × weight-normed W); patch tokens final-layer/final-epoch/probed-branch only; cadence epochs battery-lite (vectors only); purge-after-table + tar archive; hard cap enforced in code (default 500 GB) | store everything | group NFS has 2.8 TB free of 15 TB, shared; M2 peak ≈ 350 GB under this policy | USER-APPROVED |
| ~~D-006~~ | 2026-07-02 | ~~primary linear = ℓ2-norm→Linear; house parity-only; attentive token-spaces~~ | — | superseded by D-006v2 proposal | SUPERSEDED |
| **D-006v2** | 2026-07-02 | **Headline probe pair: (1) `linear_raw_v1` — plain Linear on raw (unnormalized) frozen features, AdamW 1e-3 / wd 1e-7 / 30 ep / best-val — linear separability in its purest form; (2) `knn_v1` — lightly-parity weighted cosine kNN, k=200, t=0.1.** Computed but not headline: `linear_house_v1` (LN+Linear; continuity with prior in-house tables), `linear_l2_v1` + `attentive_v1` (token spaces, M2) → E11, where probe sensitivity is itself the object; `sololearn_linear` port-validation only | l2-primary (old D-006); single probe | Berker: "main thing is linear separability + lightly-parity weighted kNN"; approved 2026-07-02 | USER-APPROVED |
| D-007 | 2026-07-02 | M2 grid adds anchors: supervised DeiT-lite + random-init | core-7 only | report §5.0 requires anchors; without them the audit matrix has no reference rows | USER-APPROVED |
| D-008 | 2026-07-02 | Per-method `z` definitions per PROTOCOL §3 table (DINO z.final = 256-d bottleneck pre-prototypes; MAE z = decoder-block taps; BYOL target space = teacher projection; I-JEPA probed branch = teacher) | naive "final head output" | uniform "final output" would mean 65k-d prototype logits / pixel tensors; per-method definitions keep z meaningful and storable | USER-APPROVED (folded into D-003v2; z-side clean, h-side flags F1–F4) |
| **D-009** | 2026-07-02 | **Relative representations** in the cross-model toolset: cosine similarities to anchors — (a) dataset anchors, A = feature dim, fixed seeded image ids per manifest, SHARED across models (true cross-model frame on shared data); (b) random-orthonormal anchor control (a rotation of the normalized space — a metric rotation-invariance check, does NOT align different models). Battery runnable on relrep space; cross-model driver at M1 when comparable models exist | none | user request 2026-07-02 (Moschella et al. 2023-style); semantics verified vs latentis @800699f: cosine = l2+dot, no centering by default, optional Centering/StandardScaling exposed as `abs_transform` | USER-DIRECTED (impl verified vs latentis) |
| **D-010** | 2026-07-02 | `docs/METRICS.md` = metric reference (definition, bounds, direction, caveats — incl. **EP rejection-test semantics: a low statistic / failing to reject ≠ isotropic Gaussian**, hence the kurt-top pairing); metric↔downstream relations: literature column now, E3 measures ours, M0 exploratory correlations non-evidential | knowledge stays in code comments | user request 2026-07-02 | USER-DIRECTED (delivered) |

## Gate decisions (data-ladder / milestone advancement)

| ID | date | gate | evidence required | status |
|---|---|---|---|---|
| G-M0 | 2026-07-02 | M0 exit → start M1 (toy trainers, LeJEPA port first) | **PASSED**: pipeline end-to-end on 6 ckpts + 2 nulls; parity vs prior stack exact (16/16 within ±0.06 pt; RankMe/PR to the printed decimal); kNN self-test green; collapse detectors verified on the λ=0 arm; D-001…D-010 ratified (D-003v2 F1–F4 resolved; D-006v2 approved); PROTOCOL v1-draft.2 in force; AUDIT_MATRIX v1 LOCKED | USER-APPROVED (Berker, 2026-07-02) |
| G-M2 | — | rung 2→3 (public IN-1k ckpts) | "consistent conclusions at toy+IN-100" — E1/E2/E11 resolved with agreed takeaways | pending |
| G-M4 | — | rung 3→4 (IN-1k retrains) | scoped only after M3/M4 review | pending |

## Agreed experimental takeaways

*(empty — filled only from experiment-card discussions; each row links its E-card and evidence)*

| ID | date | takeaway | evidence (card, tables) | status |
|---|---|---|---|---|
