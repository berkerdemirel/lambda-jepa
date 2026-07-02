# DECISIONS.md — the ledger

Append-only. Every project-shaping decision and every agreed experimental takeaway gets a row.
**Rule (see CLAUDE.md): nothing here reaches status USER-APPROVED without Berker's explicit sign-off
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
| D-001 | 2026-07-02 | `ssl_project/` is a git repo; package name `sslgap`; wandb project `sslgap` | name `gap`, `audit` | — | PROPOSED |
| D-002 | 2026-07-02 | Canonical IN-100 = `~/data/imagenet100` (CMC split); `~/data/imagenet-100` deprecated for this project | reconcile the two dirs | verified: the two share only **8/100 classes** — different datasets. CMC split matches the prior DINO control (`inv_dino-in100`) and solo-learn's published numbers | PROPOSED |
| D-003 | 2026-07-02 | `h` ≡ trunk `forward_features` output (CLS & patch-GAP, 384-d for ViT-S, per-layer taps). Anything trainable after the trunk counts as head; the lejepa `Linear(384→512)` "emb" becomes tap `z.embed` | keep 512-d emb as h (prior repos' convention) | the audit's central variable must not contain a hidden head layer; verified in `lejepa/MINIMAL.md` + `sslx/models.py` (`num_classes=512`) | PROPOSED |
| D-004 | 2026-07-02 | Budget = epochs-matched per rung; pixels-seen/epoch recorded per method as a covariate in MODELS.md | FLOP-matched budgets | FLOP-matching would force recipe changes (multi-crop DINO sees ~1.7× pixels/ep); recipe fidelity wins, the covariate keeps us honest | PROPOSED |
| D-005 | 2026-07-02 | Feature-store policy: fp16; nothing >8192-d stored (DINO prototype logits recomputed from bottleneck × weight-normed W); patch tokens final-layer/final-epoch/probed-branch only; cadence epochs battery-lite (vectors only); purge-after-table + tar archive; hard cap enforced in code (default 500 GB) | store everything | group NFS has 2.8 TB free of 15 TB, shared; M2 peak ≈ 350 GB under this policy | PROPOSED |
| D-006 | 2026-07-02 | Probe protocols frozen as PROTOCOL §4 (`linear_l2_v1` primary; `linear_house_v1` parity-only; `knn_v1` k=200/t=0.1 + k=20; `attentive_v1` token-spaces only; `sololearn_linear` port-validation only) | single probe | report §5: family rankings swing 5–17 pts with probe choice — fix probes per space, report all | PROPOSED |
| D-007 | 2026-07-02 | M2 grid adds anchors: supervised DeiT-lite + random-init | core-7 only | report §5.0 requires anchors; without them the audit matrix has no reference rows | PROPOSED |
| D-008 | 2026-07-02 | Per-method `z` definitions per PROTOCOL §3 table (DINO z.final = 256-d bottleneck pre-prototypes; MAE z = decoder-block taps; BYOL target space = teacher projection; I-JEPA probed branch = teacher) | naive "final head output" | uniform "final output" would mean 65k-d prototype logits / pixel tensors; per-method definitions keep z meaningful and storable | PROPOSED |

## Gate decisions (data-ladder / milestone advancement)

| ID | date | gate | evidence required | status |
|---|---|---|---|---|
| G-M0 | — | M0 exit → start M1 (toy trainers) | M0 exit criteria met (see docs/ROADMAP.md) + PROTOCOL v1 and E1 predictions signed off + D-001…D-008 ratified | pending |
| G-M2 | — | rung 2→3 (public IN-1k ckpts) | "consistent conclusions at toy+IN-100" — E1/E2/E11 resolved with agreed takeaways | pending |
| G-M4 | — | rung 3→4 (IN-1k retrains) | scoped only after M3/M4 review | pending |

## Agreed experimental takeaways

*(empty — filled only from experiment-card discussions; each row links its E-card and evidence)*

| ID | date | takeaway | evidence (card, tables) | status |
|---|---|---|---|---|
