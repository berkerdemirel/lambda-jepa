# ROADMAP — hierarchical: theory ▸ methods ▸ experiments

Derived from `docs/report/ssl-projector-gap-report.html` §7.6 (compiled 2026-07-01), adapted to the
binding data ladder (DECISIONS L-004) and the 2×H100 cap. Milestones exit only through user review
(gate rows in DECISIONS.md).

## 0 · The question hierarchy

```
Q0  Do SSL methods impose their stated desiderata on h, or only on z?          → E1 (audit matrix, τ)
 ├─ Q1  How do desiderata decay through the head?                              → E2 (guillotine curves)
 ├─ Q2  Which space's properties predict downstream utility?                   → E3 (+E11 probe map)
 ├─ Q3  What information does the head delete / the backbone keep?             → E4 (nuisance ledger)
 ├─ Q4  Which semantic failure modes hide behind satisfied losses?             → E5–E9 (failure probes)
 ├─ Q5  Is the dissociation CAUSED by head placement, and does moving the
 │      loss to h trade retention against invariance along a frontier?         → E10 (causal grid)
 └─ Q6  Constructive: what criterion, imposed where, closes the gap?           → M5+ synthesis
        (connects to the ssl_explore/lejepa agenda: minimal complete description up to nuisance)
```

Theory backbone (docs/theory/THEORY_MAP.md): geometry claims break at the gap; content/sufficiency
claims survive; the field's two 2025–26 anchors (LeJEPA normative isotropy; Latent Distribution
Matching unification) are both stated at the loss layer — our measurements are the missing bridge.

## 1 · Milestones

### M0 — pipeline on existing checkpoints (zero training) · ~1 wk · `gpu` partition
Docs + scaffold + audit pipeline end-to-end on: 3 lejepa Imagenette ckpts (λ=0.02 / λ=0 / InfoNCE;
h=384 trunk, z.embed=512, z.proj=16 — stress-tests dim-sensitive estimators) and 3 DINO IN-100
control ckpts (ep25/50/100; student+teacher), + random-init nulls per frame.
**Exit criteria:** battery runs from stored features alone (model not in memory); kNN parity
self-test passes; DINO GAP linear/kNN reproduce ssl_explore CAMPAIGN_LOG within ±0.5 pt under
`linear_house_v1`; lejepa emb-vs-proj gap qualitatively consistent with the user's own
`probe_emb_vs_proj.py` findings; mini audit matrix + τ emitted for 2 methods; PROTOCOL v1 + E1
predictions **pre-registered with user sign-off**; D-001…D-008 ratified. → gate G-M0.

### M1 — uniform trainers, toy rung · ~2–3 wk · `gpu`, 7 × ~2–4 h runs
Imagenette ViT-S/8@128, 150 ep. Port order: **LeJEPA first** (official minimal in-house = exact
ground truth), then SimCLR, VICReg, BYOL (solo-learn donors), DINO (restructure sslx trainer), MAE,
I-JEPA. 3-epoch smoke before each full run (WORKFLOW.md rule).
**Exit:** 7 uniform `sslgap/ckpt/v1` checkpoints with heads+taps; collapse monitors green; LeJEPA
port reproduces the official curve (loss within amp noise; probe within ~1 pt); toy E1 matrix →
dress-rehearsal discussion with user.

### M1.5 (optional, parallel) — port validation on RN18-IN-100
solo-learn recipes verbatim for SimCLR/BYOL/VICReg vs their published IN-100 numbers (the only
external ground truth at this scale). Target: within ~1–1.5 pt under `sololearn_linear`. Results →
MODELS.md validation column; never enters the audit matrix.

### M2 — controlled IN-100 grid + audit core · ~4–6 wk · H100 2-slot queue
Core-7 ViT-S/16@224 × 100 ep, seed 0 then seed 1, + supervised DeiT-lite + random-init anchors
(D-007). Full extraction at ep100 (+patch tokens), battery-lite at cadence epochs.
**Deliverables:** E1 completed matrix with CIs + τ per method; E2 guillotine curves (backbone
layers + head taps); E11 probe-sensitivity map (nearly free — features cached).
**Exit:** seed-1 replication of headline cells; per-card resolution discussions → AGREED TAKEAWAYS
in DECISIONS.md. → gate G-M2 (advance to public rung).

### M3 — public IN-1k validation rung · ~3–4 wk · `gpu` inference only
Head-inventory verification first (MODELS.md Track C), adapters per source; published linear/kNN
reproduced within ~1 pt before any battery number is trusted. Then E3 (metric→downstream
regression) + E4 (augmentation-information ledger) across both tracks.
**Exit:** E3/E4 resolved; IN-100 vs IN-1k **pattern** comparison (patterns, not raw numbers).

### M4 — E10 causal grid · ~6–10 wk · ≈15–18 H100-d per seed pass
Move-the-loss retrains on IN-100: LeJEPA {proj depth 0–3, SIGReg-on-h}, SimCLR {proj depth 0–3},
VICReg {var/cov at z | h | both}, DINO {+KoLeo-at-h}, BYOL→DirectPred (stretch). ≈15–18 configs × 2
seeds. **Exit:** the partial-transfer frontier; pre-registered frontier predictions resolved.

### M5+ — scoped later with user
Failure probes E5–E9 (features mostly cached; datasets: IN-9, Waterbirds, Stylized-IN, COCO-crops
bank to build); roster expansion (Barlow, SwAV, MoCo v3, iBOT, DINOv2-toy); IN-1k retrains (ladder
rung 4, gate G-M4); **constructive phase** (Q6): synthesize the criterion/method from the audit;
paper writing.

## 2 · Experiment index

| card | title | question | phase | status |
|---|---|---|---|---|
| [E01](experiments/E01_two_space_audit.md) | Two-space desiderata audit | Q0 | M0 (mini) → M2 (full) | draft |
| [E02](experiments/E02_guillotine_desiderata.md) | Guillotine curves for desiderata | Q1 | M2 | draft |
| [E03](experiments/E03_metric_downstream.md) | Which space's metrics predict downstream | Q2 | M3 | draft |
| [E04](experiments/E04_aug_ledger.md) | Augmentation-information ledger | Q3 | M3 | stub |
| [E05](experiments/E05_object_absent_crops.md) | Object-absent & background-only crops | Q4 | M5+ | stub |
| [E06](experiments/E06_context_group_robustness.md) | Fg/bg conflict, context swap, group robustness | Q4 | M5+ | stub |
| [E07](experiments/E07_texture_shape.md) | Texture/shape cue-conflict for MIM & JEPA | Q4 | M5+ | stub |
| [E08](experiments/E08_small_multi_object.md) | Small objects & multi-object scenes | Q4 | M5+ | stub |
| [E09](experiments/E09_masked_ambiguity.md) | Masked-region ambiguity (content) | Q4 | M5+ | stub |
| [E10](experiments/E10_causal_head_grid.md) | Causal head interventions (move the loss) | Q5 | M4 | draft |
| [E11](experiments/E11_probe_sensitivity.md) | Probe-protocol sensitivity map | Q2 | M2 | draft |
| [E12](experiments/E12_moment_floor.md) | OUR METHOD arm 1 — the moment floor at h | Q6 | M4 (one-branch vehicle, D-026) | pre-registered |
| [E13](experiments/E13_pivot_rung0.md) | PIVOT rung-0 — predictive-state distortion as a zoo ranker | Q2 (staged-merge gate for the PIVOT proposal, D-029) | M2 features, zero training | resolved (T1–T3) |

Pre-registration artifact: [experiments/AUDIT_MATRIX.md](experiments/AUDIT_MATRIX.md) (report §3.1
predictions — locked before numbers exist).

## 3 · Method index

Core-7 (L-002): [simclr](methods/simclr.md) · [byol](methods/byol.md) · [vicreg](methods/vicreg.md)
· [dino](methods/dino.md) · [mae](methods/mae.md) · [ijepa](methods/ijepa.md) ·
[lejepa](methods/lejepa.md). Later: methods/later/{barlow,swav,mocov3,ibot,dinov2}.md.

## 4 · Standing rules

Pre-register → run → numbers land on the card → **discussion** → AGREED TAKEAWAY → DECISIONS row.
No rung advance without a gate row. Provenance never mixed within a table. Estimator discipline of
PROTOCOL §6 applies to every number that leaves the repo.
