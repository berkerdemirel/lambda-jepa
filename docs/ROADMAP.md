# ROADMAP — the question hierarchy, the milestones, and where the program actually is

Re-baselined 2026-08-07 (the previous version was written 2026-07-12 and its experiment index
stopped at E13). Derived from `docs/report/ssl-projector-gap-report.html` §7.6, adapted to the
binding data ladder (DECISIONS L-004). Terminology: [GLOSSARY.md](GLOSSARY.md). Milestones exit
only through user review (gate rows in DECISIONS.md).

## 0 · The question hierarchy

```
Q0  Do SSL methods impose their stated desiderata on h, or only on z?          → E1 (audit matrix, τ)
 ├─ Q1  How do desiderata decay through the head?                              → E2 (guillotine curves)
 ├─ Q2  Which space's properties predict downstream utility?                   → E3 (+E11 probe map)
 ├─ Q3  What information does the head delete / the backbone keep?             → E4 (nuisance ledger)
 ├─ Q4  Which semantic failure modes hide behind satisfied losses?             → E5–E9 (failure probes)
 ├─ Q5  Is the dissociation CAUSED by head placement, and does moving the
 │      loss to h trade retention against invariance along a frontier?         → E10 (causal grid)
 └─ Q6  Constructive: what criterion, imposed where, closes the gap?           → E12 → E27
```

**Q6 is where the program lives.** The audit answered Q0/Q1 and the diagnostic wave answered
enough of Q2/Q5 to build on; since E12 the work has been constructive — a conditioner at h, a
dose law for placing it, a geometry (the cloud calculus) for reading it, and now a scaled
multi-arch test of the recipe. E3–E9 are not abandoned but are not the active line.

## 1 · The three eras (what actually happened)

**Era 1 — the audit (E1, E2, E11; M0–M2).** Battery + probes + guillotine curves at h and z
under a matched frame, on seven uniform trainers. Established the two-space picture the founding
report predicted.

**Era 2 — the diagnostic wave (E12 → E18).** The moment conditioner at h (E12) → the
desideratum-transfer matrix and the cloud-connectivity finding (E17) → the declared-prior test
(E18). Produced the program's two load-bearing instruments: the **dose law** (a term's effect
tracks its *realized share* of trunk pull, not its nominal weight — E19-T1) and the
**cloud-thickness lens** (Ω, and the touch law that calibrates it).

**Era 3 — the constructive program (E19 → E27), current.** The method got its own class (E21),
scaled to IN-1k (E22), had its capacity map read (E23), its dose recipe derived (E24-T1), its
transfer measured (E25), and its geometry checked against public backbones (E26). E27 is the
guided multi-ViT IN-1k program the whole chain was building toward.

## 2 · Milestones

| id | scope | status |
|---|---|---|
| **M0** | pipeline on existing checkpoints, zero training | **PASSED** (G-M0; kNN parity exact, DINO reproduction within tolerance) |
| **M1** | uniform trainers, toy rung (Imagenette ViT-S/8@128, 150 ep, core-7) | **PASSED** (G-M1; LeJEPA portval certified the trainer stack) |
| **M1.5** | RN18-IN-100 solo-learn port validation | **DEFERRED indefinitely** (D-017; "after in100 runs, if we run them at all") |
| **M2** | controlled IN-100 grid + audit core (E1 matrix, E2 curves, E11 map) | **audit core DONE at seed 0**; seed-1 replication NOT run (D-024: single-seed glyph scoring, no seed replicates) |
| **M3** | public IN-1k validation rung | **OPENED for the zoo track** (D-076) → E26 landed, read pending |
| **M4** | E10 causal head grid (move-the-loss retrains) | **NOT RUN.** The constructive program overtook it: E12/E19/E20 move the loss to h across the whole zoo, which answers Q5's core in a different design. E10 stays a card, not a plan. |
| **M5+** | failure probes E5–E9, roster expansion, paper | scoped later |

**How the ladder actually advanced.** The blanket gates **G-M2 and G-M4 are still `pending`** —
IN-1k was never opened wholesale. Instead each IN-1k run got its own USER-APPROVED gate row:
D-055 (the first scaling cell), D-074 (one run, "run your best bet"), D-076 (the public-zoo
rung), D-079a (the E27 S pair). This is the governance of record; the blanket gates remain the
route for anything not individually approved.

## 3 · Experiment index

| card | title | question | status |
|---|---|---|---|
| [E01](experiments/E01_two_space_audit.md) | Two-space desiderata audit | Q0 | matrix landed (toy + IN-100, seed 0) |
| [E02](experiments/E02_guillotine_desiderata.md) | Guillotine curves for desiderata | Q1 | landed; the format is now the program's standard figure |
| [E03](experiments/E03_metric_downstream.md) | Which space's metrics predict downstream | Q2 | stub |
| [E04](experiments/E04_aug_ledger.md) | Augmentation-information ledger | Q3 | stub |
| [E05](experiments/E05_object_absent_crops.md) | Object-absent & background-only crops | Q4 | stub |
| [E06](experiments/E06_context_group_robustness.md) | Fg/bg conflict, context swap, group robustness | Q4 | stub |
| [E07](experiments/E07_texture_shape.md) | Texture/shape cue-conflict | Q4 | stub |
| [E08](experiments/E08_small_multi_object.md) | Small objects & multi-object scenes | Q4 | stub |
| [E09](experiments/E09_masked_ambiguity.md) | Masked-region ambiguity | Q4 | stub |
| [E10](experiments/E10_causal_head_grid.md) | Causal head interventions (move the loss) | Q5 | draft; superseded in practice (see M4) |
| [E11](experiments/E11_probe_sensitivity.md) | Probe-protocol sensitivity map | Q2 | locked for IN-100 |
| [E12](experiments/E12_moment_floor.md) | The moment conditioner at h | Q6 | **CLOSED** (T1–T9; T9(i) OPEN, reminder re-arms at E27 close) |
| [E13](experiments/E13_pivot_rung0.md) | PIVOT rung-0 — distortion as a zoo ranker | Q2 | resolved (T1–T3); lineage killed at E15/E16 |
| [E14](experiments/E14_foveal_rung1.md) | PIVOT rung-1 — the foveal channel | Q2 | lineage killed |
| [E15](experiments/E15_pivot_mvi.md) | PIVOT marginal MVI | Q6 | **KILLED** (D-032…D-034: the distillation ceiling) |
| [E16](experiments/E16_pivot_dense.md) | PIVOT dense rebuild | Q6 | **KILLED**; ceiling ruled (D-038) |
| [E17](experiments/E17_hpull.md) | h-pull: the desideratum-transfer matrix | Q6 | **COMPLETE** (T1–T7, D-039) — the cloud-connectivity finding |
| [E18](experiments/E18_declared_prior.md) | Declared-prior swap (t_ν) | Q6 | **CLOSED** (E18-T1: a failed rescue; mechanism OPEN) |
| [E19](experiments/E19_floorssl.md) | The method, arm 1: conditioner as sole anti-collapse at z | Q6 | **CLOSED** (E19-T1 the dose law, E19-T2) |
| [E20](experiments/E20_zoo_floor.md) | The calibrated conditioner at h, across the zoo | Q6 | **CLOSED** (T1–T4) |
| [E21](experiments/E21_floorssl_class.md) | The method as its own class + BN-free fork | Q6 | **COMPLETE** (T1 partial, T2 the estimator-width arc) |
| [E22](experiments/E22_floorssl_in1k.md) | IN-1k scaling cells (vm2/vm3/vm4) | Q6 | cells landed; read on the E23 card |
| [E23](experiments/E23_mlp_leakage.md) | Head capacity as the leakage dial | Q6 | **CLOSED** (T1–T5) — nothing owed |
| [E24](experiments/E24_dose_interaction.md) | The three-pull interaction law | Q6 | takeaways APPROVED (T1–T5); **card open on the voas landing** |
| [E25](experiments/E25_transfer.md) | ssl-transfer benchmark on the IN-1k winner | Q6 | **CLOSED** (E25-T1) |
| [E26](experiments/E26_public_zoo.md) | Public pretrained-zoo guillotine | Q2/Q6 | landed; **read PENDING** |
| [E27](experiments/E27_guided_sbl.md) | The guided multi-ViT IN-1k program (S/B/L) | Q6 | **S phase RUNNING**; B/L gated (D-079b/c) |

Pre-registration artifact: [experiments/AUDIT_MATRIX.md](experiments/AUDIT_MATRIX.md) (report
§3.1 predictions — locked before numbers existed).

## 4 · Theory track

`docs/theory/OMEGA_CONNECTIVITY.md` is the live document: the two cloud spaces, thickness and
the touch law as the measurement tool, the impossibility lemma, the whitened-alignment
eigenspace, transport, the declared-task transfer bound, and the operational band target. §3–§6
derivations are **REVIEW-PENDING** on Berker's own timeline. `THEORY_MAP.md` indexes which
theorems bind which space; the TRD-π framework documents are the earlier proposal lineage and
are historical.

## 5 · Method index

Core-7 (L-002): [simclr](methods/simclr.md) · [byol](methods/byol.md) · [vicreg](methods/vicreg.md)
· [dino](methods/dino.md) · [mae](methods/mae.md) · [ijepa](methods/ijepa.md) ·
[lejepa](methods/lejepa.md). The house method: [floorssl](methods/floorssl.md) (renaming to
`spectral` under D-083 — see [GLOSSARY.md](GLOSSARY.md)). Public zoo (E26, D-076): dinov2,
dinov3, dino, clip, siglip, mae.

## 6 · Standing rules

Pre-register → run → numbers land on the card → **discussion** → AGREED TAKEAWAY → DECISIONS row.
No rung advance without a gate row. Provenance never mixed within a table. Estimator discipline
of PROTOCOL §6 applies to every number that leaves the repo. Vocabulary per GLOSSARY.md.
