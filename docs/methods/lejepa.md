# LeJEPA — dossier

> Derived from docs/report/ssl-projector-gap-report.html (compiled 2026-07-01, quote-verified there). Tags: [E] established · [P] plausible interpretation · [O] open question.

## Identity

- **Paper:** *LeJEPA: Provable and Scalable SSL Without the Heuristics* — Balestriero & LeCun.
  Preprint, Nov 2025 · arXiv:2511.08544. **[P — preprint, unreplicated; treat scale claims
  accordingly.]**
- Naming: "L-JEPA" resolves to LeJEPA; the only name-collision is AD-L-JEPA (arXiv:2501.04969, a
  LiDAR driving paper). [E]
- **Family:** JEPA, heuristics-free — a **single trade-off hyperparameter λ**; no stop-gradient, no
  EMA teacher–student, no schedulers. [E]
- Official code: `galilai-group/lejepa` / `rbalestr-lab/lejepa` (our donor: `../lejepa`).

## Stated desiderata (paper's own language)

Embeddings should follow an **isotropic Gaussian** — proven "optimal... to minimize downstream
prediction risk" (worst-case over tasks; Lemmas 1–2 for linear probes under ridge, Theorem 1 for
nonlinear probes: "unique minimizer of the integrated square bias" under a scalar covariance
constraint). [E quotes; P venue]

## Where the loss lives

- **Objective:** multi-view prediction + SIGReg (isotropic-Gaussian test).
- **Loss applied to:** z = the output of a **3-layer MLP projector**. Official implementation:
  `self.proj = MLP(512, [2048, 2048, proj_dim], BatchNorm1d)`; **both losses are computed on the
  projector output** — `sigreg_loss = sigreg(proj)` and
  `inv_loss = (proj.mean(0)-proj).square().mean()`. [E]
  - Multi-view prediction: each view regresses the mean of the global-view embeddings.
  - SIGReg: random 1-D projections tested against N(0,1) with the Epps–Pulley statistic.
- **Head between h and loss:** 3-layer MLP projector + BN (≈ VICReg expander). The paper's
  Table 1(d) lists *emb. dim.* [512, 2048] and *proj. dim.* [64…1024] as separate hyperparameters. [E]
- **Asymmetry machinery:** none — "no stop-gradient, no teacher-student, no hyper-parameter
  schedulers." What LeJEPA removes is the heuristic stack, **not** the projector. [E]
- **Same space as eval?** No.
- **Correction (2026-07-02), recorded verbatim from the report:** an earlier draft of the report
  claimed LeJEPA applied its loss directly to the backbone — that was wrong. A reader checked the
  official repo and found the 3-layer MLP projector with both losses on its output; the paper's
  ambiguous f_θ/z notation and its unexplained "proj. dim." hyperparameter had been misread.
  Corrected throughout: **0 of 11** surveyed methods impose their loss on the probed
  representation, and LeJEPA is the sharpest illustration of the gap rather than its exception.
  (Report §7.5 flags this as a cautionary example of the very failure it studies — reading a
  paper's stated desideratum instead of checking where the code applies it.)

## Eval space & paper protocol

h = frozen backbone (concat CLS of last 2 layers, per Table 1); frozen linear/ridge probe.
Training loss "exhibits strong correlation with downstream linear probe performance" — label-free
model selection. ~79% IN-1k linear with ViT-H/14; stability across 60+ architectures. [E — preprint]

## Head ablations & known head facts

- The projector is a full-width 3-layer MLP with BN — architecturally the VICReg expander, so all
  §2 head evidence (buffer, uniformity-projector, rank shield) plausibly applies unchanged. [P]
- No published head ablation exists yet (preprint); the projector-depth 0–3 sweep and
  SIGReg-on-backbone arm are **ours to run** (E10). [O]
- Reading [P]: LeJEPA is the sharpest case of the report's thesis, not its exception. It proves the
  isotropic Gaussian is the distribution embeddings "should follow to minimize downstream risk,"
  then enforces that distribution on a projector it discards — so the optimality theorem, read
  literally, characterizes the thrown-away space, while the probed backbone inherits isotropy only
  indirectly (the paper's notation does not distinguish the two, and does not claim the projector
  is discarded).
- The genuinely open question [O]: does a *distributional* constraint (isotropy of the whole
  embedding cloud) transfer to h more faithfully than the *pointwise* invariance other methods
  impose? Directly testable (E1, E4, E10).

## Documented weaknesses / failure modes

Table-4 row (° = prediction [P]):

| Expected strengths | Documented weaknesses | Predicted weak spots (§5 probes) |
|---|---|---|
| heuristics-free (no stop-grad/EMA/teacher/schedulers); provable isotropy target; single hyperparameter; stability across architectures; label-free selection | preprint, unreplicated; still projector-based — isotropy enforced at z, not the probed h; mild view stack persists | isotropy transfer z→h° (E1); whether a distributional constraint needs less buffering than invariance° (E10) |

## Audit expectations (this project)

Report §3.1 predicted-matrix row (h / z; predictions [P]):

| Alignment | Uniformity | Var. floor | Decorrelation | Eff. rank high | Isotropy/Gauss. | Aug.-invariance | View-predictability |
|---|---|---|---|---|---|---|---|
| ?/✓ | ?/✓ | ?/✓ | ?/~ | ?/✓ | **?/✓ (SIGReg at z)** | ? (mild view stack) | **?/✓** |

**Own-desideratum cells (bold, explicit in the report):** isotropy/Gaussianity and
view-predictability — both enforced at z; "their transfer to h is the study's central unknown."
Note the h column is all `?`: LeJEPA is the method the literature predicts *least* about at h.
E1's LeJEPA-specific test: is its z→h isotropy dissociation smaller than the invariance
dissociation of view-based methods?

## Recipe & port plan (OURS — scaffold)

- **Donor:** `../lejepa` — official minimal implementation, ported **exactly** (it is M1's ground
  truth: port must reproduce the official loss curve within amp noise, probe within ~1 pt).
- **Known recipe risks (kickstart plan):** lowest risk of the core-7 (no schedulers/EMA/stop-grad
  to mis-set; single λ). House caveat: sliced-Gaussianity (Epps–Pulley) is foolable — isotropy
  monitoring pairs it with kurt_topeig/worst-direction stats (CLAUDE.md).

### PORT_NOTES

*(empty — filled at port review time: donor commit, review findings, deviations)*
