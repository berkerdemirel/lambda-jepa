# AUDIT_MATRIX — pre-registered directional predictions

> Derived verbatim from docs/report/ssl-projector-gap-report.html §3.1 (compiled 2026-07-01).
> This is the **pre-registration artifact** for E01/E02: falsifiable content committed before any
> full-grid number is computed. **LOCKED v1 — user sign-off 2026-07-02 (gate G-M0).** Edits only via
> a DECISIONS row.

**Cell notation:** `expectation at h / expectation at z`. ✓ = desideratum satisfied · ✗ = violated ·
~ = partially/weakly · ? = genuinely unknown · — = space doesn't exist in that form.
**Bold** marks each method's own advertised desideratum — the diagonal the study tests.

| Method | Alignment | Uniformity | Variance floor | Decorrelation | Eff. rank high | Isotropy/Gauss. | Aug.-invariance | View-predictability |
|---|---|---|---|---|---|---|---|---|
| SimCLR | **?/✓** | ✗/✓ | ?/✓ | ✗/~ | ~/✗ (dim. collapse at z) | ✗/~ | **✗/✓** (RCDM) | ?/— |
| BYOL | **?/✓** | ✗/~ | ?/~ | ✗/~ | ~/~ (rank differential) | ✗/✗ | ✗/✓ | ?/✓ |
| SwAV† | ?/✓ (code space) | **✗/✓** (equipartition) | ?/~ | ✗/? | ~/? | ✗/✗ | ✗/✓ | ?/— |
| DINO | ?/✓ | **✗/~** (centering) | ?/? | ✗/? | ~/? | ✗/✗ | ✗/✓ | ?/— |
| DINOv2† | ?/✓ | **~/✓** (KoLeo at ~h) | ?/? | ✗/? | ✓/? | ?/✗ | ✗/✓ | ?/— |
| Barlow Twins† | ?/✓ | ✗/~ | ?/✓ | **✗/✓** | ~/✓ | ✗/~ | ✗/✓ | ?/— |
| VICReg | ?/✓ | ✗/~ | **~/✓** (D.8: directional) | **✗/✓** | ~/✓ | ✗/~ | ✗/✓ | ?/— |
| MAE | ✗/— (pixel loss) | ✗/— | ?/— | ✗/— | **✗** (U-MAE: collapse-prone)/— | ✗/— | ✗ (none trained)/— | **?/—** |
| I-JEPA | ?/✓ (latent L2) | ✗/✗ | ?/? | ✗/? | ?/? (C-JEPA: fragile) | ✗/✗ | ✗ (by design: no augs) | **?/✓** |
| LeJEPA | ?/✓ | ?/✓ | ?/✓ | ?/~ | ?/✓ | **?/✓** (SIGReg at z) | ? (mild view stack) | **?/✓** |

† not in the core-7 first grid (DECISIONS L-002); rows kept for the later roster and the public rung.

Note (report §3.1): LeJEPA takes the same h/z form as every other method — its bold cells (isotropy,
view-predictability) are enforced at z; their transfer to h is the study's central unknown. [P]

## What counts as confirmation / surprise

- Cells where the two spaces should diverge (✗/✓) are the report's *evidence*; cells where they
  don't diverge in measurement are its *surprises* — every surprise cell gets flagged for
  discussion before any narrative is attached.
- The `?` cells at h are the study's new measurements — no published value exists.

## Locking record

| version | date | change | sign-off |
|---|---|---|---|
| v1-draft | 2026-07-02 | transcribed from report §3.1 | — |
| **v1 (LOCKED)** | 2026-07-02 | locked at M0 exit; predictions frozen before any controlled-grid number exists (M0 touched only pre-existing lejepa/DINO checkpoints) | **Berker, 2026-07-02** |
