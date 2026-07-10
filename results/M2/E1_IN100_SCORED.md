# E1 IN-100 SCORED (D-024 mechanical pass, seed 0) — EVERY CELL VETO-OPEN

> Emitted by experiments/score_matrix.py. Glyphs are MECHANICAL-RULE outputs (delegated one-time, Berker 2026-07-10), laid beside their inputs; they are NOT agreed interpretation. Predictions = AUDIT_MATRIX v1 (LOCKED 2026-07-02); bold = own desideratum. Verdicts: MATCH · soft (a ~ on either side) · SURPRISE (✓↔✗) · NEW:<glyph> (prediction was ?, the study's new measurement — no match possible) · n/a (space absent / not measured). Nulls per D-014/T5: named per family below; own-arch nulls where landed, shared randinit otherwise (teacher→student read-across). Single seed per D-024.

## Rules (the delegated choices)

- **Alignment**: pairs (D-013/E01-T8): cos_margin + align_rel vs the shared-null same-space baseline. ✓ margin ≥ .5 AND ≥ 3x null margin · ~ align_rel ≤ .7x null align_rel OR margin ≥ 1.5x null margin (T8's 'relative-only') · ✗ else. z has no null pair store at this rung: absolute ✓ ≥ .5 / ~ ≥ .25.
- **Uniformity**: battery `uniformity` (raw|full; 0 = collapsed, more negative = more uniform): ✓ ≤ −3 · ~ ≤ −1.5 · ✗ else. Randinit null shown for reference.
- **Variance floor (scale-free)**: `min_over_mean_std` ∈[0,1] (all-dims-alive): ✓ ≥ .5 · ~ ≥ .1 · ✗ else (absolute; D-014 scale-free lead).
- **Decorrelation**: `mean_abs_corr` (absolute): ✓ ≤ .05 · ~ ≤ .15 · ✗ else.
- **Eff. rank**: `rankme` / d (fraction of ambient dim): ✓ ≥ .5 · ~ ≥ .2 · ✗ else. SHAKIEST RULE — random features are high-rank, so the null is no reference and the fraction threshold is a choice. Veto expected here first.
- **Isotropy/Gauss.**: `kurt_topeig.worst` vs the MATCHED-GAUSSIAN null (T5 which-null): q = |v|/max(|null_gauss|,.02): ✓ q ≤ 3 · ~ q ≤ 10 · ✗ else. EP row stays unscored beside kurt (T5: never EP alone).
- **Aug.-invariance**: same margin rule as Alignment (T8 scored the two jointly).
- **View-predictability**: no cell at this rung (probe-based) — not measured.

## Alignment — predicted vs measured

| method   | h inputs                                                                                              | glyph_h   | z inputs            | glyph_z   | predicted h/z   | verdict h/z      |
|:---------|:------------------------------------------------------------------------------------------------------|:----------|:--------------------|:----------|:----------------|:-----------------|
| simclr   | mrg 0.33 (null 0.15), arel 0.30 (null 0.81)                                                           | ~         | mrg 0.79, arel 0.21 | ✓         | **?/✓**         | NEW:~ / MATCH    |
| byol     | mrg 0.16 (null 0.15), arel 0.36 (null 0.81)                                                           | ~         | mrg 0.68, arel 0.17 | ✓         | **?/✓**         | NEW:~ / MATCH    |
| vicreg   | mrg 0.11 (null 0.15), arel 0.37 (null 0.81)                                                           | ~         | mrg 0.69, arel 0.30 | ✓         | ?/✓             | NEW:~ / MATCH    |
| dino     | mrg 0.54 (null 0.14), arel 0.37 (null 0.82)                                                           | ✓         | mrg 0.68, arel 0.24 | ✓         | ?/✓             | NEW:✓ / MATCH    |
| mae      | mrg 0.13 (null 0.15), arel 0.68 (null 0.81)                                                           | ✗         | —                   | —         | ✗/—             | MATCH / n/a      |
| ijepa    | mrg 0.18 (null 0.15), arel 0.75 (null 0.81)                                                           | ✗         | mrg 0.17, arel 0.77 | ✗         | ?/✓             | NEW:✗ / SURPRISE |
| lejepa   | mrg 0.37 (null — no null store (lejepa-arch gap)), arel 0.11 (null — no null store (lejepa-arch gap)) | ~         | mrg 0.91, arel 0.09 | ✓         | ?/✓             | NEW:~ / MATCH    |


## Uniformity — predicted vs measured

| method   | h inputs            | glyph_h   | z inputs            | glyph_z   | predicted h/z   | verdict h/z     |
|:---------|:--------------------|:----------|:--------------------|:----------|:----------------|:----------------|
| simclr   | -1.66 (null -1.68)  | ~         | -3.82 (null -1.52)  | ✓         | ✗/✓             | soft / MATCH    |
| byol     | -0.878 (null -1.68) | ✗         | -2.91 (null -1.19)  | ~         | ✗/~             | MATCH / MATCH   |
| vicreg   | -0.642 (null -1.68) | ✗         | -3.92 (null -1.28)  | ✓         | ✗/~             | MATCH / soft    |
| dino     | -3.24 (null -1.88)  | ✓         | -3.53 (null -1.8)   | ✓         | **✗/~**         | SURPRISE / soft |
| mae      | -1.36 (null -1.68)  | ✗         | —                   | —         | ✗/—             | MATCH / n/a     |
| ijepa    | -2.49 (null -1.68)  | ~         | -2.41 (null -0.343) | ~         | ✗/✗             | soft / soft     |
| lejepa   | -1.44 (null -1.88)  | ✗         | -3.32 (null -1.28)  | ✓         | ?/✓             | NEW:✗ / MATCH   |


## Variance floor (scale-free) — predicted vs measured

| method   | h inputs           | glyph_h   | z inputs           | glyph_z   | predicted h/z   | verdict h/z   |
|:---------|:-------------------|:----------|:-------------------|:----------|:----------------|:--------------|
| simclr   | 0.505 (null 0.255) | ✓         | 0.835 (null 0.438) | ✓         | ?/✓             | NEW:✓ / MATCH |
| byol     | 0.374 (null 0.255) | ~         | 0.529 (null 0.563) | ✓         | ?/~             | NEW:~ / soft  |
| vicreg   | 0.5 (null 0.255)   | ✓         | 0.986 (null 0.52)  | ✓         | **~/✓**         | soft / MATCH  |
| dino     | 0.155 (null 0.38)  | ~         | 0.832 (null 0.494) | ✓         | ?/?             | NEW:~ / NEW:✓ |
| mae      | 0.315 (null 0.255) | ~         | —                  | —         | ?/—             | NEW:~ / n/a   |
| ijepa    | 0.377 (null 0.255) | ~         | 0.381 (null 0.304) | ~         | ?/?             | NEW:~ / NEW:~ |
| lejepa   | 0.386 (null 0.375) | ~         | 0.945 (null 0.577) | ✓         | ?/✓             | NEW:~ / MATCH |


## Decorrelation — predicted vs measured

| method   | h inputs            | glyph_h   | z inputs            | glyph_z   | predicted h/z   | verdict h/z   |
|:---------|:--------------------|:----------|:--------------------|:----------|:----------------|:--------------|
| simclr   | 0.13 (null 0.475)   | ~         | 0.0869 (null 0.316) | ~         | ✗/~             | soft / MATCH  |
| byol     | 0.121 (null 0.475)  | ~         | 0.15 (null 0.296)   | ~         | ✗/~             | soft / MATCH  |
| vicreg   | 0.129 (null 0.475)  | ~         | 0.0365 (null 0.291) | ✓         | **✗/✓**         | soft / MATCH  |
| dino     | 0.0849 (null 0.377) | ~         | 0.0873 (null 0.333) | ~         | ✗/?             | soft / NEW:~  |
| mae      | 0.153 (null 0.475)  | ✗         | —                   | —         | ✗/—             | MATCH / n/a   |
| ijepa    | 0.1 (null 0.475)    | ~         | 0.103 (null 0.436)  | ~         | ✗/?             | soft / NEW:~  |
| lejepa   | 0.15 (null 0.371)   | ✗         | 0.0239 (null 0.264) | ✓         | ?/~             | NEW:✗ / soft  |


## Eff. rank — predicted vs measured

| method   | h inputs        | glyph_h   | z inputs        | glyph_z   | predicted h/z   | verdict h/z   |
|:---------|:----------------|:----------|:----------------|:----------|:----------------|:--------------|
| simclr   | 136 (null 53)   | ~         | 101 (null 152)  | ✗         | ~/✗             | MATCH / MATCH |
| byol     | 119 (null 53)   | ~         | 39 (null 91.6)  | ✗         | ~/~             | MATCH / soft  |
| vicreg   | 116 (null 53)   | ~         | 562 (null 418)  | ~         | ~/✓             | MATCH / soft  |
| dino     | 204 (null 91.5) | ✓         | 118 (null 82.1) | ~         | ~/?             | soft / NEW:~  |
| mae      | 103 (null 53)   | ~         | —               | —         | **✗**/—         | soft / n/a    |
| ijepa    | 159 (null 53)   | ~         | 135 (null 20.8) | ~         | ?/?             | NEW:~ / NEW:~ |
| lejepa   | 35 (null 79.5)  | ✗         | 16 (null 11)    | ✓         | ?/✓             | NEW:✗ / MATCH |


## Isotropy/Gauss. — predicted vs measured

| method   | h inputs          | glyph_h   | z inputs         | glyph_z   | predicted h/z   | verdict h/z      |
|:---------|:------------------|:----------|:-----------------|:----------|:----------------|:-----------------|
| simclr   | 0.843 (null 5.93) | ✗         | 84.7 (null 1.53) | ✗         | ✗/~             | MATCH / soft     |
| byol     | 1.4 (null 5.93)   | ✗         | 7.34 (null 1.54) | ✗         | ✗/✗             | MATCH / MATCH    |
| vicreg   | 2.98 (null 5.93)  | ✗         | 245 (null 1.54)  | ✗         | ✗/~             | MATCH / soft     |
| dino     | 0.893 (null 1.51) | ✗         | 66.2 (null 1.55) | ✗         | ✗/✗             | MATCH / MATCH    |
| mae      | 2.43 (null 5.93)  | ✗         | —                | —         | ✗/—             | MATCH / n/a      |
| ijepa    | 1.74 (null 5.93)  | ✗         | 1.69 (null 4.08) | ✗         | ✗/✗             | MATCH / MATCH    |
| lejepa   | 1.5 (null 2.11)   | ✗         | 2.06 (null 1.31) | ✗         | **?/✓**         | NEW:✗ / SURPRISE |


## Aug.-invariance — predicted vs measured

| method   | h inputs                                                                                              | glyph_h   | z inputs            | glyph_z   | predicted h/z   | verdict h/z      |
|:---------|:------------------------------------------------------------------------------------------------------|:----------|:--------------------|:----------|:----------------|:-----------------|
| simclr   | mrg 0.33 (null 0.15), arel 0.30 (null 0.81)                                                           | ~         | mrg 0.79, arel 0.21 | ✓         | **✗/✓**         | soft / MATCH     |
| byol     | mrg 0.16 (null 0.15), arel 0.36 (null 0.81)                                                           | ~         | mrg 0.68, arel 0.17 | ✓         | ✗/✓             | soft / MATCH     |
| vicreg   | mrg 0.11 (null 0.15), arel 0.37 (null 0.81)                                                           | ~         | mrg 0.69, arel 0.30 | ✓         | ✗/✓             | soft / MATCH     |
| dino     | mrg 0.54 (null 0.14), arel 0.37 (null 0.82)                                                           | ✓         | mrg 0.68, arel 0.24 | ✓         | ✗/✓             | SURPRISE / MATCH |
| mae      | mrg 0.13 (null 0.15), arel 0.68 (null 0.81)                                                           | ✗         | —                   | —         | ✗/—             | MATCH / n/a      |
| ijepa    | mrg 0.18 (null 0.15), arel 0.75 (null 0.81)                                                           | ✗         | mrg 0.17, arel 0.77 | ✗         | ✗ (no augs)     | MATCH / n/a      |
| lejepa   | mrg 0.37 (null — no null store (lejepa-arch gap)), arel 0.11 (null — no null store (lejepa-arch gap)) | ~         | mrg 0.91, arel 0.09 | ✓         | ? (mild stack)  | NEW:~ / n/a      |


## View-predictability — predicted vs measured

| method   | predicted h/z   | measured h/z              | verdict   |
|:---------|:----------------|:--------------------------|:----------|
| simclr   | ?/—             | not measured at this rung | n/a       |
| byol     | ?/✓             | not measured at this rung | n/a       |
| vicreg   | ?/—             | not measured at this rung | n/a       |
| dino     | ?/—             | not measured at this rung | n/a       |
| mae      | **?/—**         | not measured at this rung | n/a       |
| ijepa    | **?/✓**         | not measured at this rung | n/a       |
| lejepa   | **?/✓**         | not measured at this rung | n/a       |


## Summary (mechanical; no interpretation)

- h-side: MATCH 16, NEW:~ 11, NEW:✓ 2, NEW:✗ 5, SURPRISE 2, n/a 7, soft 13
- z-side: MATCH 23, NEW:~ 5, NEW:✓ 1, SURPRISE 2, n/a 16, soft 9
- SURPRISE cells (flagged for discussion before any narrative, AUDIT_MATRIX rule): ijepa/Alignment; dino/Uniformity; lejepa/Isotropy/Gauss.; dino/Aug.-invariance
