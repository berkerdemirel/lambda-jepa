# M2 — IN-100 E1 matrix, seed 0 (numbers only, SKELETON)

> Emitted by experiments/report_m2.py. NO cell is scored — glyph resolution happens in
> discussion (CLAUDE.md contract). Predicted glyphs = AUDIT_MATRIX v1 (LOCKED
> 2026-07-02); bold = the method's own desideratum. tau = value(h)/value(z).
> Probes = D-020 v2 family (patience-converged; `capped` marks best_ep within
> patience-reach of the 1000-ep cap). Null columns = shared in100.randinit-s0.ext
> (h-side only) EXCEPT methods with an own-arch null in cfg.method_null_runs
> (z-side included); remaining z-nulls NaN by construction, not by failure.
> Pair rows are margin-scored per D-013/E01-T8: results/M2/PAIR_MARGIN.md.
> Scored SINGLE-SEED per D-024 (Berker 2026-07-10; seed replicates deferred);
> agreed takeaways live on the E01 card (§IN-100, T10-T15) + DECISIONS.


## Alignment — `alignment` (pairs)

| method   | h_space         | z_final                   |   value_h |   value_z |      tau |   null_h(randinit) |   null_z(randinit) | predicted h/z   |
|:---------|:----------------|:--------------------------|----------:|----------:|---------:|-------------------:|-------------------:|:----------------|
| simclr   | student.h.gap   | student.z.proj.out        |   0.2815  |    0.4281 |   0.6577 |              1.246 |             0.7795 | **?/✓**         |
| byol     | student.h.gap   | student.z.pred.out        |   0.1731  |    0.2768 |   0.6255 |              1.246 |             0.536  | **?/✓**         |
| vicreg   | student.h.gap   | student.z.proj.out        |   0.1274  |    0.5999 |   0.2123 |              1.246 |             0.5909 | ?/✓             |
| dino     | teacher.h.cls   | teacher.z.dino.bottleneck |   0.6391  |    0.4436 |   1.441  |              1.23  |             1.005  | ?/✓             |
| mae      | student.h.gap   | —                         |   0.5628  |  nan      | nan      |              1.246 |           nan      | ✗/—             |
| ijepa    | teacher.h.gap   | student.z.pred.out        |   1.089   |    1.108  |   0.9827 |              1.246 |             0.2097 | ?/✓             |
| lejepa   | student.z.embed | student.z.proj.out        |   0.09018 |    0.1831 |   0.4925 |              1.179 |             0.6612 | ?/✓             |



## Uniformity — `uniformity` (raw|full)

| method   | h_space         | z_final                   |   value_h |   value_z |      tau |   null_h(randinit) |   null_z(randinit) | predicted h/z   |
|:---------|:----------------|:--------------------------|----------:|----------:|---------:|-------------------:|-------------------:|:----------------|
| simclr   | student.h.gap   | student.z.proj.out        |   -1.664  |    -3.821 |   0.4356 |             -1.678 |            -1.516  | ✗/✓             |
| byol     | student.h.gap   | student.z.pred.out        |   -0.8778 |    -2.913 |   0.3014 |            nan     |           nan      | ✗/~             |
| vicreg   | student.h.gap   | student.z.proj.out        |   -0.6416 |    -3.925 |   0.1635 |            nan     |           nan      | ✗/~             |
| dino     | teacher.h.cls   | teacher.z.dino.bottleneck |   -3.245  |    -3.53  |   0.9192 |             -1.884 |            -1.799  | **✗/~**         |
| mae      | student.h.gap   | —                         |   -1.363  |   nan     | nan      |             -1.678 |           nan      | ✗/—             |
| ijepa    | teacher.h.gap   | student.z.pred.out        |   -2.49   |    -2.405 |   1.035  |             -1.678 |            -0.3429 | ✗/✗             |
| lejepa   | student.z.embed | student.z.proj.out        |   -1.435  |    -3.319 |   0.4325 |             -1.879 |            -1.276  | ?/✓             |



## Variance floor (scale-free) — `variance_floor.min_over_mean_std` (raw|full)

| method   | h_space         | z_final                   |   value_h |   value_z |      tau |   null_h(randinit) |   null_z(randinit) | predicted h/z   |
|:---------|:----------------|:--------------------------|----------:|----------:|---------:|-------------------:|-------------------:|:----------------|
| simclr   | student.h.gap   | student.z.proj.out        |    0.5055 |    0.8352 |   0.6052 |             0.2555 |             0.4378 | ?/✓             |
| byol     | student.h.gap   | student.z.pred.out        |    0.3742 |    0.5292 |   0.7072 |           nan      |           nan      | ?/~             |
| vicreg   | student.h.gap   | student.z.proj.out        |    0.5001 |    0.9864 |   0.507  |           nan      |           nan      | **~/✓**         |
| dino     | teacher.h.cls   | teacher.z.dino.bottleneck |    0.1551 |    0.8319 |   0.1864 |             0.3796 |             0.4938 | ?/?             |
| mae      | student.h.gap   | —                         |    0.3154 |  nan      | nan      |             0.2555 |           nan      | ?/—             |
| ijepa    | teacher.h.gap   | student.z.pred.out        |    0.3775 |    0.3806 |   0.9917 |             0.2555 |             0.3037 | ?/?             |
| lejepa   | student.z.embed | student.z.proj.out        |    0.3856 |    0.945  |   0.408  |             0.3747 |             0.5775 | ?/✓             |



## Var. floor (hinge, paired) — `variance_floor.hinge` (raw|full)

| method   | h_space         | z_final                   |   value_h |   value_z |     tau |   null_h(randinit) |   null_z(randinit) | predicted h/z   |
|:---------|:----------------|:--------------------------|----------:|----------:|--------:|-------------------:|-------------------:|:----------------|
| simclr   | student.h.gap   | student.z.proj.out        |    0.8752 |   0       | nan     |             0.5105 |             0.8395 |                 |
| byol     | student.h.gap   | student.z.pred.out        |    0.8925 |   0       | nan     |           nan      |           nan      |                 |
| vicreg   | student.h.gap   | student.z.proj.out        |    0.9054 |   0.1908  |   4.746 |           nan      |           nan      |                 |
| dino     | teacher.h.cls   | teacher.z.dino.bottleneck |    0.6297 |   0       | nan     |             0.2294 |             0.953  |                 |
| mae      | student.h.gap   | —                         |    0.8991 | nan       | nan     |             0.5105 |           nan      |                 |
| ijepa    | teacher.h.gap   | student.z.pred.out        |    0.8565 |   0.5735  |   1.493 |             0.5105 |             0.8463 |                 |
| lejepa   | student.z.embed | student.z.proj.out        |    0.7328 |   0.02468 |  29.69  |             0.6886 |             0.9801 |                 |



## Decorrelation — `offdiag_redundancy.mean_abs_corr` (raw|full)

| method   | h_space         | z_final                   |   value_h |   value_z |      tau |   null_h(randinit) |   null_z(randinit) | predicted h/z   |
|:---------|:----------------|:--------------------------|----------:|----------:|---------:|-------------------:|-------------------:|:----------------|
| simclr   | student.h.gap   | student.z.proj.out        |   0.1298  |   0.08695 |   1.493  |             0.4748 |             0.3163 | ✗/~             |
| byol     | student.h.gap   | student.z.pred.out        |   0.1208  |   0.1498  |   0.8061 |           nan      |           nan      | ✗/~             |
| vicreg   | student.h.gap   | student.z.proj.out        |   0.1288  |   0.03649 |   3.53   |           nan      |           nan      | **✗/✓**         |
| dino     | teacher.h.cls   | teacher.z.dino.bottleneck |   0.08485 |   0.08735 |   0.9715 |             0.3766 |             0.333  | ✗/?             |
| mae      | student.h.gap   | —                         |   0.1525  | nan       | nan      |             0.4748 |           nan      | ✗/—             |
| ijepa    | teacher.h.gap   | student.z.pred.out        |   0.09999 |   0.1027  |   0.9735 |             0.4748 |             0.4364 | ✗/?             |
| lejepa   | student.z.embed | student.z.proj.out        |   0.1504  |   0.0239  |   6.294  |             0.3712 |             0.2637 | ?/~             |



## Eff. rank — `rankme` (raw|full)

| method   | h_space         | z_final                   |   value_h |   value_z |      tau |   null_h(randinit) |   null_z(randinit) | predicted h/z   |
|:---------|:----------------|:--------------------------|----------:|----------:|---------:|-------------------:|-------------------:|:----------------|
| simclr   | student.h.gap   | student.z.proj.out        |     136.4 |    101.2  |   1.348  |              53.05 |             151.7  | ~/✗             |
| byol     | student.h.gap   | student.z.pred.out        |     119.3 |     38.97 |   3.063  |             nan    |             nan    | ~/~             |
| vicreg   | student.h.gap   | student.z.proj.out        |     116.1 |    562.3  |   0.2066 |             nan    |             nan    | ~/✓             |
| dino     | teacher.h.cls   | teacher.z.dino.bottleneck |     203.6 |    117.8  |   1.728  |              91.52 |              82.12 | ~/?             |
| mae      | student.h.gap   | —                         |     103   |    nan    | nan      |              53.05 |             nan    | **✗**/—         |
| ijepa    | teacher.h.gap   | student.z.pred.out        |     158.6 |    135.3  |   1.172  |              53.05 |              20.76 | ?/?             |
| lejepa   | student.z.embed | student.z.proj.out        |      35   |     15.97 |   2.192  |              79.52 |              11.05 | ?/✓             |



## Isotropy/Gauss. — `kurt_topeig.worst` (raw|full)

| method   | h_space         | z_final                   |   value_h |   value_z |        tau |   null_h(randinit) |   null_z(randinit) | predicted h/z   |
|:---------|:----------------|:--------------------------|----------:|----------:|-----------:|-------------------:|-------------------:|:----------------|
| simclr   | student.h.gap   | student.z.proj.out        |    0.8431 |    84.72  |   0.009951 |              5.934 |              1.531 | ✗/~             |
| byol     | student.h.gap   | student.z.pred.out        |    1.399  |     7.337 |   0.1907   |            nan     |            nan     | ✗/✗             |
| vicreg   | student.h.gap   | student.z.proj.out        |    2.982  |   245.4   |   0.01215  |            nan     |            nan     | ✗/~             |
| dino     | teacher.h.cls   | teacher.z.dino.bottleneck |    0.8932 |    66.23  |   0.01349  |              1.512 |              1.546 | ✗/✗             |
| mae      | student.h.gap   | —                         |    2.429  |   nan     | nan        |              5.934 |            nan     | ✗/—             |
| ijepa    | teacher.h.gap   | student.z.pred.out        |    1.74   |     1.688 |   1.031    |              5.934 |              4.08  | ✗/✗             |
| lejepa   | student.z.embed | student.z.proj.out        |    1.497  |     2.061 |   0.7265   |              2.11  |              1.305 | **?/✓**         |



## Isotropy (EP, paired) — `epps_pulley` (raw|full)

| method   | h_space         | z_final                   |   value_h |   value_z |      tau |   null_h(randinit) |   null_z(randinit) | predicted h/z   |
|:---------|:----------------|:--------------------------|----------:|----------:|---------:|-------------------:|-------------------:|:----------------|
| simclr   | student.h.gap   | student.z.proj.out        |     330.4 |     337.6 |   0.9786 |               3408 |               2370 |                 |
| byol     | student.h.gap   | student.z.pred.out        |     336.7 |     668.4 |   0.5037 |                nan |                nan |                 |
| vicreg   | student.h.gap   | student.z.proj.out        |     338.9 |      44.3 |   7.65   |                nan |                nan |                 |
| dino     | teacher.h.cls   | teacher.z.dino.bottleneck |     159.6 |     776.9 |   0.2055 |               3005 |               2331 |                 |
| mae      | student.h.gap   | —                         |     537.9 |     nan   | nan      |               3408 |                nan |                 |
| ijepa    | teacher.h.gap   | student.z.pred.out        |     267.8 |     284.1 |   0.9426 |               3408 |               3135 |                 |
| lejepa   | student.z.embed | student.z.proj.out        |     651.9 |     152.3 |   4.281  |               3011 |               1571 |                 |



## Aug.-invariance — `cos_invariance` (pairs)

| method   | h_space         | z_final                   |   value_h |   value_z |      tau |   null_h(randinit) |   null_z(randinit) | predicted h/z   |
|:---------|:----------------|:--------------------------|----------:|----------:|---------:|-------------------:|-------------------:|:----------------|
| simclr   | student.h.gap   | student.z.proj.out        |    0.8592 |    0.786  |   1.093  |             0.3769 |             0.6102 | **✗/✓**         |
| byol     | student.h.gap   | student.z.pred.out        |    0.9134 |    0.8616 |   1.06   |             0.3769 |             0.732  | ✗/✓             |
| vicreg   | student.h.gap   | student.z.proj.out        |    0.9363 |    0.7001 |   1.337  |             0.3769 |             0.7046 | ✗/✓             |
| dino     | teacher.h.cls   | teacher.z.dino.bottleneck |    0.6804 |    0.7782 |   0.8743 |             0.3851 |             0.4976 | ✗/✓             |
| mae      | student.h.gap   | —                         |    0.7186 |  nan      | nan      |             0.3769 |           nan      | ✗/—             |
| ijepa    | teacher.h.gap   | student.z.pred.out        |    0.4556 |    0.446  |   1.021  |             0.3769 |             0.8952 | ✗ (no augs)     |
| lejepa   | student.z.embed | student.z.proj.out        |    0.9549 |    0.9084 |   1.051  |             0.4104 |             0.6694 | ? (mild stack)  |



## E01-T2 locked sign predictions vs numbers — UNSCORED (joint reading only)

| method   | locked prediction               |   kurt_worst h | kurt_worst z   | null_gauss z   |
|:---------|:--------------------------------|---------------:|:---------------|:---------------|
| simclr   | cluster-sharpening: kurt z >> h |          0.843 | 84.7           | 0.0533         |
| byol     | Gaussian-smoothing: kurt z < h  |          1.4   | 7.34           | 0.0595         |
| vicreg   | cluster-sharpening: kurt z >> h |          2.98  | 245            | 0.0467         |
| dino     | cluster-sharpening: kurt z >> h |          0.893 | 66.2           | 0.069          |
| mae      | no z prediction (pixel loss)    |          2.43  | —              | —              |
| ijepa    | flat (z trained to be h-space)  |          1.74  | 1.69           | 0.0358         |
| lejepa   | Gaussian-smoothing: kurt z < h  |          1.5   | 2.06           | 0.0397         |



## Headline probes (D-020 v2 pair) at h and z.final

> continuity columns (v1 raw @30ep — known censored at GAP spaces — and house_v2) shown for reference; headline = raw_v2 + knn.

| method   | at      | space                     |   linear_raw_v2 |   knn_v1_k200 |   linear_raw_v1 |   linear_house_v2 |
|:---------|:--------|:--------------------------|----------------:|--------------:|----------------:|------------------:|
| simclr   | h       | student.h.gap             |          0.566  |        0.3822 |          0.5116 |            0.56   |
| simclr   | z.final | student.z.proj.out        |          0.541  |        0.5114 |          0.5352 |            0.5482 |
| byol     | h       | student.h.gap             |          0.568  |        0.3898 |          0.508  |            0.562  |
| byol     | z.final | student.z.pred.out        |          0.4728 |        0.4468 |          0.453  |            0.5164 |
| vicreg   | h       | student.h.gap             |          0.5898 |        0.3936 |          0.5144 |            0.5866 |
| vicreg   | z.final | student.z.proj.out        |          0.5878 |        0.5624 |          0.5878 |            0.5834 |
| dino     | h       | teacher.h.cls             |          0.6864 |        0.5994 |          0.6852 |            0.6838 |
| dino     | z.final | teacher.z.dino.bottleneck |          0.5306 |        0.4978 |          0.5292 |            0.5362 |
| mae      | h       | student.h.gap             |          0.4344 |        0.2396 |          0.3772 |            0.4306 |
| ijepa    | h       | teacher.h.gap             |          0.4836 |        0.3474 |          0.4338 |            0.4838 |
| ijepa    | z.final | student.z.pred.out        |          0.4614 |        0.336  |          0.4388 |            0.4628 |
| lejepa   | h       | student.z.embed           |          0.6022 |        0.524  |          0.588  |            0.61   |
| lejepa   | z.final | student.z.proj.out        |          0.5008 |        0.4632 |          0.4864 |            0.4924 |


## Anchor rows (D-007/D-022) — supervised deitlite + shared randinit

> Reference rows, never comparison cells (provenance stays separate). deitlite = minimal supervised anchor (CE on CLS, RRC+flip only; its own classifier hit 0.6182 val @ training best (wandb in100.deitlite.s0)). randinit = the shared untrained floor. Battery values at anchor spaces; probes same family as above.

| desideratum                 |   deitlite h.cls |   deitlite h.gap |   deitlite z.logits |   randinit h.cls |   randinit h.gap |
|:----------------------------|-----------------:|-----------------:|--------------------:|-----------------:|-----------------:|
| Alignment                   |        1.474     |         0.717    |              1.287  |           1.23   |           1.254  |
| Uniformity                  |       -3.825     |        -1.809    |             -3.545  |          -1.884  |          -1.678  |
| Variance floor (scale-free) |        0.0009494 |         0.002239 |              0.8741 |           0.3796 |           0.2555 |
| Var. floor (hinge, paired)  |        0.3174    |         0.8149   |              0      |           0.2294 |           0.5105 |
| Decorrelation               |        0.05749   |         0.09307  |              0.153  |           0.3766 |           0.4748 |
| Eff. rank                   |      254         |       217.2      |             86.49   |          91.52   |          53.05   |
| Isotropy/Gauss.             |        0.9978    |         1.181    |              0.9977 |           1.512  |           5.934  |
| Isotropy (EP, paired)       |       79.39      |       248.9      |            421.5    |        3005      |        3408      |
| Aug.-invariance             |        0.263     |         0.6415   |              0.3563 |           0.3851 |           0.373  |


### Anchor probes

| anchor   | at       | space            |   linear_raw_v2 |   knn_v1_k200 |   linear_raw_v1 |   linear_house_v2 |
|:---------|:---------|:-----------------|----------------:|--------------:|----------------:|------------------:|
| deitlite | h.cls    | student.h.cls    |          0.6236 |        0.6158 |          0.6236 |            0.621  |
| deitlite | h.gap    | student.h.gap    |          0.5388 |        0.3664 |          0.5248 |            0.533  |
| deitlite | z.logits | student.z.logits |          0.6146 |        0.6052 |          0.614  |            0.6094 |
| randinit | h.cls    | student.h.cls    |          0.1168 |        0.0622 |          0.0894 |            0.1204 |
| randinit | h.gap    | student.h.gap    |          0.1518 |        0.0712 |          0.1102 |            0.153  |


## AGREED TAKEAWAY

*(lives on the E01 card §IN-100 rung, T10–T15, agreed 2026-07-10 — this generated file never holds takeaway text; glyph scoring per D-024 lands on the card once all own-arch nulls are in)*
