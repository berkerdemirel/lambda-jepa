# M1 — toy E1 matrix (numbers only, dress rehearsal)

> Emitted by experiments/report_m1.py. Toy rung: recipe bring-up frame — protocol
> shakedown, NOT locked interpretation (D-012 pending). Predicted glyphs = AUDIT_MATRIX
> v1 (LOCKED); bold = the method's own desideratum. τ = value(h)/value(z) (E01).
> Dim-sensitive metrics: raw|full shown; raw|pca64 in the appendix blocks (§6.2).


## Alignment — `alignment` (pairs)

| method   | h_space         | z_final                   |   value_h |   value_z |       tau |   null_h |   null_z | predicted h/z   |
|:---------|:----------------|:--------------------------|----------:|----------:|----------:|---------:|---------:|:----------------|
| simclr   | student.h.gap   | student.z.proj.out        |   0.1449  |    0.6747 |   0.2148  |    1.239 |   0.7825 | **?/✓**         |
| byol     | student.h.gap   | student.z.pred.out        |   0.1038  |    0.3916 |   0.2651  |    1.212 |   0.59   | **?/✓**         |
| vicreg   | student.h.gap   | student.z.proj.out        |   0.08969 |    1.039  |   0.08634 |    1.206 |   0.5816 | ?/✓             |
| dino     | teacher.h.cls   | teacher.z.dino.bottleneck |   0.5461  |    0.7332 |   0.7449  |    1.183 |   0.9815 | ?/✓             |
| mae      | student.h.gap   | —                         |   0.1753  |  nan      | nan       |    1.172 | nan      | ✗/—             |
| ijepa    | teacher.h.gap   | student.z.pred.out        |   0.388   |    0.3564 |   1.088   |    1.193 |   0.2439 | ?/✓             |
| lejepa   | student.z.embed | student.z.proj.out        |   0.01148 |    0.4151 |   0.02767 |    1.138 |   0.5899 | ?/✓             |



## Uniformity — `uniformity` (raw|full)

| method   | h_space         | z_final                   |   value_h |   value_z |       tau |   null_h |   null_z | predicted h/z   |
|:---------|:----------------|:--------------------------|----------:|----------:|----------:|---------:|---------:|:----------------|
| simclr   | student.h.gap   | student.z.proj.out        |  -0.4503  |   -3.606  |   0.1249  |   -1.643 |  -1.476  | ✗/✓             |
| byol     | student.h.gap   | student.z.pred.out        |  -0.3301  |   -2.051  |   0.161   |   -1.643 |  -1.172  | ✗/~             |
| vicreg   | student.h.gap   | student.z.proj.out        |  -0.3241  |   -3.843  |   0.08433 |   -1.643 |  -1.204  | ✗/~             |
| dino     | teacher.h.cls   | teacher.z.dino.bottleneck |  -1.713   |   -3.202  |   0.5351  |   -1.802 |  -1.691  | **✗/~**         |
| mae      | student.h.gap   | —                         |  -0.3638  |  nan      | nan       |   -1.643 | nan      | ✗/—             |
| ijepa    | teacher.h.gap   | student.z.pred.out        |  -0.5388  |   -0.5056 |   1.066   |   -1.643 |  -0.3851 | ✗/✗             |
| lejepa   | student.z.embed | student.z.proj.out        |  -0.09832 |   -3.182  |   0.0309  |   -1.802 |  -1.299  | ?/✓             |



## Variance floor — `variance_floor.hinge` (raw|full)

| method   | h_space         | z_final                   |   value_h |   value_z |     tau |   null_h |   null_z | predicted h/z   |
|:---------|:----------------|:--------------------------|----------:|----------:|--------:|---------:|---------:|:----------------|
| simclr   | student.h.gap   | student.z.proj.out        |    0.7626 |   0       | nan     |   0.5243 |   0.8443 | ?/✓             |
| byol     | student.h.gap   | student.z.pred.out        |    0.7806 |   0       | nan     |   0.5243 |   0.9659 | ?/~             |
| vicreg   | student.h.gap   | student.z.proj.out        |    0.7337 |   0.3076  |   2.385 |   0.5243 |   0.9449 | **~/✓**         |
| dino     | teacher.h.cls   | teacher.z.dino.bottleneck |    0.4556 |   0       | nan     |   0.2539 |   0.9569 | ?/?             |
| mae      | student.h.gap   | —                         |    0.8416 | nan       | nan     |   0.5243 | nan      | ?/—             |
| ijepa    | teacher.h.gap   | student.z.pred.out        |    0.77   |   0.7195  |   1.07  |   0.5243 |   0.8264 | ?/?             |
| lejepa   | student.z.embed | student.z.proj.out        |    0.6931 |   0.02843 |  24.38  |   0.6925 |   0.9784 | ?/✓             |



## Decorrelation — `offdiag_redundancy.mean_abs_corr` (raw|full)

| method   | h_space         | z_final                   |   value_h |   value_z |      tau |   null_h |   null_z | predicted h/z   |
|:---------|:----------------|:--------------------------|----------:|----------:|---------:|---------:|---------:|:----------------|
| simclr   | student.h.gap   | student.z.proj.out        |    0.2279 |   0.165   |   1.381  |   0.4866 |   0.3194 | ✗/~             |
| byol     | student.h.gap   | student.z.pred.out        |    0.2903 |   0.339   |   0.8563 |   0.4866 |   0.3038 | ✗/~             |
| vicreg   | student.h.gap   | student.z.proj.out        |    0.1701 |   0.04957 |   3.431  |   0.4866 |   0.2777 | **✗/✓**         |
| dino     | teacher.h.cls   | teacher.z.dino.bottleneck |    0.1552 |   0.2325  |   0.6674 |   0.3716 |   0.3145 | ✗/?             |
| mae      | student.h.gap   | —                         |    0.2182 | nan       | nan      |   0.4866 | nan      | ✗/—             |
| ijepa    | teacher.h.gap   | student.z.pred.out        |    0.199  |   0.1995  |   0.9979 |   0.4866 |   0.4919 | ✗/?             |
| lejepa   | student.z.embed | student.z.proj.out        |    0.2169 |   0.05013 |   4.327  |   0.3654 |   0.3157 | ?/~             |



## Eff. rank — `rankme` (raw|full)

| method   | h_space         | z_final                   |   value_h |   value_z |      tau |   null_h |   null_z | predicted h/z   |
|:---------|:----------------|:--------------------------|----------:|----------:|---------:|---------:|---------:|:----------------|
| simclr   | student.h.gap   | student.z.proj.out        |    54.9   |     52.87 |   1.038  |    44.28 |   137.1  | ~/✗             |
| byol     | student.h.gap   | student.z.pred.out        |    25.15  |     10.72 |   2.345  |    44.28 |    83.84 | ~/~             |
| vicreg   | student.h.gap   | student.z.proj.out        |    55.1   |    512    |   0.1076 |    44.28 |   365.7  | ~/✓             |
| dino     | teacher.h.cls   | teacher.z.dino.bottleneck |    96.7   |     68.83 |   1.405  |    77.84 |    75.01 | ~/?             |
| mae      | student.h.gap   | —                         |    37.95  |    nan    | nan      |    44.28 |   nan    | **✗**/—         |
| ijepa    | teacher.h.gap   | student.z.pred.out        |    46.06  |     30.89 |   1.491  |    44.28 |    17.98 | ?/?             |
| lejepa   | student.z.embed | student.z.proj.out        |     9.423 |     15.86 |   0.5942 |    72.07 |    11.06 | ?/✓             |



## Isotropy/Gauss. — `kurt_topeig.worst` (raw|full)

| method   | h_space         | z_final                   |   value_h |   value_z |       tau |   null_h |   null_z | predicted h/z   |
|:---------|:----------------|:--------------------------|----------:|----------:|----------:|---------:|---------:|:----------------|
| simclr   | student.h.gap   | student.z.proj.out        |    2.038  |   26.62   |   0.07655 |    5.489 |    1.531 | ✗/~             |
| byol     | student.h.gap   | student.z.pred.out        |    3.158  |    0.7568 |   4.172   |    5.489 |    1.512 | ✗/✗             |
| vicreg   | student.h.gap   | student.z.proj.out        |    1.441  |   23.98   |   0.06012 |    5.489 |    1.557 | ✗/~             |
| dino     | teacher.h.cls   | teacher.z.dino.bottleneck |    0.4819 |   20.36   |   0.02367 |    1.5   |    1.697 | ✗/✗             |
| mae      | student.h.gap   | —                         |    2.346  |  nan      | nan       |    5.489 |  nan     | ✗/—             |
| ijepa    | teacher.h.gap   | student.z.pred.out        |    2.077  |    2.496  |   0.8322  |    5.49  |    4.885 | ✗/✗             |
| lejepa   | student.z.embed | student.z.proj.out        |    3.624  |    1.516  |   2.39    |    1.449 |    1.406 | **?/✓**         |



## Isotropy (EP, paired) — `epps_pulley` (raw|full)

| method   | h_space         | z_final                   |   value_h |   value_z |      tau |   null_h |   null_z | predicted h/z   |
|:---------|:----------------|:--------------------------|----------:|----------:|---------:|---------:|---------:|:----------------|
| simclr   | student.h.gap   | student.z.proj.out        |    238.8  |    286.3  |   0.8339 |    620.2 |    390.9 |                 |
| byol     | student.h.gap   | student.z.pred.out        |    363.4  |    423.5  |   0.858  |    620.2 |    431.3 |                 |
| vicreg   | student.h.gap   | student.z.proj.out        |    126.7  |     12.94 |   9.791  |    620.2 |    374.3 |                 |
| dino     | teacher.h.cls   | teacher.z.dino.bottleneck |     83.78 |    410.6  |   0.204  |    580.8 |    477.1 |                 |
| mae      | student.h.gap   | —                         |    207.1  |    nan    | nan      |    620.2 |    nan   |                 |
| ijepa    | teacher.h.gap   | student.z.pred.out        |    175.7  |    179.8  |   0.9771 |    620.2 |    757.7 |                 |
| lejepa   | student.z.embed | student.z.proj.out        |    203    |     54.89 |   3.699  |    569.4 |    405.1 |                 |



## Aug.-invariance — `cos_invariance` (pairs)

| method   | h_space         | z_final                   |   value_h |   value_z |      tau |   null_h |   null_z | predicted h/z   |
|:---------|:----------------|:--------------------------|----------:|----------:|---------:|---------:|---------:|:----------------|
| simclr   | student.h.gap   | student.z.proj.out        |    0.9275 |    0.6627 |   1.4    |   0.3804 |   0.6088 | **✗/✓**         |
| byol     | student.h.gap   | student.z.pred.out        |    0.9481 |    0.8042 |   1.179  |   0.3942 |   0.705  | ✗/✓             |
| vicreg   | student.h.gap   | student.z.proj.out        |    0.9552 |    0.4806 |   1.987  |   0.3971 |   0.7092 | ✗/✓             |
| dino     | teacher.h.cls   | teacher.z.dino.bottleneck |    0.7269 |    0.6334 |   1.148  |   0.4087 |   0.5093 | ✗/✓             |
| mae      | student.h.gap   | —                         |    0.9124 |  nan      | nan      |   0.4139 | nan      | ✗/—             |
| ijepa    | teacher.h.gap   | student.z.pred.out        |    0.806  |    0.8218 |   0.9808 |   0.4033 |   0.878  | ✗ (no augs)     |
| lejepa   | student.z.embed | student.z.proj.out        |    0.9943 |    0.7924 |   1.255  |   0.4309 |   0.705  | ? (mild stack)  |



## Headline probes (D-006v2 pair) at h and z.final

| method   | at      | space                     | probe         |   val_acc |   null |
|:---------|:--------|:--------------------------|:--------------|----------:|-------:|
| simclr   | h       | student.h.gap             | linear_raw_v1 |    0.6698 | 0.4010 |
| simclr   | h       | student.h.gap             | knn_v1_k200   |    0.5350 | 0.3325 |
| simclr   | z.final | student.z.proj.out        | linear_raw_v1 |    0.7159 | 0.3554 |
| simclr   | z.final | student.z.proj.out        | knn_v1_k200   |    0.7200 | 0.3200 |
| byol     | h       | student.h.gap             | linear_raw_v1 |    0.6324 | 0.4010 |
| byol     | h       | student.h.gap             | knn_v1_k200   |    0.5450 | 0.3325 |
| byol     | z.final | student.z.pred.out        | linear_raw_v1 |    0.5518 | 0.2907 |
| byol     | z.final | student.z.pred.out        | knn_v1_k200   |    0.5697 | 0.3175 |
| vicreg   | h       | student.h.gap             | linear_raw_v1 |    0.7371 | 0.4010 |
| vicreg   | h       | student.h.gap             | knn_v1_k200   |    0.6204 | 0.3325 |
| vicreg   | z.final | student.z.proj.out        | linear_raw_v1 |    0.7855 | 0.3654 |
| vicreg   | z.final | student.z.proj.out        | knn_v1_k200   |    0.7725 | 0.3103 |
| dino     | h       | teacher.h.cls             | linear_raw_v1 |    0.7738 | 0.3946 |
| dino     | h       | teacher.h.cls             | knn_v1_k200   |    0.7029 | 0.3172 |
| dino     | z.final | teacher.z.dino.bottleneck | linear_raw_v1 |    0.7567 | 0.3019 |
| dino     | z.final | teacher.z.dino.bottleneck | knn_v1_k200   |    0.7218 | 0.3159 |
| mae      | h       | student.h.gap             | linear_raw_v1 |    0.6688 | 0.4010 |
| mae      | h       | student.h.gap             | knn_v1_k200   |    0.5276 | 0.3325 |
| ijepa    | h       | teacher.h.gap             | linear_raw_v1 |    0.6525 | 0.4010 |
| ijepa    | h       | teacher.h.gap             | knn_v1_k200   |    0.5195 | 0.3325 |
| ijepa    | z.final | student.z.pred.out        | linear_raw_v1 |    0.6487 | 0.3269 |
| ijepa    | z.final | student.z.pred.out        | knn_v1_k200   |    0.5480 | 0.2989 |
| lejepa   | h       | student.z.embed           | linear_raw_v1 |    0.8036 | 0.3648 |
| lejepa   | h       | student.z.embed           | knn_v1_k200   |    0.7878 | 0.3190 |
| lejepa   | z.final | student.z.proj.out        | linear_raw_v1 |    0.7911 | 0.2204 |
| lejepa   | z.final | student.z.proj.out        | knn_v1_k200   |    0.7832 | 0.2973 |



## Appendix — full per-space battery pointers + MAE decoder taps

- **simclr**: results/battery/toy.simclr.s0.ext.csv — 12 spaces: student.h.cls, student.h.cls.L03, student.h.cls.L06, student.h.cls.L09, student.h.cls.L12, student.h.gap, student.h.gap.L03, student.h.gap.L06, student.h.gap.L09, student.h.gap.L12, student.z.proj.out, student.z.proj.tap1
- **byol**: results/battery/toy.byol.s0.ext.csv — 26 spaces: student.h.cls, student.h.cls.L03, student.h.cls.L06, student.h.cls.L09, student.h.cls.L12, student.h.gap, student.h.gap.L03, student.h.gap.L06, student.h.gap.L09, student.h.gap.L12, student.z.pred.out, student.z.pred.tap1, student.z.proj.out, student.z.proj.tap1, teacher.h.cls, teacher.h.cls.L03, teacher.h.cls.L06, teacher.h.cls.L09, teacher.h.cls.L12, teacher.h.gap, teacher.h.gap.L03, teacher.h.gap.L06, teacher.h.gap.L09, teacher.h.gap.L12, teacher.z.proj.out, teacher.z.proj.tap1
- **vicreg**: results/battery/toy.vicreg.s0.ext.csv — 13 spaces: student.h.cls, student.h.cls.L03, student.h.cls.L06, student.h.cls.L09, student.h.cls.L12, student.h.gap, student.h.gap.L03, student.h.gap.L06, student.h.gap.L09, student.h.gap.L12, student.z.proj.out, student.z.proj.tap1, student.z.proj.tap2
- **dino**: results/battery/toy.dino.s0.ext.csv — 26 spaces: student.h.cls, student.h.cls.L03, student.h.cls.L06, student.h.cls.L09, student.h.cls.L12, student.h.gap, student.h.gap.L03, student.h.gap.L06, student.h.gap.L09, student.h.gap.L12, student.z.dino.bottleneck, student.z.dino.tap1, student.z.dino.tap2, teacher.h.cls, teacher.h.cls.L03, teacher.h.cls.L06, teacher.h.cls.L09, teacher.h.cls.L12, teacher.h.gap, teacher.h.gap.L03, teacher.h.gap.L06, teacher.h.gap.L09, teacher.h.gap.L12, teacher.z.dino.bottleneck, teacher.z.dino.tap1, teacher.z.dino.tap2
- **mae**: results/battery/toy.mae.s0.ext.csv — 13 spaces: student.h.cls, student.h.cls.L03, student.h.cls.L06, student.h.cls.L09, student.h.cls.L12, student.h.gap, student.h.gap.L03, student.h.gap.L06, student.h.gap.L09, student.h.gap.L12, student.z.dec.tap2, student.z.dec.tap5, student.z.dec.tap8
- **ijepa**: results/battery/toy.ijepa.s0.ext.csv — 21 spaces: student.h.cls, student.h.cls.L03, student.h.cls.L06, student.h.cls.L09, student.h.cls.L12, student.h.gap, student.h.gap.L03, student.h.gap.L06, student.h.gap.L09, student.h.gap.L12, student.z.pred.out, teacher.h.cls, teacher.h.cls.L03, teacher.h.cls.L06, teacher.h.cls.L09, teacher.h.cls.L12, teacher.h.gap, teacher.h.gap.L03, teacher.h.gap.L06, teacher.h.gap.L09, teacher.h.gap.L12
- **lejepa**: results/battery/toy.lejepa.s0.ext.csv — 14 spaces: student.h.cls, student.h.cls.L03, student.h.cls.L06, student.h.cls.L09, student.h.cls.L12, student.h.gap, student.h.gap.L03, student.h.gap.L06, student.h.gap.L09, student.h.gap.L12, student.z.embed, student.z.proj.out, student.z.proj.tap1, student.z.proj.tap2


## AGREED TAKEAWAY

*(empty — filled only after discussion; see CLAUDE.md)*
