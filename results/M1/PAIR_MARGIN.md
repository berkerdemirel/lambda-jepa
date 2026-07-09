# M1 verify pass — pair margins + scale check (numbers only)

> Emitted by experiments/pair_margin_m1.py. pair_margin = positive-pair vs random-pair
> contrast in the SAME space (random pairs: cross-view, different images, identical
> pipeline). cos_margin = pos_cos − rand_cos (higher = more view-invariance beyond
> compactness); align_rel = align_pos / align_rand (lower = same, relative version).
> Full grid incl. taps + null runs: results/M1/pair_margin.csv


## Headline h vs z (audit_v1 stack, trained runs)

| method   | h: pos/rand cos   |   h margin | z: pos/rand cos   |   z margin |   h align_rel |   z align_rel |
|:---------|:------------------|-----------:|:------------------|-----------:|--------------:|--------------:|
| simclr   | 0.928 / 0.846     |      0.081 | 0.663 / 0.000     |      0.662 |         0.471 |         0.337 |
| vicreg   | 0.955 / 0.910     |      0.045 | 0.481 / 0.037     |      0.443 |         0.498 |         0.54  |
| byol     | 0.948 / 0.882     |      0.066 | 0.804 / 0.325     |      0.479 |         0.439 |         0.29  |
| dino     | 0.727 / 0.498     |      0.228 | 0.633 / 0.135     |      0.499 |         0.544 |         0.424 |
| mae      | 0.912 / 0.876     |      0.036 | —                 |    nan     |         0.708 |       nan     |
| ijepa    | 0.806 / 0.751     |      0.055 | 0.822 / 0.774     |      0.048 |         0.778 |         0.787 |
| lejepa   | 0.994 / 0.976     |      0.018 | 0.792 / 0.025     |      0.767 |         0.243 |         0.213 |


## Scale check — per-dim std, clean manifest vs augmented pair store

> variance_floor.hinge is computed at gamma=1 on whatever scale the space has;
> VICReg trains its floor on augmented batches. hinge_clean vs hinge_aug per space:

| method   | space                     |   mean_std_clean |   hinge_clean |   mean_std_aug |   hinge_aug |   mean_std_branchA |   hinge_branchA |   mean_std_within |
|:---------|:--------------------------|-----------------:|--------------:|---------------:|------------:|-------------------:|----------------:|------------------:|
| simclr   | student.h.gap             |           0.2374 |        0.7626 |         0.2660 |      0.7340 |             0.2662 |          0.7338 |            0.1768 |
| simclr   | student.z.proj.out        |           4.2495 |        0.0000 |         4.2435 |      0.0000 |             4.2410 |          0.0000 |            2.1996 |
| vicreg   | student.h.gap             |           0.2663 |        0.7337 |         0.2698 |      0.7302 |             0.2701 |          0.7299 |            0.1615 |
| vicreg   | student.z.proj.out        |           0.6924 |        0.3076 |         0.6204 |      0.3796 |             0.6214 |          0.3786 |            0.3498 |
| byol     | student.h.gap             |           0.2194 |        0.7806 |         0.2315 |      0.7685 |             0.2331 |          0.7669 |            0.1256 |
| byol     | student.z.pred.out        |          15.5479 |        0.0000 |        15.1631 |      0.0000 |            15.2026 |          0.0000 |            6.1818 |
| dino     | teacher.z.dino.bottleneck |           5.2586 |        0.0000 |         4.7827 |      0.0000 |             4.8418 |          0.0000 |            1.7236 |
| dino     | teacher.h.cls             |           0.5547 |        0.4556 |         0.5495 |      0.4601 |             0.5493 |          0.4603 |            0.2958 |
| mae      | student.h.gap             |           0.1586 |        0.8416 |         0.1661 |      0.8345 |             0.1662 |          0.8343 |            0.0679 |
| ijepa    | teacher.h.gap             |           0.2305 |        0.7700 |         0.2359 |      0.7647 |             0.2356 |          0.7650 |            0.0725 |
| ijepa    | student.z.pred.out        |           0.2822 |        0.7195 |         0.2928 |      0.7092 |             0.2923 |          0.7098 |            0.1208 |
| lejepa   | student.z.proj.out        |           0.9862 |        0.0284 |       nan      |    nan      |           nan      |        nan      |          nan      |
| lejepa   | student.z.embed           |           0.3069 |        0.6931 |       nan      |    nan      |           nan      |        nan      |          nan      |

