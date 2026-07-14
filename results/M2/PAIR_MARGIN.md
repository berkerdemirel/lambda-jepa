# M2 pair margins + scale check — IN-100 seed 0 (numbers only)

> Emitted by experiments/pair_margin_m2.py. Definitions as in results/M1/PAIR_MARGIN.md
> (cos_margin = pos_cos − rand_cos; align_rel = align_pos/align_rand, lower = more
> view-invariant relative to the space's own compactness). audit_v1 stack only at this
> rung. Null = shared in100.randinit-s0.ext (h-side spaces only).


## Headline h vs z (audit_v1 stack, trained runs)

| method   | h: pos/rand cos   |   h margin | z: pos/rand cos   |   z margin |   h align_rel |   z align_rel |
|:---------|:------------------|-----------:|:------------------|-----------:|--------------:|--------------:|
| simclr   | 0.857 / 0.483     |      0.374 | 0.786 / 0.001     |      0.785 |         0.276 |         0.214 |
| vicreg   | 0.921 / 0.710     |      0.21  | 0.700 / 0.011     |      0.689 |         0.274 |         0.303 |
| byol     | 0.934 / 0.764     |      0.169 | 0.862 / 0.178     |      0.683 |         0.281 |         0.168 |
| dino     | 0.680 / 0.145     |      0.535 | 0.778 / 0.095     |      0.684 |         0.374 |         0.245 |
| mae      | 0.719 / 0.586     |      0.132 | —                 |    nan     |         0.68  |       nan     |
| ijepa    | 0.456 / 0.278     |      0.177 | 0.446 / 0.280     |      0.166 |         0.755 |         0.77  |
| lejepa   | 0.955 / 0.583     |      0.372 | 0.908 / -0.004    |      0.912 |         0.108 |         0.091 |


## Scale check — per-dim std, clean manifest vs audit-stack pair store

| method   | space                     |   mean_std_clean |   hinge_clean |   mean_std_aug |   hinge_aug |   mean_std_branchA |   hinge_branchA |   mean_std_within |
|:---------|:--------------------------|-----------------:|--------------:|---------------:|------------:|-------------------:|----------------:|------------------:|
| simclr   | student.z.proj.out        |           1.7763 |        0.0000 |         1.6972 |      0.0000 |             1.7011 |          0.0000 |            0.7458 |
| simclr   | student.h.cls             |           0.1950 |        0.8053 |         0.2025 |      0.7977 |             0.2025 |          0.7977 |            0.1112 |
| vicreg   | student.z.proj.out        |           0.8092 |        0.1908 |         0.7464 |      0.2536 |             0.7467 |          0.2533 |            0.3778 |
| vicreg   | student.h.cls             |           0.2087 |        0.7914 |         0.2109 |      0.7891 |             0.2109 |          0.7891 |            0.1144 |
| byol     | student.z.pred.out        |          13.8801 |        0.0000 |        11.8923 |      0.0000 |            11.9121 |          0.0000 |            4.7690 |
| byol     | student.h.cls             |           0.1573 |        0.8427 |         0.1609 |      0.8391 |             0.1610 |          0.8390 |            0.0871 |
| dino     | teacher.z.dino.bottleneck |           1.6551 |        0.0000 |         1.4733 |      0.0000 |             1.4770 |          0.0000 |            0.7183 |
| dino     | teacher.h.cls             |           0.3704 |        0.6297 |         0.3728 |      0.6274 |             0.3728 |          0.6274 |            0.2297 |
| mae      | student.h.gap             |           0.1013 |        0.8991 |         0.1195 |      0.8812 |             0.1196 |          0.8811 |            0.0981 |
| ijepa    | teacher.h.gap             |           0.1435 |        0.8565 |         0.1699 |      0.8304 |             0.1700 |          0.8304 |            0.1502 |
| ijepa    | student.z.pred.out        |           0.4431 |        0.5735 |         0.5141 |      0.5114 |             0.5145 |          0.5110 |            0.4584 |
| lejepa   | student.z.embed           |           0.2672 |        0.7328 |         0.2721 |      0.7279 |             0.2722 |          0.7278 |            0.0917 |
| lejepa   | student.z.proj.out        |           0.9753 |        0.0247 |         0.9529 |      0.0471 |             0.9531 |          0.0469 |            0.2585 |

