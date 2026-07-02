# M0 — mini two-space audit (numbers only)

> Emitted by experiments/report_m0.py. Interpretation happens in discussion; AGREED TAKEAWAYS live in DECISIONS.md after user sign-off.


## toy.lejepa-lamb002.ext

### Battery (variant raw|full)

| model.space                                  |   rankme |   effective_rank |   alpha |   epps_pulley |   kurt_topeig.worst |   uniformity |   offdiag_redundancy.mean_abs_corr |   variance_floor.hinge |
|:---------------------------------------------|---------:|-----------------:|--------:|--------------:|--------------------:|-------------:|-----------------------------------:|-----------------------:|
| toy.lejepa-lamb002.ext · student.h.cls       |    92.86 |            35.99 |   2.129 |        107.8  |              0.8267 |     -1.102   |                            0.1515  |              0.8948    |
| toy.lejepa-lamb002.ext · student.h.cls.L03   |    23.99 |            30.52 |   2.192 |        115.2  |              0.6346 |     -0.06751 |                            0.1734  |              0.9647    |
| toy.lejepa-lamb002.ext · student.h.cls.L06   |    72.02 |            42.4  |   1.742 |         76.61 |              0.712  |     -0.2499  |                            0.1377  |              0.9358    |
| toy.lejepa-lamb002.ext · student.h.cls.L09   |    99.52 |            41.16 |   1.912 |         80.04 |              0.8506 |     -0.7693  |                            0.1368  |              0.904     |
| toy.lejepa-lamb002.ext · student.h.cls.L12   |    92.86 |            35.99 |   2.129 |        107.8  |              0.8267 |     -1.102   |                            0.1515  |              0.8948    |
| toy.lejepa-lamb002.ext · student.h.gap       |   103.6  |            39.83 |   1.609 |         73.77 |              1.556  |     -0.4495  |                            0.124   |              0.9543    |
| toy.lejepa-lamb002.ext · student.h.gap.L03   |    65.74 |            19.45 |   1.929 |        108.7  |              1.42   |     -0.3975  |                            0.1579  |              0.9627    |
| toy.lejepa-lamb002.ext · student.h.gap.L06   |   108.1  |            38.29 |   1.65  |         84.52 |              4.726  |     -0.5797  |                            0.1362  |              0.9525    |
| toy.lejepa-lamb002.ext · student.h.gap.L09   |   120.7  |            45.08 |   1.557 |         68.85 |              1.357  |     -0.6065  |                            0.1231  |              0.9509    |
| toy.lejepa-lamb002.ext · student.h.gap.L12   |   103.6  |            39.83 |   1.609 |         73.77 |              1.556  |     -0.4495  |                            0.124   |              0.9543    |
| toy.lejepa-lamb002.ext · student.z.embed     |    41.19 |            30.12 |   4.728 |         89.44 |              0.9914 |     -1.512   |                            0.1381  |              0.491     |
| toy.lejepa-lamb002.ext · student.z.proj.out  |    15.98 |            15.9  |   0.137 |         24.93 |              3.543  |     -3.349   |                            0.02109 |              0.0008017 |
| toy.lejepa-lamb002.ext · student.z.proj.tap1 |  1193    |           226.6  |   1.062 |         30.93 |              1.785  |     -3.505   |                            0.0598  |              0.974     |
| toy.lejepa-lamb002.ext · student.z.proj.tap2 |   946.3  |           120.3  |   1.36  |         79.16 |              1.015  |     -3.308   |                            0.04284 |              0.9588    |



### Transfer ratios τ = value(h) / value(z)


**student.h.cls vs student.z.embed**


| metric                           |   value_h |   value_z |    tau |    delta |
|:---------------------------------|----------:|----------:|-------:|---------:|
| uniformity                       |   -1.102  |   -1.512  | 0.7284 |  0.4107  |
| variance_floor.hinge             |    0.8948 |    0.491  | 1.822  |  0.4038  |
| offdiag_redundancy.mean_abs_corr |    0.1515 |    0.1381 | 1.097  |  0.01338 |
| rankme                           |   92.86   |   41.19   | 2.254  | 51.67    |
| effective_rank                   |   35.99   |   30.12   | 1.195  |  5.876   |
| alpha                            |    2.129  |    4.728  | 0.4502 | -2.599   |
| epps_pulley                      |  107.8    |   89.44   | 1.206  | 18.41    |
| kurt_topeig.worst                |    0.8267 |    0.9914 | 0.8338 | -0.1648  |



**student.h.cls vs student.z.proj.out**


| metric                           |   value_h |    value_z |       tau |   delta |
|:---------------------------------|----------:|-----------:|----------:|--------:|
| uniformity                       |   -1.102  | -3.349     |    0.329  |  2.247  |
| variance_floor.hinge             |    0.8948 |  0.0008017 | 1116      |  0.894  |
| offdiag_redundancy.mean_abs_corr |    0.1515 |  0.02109   |    7.183  |  0.1304 |
| rankme                           |   92.86   | 15.98      |    5.812  | 76.88   |
| effective_rank                   |   35.99   | 15.9       |    2.263  | 20.09   |
| alpha                            |    2.129  |  0.137     |   15.54   |  1.992  |
| epps_pulley                      |  107.8    | 24.93      |    4.326  | 82.91   |
| kurt_topeig.worst                |    0.8267 |  3.543     |    0.2333 | -2.716  |



**student.h.cls vs student.z.proj.tap1**


| metric                           |   value_h |   value_z |     tau |       delta |
|:---------------------------------|----------:|----------:|--------:|------------:|
| uniformity                       |   -1.102  |   -3.505  | 0.3143  |     2.403   |
| variance_floor.hinge             |    0.8948 |    0.974  | 0.9186  |    -0.07927 |
| offdiag_redundancy.mean_abs_corr |    0.1515 |    0.0598 | 2.533   |     0.09166 |
| rankme                           |   92.86   | 1193      | 0.07786 | -1100       |
| effective_rank                   |   35.99   |  226.6    | 0.1589  |  -190.6     |
| alpha                            |    2.129  |    1.062  | 2.004   |     1.067   |
| epps_pulley                      |  107.8    |   30.93   | 3.487   |    76.91    |
| kurt_topeig.worst                |    0.8267 |    1.785  | 0.4631  |    -0.9583  |



**student.h.cls vs student.z.proj.tap2**


| metric                           |   value_h |   value_z |     tau |      delta |
|:---------------------------------|----------:|----------:|--------:|-----------:|
| uniformity                       |   -1.102  |  -3.308   | 0.333   |    2.206   |
| variance_floor.hinge             |    0.8948 |   0.9588  | 0.9332  |   -0.06404 |
| offdiag_redundancy.mean_abs_corr |    0.1515 |   0.04284 | 3.536   |    0.1086  |
| rankme                           |   92.86   | 946.3     | 0.09812 | -853.5     |
| effective_rank                   |   35.99   | 120.3     | 0.2993  |  -84.26    |
| alpha                            |    2.129  |   1.36    | 1.565   |    0.7689  |
| epps_pulley                      |  107.8    |  79.16    | 1.362   |   28.69    |
| kurt_topeig.worst                |    0.8267 |   1.015   | 0.8146  |   -0.1881  |



**student.h.cls.L03 vs student.z.embed**


| metric                           |   value_h |   value_z |     tau |     delta |
|:---------------------------------|----------:|----------:|--------:|----------:|
| uniformity                       |  -0.06751 |   -1.512  | 0.04464 |   1.445   |
| variance_floor.hinge             |   0.9647  |    0.491  | 1.965   |   0.4737  |
| offdiag_redundancy.mean_abs_corr |   0.1734  |    0.1381 | 1.256   |   0.03533 |
| rankme                           |  23.99    |   41.19   | 0.5824  | -17.2     |
| effective_rank                   |  30.52    |   30.12   | 1.013   |   0.4054  |
| alpha                            |   2.192   |    4.728  | 0.4635  |  -2.536   |
| epps_pulley                      | 115.2     |   89.44   | 1.288   |  25.78    |
| kurt_topeig.worst                |   0.6346  |    0.9914 | 0.6401  |  -0.3569  |



**student.h.cls.L03 vs student.z.proj.out**


| metric                           |   value_h |    value_z |        tau |   delta |
|:---------------------------------|----------:|-----------:|-----------:|--------:|
| uniformity                       |  -0.06751 | -3.349     |    0.02016 |  3.281  |
| variance_floor.hinge             |   0.9647  |  0.0008017 | 1203       |  0.9639 |
| offdiag_redundancy.mean_abs_corr |   0.1734  |  0.02109   |    8.224   |  0.1523 |
| rankme                           |  23.99    | 15.98      |    1.501   |  8.012  |
| effective_rank                   |  30.52    | 15.9       |    1.919   | 14.62   |
| alpha                            |   2.192   |  0.137     |   16       |  2.055  |
| epps_pulley                      | 115.2     | 24.93      |    4.622   | 90.29   |
| kurt_topeig.worst                |   0.6346  |  3.543     |    0.1791  | -2.908  |



**student.h.cls.L03 vs student.z.proj.tap1**


| metric                           |   value_h |   value_z |     tau |        delta |
|:---------------------------------|----------:|----------:|--------:|-------------:|
| uniformity                       |  -0.06751 |   -3.505  | 0.01926 |     3.437    |
| variance_floor.hinge             |   0.9647  |    0.974  | 0.9904  |    -0.009332 |
| offdiag_redundancy.mean_abs_corr |   0.1734  |    0.0598 | 2.9     |     0.1136   |
| rankme                           |  23.99    | 1193      | 0.02012 | -1169        |
| effective_rank                   |  30.52    |  226.6    | 0.1347  |  -196        |
| alpha                            |   2.192   |    1.062  | 2.064   |     1.13     |
| epps_pulley                      | 115.2     |   30.93   | 3.725   |    84.29     |
| kurt_topeig.worst                |   0.6346  |    1.785  | 0.3555  |    -1.15     |



**student.h.cls.L03 vs student.z.proj.tap2**


| metric                           |   value_h |   value_z |     tau |       delta |
|:---------------------------------|----------:|----------:|--------:|------------:|
| uniformity                       |  -0.06751 |  -3.308   | 0.02041 |    3.24     |
| variance_floor.hinge             |   0.9647  |   0.9588  | 1.006   |    0.005898 |
| offdiag_redundancy.mean_abs_corr |   0.1734  |   0.04284 | 4.048   |    0.1306   |
| rankme                           |  23.99    | 946.3     | 0.02535 | -922.4      |
| effective_rank                   |  30.52    | 120.3     | 0.2538  |  -89.73     |
| alpha                            |   2.192   |   1.36    | 1.612   |    0.8318   |
| epps_pulley                      | 115.2     |  79.16    | 1.456   |   36.06     |
| kurt_topeig.worst                |   0.6346  |   1.015   | 0.6253  |   -0.3802   |



**student.h.cls.L06 vs student.z.embed**


| metric                           |   value_h |   value_z |    tau |       delta |
|:---------------------------------|----------:|----------:|-------:|------------:|
| uniformity                       |   -0.2499 |   -1.512  | 0.1652 |   1.262     |
| variance_floor.hinge             |    0.9358 |    0.491  | 1.906  |   0.4448    |
| offdiag_redundancy.mean_abs_corr |    0.1377 |    0.1381 | 0.9969 |  -0.0004303 |
| rankme                           |   72.02   |   41.19   | 1.748  |  30.83      |
| effective_rank                   |   42.4    |   30.12   | 1.408  |  12.28      |
| alpha                            |    1.742  |    4.728  | 0.3685 |  -2.986     |
| epps_pulley                      |   76.61   |   89.44   | 0.8566 | -12.83      |
| kurt_topeig.worst                |    0.712  |    0.9914 | 0.7181 |  -0.2794    |



**student.h.cls.L06 vs student.z.proj.out**


| metric                           |   value_h |    value_z |        tau |   delta |
|:---------------------------------|----------:|-----------:|-----------:|--------:|
| uniformity                       |   -0.2499 | -3.349     |    0.07463 |  3.099  |
| variance_floor.hinge             |    0.9358 |  0.0008017 | 1167       |  0.935  |
| offdiag_redundancy.mean_abs_corr |    0.1377 |  0.02109   |    6.528   |  0.1166 |
| rankme                           |   72.02   | 15.98      |    4.508   | 56.04   |
| effective_rank                   |   42.4    | 15.9       |    2.666   | 26.49   |
| alpha                            |    1.742  |  0.137     |   12.72    |  1.605  |
| epps_pulley                      |   76.61   | 24.93      |    3.073   | 51.68   |
| kurt_topeig.worst                |    0.712  |  3.543     |    0.201   | -2.831  |



**student.h.cls.L06 vs student.z.proj.tap1**


| metric                           |   value_h |   value_z |     tau |       delta |
|:---------------------------------|----------:|----------:|--------:|------------:|
| uniformity                       |   -0.2499 |   -3.505  | 0.0713  |     3.255   |
| variance_floor.hinge             |    0.9358 |    0.974  | 0.9608  |    -0.03823 |
| offdiag_redundancy.mean_abs_corr |    0.1377 |    0.0598 | 2.302   |     0.07785 |
| rankme                           |   72.02   | 1193      | 0.06039 | -1121       |
| effective_rank                   |   42.4    |  226.6    | 0.1871  |  -184.2     |
| alpha                            |    1.742  |    1.062  | 1.64    |     0.6801  |
| epps_pulley                      |   76.61   |   30.93   | 2.477   |    45.68    |
| kurt_topeig.worst                |    0.712  |    1.785  | 0.3989  |    -1.073   |



**student.h.cls.L06 vs student.z.proj.tap2**


| metric                           |   value_h |   value_z |     tau |      delta |
|:---------------------------------|----------:|----------:|--------:|-----------:|
| uniformity                       |   -0.2499 |  -3.308   | 0.07554 |    3.058   |
| variance_floor.hinge             |    0.9358 |   0.9588  | 0.976   |   -0.023   |
| offdiag_redundancy.mean_abs_corr |    0.1377 |   0.04284 | 3.214   |    0.09482 |
| rankme                           |   72.02   | 946.3     | 0.0761  | -874.3     |
| effective_rank                   |   42.4    | 120.3     | 0.3526  |  -77.85    |
| alpha                            |    1.742  |   1.36    | 1.281   |    0.3823  |
| epps_pulley                      |   76.61   |  79.16    | 0.9678  |   -2.548   |
| kurt_topeig.worst                |    0.712  |   1.015   | 0.7016  |   -0.3028  |



**student.h.cls.L09 vs student.z.embed**


| metric                           |   value_h |   value_z |    tau |     delta |
|:---------------------------------|----------:|----------:|-------:|----------:|
| uniformity                       |   -0.7693 |   -1.512  | 0.5087 |  0.7431   |
| variance_floor.hinge             |    0.904  |    0.491  | 1.841  |  0.4131   |
| offdiag_redundancy.mean_abs_corr |    0.1368 |    0.1381 | 0.9907 | -0.001284 |
| rankme                           |   99.52   |   41.19   | 2.416  | 58.33     |
| effective_rank                   |   41.16   |   30.12   | 1.366  | 11.04     |
| alpha                            |    1.912  |    4.728  | 0.4044 | -2.816    |
| epps_pulley                      |   80.04   |   89.44   | 0.8949 | -9.396    |
| kurt_topeig.worst                |    0.8506 |    0.9914 | 0.8579 | -0.1409   |



**student.h.cls.L09 vs student.z.proj.out**


| metric                           |   value_h |    value_z |       tau |   delta |
|:---------------------------------|----------:|-----------:|----------:|--------:|
| uniformity                       |   -0.7693 | -3.349     |    0.2297 |  2.579  |
| variance_floor.hinge             |    0.904  |  0.0008017 | 1128      |  0.9032 |
| offdiag_redundancy.mean_abs_corr |    0.1368 |  0.02109   |    6.488  |  0.1157 |
| rankme                           |   99.52   | 15.98      |    6.229  | 83.54   |
| effective_rank                   |   41.16   | 15.9       |    2.588  | 25.25   |
| alpha                            |    1.912  |  0.137     |   13.96   |  1.775  |
| epps_pulley                      |   80.04   | 24.93      |    3.211  | 55.11   |
| kurt_topeig.worst                |    0.8506 |  3.543     |    0.2401 | -2.692  |



**student.h.cls.L09 vs student.z.proj.tap1**


| metric                           |   value_h |   value_z |     tau |       delta |
|:---------------------------------|----------:|----------:|--------:|------------:|
| uniformity                       |   -0.7693 |   -3.505  | 0.2195  |     2.735   |
| variance_floor.hinge             |    0.904  |    0.974  | 0.9281  |    -0.06999 |
| offdiag_redundancy.mean_abs_corr |    0.1368 |    0.0598 | 2.288   |     0.077   |
| rankme                           |   99.52   | 1193      | 0.08345 | -1093       |
| effective_rank                   |   41.16   |  226.6    | 0.1817  |  -185.4     |
| alpha                            |    1.912  |    1.062  | 1.8     |     0.85    |
| epps_pulley                      |   80.04   |   30.93   | 2.588   |    49.11    |
| kurt_topeig.worst                |    0.8506 |    1.785  | 0.4765  |    -0.9344  |



**student.h.cls.L09 vs student.z.proj.tap2**


| metric                           |   value_h |   value_z |    tau |      delta |
|:---------------------------------|----------:|----------:|-------:|-----------:|
| uniformity                       |   -0.7693 |  -3.308   | 0.2326 |    2.539   |
| variance_floor.hinge             |    0.904  |   0.9588  | 0.9429 |   -0.05476 |
| offdiag_redundancy.mean_abs_corr |    0.1368 |   0.04284 | 3.194  |    0.09397 |
| rankme                           |   99.52   | 946.3     | 0.1052 | -846.8     |
| effective_rank                   |   41.16   | 120.3     | 0.3422 |  -79.1     |
| alpha                            |    1.912  |   1.36    | 1.406  |    0.5522  |
| epps_pulley                      |   80.04   |  79.16    | 1.011  |    0.8831  |
| kurt_topeig.worst                |    0.8506 |   1.015   | 0.8382 |   -0.1642  |



**student.h.cls.L12 vs student.z.embed**


| metric                           |   value_h |   value_z |    tau |    delta |
|:---------------------------------|----------:|----------:|-------:|---------:|
| uniformity                       |   -1.102  |   -1.512  | 0.7284 |  0.4107  |
| variance_floor.hinge             |    0.8948 |    0.491  | 1.822  |  0.4038  |
| offdiag_redundancy.mean_abs_corr |    0.1515 |    0.1381 | 1.097  |  0.01338 |
| rankme                           |   92.86   |   41.19   | 2.254  | 51.67    |
| effective_rank                   |   35.99   |   30.12   | 1.195  |  5.876   |
| alpha                            |    2.129  |    4.728  | 0.4502 | -2.599   |
| epps_pulley                      |  107.8    |   89.44   | 1.206  | 18.41    |
| kurt_topeig.worst                |    0.8267 |    0.9914 | 0.8338 | -0.1648  |



**student.h.cls.L12 vs student.z.proj.out**


| metric                           |   value_h |    value_z |       tau |   delta |
|:---------------------------------|----------:|-----------:|----------:|--------:|
| uniformity                       |   -1.102  | -3.349     |    0.329  |  2.247  |
| variance_floor.hinge             |    0.8948 |  0.0008017 | 1116      |  0.894  |
| offdiag_redundancy.mean_abs_corr |    0.1515 |  0.02109   |    7.183  |  0.1304 |
| rankme                           |   92.86   | 15.98      |    5.812  | 76.88   |
| effective_rank                   |   35.99   | 15.9       |    2.263  | 20.09   |
| alpha                            |    2.129  |  0.137     |   15.54   |  1.992  |
| epps_pulley                      |  107.8    | 24.93      |    4.326  | 82.91   |
| kurt_topeig.worst                |    0.8267 |  3.543     |    0.2333 | -2.716  |



**student.h.cls.L12 vs student.z.proj.tap1**


| metric                           |   value_h |   value_z |     tau |       delta |
|:---------------------------------|----------:|----------:|--------:|------------:|
| uniformity                       |   -1.102  |   -3.505  | 0.3143  |     2.403   |
| variance_floor.hinge             |    0.8948 |    0.974  | 0.9186  |    -0.07927 |
| offdiag_redundancy.mean_abs_corr |    0.1515 |    0.0598 | 2.533   |     0.09166 |
| rankme                           |   92.86   | 1193      | 0.07786 | -1100       |
| effective_rank                   |   35.99   |  226.6    | 0.1589  |  -190.6     |
| alpha                            |    2.129  |    1.062  | 2.004   |     1.067   |
| epps_pulley                      |  107.8    |   30.93   | 3.487   |    76.91    |
| kurt_topeig.worst                |    0.8267 |    1.785  | 0.4631  |    -0.9583  |



**student.h.cls.L12 vs student.z.proj.tap2**


| metric                           |   value_h |   value_z |     tau |      delta |
|:---------------------------------|----------:|----------:|--------:|-----------:|
| uniformity                       |   -1.102  |  -3.308   | 0.333   |    2.206   |
| variance_floor.hinge             |    0.8948 |   0.9588  | 0.9332  |   -0.06404 |
| offdiag_redundancy.mean_abs_corr |    0.1515 |   0.04284 | 3.536   |    0.1086  |
| rankme                           |   92.86   | 946.3     | 0.09812 | -853.5     |
| effective_rank                   |   35.99   | 120.3     | 0.2993  |  -84.26    |
| alpha                            |    2.129  |   1.36    | 1.565   |    0.7689  |
| epps_pulley                      |  107.8    |  79.16    | 1.362   |   28.69    |
| kurt_topeig.worst                |    0.8267 |   1.015   | 0.8146  |   -0.1881  |



**student.h.gap vs student.z.embed**


| metric                           |   value_h |   value_z |    tau |     delta |
|:---------------------------------|----------:|----------:|-------:|----------:|
| uniformity                       |   -0.4495 |   -1.512  | 0.2972 |   1.063   |
| variance_floor.hinge             |    0.9543 |    0.491  | 1.944  |   0.4633  |
| offdiag_redundancy.mean_abs_corr |    0.124  |    0.1381 | 0.898  |  -0.01409 |
| rankme                           |  103.6    |   41.19   | 2.516  |  62.45    |
| effective_rank                   |   39.83   |   30.12   | 1.323  |   9.714   |
| alpha                            |    1.609  |    4.728  | 0.3402 |  -3.119   |
| epps_pulley                      |   73.77   |   89.44   | 0.8249 | -15.66    |
| kurt_topeig.worst                |    1.556  |    0.9914 | 1.569  |   0.5644  |



**student.h.gap vs student.z.proj.out**


| metric                           |   value_h |    value_z |       tau |   delta |
|:---------------------------------|----------:|-----------:|----------:|--------:|
| uniformity                       |   -0.4495 | -3.349     |    0.1342 |  2.899  |
| variance_floor.hinge             |    0.9543 |  0.0008017 | 1190      |  0.9535 |
| offdiag_redundancy.mean_abs_corr |    0.124  |  0.02109   |    5.88   |  0.1029 |
| rankme                           |  103.6    | 15.98      |    6.487  | 87.66   |
| effective_rank                   |   39.83   | 15.9       |    2.504  | 23.93   |
| alpha                            |    1.609  |  0.137     |   11.74   |  1.472  |
| epps_pulley                      |   73.77   | 24.93      |    2.959  | 48.84   |
| kurt_topeig.worst                |    1.556  |  3.543     |    0.4391 | -1.987  |



**student.h.gap vs student.z.proj.tap1**


| metric                           |   value_h |   value_z |     tau |       delta |
|:---------------------------------|----------:|----------:|--------:|------------:|
| uniformity                       |   -0.4495 |   -3.505  | 0.1282  |     3.055   |
| variance_floor.hinge             |    0.9543 |    0.974  | 0.9797  |    -0.01974 |
| offdiag_redundancy.mean_abs_corr |    0.124  |    0.0598 | 2.073   |     0.06419 |
| rankme                           |  103.6    | 1193      | 0.08691 | -1089       |
| effective_rank                   |   39.83   |  226.6    | 0.1758  |  -186.7     |
| alpha                            |    1.609  |    1.062  | 1.515   |     0.5467  |
| epps_pulley                      |   73.77   |   30.93   | 2.385   |    42.84    |
| kurt_topeig.worst                |    1.556  |    1.785  | 0.8716  |    -0.2291  |



**student.h.gap vs student.z.proj.tap2**


| metric                           |   value_h |   value_z |    tau |       delta |
|:---------------------------------|----------:|----------:|-------:|------------:|
| uniformity                       |   -0.4495 |  -3.308   | 0.1359 |    2.858    |
| variance_floor.hinge             |    0.9543 |   0.9588  | 0.9953 |   -0.004513 |
| offdiag_redundancy.mean_abs_corr |    0.124  |   0.04284 | 2.895  |    0.08116  |
| rankme                           |  103.6    | 946.3     | 0.1095 | -842.7      |
| effective_rank                   |   39.83   | 120.3     | 0.3312 |  -80.42     |
| alpha                            |    1.609  |   1.36    | 1.183  |    0.2489   |
| epps_pulley                      |   73.77   |  79.16    | 0.932  |   -5.385    |
| kurt_topeig.worst                |    1.556  |   1.015   | 1.533  |    0.541    |



**student.h.gap.L03 vs student.z.embed**


| metric                           |   value_h |   value_z |    tau |     delta |
|:---------------------------------|----------:|----------:|-------:|----------:|
| uniformity                       |   -0.3975 |   -1.512  | 0.2628 |   1.115   |
| variance_floor.hinge             |    0.9627 |    0.491  | 1.961  |   0.4718  |
| offdiag_redundancy.mean_abs_corr |    0.1579 |    0.1381 | 1.143  |   0.01977 |
| rankme                           |   65.74   |   41.19   | 1.596  |  24.55    |
| effective_rank                   |   19.45   |   30.12   | 0.6457 | -10.67    |
| alpha                            |    1.929  |    4.728  | 0.4081 |  -2.798   |
| epps_pulley                      |  108.7    |   89.44   | 1.215  |  19.27    |
| kurt_topeig.worst                |    1.42   |    0.9914 | 1.433  |   0.4289  |



**student.h.gap.L03 vs student.z.proj.out**


| metric                           |   value_h |    value_z |       tau |   delta |
|:---------------------------------|----------:|-----------:|----------:|--------:|
| uniformity                       |   -0.3975 | -3.349     |    0.1187 |  2.951  |
| variance_floor.hinge             |    0.9627 |  0.0008017 | 1201      |  0.9619 |
| offdiag_redundancy.mean_abs_corr |    0.1579 |  0.02109   |    7.486  |  0.1368 |
| rankme                           |   65.74   | 15.98      |    4.115  | 49.76   |
| effective_rank                   |   19.45   | 15.9       |    1.223  |  3.544  |
| alpha                            |    1.929  |  0.137     |   14.09   |  1.792  |
| epps_pulley                      |  108.7    | 24.93      |    4.361  | 83.78   |
| kurt_topeig.worst                |    1.42   |  3.543     |    0.4009 | -2.123  |



**student.h.gap.L03 vs student.z.proj.tap1**


| metric                           |   value_h |   value_z |     tau |       delta |
|:---------------------------------|----------:|----------:|--------:|------------:|
| uniformity                       |   -0.3975 |   -3.505  | 0.1134  |     3.107   |
| variance_floor.hinge             |    0.9627 |    0.974  | 0.9884  |    -0.01128 |
| offdiag_redundancy.mean_abs_corr |    0.1579 |    0.0598 | 2.64    |     0.09805 |
| rankme                           |   65.74   | 1193      | 0.05512 | -1127       |
| effective_rank                   |   19.45   |  226.6    | 0.08584 |  -207.1     |
| alpha                            |    1.929  |    1.062  | 1.817   |     0.8675  |
| epps_pulley                      |  108.7    |   30.93   | 3.514   |    77.78    |
| kurt_topeig.worst                |    1.42   |    1.785  | 0.7957  |    -0.3646  |



**student.h.gap.L03 vs student.z.proj.tap2**


| metric                           |   value_h |   value_z |     tau |       delta |
|:---------------------------------|----------:|----------:|--------:|------------:|
| uniformity                       |   -0.3975 |  -3.308   | 0.1202  |    2.91     |
| variance_floor.hinge             |    0.9627 |   0.9588  | 1.004   |    0.003945 |
| offdiag_redundancy.mean_abs_corr |    0.1579 |   0.04284 | 3.685   |    0.115    |
| rankme                           |   65.74   | 946.3     | 0.06946 | -880.6      |
| effective_rank                   |   19.45   | 120.3     | 0.1617  | -100.8      |
| alpha                            |    1.929  |   1.36    | 1.419   |    0.5697   |
| epps_pulley                      |  108.7    |  79.16    | 1.373   |   29.55     |
| kurt_topeig.worst                |    1.42   |   1.015   | 1.4     |    0.4056   |



**student.h.gap.L06 vs student.z.embed**


| metric                           |   value_h |   value_z |    tau |     delta |
|:---------------------------------|----------:|----------:|-------:|----------:|
| uniformity                       |   -0.5797 |   -1.512  | 0.3833 |  0.9327   |
| variance_floor.hinge             |    0.9525 |    0.491  | 1.94   |  0.4615   |
| offdiag_redundancy.mean_abs_corr |    0.1362 |    0.1381 | 0.9863 | -0.001896 |
| rankme                           |  108.1    |   41.19   | 2.623  | 66.86     |
| effective_rank                   |   38.29   |   30.12   | 1.271  |  8.172    |
| alpha                            |    1.65   |    4.728  | 0.3491 | -3.077    |
| epps_pulley                      |   84.52   |   89.44   | 0.9451 | -4.914    |
| kurt_topeig.worst                |    4.726  |    0.9914 | 4.767  |  3.735    |



**student.h.gap.L06 vs student.z.proj.out**


| metric                           |   value_h |    value_z |       tau |   delta |
|:---------------------------------|----------:|-----------:|----------:|--------:|
| uniformity                       |   -0.5797 | -3.349     |    0.1731 |  2.769  |
| variance_floor.hinge             |    0.9525 |  0.0008017 | 1188      |  0.9517 |
| offdiag_redundancy.mean_abs_corr |    0.1362 |  0.02109   |    6.459  |  0.1151 |
| rankme                           |  108.1    | 15.98      |    6.763  | 92.07   |
| effective_rank                   |   38.29   | 15.9       |    2.408  | 22.39   |
| alpha                            |    1.65   |  0.137     |   12.05   |  1.513  |
| epps_pulley                      |   84.52   | 24.93      |    3.39   | 59.59   |
| kurt_topeig.worst                |    4.726  |  3.543     |    1.334  |  1.183  |



**student.h.gap.L06 vs student.z.proj.tap1**


| metric                           |   value_h |   value_z |     tau |       delta |
|:---------------------------------|----------:|----------:|--------:|------------:|
| uniformity                       |   -0.5797 |   -3.505  | 0.1654  |     2.925   |
| variance_floor.hinge             |    0.9525 |    0.974  | 0.9779  |    -0.02153 |
| offdiag_redundancy.mean_abs_corr |    0.1362 |    0.0598 | 2.277   |     0.07639 |
| rankme                           |  108.1    | 1193      | 0.09061 | -1084       |
| effective_rank                   |   38.29   |  226.6    | 0.169   |  -188.3     |
| alpha                            |    1.65   |    1.062  | 1.554   |     0.5885  |
| epps_pulley                      |   84.52   |   30.93   | 2.733   |    53.59    |
| kurt_topeig.worst                |    4.726  |    1.785  | 2.648   |     2.941   |



**student.h.gap.L06 vs student.z.proj.tap2**


| metric                           |   value_h |   value_z |    tau |       delta |
|:---------------------------------|----------:|----------:|-------:|------------:|
| uniformity                       |   -0.5797 |  -3.308   | 0.1753 |    2.728    |
| variance_floor.hinge             |    0.9525 |   0.9588  | 0.9934 |   -0.006301 |
| offdiag_redundancy.mean_abs_corr |    0.1362 |   0.04284 | 3.179  |    0.09335  |
| rankme                           |  108.1    | 946.3     | 0.1142 | -838.3      |
| effective_rank                   |   38.29   | 120.3     | 0.3184 |  -81.96     |
| alpha                            |    1.65   |   1.36    | 1.214  |    0.2907   |
| epps_pulley                      |   84.52   |  79.16    | 1.068  |    5.365    |
| kurt_topeig.worst                |    4.726  |   1.015   | 4.657  |    3.712    |



**student.h.gap.L09 vs student.z.embed**


| metric                           |   value_h |   value_z |    tau |     delta |
|:---------------------------------|----------:|----------:|-------:|----------:|
| uniformity                       |   -0.6065 |   -1.512  | 0.401  |   0.9059  |
| variance_floor.hinge             |    0.9509 |    0.491  | 1.937  |   0.46    |
| offdiag_redundancy.mean_abs_corr |    0.1231 |    0.1381 | 0.8916 |  -0.01497 |
| rankme                           |  120.7    |   41.19   | 2.931  |  79.55    |
| effective_rank                   |   45.08   |   30.12   | 1.497  |  14.96    |
| alpha                            |    1.557  |    4.728  | 0.3294 |  -3.17    |
| epps_pulley                      |   68.85   |   89.44   | 0.7698 | -20.59    |
| kurt_topeig.worst                |    1.357  |    0.9914 | 1.369  |   0.3658  |



**student.h.gap.L09 vs student.z.proj.out**


| metric                           |   value_h |    value_z |       tau |    delta |
|:---------------------------------|----------:|-----------:|----------:|---------:|
| uniformity                       |   -0.6065 | -3.349     |    0.1811 |   2.742  |
| variance_floor.hinge             |    0.9509 |  0.0008017 | 1186      |   0.9501 |
| offdiag_redundancy.mean_abs_corr |    0.1231 |  0.02109   |    5.839  |   0.102  |
| rankme                           |  120.7    | 15.98      |    7.557  | 104.8    |
| effective_rank                   |   45.08   | 15.9       |    2.834  |  29.17   |
| alpha                            |    1.557  |  0.137     |   11.37   |   1.42   |
| epps_pulley                      |   68.85   | 24.93      |    2.762  |  43.92   |
| kurt_topeig.worst                |    1.357  |  3.543     |    0.3831 |  -2.186  |



**student.h.gap.L09 vs student.z.proj.tap1**


| metric                           |   value_h |   value_z |    tau |       delta |
|:---------------------------------|----------:|----------:|-------:|------------:|
| uniformity                       |   -0.6065 |   -3.505  | 0.173  |     2.898   |
| variance_floor.hinge             |    0.9509 |    0.974  | 0.9763 |    -0.02309 |
| offdiag_redundancy.mean_abs_corr |    0.1231 |    0.0598 | 2.059  |     0.06331 |
| rankme                           |  120.7    | 1193      | 0.1012 | -1072       |
| effective_rank                   |   45.08   |  226.6    | 0.199  |  -181.5     |
| alpha                            |    1.557  |    1.062  | 1.467  |     0.4955  |
| epps_pulley                      |   68.85   |   30.93   | 2.226  |    37.91    |
| kurt_topeig.worst                |    1.357  |    1.785  | 0.7604 |    -0.4277  |



**student.h.gap.L09 vs student.z.proj.tap2**


| metric                           |   value_h |   value_z |    tau |       delta |
|:---------------------------------|----------:|----------:|-------:|------------:|
| uniformity                       |   -0.6065 |  -3.308   | 0.1833 |    2.701    |
| variance_floor.hinge             |    0.9509 |   0.9588  | 0.9918 |   -0.007858 |
| offdiag_redundancy.mean_abs_corr |    0.1231 |   0.04284 | 2.874  |    0.08028  |
| rankme                           |  120.7    | 946.3     | 0.1276 | -825.6      |
| effective_rank                   |   45.08   | 120.3     | 0.3749 |  -75.17     |
| alpha                            |    1.557  |   1.36    | 1.145  |    0.1978   |
| epps_pulley                      |   68.85   |  79.16    | 0.8697 |  -10.31     |
| kurt_topeig.worst                |    1.357  |   1.015   | 1.337  |    0.3424   |



**student.h.gap.L12 vs student.z.embed**


| metric                           |   value_h |   value_z |    tau |     delta |
|:---------------------------------|----------:|----------:|-------:|----------:|
| uniformity                       |   -0.4495 |   -1.512  | 0.2972 |   1.063   |
| variance_floor.hinge             |    0.9543 |    0.491  | 1.944  |   0.4633  |
| offdiag_redundancy.mean_abs_corr |    0.124  |    0.1381 | 0.898  |  -0.01409 |
| rankme                           |  103.6    |   41.19   | 2.516  |  62.45    |
| effective_rank                   |   39.83   |   30.12   | 1.323  |   9.714   |
| alpha                            |    1.609  |    4.728  | 0.3402 |  -3.119   |
| epps_pulley                      |   73.77   |   89.44   | 0.8249 | -15.66    |
| kurt_topeig.worst                |    1.556  |    0.9914 | 1.569  |   0.5644  |



**student.h.gap.L12 vs student.z.proj.out**


| metric                           |   value_h |    value_z |       tau |   delta |
|:---------------------------------|----------:|-----------:|----------:|--------:|
| uniformity                       |   -0.4495 | -3.349     |    0.1342 |  2.899  |
| variance_floor.hinge             |    0.9543 |  0.0008017 | 1190      |  0.9535 |
| offdiag_redundancy.mean_abs_corr |    0.124  |  0.02109   |    5.88   |  0.1029 |
| rankme                           |  103.6    | 15.98      |    6.487  | 87.66   |
| effective_rank                   |   39.83   | 15.9       |    2.504  | 23.93   |
| alpha                            |    1.609  |  0.137     |   11.74   |  1.472  |
| epps_pulley                      |   73.77   | 24.93      |    2.959  | 48.84   |
| kurt_topeig.worst                |    1.556  |  3.543     |    0.4391 | -1.987  |



**student.h.gap.L12 vs student.z.proj.tap1**


| metric                           |   value_h |   value_z |     tau |       delta |
|:---------------------------------|----------:|----------:|--------:|------------:|
| uniformity                       |   -0.4495 |   -3.505  | 0.1282  |     3.055   |
| variance_floor.hinge             |    0.9543 |    0.974  | 0.9797  |    -0.01974 |
| offdiag_redundancy.mean_abs_corr |    0.124  |    0.0598 | 2.073   |     0.06419 |
| rankme                           |  103.6    | 1193      | 0.08691 | -1089       |
| effective_rank                   |   39.83   |  226.6    | 0.1758  |  -186.7     |
| alpha                            |    1.609  |    1.062  | 1.515   |     0.5467  |
| epps_pulley                      |   73.77   |   30.93   | 2.385   |    42.84    |
| kurt_topeig.worst                |    1.556  |    1.785  | 0.8716  |    -0.2291  |



**student.h.gap.L12 vs student.z.proj.tap2**


| metric                           |   value_h |   value_z |    tau |       delta |
|:---------------------------------|----------:|----------:|-------:|------------:|
| uniformity                       |   -0.4495 |  -3.308   | 0.1359 |    2.858    |
| variance_floor.hinge             |    0.9543 |   0.9588  | 0.9953 |   -0.004513 |
| offdiag_redundancy.mean_abs_corr |    0.124  |   0.04284 | 2.895  |    0.08116  |
| rankme                           |  103.6    | 946.3     | 0.1095 | -842.7      |
| effective_rank                   |   39.83   | 120.3     | 0.3312 |  -80.42     |
| alpha                            |    1.609  |   1.36    | 1.183  |    0.2489   |
| epps_pulley                      |   73.77   |  79.16    | 0.932  |   -5.385    |
| kurt_topeig.worst                |    1.556  |   1.015   | 1.533  |    0.541    |



### Pair metrics (alignment ↓ = more view-invariant)


| manifest                     | space               | metric         |   value |   ci_lo |   ci_hi |
|:-----------------------------|:--------------------|:---------------|--------:|--------:|--------:|
| imagenette.train.v1@audit_v1 | student.h.cls       | alignment      |  0.1308 |  0.129  |  0.1329 |
| imagenette.train.v1@audit_v1 | student.h.cls       | cos_invariance |  0.9346 |  0.9336 |  0.9355 |
| imagenette.train.v1@audit_v1 | student.h.gap       | alignment      |  0.1143 |  0.1126 |  0.1159 |
| imagenette.train.v1@audit_v1 | student.h.gap       | cos_invariance |  0.9428 |  0.9421 |  0.9436 |
| imagenette.train.v1@audit_v1 | student.z.embed     | alignment      |  0.1538 |  0.1514 |  0.1564 |
| imagenette.train.v1@audit_v1 | student.z.embed     | cos_invariance |  0.9231 |  0.9218 |  0.9241 |
| imagenette.train.v1@audit_v1 | student.z.proj.out  | alignment      |  0.1642 |  0.1591 |  0.17   |
| imagenette.train.v1@audit_v1 | student.z.proj.out  | cos_invariance |  0.9179 |  0.9151 |  0.9205 |
| imagenette.train.v1@audit_v1 | student.z.proj.tap1 | alignment      |  0.4921 |  0.484  |  0.4985 |
| imagenette.train.v1@audit_v1 | student.z.proj.tap1 | cos_invariance |  0.754  |  0.7504 |  0.7576 |
| imagenette.train.v1@audit_v1 | student.z.proj.tap2 | alignment      |  0.3113 |  0.3043 |  0.3184 |
| imagenette.train.v1@audit_v1 | student.z.proj.tap2 | cos_invariance |  0.8444 |  0.8409 |  0.8474 |



### Cross-space similarity (CKA is contested — read as the triple)


| space                             | metric              |   value |
|:----------------------------------|:--------------------|--------:|
| student.h.cls~student.z.embed     | cka_linear          |  0.9752 |
| student.h.cls~student.z.embed     | neighbor_jaccard    |  0.7588 |
| student.h.cls~student.z.embed     | procrustes_distance |  0.1548 |
| student.h.cls~student.z.embed     | knn_label_agreement |  0.9826 |
| student.h.cls~student.z.proj.out  | cka_linear          |  0.674  |
| student.h.cls~student.z.proj.out  | neighbor_jaccard    |  0.3427 |
| student.h.cls~student.z.proj.out  | procrustes_distance |  0.5812 |
| student.h.cls~student.z.proj.out  | knn_label_agreement |  0.9598 |
| student.h.cls~student.z.proj.tap1 | cka_linear          |  0.8647 |
| student.h.cls~student.z.proj.tap1 | neighbor_jaccard    |  0.594  |
| student.h.cls~student.z.proj.tap1 | procrustes_distance |  0.5212 |
| student.h.cls~student.z.proj.tap1 | knn_label_agreement |  0.9748 |
| student.h.cls~student.z.proj.tap2 | cka_linear          |  0.7925 |
| student.h.cls~student.z.proj.tap2 | neighbor_jaccard    |  0.4317 |
| student.h.cls~student.z.proj.tap2 | procrustes_distance |  0.6269 |
| student.h.cls~student.z.proj.tap2 | knn_label_agreement |  0.9666 |



### Probes


| space               |   knn_v1_k20 |   knn_v1_k200 |   linear_house_v1 |   linear_l2_v1 |
|:--------------------|-------------:|--------------:|------------------:|---------------:|
| student.h.cls       |       0.8966 |        0.8876 |            0.9118 |         0.8920 |
| student.h.cls.L03   |       0.6415 |        0.5855 |            0.6996 |         0.5546 |
| student.h.cls.L06   |       0.7628 |        0.6869 |            0.8161 |         0.7141 |
| student.h.cls.L09   |       0.8685 |        0.8451 |            0.8907 |         0.8540 |
| student.h.cls.L12   |       0.8966 |        0.8876 |            0.9121 |         0.8920 |
| student.h.gap       |       0.8048 |        0.7513 |            0.8629 |         0.7577 |
| student.h.gap.L03   |       0.6548 |        0.5676 |            0.7373 |         0.5763 |
| student.h.gap.L06   |       0.7544 |        0.6932 |            0.8321 |         0.7108 |
| student.h.gap.L09   |       0.7997 |        0.7404 |            0.8571 |         0.7480 |
| student.h.gap.L12   |       0.8048 |        0.7513 |            0.8629 |         0.7577 |
| student.z.embed     |       0.8981 |        0.8899 |            0.9096 |         0.8932 |
| student.z.proj.out  |       0.8978 |        0.8945 |            0.8897 |         0.8764 |
| student.z.proj.tap1 |       0.9080 |        0.9052 |            0.9139 |         0.9083 |
| student.z.proj.tap2 |       0.9017 |        0.9004 |            0.9121 |         0.9027 |



## toy.lejepa-lamb0.ext

### Battery (variant raw|full)

| model.space                                |   rankme |   effective_rank |   alpha |   epps_pulley |   kurt_topeig.worst |   uniformity |   offdiag_redundancy.mean_abs_corr |   variance_floor.hinge |
|:-------------------------------------------|---------:|-----------------:|--------:|--------------:|--------------------:|-------------:|-----------------------------------:|-----------------------:|
| toy.lejepa-lamb0.ext · student.h.cls       |    1.228 |            6.559 |   1.774 |         421.9 |              3.377  |   -5.442e-05 |                            0.4129  |                 0.9975 |
| toy.lejepa-lamb0.ext · student.h.cls.L03   |    1.53  |            4.171 |   2.04  |         537.2 |              2.823  |   -0.0009289 |                            0.4598  |                 0.99   |
| toy.lejepa-lamb0.ext · student.h.cls.L06   |    1.362 |            4.446 |   1.968 |         503.2 |              3.133  |   -0.0003327 |                            0.4484  |                 0.994  |
| toy.lejepa-lamb0.ext · student.h.cls.L09   |    1.256 |            6.072 |   1.835 |         432   |              3.039  |   -7.745e-05 |                            0.4212  |                 0.9971 |
| toy.lejepa-lamb0.ext · student.h.cls.L12   |    1.228 |            6.559 |   1.774 |         421.9 |              3.377  |   -5.442e-05 |                            0.4129  |                 0.9975 |
| toy.lejepa-lamb0.ext · student.h.gap       |    1.314 |            2.258 |   2.295 |         840.6 |             12.96   |   -0.000731  |                            0.5526  |                 0.9918 |
| toy.lejepa-lamb0.ext · student.h.gap.L03   |    1.994 |            2.096 |   3.239 |         837.4 |            572.3    |   -0.01547   |                            0.5684  |                 0.9627 |
| toy.lejepa-lamb0.ext · student.h.gap.L06   |    1.611 |            2.127 |   2.892 |         840.3 |             10.29   |   -0.005126  |                            0.5611  |                 0.9784 |
| toy.lejepa-lamb0.ext · student.h.gap.L09   |    1.355 |            2.285 |   2.403 |         845.3 |             12.32   |   -0.001022  |                            0.5469  |                 0.9902 |
| toy.lejepa-lamb0.ext · student.h.gap.L12   |    1.314 |            2.258 |   2.295 |         840.6 |             12.96   |   -0.000731  |                            0.5526  |                 0.9918 |
| toy.lejepa-lamb0.ext · student.z.embed     |    1.236 |           36.19  |   1.146 |         346.8 |              5.216  |   -9.178e-06 |                            0.1945  |                 0.9975 |
| toy.lejepa-lamb0.ext · student.z.proj.out  |    1.126 |            7.051 |   1.146 |         234.8 |              0.7709 |   -8.993e-05 |                            0.3302  |                 0.9997 |
| toy.lejepa-lamb0.ext · student.z.proj.tap1 |  134.8   |           38.31  |   1.09  |        1366   |              3.685  |   -0.05794   |                            0.06563 |                 0.997  |
| toy.lejepa-lamb0.ext · student.z.proj.tap2 |    2.552 |            1.231 |   1.211 |        1266   |              5.807  |   -0.0004922 |                            0.7942  |                 0.9966 |



### Transfer ratios τ = value(h) / value(z)


**student.h.cls vs student.z.embed**


| metric                           |     value_h |     value_z |    tau |       delta |
|:---------------------------------|------------:|------------:|-------:|------------:|
| uniformity                       |  -5.442e-05 |  -9.178e-06 | 5.929  |  -4.524e-05 |
| variance_floor.hinge             |   0.9975    |   0.9975    | 1      |  -7.591e-06 |
| offdiag_redundancy.mean_abs_corr |   0.4129    |   0.1945    | 2.123  |   0.2184    |
| rankme                           |   1.228     |   1.236     | 0.993  |  -0.008685  |
| effective_rank                   |   6.559     |  36.19      | 0.1812 | -29.63      |
| alpha                            |   1.774     |   1.146     | 1.548  |   0.628     |
| epps_pulley                      | 421.9       | 346.8       | 1.217  |  75.15      |
| kurt_topeig.worst                |   3.377     |   5.216     | 0.6473 |  -1.84      |



**student.h.cls vs student.z.proj.out**


| metric                           |     value_h |     value_z |    tau |       delta |
|:---------------------------------|------------:|------------:|-------:|------------:|
| uniformity                       |  -5.442e-05 |  -8.993e-05 | 0.6051 |   3.551e-05 |
| variance_floor.hinge             |   0.9975    |   0.9997    | 0.9979 |  -0.002146  |
| offdiag_redundancy.mean_abs_corr |   0.4129    |   0.3302    | 1.251  |   0.08279   |
| rankme                           |   1.228     |   1.126     | 1.09   |   0.1016    |
| effective_rank                   |   6.559     |   7.051     | 0.9302 |  -0.4922    |
| alpha                            |   1.774     |   1.146     | 1.548  |   0.6284    |
| epps_pulley                      | 421.9       | 234.8       | 1.797  | 187.1       |
| kurt_topeig.worst                |   3.377     |   0.7709    | 4.38   |   2.606     |



**student.h.cls vs student.z.proj.tap1**


| metric                           |     value_h |    value_z |       tau |        delta |
|:---------------------------------|------------:|-----------:|----------:|-------------:|
| uniformity                       |  -5.442e-05 |   -0.05794 | 0.0009393 |    0.05789   |
| variance_floor.hinge             |   0.9975    |    0.997   | 1.001     |    0.0005662 |
| offdiag_redundancy.mean_abs_corr |   0.4129    |    0.06563 | 6.292     |    0.3473    |
| rankme                           |   1.228     |  134.8     | 0.009107  | -133.6       |
| effective_rank                   |   6.559     |   38.31    | 0.1712    |  -31.75      |
| alpha                            |   1.774     |    1.09    | 1.628     |    0.6841    |
| epps_pulley                      | 421.9       | 1366       | 0.3089    | -944         |
| kurt_topeig.worst                |   3.377     |    3.685   | 0.9163    |   -0.3085    |



**student.h.cls vs student.z.proj.tap2**


| metric                           |     value_h |      value_z |    tau |        delta |
|:---------------------------------|------------:|-------------:|-------:|-------------:|
| uniformity                       |  -5.442e-05 |   -0.0004922 | 0.1106 |    0.0004378 |
| variance_floor.hinge             |   0.9975    |    0.9966    | 1.001  |    0.0009244 |
| offdiag_redundancy.mean_abs_corr |   0.4129    |    0.7942    | 0.5199 |   -0.3813    |
| rankme                           |   1.228     |    2.552     | 0.481  |   -1.325     |
| effective_rank                   |   6.559     |    1.231     | 5.33   |    5.328     |
| alpha                            |   1.774     |    1.211     | 1.465  |    0.5632    |
| epps_pulley                      | 421.9       | 1266         | 0.3333 | -844.1       |
| kurt_topeig.worst                |   3.377     |    5.807     | 0.5815 |   -2.43      |



**student.h.cls.L03 vs student.z.embed**


| metric                           |     value_h |     value_z |      tau |       delta |
|:---------------------------------|------------:|------------:|---------:|------------:|
| uniformity                       |  -0.0009289 |  -9.178e-06 | 101.2    |  -0.0009198 |
| variance_floor.hinge             |   0.99      |   0.9975    |   0.9924 |  -0.007552  |
| offdiag_redundancy.mean_abs_corr |   0.4598    |   0.1945    |   2.364  |   0.2653    |
| rankme                           |   1.53      |   1.236     |   1.237  |   0.2932    |
| effective_rank                   |   4.171     |  36.19      |   0.1153 | -32.02      |
| alpha                            |   2.04      |   1.146     |   1.78   |   0.8936    |
| epps_pulley                      | 537.2       | 346.8       |   1.549  | 190.4       |
| kurt_topeig.worst                |   2.823     |   5.216     |   0.5412 |  -2.393     |



**student.h.cls.L03 vs student.z.proj.out**


| metric                           |     value_h |     value_z |     tau |      delta |
|:---------------------------------|------------:|------------:|--------:|-----------:|
| uniformity                       |  -0.0009289 |  -8.993e-05 | 10.33   |  -0.000839 |
| variance_floor.hinge             |   0.99      |   0.9997    |  0.9903 |  -0.00969  |
| offdiag_redundancy.mean_abs_corr |   0.4598    |   0.3302    |  1.393  |   0.1297   |
| rankme                           |   1.53      |   1.126     |  1.358  |   0.4034   |
| effective_rank                   |   4.171     |   7.051     |  0.5916 |  -2.88     |
| alpha                            |   2.04      |   1.146     |  1.78   |   0.8939   |
| epps_pulley                      | 537.2       | 234.8       |  2.288  | 302.4      |
| kurt_topeig.worst                |   2.823     |   0.7709    |  3.662  |   2.052    |



**student.h.cls.L03 vs student.z.proj.tap1**


| metric                           |     value_h |    value_z |     tau |       delta |
|:---------------------------------|------------:|-----------:|--------:|------------:|
| uniformity                       |  -0.0009289 |   -0.05794 | 0.01603 |    0.05701  |
| variance_floor.hinge             |   0.99      |    0.997   | 0.993   |   -0.006979 |
| offdiag_redundancy.mean_abs_corr |   0.4598    |    0.06563 | 7.007   |    0.3942   |
| rankme                           |   1.53      |  134.8     | 0.01135 | -133.3      |
| effective_rank                   |   4.171     |   38.31    | 0.1089  |  -34.14     |
| alpha                            |   2.04      |    1.09    | 1.871   |    0.9497   |
| epps_pulley                      | 537.2       | 1366       | 0.3933  | -828.8      |
| kurt_topeig.worst                |   2.823     |    3.685   | 0.7661  |   -0.8619   |



**student.h.cls.L03 vs student.z.proj.tap2**


| metric                           |     value_h |      value_z |    tau |        delta |
|:---------------------------------|------------:|-------------:|-------:|-------------:|
| uniformity                       |  -0.0009289 |   -0.0004922 | 1.887  |   -0.0004367 |
| variance_floor.hinge             |   0.99      |    0.9966    | 0.9934 |   -0.00662   |
| offdiag_redundancy.mean_abs_corr |   0.4598    |    0.7942    | 0.579  |   -0.3344    |
| rankme                           |   1.53      |    2.552     | 0.5993 |   -1.023     |
| effective_rank                   |   4.171     |    1.231     | 3.39   |    2.941     |
| alpha                            |   2.04      |    1.211     | 1.684  |    0.8287    |
| epps_pulley                      | 537.2       | 1266         | 0.4243 | -728.8       |
| kurt_topeig.worst                |   2.823     |    5.807     | 0.4862 |   -2.984     |



**student.h.cls.L06 vs student.z.embed**


| metric                           |     value_h |     value_z |     tau |       delta |
|:---------------------------------|------------:|------------:|--------:|------------:|
| uniformity                       |  -0.0003327 |  -9.178e-06 | 36.24   |  -0.0003235 |
| variance_floor.hinge             |   0.994     |   0.9975    |  0.9964 |  -0.003545  |
| offdiag_redundancy.mean_abs_corr |   0.4484    |   0.1945    |  2.305  |   0.2539    |
| rankme                           |   1.362     |   1.236     |  1.101  |   0.1252    |
| effective_rank                   |   4.446     |  36.19      |  0.1229 | -31.74      |
| alpha                            |   1.968     |   1.146     |  1.717  |   0.8221    |
| epps_pulley                      | 503.2       | 346.8       |  1.451  | 156.4       |
| kurt_topeig.worst                |   3.133     |   5.216     |  0.6005 |  -2.084     |



**student.h.cls.L06 vs student.z.proj.out**


| metric                           |     value_h |     value_z |    tau |       delta |
|:---------------------------------|------------:|------------:|-------:|------------:|
| uniformity                       |  -0.0003327 |  -8.993e-05 | 3.699  |  -0.0002427 |
| variance_floor.hinge             |   0.994     |   0.9997    | 0.9943 |  -0.005683  |
| offdiag_redundancy.mean_abs_corr |   0.4484    |   0.3302    | 1.358  |   0.1182    |
| rankme                           |   1.362     |   1.126     | 1.209  |   0.2355    |
| effective_rank                   |   4.446     |   7.051     | 0.6306 |  -2.605     |
| alpha                            |   1.968     |   1.146     | 1.718  |   0.8224    |
| epps_pulley                      | 503.2       | 234.8       | 2.143  | 268.3       |
| kurt_topeig.worst                |   3.133     |   0.7709    | 4.063  |   2.362     |



**student.h.cls.L06 vs student.z.proj.tap1**


| metric                           |     value_h |    value_z |      tau |       delta |
|:---------------------------------|------------:|-----------:|---------:|------------:|
| uniformity                       |  -0.0003327 |   -0.05794 | 0.005742 |    0.05761  |
| variance_floor.hinge             |   0.994     |    0.997   | 0.997    |   -0.002971 |
| offdiag_redundancy.mean_abs_corr |   0.4484    |    0.06563 | 6.832    |    0.3828   |
| rankme                           |   1.362     |  134.8     | 0.0101   | -133.4      |
| effective_rank                   |   4.446     |   38.31    | 0.1161   |  -33.87     |
| alpha                            |   1.968     |    1.09    | 1.806    |    0.8781   |
| epps_pulley                      | 503.2       | 1366       | 0.3683   | -862.8      |
| kurt_topeig.worst                |   3.133     |    3.685   | 0.8501   |   -0.5526   |



**student.h.cls.L06 vs student.z.proj.tap2**


| metric                           |     value_h |      value_z |    tau |        delta |
|:---------------------------------|------------:|-------------:|-------:|-------------:|
| uniformity                       |  -0.0003327 |   -0.0004922 | 0.6758 |    0.0001596 |
| variance_floor.hinge             |   0.994     |    0.9966    | 0.9974 |   -0.002613  |
| offdiag_redundancy.mean_abs_corr |   0.4484    |    0.7942    | 0.5645 |   -0.3459    |
| rankme                           |   1.362     |    2.552     | 0.5335 |   -1.191     |
| effective_rank                   |   4.446     |    1.231     | 3.613  |    3.216     |
| alpha                            |   1.968     |    1.211     | 1.625  |    0.7572    |
| epps_pulley                      | 503.2       | 1266         | 0.3974 | -762.9       |
| kurt_topeig.worst                |   3.133     |    5.807     | 0.5394 |   -2.674     |



**student.h.cls.L09 vs student.z.embed**


| metric                           |     value_h |     value_z |    tau |       delta |
|:---------------------------------|------------:|------------:|-------:|------------:|
| uniformity                       |  -7.745e-05 |  -9.178e-06 | 8.438  |  -6.827e-05 |
| variance_floor.hinge             |   0.9971    |   0.9975    | 0.9995 |  -0.0004771 |
| offdiag_redundancy.mean_abs_corr |   0.4212    |   0.1945    | 2.165  |   0.2266    |
| rankme                           |   1.256     |   1.236     | 1.016  |   0.01943   |
| effective_rank                   |   6.072     |  36.19      | 0.1678 | -30.12      |
| alpha                            |   1.835     |   1.146     | 1.601  |   0.6889    |
| epps_pulley                      | 432         | 346.8       | 1.246  |  85.16      |
| kurt_topeig.worst                |   3.039     |   5.216     | 0.5826 |  -2.178     |



**student.h.cls.L09 vs student.z.proj.out**


| metric                           |     value_h |     value_z |    tau |       delta |
|:---------------------------------|------------:|------------:|-------:|------------:|
| uniformity                       |  -7.745e-05 |  -8.993e-05 | 0.8612 |   1.248e-05 |
| variance_floor.hinge             |   0.9971    |   0.9997    | 0.9974 |  -0.002615  |
| offdiag_redundancy.mean_abs_corr |   0.4212    |   0.3302    | 1.276  |   0.091     |
| rankme                           |   1.256     |   1.126     | 1.115  |   0.1297    |
| effective_rank                   |   6.072     |   7.051     | 0.8612 |  -0.9786    |
| alpha                            |   1.835     |   1.146     | 1.602  |   0.6892    |
| epps_pulley                      | 432         | 234.8       | 1.839  | 197.1       |
| kurt_topeig.worst                |   3.039     |   0.7709    | 3.942  |   2.268     |



**student.h.cls.L09 vs student.z.proj.tap1**


| metric                           |     value_h |    value_z |      tau |        delta |
|:---------------------------------|------------:|-----------:|---------:|-------------:|
| uniformity                       |  -7.745e-05 |   -0.05794 | 0.001337 |    0.05786   |
| variance_floor.hinge             |   0.9971    |    0.997   | 1        |    9.671e-05 |
| offdiag_redundancy.mean_abs_corr |   0.4212    |    0.06563 | 6.417    |    0.3555    |
| rankme                           |   1.256     |  134.8     | 0.009316 | -133.6       |
| effective_rank                   |   6.072     |   38.31    | 0.1585   |  -32.24      |
| alpha                            |   1.835     |    1.09    | 1.683    |    0.745     |
| epps_pulley                      | 432         | 1366       | 0.3162   | -934         |
| kurt_topeig.worst                |   3.039     |    3.685   | 0.8246   |   -0.6463    |



**student.h.cls.L09 vs student.z.proj.tap2**


| metric                           |     value_h |      value_z |    tau |        delta |
|:---------------------------------|------------:|-------------:|-------:|-------------:|
| uniformity                       |  -7.745e-05 |   -0.0004922 | 0.1573 |    0.0004148 |
| variance_floor.hinge             |   0.9971    |    0.9966    | 1      |    0.0004549 |
| offdiag_redundancy.mean_abs_corr |   0.4212    |    0.7942    | 0.5303 |   -0.3731    |
| rankme                           |   1.256     |    2.552     | 0.492  |   -1.297     |
| effective_rank                   |   6.072     |    1.231     | 4.935  |    4.842     |
| alpha                            |   1.835     |    1.211     | 1.515  |    0.6241    |
| epps_pulley                      | 432         | 1266         | 0.3412 | -834.1       |
| kurt_topeig.worst                |   3.039     |    5.807     | 0.5233 |   -2.768     |



**student.h.cls.L12 vs student.z.embed**


| metric                           |     value_h |     value_z |    tau |       delta |
|:---------------------------------|------------:|------------:|-------:|------------:|
| uniformity                       |  -5.442e-05 |  -9.178e-06 | 5.929  |  -4.524e-05 |
| variance_floor.hinge             |   0.9975    |   0.9975    | 1      |  -7.591e-06 |
| offdiag_redundancy.mean_abs_corr |   0.4129    |   0.1945    | 2.123  |   0.2184    |
| rankme                           |   1.228     |   1.236     | 0.993  |  -0.008685  |
| effective_rank                   |   6.559     |  36.19      | 0.1812 | -29.63      |
| alpha                            |   1.774     |   1.146     | 1.548  |   0.628     |
| epps_pulley                      | 421.9       | 346.8       | 1.217  |  75.15      |
| kurt_topeig.worst                |   3.377     |   5.216     | 0.6473 |  -1.84      |



**student.h.cls.L12 vs student.z.proj.out**


| metric                           |     value_h |     value_z |    tau |       delta |
|:---------------------------------|------------:|------------:|-------:|------------:|
| uniformity                       |  -5.442e-05 |  -8.993e-05 | 0.6051 |   3.551e-05 |
| variance_floor.hinge             |   0.9975    |   0.9997    | 0.9979 |  -0.002146  |
| offdiag_redundancy.mean_abs_corr |   0.4129    |   0.3302    | 1.251  |   0.08279   |
| rankme                           |   1.228     |   1.126     | 1.09   |   0.1016    |
| effective_rank                   |   6.559     |   7.051     | 0.9302 |  -0.4922    |
| alpha                            |   1.774     |   1.146     | 1.548  |   0.6284    |
| epps_pulley                      | 421.9       | 234.8       | 1.797  | 187.1       |
| kurt_topeig.worst                |   3.377     |   0.7709    | 4.38   |   2.606     |



**student.h.cls.L12 vs student.z.proj.tap1**


| metric                           |     value_h |    value_z |       tau |        delta |
|:---------------------------------|------------:|-----------:|----------:|-------------:|
| uniformity                       |  -5.442e-05 |   -0.05794 | 0.0009393 |    0.05789   |
| variance_floor.hinge             |   0.9975    |    0.997   | 1.001     |    0.0005662 |
| offdiag_redundancy.mean_abs_corr |   0.4129    |    0.06563 | 6.292     |    0.3473    |
| rankme                           |   1.228     |  134.8     | 0.009107  | -133.6       |
| effective_rank                   |   6.559     |   38.31    | 0.1712    |  -31.75      |
| alpha                            |   1.774     |    1.09    | 1.628     |    0.6841    |
| epps_pulley                      | 421.9       | 1366       | 0.3089    | -944         |
| kurt_topeig.worst                |   3.377     |    3.685   | 0.9163    |   -0.3085    |



**student.h.cls.L12 vs student.z.proj.tap2**


| metric                           |     value_h |      value_z |    tau |        delta |
|:---------------------------------|------------:|-------------:|-------:|-------------:|
| uniformity                       |  -5.442e-05 |   -0.0004922 | 0.1106 |    0.0004378 |
| variance_floor.hinge             |   0.9975    |    0.9966    | 1.001  |    0.0009244 |
| offdiag_redundancy.mean_abs_corr |   0.4129    |    0.7942    | 0.5199 |   -0.3813    |
| rankme                           |   1.228     |    2.552     | 0.481  |   -1.325     |
| effective_rank                   |   6.559     |    1.231     | 5.33   |    5.328     |
| alpha                            |   1.774     |    1.211     | 1.465  |    0.5632    |
| epps_pulley                      | 421.9       | 1266         | 0.3333 | -844.1       |
| kurt_topeig.worst                |   3.377     |    5.807     | 0.5815 |   -2.43      |



**student.h.gap vs student.z.embed**


| metric                           |    value_h |     value_z |      tau |       delta |
|:---------------------------------|-----------:|------------:|---------:|------------:|
| uniformity                       |  -0.000731 |  -9.178e-06 | 79.64    |  -0.0007218 |
| variance_floor.hinge             |   0.9918   |   0.9975    |  0.9942  |  -0.005765  |
| offdiag_redundancy.mean_abs_corr |   0.5526   |   0.1945    |  2.841   |   0.3581    |
| rankme                           |   1.314    |   1.236     |  1.063   |   0.07756   |
| effective_rank                   |   2.258    |  36.19      |  0.06238 | -33.93      |
| alpha                            |   2.295    |   1.146     |  2.003   |   1.149     |
| epps_pulley                      | 840.6      | 346.8       |  2.424   | 493.8       |
| kurt_topeig.worst                |  12.96     |   5.216     |  2.484   |   7.742     |



**student.h.gap vs student.z.proj.out**


| metric                           |    value_h |     value_z |     tau |       delta |
|:---------------------------------|-----------:|------------:|--------:|------------:|
| uniformity                       |  -0.000731 |  -8.993e-05 |  8.128  |  -0.0006411 |
| variance_floor.hinge             |   0.9918   |   0.9997    |  0.9921 |  -0.007903  |
| offdiag_redundancy.mean_abs_corr |   0.5526   |   0.3302    |  1.674  |   0.2224    |
| rankme                           |   1.314    |   1.126     |  1.167  |   0.1878    |
| effective_rank                   |   2.258    |   7.051     |  0.3202 |  -4.793     |
| alpha                            |   2.295    |   1.146     |  2.003  |   1.149     |
| epps_pulley                      | 840.6      | 234.8       |  3.58   | 605.8       |
| kurt_topeig.worst                |  12.96     |   0.7709    | 16.81   |  12.19      |



**student.h.gap vs student.z.proj.tap1**


| metric                           |    value_h |    value_z |      tau |       delta |
|:---------------------------------|-----------:|-----------:|---------:|------------:|
| uniformity                       |  -0.000731 |   -0.05794 | 0.01262  |    0.05721  |
| variance_floor.hinge             |   0.9918   |    0.997   | 0.9948   |   -0.005191 |
| offdiag_redundancy.mean_abs_corr |   0.5526   |    0.06563 | 8.42     |    0.487    |
| rankme                           |   1.314    |  134.8     | 0.009747 | -133.5      |
| effective_rank                   |   2.258    |   38.31    | 0.05893  |  -36.05     |
| alpha                            |   2.295    |    1.09    | 2.106    |    1.205    |
| epps_pulley                      | 840.6      | 1366       | 0.6154   | -525.4      |
| kurt_topeig.worst                |  12.96     |    3.685   | 3.517    |    9.274    |



**student.h.gap vs student.z.proj.tap2**


| metric                           |    value_h |      value_z |    tau |        delta |
|:---------------------------------|-----------:|-------------:|-------:|-------------:|
| uniformity                       |  -0.000731 |   -0.0004922 | 1.485  |   -0.0002388 |
| variance_floor.hinge             |   0.9918   |    0.9966    | 0.9952 |   -0.004833  |
| offdiag_redundancy.mean_abs_corr |   0.5526   |    0.7942    | 0.6957 |   -0.2417    |
| rankme                           |   1.314    |    2.552     | 0.5148 |   -1.238     |
| effective_rank                   |   2.258    |    1.231     | 1.835  |    1.027     |
| alpha                            |   2.295    |    1.211     | 1.895  |    1.084     |
| epps_pulley                      | 840.6      | 1266         | 0.664  | -425.4       |
| kurt_topeig.worst                |  12.96     |    5.807     | 2.232  |    7.152     |



**student.h.gap.L03 vs student.z.embed**


| metric                           |   value_h |     value_z |        tau |     delta |
|:---------------------------------|----------:|------------:|-----------:|----------:|
| uniformity                       |  -0.01547 |  -9.178e-06 | 1685       |  -0.01546 |
| variance_floor.hinge             |   0.9627  |   0.9975    |    0.9651  |  -0.03479 |
| offdiag_redundancy.mean_abs_corr |   0.5684  |   0.1945    |    2.922   |   0.3739  |
| rankme                           |   1.994   |   1.236     |    1.612   |   0.7572  |
| effective_rank                   |   2.096   |  36.19      |    0.05791 | -34.09    |
| alpha                            |   3.239   |   1.146     |    2.826   |   2.093   |
| epps_pulley                      | 837.4     | 346.8       |    2.415   | 490.6     |
| kurt_topeig.worst                | 572.3     |   5.216     |  109.7     | 567.1     |



**student.h.gap.L03 vs student.z.proj.out**


| metric                           |   value_h |     value_z |      tau |     delta |
|:---------------------------------|----------:|------------:|---------:|----------:|
| uniformity                       |  -0.01547 |  -8.993e-05 | 172      |  -0.01538 |
| variance_floor.hinge             |   0.9627  |   0.9997    |   0.9631 |  -0.03693 |
| offdiag_redundancy.mean_abs_corr |   0.5684  |   0.3302    |   1.722  |   0.2383  |
| rankme                           |   1.994   |   1.126     |   1.77   |   0.8674  |
| effective_rank                   |   2.096   |   7.051     |   0.2973 |  -4.955   |
| alpha                            |   3.239   |   1.146     |   2.827  |   2.093   |
| epps_pulley                      | 837.4     | 234.8       |   3.566  | 602.6     |
| kurt_topeig.worst                | 572.3     |   0.7709    | 742.3    | 571.5     |



**student.h.gap.L03 vs student.z.proj.tap1**


| metric                           |   value_h |    value_z |       tau |      delta |
|:---------------------------------|----------:|-----------:|----------:|-----------:|
| uniformity                       |  -0.01547 |   -0.05794 |   0.267   |    0.04247 |
| variance_floor.hinge             |   0.9627  |    0.997   |   0.9657  |   -0.03422 |
| offdiag_redundancy.mean_abs_corr |   0.5684  |    0.06563 |   8.661   |    0.5028  |
| rankme                           |   1.994   |  134.8     |   0.01479 | -132.8     |
| effective_rank                   |   2.096   |   38.31    |   0.05471 |  -36.22    |
| alpha                            |   3.239   |    1.09    |   2.971   |    2.149   |
| epps_pulley                      | 837.4     | 1366       |   0.613   | -528.6     |
| kurt_topeig.worst                | 572.3     |    3.685   | 155.3     |  568.6     |



**student.h.gap.L03 vs student.z.proj.tap2**


| metric                           |   value_h |      value_z |     tau |      delta |
|:---------------------------------|----------:|-------------:|--------:|-----------:|
| uniformity                       |  -0.01547 |   -0.0004922 | 31.42   |   -0.01498 |
| variance_floor.hinge             |   0.9627  |    0.9966    |  0.966  |   -0.03386 |
| offdiag_redundancy.mean_abs_corr |   0.5684  |    0.7942    |  0.7157 |   -0.2258  |
| rankme                           |   1.994   |    2.552     |  0.7811 |   -0.5589  |
| effective_rank                   |   2.096   |    1.231     |  1.703  |    0.8654  |
| alpha                            |   3.239   |    1.211     |  2.675  |    2.028   |
| epps_pulley                      | 837.4     | 1266         |  0.6615 | -428.6     |
| kurt_topeig.worst                | 572.3     |    5.807     | 98.55   |  566.5     |



**student.h.gap.L06 vs student.z.embed**


| metric                           |    value_h |     value_z |       tau |      delta |
|:---------------------------------|-----------:|------------:|----------:|-----------:|
| uniformity                       |  -0.005126 |  -9.178e-06 | 558.4     |  -0.005116 |
| variance_floor.hinge             |   0.9784   |   0.9975    |   0.9809  |  -0.01909  |
| offdiag_redundancy.mean_abs_corr |   0.5611   |   0.1945    |   2.885   |   0.3666   |
| rankme                           |   1.611    |   1.236     |   1.303   |   0.3741   |
| effective_rank                   |   2.127    |  36.19      |   0.05877 | -34.06     |
| alpha                            |   2.892    |   1.146     |   2.524   |   1.746    |
| epps_pulley                      | 840.3      | 346.8       |   2.423   | 493.5      |
| kurt_topeig.worst                |  10.29     |   5.216     |   1.973   |   5.077    |



**student.h.gap.L06 vs student.z.proj.out**


| metric                           |    value_h |     value_z |     tau |      delta |
|:---------------------------------|-----------:|------------:|--------:|-----------:|
| uniformity                       |  -0.005126 |  -8.993e-05 | 56.99   |  -0.005036 |
| variance_floor.hinge             |   0.9784   |   0.9997    |  0.9788 |  -0.02123  |
| offdiag_redundancy.mean_abs_corr |   0.5611   |   0.3302    |  1.7    |   0.231    |
| rankme                           |   1.611    |   1.126     |  1.43   |   0.4844   |
| effective_rank                   |   2.127    |   7.051     |  0.3016 |  -4.924    |
| alpha                            |   2.892    |   1.146     |  2.524  |   1.746    |
| epps_pulley                      | 840.3      | 234.8       |  3.578  | 605.5      |
| kurt_topeig.worst                |  10.29     |   0.7709    | 13.35   |   9.523    |



**student.h.gap.L06 vs student.z.proj.tap1**


| metric                           |    value_h |    value_z |     tau |      delta |
|:---------------------------------|-----------:|-----------:|--------:|-----------:|
| uniformity                       |  -0.005126 |   -0.05794 | 0.08846 |    0.05281 |
| variance_floor.hinge             |   0.9784   |    0.997   | 0.9814  |   -0.01852 |
| offdiag_redundancy.mean_abs_corr |   0.5611   |    0.06563 | 8.55    |    0.4955  |
| rankme                           |   1.611    |  134.8     | 0.01195 | -133.2     |
| effective_rank                   |   2.127    |   38.31    | 0.05551 |  -36.18    |
| alpha                            |   2.892    |    1.09    | 2.653   |    1.802   |
| epps_pulley                      | 840.3      | 1366       | 0.6152  | -525.7     |
| kurt_topeig.worst                |  10.29     |    3.685   | 2.793   |    6.608   |



**student.h.gap.L06 vs student.z.proj.tap2**


| metric                           |    value_h |      value_z |     tau |       delta |
|:---------------------------------|-----------:|-------------:|--------:|------------:|
| uniformity                       |  -0.005126 |   -0.0004922 | 10.41   |   -0.004633 |
| variance_floor.hinge             |   0.9784   |    0.9966    |  0.9818 |   -0.01816  |
| offdiag_redundancy.mean_abs_corr |   0.5611   |    0.7942    |  0.7065 |   -0.2331   |
| rankme                           |   1.611    |    2.552     |  0.631  |   -0.9419   |
| effective_rank                   |   2.127    |    1.231     |  1.728  |    0.8963   |
| alpha                            |   2.892    |    1.211     |  2.388  |    1.681    |
| epps_pulley                      | 840.3      | 1266         |  0.6637 | -425.7      |
| kurt_topeig.worst                |  10.29     |    5.807     |  1.773  |    4.487    |



**student.h.gap.L09 vs student.z.embed**


| metric                           |    value_h |     value_z |       tau |      delta |
|:---------------------------------|-----------:|------------:|----------:|-----------:|
| uniformity                       |  -0.001022 |  -9.178e-06 | 111.4     |  -0.001013 |
| variance_floor.hinge             |   0.9902   |   0.9975    |   0.9927  |  -0.007286 |
| offdiag_redundancy.mean_abs_corr |   0.5469   |   0.1945    |   2.812   |   0.3524   |
| rankme                           |   1.355    |   1.236     |   1.096   |   0.119    |
| effective_rank                   |   2.285    |  36.19      |   0.06313 | -33.91     |
| alpha                            |   2.403    |   1.146     |   2.097   |   1.257    |
| epps_pulley                      | 845.3      | 346.8       |   2.437   | 498.5      |
| kurt_topeig.worst                |  12.32     |   5.216     |   2.361   |   7.1      |



**student.h.gap.L09 vs student.z.proj.out**


| metric                           |    value_h |     value_z |     tau |       delta |
|:---------------------------------|-----------:|------------:|--------:|------------:|
| uniformity                       |  -0.001022 |  -8.993e-05 | 11.37   |  -0.0009325 |
| variance_floor.hinge             |   0.9902   |   0.9997    |  0.9906 |  -0.009424  |
| offdiag_redundancy.mean_abs_corr |   0.5469   |   0.3302    |  1.657  |   0.2168    |
| rankme                           |   1.355    |   1.126     |  1.204  |   0.2292    |
| effective_rank                   |   2.285    |   7.051     |  0.324  |  -4.766     |
| alpha                            |   2.403    |   1.146     |  2.097  |   1.257     |
| epps_pulley                      | 845.3      | 234.8       |  3.6    | 610.4       |
| kurt_topeig.worst                |  12.32     |   0.7709    | 15.98   |  11.55      |



**student.h.gap.L09 vs student.z.proj.tap1**


| metric                           |    value_h |    value_z |     tau |       delta |
|:---------------------------------|-----------:|-----------:|--------:|------------:|
| uniformity                       |  -0.001022 |   -0.05794 | 0.01765 |    0.05692  |
| variance_floor.hinge             |   0.9902   |    0.997   | 0.9933  |   -0.006713 |
| offdiag_redundancy.mean_abs_corr |   0.5469   |    0.06563 | 8.334   |    0.4813   |
| rankme                           |   1.355    |  134.8     | 0.01005 | -133.5      |
| effective_rank                   |   2.285    |   38.31    | 0.05963 |  -36.03     |
| alpha                            |   2.403    |    1.09    | 2.204   |    1.313    |
| epps_pulley                      | 845.3      | 1366       | 0.6188  | -520.7      |
| kurt_topeig.worst                |  12.32     |    3.685   | 3.342   |    8.631    |



**student.h.gap.L09 vs student.z.proj.tap2**


| metric                           |    value_h |      value_z |    tau |        delta |
|:---------------------------------|-----------:|-------------:|-------:|-------------:|
| uniformity                       |  -0.001022 |   -0.0004922 | 2.077  |   -0.0005302 |
| variance_floor.hinge             |   0.9902   |    0.9966    | 0.9936 |   -0.006354  |
| offdiag_redundancy.mean_abs_corr |   0.5469   |    0.7942    | 0.6886 |   -0.2473    |
| rankme                           |   1.355    |    2.552     | 0.531  |   -1.197     |
| effective_rank                   |   2.285    |    1.231     | 1.857  |    1.054     |
| alpha                            |   2.403    |    1.211     | 1.984  |    1.192     |
| epps_pulley                      | 845.3      | 1266         | 0.6677 | -420.8       |
| kurt_topeig.worst                |  12.32     |    5.807     | 2.121  |    6.509     |



**student.h.gap.L12 vs student.z.embed**


| metric                           |    value_h |     value_z |      tau |       delta |
|:---------------------------------|-----------:|------------:|---------:|------------:|
| uniformity                       |  -0.000731 |  -9.178e-06 | 79.64    |  -0.0007218 |
| variance_floor.hinge             |   0.9918   |   0.9975    |  0.9942  |  -0.005765  |
| offdiag_redundancy.mean_abs_corr |   0.5526   |   0.1945    |  2.841   |   0.3581    |
| rankme                           |   1.314    |   1.236     |  1.063   |   0.07756   |
| effective_rank                   |   2.258    |  36.19      |  0.06238 | -33.93      |
| alpha                            |   2.295    |   1.146     |  2.003   |   1.149     |
| epps_pulley                      | 840.6      | 346.8       |  2.424   | 493.8       |
| kurt_topeig.worst                |  12.96     |   5.216     |  2.484   |   7.742     |



**student.h.gap.L12 vs student.z.proj.out**


| metric                           |    value_h |     value_z |     tau |       delta |
|:---------------------------------|-----------:|------------:|--------:|------------:|
| uniformity                       |  -0.000731 |  -8.993e-05 |  8.128  |  -0.0006411 |
| variance_floor.hinge             |   0.9918   |   0.9997    |  0.9921 |  -0.007903  |
| offdiag_redundancy.mean_abs_corr |   0.5526   |   0.3302    |  1.674  |   0.2224    |
| rankme                           |   1.314    |   1.126     |  1.167  |   0.1878    |
| effective_rank                   |   2.258    |   7.051     |  0.3202 |  -4.793     |
| alpha                            |   2.295    |   1.146     |  2.003  |   1.149     |
| epps_pulley                      | 840.6      | 234.8       |  3.58   | 605.8       |
| kurt_topeig.worst                |  12.96     |   0.7709    | 16.81   |  12.19      |



**student.h.gap.L12 vs student.z.proj.tap1**


| metric                           |    value_h |    value_z |      tau |       delta |
|:---------------------------------|-----------:|-----------:|---------:|------------:|
| uniformity                       |  -0.000731 |   -0.05794 | 0.01262  |    0.05721  |
| variance_floor.hinge             |   0.9918   |    0.997   | 0.9948   |   -0.005191 |
| offdiag_redundancy.mean_abs_corr |   0.5526   |    0.06563 | 8.42     |    0.487    |
| rankme                           |   1.314    |  134.8     | 0.009747 | -133.5      |
| effective_rank                   |   2.258    |   38.31    | 0.05893  |  -36.05     |
| alpha                            |   2.295    |    1.09    | 2.106    |    1.205    |
| epps_pulley                      | 840.6      | 1366       | 0.6154   | -525.4      |
| kurt_topeig.worst                |  12.96     |    3.685   | 3.517    |    9.274    |



**student.h.gap.L12 vs student.z.proj.tap2**


| metric                           |    value_h |      value_z |    tau |        delta |
|:---------------------------------|-----------:|-------------:|-------:|-------------:|
| uniformity                       |  -0.000731 |   -0.0004922 | 1.485  |   -0.0002388 |
| variance_floor.hinge             |   0.9918   |    0.9966    | 0.9952 |   -0.004833  |
| offdiag_redundancy.mean_abs_corr |   0.5526   |    0.7942    | 0.6957 |   -0.2417    |
| rankme                           |   1.314    |    2.552     | 0.5148 |   -1.238     |
| effective_rank                   |   2.258    |    1.231     | 1.835  |    1.027     |
| alpha                            |   2.295    |    1.211     | 1.895  |    1.084     |
| epps_pulley                      | 840.6      | 1266         | 0.664  | -425.4       |
| kurt_topeig.worst                |  12.96     |    5.807     | 2.232  |    7.152     |



### Pair metrics (alignment ↓ = more view-invariant)


| manifest                     | space               | metric         |     value |     ci_lo |     ci_hi |
|:-----------------------------|:--------------------|:---------------|----------:|----------:|----------:|
| imagenette.train.v1@audit_v1 | student.h.cls       | alignment      | 9.657e-05 | 9.448e-05 | 9.852e-05 |
| imagenette.train.v1@audit_v1 | student.h.cls       | cos_invariance | 1         | 1         | 1         |
| imagenette.train.v1@audit_v1 | student.h.gap       | alignment      | 0.000649  | 0.0006344 | 0.0006613 |
| imagenette.train.v1@audit_v1 | student.h.gap       | cos_invariance | 0.9997    | 0.9997    | 0.9997    |
| imagenette.train.v1@audit_v1 | student.z.embed     | alignment      | 7.591e-06 | 7.499e-06 | 7.678e-06 |
| imagenette.train.v1@audit_v1 | student.z.embed     | cos_invariance | 1         | 1         | 1         |
| imagenette.train.v1@audit_v1 | student.z.proj.out  | alignment      | 4.669e-05 | 4.574e-05 | 4.76e-05  |
| imagenette.train.v1@audit_v1 | student.z.proj.out  | cos_invariance | 1         | 1         | 1         |
| imagenette.train.v1@audit_v1 | student.z.proj.tap1 | alignment      | 0.0366    | 0.03627   | 0.03696   |
| imagenette.train.v1@audit_v1 | student.z.proj.tap1 | cos_invariance | 0.9817    | 0.9815    | 0.9819    |
| imagenette.train.v1@audit_v1 | student.z.proj.tap2 | alignment      | 0.0002529 | 0.0002438 | 0.0002613 |
| imagenette.train.v1@audit_v1 | student.z.proj.tap2 | cos_invariance | 0.9999    | 0.9999    | 0.9999    |



### Cross-space similarity (CKA is contested — read as the triple)


| space                             | metric              |   value |
|:----------------------------------|:--------------------|--------:|
| student.h.cls~student.z.embed     | cka_linear          | 0.5955  |
| student.h.cls~student.z.embed     | neighbor_jaccard    | 0.09869 |
| student.h.cls~student.z.embed     | procrustes_distance | 0.8224  |
| student.h.cls~student.z.embed     | knn_label_agreement | 0.2458  |
| student.h.cls~student.z.proj.out  | cka_linear          | 0.39    |
| student.h.cls~student.z.proj.out  | neighbor_jaccard    | 0.0101  |
| student.h.cls~student.z.proj.out  | procrustes_distance | 1.025   |
| student.h.cls~student.z.proj.out  | knn_label_agreement | 0.1526  |
| student.h.cls~student.z.proj.tap1 | cka_linear          | 0.4351  |
| student.h.cls~student.z.proj.tap1 | neighbor_jaccard    | 0.05777 |
| student.h.cls~student.z.proj.tap1 | procrustes_distance | 0.9399  |
| student.h.cls~student.z.proj.tap1 | knn_label_agreement | 0.2082  |
| student.h.cls~student.z.proj.tap2 | cka_linear          | 0.3472  |
| student.h.cls~student.z.proj.tap2 | neighbor_jaccard    | 0.02723 |
| student.h.cls~student.z.proj.tap2 | procrustes_distance | 1.017   |
| student.h.cls~student.z.proj.tap2 | knn_label_agreement | 0.189   |



### Probes


| space               |   knn_v1_k20 |   knn_v1_k200 |   linear_house_v1 |   linear_l2_v1 |
|:--------------------|-------------:|--------------:|------------------:|---------------:|
| student.h.cls       |       0.1725 |        0.1829 |            0.1149 |         0.1144 |
| student.h.cls.L03   |       0.1743 |        0.1868 |            0.1501 |         0.1322 |
| student.h.cls.L06   |       0.1809 |        0.1878 |            0.1304 |         0.1215 |
| student.h.cls.L09   |       0.1794 |        0.1763 |            0.1169 |         0.1195 |
| student.h.cls.L12   |       0.1725 |        0.1829 |            0.1146 |         0.1144 |
| student.h.gap       |       0.2617 |        0.2484 |            0.1567 |         0.1355 |
| student.h.gap.L03   |       0.2515 |        0.2456 |            0.2076 |         0.1625 |
| student.h.gap.L06   |       0.2589 |        0.2476 |            0.1676 |         0.1513 |
| student.h.gap.L09   |       0.2530 |        0.2451 |            0.1575 |         0.1376 |
| student.h.gap.L12   |       0.2617 |        0.2484 |            0.1567 |         0.1355 |
| student.z.embed     |       0.1511 |        0.1600 |            0.1068 |         0.1068 |
| student.z.proj.out  |       0.1215 |        0.1389 |            0.1068 |         0.1068 |
| student.z.proj.tap1 |       0.1572 |        0.1654 |            0.1847 |         0.1608 |
| student.z.proj.tap2 |       0.1422 |        0.1544 |            0.1368 |         0.1136 |



## toy.infonce.ext

### Battery (variant raw|full)

| model.space                           |   rankme |   effective_rank |   alpha |   epps_pulley |   kurt_topeig.worst |   uniformity |   offdiag_redundancy.mean_abs_corr |   variance_floor.hinge |
|:--------------------------------------|---------:|-----------------:|--------:|--------------:|--------------------:|-------------:|-----------------------------------:|-----------------------:|
| toy.infonce.ext · student.h.cls       |    17.48 |            19.98 |  2.705  |        172.5  |              0.708  |     -0.09946 |                             0.2053 |                 0.879  |
| toy.infonce.ext · student.h.cls.L03   |     5.33 |            19.96 |  2.98   |        171    |              1.292  |     -0.01074 |                             0.2188 |                 0.9602 |
| toy.infonce.ext · student.h.cls.L06   |    10.89 |            21.4  |  2.604  |        180.2  |              1.543  |     -0.03251 |                             0.2073 |                 0.9307 |
| toy.infonce.ext · student.h.cls.L09   |    15.22 |            20.33 |  2.648  |        174.7  |              0.573  |     -0.06832 |                             0.2062 |                 0.8999 |
| toy.infonce.ext · student.h.cls.L12   |    17.48 |            19.98 |  2.705  |        172.5  |              0.708  |     -0.09946 |                             0.2053 |                 0.879  |
| toy.infonce.ext · student.h.gap       |    40.1  |            15.33 |  2.176  |        181.5  |              1.826  |     -0.2347  |                             0.2118 |                 0.8731 |
| toy.infonce.ext · student.h.gap.L03   |    31.31 |            12.4  |  2.472  |        219.6  |              3.898  |     -0.2217  |                             0.2338 |                 0.8854 |
| toy.infonce.ext · student.h.gap.L06   |    40.51 |            16.44 |  2.261  |        190.4  |              3.686  |     -0.2493  |                             0.2145 |                 0.8716 |
| toy.infonce.ext · student.h.gap.L09   |    40.9  |            16.62 |  2.181  |        175.7  |              2.851  |     -0.2278  |                             0.2098 |                 0.8759 |
| toy.infonce.ext · student.h.gap.L12   |    40.1  |            15.33 |  2.176  |        181.5  |              1.826  |     -0.2347  |                             0.2118 |                 0.8731 |
| toy.infonce.ext · student.z.embed     |    19.2  |            17.33 |  3.948  |        153.1  |              0.5799 |     -0.6214  |                             0.204  |                 0.664  |
| toy.infonce.ext · student.z.proj.out  |    15.85 |            15.43 |  0.3328 |         16.22 |              0.5958 |     -3.418   |                             0.0496 |                 0      |
| toy.infonce.ext · student.z.proj.tap1 |   717.6  |            56.83 |  1.438  |        115.9  |              1.167  |     -2.587   |                             0.151  |                 0.6657 |
| toy.infonce.ext · student.z.proj.tap2 |   757.5  |            62.05 |  1.484  |        101.2  |              0.8866 |     -2.539   |                             0.154  |                 0.6601 |



### Transfer ratios τ = value(h) / value(z)


**student.h.cls vs student.z.embed**


| metric                           |   value_h |   value_z |    tau |     delta |
|:---------------------------------|----------:|----------:|-------:|----------:|
| uniformity                       |  -0.09946 |   -0.6214 | 0.16   |  0.522    |
| variance_floor.hinge             |   0.879   |    0.664  | 1.324  |  0.215    |
| offdiag_redundancy.mean_abs_corr |   0.2053  |    0.204  | 1.007  |  0.001384 |
| rankme                           |  17.48    |   19.2    | 0.9102 | -1.725    |
| effective_rank                   |  19.98    |   17.33   | 1.153  |  2.652    |
| alpha                            |   2.705   |    3.948  | 0.685  | -1.244    |
| epps_pulley                      | 172.5     |  153.1    | 1.127  | 19.44     |
| kurt_topeig.worst                |   0.708   |    0.5799 | 1.221  |  0.1281   |



**student.h.cls vs student.z.proj.out**


| metric                           |   value_h |   value_z |      tau |    delta |
|:---------------------------------|----------:|----------:|---------:|---------:|
| uniformity                       |  -0.09946 |   -3.418  |   0.0291 |   3.318  |
| variance_floor.hinge             |   0.879   |    0      | inf      |   0.879  |
| offdiag_redundancy.mean_abs_corr |   0.2053  |    0.0496 |   4.141  |   0.1558 |
| rankme                           |  17.48    |   15.85   |   1.103  |   1.63   |
| effective_rank                   |  19.98    |   15.43   |   1.295  |   4.551  |
| alpha                            |   2.705   |    0.3328 |   8.127  |   2.372  |
| epps_pulley                      | 172.5     |   16.22   |  10.63   | 156.3    |
| kurt_topeig.worst                |   0.708   |    0.5958 |   1.188  |   0.1123 |



**student.h.cls vs student.z.proj.tap1**


| metric                           |   value_h |   value_z |     tau |      delta |
|:---------------------------------|----------:|----------:|--------:|-----------:|
| uniformity                       |  -0.09946 |   -2.587  | 0.03845 |    2.487   |
| variance_floor.hinge             |   0.879   |    0.6657 | 1.32    |    0.2133  |
| offdiag_redundancy.mean_abs_corr |   0.2053  |    0.151  | 1.36    |    0.05433 |
| rankme                           |  17.48    |  717.6    | 0.02436 | -700.2     |
| effective_rank                   |  19.98    |   56.83   | 0.3516  |  -36.85    |
| alpha                            |   2.705   |    1.438  | 1.88    |    1.266   |
| epps_pulley                      | 172.5     |  115.9    | 1.488   |   56.58    |
| kurt_topeig.worst                |   0.708   |    1.167  | 0.6069  |   -0.4585  |



**student.h.cls vs student.z.proj.tap2**


| metric                           |   value_h |   value_z |     tau |      delta |
|:---------------------------------|----------:|----------:|--------:|-----------:|
| uniformity                       |  -0.09946 |   -2.539  | 0.03918 |    2.439   |
| variance_floor.hinge             |   0.879   |    0.6601 | 1.332   |    0.2189  |
| offdiag_redundancy.mean_abs_corr |   0.2053  |    0.154  | 1.333   |    0.05132 |
| rankme                           |  17.48    |  757.5    | 0.02307 | -740       |
| effective_rank                   |  19.98    |   62.05   | 0.322   |  -42.07    |
| alpha                            |   2.705   |    1.484  | 1.823   |    1.221   |
| epps_pulley                      | 172.5     |  101.2    | 1.705   |   71.34    |
| kurt_topeig.worst                |   0.708   |    0.8866 | 0.7986  |   -0.1786  |



**student.h.cls.L03 vs student.z.embed**


| metric                           |   value_h |   value_z |     tau |     delta |
|:---------------------------------|----------:|----------:|--------:|----------:|
| uniformity                       |  -0.01074 |   -0.6214 | 0.01728 |   0.6107  |
| variance_floor.hinge             |   0.9602  |    0.664  | 1.446   |   0.2961  |
| offdiag_redundancy.mean_abs_corr |   0.2188  |    0.204  | 1.073   |   0.01487 |
| rankme                           |   5.33    |   19.2    | 0.2776  | -13.87    |
| effective_rank                   |  19.96    |   17.33   | 1.152   |   2.629   |
| alpha                            |   2.98    |    3.948  | 0.7547  |  -0.9686  |
| epps_pulley                      | 171       |  153.1    | 1.117   |  17.95    |
| kurt_topeig.worst                |   1.292   |    0.5799 | 2.228   |   0.7123  |



**student.h.cls.L03 vs student.z.proj.out**


| metric                           |   value_h |   value_z |        tau |    delta |
|:---------------------------------|----------:|----------:|-----------:|---------:|
| uniformity                       |  -0.01074 |   -3.418  |   0.003141 |   3.407  |
| variance_floor.hinge             |   0.9602  |    0      | inf        |   0.9602 |
| offdiag_redundancy.mean_abs_corr |   0.2188  |    0.0496 |   4.412    |   0.1692 |
| rankme                           |   5.33    |   15.85   |   0.3363   | -10.52   |
| effective_rank                   |  19.96    |   15.43   |   1.294    |   4.529  |
| alpha                            |   2.98    |    0.3328 |   8.954    |   2.647  |
| epps_pulley                      | 171       |   16.22   |  10.54     | 154.8    |
| kurt_topeig.worst                |   1.292   |    0.5958 |   2.169    |   0.6965 |



**student.h.cls.L03 vs student.z.proj.tap1**


| metric                           |   value_h |   value_z |      tau |      delta |
|:---------------------------------|----------:|----------:|---------:|-----------:|
| uniformity                       |  -0.01074 |   -2.587  | 0.00415  |    2.576   |
| variance_floor.hinge             |   0.9602  |    0.6657 | 1.442    |    0.2945  |
| offdiag_redundancy.mean_abs_corr |   0.2188  |    0.151  | 1.449    |    0.06781 |
| rankme                           |   5.33    |  717.6    | 0.007427 | -712.3     |
| effective_rank                   |  19.96    |   56.83   | 0.3512   |  -36.87    |
| alpha                            |   2.98    |    1.438  | 2.072    |    1.541   |
| epps_pulley                      | 171       |  115.9    | 1.475    |   55.09    |
| kurt_topeig.worst                |   1.292   |    1.167  | 1.108    |    0.1257  |



**student.h.cls.L03 vs student.z.proj.tap2**


| metric                           |   value_h |   value_z |      tau |      delta |
|:---------------------------------|----------:|----------:|---------:|-----------:|
| uniformity                       |  -0.01074 |   -2.539  | 0.004229 |    2.528   |
| variance_floor.hinge             |   0.9602  |    0.6601 | 1.455    |    0.3     |
| offdiag_redundancy.mean_abs_corr |   0.2188  |    0.154  | 1.421    |    0.06481 |
| rankme                           |   5.33    |  757.5    | 0.007037 | -752.2     |
| effective_rank                   |  19.96    |   62.05   | 0.3217   |  -42.09    |
| alpha                            |   2.98    |    1.484  | 2.008    |    1.496   |
| epps_pulley                      | 171       |  101.2    | 1.69     |   69.85    |
| kurt_topeig.worst                |   1.292   |    0.8866 | 1.458    |    0.4056  |



**student.h.cls.L06 vs student.z.embed**


| metric                           |   value_h |   value_z |     tau |     delta |
|:---------------------------------|----------:|----------:|--------:|----------:|
| uniformity                       |  -0.03251 |   -0.6214 | 0.05232 |  0.5889   |
| variance_floor.hinge             |   0.9307  |    0.664  | 1.402   |  0.2666   |
| offdiag_redundancy.mean_abs_corr |   0.2073  |    0.204  | 1.016   |  0.003355 |
| rankme                           |  10.89    |   19.2    | 0.567   | -8.314    |
| effective_rank                   |  21.4     |   17.33   | 1.235   |  4.072    |
| alpha                            |   2.604   |    3.948  | 0.6594  | -1.345    |
| epps_pulley                      | 180.2     |  153.1    | 1.178   | 27.17     |
| kurt_topeig.worst                |   1.543   |    0.5799 | 2.66    |  0.963    |



**student.h.cls.L06 vs student.z.proj.out**


| metric                           |   value_h |   value_z |        tau |    delta |
|:---------------------------------|----------:|----------:|-----------:|---------:|
| uniformity                       |  -0.03251 |   -3.418  |   0.009512 |   3.385  |
| variance_floor.hinge             |   0.9307  |    0      | inf        |   0.9307 |
| offdiag_redundancy.mean_abs_corr |   0.2073  |    0.0496 |   4.18     |   0.1577 |
| rankme                           |  10.89    |   15.85   |   0.6871   |  -4.959  |
| effective_rank                   |  21.4     |   15.43   |   1.387    |   5.971  |
| alpha                            |   2.604   |    0.3328 |   7.824    |   2.271  |
| epps_pulley                      | 180.2     |   16.22   |  11.11     | 164      |
| kurt_topeig.worst                |   1.543   |    0.5958 |   2.59     |   0.9471 |



**student.h.cls.L06 vs student.z.proj.tap1**


| metric                           |   value_h |   value_z |     tau |     delta |
|:---------------------------------|----------:|----------:|--------:|----------:|
| uniformity                       |  -0.03251 |   -2.587  | 0.01257 |    2.554  |
| variance_floor.hinge             |   0.9307  |    0.6657 | 1.398   |    0.265  |
| offdiag_redundancy.mean_abs_corr |   0.2073  |    0.151  | 1.373   |    0.0563 |
| rankme                           |  10.89    |  717.6    | 0.01517 | -706.7    |
| effective_rank                   |  21.4     |   56.83   | 0.3766  |  -35.43   |
| alpha                            |   2.604   |    1.438  | 1.81    |    1.165  |
| epps_pulley                      | 180.2     |  115.9    | 1.555   |   64.31   |
| kurt_topeig.worst                |   1.543   |    1.167  | 1.323   |    0.3763 |



**student.h.cls.L06 vs student.z.proj.tap2**


| metric                           |   value_h |   value_z |     tau |      delta |
|:---------------------------------|----------:|----------:|--------:|-----------:|
| uniformity                       |  -0.03251 |   -2.539  | 0.01281 |    2.506   |
| variance_floor.hinge             |   0.9307  |    0.6601 | 1.41    |    0.2706  |
| offdiag_redundancy.mean_abs_corr |   0.2073  |    0.154  | 1.346   |    0.05329 |
| rankme                           |  10.89    |  757.5    | 0.01438 | -746.6     |
| effective_rank                   |  21.4     |   62.05   | 0.3449  |  -40.65    |
| alpha                            |   2.604   |    1.484  | 1.754   |    1.12    |
| epps_pulley                      | 180.2     |  101.2    | 1.782   |   79.07    |
| kurt_topeig.worst                |   1.543   |    0.8866 | 1.74    |    0.6563  |



**student.h.cls.L09 vs student.z.embed**


| metric                           |   value_h |   value_z |    tau |     delta |
|:---------------------------------|----------:|----------:|-------:|----------:|
| uniformity                       |  -0.06832 |   -0.6214 | 0.1099 |  0.5531   |
| variance_floor.hinge             |   0.8999  |    0.664  | 1.355  |  0.2358   |
| offdiag_redundancy.mean_abs_corr |   0.2062  |    0.204  | 1.011  |  0.002211 |
| rankme                           |  15.22    |   19.2    | 0.7923 | -3.988    |
| effective_rank                   |  20.33    |   17.33   | 1.173  |  3        |
| alpha                            |   2.648   |    3.948  | 0.6707 | -1.3      |
| epps_pulley                      | 174.7     |  153.1    | 1.141  | 21.61     |
| kurt_topeig.worst                |   0.573   |    0.5799 | 0.988  | -0.006985 |



**student.h.cls.L09 vs student.z.proj.out**


| metric                           |   value_h |   value_z |       tau |     delta |
|:---------------------------------|----------:|----------:|----------:|----------:|
| uniformity                       |  -0.06832 |   -3.418  |   0.01999 |   3.349   |
| variance_floor.hinge             |   0.8999  |    0      | inf       |   0.8999  |
| offdiag_redundancy.mean_abs_corr |   0.2062  |    0.0496 |   4.157   |   0.1566  |
| rankme                           |  15.22    |   15.85   |   0.96    |  -0.6333  |
| effective_rank                   |  20.33    |   15.43   |   1.318   |   4.899   |
| alpha                            |   2.648   |    0.3328 |   7.958   |   2.315   |
| epps_pulley                      | 174.7     |   16.22   |  10.77    | 158.5     |
| kurt_topeig.worst                |   0.573   |    0.5958 |   0.9617  |  -0.02281 |



**student.h.cls.L09 vs student.z.proj.tap1**


| metric                           |   value_h |   value_z |     tau |      delta |
|:---------------------------------|----------:|----------:|--------:|-----------:|
| uniformity                       |  -0.06832 |   -2.587  | 0.02641 |    2.519   |
| variance_floor.hinge             |   0.8999  |    0.6657 | 1.352   |    0.2341  |
| offdiag_redundancy.mean_abs_corr |   0.2062  |    0.151  | 1.365   |    0.05516 |
| rankme                           |  15.22    |  717.6    | 0.0212  | -702.4     |
| effective_rank                   |  20.33    |   56.83   | 0.3577  |  -36.5     |
| alpha                            |   2.648   |    1.438  | 1.841   |    1.21    |
| epps_pulley                      | 174.7     |  115.9    | 1.507   |   58.75    |
| kurt_topeig.worst                |   0.573   |    1.167  | 0.4912  |   -0.5936  |



**student.h.cls.L09 vs student.z.proj.tap2**


| metric                           |   value_h |   value_z |     tau |      delta |
|:---------------------------------|----------:|----------:|--------:|-----------:|
| uniformity                       |  -0.06832 |   -2.539  | 0.02691 |    2.47    |
| variance_floor.hinge             |   0.8999  |    0.6601 | 1.363   |    0.2397  |
| offdiag_redundancy.mean_abs_corr |   0.2062  |    0.154  | 1.339   |    0.05215 |
| rankme                           |  15.22    |  757.5    | 0.02009 | -742.3     |
| effective_rank                   |  20.33    |   62.05   | 0.3276  |  -41.72    |
| alpha                            |   2.648   |    1.484  | 1.785   |    1.164   |
| epps_pulley                      | 174.7     |  101.2    | 1.727   |   73.51    |
| kurt_topeig.worst                |   0.573   |    0.8866 | 0.6462  |   -0.3136  |



**student.h.cls.L12 vs student.z.embed**


| metric                           |   value_h |   value_z |    tau |     delta |
|:---------------------------------|----------:|----------:|-------:|----------:|
| uniformity                       |  -0.09946 |   -0.6214 | 0.16   |  0.522    |
| variance_floor.hinge             |   0.879   |    0.664  | 1.324  |  0.215    |
| offdiag_redundancy.mean_abs_corr |   0.2053  |    0.204  | 1.007  |  0.001384 |
| rankme                           |  17.48    |   19.2    | 0.9102 | -1.725    |
| effective_rank                   |  19.98    |   17.33   | 1.153  |  2.652    |
| alpha                            |   2.705   |    3.948  | 0.685  | -1.244    |
| epps_pulley                      | 172.5     |  153.1    | 1.127  | 19.44     |
| kurt_topeig.worst                |   0.708   |    0.5799 | 1.221  |  0.1281   |



**student.h.cls.L12 vs student.z.proj.out**


| metric                           |   value_h |   value_z |      tau |    delta |
|:---------------------------------|----------:|----------:|---------:|---------:|
| uniformity                       |  -0.09946 |   -3.418  |   0.0291 |   3.318  |
| variance_floor.hinge             |   0.879   |    0      | inf      |   0.879  |
| offdiag_redundancy.mean_abs_corr |   0.2053  |    0.0496 |   4.141  |   0.1558 |
| rankme                           |  17.48    |   15.85   |   1.103  |   1.63   |
| effective_rank                   |  19.98    |   15.43   |   1.295  |   4.551  |
| alpha                            |   2.705   |    0.3328 |   8.127  |   2.372  |
| epps_pulley                      | 172.5     |   16.22   |  10.63   | 156.3    |
| kurt_topeig.worst                |   0.708   |    0.5958 |   1.188  |   0.1123 |



**student.h.cls.L12 vs student.z.proj.tap1**


| metric                           |   value_h |   value_z |     tau |      delta |
|:---------------------------------|----------:|----------:|--------:|-----------:|
| uniformity                       |  -0.09946 |   -2.587  | 0.03845 |    2.487   |
| variance_floor.hinge             |   0.879   |    0.6657 | 1.32    |    0.2133  |
| offdiag_redundancy.mean_abs_corr |   0.2053  |    0.151  | 1.36    |    0.05433 |
| rankme                           |  17.48    |  717.6    | 0.02436 | -700.2     |
| effective_rank                   |  19.98    |   56.83   | 0.3516  |  -36.85    |
| alpha                            |   2.705   |    1.438  | 1.88    |    1.266   |
| epps_pulley                      | 172.5     |  115.9    | 1.488   |   56.58    |
| kurt_topeig.worst                |   0.708   |    1.167  | 0.6069  |   -0.4585  |



**student.h.cls.L12 vs student.z.proj.tap2**


| metric                           |   value_h |   value_z |     tau |      delta |
|:---------------------------------|----------:|----------:|--------:|-----------:|
| uniformity                       |  -0.09946 |   -2.539  | 0.03918 |    2.439   |
| variance_floor.hinge             |   0.879   |    0.6601 | 1.332   |    0.2189  |
| offdiag_redundancy.mean_abs_corr |   0.2053  |    0.154  | 1.333   |    0.05132 |
| rankme                           |  17.48    |  757.5    | 0.02307 | -740       |
| effective_rank                   |  19.98    |   62.05   | 0.322   |  -42.07    |
| alpha                            |   2.705   |    1.484  | 1.823   |    1.221   |
| epps_pulley                      | 172.5     |  101.2    | 1.705   |   71.34    |
| kurt_topeig.worst                |   0.708   |    0.8866 | 0.7986  |   -0.1786  |



**student.h.gap vs student.z.embed**


| metric                           |   value_h |   value_z |    tau |     delta |
|:---------------------------------|----------:|----------:|-------:|----------:|
| uniformity                       |   -0.2347 |   -0.6214 | 0.3776 |  0.3867   |
| variance_floor.hinge             |    0.8731 |    0.664  | 1.315  |  0.2091   |
| offdiag_redundancy.mean_abs_corr |    0.2118 |    0.204  | 1.039  |  0.007883 |
| rankme                           |   40.1    |   19.2    | 2.088  | 20.9      |
| effective_rank                   |   15.33   |   17.33   | 0.8847 | -1.999    |
| alpha                            |    2.176  |    3.948  | 0.5512 | -1.772    |
| epps_pulley                      |  181.5    |  153.1    | 1.186  | 28.4      |
| kurt_topeig.worst                |    1.826  |    0.5799 | 3.149  |  1.246    |



**student.h.gap vs student.z.proj.out**


| metric                           |   value_h |   value_z |       tau |     delta |
|:---------------------------------|----------:|----------:|----------:|----------:|
| uniformity                       |   -0.2347 |   -3.418  |   0.06866 |   3.183   |
| variance_floor.hinge             |    0.8731 |    0      | inf       |   0.8731  |
| offdiag_redundancy.mean_abs_corr |    0.2118 |    0.0496 |   4.272   |   0.1623  |
| rankme                           |   40.1    |   15.85   |   2.53    |  24.25    |
| effective_rank                   |   15.33   |   15.43   |   0.9936  |  -0.09928 |
| alpha                            |    2.176  |    0.3328 |   6.54    |   1.844   |
| epps_pulley                      |  181.5    |   16.22   |  11.19    | 165.3     |
| kurt_topeig.worst                |    1.826  |    0.5958 |   3.065   |   1.23    |



**student.h.gap vs student.z.proj.tap1**


| metric                           |   value_h |   value_z |     tau |      delta |
|:---------------------------------|----------:|----------:|--------:|-----------:|
| uniformity                       |   -0.2347 |   -2.587  | 0.09072 |    2.352   |
| variance_floor.hinge             |    0.8731 |    0.6657 | 1.312   |    0.2074  |
| offdiag_redundancy.mean_abs_corr |    0.2118 |    0.151  | 1.403   |    0.06083 |
| rankme                           |   40.1    |  717.6    | 0.05588 | -677.5     |
| effective_rank                   |   15.33   |   56.83   | 0.2698  |  -41.5     |
| alpha                            |    2.176  |    1.438  | 1.513   |    0.7379  |
| epps_pulley                      |  181.5    |  115.9    | 1.565   |   65.54    |
| kurt_topeig.worst                |    1.826  |    1.167  | 1.565   |    0.6596  |



**student.h.gap vs student.z.proj.tap2**


| metric                           |   value_h |   value_z |     tau |      delta |
|:---------------------------------|----------:|----------:|--------:|-----------:|
| uniformity                       |   -0.2347 |   -2.539  | 0.09244 |    2.304   |
| variance_floor.hinge             |    0.8731 |    0.6601 | 1.323   |    0.213   |
| offdiag_redundancy.mean_abs_corr |    0.2118 |    0.154  | 1.375   |    0.05782 |
| rankme                           |   40.1    |  757.5    | 0.05294 | -717.4     |
| effective_rank                   |   15.33   |   62.05   | 0.2471  |  -46.72    |
| alpha                            |    2.176  |    1.484  | 1.467   |    0.6923  |
| epps_pulley                      |  181.5    |  101.2    | 1.794   |   80.31    |
| kurt_topeig.worst                |    1.826  |    0.8866 | 2.06    |    0.9395  |



**student.h.gap.L03 vs student.z.embed**


| metric                           |   value_h |   value_z |    tau |    delta |
|:---------------------------------|----------:|----------:|-------:|---------:|
| uniformity                       |   -0.2217 |   -0.6214 | 0.3568 |  0.3997  |
| variance_floor.hinge             |    0.8854 |    0.664  | 1.333  |  0.2214  |
| offdiag_redundancy.mean_abs_corr |    0.2338 |    0.204  | 1.146  |  0.02988 |
| rankme                           |   31.31   |   19.2    | 1.63   | 12.11    |
| effective_rank                   |   12.4    |   17.33   | 0.7155 | -4.931   |
| alpha                            |    2.472  |    3.948  | 0.6262 | -1.476   |
| epps_pulley                      |  219.6    |  153.1    | 1.435  | 66.57    |
| kurt_topeig.worst                |    3.898  |    0.5799 | 6.722  |  3.318   |



**student.h.gap.L03 vs student.z.proj.out**


| metric                           |   value_h |   value_z |       tau |    delta |
|:---------------------------------|----------:|----------:|----------:|---------:|
| uniformity                       |   -0.2217 |   -3.418  |   0.06488 |   3.196  |
| variance_floor.hinge             |    0.8854 |    0      | inf       |   0.8854 |
| offdiag_redundancy.mean_abs_corr |    0.2338 |    0.0496 |   4.715   |   0.1842 |
| rankme                           |   31.31   |   15.85   |   1.976   |  15.46   |
| effective_rank                   |   12.4    |   15.43   |   0.8036  |  -3.031  |
| alpha                            |    2.472  |    0.3328 |   7.429   |   2.14   |
| epps_pulley                      |  219.6    |   16.22   |  13.54    | 203.4    |
| kurt_topeig.worst                |    3.898  |    0.5958 |   6.543   |   3.302  |



**student.h.gap.L03 vs student.z.proj.tap1**


| metric                           |   value_h |   value_z |     tau |      delta |
|:---------------------------------|----------:|----------:|--------:|-----------:|
| uniformity                       |   -0.2217 |   -2.587  | 0.08571 |    2.365   |
| variance_floor.hinge             |    0.8854 |    0.6657 | 1.33    |    0.2197  |
| offdiag_redundancy.mean_abs_corr |    0.2338 |    0.151  | 1.548   |    0.08282 |
| rankme                           |   31.31   |  717.6    | 0.04363 | -686.3     |
| effective_rank                   |   12.4    |   56.83   | 0.2182  |  -44.44    |
| alpha                            |    2.472  |    1.438  | 1.719   |    1.034   |
| epps_pulley                      |  219.6    |  115.9    | 1.895   |  103.7     |
| kurt_topeig.worst                |    3.898  |    1.167  | 3.342   |    2.732   |



**student.h.gap.L03 vs student.z.proj.tap2**


| metric                           |   value_h |   value_z |     tau |      delta |
|:---------------------------------|----------:|----------:|--------:|-----------:|
| uniformity                       |   -0.2217 |   -2.539  | 0.08734 |    2.317   |
| variance_floor.hinge             |    0.8854 |    0.6601 | 1.341   |    0.2253  |
| offdiag_redundancy.mean_abs_corr |    0.2338 |    0.154  | 1.518   |    0.07982 |
| rankme                           |   31.31   |  757.5    | 0.04133 | -726.2     |
| effective_rank                   |   12.4    |   62.05   | 0.1998  |  -49.65    |
| alpha                            |    2.472  |    1.484  | 1.666   |    0.9884  |
| epps_pulley                      |  219.6    |  101.2    | 2.171   |  118.5     |
| kurt_topeig.worst                |    3.898  |    0.8866 | 4.397   |    3.012   |



**student.h.gap.L06 vs student.z.embed**


| metric                           |   value_h |   value_z |    tau |    delta |
|:---------------------------------|----------:|----------:|-------:|---------:|
| uniformity                       |   -0.2493 |   -0.6214 | 0.4012 |  0.3721  |
| variance_floor.hinge             |    0.8716 |    0.664  | 1.313  |  0.2076  |
| offdiag_redundancy.mean_abs_corr |    0.2145 |    0.204  | 1.052  |  0.01052 |
| rankme                           |   40.51   |   19.2    | 2.11   | 21.31    |
| effective_rank                   |   16.44   |   17.33   | 0.9488 | -0.8873  |
| alpha                            |    2.261  |    3.948  | 0.5726 | -1.688   |
| epps_pulley                      |  190.4    |  153.1    | 1.244  | 37.34    |
| kurt_topeig.worst                |    3.686  |    0.5799 | 6.356  |  3.106   |



**student.h.gap.L06 vs student.z.proj.out**


| metric                           |   value_h |   value_z |       tau |    delta |
|:---------------------------------|----------:|----------:|----------:|---------:|
| uniformity                       |   -0.2493 |   -3.418  |   0.07295 |   3.168  |
| variance_floor.hinge             |    0.8716 |    0      | inf       |   0.8716 |
| offdiag_redundancy.mean_abs_corr |    0.2145 |    0.0496 |   4.325   |   0.1649 |
| rankme                           |   40.51   |   15.85   |   2.556   |  24.66   |
| effective_rank                   |   16.44   |   15.43   |   1.066   |   1.012  |
| alpha                            |    2.261  |    0.3328 |   6.794   |   1.928  |
| epps_pulley                      |  190.4    |   16.22   |  11.74    | 174.2    |
| kurt_topeig.worst                |    3.686  |    0.5958 |   6.187   |   3.09   |



**student.h.gap.L06 vs student.z.proj.tap1**


| metric                           |   value_h |   value_z |     tau |      delta |
|:---------------------------------|----------:|----------:|--------:|-----------:|
| uniformity                       |   -0.2493 |   -2.587  | 0.09638 |    2.338   |
| variance_floor.hinge             |    0.8716 |    0.6657 | 1.309   |    0.2059  |
| offdiag_redundancy.mean_abs_corr |    0.2145 |    0.151  | 1.42    |    0.06347 |
| rankme                           |   40.51   |  717.6    | 0.05645 | -677.1     |
| effective_rank                   |   16.44   |   56.83   | 0.2893  |  -40.39    |
| alpha                            |    2.261  |    1.438  | 1.572   |    0.8224  |
| epps_pulley                      |  190.4    |  115.9    | 1.642   |   74.48    |
| kurt_topeig.worst                |    3.686  |    1.167  | 3.16    |    2.519   |



**student.h.gap.L06 vs student.z.proj.tap2**


| metric                           |   value_h |   value_z |     tau |      delta |
|:---------------------------------|----------:|----------:|--------:|-----------:|
| uniformity                       |   -0.2493 |   -2.539  | 0.09822 |    2.289   |
| variance_floor.hinge             |    0.8716 |    0.6601 | 1.32    |    0.2115  |
| offdiag_redundancy.mean_abs_corr |    0.2145 |    0.154  | 1.393   |    0.06046 |
| rankme                           |   40.51   |  757.5    | 0.05348 | -717       |
| effective_rank                   |   16.44   |   62.05   | 0.265   |  -45.61    |
| alpha                            |    2.261  |    1.484  | 1.523   |    0.7769  |
| epps_pulley                      |  190.4    |  101.2    | 1.882   |   89.24    |
| kurt_topeig.worst                |    3.686  |    0.8866 | 4.157   |    2.799   |



**student.h.gap.L09 vs student.z.embed**


| metric                           |   value_h |   value_z |    tau |     delta |
|:---------------------------------|----------:|----------:|-------:|----------:|
| uniformity                       |   -0.2278 |   -0.6214 | 0.3666 |  0.3936   |
| variance_floor.hinge             |    0.8759 |    0.664  | 1.319  |  0.2119   |
| offdiag_redundancy.mean_abs_corr |    0.2098 |    0.204  | 1.029  |  0.005821 |
| rankme                           |   40.9    |   19.2    | 2.13   | 21.69     |
| effective_rank                   |   16.62   |   17.33   | 0.9588 | -0.7144   |
| alpha                            |    2.181  |    3.948  | 0.5523 | -1.768    |
| epps_pulley                      |  175.7    |  153.1    | 1.148  | 22.67     |
| kurt_topeig.worst                |    2.851  |    0.5799 | 4.916  |  2.271    |



**student.h.gap.L09 vs student.z.proj.out**


| metric                           |   value_h |   value_z |       tau |    delta |
|:---------------------------------|----------:|----------:|----------:|---------:|
| uniformity                       |   -0.2278 |   -3.418  |   0.06665 |   3.19   |
| variance_floor.hinge             |    0.8759 |    0      | inf       |   0.8759 |
| offdiag_redundancy.mean_abs_corr |    0.2098 |    0.0496 |   4.23    |   0.1602 |
| rankme                           |   40.9    |   15.85   |   2.58    |  25.05   |
| effective_rank                   |   16.62   |   15.43   |   1.077   |   1.185  |
| alpha                            |    2.181  |    0.3328 |   6.553   |   1.848  |
| epps_pulley                      |  175.7    |   16.22   |  10.83    | 159.5    |
| kurt_topeig.worst                |    2.851  |    0.5958 |   4.786   |   2.255  |



**student.h.gap.L09 vs student.z.proj.tap1**


| metric                           |   value_h |   value_z |     tau |      delta |
|:---------------------------------|----------:|----------:|--------:|-----------:|
| uniformity                       |   -0.2278 |   -2.587  | 0.08805 |    2.359   |
| variance_floor.hinge             |    0.8759 |    0.6657 | 1.316   |    0.2102  |
| offdiag_redundancy.mean_abs_corr |    0.2098 |    0.151  | 1.389   |    0.05877 |
| rankme                           |   40.9    |  717.6    | 0.05699 | -676.7     |
| effective_rank                   |   16.62   |   56.83   | 0.2924  |  -40.22    |
| alpha                            |    2.181  |    1.438  | 1.516   |    0.7423  |
| epps_pulley                      |  175.7    |  115.9    | 1.516   |   59.81    |
| kurt_topeig.worst                |    2.851  |    1.167  | 2.444   |    1.684   |



**student.h.gap.L09 vs student.z.proj.tap2**


| metric                           |   value_h |   value_z |     tau |      delta |
|:---------------------------------|----------:|----------:|--------:|-----------:|
| uniformity                       |   -0.2278 |   -2.539  | 0.08973 |    2.311   |
| variance_floor.hinge             |    0.8759 |    0.6601 | 1.327   |    0.2158  |
| offdiag_redundancy.mean_abs_corr |    0.2098 |    0.154  | 1.362   |    0.05576 |
| rankme                           |   40.9    |  757.5    | 0.05399 | -716.6     |
| effective_rank                   |   16.62   |   62.05   | 0.2678  |  -45.44    |
| alpha                            |    2.181  |    1.484  | 1.47    |    0.6967  |
| epps_pulley                      |  175.7    |  101.2    | 1.737   |   74.57    |
| kurt_topeig.worst                |    2.851  |    0.8866 | 3.216   |    1.964   |



**student.h.gap.L12 vs student.z.embed**


| metric                           |   value_h |   value_z |    tau |     delta |
|:---------------------------------|----------:|----------:|-------:|----------:|
| uniformity                       |   -0.2347 |   -0.6214 | 0.3776 |  0.3867   |
| variance_floor.hinge             |    0.8731 |    0.664  | 1.315  |  0.2091   |
| offdiag_redundancy.mean_abs_corr |    0.2118 |    0.204  | 1.039  |  0.007883 |
| rankme                           |   40.1    |   19.2    | 2.088  | 20.9      |
| effective_rank                   |   15.33   |   17.33   | 0.8847 | -1.999    |
| alpha                            |    2.176  |    3.948  | 0.5512 | -1.772    |
| epps_pulley                      |  181.5    |  153.1    | 1.186  | 28.4      |
| kurt_topeig.worst                |    1.826  |    0.5799 | 3.149  |  1.246    |



**student.h.gap.L12 vs student.z.proj.out**


| metric                           |   value_h |   value_z |       tau |     delta |
|:---------------------------------|----------:|----------:|----------:|----------:|
| uniformity                       |   -0.2347 |   -3.418  |   0.06866 |   3.183   |
| variance_floor.hinge             |    0.8731 |    0      | inf       |   0.8731  |
| offdiag_redundancy.mean_abs_corr |    0.2118 |    0.0496 |   4.272   |   0.1623  |
| rankme                           |   40.1    |   15.85   |   2.53    |  24.25    |
| effective_rank                   |   15.33   |   15.43   |   0.9936  |  -0.09928 |
| alpha                            |    2.176  |    0.3328 |   6.54    |   1.844   |
| epps_pulley                      |  181.5    |   16.22   |  11.19    | 165.3     |
| kurt_topeig.worst                |    1.826  |    0.5958 |   3.065   |   1.23    |



**student.h.gap.L12 vs student.z.proj.tap1**


| metric                           |   value_h |   value_z |     tau |      delta |
|:---------------------------------|----------:|----------:|--------:|-----------:|
| uniformity                       |   -0.2347 |   -2.587  | 0.09072 |    2.352   |
| variance_floor.hinge             |    0.8731 |    0.6657 | 1.312   |    0.2074  |
| offdiag_redundancy.mean_abs_corr |    0.2118 |    0.151  | 1.403   |    0.06083 |
| rankme                           |   40.1    |  717.6    | 0.05588 | -677.5     |
| effective_rank                   |   15.33   |   56.83   | 0.2698  |  -41.5     |
| alpha                            |    2.176  |    1.438  | 1.513   |    0.7379  |
| epps_pulley                      |  181.5    |  115.9    | 1.565   |   65.54    |
| kurt_topeig.worst                |    1.826  |    1.167  | 1.565   |    0.6596  |



**student.h.gap.L12 vs student.z.proj.tap2**


| metric                           |   value_h |   value_z |     tau |      delta |
|:---------------------------------|----------:|----------:|--------:|-----------:|
| uniformity                       |   -0.2347 |   -2.539  | 0.09244 |    2.304   |
| variance_floor.hinge             |    0.8731 |    0.6601 | 1.323   |    0.213   |
| offdiag_redundancy.mean_abs_corr |    0.2118 |    0.154  | 1.375   |    0.05782 |
| rankme                           |   40.1    |  757.5    | 0.05294 | -717.4     |
| effective_rank                   |   15.33   |   62.05   | 0.2471  |  -46.72    |
| alpha                            |    2.176  |    1.484  | 1.467   |    0.6923  |
| epps_pulley                      |  181.5    |  101.2    | 1.794   |   80.31    |
| kurt_topeig.worst                |    1.826  |    0.8866 | 2.06    |    0.9395  |



### Pair metrics (alignment ↓ = more view-invariant)


| manifest                     | space               | metric         |   value |   ci_lo |   ci_hi |
|:-----------------------------|:--------------------|:---------------|--------:|--------:|--------:|
| imagenette.train.v1@audit_v1 | student.h.cls       | alignment      | 0.01591 | 0.01569 | 0.01618 |
| imagenette.train.v1@audit_v1 | student.h.cls       | cos_invariance | 0.992   | 0.9919  | 0.9922  |
| imagenette.train.v1@audit_v1 | student.h.gap       | alignment      | 0.0571  | 0.05596 | 0.05824 |
| imagenette.train.v1@audit_v1 | student.h.gap       | cos_invariance | 0.9715  | 0.971   | 0.9721  |
| imagenette.train.v1@audit_v1 | student.z.embed     | alignment      | 0.09552 | 0.09418 | 0.09724 |
| imagenette.train.v1@audit_v1 | student.z.embed     | cos_invariance | 0.9522  | 0.9515  | 0.9531  |
| imagenette.train.v1@audit_v1 | student.z.proj.out  | alignment      | 0.5179  | 0.5099  | 0.5255  |
| imagenette.train.v1@audit_v1 | student.z.proj.out  | cos_invariance | 0.7411  | 0.737   | 0.7451  |
| imagenette.train.v1@audit_v1 | student.z.proj.tap1 | alignment      | 0.5155  | 0.509   | 0.5225  |
| imagenette.train.v1@audit_v1 | student.z.proj.tap1 | cos_invariance | 0.7422  | 0.7389  | 0.7451  |
| imagenette.train.v1@audit_v1 | student.z.proj.tap2 | alignment      | 0.4774  | 0.471   | 0.4841  |
| imagenette.train.v1@audit_v1 | student.z.proj.tap2 | cos_invariance | 0.7613  | 0.7584  | 0.7641  |



### Cross-space similarity (CKA is contested — read as the triple)


| space                             | metric              |   value |
|:----------------------------------|:--------------------|--------:|
| student.h.cls~student.z.embed     | cka_linear          |  0.9902 |
| student.h.cls~student.z.embed     | neighbor_jaccard    |  0.7612 |
| student.h.cls~student.z.embed     | procrustes_distance |  0.1643 |
| student.h.cls~student.z.embed     | knn_label_agreement |  0.9238 |
| student.h.cls~student.z.proj.out  | cka_linear          |  0.7578 |
| student.h.cls~student.z.proj.out  | neighbor_jaccard    |  0.4023 |
| student.h.cls~student.z.proj.out  | procrustes_distance |  0.4555 |
| student.h.cls~student.z.proj.out  | knn_label_agreement |  0.8468 |
| student.h.cls~student.z.proj.tap1 | cka_linear          |  0.9643 |
| student.h.cls~student.z.proj.tap1 | neighbor_jaccard    |  0.6509 |
| student.h.cls~student.z.proj.tap1 | procrustes_distance |  0.4165 |
| student.h.cls~student.z.proj.tap1 | knn_label_agreement |  0.9028 |
| student.h.cls~student.z.proj.tap2 | cka_linear          |  0.9186 |
| student.h.cls~student.z.proj.tap2 | neighbor_jaccard    |  0.5238 |
| student.h.cls~student.z.proj.tap2 | procrustes_distance |  0.509  |
| student.h.cls~student.z.proj.tap2 | knn_label_agreement |  0.8768 |



### Probes


| space               |   knn_v1_k20 |   knn_v1_k200 |   linear_house_v1 |   linear_l2_v1 |
|:--------------------|-------------:|--------------:|------------------:|---------------:|
| student.h.cls       |       0.7633 |        0.7276 |            0.7679 |         0.6943 |
| student.h.cls.L03   |       0.6066 |        0.5422 |            0.6033 |         0.4739 |
| student.h.cls.L06   |       0.6935 |        0.6418 |            0.7177 |         0.5880 |
| student.h.cls.L09   |       0.7462 |        0.7006 |            0.7580 |         0.6637 |
| student.h.cls.L12   |       0.7633 |        0.7276 |            0.7679 |         0.6943 |
| student.h.gap       |       0.6780 |        0.6163 |            0.7577 |         0.6150 |
| student.h.gap.L03   |       0.5875 |        0.5169 |            0.6731 |         0.5205 |
| student.h.gap.L06   |       0.6425 |        0.5620 |            0.7248 |         0.5778 |
| student.h.gap.L09   |       0.6591 |        0.5870 |            0.7468 |         0.6046 |
| student.h.gap.L12   |       0.6780 |        0.6163 |            0.7577 |         0.6150 |
| student.z.embed     |       0.7592 |        0.7289 |            0.7608 |         0.7238 |
| student.z.proj.out  |       0.7738 |        0.7707 |            0.7004 |         0.6986 |
| student.z.proj.tap1 |       0.7778 |        0.7631 |            0.7939 |         0.7743 |
| student.z.proj.tap2 |       0.7753 |        0.7623 |            0.7911 |         0.7727 |



## toy.randinit-s0.ext

### Battery (variant raw|full)

| model.space                               |   rankme |   effective_rank |   alpha |   epps_pulley |   kurt_topeig.worst |   uniformity |   offdiag_redundancy.mean_abs_corr |   variance_floor.hinge |
|:------------------------------------------|---------:|-----------------:|--------:|--------------:|--------------------:|-------------:|-----------------------------------:|-----------------------:|
| toy.randinit-s0.ext · student.h.cls       |   77.84  |            7.008 |   2.128 |         580.8 |               1.5   |      -1.802  |                             0.3716 |                 0.2539 |
| toy.randinit-s0.ext · student.h.cls.L03   |   61.28  |            5.401 |   2.233 |         683.1 |               1.507 |      -1.725  |                             0.4083 |                 0.2584 |
| toy.randinit-s0.ext · student.h.cls.L06   |   71.2   |            6.483 |   2.182 |         567.3 |               2.116 |      -1.804  |                             0.3747 |                 0.2554 |
| toy.randinit-s0.ext · student.h.cls.L09   |   75.52  |            6.723 |   2.146 |         597.5 |               2.134 |      -1.81   |                             0.3768 |                 0.247  |
| toy.randinit-s0.ext · student.h.cls.L12   |   77.84  |            7.008 |   2.128 |         580.8 |               1.5   |      -1.802  |                             0.3716 |                 0.2539 |
| toy.randinit-s0.ext · student.h.gap       |   44.28  |            3.386 |   2.34  |         620.2 |               5.489 |      -1.643  |                             0.4866 |                 0.5243 |
| toy.randinit-s0.ext · student.h.gap.L03   |   34.26  |            2.834 |   2.431 |         686.4 |              10.99  |      -1.705  |                             0.5173 |                 0.5548 |
| toy.randinit-s0.ext · student.h.gap.L06   |   39.83  |            3.091 |   2.371 |         621.3 |              11.45  |      -1.683  |                             0.5143 |                 0.5427 |
| toy.randinit-s0.ext · student.h.gap.L09   |   42.73  |            3.201 |   2.35  |         663   |               6.557 |      -1.683  |                             0.4862 |                 0.5339 |
| toy.randinit-s0.ext · student.h.gap.L12   |   44.28  |            3.386 |   2.34  |         620.2 |               5.489 |      -1.643  |                             0.4866 |                 0.5243 |
| toy.randinit-s0.ext · student.z.embed     |   69.36  |            6.403 |   2.224 |         518.8 |               1.508 |      -1.759  |                             0.3758 |                 0.539  |
| toy.randinit-s0.ext · student.z.proj.out  |    9.922 |            7.123 |   1.466 |         365.2 |               1.413 |      -0.9297 |                             0.3035 |                 0.9665 |
| toy.randinit-s0.ext · student.z.proj.tap1 |  363.2   |           10.67  |   1.812 |         348.3 |               1.546 |      -1.477  |                             0.2852 |                 0.8515 |
| toy.randinit-s0.ext · student.z.proj.tap2 |  441.6   |           15.09  |   1.674 |         246.7 |               1.542 |      -1.226  |                             0.2093 |                 0.9504 |



### Transfer ratios τ = value(h) / value(z)


**student.h.cls vs student.z.embed**


| metric                           |   value_h |   value_z |    tau |     delta |
|:---------------------------------|----------:|----------:|-------:|----------:|
| uniformity                       |   -1.802  |   -1.759  | 1.024  | -0.04301  |
| variance_floor.hinge             |    0.2539 |    0.539  | 0.471  | -0.2851   |
| offdiag_redundancy.mean_abs_corr |    0.3716 |    0.3758 | 0.9888 | -0.00422  |
| rankme                           |   77.84   |   69.36   | 1.122  |  8.474    |
| effective_rank                   |    7.008  |    6.403  | 1.094  |  0.6045   |
| alpha                            |    2.128  |    2.224  | 0.9567 | -0.09638  |
| epps_pulley                      |  580.8    |  518.8    | 1.12   | 62.02     |
| kurt_topeig.worst                |    1.5    |    1.508  | 0.9943 | -0.008641 |



**student.h.cls vs student.z.proj.out**


| metric                           |   value_h |   value_z |    tau |     delta |
|:---------------------------------|----------:|----------:|-------:|----------:|
| uniformity                       |   -1.802  |   -0.9297 | 1.938  |  -0.8722  |
| variance_floor.hinge             |    0.2539 |    0.9665 | 0.2627 |  -0.7126  |
| offdiag_redundancy.mean_abs_corr |    0.3716 |    0.3035 | 1.225  |   0.06813 |
| rankme                           |   77.84   |    9.922  | 7.844  |  67.91    |
| effective_rank                   |    7.008  |    7.123  | 0.9838 |  -0.1151  |
| alpha                            |    2.128  |    1.466  | 1.452  |   0.6619  |
| epps_pulley                      |  580.8    |  365.2    | 1.59   | 215.6     |
| kurt_topeig.worst                |    1.5    |    1.413  | 1.061  |   0.08651 |



**student.h.cls vs student.z.proj.tap1**


| metric                           |   value_h |   value_z |    tau |      delta |
|:---------------------------------|----------:|----------:|-------:|-----------:|
| uniformity                       |   -1.802  |   -1.477  | 1.22   |   -0.3245  |
| variance_floor.hinge             |    0.2539 |    0.8515 | 0.2982 |   -0.5976  |
| offdiag_redundancy.mean_abs_corr |    0.3716 |    0.2852 | 1.303  |    0.08635 |
| rankme                           |   77.84   |  363.2    | 0.2143 | -285.3     |
| effective_rank                   |    7.008  |   10.67   | 0.6569 |   -3.66    |
| alpha                            |    2.128  |    1.812  | 1.174  |    0.3155  |
| epps_pulley                      |  580.8    |  348.3    | 1.668  |  232.5     |
| kurt_topeig.worst                |    1.5    |    1.546  | 0.9697 |   -0.04683 |



**student.h.cls vs student.z.proj.tap2**


| metric                           |   value_h |   value_z |    tau |     delta |
|:---------------------------------|----------:|----------:|-------:|----------:|
| uniformity                       |   -1.802  |   -1.226  | 1.47   |   -0.5763 |
| variance_floor.hinge             |    0.2539 |    0.9504 | 0.2672 |   -0.6965 |
| offdiag_redundancy.mean_abs_corr |    0.3716 |    0.2093 | 1.776  |    0.1623 |
| rankme                           |   77.84   |  441.6    | 0.1763 | -363.7    |
| effective_rank                   |    7.008  |   15.09   | 0.4645 |   -8.077  |
| alpha                            |    2.128  |    1.674  | 1.271  |    0.454  |
| epps_pulley                      |  580.8    |  246.7    | 2.355  |  334.2    |
| kurt_topeig.worst                |    1.5    |    1.542  | 0.9724 |   -0.0426 |



**student.h.cls.L03 vs student.z.embed**


| metric                           |   value_h |   value_z |    tau |      delta |
|:---------------------------------|----------:|----------:|-------:|-----------:|
| uniformity                       |   -1.725  |   -1.759  | 0.9805 |   0.03437  |
| variance_floor.hinge             |    0.2584 |    0.539  | 0.4794 |  -0.2806   |
| offdiag_redundancy.mean_abs_corr |    0.4083 |    0.3758 | 1.086  |   0.03246  |
| rankme                           |   61.28   |   69.36   | 0.8834 |  -8.087    |
| effective_rank                   |    5.401  |    6.403  | 0.8434 |  -1.003    |
| alpha                            |    2.233  |    2.224  | 1.004  |   0.009235 |
| epps_pulley                      |  683.1    |  518.8    | 1.317  | 164.3      |
| kurt_topeig.worst                |    1.507  |    1.508  | 0.9992 |  -0.001174 |



**student.h.cls.L03 vs student.z.proj.out**


| metric                           |   value_h |   value_z |    tau |     delta |
|:---------------------------------|----------:|----------:|-------:|----------:|
| uniformity                       |   -1.725  |   -0.9297 | 1.855  |  -0.7948  |
| variance_floor.hinge             |    0.2584 |    0.9665 | 0.2674 |  -0.7081  |
| offdiag_redundancy.mean_abs_corr |    0.4083 |    0.3035 | 1.345  |   0.1048  |
| rankme                           |   61.28   |    9.922  | 6.175  |  51.35    |
| effective_rank                   |    5.401  |    7.123  | 0.7582 |  -1.722   |
| alpha                            |    2.233  |    1.466  | 1.524  |   0.7675  |
| epps_pulley                      |  683.1    |  365.2    | 1.871  | 317.9     |
| kurt_topeig.worst                |    1.507  |    1.413  | 1.067  |   0.09398 |



**student.h.cls.L03 vs student.z.proj.tap1**


| metric                           |   value_h |   value_z |    tau |      delta |
|:---------------------------------|----------:|----------:|-------:|-----------:|
| uniformity                       |   -1.725  |   -1.477  | 1.167  |   -0.2471  |
| variance_floor.hinge             |    0.2584 |    0.8515 | 0.3034 |   -0.5931  |
| offdiag_redundancy.mean_abs_corr |    0.4083 |    0.2852 | 1.431  |    0.123   |
| rankme                           |   61.28   |  363.2    | 0.1687 | -301.9     |
| effective_rank                   |    5.401  |   10.67   | 0.5062 |   -5.267   |
| alpha                            |    2.233  |    1.812  | 1.232  |    0.4211  |
| epps_pulley                      |  683.1    |  348.3    | 1.961  |  334.8     |
| kurt_topeig.worst                |    1.507  |    1.546  | 0.9745 |   -0.03936 |



**student.h.cls.L03 vs student.z.proj.tap2**


| metric                           |   value_h |   value_z |    tau |      delta |
|:---------------------------------|----------:|----------:|-------:|-----------:|
| uniformity                       |   -1.725  |   -1.226  | 1.407  |   -0.4989  |
| variance_floor.hinge             |    0.2584 |    0.9504 | 0.2719 |   -0.692   |
| offdiag_redundancy.mean_abs_corr |    0.4083 |    0.2093 | 1.951  |    0.199   |
| rankme                           |   61.28   |  441.6    | 0.1388 | -380.3     |
| effective_rank                   |    5.401  |   15.09   | 0.358  |   -9.685   |
| alpha                            |    2.233  |    1.674  | 1.334  |    0.5596  |
| epps_pulley                      |  683.1    |  246.7    | 2.769  |  436.4     |
| kurt_topeig.worst                |    1.507  |    1.542  | 0.9772 |   -0.03513 |



**student.h.cls.L06 vs student.z.embed**


| metric                           |   value_h |   value_z |    tau |     delta |
|:---------------------------------|----------:|----------:|-------:|----------:|
| uniformity                       |   -1.804  |   -1.759  | 1.026  | -0.04514  |
| variance_floor.hinge             |    0.2554 |    0.539  | 0.4738 | -0.2836   |
| offdiag_redundancy.mean_abs_corr |    0.3747 |    0.3758 | 0.9971 | -0.001078 |
| rankme                           |   71.2    |   69.36   | 1.026  |  1.833    |
| effective_rank                   |    6.483  |    6.403  | 1.013  |  0.0803   |
| alpha                            |    2.182  |    2.224  | 0.981  | -0.0423   |
| epps_pulley                      |  567.3    |  518.8    | 1.093  | 48.51     |
| kurt_topeig.worst                |    2.116  |    1.508  | 1.403  |  0.6081   |



**student.h.cls.L06 vs student.z.proj.out**


| metric                           |   value_h |   value_z |    tau |     delta |
|:---------------------------------|----------:|----------:|-------:|----------:|
| uniformity                       |   -1.804  |   -0.9297 | 1.94   |  -0.8743  |
| variance_floor.hinge             |    0.2554 |    0.9665 | 0.2643 |  -0.7111  |
| offdiag_redundancy.mean_abs_corr |    0.3747 |    0.3035 | 1.235  |   0.07127 |
| rankme                           |   71.2    |    9.922  | 7.175  |  61.27    |
| effective_rank                   |    6.483  |    7.123  | 0.9102 |  -0.6394  |
| alpha                            |    2.182  |    1.466  | 1.488  |   0.7159  |
| epps_pulley                      |  567.3    |  365.2    | 1.553  | 202.1     |
| kurt_topeig.worst                |    2.116  |    1.413  | 1.498  |   0.7032  |



**student.h.cls.L06 vs student.z.proj.tap1**


| metric                           |   value_h |   value_z |    tau |      delta |
|:---------------------------------|----------:|----------:|-------:|-----------:|
| uniformity                       |   -1.804  |   -1.477  | 1.221  |   -0.3266  |
| variance_floor.hinge             |    0.2554 |    0.8515 | 0.3    |   -0.5961  |
| offdiag_redundancy.mean_abs_corr |    0.3747 |    0.2852 | 1.314  |    0.08949 |
| rankme                           |   71.2    |  363.2    | 0.196  | -292       |
| effective_rank                   |    6.483  |   10.67   | 0.6077 |   -4.185   |
| alpha                            |    2.182  |    1.812  | 1.204  |    0.3696  |
| epps_pulley                      |  567.3    |  348.3    | 1.629  |  219       |
| kurt_topeig.worst                |    2.116  |    1.546  | 1.369  |    0.5699  |



**student.h.cls.L06 vs student.z.proj.tap2**


| metric                           |   value_h |   value_z |    tau |     delta |
|:---------------------------------|----------:|----------:|-------:|----------:|
| uniformity                       |   -1.804  |   -1.226  | 1.472  |   -0.5785 |
| variance_floor.hinge             |    0.2554 |    0.9504 | 0.2688 |   -0.695  |
| offdiag_redundancy.mean_abs_corr |    0.3747 |    0.2093 | 1.791  |    0.1655 |
| rankme                           |   71.2    |  441.6    | 0.1612 | -370.4    |
| effective_rank                   |    6.483  |   15.09   | 0.4298 |   -8.602  |
| alpha                            |    2.182  |    1.674  | 1.304  |    0.5081 |
| epps_pulley                      |  567.3    |  246.7    | 2.3    |  320.7    |
| kurt_topeig.worst                |    2.116  |    1.542  | 1.372  |    0.5741 |



**student.h.cls.L09 vs student.z.embed**


| metric                           |   value_h |   value_z |    tau |      delta |
|:---------------------------------|----------:|----------:|-------:|-----------:|
| uniformity                       |   -1.81   |   -1.759  | 1.029  | -0.05084   |
| variance_floor.hinge             |    0.247  |    0.539  | 0.4583 | -0.292     |
| offdiag_redundancy.mean_abs_corr |    0.3768 |    0.3758 | 1.003  |  0.0009885 |
| rankme                           |   75.52   |   69.36   | 1.089  |  6.16      |
| effective_rank                   |    6.723  |    6.403  | 1.05   |  0.3197    |
| alpha                            |    2.146  |    2.224  | 0.9651 | -0.07764   |
| epps_pulley                      |  597.5    |  518.8    | 1.152  | 78.72      |
| kurt_topeig.worst                |    2.134  |    1.508  | 1.415  |  0.6261    |



**student.h.cls.L09 vs student.z.proj.out**


| metric                           |   value_h |   value_z |    tau |     delta |
|:---------------------------------|----------:|----------:|-------:|----------:|
| uniformity                       |   -1.81   |   -0.9297 | 1.947  |  -0.88    |
| variance_floor.hinge             |    0.247  |    0.9665 | 0.2556 |  -0.7195  |
| offdiag_redundancy.mean_abs_corr |    0.3768 |    0.3035 | 1.242  |   0.07334 |
| rankme                           |   75.52   |    9.922  | 7.611  |  65.6     |
| effective_rank                   |    6.723  |    7.123  | 0.9438 |  -0.4     |
| alpha                            |    2.146  |    1.466  | 1.464  |   0.6806  |
| epps_pulley                      |  597.5    |  365.2    | 1.636  | 232.3     |
| kurt_topeig.worst                |    2.134  |    1.413  | 1.51   |   0.7213  |



**student.h.cls.L09 vs student.z.proj.tap1**


| metric                           |   value_h |   value_z |    tau |      delta |
|:---------------------------------|----------:|----------:|-------:|-----------:|
| uniformity                       |   -1.81   |   -1.477  | 1.225  |   -0.3323  |
| variance_floor.hinge             |    0.247  |    0.8515 | 0.2901 |   -0.6045  |
| offdiag_redundancy.mean_abs_corr |    0.3768 |    0.2852 | 1.321  |    0.09156 |
| rankme                           |   75.52   |  363.2    | 0.208  | -287.6     |
| effective_rank                   |    6.723  |   10.67   | 0.6302 |   -3.945   |
| alpha                            |    2.146  |    1.812  | 1.184  |    0.3342  |
| epps_pulley                      |  597.5    |  348.3    | 1.715  |  249.2     |
| kurt_topeig.worst                |    2.134  |    1.546  | 1.38   |    0.588   |



**student.h.cls.L09 vs student.z.proj.tap2**


| metric                           |   value_h |   value_z |    tau |     delta |
|:---------------------------------|----------:|----------:|-------:|----------:|
| uniformity                       |   -1.81   |   -1.226  | 1.477  |   -0.5841 |
| variance_floor.hinge             |    0.247  |    0.9504 | 0.2599 |   -0.7034 |
| offdiag_redundancy.mean_abs_corr |    0.3768 |    0.2093 | 1.801  |    0.1675 |
| rankme                           |   75.52   |  441.6    | 0.171  | -366.1    |
| effective_rank                   |    6.723  |   15.09   | 0.4457 |   -8.362  |
| alpha                            |    2.146  |    1.674  | 1.282  |    0.4727 |
| epps_pulley                      |  597.5    |  246.7    | 2.422  |  350.9    |
| kurt_topeig.worst                |    2.134  |    1.542  | 1.384  |    0.5922 |



**student.h.cls.L12 vs student.z.embed**


| metric                           |   value_h |   value_z |    tau |     delta |
|:---------------------------------|----------:|----------:|-------:|----------:|
| uniformity                       |   -1.802  |   -1.759  | 1.024  | -0.04301  |
| variance_floor.hinge             |    0.2539 |    0.539  | 0.471  | -0.2851   |
| offdiag_redundancy.mean_abs_corr |    0.3716 |    0.3758 | 0.9888 | -0.00422  |
| rankme                           |   77.84   |   69.36   | 1.122  |  8.474    |
| effective_rank                   |    7.008  |    6.403  | 1.094  |  0.6045   |
| alpha                            |    2.128  |    2.224  | 0.9567 | -0.09638  |
| epps_pulley                      |  580.8    |  518.8    | 1.12   | 62.02     |
| kurt_topeig.worst                |    1.5    |    1.508  | 0.9943 | -0.008641 |



**student.h.cls.L12 vs student.z.proj.out**


| metric                           |   value_h |   value_z |    tau |     delta |
|:---------------------------------|----------:|----------:|-------:|----------:|
| uniformity                       |   -1.802  |   -0.9297 | 1.938  |  -0.8722  |
| variance_floor.hinge             |    0.2539 |    0.9665 | 0.2627 |  -0.7126  |
| offdiag_redundancy.mean_abs_corr |    0.3716 |    0.3035 | 1.225  |   0.06813 |
| rankme                           |   77.84   |    9.922  | 7.844  |  67.91    |
| effective_rank                   |    7.008  |    7.123  | 0.9838 |  -0.1151  |
| alpha                            |    2.128  |    1.466  | 1.452  |   0.6619  |
| epps_pulley                      |  580.8    |  365.2    | 1.59   | 215.6     |
| kurt_topeig.worst                |    1.5    |    1.413  | 1.061  |   0.08651 |



**student.h.cls.L12 vs student.z.proj.tap1**


| metric                           |   value_h |   value_z |    tau |      delta |
|:---------------------------------|----------:|----------:|-------:|-----------:|
| uniformity                       |   -1.802  |   -1.477  | 1.22   |   -0.3245  |
| variance_floor.hinge             |    0.2539 |    0.8515 | 0.2982 |   -0.5976  |
| offdiag_redundancy.mean_abs_corr |    0.3716 |    0.2852 | 1.303  |    0.08635 |
| rankme                           |   77.84   |  363.2    | 0.2143 | -285.3     |
| effective_rank                   |    7.008  |   10.67   | 0.6569 |   -3.66    |
| alpha                            |    2.128  |    1.812  | 1.174  |    0.3155  |
| epps_pulley                      |  580.8    |  348.3    | 1.668  |  232.5     |
| kurt_topeig.worst                |    1.5    |    1.546  | 0.9697 |   -0.04683 |



**student.h.cls.L12 vs student.z.proj.tap2**


| metric                           |   value_h |   value_z |    tau |     delta |
|:---------------------------------|----------:|----------:|-------:|----------:|
| uniformity                       |   -1.802  |   -1.226  | 1.47   |   -0.5763 |
| variance_floor.hinge             |    0.2539 |    0.9504 | 0.2672 |   -0.6965 |
| offdiag_redundancy.mean_abs_corr |    0.3716 |    0.2093 | 1.776  |    0.1623 |
| rankme                           |   77.84   |  441.6    | 0.1763 | -363.7    |
| effective_rank                   |    7.008  |   15.09   | 0.4645 |   -8.077  |
| alpha                            |    2.128  |    1.674  | 1.271  |    0.454  |
| epps_pulley                      |  580.8    |  246.7    | 2.355  |  334.2    |
| kurt_topeig.worst                |    1.5    |    1.542  | 0.9724 |   -0.0426 |



**student.h.gap vs student.z.embed**


| metric                           |   value_h |   value_z |    tau |     delta |
|:---------------------------------|----------:|----------:|-------:|----------:|
| uniformity                       |   -1.643  |   -1.759  | 0.9342 |   0.1158  |
| variance_floor.hinge             |    0.5243 |    0.539  | 0.9726 |  -0.01479 |
| offdiag_redundancy.mean_abs_corr |    0.4866 |    0.3758 | 1.295  |   0.1108  |
| rankme                           |   44.28   |   69.36   | 0.6384 | -25.08    |
| effective_rank                   |    3.386  |    6.403  | 0.5288 |  -3.017   |
| alpha                            |    2.34   |    2.224  | 1.052  |   0.1165  |
| epps_pulley                      |  620.2    |  518.8    | 1.196  | 101.4     |
| kurt_topeig.worst                |    5.489  |    1.508  | 3.64   |   3.981   |



**student.h.gap vs student.z.proj.out**


| metric                           |   value_h |   value_z |    tau |    delta |
|:---------------------------------|----------:|----------:|-------:|---------:|
| uniformity                       |   -1.643  |   -0.9297 | 1.767  |  -0.7134 |
| variance_floor.hinge             |    0.5243 |    0.9665 | 0.5424 |  -0.4422 |
| offdiag_redundancy.mean_abs_corr |    0.4866 |    0.3035 | 1.604  |   0.1831 |
| rankme                           |   44.28   |    9.922  | 4.463  |  34.36   |
| effective_rank                   |    3.386  |    7.123  | 0.4754 |  -3.737  |
| alpha                            |    2.34   |    1.466  | 1.597  |   0.8747 |
| epps_pulley                      |  620.2    |  365.2    | 1.698  | 255      |
| kurt_topeig.worst                |    5.489  |    1.413  | 3.885  |   4.076  |



**student.h.gap vs student.z.proj.tap1**


| metric                           |   value_h |   value_z |    tau |     delta |
|:---------------------------------|----------:|----------:|-------:|----------:|
| uniformity                       |   -1.643  |   -1.477  | 1.112  |   -0.1657 |
| variance_floor.hinge             |    0.5243 |    0.8515 | 0.6157 |   -0.3273 |
| offdiag_redundancy.mean_abs_corr |    0.4866 |    0.2852 | 1.706  |    0.2014 |
| rankme                           |   44.28   |  363.2    | 0.1219 | -318.9    |
| effective_rank                   |    3.386  |   10.67   | 0.3174 |   -7.282  |
| alpha                            |    2.34   |    1.812  | 1.292  |    0.5284 |
| epps_pulley                      |  620.2    |  348.3    | 1.781  |  271.9    |
| kurt_topeig.worst                |    5.489  |    1.546  | 3.55   |    3.943  |



**student.h.gap vs student.z.proj.tap2**


| metric                           |   value_h |   value_z |    tau |     delta |
|:---------------------------------|----------:|----------:|-------:|----------:|
| uniformity                       |   -1.643  |   -1.226  | 1.341  |   -0.4175 |
| variance_floor.hinge             |    0.5243 |    0.9504 | 0.5516 |   -0.4262 |
| offdiag_redundancy.mean_abs_corr |    0.4866 |    0.2093 | 2.325  |    0.2773 |
| rankme                           |   44.28   |  441.6    | 0.1003 | -397.3    |
| effective_rank                   |    3.386  |   15.09   | 0.2245 |  -11.7    |
| alpha                            |    2.34   |    1.674  | 1.398  |    0.6669 |
| epps_pulley                      |  620.2    |  246.7    | 2.514  |  373.6    |
| kurt_topeig.worst                |    5.489  |    1.542  | 3.559  |    3.947  |



**student.h.gap.L03 vs student.z.embed**


| metric                           |   value_h |   value_z |    tau |     delta |
|:---------------------------------|----------:|----------:|-------:|----------:|
| uniformity                       |   -1.705  |   -1.759  | 0.9695 |   0.05373 |
| variance_floor.hinge             |    0.5548 |    0.539  | 1.029  |   0.01577 |
| offdiag_redundancy.mean_abs_corr |    0.5173 |    0.3758 | 1.377  |   0.1415  |
| rankme                           |   34.26   |   69.36   | 0.4939 | -35.1     |
| effective_rank                   |    2.834  |    6.403  | 0.4426 |  -3.569   |
| alpha                            |    2.431  |    2.224  | 1.093  |   0.2069  |
| epps_pulley                      |  686.4    |  518.8    | 1.323  | 167.6     |
| kurt_topeig.worst                |   10.99   |    1.508  | 7.289  |   9.485   |



**student.h.gap.L03 vs student.z.proj.out**


| metric                           |   value_h |   value_z |    tau |    delta |
|:---------------------------------|----------:|----------:|-------:|---------:|
| uniformity                       |   -1.705  |   -0.9297 | 1.834  |  -0.7755 |
| variance_floor.hinge             |    0.5548 |    0.9665 | 0.5741 |  -0.4117 |
| offdiag_redundancy.mean_abs_corr |    0.5173 |    0.3035 | 1.705  |   0.2139 |
| rankme                           |   34.26   |    9.922  | 3.453  |  24.34   |
| effective_rank                   |    2.834  |    7.123  | 0.3978 |  -4.289  |
| alpha                            |    2.431  |    1.466  | 1.658  |   0.9651 |
| epps_pulley                      |  686.4    |  365.2    | 1.88   | 321.2    |
| kurt_topeig.worst                |   10.99   |    1.413  | 7.779  |   9.58   |



**student.h.gap.L03 vs student.z.proj.tap1**


| metric                           |   value_h |   value_z |     tau |     delta |
|:---------------------------------|----------:|----------:|--------:|----------:|
| uniformity                       |   -1.705  |   -1.477  | 1.154   |   -0.2278 |
| variance_floor.hinge             |    0.5548 |    0.8515 | 0.6515  |   -0.2967 |
| offdiag_redundancy.mean_abs_corr |    0.5173 |    0.2852 | 1.814   |    0.2321 |
| rankme                           |   34.26   |  363.2    | 0.09434 | -328.9    |
| effective_rank                   |    2.834  |   10.67   | 0.2656  |   -7.834  |
| alpha                            |    2.431  |    1.812  | 1.341   |    0.6187 |
| epps_pulley                      |  686.4    |  348.3    | 1.971   |  338.1    |
| kurt_topeig.worst                |   10.99   |    1.546  | 7.109   |    9.447  |



**student.h.gap.L03 vs student.z.proj.tap2**


| metric                           |   value_h |   value_z |     tau |     delta |
|:---------------------------------|----------:|----------:|--------:|----------:|
| uniformity                       |   -1.705  |   -1.226  | 1.391   |   -0.4796 |
| variance_floor.hinge             |    0.5548 |    0.9504 | 0.5838  |   -0.3956 |
| offdiag_redundancy.mean_abs_corr |    0.5173 |    0.2093 | 2.472   |    0.3081 |
| rankme                           |   34.26   |  441.6    | 0.07759 | -407.3    |
| effective_rank                   |    2.834  |   15.09   | 0.1879  |  -12.25   |
| alpha                            |    2.431  |    1.674  | 1.452   |    0.7572 |
| epps_pulley                      |  686.4    |  246.7    | 2.783   |  439.7    |
| kurt_topeig.worst                |   10.99   |    1.542  | 7.128   |    9.451  |



**student.h.gap.L06 vs student.z.embed**


| metric                           |   value_h |   value_z |    tau |      delta |
|:---------------------------------|----------:|----------:|-------:|-----------:|
| uniformity                       |   -1.683  |   -1.759  | 0.9571 |   0.0754   |
| variance_floor.hinge             |    0.5427 |    0.539  | 1.007  |   0.003674 |
| offdiag_redundancy.mean_abs_corr |    0.5143 |    0.3758 | 1.369  |   0.1385   |
| rankme                           |   39.83   |   69.36   | 0.5742 | -29.53     |
| effective_rank                   |    3.091  |    6.403  | 0.4827 |  -3.313    |
| alpha                            |    2.371  |    2.224  | 1.066  |   0.1473   |
| epps_pulley                      |  621.3    |  518.8    | 1.198  | 102.5      |
| kurt_topeig.worst                |   11.45   |    1.508  | 7.591  |   9.941    |



**student.h.gap.L06 vs student.z.proj.out**


| metric                           |   value_h |   value_z |    tau |    delta |
|:---------------------------------|----------:|----------:|-------:|---------:|
| uniformity                       |   -1.683  |   -0.9297 | 1.811  |  -0.7538 |
| variance_floor.hinge             |    0.5427 |    0.9665 | 0.5615 |  -0.4238 |
| offdiag_redundancy.mean_abs_corr |    0.5143 |    0.3035 | 1.695  |   0.2108 |
| rankme                           |   39.83   |    9.922  | 4.014  |  29.91   |
| effective_rank                   |    3.091  |    7.123  | 0.4339 |  -4.032  |
| alpha                            |    2.371  |    1.466  | 1.618  |   0.9056 |
| epps_pulley                      |  621.3    |  365.2    | 1.701  | 256.1    |
| kurt_topeig.worst                |   11.45   |    1.413  | 8.102  |  10.04   |



**student.h.gap.L06 vs student.z.proj.tap1**


| metric                           |   value_h |   value_z |    tau |     delta |
|:---------------------------------|----------:|----------:|-------:|----------:|
| uniformity                       |   -1.683  |   -1.477  | 1.14   |   -0.2061 |
| variance_floor.hinge             |    0.5427 |    0.8515 | 0.6373 |   -0.3088 |
| offdiag_redundancy.mean_abs_corr |    0.5143 |    0.2852 | 1.803  |    0.2291 |
| rankme                           |   39.83   |  363.2    | 0.1097 | -323.3    |
| effective_rank                   |    3.091  |   10.67   | 0.2897 |   -7.577  |
| alpha                            |    2.371  |    1.812  | 1.309  |    0.5592 |
| epps_pulley                      |  621.3    |  348.3    | 1.784  |  273      |
| kurt_topeig.worst                |   11.45   |    1.546  | 7.404  |    9.903  |



**student.h.gap.L06 vs student.z.proj.tap2**


| metric                           |   value_h |   value_z |    tau |     delta |
|:---------------------------------|----------:|----------:|-------:|----------:|
| uniformity                       |   -1.683  |   -1.226  | 1.374  |   -0.4579 |
| variance_floor.hinge             |    0.5427 |    0.9504 | 0.571  |   -0.4077 |
| offdiag_redundancy.mean_abs_corr |    0.5143 |    0.2093 | 2.458  |    0.305  |
| rankme                           |   39.83   |  441.6    | 0.0902 | -401.8    |
| effective_rank                   |    3.091  |   15.09   | 0.2049 |  -11.99   |
| alpha                            |    2.371  |    1.674  | 1.417  |    0.6977 |
| epps_pulley                      |  621.3    |  246.7    | 2.519  |  374.6    |
| kurt_topeig.worst                |   11.45   |    1.542  | 7.424  |    9.907  |



**student.h.gap.L09 vs student.z.embed**


| metric                           |   value_h |   value_z |    tau |      delta |
|:---------------------------------|----------:|----------:|-------:|-----------:|
| uniformity                       |   -1.683  |   -1.759  | 0.9571 |   0.07552  |
| variance_floor.hinge             |    0.5339 |    0.539  | 0.9905 |  -0.005142 |
| offdiag_redundancy.mean_abs_corr |    0.4862 |    0.3758 | 1.294  |   0.1104   |
| rankme                           |   42.73   |   69.36   | 0.616  | -26.63     |
| effective_rank                   |    3.201  |    6.403  | 0.5    |  -3.202    |
| alpha                            |    2.35   |    2.224  | 1.057  |   0.1264   |
| epps_pulley                      |  663      |  518.8    | 1.278  | 144.2      |
| kurt_topeig.worst                |    6.557  |    1.508  | 4.347  |   5.049    |



**student.h.gap.L09 vs student.z.proj.out**


| metric                           |   value_h |   value_z |    tau |    delta |
|:---------------------------------|----------:|----------:|-------:|---------:|
| uniformity                       |   -1.683  |   -0.9297 | 1.811  |  -0.7537 |
| variance_floor.hinge             |    0.5339 |    0.9665 | 0.5524 |  -0.4326 |
| offdiag_redundancy.mean_abs_corr |    0.4862 |    0.3035 | 1.602  |   0.1828 |
| rankme                           |   42.73   |    9.922  | 4.306  |  32.81   |
| effective_rank                   |    3.201  |    7.123  | 0.4494 |  -3.922  |
| alpha                            |    2.35   |    1.466  | 1.604  |   0.8847 |
| epps_pulley                      |  663      |  365.2    | 1.815  | 297.8    |
| kurt_topeig.worst                |    6.557  |    1.413  | 4.64   |   5.144  |



**student.h.gap.L09 vs student.z.proj.tap1**


| metric                           |   value_h |   value_z |    tau |     delta |
|:---------------------------------|----------:|----------:|-------:|----------:|
| uniformity                       |   -1.683  |   -1.477  | 1.139  |   -0.206  |
| variance_floor.hinge             |    0.5339 |    0.8515 | 0.627  |   -0.3176 |
| offdiag_redundancy.mean_abs_corr |    0.4862 |    0.2852 | 1.705  |    0.201  |
| rankme                           |   42.73   |  363.2    | 0.1177 | -320.4    |
| effective_rank                   |    3.201  |   10.67   | 0.3001 |   -7.467  |
| alpha                            |    2.35   |    1.812  | 1.297  |    0.5383 |
| epps_pulley                      |  663      |  348.3    | 1.903  |  314.6    |
| kurt_topeig.worst                |    6.557  |    1.546  | 4.24   |    5.011  |



**student.h.gap.L09 vs student.z.proj.tap2**


| metric                           |   value_h |   value_z |     tau |     delta |
|:---------------------------------|----------:|----------:|--------:|----------:|
| uniformity                       |   -1.683  |   -1.226  | 1.374   |   -0.4578 |
| variance_floor.hinge             |    0.5339 |    0.9504 | 0.5618  |   -0.4165 |
| offdiag_redundancy.mean_abs_corr |    0.4862 |    0.2093 | 2.323   |    0.2769 |
| rankme                           |   42.73   |  441.6    | 0.09677 | -398.8    |
| effective_rank                   |    3.201  |   15.09   | 0.2122  |  -11.88   |
| alpha                            |    2.35   |    1.674  | 1.404   |    0.6768 |
| epps_pulley                      |  663      |  246.7    | 2.688   |  416.3    |
| kurt_topeig.worst                |    6.557  |    1.542  | 4.252   |    5.015  |



**student.h.gap.L12 vs student.z.embed**


| metric                           |   value_h |   value_z |    tau |     delta |
|:---------------------------------|----------:|----------:|-------:|----------:|
| uniformity                       |   -1.643  |   -1.759  | 0.9342 |   0.1158  |
| variance_floor.hinge             |    0.5243 |    0.539  | 0.9726 |  -0.01479 |
| offdiag_redundancy.mean_abs_corr |    0.4866 |    0.3758 | 1.295  |   0.1108  |
| rankme                           |   44.28   |   69.36   | 0.6384 | -25.08    |
| effective_rank                   |    3.386  |    6.403  | 0.5288 |  -3.017   |
| alpha                            |    2.34   |    2.224  | 1.052  |   0.1165  |
| epps_pulley                      |  620.2    |  518.8    | 1.196  | 101.4     |
| kurt_topeig.worst                |    5.489  |    1.508  | 3.64   |   3.981   |



**student.h.gap.L12 vs student.z.proj.out**


| metric                           |   value_h |   value_z |    tau |    delta |
|:---------------------------------|----------:|----------:|-------:|---------:|
| uniformity                       |   -1.643  |   -0.9297 | 1.767  |  -0.7134 |
| variance_floor.hinge             |    0.5243 |    0.9665 | 0.5424 |  -0.4422 |
| offdiag_redundancy.mean_abs_corr |    0.4866 |    0.3035 | 1.604  |   0.1831 |
| rankme                           |   44.28   |    9.922  | 4.463  |  34.36   |
| effective_rank                   |    3.386  |    7.123  | 0.4754 |  -3.737  |
| alpha                            |    2.34   |    1.466  | 1.597  |   0.8747 |
| epps_pulley                      |  620.2    |  365.2    | 1.698  | 255      |
| kurt_topeig.worst                |    5.489  |    1.413  | 3.885  |   4.076  |



**student.h.gap.L12 vs student.z.proj.tap1**


| metric                           |   value_h |   value_z |    tau |     delta |
|:---------------------------------|----------:|----------:|-------:|----------:|
| uniformity                       |   -1.643  |   -1.477  | 1.112  |   -0.1657 |
| variance_floor.hinge             |    0.5243 |    0.8515 | 0.6157 |   -0.3273 |
| offdiag_redundancy.mean_abs_corr |    0.4866 |    0.2852 | 1.706  |    0.2014 |
| rankme                           |   44.28   |  363.2    | 0.1219 | -318.9    |
| effective_rank                   |    3.386  |   10.67   | 0.3174 |   -7.282  |
| alpha                            |    2.34   |    1.812  | 1.292  |    0.5284 |
| epps_pulley                      |  620.2    |  348.3    | 1.781  |  271.9    |
| kurt_topeig.worst                |    5.489  |    1.546  | 3.55   |    3.943  |



**student.h.gap.L12 vs student.z.proj.tap2**


| metric                           |   value_h |   value_z |    tau |     delta |
|:---------------------------------|----------:|----------:|-------:|----------:|
| uniformity                       |   -1.643  |   -1.226  | 1.341  |   -0.4175 |
| variance_floor.hinge             |    0.5243 |    0.9504 | 0.5516 |   -0.4262 |
| offdiag_redundancy.mean_abs_corr |    0.4866 |    0.2093 | 2.325  |    0.2773 |
| rankme                           |   44.28   |  441.6    | 0.1003 | -397.3    |
| effective_rank                   |    3.386  |   15.09   | 0.2245 |  -11.7    |
| alpha                            |    2.34   |    1.674  | 1.398  |    0.6669 |
| epps_pulley                      |  620.2    |  246.7    | 2.514  |  373.6    |
| kurt_topeig.worst                |    5.489  |    1.542  | 3.559  |    3.947  |



### Pair metrics (alignment ↓ = more view-invariant)


| manifest                     | space               | metric         |   value |   ci_lo |   ci_hi |
|:-----------------------------|:--------------------|:---------------|--------:|--------:|--------:|
| imagenette.train.v1@audit_v1 | student.h.cls       | alignment      |  1.162  |  1.14   |  1.181  |
| imagenette.train.v1@audit_v1 | student.h.cls       | cos_invariance |  0.4192 |  0.4098 |  0.4293 |
| imagenette.train.v1@audit_v1 | student.h.gap       | alignment      |  1.202  |  1.177  |  1.225  |
| imagenette.train.v1@audit_v1 | student.h.gap       | cos_invariance |  0.3989 |  0.3879 |  0.4107 |
| imagenette.train.v1@audit_v1 | student.z.embed     | alignment      |  1.178  |  1.158  |  1.198  |
| imagenette.train.v1@audit_v1 | student.z.embed     | cos_invariance |  0.4108 |  0.4013 |  0.421  |
| imagenette.train.v1@audit_v1 | student.z.proj.out  | alignment      |  0.4092 |  0.4038 |  0.4161 |
| imagenette.train.v1@audit_v1 | student.z.proj.out  | cos_invariance |  0.7954 |  0.7919 |  0.7987 |
| imagenette.train.v1@audit_v1 | student.z.proj.tap1 | alignment      |  0.7952 |  0.7827 |  0.8072 |
| imagenette.train.v1@audit_v1 | student.z.proj.tap1 | cos_invariance |  0.6024 |  0.5968 |  0.6094 |
| imagenette.train.v1@audit_v1 | student.z.proj.tap2 | alignment      |  0.5895 |  0.581  |  0.5976 |
| imagenette.train.v1@audit_v1 | student.z.proj.tap2 | cos_invariance |  0.7053 |  0.7016 |  0.7101 |



### Cross-space similarity (CKA is contested — read as the triple)


| space                             | metric              |   value |
|:----------------------------------|:--------------------|--------:|
| student.h.cls~student.z.embed     | cka_linear          | 0.994   |
| student.h.cls~student.z.embed     | neighbor_jaccard    | 0.8369  |
| student.h.cls~student.z.embed     | procrustes_distance | 0.09562 |
| student.h.cls~student.z.embed     | knn_label_agreement | 0.8524  |
| student.h.cls~student.z.proj.out  | cka_linear          | 0.8889  |
| student.h.cls~student.z.proj.out  | neighbor_jaccard    | 0.3719  |
| student.h.cls~student.z.proj.out  | procrustes_distance | 0.5152  |
| student.h.cls~student.z.proj.out  | knn_label_agreement | 0.5574  |
| student.h.cls~student.z.proj.tap1 | cka_linear          | 0.9926  |
| student.h.cls~student.z.proj.tap1 | neighbor_jaccard    | 0.8037  |
| student.h.cls~student.z.proj.tap1 | procrustes_distance | 0.1758  |
| student.h.cls~student.z.proj.tap1 | knn_label_agreement | 0.8342  |
| student.h.cls~student.z.proj.tap2 | cka_linear          | 0.9871  |
| student.h.cls~student.z.proj.tap2 | neighbor_jaccard    | 0.7819  |
| student.h.cls~student.z.proj.tap2 | procrustes_distance | 0.2534  |
| student.h.cls~student.z.proj.tap2 | knn_label_agreement | 0.8152  |



### Probes


| space               |   knn_v1_k20 |   knn_v1_k200 |   linear_house_v1 |   linear_l2_v1 |
|:--------------------|-------------:|--------------:|------------------:|---------------:|
| student.h.cls       |       0.3304 |        0.3172 |            0.3985 |         0.3146 |
| student.h.cls.L03   |       0.3162 |        0.3106 |            0.3898 |         0.3014 |
| student.h.cls.L06   |       0.3284 |        0.3154 |            0.3980 |         0.3118 |
| student.h.cls.L09   |       0.3322 |        0.3146 |            0.3977 |         0.3121 |
| student.h.cls.L12   |       0.3304 |        0.3172 |            0.3995 |         0.3146 |
| student.h.gap       |       0.3432 |        0.3325 |            0.4331 |         0.3269 |
| student.h.gap.L03   |       0.3335 |        0.3233 |            0.4043 |         0.3121 |
| student.h.gap.L06   |       0.3394 |        0.3294 |            0.4209 |         0.3208 |
| student.h.gap.L09   |       0.3432 |        0.3340 |            0.4275 |         0.3274 |
| student.h.gap.L12   |       0.3432 |        0.3325 |            0.4331 |         0.3269 |
| student.z.embed     |       0.3269 |        0.3129 |            0.3946 |         0.3136 |
| student.z.proj.out  |       0.3085 |        0.3014 |            0.2479 |         0.2211 |
| student.z.proj.tap1 |       0.3307 |        0.3126 |            0.4362 |         0.3266 |
| student.z.proj.tap2 |       0.3304 |        0.3149 |            0.4217 |         0.3208 |



## AGREED TAKEAWAY

*(empty — filled only after discussion; see CLAUDE.md)*
