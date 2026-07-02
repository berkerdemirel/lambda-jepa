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


| space               |   knn_v1_k20 |   knn_v1_k200 |   linear_house_v1 |   linear_l2_v1 |   linear_raw_v1 |
|:--------------------|-------------:|--------------:|------------------:|---------------:|----------------:|
| student.h.cls       |       0.8966 |        0.8876 |            0.9121 |         0.8920 |          0.9019 |
| student.h.cls.L03   |       0.6415 |        0.5855 |            0.6996 |         0.5546 |          0.6234 |
| student.h.cls.L06   |       0.7628 |        0.6869 |            0.8161 |         0.7141 |          0.7791 |
| student.h.cls.L09   |       0.8685 |        0.8451 |            0.8907 |         0.8540 |          0.8792 |
| student.h.cls.L12   |       0.8966 |        0.8876 |            0.9121 |         0.8920 |          0.9022 |
| student.h.gap       |       0.8048 |        0.7513 |            0.8629 |         0.7577 |          0.8043 |
| student.h.gap.L03   |       0.6548 |        0.5676 |            0.7373 |         0.5763 |          0.6362 |
| student.h.gap.L06   |       0.7544 |        0.6932 |            0.8321 |         0.7108 |          0.7648 |
| student.h.gap.L09   |       0.7997 |        0.7404 |            0.8571 |         0.7480 |          0.7924 |
| student.h.gap.L12   |       0.8048 |        0.7513 |            0.8629 |         0.7577 |          0.8043 |
| student.z.embed     |       0.8981 |        0.8899 |            0.9096 |         0.8932 |          0.9103 |
| student.z.proj.out  |       0.8978 |        0.8945 |            0.8897 |         0.8764 |          0.8884 |
| student.z.proj.tap1 |       0.9080 |        0.9052 |            0.9139 |         0.9083 |          0.9106 |
| student.z.proj.tap2 |       0.9017 |        0.9004 |            0.9121 |         0.9027 |          0.9111 |



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


| space               |   knn_v1_k20 |   knn_v1_k200 |   linear_house_v1 |   linear_l2_v1 |   linear_raw_v1 |
|:--------------------|-------------:|--------------:|------------------:|---------------:|----------------:|
| student.h.cls       |       0.1725 |        0.1829 |            0.1146 |         0.1144 |          0.1180 |
| student.h.cls.L03   |       0.1743 |        0.1868 |            0.1501 |         0.1322 |          0.1462 |
| student.h.cls.L06   |       0.1809 |        0.1878 |            0.1304 |         0.1215 |          0.1304 |
| student.h.cls.L09   |       0.1794 |        0.1763 |            0.1169 |         0.1195 |          0.1205 |
| student.h.cls.L12   |       0.1725 |        0.1829 |            0.1146 |         0.1144 |          0.1162 |
| student.h.gap       |       0.2617 |        0.2484 |            0.1567 |         0.1355 |          0.1437 |
| student.h.gap.L03   |       0.2515 |        0.2456 |            0.2076 |         0.1625 |          0.1941 |
| student.h.gap.L06   |       0.2589 |        0.2476 |            0.1676 |         0.1513 |          0.1727 |
| student.h.gap.L09   |       0.2530 |        0.2451 |            0.1575 |         0.1376 |          0.1496 |
| student.h.gap.L12   |       0.2617 |        0.2484 |            0.1567 |         0.1355 |          0.1437 |
| student.z.embed     |       0.1511 |        0.1600 |            0.1068 |         0.1068 |          0.1042 |
| student.z.proj.out  |       0.1215 |        0.1389 |            0.1068 |         0.1068 |          0.1068 |
| student.z.proj.tap1 |       0.1572 |        0.1654 |            0.1847 |         0.1608 |          0.1631 |
| student.z.proj.tap2 |       0.1422 |        0.1544 |            0.1368 |         0.1136 |          0.1254 |



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


| space               |   knn_v1_k20 |   knn_v1_k200 |   linear_house_v1 |   linear_l2_v1 |   linear_raw_v1 |
|:--------------------|-------------:|--------------:|------------------:|---------------:|----------------:|
| student.h.cls       |       0.7633 |        0.7276 |            0.7679 |         0.6943 |          0.7582 |
| student.h.cls.L03   |       0.6066 |        0.5422 |            0.6033 |         0.4739 |          0.5776 |
| student.h.cls.L06   |       0.6935 |        0.6418 |            0.7177 |         0.5880 |          0.6945 |
| student.h.cls.L09   |       0.7462 |        0.7006 |            0.7580 |         0.6637 |          0.7468 |
| student.h.cls.L12   |       0.7633 |        0.7276 |            0.7679 |         0.6943 |          0.7577 |
| student.h.gap       |       0.6780 |        0.6163 |            0.7577 |         0.6150 |          0.7307 |
| student.h.gap.L03   |       0.5875 |        0.5169 |            0.6731 |         0.5205 |          0.6352 |
| student.h.gap.L06   |       0.6425 |        0.5620 |            0.7248 |         0.5778 |          0.6945 |
| student.h.gap.L09   |       0.6591 |        0.5870 |            0.7468 |         0.6046 |          0.7116 |
| student.h.gap.L12   |       0.6780 |        0.6163 |            0.7577 |         0.6150 |          0.7307 |
| student.z.embed     |       0.7592 |        0.7289 |            0.7608 |         0.7238 |          0.7590 |
| student.z.proj.out  |       0.7738 |        0.7707 |            0.7004 |         0.6986 |          0.7424 |
| student.z.proj.tap1 |       0.7778 |        0.7631 |            0.7939 |         0.7743 |          0.7954 |
| student.z.proj.tap2 |       0.7753 |        0.7623 |            0.7911 |         0.7727 |          0.7959 |



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


| space               |   knn_v1_k20 |   knn_v1_k200 |   linear_house_v1 |   linear_l2_v1 |   linear_raw_v1 |
|:--------------------|-------------:|--------------:|------------------:|---------------:|----------------:|
| student.h.cls       |       0.3304 |        0.3172 |            0.3995 |         0.3146 |          0.3941 |
| student.h.cls.L03   |       0.3162 |        0.3106 |            0.3898 |         0.3014 |          0.3842 |
| student.h.cls.L06   |       0.3284 |        0.3154 |            0.3980 |         0.3118 |          0.3969 |
| student.h.cls.L09   |       0.3322 |        0.3146 |            0.3977 |         0.3121 |          0.3918 |
| student.h.cls.L12   |       0.3304 |        0.3172 |            0.3995 |         0.3146 |          0.3949 |
| student.h.gap       |       0.3432 |        0.3325 |            0.4331 |         0.3269 |          0.4010 |
| student.h.gap.L03   |       0.3335 |        0.3233 |            0.4043 |         0.3121 |          0.3738 |
| student.h.gap.L06   |       0.3394 |        0.3294 |            0.4209 |         0.3208 |          0.3913 |
| student.h.gap.L09   |       0.3432 |        0.3340 |            0.4275 |         0.3274 |          0.4003 |
| student.h.gap.L12   |       0.3432 |        0.3325 |            0.4331 |         0.3269 |          0.4010 |
| student.z.embed     |       0.3269 |        0.3129 |            0.3946 |         0.3136 |          0.3832 |
| student.z.proj.out  |       0.3085 |        0.3014 |            0.2479 |         0.2211 |          0.2120 |
| student.z.proj.tap1 |       0.3307 |        0.3126 |            0.4362 |         0.3266 |          0.3957 |
| student.z.proj.tap2 |       0.3304 |        0.3149 |            0.4217 |         0.3208 |          0.3682 |



## in100.dino-ctrl.ep25.ext

### Battery (variant raw|full)

| model.space                                          |   rankme |   effective_rank |   alpha |   epps_pulley |   kurt_topeig.worst |   uniformity |   offdiag_redundancy.mean_abs_corr |   variance_floor.hinge |
|:-----------------------------------------------------|---------:|-----------------:|--------:|--------------:|--------------------:|-------------:|-----------------------------------:|-----------------------:|
| in100.dino-ctrl.ep25.ext · student.h.cls             |   141.6  |            44.8  |   1.961 |         370.7 |              0.8385 |      -2.023  |                            0.1301  |                 0.3149 |
| in100.dino-ctrl.ep25.ext · student.h.cls.L03         |    71.57 |            19.79 |   1.962 |         932.4 |              2.564  |      -0.5206 |                            0.2075  |                 0.6501 |
| in100.dino-ctrl.ep25.ext · student.h.cls.L06         |   108.9  |            31.07 |   1.849 |         501.4 |              1.48   |      -0.9216 |                            0.1572  |                 0.5453 |
| in100.dino-ctrl.ep25.ext · student.h.cls.L09         |   133.1  |            39.38 |   1.888 |         432.7 |              0.7973 |      -1.509  |                            0.1348  |                 0.4209 |
| in100.dino-ctrl.ep25.ext · student.h.cls.L12         |   141.6  |            44.8  |   1.961 |         370.7 |              0.8385 |      -2.023  |                            0.1301  |                 0.3149 |
| in100.dino-ctrl.ep25.ext · student.h.gap             |   160.9  |            34.47 |   1.782 |         443.1 |              1.964  |      -2.549  |                            0.1401  |                 0.5693 |
| in100.dino-ctrl.ep25.ext · student.h.gap.L03         |   104.8  |            18.08 |   1.903 |         818.8 |              1.754  |      -1.27   |                            0.1946  |                 0.7441 |
| in100.dino-ctrl.ep25.ext · student.h.gap.L06         |   136.2  |            27.32 |   1.839 |         532.4 |              2.102  |      -1.822  |                            0.1626  |                 0.6782 |
| in100.dino-ctrl.ep25.ext · student.h.gap.L09         |   157.6  |            33.2  |   1.743 |         481.2 |              1.781  |      -2.243  |                            0.1456  |                 0.6198 |
| in100.dino-ctrl.ep25.ext · student.h.gap.L12         |   160.9  |            34.47 |   1.782 |         443.1 |              1.964  |      -2.549  |                            0.1401  |                 0.5693 |
| in100.dino-ctrl.ep25.ext · student.z.dino.bottleneck |    95.14 |            43.91 |   2.743 |         527.2 |              5.263  |      -3.37   |                            0.1306  |                 0      |
| in100.dino-ctrl.ep25.ext · student.z.dino.tap1       |  1115    |           143.5  |   1.186 |         215.5 |              0.9966 |      -3.148  |                            0.05892 |                 0.6527 |
| in100.dino-ctrl.ep25.ext · student.z.dino.tap2       |  1196    |           197.1  |   1.117 |         208.4 |              3.96   |      -3.527  |                            0.0548  |                 0.5609 |
| in100.dino-ctrl.ep25.ext · teacher.h.cls             |   140.1  |            44.87 |   1.98  |         373.4 |              0.8217 |      -2.024  |                            0.1293  |                 0.3155 |
| in100.dino-ctrl.ep25.ext · teacher.h.cls.L03         |    71.34 |            20.19 |   1.97  |         864.2 |              2.801  |      -0.5173 |                            0.2009  |                 0.6553 |
| in100.dino-ctrl.ep25.ext · teacher.h.cls.L06         |   107.3  |            30.6  |   1.866 |         494.8 |              1.868  |      -0.9259 |                            0.1557  |                 0.5466 |
| in100.dino-ctrl.ep25.ext · teacher.h.cls.L09         |   131.5  |            39.29 |   1.909 |         441   |              0.7846 |      -1.521  |                            0.1343  |                 0.4195 |
| in100.dino-ctrl.ep25.ext · teacher.h.cls.L12         |   140.1  |            44.87 |   1.98  |         373.4 |              0.8217 |      -2.024  |                            0.1293  |                 0.3155 |
| in100.dino-ctrl.ep25.ext · teacher.h.gap             |   160.6  |            35.21 |   1.792 |         410.7 |              2.249  |      -2.572  |                            0.1367  |                 0.5717 |
| in100.dino-ctrl.ep25.ext · teacher.h.gap.L03         |   105.1  |            17.96 |   1.895 |         755.8 |              1.876  |      -1.255  |                            0.1844  |                 0.756  |
| in100.dino-ctrl.ep25.ext · teacher.h.gap.L06         |   135    |            26.87 |   1.849 |         556.3 |              2.988  |      -1.816  |                            0.161   |                 0.6825 |
| in100.dino-ctrl.ep25.ext · teacher.h.gap.L09         |   156.5  |            32.87 |   1.756 |         455.5 |              1.418  |      -2.266  |                            0.1441  |                 0.6213 |
| in100.dino-ctrl.ep25.ext · teacher.h.gap.L12         |   160.6  |            35.21 |   1.792 |         410.7 |              2.249  |      -2.572  |                            0.1367  |                 0.5717 |
| in100.dino-ctrl.ep25.ext · teacher.z.dino.bottleneck |    95.5  |            45.21 |   2.746 |         487   |              3.534  |      -3.376  |                            0.1271  |                 0      |
| in100.dino-ctrl.ep25.ext · teacher.z.dino.tap1       |  1119    |           146.8  |   1.179 |         219.6 |              1.268  |      -3.178  |                            0.05789 |                 0.6548 |
| in100.dino-ctrl.ep25.ext · teacher.z.dino.tap2       |  1196    |           202    |   1.108 |         200.8 |              4.293  |      -3.542  |                            0.05524 |                 0.5589 |



### Transfer ratios τ = value(h) / value(z)


**student.h.cls vs student.z.dino.bottleneck**


| metric                           |   value_h |   value_z |      tau |        delta |
|:---------------------------------|----------:|----------:|---------:|-------------:|
| uniformity                       |   -2.023  |   -3.37   |   0.6002 |    1.348     |
| variance_floor.hinge             |    0.3149 |    0      | inf      |    0.3149    |
| offdiag_redundancy.mean_abs_corr |    0.1301 |    0.1306 |   0.9964 |   -0.0004679 |
| rankme                           |  141.6    |   95.14   |   1.488  |   46.46      |
| effective_rank                   |   44.8    |   43.91   |   1.02   |    0.8912    |
| alpha                            |    1.961  |    2.743  |   0.7151 |   -0.7812    |
| epps_pulley                      |  370.7    |  527.2    |   0.7032 | -156.5       |
| kurt_topeig.worst                |    0.8385 |    5.263  |   0.1593 |   -4.425     |



**student.h.cls vs student.z.dino.tap1**


| metric                           |   value_h |    value_z |    tau |      delta |
|:---------------------------------|----------:|-----------:|-------:|-----------:|
| uniformity                       |   -2.023  |   -3.148   | 0.6426 |    1.125   |
| variance_floor.hinge             |    0.3149 |    0.6527  | 0.4825 |   -0.3377  |
| offdiag_redundancy.mean_abs_corr |    0.1301 |    0.05892 | 2.208  |    0.07118 |
| rankme                           |  141.6    | 1115       | 0.127  | -973.5     |
| effective_rank                   |   44.8    |  143.5     | 0.3122 |  -98.72    |
| alpha                            |    1.961  |    1.186   | 1.654  |    0.7757  |
| epps_pulley                      |  370.7    |  215.5     | 1.72   |  155.2     |
| kurt_topeig.worst                |    0.8385 |    0.9966  | 0.8413 |   -0.1581  |



**student.h.cls vs student.z.dino.tap2**


| metric                           |   value_h |   value_z |    tau |      delta |
|:---------------------------------|----------:|----------:|-------:|-----------:|
| uniformity                       |   -2.023  |   -3.527  | 0.5735 |     1.504  |
| variance_floor.hinge             |    0.3149 |    0.5609 | 0.5615 |    -0.246  |
| offdiag_redundancy.mean_abs_corr |    0.1301 |    0.0548 | 2.374  |     0.0753 |
| rankme                           |  141.6    | 1196      | 0.1184 | -1055      |
| effective_rank                   |   44.8    |  197.1    | 0.2273 |  -152.3    |
| alpha                            |    1.961  |    1.117  | 1.757  |     0.8447 |
| epps_pulley                      |  370.7    |  208.4    | 1.779  |   162.3    |
| kurt_topeig.worst                |    0.8385 |    3.96   | 0.2117 |    -3.121  |



**student.h.cls.L03 vs student.z.dino.bottleneck**


| metric                           |   value_h |   value_z |      tau |     delta |
|:---------------------------------|----------:|----------:|---------:|----------:|
| uniformity                       |   -0.5206 |   -3.37   |   0.1545 |   2.85    |
| variance_floor.hinge             |    0.6501 |    0      | inf      |   0.6501  |
| offdiag_redundancy.mean_abs_corr |    0.2075 |    0.1306 |   1.589  |   0.07695 |
| rankme                           |   71.57   |   95.14   |   0.7522 | -23.57    |
| effective_rank                   |   19.79   |   43.91   |   0.4506 | -24.12    |
| alpha                            |    1.962  |    2.743  |   0.7153 |  -0.7809  |
| epps_pulley                      |  932.4    |  527.2    |   1.769  | 405.2     |
| kurt_topeig.worst                |    2.564  |    5.263  |   0.4872 |  -2.699   |



**student.h.cls.L03 vs student.z.dino.tap1**


| metric                           |   value_h |    value_z |     tau |        delta |
|:---------------------------------|----------:|-----------:|--------:|-------------:|
| uniformity                       |   -0.5206 |   -3.148   | 0.1654  |     2.627    |
| variance_floor.hinge             |    0.6501 |    0.6527  | 0.9961  |    -0.002563 |
| offdiag_redundancy.mean_abs_corr |    0.2075 |    0.05892 | 3.522   |     0.1486   |
| rankme                           |   71.57   | 1115       | 0.06418 | -1044        |
| effective_rank                   |   19.79   |  143.5     | 0.1379  |  -123.7      |
| alpha                            |    1.962  |    1.186   | 1.655   |     0.776    |
| epps_pulley                      |  932.4    |  215.5     | 4.326   |   716.8      |
| kurt_topeig.worst                |    2.564  |    0.9966  | 2.573   |     1.567    |



**student.h.cls.L03 vs student.z.dino.tap2**


| metric                           |   value_h |   value_z |     tau |       delta |
|:---------------------------------|----------:|----------:|--------:|------------:|
| uniformity                       |   -0.5206 |   -3.527  | 0.1476  |     3.006   |
| variance_floor.hinge             |    0.6501 |    0.5609 | 1.159   |     0.08921 |
| offdiag_redundancy.mean_abs_corr |    0.2075 |    0.0548 | 3.787   |     0.1527  |
| rankme                           |   71.57   | 1196      | 0.05982 | -1125       |
| effective_rank                   |   19.79   |  197.1    | 0.1004  |  -177.3     |
| alpha                            |    1.962  |    1.117  | 1.757   |     0.845   |
| epps_pulley                      |  932.4    |  208.4    | 4.474   |   724       |
| kurt_topeig.worst                |    2.564  |    3.96   | 0.6475  |    -1.396   |



**student.h.cls.L06 vs student.z.dino.bottleneck**


| metric                           |   value_h |   value_z |      tau |     delta |
|:---------------------------------|----------:|----------:|---------:|----------:|
| uniformity                       |   -0.9216 |   -3.37   |   0.2735 |   2.449   |
| variance_floor.hinge             |    0.5453 |    0      | inf      |   0.5453  |
| offdiag_redundancy.mean_abs_corr |    0.1572 |    0.1306 |   1.204  |   0.02663 |
| rankme                           |  108.9    |   95.14   |   1.145  |  13.77    |
| effective_rank                   |   31.07   |   43.91   |   0.7077 | -12.83    |
| alpha                            |    1.849  |    2.743  |   0.674  |  -0.8939  |
| epps_pulley                      |  501.4    |  527.2    |   0.9511 | -25.76    |
| kurt_topeig.worst                |    1.48   |    5.263  |   0.2812 |  -3.783   |



**student.h.cls.L06 vs student.z.dino.tap1**


| metric                           |   value_h |    value_z |     tau |       delta |
|:---------------------------------|----------:|-----------:|--------:|------------:|
| uniformity                       |   -0.9216 |   -3.148   | 0.2928  |     2.226   |
| variance_floor.hinge             |    0.5453 |    0.6527  | 0.8355  |    -0.1074  |
| offdiag_redundancy.mean_abs_corr |    0.1572 |    0.05892 | 2.668   |     0.09827 |
| rankme                           |  108.9    | 1115       | 0.09767 | -1006       |
| effective_rank                   |   31.07   |  143.5     | 0.2165  |  -112.4     |
| alpha                            |    1.849  |    1.186   | 1.559   |     0.663   |
| epps_pulley                      |  501.4    |  215.5     | 2.326   |   285.9     |
| kurt_topeig.worst                |    1.48   |    0.9966  | 1.485   |     0.4835  |



**student.h.cls.L06 vs student.z.dino.tap2**


| metric                           |   value_h |   value_z |     tau |      delta |
|:---------------------------------|----------:|----------:|--------:|-----------:|
| uniformity                       |   -0.9216 |   -3.527  | 0.2613  |     2.605  |
| variance_floor.hinge             |    0.5453 |    0.5609 | 0.9722  |    -0.0156 |
| offdiag_redundancy.mean_abs_corr |    0.1572 |    0.0548 | 2.869   |     0.1024 |
| rankme                           |  108.9    | 1196      | 0.09103 | -1087      |
| effective_rank                   |   31.07   |  197.1    | 0.1577  |  -166      |
| alpha                            |    1.849  |    1.117  | 1.656   |     0.732  |
| epps_pulley                      |  501.4    |  208.4    | 2.406   |   293      |
| kurt_topeig.worst                |    1.48   |    3.96   | 0.3738  |    -2.48   |



**student.h.cls.L09 vs student.z.dino.bottleneck**


| metric                           |   value_h |   value_z |      tau |      delta |
|:---------------------------------|----------:|----------:|---------:|-----------:|
| uniformity                       |   -1.509  |   -3.37   |   0.4477 |   1.861    |
| variance_floor.hinge             |    0.4209 |    0      | inf      |   0.4209   |
| offdiag_redundancy.mean_abs_corr |    0.1348 |    0.1306 |   1.032  |   0.004221 |
| rankme                           |  133.1    |   95.14   |   1.399  |  37.97     |
| effective_rank                   |   39.38   |   43.91   |   0.8969 |  -4.525    |
| alpha                            |    1.888  |    2.743  |   0.6883 |  -0.8548   |
| epps_pulley                      |  432.7    |  527.2    |   0.8207 | -94.51     |
| kurt_topeig.worst                |    0.7973 |    5.263  |   0.1515 |  -4.466    |



**student.h.cls.L09 vs student.z.dino.tap1**


| metric                           |   value_h |    value_z |    tau |      delta |
|:---------------------------------|----------:|-----------:|-------:|-----------:|
| uniformity                       |   -1.509  |   -3.148   | 0.4794 |    1.639   |
| variance_floor.hinge             |    0.4209 |    0.6527  | 0.6449 |   -0.2317  |
| offdiag_redundancy.mean_abs_corr |    0.1348 |    0.05892 | 2.288  |    0.07586 |
| rankme                           |  133.1    | 1115       | 0.1194 | -982       |
| effective_rank                   |   39.38   |  143.5     | 0.2744 | -104.1     |
| alpha                            |    1.888  |    1.186   | 1.592  |    0.7021  |
| epps_pulley                      |  432.7    |  215.5     | 2.007  |  217.1     |
| kurt_topeig.worst                |    0.7973 |    0.9966  | 0.7999 |   -0.1994  |



**student.h.cls.L09 vs student.z.dino.tap2**


| metric                           |   value_h |   value_z |    tau |       delta |
|:---------------------------------|----------:|----------:|-------:|------------:|
| uniformity                       |   -1.509  |   -3.527  | 0.4279 |     2.018   |
| variance_floor.hinge             |    0.4209 |    0.5609 | 0.7505 |    -0.14    |
| offdiag_redundancy.mean_abs_corr |    0.1348 |    0.0548 | 2.46   |     0.07999 |
| rankme                           |  133.1    | 1196      | 0.1113 | -1063       |
| effective_rank                   |   39.38   |  197.1    | 0.1998 |  -157.7     |
| alpha                            |    1.888  |    1.117  | 1.691  |     0.7711  |
| epps_pulley                      |  432.7    |  208.4    | 2.076  |   224.3     |
| kurt_topeig.worst                |    0.7973 |    3.96   | 0.2013 |    -3.163   |



**student.h.cls.L12 vs student.z.dino.bottleneck**


| metric                           |   value_h |   value_z |      tau |        delta |
|:---------------------------------|----------:|----------:|---------:|-------------:|
| uniformity                       |   -2.023  |   -3.37   |   0.6002 |    1.348     |
| variance_floor.hinge             |    0.3149 |    0      | inf      |    0.3149    |
| offdiag_redundancy.mean_abs_corr |    0.1301 |    0.1306 |   0.9964 |   -0.0004679 |
| rankme                           |  141.6    |   95.14   |   1.488  |   46.46      |
| effective_rank                   |   44.8    |   43.91   |   1.02   |    0.8912    |
| alpha                            |    1.961  |    2.743  |   0.7151 |   -0.7812    |
| epps_pulley                      |  370.7    |  527.2    |   0.7032 | -156.5       |
| kurt_topeig.worst                |    0.8385 |    5.263  |   0.1593 |   -4.425     |



**student.h.cls.L12 vs student.z.dino.tap1**


| metric                           |   value_h |    value_z |    tau |      delta |
|:---------------------------------|----------:|-----------:|-------:|-----------:|
| uniformity                       |   -2.023  |   -3.148   | 0.6426 |    1.125   |
| variance_floor.hinge             |    0.3149 |    0.6527  | 0.4825 |   -0.3377  |
| offdiag_redundancy.mean_abs_corr |    0.1301 |    0.05892 | 2.208  |    0.07118 |
| rankme                           |  141.6    | 1115       | 0.127  | -973.5     |
| effective_rank                   |   44.8    |  143.5     | 0.3122 |  -98.72    |
| alpha                            |    1.961  |    1.186   | 1.654  |    0.7757  |
| epps_pulley                      |  370.7    |  215.5     | 1.72   |  155.2     |
| kurt_topeig.worst                |    0.8385 |    0.9966  | 0.8413 |   -0.1581  |



**student.h.cls.L12 vs student.z.dino.tap2**


| metric                           |   value_h |   value_z |    tau |      delta |
|:---------------------------------|----------:|----------:|-------:|-----------:|
| uniformity                       |   -2.023  |   -3.527  | 0.5735 |     1.504  |
| variance_floor.hinge             |    0.3149 |    0.5609 | 0.5615 |    -0.246  |
| offdiag_redundancy.mean_abs_corr |    0.1301 |    0.0548 | 2.374  |     0.0753 |
| rankme                           |  141.6    | 1196      | 0.1184 | -1055      |
| effective_rank                   |   44.8    |  197.1    | 0.2273 |  -152.3    |
| alpha                            |    1.961  |    1.117  | 1.757  |     0.8447 |
| epps_pulley                      |  370.7    |  208.4    | 1.779  |   162.3    |
| kurt_topeig.worst                |    0.8385 |    3.96   | 0.2117 |    -3.121  |



**student.h.gap vs student.z.dino.bottleneck**


| metric                           |   value_h |   value_z |      tau |      delta |
|:---------------------------------|----------:|----------:|---------:|-----------:|
| uniformity                       |   -2.549  |   -3.37   |   0.7564 |   0.821    |
| variance_floor.hinge             |    0.5693 |    0      | inf      |   0.5693   |
| offdiag_redundancy.mean_abs_corr |    0.1401 |    0.1306 |   1.073  |   0.009491 |
| rankme                           |  160.9    |   95.14   |   1.692  |  65.79     |
| effective_rank                   |   34.47   |   43.91   |   0.785  |  -9.441    |
| alpha                            |    1.782  |    2.743  |   0.6499 |  -0.96     |
| epps_pulley                      |  443.1    |  527.2    |   0.8405 | -84.06     |
| kurt_topeig.worst                |    1.964  |    5.263  |   0.3731 |  -3.3      |



**student.h.gap vs student.z.dino.tap1**


| metric                           |   value_h |    value_z |    tau |      delta |
|:---------------------------------|----------:|-----------:|-------:|-----------:|
| uniformity                       |   -2.549  |   -3.148   | 0.81   |    0.5982  |
| variance_floor.hinge             |    0.5693 |    0.6527  | 0.8723 |   -0.08332 |
| offdiag_redundancy.mean_abs_corr |    0.1401 |    0.05892 | 2.377  |    0.08114 |
| rankme                           |  160.9    | 1115       | 0.1443 | -954.1     |
| effective_rank                   |   34.47   |  143.5     | 0.2402 | -109       |
| alpha                            |    1.782  |    1.186   | 1.503  |    0.5969  |
| epps_pulley                      |  443.1    |  215.5     | 2.056  |  227.6     |
| kurt_topeig.worst                |    1.964  |    0.9966  | 1.97   |    0.967   |



**student.h.gap vs student.z.dino.tap2**


| metric                           |   value_h |   value_z |    tau |        delta |
|:---------------------------------|----------:|----------:|-------:|-------------:|
| uniformity                       |   -2.549  |   -3.527  | 0.7229 |     0.9774   |
| variance_floor.hinge             |    0.5693 |    0.5609 | 1.015  |     0.008448 |
| offdiag_redundancy.mean_abs_corr |    0.1401 |    0.0548 | 2.556  |     0.08526  |
| rankme                           |  160.9    | 1196      | 0.1345 | -1035        |
| effective_rank                   |   34.47   |  197.1    | 0.1749 |  -162.6      |
| alpha                            |    1.782  |    1.117  | 1.596  |     0.6659   |
| epps_pulley                      |  443.1    |  208.4    | 2.126  |   234.7      |
| kurt_topeig.worst                |    1.964  |    3.96   | 0.4959 |    -1.996    |



**student.h.gap.L03 vs student.z.dino.bottleneck**


| metric                           |   value_h |   value_z |      tau |     delta |
|:---------------------------------|----------:|----------:|---------:|----------:|
| uniformity                       |   -1.27   |   -3.37   |   0.3769 |   2.1     |
| variance_floor.hinge             |    0.7441 |    0      | inf      |   0.7441  |
| offdiag_redundancy.mean_abs_corr |    0.1946 |    0.1306 |   1.49   |   0.06399 |
| rankme                           |  104.8    |   95.14   |   1.101  |   9.634   |
| effective_rank                   |   18.08   |   43.91   |   0.4117 | -25.83    |
| alpha                            |    1.903  |    2.743  |   0.6937 |  -0.8399  |
| epps_pulley                      |  818.8    |  527.2    |   1.553  | 291.6     |
| kurt_topeig.worst                |    1.754  |    5.263  |   0.3333 |  -3.509   |



**student.h.gap.L03 vs student.z.dino.tap1**


| metric                           |   value_h |    value_z |     tau |       delta |
|:---------------------------------|----------:|-----------:|--------:|------------:|
| uniformity                       |   -1.27   |   -3.148   | 0.4036  |     1.877   |
| variance_floor.hinge             |    0.7441 |    0.6527  | 1.14    |     0.09145 |
| offdiag_redundancy.mean_abs_corr |    0.1946 |    0.05892 | 3.302   |     0.1356  |
| rankme                           |  104.8    | 1115       | 0.09396 | -1010       |
| effective_rank                   |   18.08   |  143.5     | 0.126   |  -125.4     |
| alpha                            |    1.903  |    1.186   | 1.605   |     0.717   |
| epps_pulley                      |  818.8    |  215.5     | 3.799   |   603.2     |
| kurt_topeig.worst                |    1.754  |    0.9966  | 1.76    |     0.7578  |



**student.h.gap.L03 vs student.z.dino.tap2**


| metric                           |   value_h |   value_z |     tau |      delta |
|:---------------------------------|----------:|----------:|--------:|-----------:|
| uniformity                       |   -1.27   |   -3.527  | 0.3602  |     2.256  |
| variance_floor.hinge             |    0.7441 |    0.5609 | 1.327   |     0.1832 |
| offdiag_redundancy.mean_abs_corr |    0.1946 |    0.0548 | 3.55    |     0.1398 |
| rankme                           |  104.8    | 1196      | 0.08758 | -1092      |
| effective_rank                   |   18.08   |  197.1    | 0.09172 |  -179      |
| alpha                            |    1.903  |    1.117  | 1.704   |     0.786  |
| epps_pulley                      |  818.8    |  208.4    | 3.929   |   610.3    |
| kurt_topeig.worst                |    1.754  |    3.96   | 0.443   |    -2.206  |



**student.h.gap.L06 vs student.z.dino.bottleneck**


| metric                           |   value_h |   value_z |      tau |     delta |
|:---------------------------------|----------:|----------:|---------:|----------:|
| uniformity                       |   -1.822  |   -3.37   |   0.5407 |   1.548   |
| variance_floor.hinge             |    0.6782 |    0      | inf      |   0.6782  |
| offdiag_redundancy.mean_abs_corr |    0.1626 |    0.1306 |   1.246  |   0.03208 |
| rankme                           |  136.2    |   95.14   |   1.431  |  41.02    |
| effective_rank                   |   27.32   |   43.91   |   0.6222 | -16.59    |
| alpha                            |    1.839  |    2.743  |   0.6704 |  -0.9039  |
| epps_pulley                      |  532.4    |  527.2    |   1.01   |   5.222   |
| kurt_topeig.worst                |    2.102  |    5.263  |   0.3994 |  -3.161   |



**student.h.gap.L06 vs student.z.dino.tap1**


| metric                           |   value_h |    value_z |    tau |      delta |
|:---------------------------------|----------:|-----------:|-------:|-----------:|
| uniformity                       |   -1.822  |   -3.148   | 0.579  |    1.325   |
| variance_floor.hinge             |    0.6782 |    0.6527  | 1.039  |    0.02556 |
| offdiag_redundancy.mean_abs_corr |    0.1626 |    0.05892 | 2.76   |    0.1037  |
| rankme                           |  136.2    | 1115       | 0.1221 | -978.9     |
| effective_rank                   |   27.32   |  143.5     | 0.1904 | -116.2     |
| alpha                            |    1.839  |    1.186   | 1.551  |    0.653   |
| epps_pulley                      |  532.4    |  215.5     | 2.47   |  316.9     |
| kurt_topeig.worst                |    2.102  |    0.9966  | 2.109  |    1.105   |



**student.h.gap.L06 vs student.z.dino.tap2**


| metric                           |   value_h |   value_z |    tau |      delta |
|:---------------------------------|----------:|----------:|-------:|-----------:|
| uniformity                       |   -1.822  |   -3.527  | 0.5167 |     1.704  |
| variance_floor.hinge             |    0.6782 |    0.5609 | 1.209  |     0.1173 |
| offdiag_redundancy.mean_abs_corr |    0.1626 |    0.0548 | 2.968  |     0.1078 |
| rankme                           |  136.2    | 1196      | 0.1138 | -1060      |
| effective_rank                   |   27.32   |  197.1    | 0.1386 |  -169.8    |
| alpha                            |    1.839  |    1.117  | 1.647  |     0.722  |
| epps_pulley                      |  532.4    |  208.4    | 2.555  |   324      |
| kurt_topeig.worst                |    2.102  |    3.96   | 0.5308 |    -1.858  |



**student.h.gap.L09 vs student.z.dino.bottleneck**


| metric                           |   value_h |   value_z |      tau |     delta |
|:---------------------------------|----------:|----------:|---------:|----------:|
| uniformity                       |   -2.243  |   -3.37   |   0.6654 |   1.128   |
| variance_floor.hinge             |    0.6198 |    0      | inf      |   0.6198  |
| offdiag_redundancy.mean_abs_corr |    0.1456 |    0.1306 |   1.115  |   0.01508 |
| rankme                           |  157.6    |   95.14   |   1.656  |  62.46    |
| effective_rank                   |   33.2    |   43.91   |   0.7561 | -10.71    |
| alpha                            |    1.743  |    2.743  |   0.6355 |  -0.9998  |
| epps_pulley                      |  481.2    |  527.2    |   0.9129 | -45.94    |
| kurt_topeig.worst                |    1.781  |    5.263  |   0.3384 |  -3.482   |



**student.h.gap.L09 vs student.z.dino.tap1**


| metric                           |   value_h |    value_z |    tau |      delta |
|:---------------------------------|----------:|-----------:|-------:|-----------:|
| uniformity                       |   -2.243  |   -3.148   | 0.7125 |    0.905   |
| variance_floor.hinge             |    0.6198 |    0.6527  | 0.9496 |   -0.03287 |
| offdiag_redundancy.mean_abs_corr |    0.1456 |    0.05892 | 2.472  |    0.08672 |
| rankme                           |  157.6    | 1115       | 0.1413 | -957.5     |
| effective_rank                   |   33.2    |  143.5     | 0.2313 | -110.3     |
| alpha                            |    1.743  |    1.186   | 1.47   |    0.5572  |
| epps_pulley                      |  481.2    |  215.5     | 2.233  |  265.7     |
| kurt_topeig.worst                |    1.781  |    0.9966  | 1.787  |    0.7844  |



**student.h.gap.L09 vs student.z.dino.tap2**


| metric                           |   value_h |   value_z |    tau |       delta |
|:---------------------------------|----------:|----------:|-------:|------------:|
| uniformity                       |   -2.243  |   -3.527  | 0.6359 |     1.284   |
| variance_floor.hinge             |    0.6198 |    0.5609 | 1.105  |     0.0589  |
| offdiag_redundancy.mean_abs_corr |    0.1456 |    0.0548 | 2.658  |     0.09085 |
| rankme                           |  157.6    | 1196      | 0.1317 | -1039       |
| effective_rank                   |   33.2    |  197.1    | 0.1684 |  -163.9     |
| alpha                            |    1.743  |    1.117  | 1.561  |     0.6262  |
| epps_pulley                      |  481.2    |  208.4    | 2.309  |   272.8     |
| kurt_topeig.worst                |    1.781  |    3.96   | 0.4498 |    -2.179   |



**student.h.gap.L12 vs student.z.dino.bottleneck**


| metric                           |   value_h |   value_z |      tau |      delta |
|:---------------------------------|----------:|----------:|---------:|-----------:|
| uniformity                       |   -2.549  |   -3.37   |   0.7564 |   0.821    |
| variance_floor.hinge             |    0.5693 |    0      | inf      |   0.5693   |
| offdiag_redundancy.mean_abs_corr |    0.1401 |    0.1306 |   1.073  |   0.009491 |
| rankme                           |  160.9    |   95.14   |   1.692  |  65.79     |
| effective_rank                   |   34.47   |   43.91   |   0.785  |  -9.441    |
| alpha                            |    1.782  |    2.743  |   0.6499 |  -0.96     |
| epps_pulley                      |  443.1    |  527.2    |   0.8405 | -84.06     |
| kurt_topeig.worst                |    1.964  |    5.263  |   0.3731 |  -3.3      |



**student.h.gap.L12 vs student.z.dino.tap1**


| metric                           |   value_h |    value_z |    tau |      delta |
|:---------------------------------|----------:|-----------:|-------:|-----------:|
| uniformity                       |   -2.549  |   -3.148   | 0.81   |    0.5982  |
| variance_floor.hinge             |    0.5693 |    0.6527  | 0.8723 |   -0.08332 |
| offdiag_redundancy.mean_abs_corr |    0.1401 |    0.05892 | 2.377  |    0.08114 |
| rankme                           |  160.9    | 1115       | 0.1443 | -954.1     |
| effective_rank                   |   34.47   |  143.5     | 0.2402 | -109       |
| alpha                            |    1.782  |    1.186   | 1.503  |    0.5969  |
| epps_pulley                      |  443.1    |  215.5     | 2.056  |  227.6     |
| kurt_topeig.worst                |    1.964  |    0.9966  | 1.97   |    0.967   |



**student.h.gap.L12 vs student.z.dino.tap2**


| metric                           |   value_h |   value_z |    tau |        delta |
|:---------------------------------|----------:|----------:|-------:|-------------:|
| uniformity                       |   -2.549  |   -3.527  | 0.7229 |     0.9774   |
| variance_floor.hinge             |    0.5693 |    0.5609 | 1.015  |     0.008448 |
| offdiag_redundancy.mean_abs_corr |    0.1401 |    0.0548 | 2.556  |     0.08526  |
| rankme                           |  160.9    | 1196      | 0.1345 | -1035        |
| effective_rank                   |   34.47   |  197.1    | 0.1749 |  -162.6      |
| alpha                            |    1.782  |    1.117  | 1.596  |     0.6659   |
| epps_pulley                      |  443.1    |  208.4    | 2.126  |   234.7      |
| kurt_topeig.worst                |    1.964  |    3.96   | 0.4959 |    -1.996    |



**teacher.h.cls vs teacher.z.dino.bottleneck**


| metric                           |   value_h |   value_z |      tau |       delta |
|:---------------------------------|----------:|----------:|---------:|------------:|
| uniformity                       |   -2.024  |   -3.376  |   0.5996 |    1.352    |
| variance_floor.hinge             |    0.3155 |    0      | inf      |    0.3155   |
| offdiag_redundancy.mean_abs_corr |    0.1293 |    0.1271 |   1.017  |    0.002213 |
| rankme                           |  140.1    |   95.5    |   1.467  |   44.63     |
| effective_rank                   |   44.87   |   45.21   |   0.9925 |   -0.3385   |
| alpha                            |    1.98   |    2.746  |   0.721  |   -0.7661   |
| epps_pulley                      |  373.4    |  487      |   0.7668 | -113.6      |
| kurt_topeig.worst                |    0.8217 |    3.534  |   0.2325 |   -2.712    |



**teacher.h.cls vs teacher.z.dino.tap1**


| metric                           |   value_h |    value_z |    tau |     delta |
|:---------------------------------|----------:|-----------:|-------:|----------:|
| uniformity                       |   -2.024  |   -3.178   | 0.6369 |    1.154  |
| variance_floor.hinge             |    0.3155 |    0.6548  | 0.4817 |   -0.3394 |
| offdiag_redundancy.mean_abs_corr |    0.1293 |    0.05789 | 2.233  |    0.0714 |
| rankme                           |  140.1    | 1119       | 0.1253 | -978.6    |
| effective_rank                   |   44.87   |  146.8     | 0.3056 | -102      |
| alpha                            |    1.98   |    1.179   | 1.679  |    0.801  |
| epps_pulley                      |  373.4    |  219.6     | 1.701  |  153.9    |
| kurt_topeig.worst                |    0.8217 |    1.268   | 0.6483 |   -0.4458 |



**teacher.h.cls vs teacher.z.dino.tap2**


| metric                           |   value_h |    value_z |    tau |       delta |
|:---------------------------------|----------:|-----------:|-------:|------------:|
| uniformity                       |   -2.024  |   -3.542   | 0.5715 |     1.518   |
| variance_floor.hinge             |    0.3155 |    0.5589  | 0.5644 |    -0.2434  |
| offdiag_redundancy.mean_abs_corr |    0.1293 |    0.05524 | 2.34   |     0.07405 |
| rankme                           |  140.1    | 1196       | 0.1171 | -1056       |
| effective_rank                   |   44.87   |  202       | 0.2222 |  -157.1     |
| alpha                            |    1.98   |    1.108   | 1.787  |     0.872   |
| epps_pulley                      |  373.4    |  200.8     | 1.86   |   172.7     |
| kurt_topeig.worst                |    0.8217 |    4.293   | 0.1914 |    -3.471   |



**teacher.h.cls.L03 vs teacher.z.dino.bottleneck**


| metric                           |   value_h |   value_z |      tau |     delta |
|:---------------------------------|----------:|----------:|---------:|----------:|
| uniformity                       |   -0.5173 |   -3.376  |   0.1532 |   2.859   |
| variance_floor.hinge             |    0.6553 |    0      | inf      |   0.6553  |
| offdiag_redundancy.mean_abs_corr |    0.2009 |    0.1271 |   1.581  |   0.07385 |
| rankme                           |   71.34   |   95.5    |   0.7471 | -24.15    |
| effective_rank                   |   20.19   |   45.21   |   0.4465 | -25.02    |
| alpha                            |    1.97   |    2.746  |   0.7173 |  -0.7762  |
| epps_pulley                      |  864.2    |  487      |   1.774  | 377.2     |
| kurt_topeig.worst                |    2.801  |    3.534  |   0.7926 |  -0.733   |



**teacher.h.cls.L03 vs teacher.z.dino.tap1**


| metric                           |   value_h |    value_z |     tau |         delta |
|:---------------------------------|----------:|-----------:|--------:|--------------:|
| uniformity                       |   -0.5173 |   -3.178   | 0.1628  |     2.661     |
| variance_floor.hinge             |    0.6553 |    0.6548  | 1.001   |     0.0004933 |
| offdiag_redundancy.mean_abs_corr |    0.2009 |    0.05789 | 3.471   |     0.143     |
| rankme                           |   71.34   | 1119       | 0.06377 | -1047         |
| effective_rank                   |   20.19   |  146.8     | 0.1375  |  -126.6       |
| alpha                            |    1.97   |    1.179   | 1.671   |     0.7908    |
| epps_pulley                      |  864.2    |  219.6     | 3.936   |   644.6       |
| kurt_topeig.worst                |    2.801  |    1.268   | 2.21    |     1.534     |



**teacher.h.cls.L03 vs teacher.z.dino.tap2**


| metric                           |   value_h |    value_z |     tau |       delta |
|:---------------------------------|----------:|-----------:|--------:|------------:|
| uniformity                       |   -0.5173 |   -3.542   | 0.1461  |     3.025   |
| variance_floor.hinge             |    0.6553 |    0.5589  | 1.173   |     0.09641 |
| offdiag_redundancy.mean_abs_corr |    0.2009 |    0.05524 | 3.637   |     0.1457  |
| rankme                           |   71.34   | 1196       | 0.05963 | -1125       |
| effective_rank                   |   20.19   |  202       | 0.09995 |  -181.8     |
| alpha                            |    1.97   |    1.108   | 1.778   |     0.8619  |
| epps_pulley                      |  864.2    |  200.8     | 4.304   |   663.4     |
| kurt_topeig.worst                |    2.801  |    4.293   | 0.6525  |    -1.492   |



**teacher.h.cls.L06 vs teacher.z.dino.bottleneck**


| metric                           |   value_h |   value_z |      tau |     delta |
|:---------------------------------|----------:|----------:|---------:|----------:|
| uniformity                       |   -0.9259 |   -3.376  |   0.2742 |   2.45    |
| variance_floor.hinge             |    0.5466 |    0      | inf      |   0.5466  |
| offdiag_redundancy.mean_abs_corr |    0.1557 |    0.1271 |   1.225  |   0.02859 |
| rankme                           |  107.3    |   95.5    |   1.124  |  11.81    |
| effective_rank                   |   30.6    |   45.21   |   0.6768 | -14.61    |
| alpha                            |    1.866  |    2.746  |   0.6797 |  -0.8797  |
| epps_pulley                      |  494.8    |  487      |   1.016  |   7.834   |
| kurt_topeig.worst                |    1.868  |    3.534  |   0.5286 |  -1.666   |



**teacher.h.cls.L06 vs teacher.z.dino.tap1**


| metric                           |   value_h |    value_z |     tau |       delta |
|:---------------------------------|----------:|-----------:|--------:|------------:|
| uniformity                       |   -0.9259 |   -3.178   | 0.2913  |     2.253   |
| variance_floor.hinge             |    0.5466 |    0.6548  | 0.8347  |    -0.1082  |
| offdiag_redundancy.mean_abs_corr |    0.1557 |    0.05789 | 2.689   |     0.09777 |
| rankme                           |  107.3    | 1119       | 0.09592 | -1011       |
| effective_rank                   |   30.6    |  146.8     | 0.2084  |  -116.2     |
| alpha                            |    1.866  |    1.179   | 1.583   |     0.6874  |
| epps_pulley                      |  494.8    |  219.6     | 2.254   |   275.3     |
| kurt_topeig.worst                |    1.868  |    1.268   | 1.474   |     0.6005  |



**teacher.h.cls.L06 vs teacher.z.dino.tap2**


| metric                           |   value_h |    value_z |     tau |      delta |
|:---------------------------------|----------:|-----------:|--------:|-----------:|
| uniformity                       |   -0.9259 |   -3.542   | 0.2614  |     2.616  |
| variance_floor.hinge             |    0.5466 |    0.5589  | 0.978   |    -0.0123 |
| offdiag_redundancy.mean_abs_corr |    0.1557 |    0.05524 | 2.818   |     0.1004 |
| rankme                           |  107.3    | 1196       | 0.08968 | -1089      |
| effective_rank                   |   30.6    |  202       | 0.1515  |  -171.4    |
| alpha                            |    1.866  |    1.108   | 1.684   |     0.7584 |
| epps_pulley                      |  494.8    |  200.8     | 2.465   |   294.1    |
| kurt_topeig.worst                |    1.868  |    4.293   | 0.4351  |    -2.425  |



**teacher.h.cls.L09 vs teacher.z.dino.bottleneck**


| metric                           |   value_h |   value_z |      tau |      delta |
|:---------------------------------|----------:|----------:|---------:|-----------:|
| uniformity                       |   -1.521  |   -3.376  |   0.4504 |   1.856    |
| variance_floor.hinge             |    0.4195 |    0      | inf      |   0.4195   |
| offdiag_redundancy.mean_abs_corr |    0.1343 |    0.1271 |   1.057  |   0.007223 |
| rankme                           |  131.5    |   95.5    |   1.377  |  36        |
| effective_rank                   |   39.29   |   45.21   |   0.8691 |  -5.919    |
| alpha                            |    1.909  |    2.746  |   0.6953 |  -0.8368   |
| epps_pulley                      |  441      |  487      |   0.9055 | -46.01     |
| kurt_topeig.worst                |    0.7846 |    3.534  |   0.222  |  -2.75     |



**teacher.h.cls.L09 vs teacher.z.dino.tap1**


| metric                           |   value_h |    value_z |    tau |      delta |
|:---------------------------------|----------:|-----------:|-------:|-----------:|
| uniformity                       |   -1.521  |   -3.178   | 0.4784 |    1.658   |
| variance_floor.hinge             |    0.4195 |    0.6548  | 0.6406 |   -0.2353  |
| offdiag_redundancy.mean_abs_corr |    0.1343 |    0.05789 | 2.32   |    0.07641 |
| rankme                           |  131.5    | 1119       | 0.1175 | -987.2     |
| effective_rank                   |   39.29   |  146.8     | 0.2676 | -107.5     |
| alpha                            |    1.909  |    1.179   | 1.619  |    0.7303  |
| epps_pulley                      |  441      |  219.6     | 2.009  |  221.4     |
| kurt_topeig.worst                |    0.7846 |    1.268   | 0.619  |   -0.4829  |



**teacher.h.cls.L09 vs teacher.z.dino.tap2**


| metric                           |   value_h |    value_z |    tau |       delta |
|:---------------------------------|----------:|-----------:|-------:|------------:|
| uniformity                       |   -1.521  |   -3.542   | 0.4293 |     2.022   |
| variance_floor.hinge             |    0.4195 |    0.5589  | 0.7505 |    -0.1394  |
| offdiag_redundancy.mean_abs_corr |    0.1343 |    0.05524 | 2.431  |     0.07906 |
| rankme                           |  131.5    | 1196       | 0.1099 | -1065       |
| effective_rank                   |   39.29   |  202       | 0.1945 |  -162.7     |
| alpha                            |    1.909  |    1.108   | 1.723  |     0.8013  |
| epps_pulley                      |  441      |  200.8     | 2.197  |   240.2     |
| kurt_topeig.worst                |    0.7846 |    4.293   | 0.1828 |    -3.508   |



**teacher.h.cls.L12 vs teacher.z.dino.bottleneck**


| metric                           |   value_h |   value_z |      tau |       delta |
|:---------------------------------|----------:|----------:|---------:|------------:|
| uniformity                       |   -2.024  |   -3.376  |   0.5996 |    1.352    |
| variance_floor.hinge             |    0.3155 |    0      | inf      |    0.3155   |
| offdiag_redundancy.mean_abs_corr |    0.1293 |    0.1271 |   1.017  |    0.002213 |
| rankme                           |  140.1    |   95.5    |   1.467  |   44.63     |
| effective_rank                   |   44.87   |   45.21   |   0.9925 |   -0.3385   |
| alpha                            |    1.98   |    2.746  |   0.721  |   -0.7661   |
| epps_pulley                      |  373.4    |  487      |   0.7668 | -113.6      |
| kurt_topeig.worst                |    0.8217 |    3.534  |   0.2325 |   -2.712    |



**teacher.h.cls.L12 vs teacher.z.dino.tap1**


| metric                           |   value_h |    value_z |    tau |     delta |
|:---------------------------------|----------:|-----------:|-------:|----------:|
| uniformity                       |   -2.024  |   -3.178   | 0.6369 |    1.154  |
| variance_floor.hinge             |    0.3155 |    0.6548  | 0.4817 |   -0.3394 |
| offdiag_redundancy.mean_abs_corr |    0.1293 |    0.05789 | 2.233  |    0.0714 |
| rankme                           |  140.1    | 1119       | 0.1253 | -978.6    |
| effective_rank                   |   44.87   |  146.8     | 0.3056 | -102      |
| alpha                            |    1.98   |    1.179   | 1.679  |    0.801  |
| epps_pulley                      |  373.4    |  219.6     | 1.701  |  153.9    |
| kurt_topeig.worst                |    0.8217 |    1.268   | 0.6483 |   -0.4458 |



**teacher.h.cls.L12 vs teacher.z.dino.tap2**


| metric                           |   value_h |    value_z |    tau |       delta |
|:---------------------------------|----------:|-----------:|-------:|------------:|
| uniformity                       |   -2.024  |   -3.542   | 0.5715 |     1.518   |
| variance_floor.hinge             |    0.3155 |    0.5589  | 0.5644 |    -0.2434  |
| offdiag_redundancy.mean_abs_corr |    0.1293 |    0.05524 | 2.34   |     0.07405 |
| rankme                           |  140.1    | 1196       | 0.1171 | -1056       |
| effective_rank                   |   44.87   |  202       | 0.2222 |  -157.1     |
| alpha                            |    1.98   |    1.108   | 1.787  |     0.872   |
| epps_pulley                      |  373.4    |  200.8     | 1.86   |   172.7     |
| kurt_topeig.worst                |    0.8217 |    4.293   | 0.1914 |    -3.471   |



**teacher.h.gap vs teacher.z.dino.bottleneck**


| metric                           |   value_h |   value_z |      tau |      delta |
|:---------------------------------|----------:|----------:|---------:|-----------:|
| uniformity                       |   -2.572  |   -3.376  |   0.7618 |   0.8043   |
| variance_floor.hinge             |    0.5717 |    0      | inf      |   0.5717   |
| offdiag_redundancy.mean_abs_corr |    0.1367 |    0.1271 |   1.075  |   0.009576 |
| rankme                           |  160.6    |   95.5    |   1.682  |  65.15     |
| effective_rank                   |   35.21   |   45.21   |   0.779  |  -9.992    |
| alpha                            |    1.792  |    2.746  |   0.6527 |  -0.9536   |
| epps_pulley                      |  410.7    |  487      |   0.8434 | -76.27     |
| kurt_topeig.worst                |    2.249  |    3.534  |   0.6363 |  -1.285    |



**teacher.h.gap vs teacher.z.dino.tap1**


| metric                           |   value_h |    value_z |    tau |      delta |
|:---------------------------------|----------:|-----------:|-------:|-----------:|
| uniformity                       |   -2.572  |   -3.178   | 0.8092 |    0.6065  |
| variance_floor.hinge             |    0.5717 |    0.6548  | 0.8731 |   -0.08311 |
| offdiag_redundancy.mean_abs_corr |    0.1367 |    0.05789 | 2.36   |    0.07876 |
| rankme                           |  160.6    | 1119       | 0.1436 | -958.1     |
| effective_rank                   |   35.21   |  146.8     | 0.2398 | -111.6     |
| alpha                            |    1.792  |    1.179   | 1.52   |    0.6134  |
| epps_pulley                      |  410.7    |  219.6     | 1.871  |  191.2     |
| kurt_topeig.worst                |    2.249  |    1.268   | 1.774  |    0.9812  |



**teacher.h.gap vs teacher.z.dino.tap2**


| metric                           |   value_h |    value_z |    tau |       delta |
|:---------------------------------|----------:|-----------:|-------:|------------:|
| uniformity                       |   -2.572  |   -3.542   | 0.7261 |     0.9703  |
| variance_floor.hinge             |    0.5717 |    0.5589  | 1.023  |     0.01282 |
| offdiag_redundancy.mean_abs_corr |    0.1367 |    0.05524 | 2.474  |     0.08141 |
| rankme                           |  160.6    | 1196       | 0.1343 | -1036       |
| effective_rank                   |   35.21   |  202       | 0.1744 |  -166.7     |
| alpha                            |    1.792  |    1.108   | 1.618  |     0.6845  |
| epps_pulley                      |  410.7    |  200.8     | 2.046  |   210       |
| kurt_topeig.worst                |    2.249  |    4.293   | 0.5238 |    -2.044   |



**teacher.h.gap.L03 vs teacher.z.dino.bottleneck**


| metric                           |   value_h |   value_z |      tau |     delta |
|:---------------------------------|----------:|----------:|---------:|----------:|
| uniformity                       |   -1.255  |   -3.376  |   0.3718 |   2.121   |
| variance_floor.hinge             |    0.756  |    0      | inf      |   0.756   |
| offdiag_redundancy.mean_abs_corr |    0.1844 |    0.1271 |   1.451  |   0.05731 |
| rankme                           |  105.1    |   95.5    |   1.101  |   9.646   |
| effective_rank                   |   17.96   |   45.21   |   0.3972 | -27.25    |
| alpha                            |    1.895  |    2.746  |   0.6901 |  -0.8509  |
| epps_pulley                      |  755.8    |  487      |   1.552  | 268.8     |
| kurt_topeig.worst                |    1.876  |    3.534  |   0.5307 |  -1.659   |



**teacher.h.gap.L03 vs teacher.z.dino.tap1**


| metric                           |   value_h |    value_z |     tau |      delta |
|:---------------------------------|----------:|-----------:|--------:|-----------:|
| uniformity                       |   -1.255  |   -3.178   | 0.395   |     1.923  |
| variance_floor.hinge             |    0.756  |    0.6548  | 1.155   |     0.1012 |
| offdiag_redundancy.mean_abs_corr |    0.1844 |    0.05789 | 3.185   |     0.1265 |
| rankme                           |  105.1    | 1119       | 0.09399 | -1014      |
| effective_rank                   |   17.96   |  146.8     | 0.1223  |  -128.9    |
| alpha                            |    1.895  |    1.179   | 1.607   |     0.7161 |
| epps_pulley                      |  755.8    |  219.6     | 3.442   |   536.3    |
| kurt_topeig.worst                |    1.876  |    1.268   | 1.48    |     0.6081 |



**teacher.h.gap.L03 vs teacher.z.dino.tap2**


| metric                           |   value_h |    value_z |     tau |      delta |
|:---------------------------------|----------:|-----------:|--------:|-----------:|
| uniformity                       |   -1.255  |   -3.542   | 0.3544  |     2.287  |
| variance_floor.hinge             |    0.756  |    0.5589  | 1.353   |     0.1971 |
| offdiag_redundancy.mean_abs_corr |    0.1844 |    0.05524 | 3.338   |     0.1291 |
| rankme                           |  105.1    | 1196       | 0.08788 | -1091      |
| effective_rank                   |   17.96   |  202       | 0.08891 |  -184      |
| alpha                            |    1.895  |    1.108   | 1.71    |     0.7872 |
| epps_pulley                      |  755.8    |  200.8     | 3.765   |   555.1    |
| kurt_topeig.worst                |    1.876  |    4.293   | 0.4369  |    -2.417  |



**teacher.h.gap.L06 vs teacher.z.dino.bottleneck**


| metric                           |   value_h |   value_z |      tau |     delta |
|:---------------------------------|----------:|----------:|---------:|----------:|
| uniformity                       |   -1.816  |   -3.376  |   0.5378 |   1.56    |
| variance_floor.hinge             |    0.6825 |    0      | inf      |   0.6825  |
| offdiag_redundancy.mean_abs_corr |    0.161  |    0.1271 |   1.267  |   0.03388 |
| rankme                           |  135      |   95.5    |   1.413  |  39.48    |
| effective_rank                   |   26.87   |   45.21   |   0.5945 | -18.33    |
| alpha                            |    1.849  |    2.746  |   0.6733 |  -0.897   |
| epps_pulley                      |  556.3    |  487      |   1.142  |  69.31    |
| kurt_topeig.worst                |    2.988  |    3.534  |   0.8454 |  -0.5463  |



**teacher.h.gap.L06 vs teacher.z.dino.tap1**


| metric                           |   value_h |    value_z |    tau |      delta |
|:---------------------------------|----------:|-----------:|-------:|-----------:|
| uniformity                       |   -1.816  |   -3.178   | 0.5713 |    1.363   |
| variance_floor.hinge             |    0.6825 |    0.6548  | 1.042  |    0.02773 |
| offdiag_redundancy.mean_abs_corr |    0.161  |    0.05789 | 2.78   |    0.1031  |
| rankme                           |  135      | 1119       | 0.1207 | -983.8     |
| effective_rank                   |   26.87   |  146.8     | 0.183  | -120       |
| alpha                            |    1.849  |    1.179   | 1.568  |    0.67    |
| epps_pulley                      |  556.3    |  219.6     | 2.534  |  336.8     |
| kurt_topeig.worst                |    2.988  |    1.268   | 2.357  |    1.72    |



**teacher.h.gap.L06 vs teacher.z.dino.tap2**


| metric                           |   value_h |    value_z |    tau |      delta |
|:---------------------------------|----------:|-----------:|-------:|-----------:|
| uniformity                       |   -1.816  |   -3.542   | 0.5126 |     1.726  |
| variance_floor.hinge             |    0.6825 |    0.5589  | 1.221  |     0.1236 |
| offdiag_redundancy.mean_abs_corr |    0.161  |    0.05524 | 2.914  |     0.1057 |
| rankme                           |  135      | 1196       | 0.1128 | -1062      |
| effective_rank                   |   26.87   |  202       | 0.1331 |  -175.1    |
| alpha                            |    1.849  |    1.108   | 1.669  |     0.7411 |
| epps_pulley                      |  556.3    |  200.8     | 2.771  |   355.6    |
| kurt_topeig.worst                |    2.988  |    4.293   | 0.696  |    -1.305  |



**teacher.h.gap.L09 vs teacher.z.dino.bottleneck**


| metric                           |   value_h |   value_z |      tau |     delta |
|:---------------------------------|----------:|----------:|---------:|----------:|
| uniformity                       |   -2.266  |   -3.376  |   0.6711 |   1.11    |
| variance_floor.hinge             |    0.6213 |    0      | inf      |   0.6213  |
| offdiag_redundancy.mean_abs_corr |    0.1441 |    0.1271 |   1.134  |   0.01703 |
| rankme                           |  156.5    |   95.5    |   1.638  |  60.96    |
| effective_rank                   |   32.87   |   45.21   |   0.7272 | -12.33    |
| alpha                            |    1.756  |    2.746  |   0.6394 |  -0.9904  |
| epps_pulley                      |  455.5    |  487      |   0.9353 | -31.49    |
| kurt_topeig.worst                |    1.418  |    3.534  |   0.4012 |  -2.116   |



**teacher.h.gap.L09 vs teacher.z.dino.tap1**


| metric                           |   value_h |    value_z |    tau |      delta |
|:---------------------------------|----------:|-----------:|-------:|-----------:|
| uniformity                       |   -2.266  |   -3.178   | 0.7129 |    0.9126  |
| variance_floor.hinge             |    0.6213 |    0.6548  | 0.9488 |   -0.03351 |
| offdiag_redundancy.mean_abs_corr |    0.1441 |    0.05789 | 2.489  |    0.08622 |
| rankme                           |  156.5    | 1119       | 0.1398 | -962.3     |
| effective_rank                   |   32.87   |  146.8     | 0.2239 | -114       |
| alpha                            |    1.756  |    1.179   | 1.489  |    0.5767  |
| epps_pulley                      |  455.5    |  219.6     | 2.075  |  236       |
| kurt_topeig.worst                |    1.418  |    1.268   | 1.119  |    0.1505  |



**teacher.h.gap.L09 vs teacher.z.dino.tap2**


| metric                           |   value_h |    value_z |    tau |       delta |
|:---------------------------------|----------:|-----------:|-------:|------------:|
| uniformity                       |   -2.266  |   -3.542   | 0.6397 |     1.276   |
| variance_floor.hinge             |    0.6213 |    0.5589  | 1.112  |     0.06241 |
| offdiag_redundancy.mean_abs_corr |    0.1441 |    0.05524 | 2.609  |     0.08887 |
| rankme                           |  156.5    | 1196       | 0.1308 | -1040       |
| effective_rank                   |   32.87   |  202       | 0.1628 |  -169.1     |
| alpha                            |    1.756  |    1.108   | 1.585  |     0.6477  |
| epps_pulley                      |  455.5    |  200.8     | 2.269  |   254.8     |
| kurt_topeig.worst                |    1.418  |    4.293   | 0.3303 |    -2.875   |



**teacher.h.gap.L12 vs teacher.z.dino.bottleneck**


| metric                           |   value_h |   value_z |      tau |      delta |
|:---------------------------------|----------:|----------:|---------:|-----------:|
| uniformity                       |   -2.572  |   -3.376  |   0.7618 |   0.8043   |
| variance_floor.hinge             |    0.5717 |    0      | inf      |   0.5717   |
| offdiag_redundancy.mean_abs_corr |    0.1367 |    0.1271 |   1.075  |   0.009576 |
| rankme                           |  160.6    |   95.5    |   1.682  |  65.15     |
| effective_rank                   |   35.21   |   45.21   |   0.779  |  -9.992    |
| alpha                            |    1.792  |    2.746  |   0.6527 |  -0.9536   |
| epps_pulley                      |  410.7    |  487      |   0.8434 | -76.27     |
| kurt_topeig.worst                |    2.249  |    3.534  |   0.6363 |  -1.285    |



**teacher.h.gap.L12 vs teacher.z.dino.tap1**


| metric                           |   value_h |    value_z |    tau |      delta |
|:---------------------------------|----------:|-----------:|-------:|-----------:|
| uniformity                       |   -2.572  |   -3.178   | 0.8092 |    0.6065  |
| variance_floor.hinge             |    0.5717 |    0.6548  | 0.8731 |   -0.08311 |
| offdiag_redundancy.mean_abs_corr |    0.1367 |    0.05789 | 2.36   |    0.07876 |
| rankme                           |  160.6    | 1119       | 0.1436 | -958.1     |
| effective_rank                   |   35.21   |  146.8     | 0.2398 | -111.6     |
| alpha                            |    1.792  |    1.179   | 1.52   |    0.6134  |
| epps_pulley                      |  410.7    |  219.6     | 1.871  |  191.2     |
| kurt_topeig.worst                |    2.249  |    1.268   | 1.774  |    0.9812  |



**teacher.h.gap.L12 vs teacher.z.dino.tap2**


| metric                           |   value_h |    value_z |    tau |       delta |
|:---------------------------------|----------:|-----------:|-------:|------------:|
| uniformity                       |   -2.572  |   -3.542   | 0.7261 |     0.9703  |
| variance_floor.hinge             |    0.5717 |    0.5589  | 1.023  |     0.01282 |
| offdiag_redundancy.mean_abs_corr |    0.1367 |    0.05524 | 2.474  |     0.08141 |
| rankme                           |  160.6    | 1196       | 0.1343 | -1036       |
| effective_rank                   |   35.21   |  202       | 0.1744 |  -166.7     |
| alpha                            |    1.792  |    1.108   | 1.618  |     0.6845  |
| epps_pulley                      |  410.7    |  200.8     | 2.046  |   210       |
| kurt_topeig.worst                |    2.249  |    4.293   | 0.5238 |    -2.044   |



### Pair metrics (alignment ↓ = more view-invariant)


| manifest                   | space                     | metric         |   value |   ci_lo |   ci_hi |
|:---------------------------|:--------------------------|:---------------|--------:|--------:|--------:|
| in100.pairs100.v1@audit_v1 | student.h.cls             | alignment      |  0.5046 |  0.499  |  0.5093 |
| in100.pairs100.v1@audit_v1 | student.h.cls             | cos_invariance |  0.7477 |  0.7446 |  0.7501 |
| in100.pairs100.v1@audit_v1 | student.h.gap             | alignment      |  0.6302 |  0.6239 |  0.6364 |
| in100.pairs100.v1@audit_v1 | student.h.gap             | cos_invariance |  0.6849 |  0.6817 |  0.6878 |
| in100.pairs100.v1@audit_v1 | student.z.dino.bottleneck | alignment      |  0.5293 |  0.5227 |  0.5355 |
| in100.pairs100.v1@audit_v1 | student.z.dino.bottleneck | cos_invariance |  0.7354 |  0.7314 |  0.7391 |
| in100.pairs100.v1@audit_v1 | student.z.dino.tap1       | alignment      |  0.751  |  0.7439 |  0.7579 |
| in100.pairs100.v1@audit_v1 | student.z.dino.tap1       | cos_invariance |  0.6245 |  0.6199 |  0.6281 |
| in100.pairs100.v1@audit_v1 | student.z.dino.tap2       | alignment      |  0.7587 |  0.7512 |  0.766  |
| in100.pairs100.v1@audit_v1 | student.z.dino.tap2       | cos_invariance |  0.6207 |  0.6161 |  0.6247 |
| in100.pairs100.v1@audit_v1 | teacher.h.cls             | alignment      |  0.4913 |  0.486  |  0.4957 |
| in100.pairs100.v1@audit_v1 | teacher.h.cls             | cos_invariance |  0.7543 |  0.7512 |  0.7567 |
| in100.pairs100.v1@audit_v1 | teacher.h.gap             | alignment      |  0.6165 |  0.61   |  0.6222 |
| in100.pairs100.v1@audit_v1 | teacher.h.gap             | cos_invariance |  0.6918 |  0.6885 |  0.6947 |
| in100.pairs100.v1@audit_v1 | teacher.z.dino.bottleneck | alignment      |  0.4998 |  0.4932 |  0.506  |
| in100.pairs100.v1@audit_v1 | teacher.z.dino.bottleneck | cos_invariance |  0.7501 |  0.7458 |  0.7533 |
| in100.pairs100.v1@audit_v1 | teacher.z.dino.tap1       | alignment      |  0.7348 |  0.7279 |  0.7421 |
| in100.pairs100.v1@audit_v1 | teacher.z.dino.tap1       | cos_invariance |  0.6326 |  0.6281 |  0.6362 |
| in100.pairs100.v1@audit_v1 | teacher.z.dino.tap2       | alignment      |  0.732  |  0.7243 |  0.7401 |
| in100.pairs100.v1@audit_v1 | teacher.z.dino.tap2       | cos_invariance |  0.634  |  0.6295 |  0.638  |
| in100.pairs100.v1@own_dino | student.h.cls             | alignment      |  0.2579 |  0.2537 |  0.2622 |
| in100.pairs100.v1@own_dino | student.h.cls             | cos_invariance |  0.871  |  0.8692 |  0.8735 |
| in100.pairs100.v1@own_dino | student.h.gap             | alignment      |  0.364  |  0.36   |  0.368  |
| in100.pairs100.v1@own_dino | student.h.gap             | cos_invariance |  0.818  |  0.8164 |  0.8203 |
| in100.pairs100.v1@own_dino | student.z.dino.bottleneck | alignment      |  0.2007 |  0.1969 |  0.2042 |
| in100.pairs100.v1@own_dino | student.z.dino.bottleneck | cos_invariance |  0.8996 |  0.8977 |  0.9016 |
| in100.pairs100.v1@own_dino | student.z.dino.tap1       | alignment      |  0.3636 |  0.3573 |  0.3697 |
| in100.pairs100.v1@own_dino | student.z.dino.tap1       | cos_invariance |  0.8182 |  0.8152 |  0.8215 |
| in100.pairs100.v1@own_dino | student.z.dino.tap2       | alignment      |  0.3381 |  0.332  |  0.3439 |
| in100.pairs100.v1@own_dino | student.z.dino.tap2       | cos_invariance |  0.831  |  0.8281 |  0.8342 |
| in100.pairs100.v1@own_dino | teacher.h.cls             | alignment      |  0.2487 |  0.2445 |  0.2529 |
| in100.pairs100.v1@own_dino | teacher.h.cls             | cos_invariance |  0.8757 |  0.874  |  0.8781 |
| in100.pairs100.v1@own_dino | teacher.h.gap             | alignment      |  0.3481 |  0.3443 |  0.352  |
| in100.pairs100.v1@own_dino | teacher.h.gap             | cos_invariance |  0.826  |  0.8244 |  0.8283 |
| in100.pairs100.v1@own_dino | teacher.z.dino.bottleneck | alignment      |  0.1866 |  0.183  |  0.1901 |
| in100.pairs100.v1@own_dino | teacher.z.dino.bottleneck | cos_invariance |  0.9067 |  0.905  |  0.9086 |
| in100.pairs100.v1@own_dino | teacher.z.dino.tap1       | alignment      |  0.3546 |  0.3485 |  0.361  |
| in100.pairs100.v1@own_dino | teacher.z.dino.tap1       | cos_invariance |  0.8227 |  0.8198 |  0.8262 |
| in100.pairs100.v1@own_dino | teacher.z.dino.tap2       | alignment      |  0.3239 |  0.3179 |  0.3296 |
| in100.pairs100.v1@own_dino | teacher.z.dino.tap2       | cos_invariance |  0.8381 |  0.8352 |  0.8413 |



### Cross-space similarity (CKA is contested — read as the triple)


| space                                   | metric              |   value |
|:----------------------------------------|:--------------------|--------:|
| teacher.h.cls~teacher.z.dino.bottleneck | cka_linear          |  0.717  |
| teacher.h.cls~teacher.z.dino.bottleneck | neighbor_jaccard    |  0.302  |
| teacher.h.cls~teacher.z.dino.bottleneck | procrustes_distance |  0.6808 |
| teacher.h.cls~teacher.z.dino.bottleneck | knn_label_agreement |  0.4972 |
| teacher.h.cls~teacher.z.dino.tap1       | cka_linear          |  0.9387 |
| teacher.h.cls~teacher.z.dino.tap1       | neighbor_jaccard    |  0.5506 |
| teacher.h.cls~teacher.z.dino.tap1       | procrustes_distance |  0.4434 |
| teacher.h.cls~teacher.z.dino.tap1       | knn_label_agreement |  0.644  |
| teacher.h.cls~teacher.z.dino.tap2       | cka_linear          |  0.7996 |
| teacher.h.cls~teacher.z.dino.tap2       | neighbor_jaccard    |  0.3698 |
| teacher.h.cls~teacher.z.dino.tap2       | procrustes_distance |  0.6732 |
| teacher.h.cls~teacher.z.dino.tap2       | knn_label_agreement |  0.5272 |



### Probes


| space                     |   knn_v1_k20 |   knn_v1_k200 |   linear_house_v1 |   linear_l2_v1 |   linear_raw_v1 |
|:--------------------------|-------------:|--------------:|------------------:|---------------:|----------------:|
| student.h.cls             |       0.4762 |        0.4528 |            0.5392 |         0.4888 |          0.5378 |
| student.h.cls.L03         |       0.1798 |        0.1724 |            0.2676 |         0.1766 |          0.2528 |
| student.h.cls.L06         |       0.2980 |        0.2792 |            0.3890 |         0.2978 |          0.3798 |
| student.h.cls.L09         |       0.4090 |        0.3814 |            0.4962 |         0.4282 |          0.4926 |
| student.h.cls.L12         |       0.4762 |        0.4528 |            0.5392 |         0.4888 |          0.5372 |
| student.h.gap             |       0.3952 |        0.3682 |            0.5022 |         0.4346 |          0.4958 |
| student.h.gap.L03         |       0.2198 |        0.2108 |            0.3256 |         0.2428 |          0.3006 |
| student.h.gap.L06         |       0.3038 |        0.2800 |            0.4150 |         0.3180 |          0.3906 |
| student.h.gap.L09         |       0.3622 |        0.3352 |            0.4788 |         0.3934 |          0.4632 |
| student.h.gap.L12         |       0.3952 |        0.3682 |            0.5022 |         0.4346 |          0.4958 |
| student.z.dino.bottleneck |       0.4276 |        0.4102 |            0.4562 |         0.4316 |          0.4570 |
| student.z.dino.tap1       |       0.4738 |        0.4434 |            0.5322 |         0.5152 |          0.5434 |
| student.z.dino.tap2       |       0.4374 |        0.4300 |            0.4934 |         0.4824 |          0.5018 |
| teacher.h.cls             |       0.4846 |        0.4590 |            0.5468 |         0.4926 |          0.5444 |
| teacher.h.cls.L03         |       0.1860 |        0.1822 |            0.2782 |         0.1846 |          0.2660 |
| teacher.h.cls.L06         |       0.2946 |        0.2788 |            0.3964 |         0.3018 |          0.3858 |
| teacher.h.cls.L09         |       0.4140 |        0.3874 |            0.5034 |         0.4342 |          0.4982 |
| teacher.h.cls.L12         |       0.4846 |        0.4590 |            0.5468 |         0.4926 |          0.5444 |
| teacher.h.gap             |       0.4050 |        0.3794 |            0.5102 |         0.4410 |          0.5040 |
| teacher.h.gap.L03         |       0.2264 |        0.2068 |            0.3308 |         0.2464 |          0.3068 |
| teacher.h.gap.L06         |       0.3060 |        0.2874 |            0.4208 |         0.3236 |          0.3968 |
| teacher.h.gap.L09         |       0.3678 |        0.3406 |            0.4878 |         0.3976 |          0.4676 |
| teacher.h.gap.L12         |       0.4050 |        0.3794 |            0.5102 |         0.4410 |          0.5040 |
| teacher.z.dino.bottleneck |       0.4274 |        0.4202 |            0.4662 |         0.4416 |          0.4634 |
| teacher.z.dino.tap1       |       0.4754 |        0.4510 |            0.5380 |         0.5212 |          0.5494 |
| teacher.z.dino.tap2       |       0.4386 |        0.4378 |            0.4980 |         0.4888 |          0.5090 |



## in100.dino-ctrl.ep50.ext

### Battery (variant raw|full)

| model.space                                          |   rankme |   effective_rank |   alpha |   epps_pulley |   kurt_topeig.worst |   uniformity |   offdiag_redundancy.mean_abs_corr |   variance_floor.hinge |
|:-----------------------------------------------------|---------:|-----------------:|--------:|--------------:|--------------------:|-------------:|-----------------------------------:|-----------------------:|
| in100.dino-ctrl.ep50.ext · student.h.cls             |   199.3  |            80.44 |  1.562  |        212.6  |              0.8212 |      -3.063  |                            0.09755 |                0.07919 |
| in100.dino-ctrl.ep50.ext · student.h.cls.L03         |    74.09 |            22.53 |  1.947  |        802.5  |              1.306  |      -0.5464 |                            0.1929  |                0.6052  |
| in100.dino-ctrl.ep50.ext · student.h.cls.L06         |   150.1  |            50.55 |  1.664  |        321.9  |              0.7279 |      -1.31   |                            0.1252  |                0.387   |
| in100.dino-ctrl.ep50.ext · student.h.cls.L09         |   196.6  |            75.9  |  1.498  |        230.1  |              0.8588 |      -2.421  |                            0.09961 |                0.1696  |
| in100.dino-ctrl.ep50.ext · student.h.cls.L12         |   199.3  |            80.44 |  1.562  |        212.6  |              0.8212 |      -3.063  |                            0.09755 |                0.07919 |
| in100.dino-ctrl.ep50.ext · student.h.gap             |   193.8  |            56.53 |  1.587  |        330.7  |              2.177  |      -2.556  |                            0.119   |                0.5726  |
| in100.dino-ctrl.ep50.ext · student.h.gap.L03         |   123.6  |            22.57 |  1.776  |        645.9  |              1.306  |      -1.354  |                            0.169   |                0.7336  |
| in100.dino-ctrl.ep50.ext · student.h.gap.L06         |   164.3  |            42.26 |  1.649  |        423.1  |              1.752  |      -1.786  |                            0.1335  |                0.6684  |
| in100.dino-ctrl.ep50.ext · student.h.gap.L09         |   196.7  |            58.91 |  1.488  |        335.5  |              1.346  |      -2.325  |                            0.1194  |                0.6004  |
| in100.dino-ctrl.ep50.ext · student.h.gap.L12         |   193.8  |            56.53 |  1.587  |        330.7  |              2.177  |      -2.556  |                            0.119   |                0.5726  |
| in100.dino-ctrl.ep50.ext · student.z.dino.bottleneck |   137.5  |            78.07 |  2.071  |        330.8  |              8.958  |      -3.601  |                            0.09019 |                0.4167  |
| in100.dino-ctrl.ep50.ext · student.z.dino.tap1       |  1405    |           317.6  |  1.001  |         81.64 |              0.8385 |      -3.471  |                            0.04801 |                0.6062  |
| in100.dino-ctrl.ep50.ext · student.z.dino.tap2       |  1515    |           484.8  |  0.8231 |         85.7  |              9.405  |      -3.73   |                            0.02994 |                0.7776  |
| in100.dino-ctrl.ep50.ext · teacher.h.cls             |   200.1  |            80.46 |  1.563  |        218    |              0.8351 |      -3.127  |                            0.09663 |                0.07743 |
| in100.dino-ctrl.ep50.ext · teacher.h.cls.L03         |    74.79 |            23.17 |  1.941  |        794.5  |              1.374  |      -0.5477 |                            0.1881  |                0.6074  |
| in100.dino-ctrl.ep50.ext · teacher.h.cls.L06         |   152    |            49.58 |  1.664  |        338.8  |              0.8053 |      -1.386  |                            0.1256  |                0.3755  |
| in100.dino-ctrl.ep50.ext · teacher.h.cls.L09         |   198.6  |            76.15 |  1.494  |        235.4  |              0.8977 |      -2.516  |                            0.09836 |                0.1609  |
| in100.dino-ctrl.ep50.ext · teacher.h.cls.L12         |   200.1  |            80.46 |  1.563  |        218    |              0.8351 |      -3.127  |                            0.09663 |                0.07743 |
| in100.dino-ctrl.ep50.ext · teacher.h.gap             |   192.7  |            55.69 |  1.597  |        304.9  |              2.621  |      -2.58   |                            0.1189  |                0.5733  |
| in100.dino-ctrl.ep50.ext · teacher.h.gap.L03         |   124    |            22.76 |  1.771  |        596.3  |              1.697  |      -1.336  |                            0.1598  |                0.7449  |
| in100.dino-ctrl.ep50.ext · teacher.h.gap.L06         |   163    |            41.14 |  1.659  |        424.7  |              1.906  |      -1.783  |                            0.1323  |                0.6719  |
| in100.dino-ctrl.ep50.ext · teacher.h.gap.L09         |   195.7  |            57.95 |  1.496  |        336.5  |              2.326  |      -2.347  |                            0.1195  |                0.5999  |
| in100.dino-ctrl.ep50.ext · teacher.h.gap.L12         |   192.7  |            55.69 |  1.597  |        304.9  |              2.621  |      -2.58   |                            0.1189  |                0.5733  |
| in100.dino-ctrl.ep50.ext · teacher.z.dino.bottleneck |   138    |            78.64 |  2.062  |        333.6  |             13.8    |      -3.608  |                            0.08971 |                0.3994  |
| in100.dino-ctrl.ep50.ext · teacher.z.dino.tap1       |  1407    |           318.1  |  1.001  |         80.03 |              0.7784 |      -3.486  |                            0.04901 |                0.5965  |
| in100.dino-ctrl.ep50.ext · teacher.z.dino.tap2       |  1516    |           484    |  0.8207 |         82.3  |             24.56   |      -3.757  |                            0.02939 |                0.7716  |



### Transfer ratios τ = value(h) / value(z)


**student.h.cls vs student.z.dino.bottleneck**


| metric                           |   value_h |   value_z |     tau |       delta |
|:---------------------------------|----------:|----------:|--------:|------------:|
| uniformity                       |  -3.063   |  -3.601   | 0.8505  |    0.5385   |
| variance_floor.hinge             |   0.07919 |   0.4167  | 0.19    |   -0.3375   |
| offdiag_redundancy.mean_abs_corr |   0.09755 |   0.09019 | 1.082   |    0.007361 |
| rankme                           | 199.3     | 137.5     | 1.45    |   61.82     |
| effective_rank                   |  80.44    |  78.07    | 1.03    |    2.37     |
| alpha                            |   1.562   |   2.071   | 0.7539  |   -0.5099   |
| epps_pulley                      | 212.6     | 330.8     | 0.6427  | -118.2      |
| kurt_topeig.worst                |   0.8212  |   8.958   | 0.09167 |   -8.136    |



**student.h.cls vs student.z.dino.tap1**


| metric                           |   value_h |    value_z |    tau |       delta |
|:---------------------------------|----------:|-----------:|-------:|------------:|
| uniformity                       |  -3.063   |   -3.471   | 0.8822 |     0.4088  |
| variance_floor.hinge             |   0.07919 |    0.6062  | 0.1306 |    -0.527   |
| offdiag_redundancy.mean_abs_corr |   0.09755 |    0.04801 | 2.032  |     0.04954 |
| rankme                           | 199.3     | 1405       | 0.1418 | -1206       |
| effective_rank                   |  80.44    |  317.6     | 0.2533 |  -237.1     |
| alpha                            |   1.562   |    1.001   | 1.56   |     0.5603  |
| epps_pulley                      | 212.6     |   81.64    | 2.604  |   131       |
| kurt_topeig.worst                |   0.8212  |    0.8385  | 0.9794 |    -0.0173  |



**student.h.cls vs student.z.dino.tap2**


| metric                           |   value_h |    value_z |     tau |       delta |
|:---------------------------------|----------:|-----------:|--------:|------------:|
| uniformity                       |  -3.063   |   -3.73    | 0.8211  |     0.6675  |
| variance_floor.hinge             |   0.07919 |    0.7776  | 0.1018  |    -0.6984  |
| offdiag_redundancy.mean_abs_corr |   0.09755 |    0.02994 | 3.259   |     0.06762 |
| rankme                           | 199.3     | 1515       | 0.1315  | -1316       |
| effective_rank                   |  80.44    |  484.8     | 0.1659  |  -404.3     |
| alpha                            |   1.562   |    0.8231  | 1.897   |     0.7384  |
| epps_pulley                      | 212.6     |   85.7     | 2.481   |   126.9     |
| kurt_topeig.worst                |   0.8212  |    9.405   | 0.08731 |    -8.584   |



**student.h.cls.L03 vs student.z.dino.bottleneck**


| metric                           |   value_h |   value_z |    tau |    delta |
|:---------------------------------|----------:|----------:|-------:|---------:|
| uniformity                       |   -0.5464 |  -3.601   | 0.1517 |   3.055  |
| variance_floor.hinge             |    0.6052 |   0.4167  | 1.452  |   0.1885 |
| offdiag_redundancy.mean_abs_corr |    0.1929 |   0.09019 | 2.139  |   0.1027 |
| rankme                           |   74.09   | 137.5     | 0.5388 | -63.41   |
| effective_rank                   |   22.53   |  78.07    | 0.2885 | -55.54   |
| alpha                            |    1.947  |   2.071   | 0.9401 |  -0.1241 |
| epps_pulley                      |  802.5    | 330.8     | 2.426  | 471.7    |
| kurt_topeig.worst                |    1.306  |   8.958   | 0.1458 |  -7.651  |



**student.h.cls.L03 vs student.z.dino.tap1**


| metric                           |   value_h |    value_z |     tau |        delta |
|:---------------------------------|----------:|-----------:|--------:|-------------:|
| uniformity                       |   -0.5464 |   -3.471   | 0.1574  |     2.925    |
| variance_floor.hinge             |    0.6052 |    0.6062  | 0.9983  |    -0.001047 |
| offdiag_redundancy.mean_abs_corr |    0.1929 |    0.04801 | 4.018   |     0.1449   |
| rankme                           |   74.09   | 1405       | 0.05273 | -1331        |
| effective_rank                   |   22.53   |  317.6     | 0.07093 |  -295        |
| alpha                            |    1.947  |    1.001   | 1.945   |     0.946    |
| epps_pulley                      |  802.5    |   81.64    | 9.83    |   720.8      |
| kurt_topeig.worst                |    1.306  |    0.8385  | 1.558   |     0.468    |



**student.h.cls.L03 vs student.z.dino.tap2**


| metric                           |   value_h |    value_z |     tau |      delta |
|:---------------------------------|----------:|-----------:|--------:|-----------:|
| uniformity                       |   -0.5464 |   -3.73    | 0.1465  |     3.184  |
| variance_floor.hinge             |    0.6052 |    0.7776  | 0.7783  |    -0.1724 |
| offdiag_redundancy.mean_abs_corr |    0.1929 |    0.02994 | 6.443   |     0.163  |
| rankme                           |   74.09   | 1515       | 0.04889 | -1441      |
| effective_rank                   |   22.53   |  484.8     | 0.04647 |  -462.3    |
| alpha                            |    1.947  |    0.8231  | 2.366   |     1.124  |
| epps_pulley                      |  802.5    |   85.7     | 9.363   |   716.8    |
| kurt_topeig.worst                |    1.306  |    9.405   | 0.1389  |    -8.099  |



**student.h.cls.L06 vs student.z.dino.bottleneck**


| metric                           |   value_h |   value_z |     tau |     delta |
|:---------------------------------|----------:|----------:|--------:|----------:|
| uniformity                       |   -1.31   |  -3.601   | 0.3638  |   2.291   |
| variance_floor.hinge             |    0.387  |   0.4167  | 0.9289  |  -0.02965 |
| offdiag_redundancy.mean_abs_corr |    0.1252 |   0.09019 | 1.388   |   0.03499 |
| rankme                           |  150.1    | 137.5     | 1.091   |  12.56    |
| effective_rank                   |   50.55   |  78.07    | 0.6476  | -27.52    |
| alpha                            |    1.664  |   2.071   | 0.8034  |  -0.4072  |
| epps_pulley                      |  321.9    | 330.8     | 0.9731  |  -8.882   |
| kurt_topeig.worst                |    0.7279 |   8.958   | 0.08126 |  -8.23    |



**student.h.cls.L06 vs student.z.dino.tap1**


| metric                           |   value_h |    value_z |    tau |       delta |
|:---------------------------------|----------:|-----------:|-------:|------------:|
| uniformity                       |   -1.31   |   -3.471   | 0.3774 |     2.161   |
| variance_floor.hinge             |    0.387  |    0.6062  | 0.6385 |    -0.2192  |
| offdiag_redundancy.mean_abs_corr |    0.1252 |    0.04801 | 2.607  |     0.07717 |
| rankme                           |  150.1    | 1405       | 0.1068 | -1255       |
| effective_rank                   |   50.55   |  317.6     | 0.1592 |  -267       |
| alpha                            |    1.664  |    1.001   | 1.662  |     0.6629  |
| epps_pulley                      |  321.9    |   81.64    | 3.943  |   240.3     |
| kurt_topeig.worst                |    0.7279 |    0.8385  | 0.8681 |    -0.1106  |



**student.h.cls.L06 vs student.z.dino.tap2**


| metric                           |   value_h |    value_z |     tau |       delta |
|:---------------------------------|----------:|-----------:|--------:|------------:|
| uniformity                       |   -1.31   |   -3.73    | 0.3512  |     2.42    |
| variance_floor.hinge             |    0.387  |    0.7776  | 0.4978  |    -0.3905  |
| offdiag_redundancy.mean_abs_corr |    0.1252 |    0.02994 | 4.182   |     0.09525 |
| rankme                           |  150.1    | 1515       | 0.09903 | -1365       |
| effective_rank                   |   50.55   |  484.8     | 0.1043  |  -434.2     |
| alpha                            |    1.664  |    0.8231  | 2.022   |     0.8411  |
| epps_pulley                      |  321.9    |   85.7     | 3.756   |   236.2     |
| kurt_topeig.worst                |    0.7279 |    9.405   | 0.07739 |    -8.677   |



**student.h.cls.L09 vs student.z.dino.bottleneck**


| metric                           |   value_h |   value_z |     tau |       delta |
|:---------------------------------|----------:|----------:|--------:|------------:|
| uniformity                       |  -2.421   |  -3.601   | 0.6724  |    1.18     |
| variance_floor.hinge             |   0.1696  |   0.4167  | 0.4069  |   -0.2471   |
| offdiag_redundancy.mean_abs_corr |   0.09961 |   0.09019 | 1.104   |    0.009422 |
| rankme                           | 196.6     | 137.5     | 1.43    |   59.08     |
| effective_rank                   |  75.9     |  78.07    | 0.9722  |   -2.17     |
| alpha                            |   1.498   |   2.071   | 0.723   |   -0.5738   |
| epps_pulley                      | 230.1     | 330.8     | 0.6956  | -100.7      |
| kurt_topeig.worst                |   0.8588  |   8.958   | 0.09587 |   -8.099    |



**student.h.cls.L09 vs student.z.dino.tap1**


| metric                           |   value_h |    value_z |    tau |       delta |
|:---------------------------------|----------:|-----------:|-------:|------------:|
| uniformity                       |  -2.421   |   -3.471   | 0.6976 |     1.05    |
| variance_floor.hinge             |   0.1696  |    0.6062  | 0.2797 |    -0.4366  |
| offdiag_redundancy.mean_abs_corr |   0.09961 |    0.04801 | 2.075  |     0.0516  |
| rankme                           | 196.6     | 1405       | 0.1399 | -1209       |
| effective_rank                   |  75.9     |  317.6     | 0.239  |  -241.7     |
| alpha                            |   1.498   |    1.001   | 1.496  |     0.4964  |
| epps_pulley                      | 230.1     |   81.64    | 2.819  |   148.5     |
| kurt_topeig.worst                |   0.8588  |    0.8385  | 1.024  |     0.02033 |



**student.h.cls.L09 vs student.z.dino.tap2**


| metric                           |   value_h |    value_z |     tau |       delta |
|:---------------------------------|----------:|-----------:|--------:|------------:|
| uniformity                       |  -2.421   |   -3.73    | 0.6492  |     1.308   |
| variance_floor.hinge             |   0.1696  |    0.7776  | 0.2181  |    -0.608   |
| offdiag_redundancy.mean_abs_corr |   0.09961 |    0.02994 | 3.327   |     0.06968 |
| rankme                           | 196.6     | 1515       | 0.1297  | -1319       |
| effective_rank                   |  75.9     |  484.8     | 0.1566  |  -408.9     |
| alpha                            |   1.498   |    0.8231  | 1.819   |     0.6745  |
| epps_pulley                      | 230.1     |   85.7     | 2.685   |   144.4     |
| kurt_topeig.worst                |   0.8588  |    9.405   | 0.09131 |    -8.546   |



**student.h.cls.L12 vs student.z.dino.bottleneck**


| metric                           |   value_h |   value_z |     tau |       delta |
|:---------------------------------|----------:|----------:|--------:|------------:|
| uniformity                       |  -3.063   |  -3.601   | 0.8505  |    0.5385   |
| variance_floor.hinge             |   0.07919 |   0.4167  | 0.19    |   -0.3375   |
| offdiag_redundancy.mean_abs_corr |   0.09755 |   0.09019 | 1.082   |    0.007361 |
| rankme                           | 199.3     | 137.5     | 1.45    |   61.82     |
| effective_rank                   |  80.44    |  78.07    | 1.03    |    2.37     |
| alpha                            |   1.562   |   2.071   | 0.7539  |   -0.5099   |
| epps_pulley                      | 212.6     | 330.8     | 0.6427  | -118.2      |
| kurt_topeig.worst                |   0.8212  |   8.958   | 0.09167 |   -8.136    |



**student.h.cls.L12 vs student.z.dino.tap1**


| metric                           |   value_h |    value_z |    tau |       delta |
|:---------------------------------|----------:|-----------:|-------:|------------:|
| uniformity                       |  -3.063   |   -3.471   | 0.8822 |     0.4088  |
| variance_floor.hinge             |   0.07919 |    0.6062  | 0.1306 |    -0.527   |
| offdiag_redundancy.mean_abs_corr |   0.09755 |    0.04801 | 2.032  |     0.04954 |
| rankme                           | 199.3     | 1405       | 0.1418 | -1206       |
| effective_rank                   |  80.44    |  317.6     | 0.2533 |  -237.1     |
| alpha                            |   1.562   |    1.001   | 1.56   |     0.5603  |
| epps_pulley                      | 212.6     |   81.64    | 2.604  |   131       |
| kurt_topeig.worst                |   0.8212  |    0.8385  | 0.9794 |    -0.0173  |



**student.h.cls.L12 vs student.z.dino.tap2**


| metric                           |   value_h |    value_z |     tau |       delta |
|:---------------------------------|----------:|-----------:|--------:|------------:|
| uniformity                       |  -3.063   |   -3.73    | 0.8211  |     0.6675  |
| variance_floor.hinge             |   0.07919 |    0.7776  | 0.1018  |    -0.6984  |
| offdiag_redundancy.mean_abs_corr |   0.09755 |    0.02994 | 3.259   |     0.06762 |
| rankme                           | 199.3     | 1515       | 0.1315  | -1316       |
| effective_rank                   |  80.44    |  484.8     | 0.1659  |  -404.3     |
| alpha                            |   1.562   |    0.8231  | 1.897   |     0.7384  |
| epps_pulley                      | 212.6     |   85.7     | 2.481   |   126.9     |
| kurt_topeig.worst                |   0.8212  |    9.405   | 0.08731 |    -8.584   |



**student.h.gap vs student.z.dino.bottleneck**


| metric                           |   value_h |   value_z |    tau |     delta |
|:---------------------------------|----------:|----------:|-------:|----------:|
| uniformity                       |   -2.556  |  -3.601   | 0.7098 |   1.045   |
| variance_floor.hinge             |    0.5726 |   0.4167  | 1.374  |   0.1559  |
| offdiag_redundancy.mean_abs_corr |    0.119  |   0.09019 | 1.319  |   0.02877 |
| rankme                           |  193.8    | 137.5     | 1.409  |  56.28    |
| effective_rank                   |   56.53   |  78.07    | 0.7241 | -21.54    |
| alpha                            |    1.587  |   2.071   | 0.766  |  -0.4847  |
| epps_pulley                      |  330.7    | 330.8     | 0.9996 |  -0.1374  |
| kurt_topeig.worst                |    2.177  |   8.958   | 0.2431 |  -6.78    |



**student.h.gap vs student.z.dino.tap1**


| metric                           |   value_h |    value_z |    tau |       delta |
|:---------------------------------|----------:|-----------:|-------:|------------:|
| uniformity                       |   -2.556  |   -3.471   | 0.7363 |     0.9153  |
| variance_floor.hinge             |    0.5726 |    0.6062  | 0.9445 |    -0.03364 |
| offdiag_redundancy.mean_abs_corr |    0.119  |    0.04801 | 2.478  |     0.07096 |
| rankme                           |  193.8    | 1405       | 0.1379 | -1211       |
| effective_rank                   |   56.53   |  317.6     | 0.178  |  -261       |
| alpha                            |    1.587  |    1.001   | 1.585  |     0.5854  |
| epps_pulley                      |  330.7    |   81.64    | 4.05   |   249       |
| kurt_topeig.worst                |    2.177  |    0.8385  | 2.597  |     1.339   |



**student.h.gap vs student.z.dino.tap2**


| metric                           |   value_h |    value_z |    tau |       delta |
|:---------------------------------|----------:|-----------:|-------:|------------:|
| uniformity                       |   -2.556  |   -3.73    | 0.6853 |     1.174   |
| variance_floor.hinge             |    0.5726 |    0.7776  | 0.7364 |    -0.205   |
| offdiag_redundancy.mean_abs_corr |    0.119  |    0.02994 | 3.974  |     0.08903 |
| rankme                           |  193.8    | 1515       | 0.1279 | -1322       |
| effective_rank                   |   56.53   |  484.8     | 0.1166 |  -428.2     |
| alpha                            |    1.587  |    0.8231  | 1.928  |     0.7636  |
| epps_pulley                      |  330.7    |   85.7     | 3.858  |   245       |
| kurt_topeig.worst                |    2.177  |    9.405   | 0.2315 |    -7.228   |



**student.h.gap.L03 vs student.z.dino.bottleneck**


| metric                           |   value_h |   value_z |    tau |     delta |
|:---------------------------------|----------:|----------:|-------:|----------:|
| uniformity                       |   -1.354  |  -3.601   | 0.3761 |   2.247   |
| variance_floor.hinge             |    0.7336 |   0.4167  | 1.761  |   0.3169  |
| offdiag_redundancy.mean_abs_corr |    0.169  |   0.09019 | 1.874  |   0.07883 |
| rankme                           |  123.6    | 137.5     | 0.899  | -13.88    |
| effective_rank                   |   22.57   |  78.07    | 0.2891 | -55.5     |
| alpha                            |    1.776  |   2.071   | 0.8575 |  -0.2952  |
| epps_pulley                      |  645.9    | 330.8     | 1.953  | 315.1     |
| kurt_topeig.worst                |    1.306  |   8.958   | 0.1458 |  -7.652   |



**student.h.gap.L03 vs student.z.dino.tap1**


| metric                           |   value_h |    value_z |     tau |      delta |
|:---------------------------------|----------:|-----------:|--------:|-----------:|
| uniformity                       |   -1.354  |   -3.471   | 0.3902  |     2.117  |
| variance_floor.hinge             |    0.7336 |    0.6062  | 1.21    |     0.1274 |
| offdiag_redundancy.mean_abs_corr |    0.169  |    0.04801 | 3.521   |     0.121  |
| rankme                           |  123.6    | 1405       | 0.08798 | -1282      |
| effective_rank                   |   22.57   |  317.6     | 0.07107 |  -295      |
| alpha                            |    1.776  |    1.001   | 1.774   |     0.775  |
| epps_pulley                      |  645.9    |   81.64    | 7.912   |   564.3    |
| kurt_topeig.worst                |    1.306  |    0.8385  | 1.558   |     0.4676 |



**student.h.gap.L03 vs student.z.dino.tap2**


| metric                           |   value_h |    value_z |     tau |       delta |
|:---------------------------------|----------:|-----------:|--------:|------------:|
| uniformity                       |   -1.354  |   -3.73    | 0.3631  |     2.376   |
| variance_floor.hinge             |    0.7336 |    0.7776  | 0.9434  |    -0.04398 |
| offdiag_redundancy.mean_abs_corr |    0.169  |    0.02994 | 5.646   |     0.1391  |
| rankme                           |  123.6    | 1515       | 0.08158 | -1392       |
| effective_rank                   |   22.57   |  484.8     | 0.04656 |  -462.2     |
| alpha                            |    1.776  |    0.8231  | 2.158   |     0.9531  |
| epps_pulley                      |  645.9    |   85.7     | 7.537   |   560.2     |
| kurt_topeig.worst                |    1.306  |    9.405   | 0.1389  |    -8.099   |



**student.h.gap.L06 vs student.z.dino.bottleneck**


| metric                           |   value_h |   value_z |    tau |     delta |
|:---------------------------------|----------:|----------:|-------:|----------:|
| uniformity                       |   -1.786  |  -3.601   | 0.496  |   1.815   |
| variance_floor.hinge             |    0.6684 |   0.4167  | 1.604  |   0.2517  |
| offdiag_redundancy.mean_abs_corr |    0.1335 |   0.09019 | 1.48   |   0.04331 |
| rankme                           |  164.3    | 137.5     | 1.195  |  26.82    |
| effective_rank                   |   42.26   |  78.07    | 0.5414 | -35.8     |
| alpha                            |    1.649  |   2.071   | 0.796  |  -0.4226  |
| epps_pulley                      |  423.1    | 330.8     | 1.279  |  92.33    |
| kurt_topeig.worst                |    1.752  |   8.958   | 0.1956 |  -7.206   |



**student.h.gap.L06 vs student.z.dino.tap1**


| metric                           |   value_h |    value_z |    tau |       delta |
|:---------------------------------|----------:|-----------:|-------:|------------:|
| uniformity                       |   -1.786  |   -3.471   | 0.5146 |     1.685   |
| variance_floor.hinge             |    0.6684 |    0.6062  | 1.103  |     0.06217 |
| offdiag_redundancy.mean_abs_corr |    0.1335 |    0.04801 | 2.781  |     0.08549 |
| rankme                           |  164.3    | 1405       | 0.1169 | -1241       |
| effective_rank                   |   42.26   |  317.6     | 0.1331 |  -275.3     |
| alpha                            |    1.649  |    1.001   | 1.647  |     0.6475  |
| epps_pulley                      |  423.1    |   81.64    | 5.183  |   341.5     |
| kurt_topeig.worst                |    1.752  |    0.8385  | 2.089  |     0.9134  |



**student.h.gap.L06 vs student.z.dino.tap2**


| metric                           |   value_h |    value_z |     tau |      delta |
|:---------------------------------|----------:|-----------:|--------:|-----------:|
| uniformity                       |   -1.786  |   -3.73    | 0.4789  |     1.944  |
| variance_floor.hinge             |    0.6684 |    0.7776  | 0.8596  |    -0.1092 |
| offdiag_redundancy.mean_abs_corr |    0.1335 |    0.02994 | 4.459   |     0.1036 |
| rankme                           |  164.3    | 1515       | 0.1084  | -1351      |
| effective_rank                   |   42.26   |  484.8     | 0.08718 |  -442.5    |
| alpha                            |    1.649  |    0.8231  | 2.003   |     0.8257 |
| epps_pulley                      |  423.1    |   85.7     | 4.937   |   337.4    |
| kurt_topeig.worst                |    1.752  |    9.405   | 0.1863  |    -7.653  |



**student.h.gap.L09 vs student.z.dino.bottleneck**


| metric                           |   value_h |   value_z |    tau |     delta |
|:---------------------------------|----------:|----------:|-------:|----------:|
| uniformity                       |   -2.325  |  -3.601   | 0.6456 |   1.276   |
| variance_floor.hinge             |    0.6004 |   0.4167  | 1.441  |   0.1837  |
| offdiag_redundancy.mean_abs_corr |    0.1194 |   0.09019 | 1.324  |   0.02918 |
| rankme                           |  196.7    | 137.5     | 1.43   |  59.17    |
| effective_rank                   |   58.91   |  78.07    | 0.7546 | -19.16    |
| alpha                            |    1.488  |   2.071   | 0.7186 |  -0.5829  |
| epps_pulley                      |  335.5    | 330.8     | 1.014  |   4.689   |
| kurt_topeig.worst                |    1.346  |   8.958   | 0.1503 |  -7.611   |



**student.h.gap.L09 vs student.z.dino.tap1**


| metric                           |   value_h |    value_z |    tau |        delta |
|:---------------------------------|----------:|-----------:|-------:|-------------:|
| uniformity                       |   -2.325  |   -3.471   | 0.6698 |     1.146    |
| variance_floor.hinge             |    0.6004 |    0.6062  | 0.9905 |    -0.005774 |
| offdiag_redundancy.mean_abs_corr |    0.1194 |    0.04801 | 2.486  |     0.07136  |
| rankme                           |  196.7    | 1405       | 0.14   | -1208        |
| effective_rank                   |   58.91   |  317.6     | 0.1855 |  -258.6      |
| alpha                            |    1.488  |    1.001   | 1.487  |     0.4873   |
| epps_pulley                      |  335.5    |   81.64    | 4.11   |   253.8      |
| kurt_topeig.worst                |    1.346  |    0.8385  | 1.606  |     0.5079   |



**student.h.gap.L09 vs student.z.dino.tap2**


| metric                           |   value_h |    value_z |    tau |       delta |
|:---------------------------------|----------:|-----------:|-------:|------------:|
| uniformity                       |   -2.325  |   -3.73    | 0.6233 |     1.405   |
| variance_floor.hinge             |    0.6004 |    0.7776  | 0.7722 |    -0.1771  |
| offdiag_redundancy.mean_abs_corr |    0.1194 |    0.02994 | 3.987  |     0.08943 |
| rankme                           |  196.7    | 1515       | 0.1298 | -1319       |
| effective_rank                   |   58.91   |  484.8     | 0.1215 |  -425.9     |
| alpha                            |    1.488  |    0.8231  | 1.808  |     0.6654  |
| epps_pulley                      |  335.5    |   85.7     | 3.915  |   249.8     |
| kurt_topeig.worst                |    1.346  |    9.405   | 0.1432 |    -8.059   |



**student.h.gap.L12 vs student.z.dino.bottleneck**


| metric                           |   value_h |   value_z |    tau |     delta |
|:---------------------------------|----------:|----------:|-------:|----------:|
| uniformity                       |   -2.556  |  -3.601   | 0.7098 |   1.045   |
| variance_floor.hinge             |    0.5726 |   0.4167  | 1.374  |   0.1559  |
| offdiag_redundancy.mean_abs_corr |    0.119  |   0.09019 | 1.319  |   0.02877 |
| rankme                           |  193.8    | 137.5     | 1.409  |  56.28    |
| effective_rank                   |   56.53   |  78.07    | 0.7241 | -21.54    |
| alpha                            |    1.587  |   2.071   | 0.766  |  -0.4847  |
| epps_pulley                      |  330.7    | 330.8     | 0.9996 |  -0.1374  |
| kurt_topeig.worst                |    2.177  |   8.958   | 0.2431 |  -6.78    |



**student.h.gap.L12 vs student.z.dino.tap1**


| metric                           |   value_h |    value_z |    tau |       delta |
|:---------------------------------|----------:|-----------:|-------:|------------:|
| uniformity                       |   -2.556  |   -3.471   | 0.7363 |     0.9153  |
| variance_floor.hinge             |    0.5726 |    0.6062  | 0.9445 |    -0.03364 |
| offdiag_redundancy.mean_abs_corr |    0.119  |    0.04801 | 2.478  |     0.07096 |
| rankme                           |  193.8    | 1405       | 0.1379 | -1211       |
| effective_rank                   |   56.53   |  317.6     | 0.178  |  -261       |
| alpha                            |    1.587  |    1.001   | 1.585  |     0.5854  |
| epps_pulley                      |  330.7    |   81.64    | 4.05   |   249       |
| kurt_topeig.worst                |    2.177  |    0.8385  | 2.597  |     1.339   |



**student.h.gap.L12 vs student.z.dino.tap2**


| metric                           |   value_h |    value_z |    tau |       delta |
|:---------------------------------|----------:|-----------:|-------:|------------:|
| uniformity                       |   -2.556  |   -3.73    | 0.6853 |     1.174   |
| variance_floor.hinge             |    0.5726 |    0.7776  | 0.7364 |    -0.205   |
| offdiag_redundancy.mean_abs_corr |    0.119  |    0.02994 | 3.974  |     0.08903 |
| rankme                           |  193.8    | 1515       | 0.1279 | -1322       |
| effective_rank                   |   56.53   |  484.8     | 0.1166 |  -428.2     |
| alpha                            |    1.587  |    0.8231  | 1.928  |     0.7636  |
| epps_pulley                      |  330.7    |   85.7     | 3.858  |   245       |
| kurt_topeig.worst                |    2.177  |    9.405   | 0.2315 |    -7.228   |



**teacher.h.cls vs teacher.z.dino.bottleneck**


| metric                           |   value_h |   value_z |     tau |       delta |
|:---------------------------------|----------:|----------:|--------:|------------:|
| uniformity                       |  -3.127   |  -3.608   | 0.8666  |    0.4812   |
| variance_floor.hinge             |   0.07743 |   0.3994  | 0.1939  |   -0.322    |
| offdiag_redundancy.mean_abs_corr |   0.09663 |   0.08971 | 1.077   |    0.006915 |
| rankme                           | 200.1     | 138       | 1.45    |   62.08     |
| effective_rank                   |  80.46    |  78.64    | 1.023   |    1.811    |
| alpha                            |   1.563   |   2.062   | 0.7581  |   -0.4989   |
| epps_pulley                      | 218       | 333.6     | 0.6535  | -115.6      |
| kurt_topeig.worst                |   0.8351  |  13.8     | 0.06051 |  -12.97     |



**teacher.h.cls vs teacher.z.dino.tap1**


| metric                           |   value_h |    value_z |    tau |       delta |
|:---------------------------------|----------:|-----------:|-------:|------------:|
| uniformity                       |  -3.127   |   -3.486   | 0.8971 |     0.3588  |
| variance_floor.hinge             |   0.07743 |    0.5965  | 0.1298 |    -0.5191  |
| offdiag_redundancy.mean_abs_corr |   0.09663 |    0.04901 | 1.972  |     0.04761 |
| rankme                           | 200.1     | 1407       | 0.1423 | -1206       |
| effective_rank                   |  80.46    |  318.1     | 0.2529 |  -237.7     |
| alpha                            |   1.563   |    1.001   | 1.561  |     0.5618  |
| epps_pulley                      | 218       |   80.03    | 2.724  |   138       |
| kurt_topeig.worst                |   0.8351  |    0.7784  | 1.073  |     0.05668 |



**teacher.h.cls vs teacher.z.dino.tap2**


| metric                           |   value_h |    value_z |    tau |       delta |
|:---------------------------------|----------:|-----------:|-------:|------------:|
| uniformity                       |  -3.127   |   -3.757   | 0.8322 |     0.6305  |
| variance_floor.hinge             |   0.07743 |    0.7716  | 0.1004 |    -0.6941  |
| offdiag_redundancy.mean_abs_corr |   0.09663 |    0.02939 | 3.287  |     0.06723 |
| rankme                           | 200.1     | 1516       | 0.132  | -1316       |
| effective_rank                   |  80.46    |  484       | 0.1662 |  -403.6     |
| alpha                            |   1.563   |    0.8207  | 1.905  |     0.7425  |
| epps_pulley                      | 218       |   82.3     | 2.649  |   135.7     |
| kurt_topeig.worst                |   0.8351  |   24.56    | 0.034  |   -23.73    |



**teacher.h.cls.L03 vs teacher.z.dino.bottleneck**


| metric                           |   value_h |   value_z |     tau |     delta |
|:---------------------------------|----------:|----------:|--------:|----------:|
| uniformity                       |   -0.5477 |  -3.608   | 0.1518  |   3.061   |
| variance_floor.hinge             |    0.6074 |   0.3994  | 1.521   |   0.208   |
| offdiag_redundancy.mean_abs_corr |    0.1881 |   0.08971 | 2.096   |   0.09836 |
| rankme                           |   74.79   | 138       | 0.5419  | -63.22    |
| effective_rank                   |   23.17   |  78.64    | 0.2946  | -55.48    |
| alpha                            |    1.941  |   2.062   | 0.9411  |  -0.1214  |
| epps_pulley                      |  794.5    | 333.6     | 2.382   | 460.9     |
| kurt_topeig.worst                |    1.374  |  13.8     | 0.09957 | -12.43    |



**teacher.h.cls.L03 vs teacher.z.dino.tap1**


| metric                           |   value_h |    value_z |     tau |       delta |
|:---------------------------------|----------:|-----------:|--------:|------------:|
| uniformity                       |   -0.5477 |   -3.486   | 0.1571  |     2.938   |
| variance_floor.hinge             |    0.6074 |    0.5965  | 1.018   |     0.01088 |
| offdiag_redundancy.mean_abs_corr |    0.1881 |    0.04901 | 3.837   |     0.1391  |
| rankme                           |   74.79   | 1407       | 0.05317 | -1332       |
| effective_rank                   |   23.17   |  318.1     | 0.07282 |  -294.9     |
| alpha                            |    1.941  |    1.001   | 1.938   |     0.9394  |
| epps_pulley                      |  794.5    |   80.03    | 9.927   |   714.4     |
| kurt_topeig.worst                |    1.374  |    0.7784  | 1.765   |     0.5959  |



**teacher.h.cls.L03 vs teacher.z.dino.tap2**


| metric                           |   value_h |    value_z |     tau |      delta |
|:---------------------------------|----------:|-----------:|--------:|-----------:|
| uniformity                       |   -0.5477 |   -3.757   | 0.1458  |     3.21   |
| variance_floor.hinge             |    0.6074 |    0.7716  | 0.7872  |    -0.1642 |
| offdiag_redundancy.mean_abs_corr |    0.1881 |    0.02939 | 6.398   |     0.1587 |
| rankme                           |   74.79   | 1516       | 0.04932 | -1442      |
| effective_rank                   |   23.17   |  484       | 0.04786 |  -460.9    |
| alpha                            |    1.941  |    0.8207  | 2.365   |     1.12   |
| epps_pulley                      |  794.5    |   82.3     | 9.654   |   712.2    |
| kurt_topeig.worst                |    1.374  |   24.56    | 0.05595 |   -23.19   |



**teacher.h.cls.L06 vs teacher.z.dino.bottleneck**


| metric                           |   value_h |   value_z |     tau |     delta |
|:---------------------------------|----------:|----------:|--------:|----------:|
| uniformity                       |   -1.386  |  -3.608   | 0.3842  |   2.222   |
| variance_floor.hinge             |    0.3755 |   0.3994  | 0.94    |  -0.02395 |
| offdiag_redundancy.mean_abs_corr |    0.1256 |   0.08971 | 1.4     |   0.03585 |
| rankme                           |  152      | 138       | 1.101   |  13.94    |
| effective_rank                   |   49.58   |  78.64    | 0.6304  | -29.07    |
| alpha                            |    1.664  |   2.062   | 0.8071  |  -0.3978  |
| epps_pulley                      |  338.8    | 333.6     | 1.016   |   5.185   |
| kurt_topeig.worst                |    0.8053 |  13.8     | 0.05834 | -13       |



**teacher.h.cls.L06 vs teacher.z.dino.tap1**


| metric                           |   value_h |    value_z |    tau |       delta |
|:---------------------------------|----------:|-----------:|-------:|------------:|
| uniformity                       |   -1.386  |   -3.486   | 0.3977 |     2.099   |
| variance_floor.hinge             |    0.3755 |    0.5965  | 0.6294 |    -0.2211  |
| offdiag_redundancy.mean_abs_corr |    0.1256 |    0.04901 | 2.562  |     0.07655 |
| rankme                           |  152      | 1407       | 0.108  | -1255       |
| effective_rank                   |   49.58   |  318.1     | 0.1558 |  -268.5     |
| alpha                            |    1.664  |    1.001   | 1.662  |     0.6629  |
| epps_pulley                      |  338.8    |   80.03    | 4.233  |   258.7     |
| kurt_topeig.worst                |    0.8053 |    0.7784  | 1.034  |     0.02681 |



**teacher.h.cls.L06 vs teacher.z.dino.tap2**


| metric                           |   value_h |    value_z |     tau |       delta |
|:---------------------------------|----------:|-----------:|--------:|------------:|
| uniformity                       |   -1.386  |   -3.757   | 0.369   |     2.371   |
| variance_floor.hinge             |    0.3755 |    0.7716  | 0.4866  |    -0.3961  |
| offdiag_redundancy.mean_abs_corr |    0.1256 |    0.02939 | 4.272   |     0.09617 |
| rankme                           |  152      | 1516       | 0.1002  | -1364       |
| effective_rank                   |   49.58   |  484       | 0.1024  |  -434.4     |
| alpha                            |    1.664  |    0.8207  | 2.028   |     0.8436  |
| epps_pulley                      |  338.8    |   82.3     | 4.116   |   256.5     |
| kurt_topeig.worst                |    0.8053 |   24.56    | 0.03278 |   -23.76    |



**teacher.h.cls.L09 vs teacher.z.dino.bottleneck**


| metric                           |   value_h |   value_z |     tau |      delta |
|:---------------------------------|----------:|----------:|--------:|-----------:|
| uniformity                       |  -2.516   |  -3.608   | 0.6972  |   1.092    |
| variance_floor.hinge             |   0.1609  |   0.3994  | 0.4029  |  -0.2385   |
| offdiag_redundancy.mean_abs_corr |   0.09836 |   0.08971 | 1.096   |   0.008645 |
| rankme                           | 198.6     | 138       | 1.439   |  60.57     |
| effective_rank                   |  76.15    |  78.64    | 0.9683  |  -2.492    |
| alpha                            |   1.494   |   2.062   | 0.7244  |  -0.5683   |
| epps_pulley                      | 235.4     | 333.6     | 0.7058  | -98.15     |
| kurt_topeig.worst                |   0.8977  |  13.8     | 0.06504 | -12.9      |



**teacher.h.cls.L09 vs teacher.z.dino.tap1**


| metric                           |   value_h |    value_z |    tau |       delta |
|:---------------------------------|----------:|-----------:|-------:|------------:|
| uniformity                       |  -2.516   |   -3.486   | 0.7217 |     0.97    |
| variance_floor.hinge             |   0.1609  |    0.5965  | 0.2698 |    -0.4356  |
| offdiag_redundancy.mean_abs_corr |   0.09836 |    0.04901 | 2.007  |     0.04934 |
| rankme                           | 198.6     | 1407       | 0.1412 | -1208       |
| effective_rank                   |  76.15    |  318.1     | 0.2394 |  -242       |
| alpha                            |   1.494   |    1.001   | 1.492  |     0.4924  |
| epps_pulley                      | 235.4     |   80.03    | 2.942  |   155.4     |
| kurt_topeig.worst                |   0.8977  |    0.7784  | 1.153  |     0.1193  |



**teacher.h.cls.L09 vs teacher.z.dino.tap2**


| metric                           |   value_h |    value_z |     tau |       delta |
|:---------------------------------|----------:|-----------:|--------:|------------:|
| uniformity                       |  -2.516   |   -3.757   | 0.6695  |     1.242   |
| variance_floor.hinge             |   0.1609  |    0.7716  | 0.2086  |    -0.6106  |
| offdiag_redundancy.mean_abs_corr |   0.09836 |    0.02939 | 3.346   |     0.06896 |
| rankme                           | 198.6     | 1516       | 0.131   | -1318       |
| effective_rank                   |  76.15    |  484       | 0.1573  |  -407.9     |
| alpha                            |   1.494   |    0.8207  | 1.82    |     0.6731  |
| epps_pulley                      | 235.4     |   82.3     | 2.861   |   153.1     |
| kurt_topeig.worst                |   0.8977  |   24.56    | 0.03655 |   -23.67    |



**teacher.h.cls.L12 vs teacher.z.dino.bottleneck**


| metric                           |   value_h |   value_z |     tau |       delta |
|:---------------------------------|----------:|----------:|--------:|------------:|
| uniformity                       |  -3.127   |  -3.608   | 0.8666  |    0.4812   |
| variance_floor.hinge             |   0.07743 |   0.3994  | 0.1939  |   -0.322    |
| offdiag_redundancy.mean_abs_corr |   0.09663 |   0.08971 | 1.077   |    0.006915 |
| rankme                           | 200.1     | 138       | 1.45    |   62.08     |
| effective_rank                   |  80.46    |  78.64    | 1.023   |    1.811    |
| alpha                            |   1.563   |   2.062   | 0.7581  |   -0.4989   |
| epps_pulley                      | 218       | 333.6     | 0.6535  | -115.6      |
| kurt_topeig.worst                |   0.8351  |  13.8     | 0.06051 |  -12.97     |



**teacher.h.cls.L12 vs teacher.z.dino.tap1**


| metric                           |   value_h |    value_z |    tau |       delta |
|:---------------------------------|----------:|-----------:|-------:|------------:|
| uniformity                       |  -3.127   |   -3.486   | 0.8971 |     0.3588  |
| variance_floor.hinge             |   0.07743 |    0.5965  | 0.1298 |    -0.5191  |
| offdiag_redundancy.mean_abs_corr |   0.09663 |    0.04901 | 1.972  |     0.04761 |
| rankme                           | 200.1     | 1407       | 0.1423 | -1206       |
| effective_rank                   |  80.46    |  318.1     | 0.2529 |  -237.7     |
| alpha                            |   1.563   |    1.001   | 1.561  |     0.5618  |
| epps_pulley                      | 218       |   80.03    | 2.724  |   138       |
| kurt_topeig.worst                |   0.8351  |    0.7784  | 1.073  |     0.05668 |



**teacher.h.cls.L12 vs teacher.z.dino.tap2**


| metric                           |   value_h |    value_z |    tau |       delta |
|:---------------------------------|----------:|-----------:|-------:|------------:|
| uniformity                       |  -3.127   |   -3.757   | 0.8322 |     0.6305  |
| variance_floor.hinge             |   0.07743 |    0.7716  | 0.1004 |    -0.6941  |
| offdiag_redundancy.mean_abs_corr |   0.09663 |    0.02939 | 3.287  |     0.06723 |
| rankme                           | 200.1     | 1516       | 0.132  | -1316       |
| effective_rank                   |  80.46    |  484       | 0.1662 |  -403.6     |
| alpha                            |   1.563   |    0.8207  | 1.905  |     0.7425  |
| epps_pulley                      | 218       |   82.3     | 2.649  |   135.7     |
| kurt_topeig.worst                |   0.8351  |   24.56    | 0.034  |   -23.73    |



**teacher.h.gap vs teacher.z.dino.bottleneck**


| metric                           |   value_h |   value_z |    tau |     delta |
|:---------------------------------|----------:|----------:|-------:|----------:|
| uniformity                       |   -2.58   |  -3.608   | 0.7152 |   1.028   |
| variance_floor.hinge             |    0.5733 |   0.3994  | 1.435  |   0.1739  |
| offdiag_redundancy.mean_abs_corr |    0.1189 |   0.08971 | 1.326  |   0.02921 |
| rankme                           |  192.7    | 138       | 1.397  |  54.73    |
| effective_rank                   |   55.69   |  78.64    | 0.7082 | -22.95    |
| alpha                            |    1.597  |   2.062   | 0.7744 |  -0.4653  |
| epps_pulley                      |  304.9    | 333.6     | 0.914  | -28.69    |
| kurt_topeig.worst                |    2.621  |  13.8     | 0.1899 | -11.18    |



**teacher.h.gap vs teacher.z.dino.tap1**


| metric                           |   value_h |    value_z |    tau |       delta |
|:---------------------------------|----------:|-----------:|-------:|------------:|
| uniformity                       |   -2.58   |   -3.486   | 0.7403 |     0.9054  |
| variance_floor.hinge             |    0.5733 |    0.5965  | 0.9611 |    -0.0232  |
| offdiag_redundancy.mean_abs_corr |    0.1189 |    0.04901 | 2.426  |     0.06991 |
| rankme                           |  192.7    | 1407       | 0.137  | -1214       |
| effective_rank                   |   55.69   |  318.1     | 0.1751 |  -262.4     |
| alpha                            |    1.597  |    1.001   | 1.595  |     0.5955  |
| epps_pulley                      |  304.9    |   80.03    | 3.81   |   224.9     |
| kurt_topeig.worst                |    2.621  |    0.7784  | 3.366  |     1.842   |



**teacher.h.gap vs teacher.z.dino.tap2**


| metric                           |   value_h |    value_z |    tau |       delta |
|:---------------------------------|----------:|-----------:|-------:|------------:|
| uniformity                       |   -2.58   |   -3.757   | 0.6867 |     1.177   |
| variance_floor.hinge             |    0.5733 |    0.7716  | 0.7431 |    -0.1982  |
| offdiag_redundancy.mean_abs_corr |    0.1189 |    0.02939 | 4.046  |     0.08953 |
| rankme                           |  192.7    | 1516       | 0.1271 | -1324       |
| effective_rank                   |   55.69   |  484       | 0.1151 |  -428.3     |
| alpha                            |    1.597  |    0.8207  | 1.946  |     0.7761  |
| epps_pulley                      |  304.9    |   82.3     | 3.705  |   222.6     |
| kurt_topeig.worst                |    2.621  |   24.56    | 0.1067 |   -21.94    |



**teacher.h.gap.L03 vs teacher.z.dino.bottleneck**


| metric                           |   value_h |   value_z |    tau |     delta |
|:---------------------------------|----------:|----------:|-------:|----------:|
| uniformity                       |   -1.336  |  -3.608   | 0.3702 |   2.273   |
| variance_floor.hinge             |    0.7449 |   0.3994  | 1.865  |   0.3455  |
| offdiag_redundancy.mean_abs_corr |    0.1598 |   0.08971 | 1.781  |   0.07008 |
| rankme                           |  124      | 138       | 0.8986 | -13.99    |
| effective_rank                   |   22.76   |  78.64    | 0.2893 | -55.89    |
| alpha                            |    1.771  |   2.062   | 0.8587 |  -0.2913  |
| epps_pulley                      |  596.3    | 333.6     | 1.788  | 262.7     |
| kurt_topeig.worst                |    1.697  |  13.8     | 0.123  | -12.1     |



**teacher.h.gap.L03 vs teacher.z.dino.tap1**


| metric                           |   value_h |    value_z |     tau |      delta |
|:---------------------------------|----------:|-----------:|--------:|-----------:|
| uniformity                       |   -1.336  |   -3.486   | 0.3832  |     2.15   |
| variance_floor.hinge             |    0.7449 |    0.5965  | 1.249   |     0.1484 |
| offdiag_redundancy.mean_abs_corr |    0.1598 |    0.04901 | 3.26    |     0.1108 |
| rankme                           |  124      | 1407       | 0.08817 | -1283      |
| effective_rank                   |   22.76   |  318.1     | 0.07153 |  -295.4    |
| alpha                            |    1.771  |    1.001   | 1.768   |     0.7695 |
| epps_pulley                      |  596.3    |   80.03    | 7.451   |   516.3    |
| kurt_topeig.worst                |    1.697  |    0.7784  | 2.18    |     0.9188 |



**teacher.h.gap.L03 vs teacher.z.dino.tap2**


| metric                           |   value_h |    value_z |     tau |       delta |
|:---------------------------------|----------:|-----------:|--------:|------------:|
| uniformity                       |   -1.336  |   -3.757   | 0.3555  |     2.422   |
| variance_floor.hinge             |    0.7449 |    0.7716  | 0.9655  |    -0.02663 |
| offdiag_redundancy.mean_abs_corr |    0.1598 |    0.02939 | 5.436   |     0.1304  |
| rankme                           |  124      | 1516       | 0.08179 | -1392       |
| effective_rank                   |   22.76   |  484       | 0.04701 |  -461.3     |
| alpha                            |    1.771  |    0.8207  | 2.158   |     0.9501  |
| epps_pulley                      |  596.3    |   82.3     | 7.246   |   514       |
| kurt_topeig.worst                |    1.697  |   24.56    | 0.0691  |   -22.87    |



**teacher.h.gap.L06 vs teacher.z.dino.bottleneck**


| metric                           |   value_h |   value_z |    tau |    delta |
|:---------------------------------|----------:|----------:|-------:|---------:|
| uniformity                       |   -1.783  |  -3.608   | 0.4941 |   1.825  |
| variance_floor.hinge             |    0.6719 |   0.3994  | 1.682  |   0.2725 |
| offdiag_redundancy.mean_abs_corr |    0.1323 |   0.08971 | 1.475  |   0.0426 |
| rankme                           |  163      | 138       | 1.181  |  24.94   |
| effective_rank                   |   41.14   |  78.64    | 0.5231 | -37.5    |
| alpha                            |    1.659  |   2.062   | 0.8045 |  -0.4031 |
| epps_pulley                      |  424.7    | 333.6     | 1.273  |  91.13   |
| kurt_topeig.worst                |    1.906  |  13.8     | 0.1381 | -11.9    |



**teacher.h.gap.L06 vs teacher.z.dino.tap1**


| metric                           |   value_h |    value_z |    tau |       delta |
|:---------------------------------|----------:|-----------:|-------:|------------:|
| uniformity                       |   -1.783  |   -3.486   | 0.5115 |     1.703   |
| variance_floor.hinge             |    0.6719 |    0.5965  | 1.126  |     0.07535 |
| offdiag_redundancy.mean_abs_corr |    0.1323 |    0.04901 | 2.7    |     0.0833  |
| rankme                           |  163      | 1407       | 0.1159 | -1244       |
| effective_rank                   |   41.14   |  318.1     | 0.1293 |  -277       |
| alpha                            |    1.659  |    1.001   | 1.657  |     0.6576  |
| epps_pulley                      |  424.7    |   80.03    | 5.307  |   344.7     |
| kurt_topeig.worst                |    1.906  |    0.7784  | 2.449  |     1.128   |



**teacher.h.gap.L06 vs teacher.z.dino.tap2**


| metric                           |   value_h |    value_z |     tau |       delta |
|:---------------------------------|----------:|-----------:|--------:|------------:|
| uniformity                       |   -1.783  |   -3.757   | 0.4745  |     1.975   |
| variance_floor.hinge             |    0.6719 |    0.7716  | 0.8708  |    -0.09968 |
| offdiag_redundancy.mean_abs_corr |    0.1323 |    0.02939 | 4.501   |     0.1029  |
| rankme                           |  163      | 1516       | 0.1075  | -1353       |
| effective_rank                   |   41.14   |  484       | 0.085   |  -442.9     |
| alpha                            |    1.659  |    0.8207  | 2.021   |     0.8382  |
| epps_pulley                      |  424.7    |   82.3     | 5.161   |   342.4     |
| kurt_topeig.worst                |    1.906  |   24.56    | 0.07761 |   -22.66    |



**teacher.h.gap.L09 vs teacher.z.dino.bottleneck**


| metric                           |   value_h |   value_z |    tau |     delta |
|:---------------------------------|----------:|----------:|-------:|----------:|
| uniformity                       |   -2.347  |  -3.608   | 0.6505 |   1.261   |
| variance_floor.hinge             |    0.5999 |   0.3994  | 1.502  |   0.2004  |
| offdiag_redundancy.mean_abs_corr |    0.1195 |   0.08971 | 1.332  |   0.02979 |
| rankme                           |  195.7    | 138       | 1.418  |  57.66    |
| effective_rank                   |   57.95   |  78.64    | 0.7369 | -20.69    |
| alpha                            |    1.496  |   2.062   | 0.7254 |  -0.5664  |
| epps_pulley                      |  336.5    | 333.6     | 1.009  |   2.945   |
| kurt_topeig.worst                |    2.326  |  13.8     | 0.1685 | -11.48    |



**teacher.h.gap.L09 vs teacher.z.dino.tap1**


| metric                           |   value_h |    value_z |    tau |        delta |
|:---------------------------------|----------:|-----------:|-------:|-------------:|
| uniformity                       |   -2.347  |   -3.486   | 0.6733 |     1.139    |
| variance_floor.hinge             |    0.5999 |    0.5965  | 1.006  |     0.003328 |
| offdiag_redundancy.mean_abs_corr |    0.1195 |    0.04901 | 2.438  |     0.07049  |
| rankme                           |  195.7    | 1407       | 0.1391 | -1211        |
| effective_rank                   |   57.95   |  318.1     | 0.1822 |  -260.2      |
| alpha                            |    1.496  |    1.001   | 1.494  |     0.4944   |
| epps_pulley                      |  336.5    |   80.03    | 4.205  |   256.5      |
| kurt_topeig.worst                |    2.326  |    0.7784  | 2.988  |     1.548    |



**teacher.h.gap.L09 vs teacher.z.dino.tap2**


| metric                           |   value_h |    value_z |    tau |       delta |
|:---------------------------------|----------:|-----------:|-------:|------------:|
| uniformity                       |   -2.347  |   -3.757   | 0.6246 |     1.41    |
| variance_floor.hinge             |    0.5999 |    0.7716  | 0.7775 |    -0.1717  |
| offdiag_redundancy.mean_abs_corr |    0.1195 |    0.02939 | 4.065  |     0.09011 |
| rankme                           |  195.7    | 1516       | 0.129  | -1321       |
| effective_rank                   |   57.95   |  484       | 0.1197 |  -426.1     |
| alpha                            |    1.496  |    0.8207  | 1.822  |     0.675   |
| epps_pulley                      |  336.5    |   82.3     | 4.089  |   254.2     |
| kurt_topeig.worst                |    2.326  |   24.56    | 0.0947 |   -22.24    |



**teacher.h.gap.L12 vs teacher.z.dino.bottleneck**


| metric                           |   value_h |   value_z |    tau |     delta |
|:---------------------------------|----------:|----------:|-------:|----------:|
| uniformity                       |   -2.58   |  -3.608   | 0.7152 |   1.028   |
| variance_floor.hinge             |    0.5733 |   0.3994  | 1.435  |   0.1739  |
| offdiag_redundancy.mean_abs_corr |    0.1189 |   0.08971 | 1.326  |   0.02921 |
| rankme                           |  192.7    | 138       | 1.397  |  54.73    |
| effective_rank                   |   55.69   |  78.64    | 0.7082 | -22.95    |
| alpha                            |    1.597  |   2.062   | 0.7744 |  -0.4653  |
| epps_pulley                      |  304.9    | 333.6     | 0.914  | -28.69    |
| kurt_topeig.worst                |    2.621  |  13.8     | 0.1899 | -11.18    |



**teacher.h.gap.L12 vs teacher.z.dino.tap1**


| metric                           |   value_h |    value_z |    tau |       delta |
|:---------------------------------|----------:|-----------:|-------:|------------:|
| uniformity                       |   -2.58   |   -3.486   | 0.7403 |     0.9054  |
| variance_floor.hinge             |    0.5733 |    0.5965  | 0.9611 |    -0.0232  |
| offdiag_redundancy.mean_abs_corr |    0.1189 |    0.04901 | 2.426  |     0.06991 |
| rankme                           |  192.7    | 1407       | 0.137  | -1214       |
| effective_rank                   |   55.69   |  318.1     | 0.1751 |  -262.4     |
| alpha                            |    1.597  |    1.001   | 1.595  |     0.5955  |
| epps_pulley                      |  304.9    |   80.03    | 3.81   |   224.9     |
| kurt_topeig.worst                |    2.621  |    0.7784  | 3.366  |     1.842   |



**teacher.h.gap.L12 vs teacher.z.dino.tap2**


| metric                           |   value_h |    value_z |    tau |       delta |
|:---------------------------------|----------:|-----------:|-------:|------------:|
| uniformity                       |   -2.58   |   -3.757   | 0.6867 |     1.177   |
| variance_floor.hinge             |    0.5733 |    0.7716  | 0.7431 |    -0.1982  |
| offdiag_redundancy.mean_abs_corr |    0.1189 |    0.02939 | 4.046  |     0.08953 |
| rankme                           |  192.7    | 1516       | 0.1271 | -1324       |
| effective_rank                   |   55.69   |  484       | 0.1151 |  -428.3     |
| alpha                            |    1.597  |    0.8207  | 1.946  |     0.7761  |
| epps_pulley                      |  304.9    |   82.3     | 3.705  |   222.6     |
| kurt_topeig.worst                |    2.621  |   24.56    | 0.1067 |   -21.94    |



### Pair metrics (alignment ↓ = more view-invariant)


| manifest                   | space                     | metric         |   value |   ci_lo |   ci_hi |
|:---------------------------|:--------------------------|:---------------|--------:|--------:|--------:|
| in100.pairs100.v1@audit_v1 | student.h.cls             | alignment      |  0.6537 |  0.6474 |  0.6603 |
| in100.pairs100.v1@audit_v1 | student.h.cls             | cos_invariance |  0.6731 |  0.6701 |  0.6759 |
| in100.pairs100.v1@audit_v1 | student.h.gap             | alignment      |  0.6053 |  0.5998 |  0.6105 |
| in100.pairs100.v1@audit_v1 | student.h.gap             | cos_invariance |  0.6974 |  0.6942 |  0.7    |
| in100.pairs100.v1@audit_v1 | student.z.dino.bottleneck | alignment      |  0.4977 |  0.4903 |  0.5061 |
| in100.pairs100.v1@audit_v1 | student.z.dino.bottleneck | cos_invariance |  0.7512 |  0.7473 |  0.7548 |
| in100.pairs100.v1@audit_v1 | student.z.dino.tap1       | alignment      |  0.7401 |  0.7329 |  0.7485 |
| in100.pairs100.v1@audit_v1 | student.z.dino.tap1       | cos_invariance |  0.63   |  0.626  |  0.6332 |
| in100.pairs100.v1@audit_v1 | student.z.dino.tap2       | alignment      |  0.7472 |  0.7397 |  0.7556 |
| in100.pairs100.v1@audit_v1 | student.z.dino.tap2       | cos_invariance |  0.6264 |  0.6222 |  0.6303 |
| in100.pairs100.v1@audit_v1 | teacher.h.cls             | alignment      |  0.6413 |  0.6352 |  0.6486 |
| in100.pairs100.v1@audit_v1 | teacher.h.cls             | cos_invariance |  0.6793 |  0.6761 |  0.6823 |
| in100.pairs100.v1@audit_v1 | teacher.h.gap             | alignment      |  0.5841 |  0.5788 |  0.5893 |
| in100.pairs100.v1@audit_v1 | teacher.h.gap             | cos_invariance |  0.708  |  0.7048 |  0.7105 |
| in100.pairs100.v1@audit_v1 | teacher.z.dino.bottleneck | alignment      |  0.4681 |  0.4611 |  0.4764 |
| in100.pairs100.v1@audit_v1 | teacher.z.dino.bottleneck | cos_invariance |  0.7659 |  0.7622 |  0.7697 |
| in100.pairs100.v1@audit_v1 | teacher.z.dino.tap1       | alignment      |  0.7117 |  0.7044 |  0.7198 |
| in100.pairs100.v1@audit_v1 | teacher.z.dino.tap1       | cos_invariance |  0.6441 |  0.6403 |  0.6475 |
| in100.pairs100.v1@audit_v1 | teacher.z.dino.tap2       | alignment      |  0.7199 |  0.7126 |  0.7285 |
| in100.pairs100.v1@audit_v1 | teacher.z.dino.tap2       | cos_invariance |  0.64   |  0.6357 |  0.6439 |
| in100.pairs100.v1@own_dino | student.h.cls             | alignment      |  0.3026 |  0.2988 |  0.306  |
| in100.pairs100.v1@own_dino | student.h.cls             | cos_invariance |  0.8487 |  0.847  |  0.8506 |
| in100.pairs100.v1@own_dino | student.h.gap             | alignment      |  0.3163 |  0.3128 |  0.3198 |
| in100.pairs100.v1@own_dino | student.h.gap             | cos_invariance |  0.8419 |  0.84   |  0.8438 |
| in100.pairs100.v1@own_dino | student.z.dino.bottleneck | alignment      |  0.169  |  0.1651 |  0.1723 |
| in100.pairs100.v1@own_dino | student.z.dino.bottleneck | cos_invariance |  0.9155 |  0.9138 |  0.9171 |
| in100.pairs100.v1@own_dino | student.z.dino.tap1       | alignment      |  0.3237 |  0.3191 |  0.3277 |
| in100.pairs100.v1@own_dino | student.z.dino.tap1       | cos_invariance |  0.8381 |  0.8362 |  0.8401 |
| in100.pairs100.v1@own_dino | student.z.dino.tap2       | alignment      |  0.3079 |  0.3028 |  0.3125 |
| in100.pairs100.v1@own_dino | student.z.dino.tap2       | cos_invariance |  0.8461 |  0.8439 |  0.8481 |
| in100.pairs100.v1@own_dino | teacher.h.cls             | alignment      |  0.2919 |  0.2878 |  0.2953 |
| in100.pairs100.v1@own_dino | teacher.h.cls             | cos_invariance |  0.854  |  0.8524 |  0.8559 |
| in100.pairs100.v1@own_dino | teacher.h.gap             | alignment      |  0.3053 |  0.3019 |  0.3087 |
| in100.pairs100.v1@own_dino | teacher.h.gap             | cos_invariance |  0.8473 |  0.8456 |  0.8492 |
| in100.pairs100.v1@own_dino | teacher.z.dino.bottleneck | alignment      |  0.1543 |  0.1506 |  0.1576 |
| in100.pairs100.v1@own_dino | teacher.z.dino.bottleneck | cos_invariance |  0.9228 |  0.9213 |  0.9243 |
| in100.pairs100.v1@own_dino | teacher.z.dino.tap1       | alignment      |  0.3052 |  0.3007 |  0.3089 |
| in100.pairs100.v1@own_dino | teacher.z.dino.tap1       | cos_invariance |  0.8474 |  0.8457 |  0.8492 |
| in100.pairs100.v1@own_dino | teacher.z.dino.tap2       | alignment      |  0.2895 |  0.2851 |  0.2941 |
| in100.pairs100.v1@own_dino | teacher.z.dino.tap2       | cos_invariance |  0.8553 |  0.8531 |  0.8571 |



### Cross-space similarity (CKA is contested — read as the triple)


| space                                   | metric              |   value |
|:----------------------------------------|:--------------------|--------:|
| teacher.h.cls~teacher.z.dino.bottleneck | cka_linear          |  0.5767 |
| teacher.h.cls~teacher.z.dino.bottleneck | neighbor_jaccard    |  0.2555 |
| teacher.h.cls~teacher.z.dino.bottleneck | procrustes_distance |  0.8044 |
| teacher.h.cls~teacher.z.dino.bottleneck | knn_label_agreement |  0.563  |
| teacher.h.cls~teacher.z.dino.tap1       | cka_linear          |  0.9136 |
| teacher.h.cls~teacher.z.dino.tap1       | neighbor_jaccard    |  0.5294 |
| teacher.h.cls~teacher.z.dino.tap1       | procrustes_distance |  0.4395 |
| teacher.h.cls~teacher.z.dino.tap1       | knn_label_agreement |  0.689  |
| teacher.h.cls~teacher.z.dino.tap2       | cka_linear          |  0.6292 |
| teacher.h.cls~teacher.z.dino.tap2       | neighbor_jaccard    |  0.3176 |
| teacher.h.cls~teacher.z.dino.tap2       | procrustes_distance |  0.7971 |
| teacher.h.cls~teacher.z.dino.tap2       | knn_label_agreement |  0.5916 |



### Probes


| space                     |   knn_v1_k20 |   knn_v1_k200 |   linear_house_v1 |   linear_l2_v1 |   linear_raw_v1 |
|:--------------------------|-------------:|--------------:|------------------:|---------------:|----------------:|
| student.h.cls             |       0.6036 |        0.5760 |            0.6596 |         0.6292 |          0.6608 |
| student.h.cls.L03         |       0.2130 |        0.1966 |            0.2906 |         0.1994 |          0.2836 |
| student.h.cls.L06         |       0.3522 |        0.3296 |            0.4606 |         0.3660 |          0.4564 |
| student.h.cls.L09         |       0.5274 |        0.4958 |            0.6044 |         0.5558 |          0.6056 |
| student.h.cls.L12         |       0.6036 |        0.5760 |            0.6596 |         0.6292 |          0.6598 |
| student.h.gap             |       0.4848 |        0.4488 |            0.6088 |         0.5302 |          0.6024 |
| student.h.gap.L03         |       0.2434 |        0.2288 |            0.3658 |         0.2628 |          0.3426 |
| student.h.gap.L06         |       0.3646 |        0.3366 |            0.4928 |         0.3860 |          0.4704 |
| student.h.gap.L09         |       0.4542 |        0.4204 |            0.5828 |         0.4960 |          0.5702 |
| student.h.gap.L12         |       0.4848 |        0.4488 |            0.6088 |         0.5302 |          0.6024 |
| student.z.dino.bottleneck |       0.5542 |        0.5436 |            0.5846 |         0.5680 |          0.5804 |
| student.z.dino.tap1       |       0.5990 |        0.5746 |            0.6528 |         0.6460 |          0.6634 |
| student.z.dino.tap2       |       0.5792 |        0.5732 |            0.6206 |         0.6126 |          0.6294 |
| teacher.h.cls             |       0.6138 |        0.5880 |            0.6682 |         0.6368 |          0.6724 |
| teacher.h.cls.L03         |       0.2118 |        0.1996 |            0.2940 |         0.2018 |          0.2850 |
| teacher.h.cls.L06         |       0.3616 |        0.3422 |            0.4712 |         0.3740 |          0.4656 |
| teacher.h.cls.L09         |       0.5368 |        0.5034 |            0.6172 |         0.5620 |          0.6156 |
| teacher.h.cls.L12         |       0.6138 |        0.5880 |            0.6682 |         0.6368 |          0.6724 |
| teacher.h.gap             |       0.4830 |        0.4460 |            0.6180 |         0.5376 |          0.6146 |
| teacher.h.gap.L03         |       0.2442 |        0.2268 |            0.3652 |         0.2668 |          0.3426 |
| teacher.h.gap.L06         |       0.3702 |        0.3464 |            0.4966 |         0.3890 |          0.4778 |
| teacher.h.gap.L09         |       0.4622 |        0.4228 |            0.5908 |         0.5006 |          0.5794 |
| teacher.h.gap.L12         |       0.4830 |        0.4460 |            0.6180 |         0.5376 |          0.6146 |
| teacher.z.dino.bottleneck |       0.5560 |        0.5462 |            0.5900 |         0.5734 |          0.5874 |
| teacher.z.dino.tap1       |       0.6062 |        0.5782 |            0.6612 |         0.6558 |          0.6748 |
| teacher.z.dino.tap2       |       0.5866 |        0.5738 |            0.6324 |         0.6190 |          0.6376 |



## in100.dino-ctrl.ep100.ext

### Battery (variant raw|full)

| model.space                                           |   rankme |   effective_rank |   alpha |   epps_pulley |   kurt_topeig.worst |   uniformity |   offdiag_redundancy.mean_abs_corr |   variance_floor.hinge |
|:------------------------------------------------------|---------:|-----------------:|--------:|--------------:|--------------------:|-------------:|-----------------------------------:|-----------------------:|
| in100.dino-ctrl.ep100.ext · student.h.cls             |   243.3  |           117.4  |  1.231  |        142.9  |              0.8165 |      -3.447  |                            0.08088 |               0.007166 |
| in100.dino-ctrl.ep100.ext · student.h.cls.L03         |    69.23 |            25.21 |  2.049  |        763.9  |              2.337  |      -0.5072 |                            0.1869  |               0.5464   |
| in100.dino-ctrl.ep100.ext · student.h.cls.L06         |   179.8  |            63.96 |  1.545  |        278.6  |              0.8579 |      -1.84   |                            0.1179  |               0.1329   |
| in100.dino-ctrl.ep100.ext · student.h.cls.L09         |   243.7  |           115.6  |  1.196  |        150.1  |              0.9573 |      -2.985  |                            0.0828  |               0.008414 |
| in100.dino-ctrl.ep100.ext · student.h.cls.L12         |   243.3  |           117.4  |  1.231  |        142.9  |              0.8165 |      -3.447  |                            0.08088 |               0.007166 |
| in100.dino-ctrl.ep100.ext · student.h.gap             |   221.6  |            84.43 |  1.379  |        227.5  |              1.703  |      -2.53   |                            0.1018  |               0.545    |
| in100.dino-ctrl.ep100.ext · student.h.gap.L03         |   130.5  |            30.74 |  1.72   |        578.5  |              1.114  |      -1.134  |                            0.1487  |               0.6973   |
| in100.dino-ctrl.ep100.ext · student.h.gap.L06         |   179.5  |            55.4  |  1.547  |        387    |              2.146  |      -1.724  |                            0.1231  |               0.6105   |
| in100.dino-ctrl.ep100.ext · student.h.gap.L09         |   220.9  |            84.49 |  1.326  |        262.7  |              2.218  |      -2.207  |                            0.1042  |               0.552    |
| in100.dino-ctrl.ep100.ext · student.h.gap.L12         |   221.6  |            84.43 |  1.379  |        227.5  |              1.703  |      -2.53   |                            0.1018  |               0.545    |
| in100.dino-ctrl.ep100.ext · student.z.dino.bottleneck |   160.2  |            94.16 |  1.661  |        453    |             46.94   |      -3.665  |                            0.08262 |               0.889    |
| in100.dino-ctrl.ep100.ext · student.z.dino.tap1       |  1434    |           374.1  |  0.9755 |         78.09 |              0.8352 |      -3.371  |                            0.05546 |               0.5857   |
| in100.dino-ctrl.ep100.ext · student.z.dino.tap2       |  1513    |           650.7  |  0.7619 |         52.82 |             37.39   |      -1.677  |                            0.02812 |               0.9068   |
| in100.dino-ctrl.ep100.ext · teacher.h.cls             |   239.2  |           113.9  |  1.261  |        149.9  |              0.823  |      -3.446  |                            0.08209 |               0.008027 |
| in100.dino-ctrl.ep100.ext · teacher.h.cls.L03         |    70.02 |            25.1  |  2.035  |        772.2  |              2.526  |      -0.5134 |                            0.1869  |               0.5468   |
| in100.dino-ctrl.ep100.ext · teacher.h.cls.L06         |   177.2  |            63    |  1.557  |        280.1  |              0.8594 |      -1.795  |                            0.1182  |               0.148    |
| in100.dino-ctrl.ep100.ext · teacher.h.cls.L09         |   239.7  |           112.3  |  1.222  |        157.2  |              0.9633 |      -2.958  |                            0.08394 |               0.009887 |
| in100.dino-ctrl.ep100.ext · teacher.h.cls.L12         |   239.2  |           113.9  |  1.261  |        149.9  |              0.823  |      -3.446  |                            0.08209 |               0.008027 |
| in100.dino-ctrl.ep100.ext · teacher.h.gap             |   220    |            81.44 |  1.4    |        234.4  |              1.765  |      -2.581  |                            0.104   |               0.5455   |
| in100.dino-ctrl.ep100.ext · teacher.h.gap.L03         |   131.3  |            29.93 |  1.725  |        595.6  |              1.282  |      -1.185  |                            0.1494  |               0.6995   |
| in100.dino-ctrl.ep100.ext · teacher.h.gap.L06         |   178    |            54.05 |  1.56   |        393.6  |              2.269  |      -1.74   |                            0.124   |               0.6127   |
| in100.dino-ctrl.ep100.ext · teacher.h.gap.L09         |   219.2  |            81.69 |  1.341  |        274.5  |              2.804  |      -2.249  |                            0.1064  |               0.553    |
| in100.dino-ctrl.ep100.ext · teacher.h.gap.L12         |   220    |            81.44 |  1.4    |        234.4  |              1.765  |      -2.581  |                            0.104   |               0.5455   |
| in100.dino-ctrl.ep100.ext · teacher.z.dino.bottleneck |   159.9  |            94    |  1.666  |        419.9  |             45.09   |      -3.662  |                            0.08261 |               0.8729   |
| in100.dino-ctrl.ep100.ext · teacher.z.dino.tap1       |  1431    |           367.8  |  0.98   |         80.12 |              0.8372 |      -3.419  |                            0.05596 |               0.5769   |
| in100.dino-ctrl.ep100.ext · teacher.z.dino.tap2       |  1539    |           662.1  |  0.7525 |         50.26 |             37.03   |      -1.953  |                            0.02763 |               0.8978   |



### Transfer ratios τ = value(h) / value(z)


**student.h.cls vs student.z.dino.bottleneck**


| metric                           |    value_h |   value_z |     tau |       delta |
|:---------------------------------|-----------:|----------:|--------:|------------:|
| uniformity                       |  -3.447    |  -3.665   | 0.9405  |    0.2181   |
| variance_floor.hinge             |   0.007166 |   0.889   | 0.00806 |   -0.8819   |
| offdiag_redundancy.mean_abs_corr |   0.08088  |   0.08262 | 0.979   |   -0.001732 |
| rankme                           | 243.3      | 160.2     | 1.519   |   83.09     |
| effective_rank                   | 117.4      |  94.16    | 1.246   |   23.19     |
| alpha                            |   1.231    |   1.661   | 0.7412  |   -0.4297   |
| epps_pulley                      | 142.9      | 453       | 0.3155  | -310        |
| kurt_topeig.worst                |   0.8165   |  46.94    | 0.0174  |  -46.12     |



**student.h.cls vs student.z.dino.tap1**


| metric                           |    value_h |    value_z |     tau |       delta |
|:---------------------------------|-----------:|-----------:|--------:|------------:|
| uniformity                       |  -3.447    |   -3.371   | 1.023   |    -0.07596 |
| variance_floor.hinge             |   0.007166 |    0.5857  | 0.01223 |    -0.5785  |
| offdiag_redundancy.mean_abs_corr |   0.08088  |    0.05546 | 1.458   |     0.02543 |
| rankme                           | 243.3      | 1434       | 0.1697  | -1190       |
| effective_rank                   | 117.4      |  374.1     | 0.3137  |  -256.8     |
| alpha                            |   1.231    |    0.9755  | 1.262   |     0.2553  |
| epps_pulley                      | 142.9      |   78.09    | 1.83    |    64.84    |
| kurt_topeig.worst                |   0.8165   |    0.8352  | 0.9775  |    -0.01878 |



**student.h.cls vs student.z.dino.tap2**


| metric                           |    value_h |    value_z |      tau |       delta |
|:---------------------------------|-----------:|-----------:|---------:|------------:|
| uniformity                       |  -3.447    |   -1.677   | 2.055    |    -1.769   |
| variance_floor.hinge             |   0.007166 |    0.9068  | 0.007902 |    -0.8996  |
| offdiag_redundancy.mean_abs_corr |   0.08088  |    0.02812 | 2.876    |     0.05276 |
| rankme                           | 243.3      | 1513       | 0.1608   | -1270       |
| effective_rank                   | 117.4      |  650.7     | 0.1803   |  -533.4     |
| alpha                            |   1.231    |    0.7619  | 1.615    |     0.4689  |
| epps_pulley                      | 142.9      |   52.82    | 2.706    |    90.11    |
| kurt_topeig.worst                |   0.8165   |   37.39    | 0.02183  |   -36.58    |



**student.h.cls.L03 vs student.z.dino.bottleneck**


| metric                           |   value_h |   value_z |     tau |    delta |
|:---------------------------------|----------:|----------:|--------:|---------:|
| uniformity                       |   -0.5072 |  -3.665   | 0.1384  |   3.158  |
| variance_floor.hinge             |    0.5464 |   0.889   | 0.6146  |  -0.3426 |
| offdiag_redundancy.mean_abs_corr |    0.1869 |   0.08262 | 2.263   |   0.1043 |
| rankme                           |   69.23   | 160.2     | 0.4321  | -91      |
| effective_rank                   |   25.21   |  94.16    | 0.2677  | -68.95   |
| alpha                            |    2.049  |   1.661   | 1.234   |   0.3885 |
| epps_pulley                      |  763.9    | 453       | 1.686   | 310.9    |
| kurt_topeig.worst                |    2.337  |  46.94    | 0.04979 | -44.6    |



**student.h.cls.L03 vs student.z.dino.tap1**


| metric                           |   value_h |    value_z |     tau |       delta |
|:---------------------------------|----------:|-----------:|--------:|------------:|
| uniformity                       |   -0.5072 |   -3.371   | 0.1505  |     2.864   |
| variance_floor.hinge             |    0.5464 |    0.5857  | 0.933   |    -0.03926 |
| offdiag_redundancy.mean_abs_corr |    0.1869 |    0.05546 | 3.371   |     0.1315  |
| rankme                           |   69.23   | 1434       | 0.04829 | -1364       |
| effective_rank                   |   25.21   |  374.1     | 0.06739 |  -348.9     |
| alpha                            |    2.049  |    0.9755  | 2.1     |     1.073   |
| epps_pulley                      |  763.9    |   78.09    | 9.782   |   685.8     |
| kurt_topeig.worst                |    2.337  |    0.8352  | 2.798   |     1.502   |



**student.h.cls.L03 vs student.z.dino.tap2**


| metric                           |   value_h |    value_z |      tau |      delta |
|:---------------------------------|----------:|-----------:|---------:|-----------:|
| uniformity                       |   -0.5072 |   -1.677   |  0.3024  |     1.17   |
| variance_floor.hinge             |    0.5464 |    0.9068  |  0.6026  |    -0.3604 |
| offdiag_redundancy.mean_abs_corr |    0.1869 |    0.02812 |  6.647   |     0.1588 |
| rankme                           |   69.23   | 1513       |  0.04574 | -1444      |
| effective_rank                   |   25.21   |  650.7     |  0.03875 |  -625.5    |
| alpha                            |    2.049  |    0.7619  |  2.689   |     1.287  |
| epps_pulley                      |  763.9    |   52.82    | 14.46    |   711.1    |
| kurt_topeig.worst                |    2.337  |   37.39    |  0.06249 |   -35.06   |



**student.h.cls.L06 vs student.z.dino.bottleneck**


| metric                           |   value_h |   value_z |     tau |      delta |
|:---------------------------------|----------:|----------:|--------:|-----------:|
| uniformity                       |   -1.84   |  -3.665   | 0.502   |    1.825   |
| variance_floor.hinge             |    0.1329 |   0.889   | 0.1494  |   -0.7562  |
| offdiag_redundancy.mean_abs_corr |    0.1179 |   0.08262 | 1.427   |    0.03526 |
| rankme                           |  179.8    | 160.2     | 1.122   |   19.55    |
| effective_rank                   |   63.96   |  94.16    | 0.6792  |  -30.2     |
| alpha                            |    1.545  |   1.661   | 0.9307  |   -0.115   |
| epps_pulley                      |  278.6    | 453       | 0.6151  | -174.3     |
| kurt_topeig.worst                |    0.8579 |  46.94    | 0.01828 |  -46.08    |



**student.h.cls.L06 vs student.z.dino.tap1**


| metric                           |   value_h |    value_z |    tau |       delta |
|:---------------------------------|----------:|-----------:|-------:|------------:|
| uniformity                       |   -1.84   |   -3.371   | 0.5458 |     1.531   |
| variance_floor.hinge             |    0.1329 |    0.5857  | 0.2268 |    -0.4528  |
| offdiag_redundancy.mean_abs_corr |    0.1179 |    0.05546 | 2.125  |     0.06242 |
| rankme                           |  179.8    | 1434       | 0.1254 | -1254       |
| effective_rank                   |   63.96   |  374.1     | 0.171  |  -310.2     |
| alpha                            |    1.545  |    0.9755  | 1.584  |     0.5699  |
| epps_pulley                      |  278.6    |   78.09    | 3.568  |   200.5     |
| kurt_topeig.worst                |    0.8579 |    0.8352  | 1.027  |     0.02263 |



**student.h.cls.L06 vs student.z.dino.tap2**


| metric                           |   value_h |    value_z |     tau |       delta |
|:---------------------------------|----------:|-----------:|--------:|------------:|
| uniformity                       |   -1.84   |   -1.677   | 1.097   |    -0.1623  |
| variance_floor.hinge             |    0.1329 |    0.9068  | 0.1465  |    -0.774   |
| offdiag_redundancy.mean_abs_corr |    0.1179 |    0.02812 | 4.192   |     0.08975 |
| rankme                           |  179.8    | 1513       | 0.1188  | -1334       |
| effective_rank                   |   63.96   |  650.7     | 0.09829 |  -586.7     |
| alpha                            |    1.545  |    0.7619  | 2.028   |     0.7836  |
| epps_pulley                      |  278.6    |   52.82    | 5.275   |   225.8     |
| kurt_topeig.worst                |    0.8579 |   37.39    | 0.02294 |   -36.53    |



**student.h.cls.L09 vs student.z.dino.bottleneck**


| metric                           |    value_h |   value_z |      tau |        delta |
|:---------------------------------|-----------:|----------:|---------:|-------------:|
| uniformity                       |  -2.985    |  -3.665   | 0.8145   |    0.68      |
| variance_floor.hinge             |   0.008414 |   0.889   | 0.009465 |   -0.8806    |
| offdiag_redundancy.mean_abs_corr |   0.0828   |   0.08262 | 1.002    |    0.0001885 |
| rankme                           | 243.7      | 160.2     | 1.521    |   83.44      |
| effective_rank                   | 115.6      |  94.16    | 1.228    |   21.47      |
| alpha                            |   1.196    |   1.661   | 0.7206   |   -0.464     |
| epps_pulley                      | 150.1      | 453       | 0.3313   | -302.9       |
| kurt_topeig.worst                |   0.9573   |  46.94    | 0.0204   |  -45.98      |



**student.h.cls.L09 vs student.z.dino.tap1**


| metric                           |    value_h |    value_z |     tau |       delta |
|:---------------------------------|-----------:|-----------:|--------:|------------:|
| uniformity                       |  -2.985    |   -3.371   | 0.8855  |     0.3859  |
| variance_floor.hinge             |   0.008414 |    0.5857  | 0.01437 |    -0.5773  |
| offdiag_redundancy.mean_abs_corr |   0.0828   |    0.05546 | 1.493   |     0.02735 |
| rankme                           | 243.7      | 1434       | 0.17    | -1190       |
| effective_rank                   | 115.6      |  374.1     | 0.3091  |  -258.5     |
| alpha                            |   1.196    |    0.9755  | 1.227   |     0.221   |
| epps_pulley                      | 150.1      |   78.09    | 1.922   |    71.99    |
| kurt_topeig.worst                |   0.9573   |    0.8352  | 1.146   |     0.122   |



**student.h.cls.L09 vs student.z.dino.tap2**


| metric                           |    value_h |    value_z |      tau |       delta |
|:---------------------------------|-----------:|-----------:|---------:|------------:|
| uniformity                       |  -2.985    |   -1.677   | 1.78     |    -1.307   |
| variance_floor.hinge             |   0.008414 |    0.9068  | 0.009279 |    -0.8984  |
| offdiag_redundancy.mean_abs_corr |   0.0828   |    0.02812 | 2.945    |     0.05468 |
| rankme                           | 243.7      | 1513       | 0.161    | -1270       |
| effective_rank                   | 115.6      |  650.7     | 0.1777   |  -535.1     |
| alpha                            |   1.196    |    0.7619  | 1.57     |     0.4346  |
| epps_pulley                      | 150.1      |   52.82    | 2.841    |    97.26    |
| kurt_topeig.worst                |   0.9573   |   37.39    | 0.0256   |   -36.44    |



**student.h.cls.L12 vs student.z.dino.bottleneck**


| metric                           |    value_h |   value_z |     tau |       delta |
|:---------------------------------|-----------:|----------:|--------:|------------:|
| uniformity                       |  -3.447    |  -3.665   | 0.9405  |    0.2181   |
| variance_floor.hinge             |   0.007166 |   0.889   | 0.00806 |   -0.8819   |
| offdiag_redundancy.mean_abs_corr |   0.08088  |   0.08262 | 0.979   |   -0.001732 |
| rankme                           | 243.3      | 160.2     | 1.519   |   83.09     |
| effective_rank                   | 117.4      |  94.16    | 1.246   |   23.19     |
| alpha                            |   1.231    |   1.661   | 0.7412  |   -0.4297   |
| epps_pulley                      | 142.9      | 453       | 0.3155  | -310        |
| kurt_topeig.worst                |   0.8165   |  46.94    | 0.0174  |  -46.12     |



**student.h.cls.L12 vs student.z.dino.tap1**


| metric                           |    value_h |    value_z |     tau |       delta |
|:---------------------------------|-----------:|-----------:|--------:|------------:|
| uniformity                       |  -3.447    |   -3.371   | 1.023   |    -0.07596 |
| variance_floor.hinge             |   0.007166 |    0.5857  | 0.01223 |    -0.5785  |
| offdiag_redundancy.mean_abs_corr |   0.08088  |    0.05546 | 1.458   |     0.02543 |
| rankme                           | 243.3      | 1434       | 0.1697  | -1190       |
| effective_rank                   | 117.4      |  374.1     | 0.3137  |  -256.8     |
| alpha                            |   1.231    |    0.9755  | 1.262   |     0.2553  |
| epps_pulley                      | 142.9      |   78.09    | 1.83    |    64.84    |
| kurt_topeig.worst                |   0.8165   |    0.8352  | 0.9775  |    -0.01878 |



**student.h.cls.L12 vs student.z.dino.tap2**


| metric                           |    value_h |    value_z |      tau |       delta |
|:---------------------------------|-----------:|-----------:|---------:|------------:|
| uniformity                       |  -3.447    |   -1.677   | 2.055    |    -1.769   |
| variance_floor.hinge             |   0.007166 |    0.9068  | 0.007902 |    -0.8996  |
| offdiag_redundancy.mean_abs_corr |   0.08088  |    0.02812 | 2.876    |     0.05276 |
| rankme                           | 243.3      | 1513       | 0.1608   | -1270       |
| effective_rank                   | 117.4      |  650.7     | 0.1803   |  -533.4     |
| alpha                            |   1.231    |    0.7619  | 1.615    |     0.4689  |
| epps_pulley                      | 142.9      |   52.82    | 2.706    |    90.11    |
| kurt_topeig.worst                |   0.8165   |   37.39    | 0.02183  |   -36.58    |



**student.h.gap vs student.z.dino.bottleneck**


| metric                           |   value_h |   value_z |     tau |      delta |
|:---------------------------------|----------:|----------:|--------:|-----------:|
| uniformity                       |   -2.53   |  -3.665   | 0.6903  |    1.135   |
| variance_floor.hinge             |    0.545  |   0.889   | 0.613   |   -0.3441  |
| offdiag_redundancy.mean_abs_corr |    0.1018 |   0.08262 | 1.232   |    0.01916 |
| rankme                           |  221.6    | 160.2     | 1.383   |   61.35    |
| effective_rank                   |   84.43   |  94.16    | 0.8967  |   -9.731   |
| alpha                            |    1.379  |   1.661   | 0.8307  |   -0.2811  |
| epps_pulley                      |  227.5    | 453       | 0.5021  | -225.5     |
| kurt_topeig.worst                |    1.703  |  46.94    | 0.03628 |  -45.23    |



**student.h.gap vs student.z.dino.tap1**


| metric                           |   value_h |    value_z |    tau |       delta |
|:---------------------------------|----------:|-----------:|-------:|------------:|
| uniformity                       |   -2.53   |   -3.371   | 0.7505 |     0.8411  |
| variance_floor.hinge             |    0.545  |    0.5857  | 0.9305 |    -0.04072 |
| offdiag_redundancy.mean_abs_corr |    0.1018 |    0.05546 | 1.835  |     0.04631 |
| rankme                           |  221.6    | 1434       | 0.1546 | -1212       |
| effective_rank                   |   84.43   |  374.1     | 0.2257 |  -289.7     |
| alpha                            |    1.379  |    0.9755  | 1.414  |     0.4039  |
| epps_pulley                      |  227.5    |   78.09    | 2.913  |   149.4     |
| kurt_topeig.worst                |    1.703  |    0.8352  | 2.039  |     0.8678  |



**student.h.gap vs student.z.dino.tap2**


| metric                           |   value_h |    value_z |     tau |       delta |
|:---------------------------------|----------:|-----------:|--------:|------------:|
| uniformity                       |   -2.53   |   -1.677   | 1.508   |    -0.8523  |
| variance_floor.hinge             |    0.545  |    0.9068  | 0.601   |    -0.3618  |
| offdiag_redundancy.mean_abs_corr |    0.1018 |    0.02812 | 3.619   |     0.07365 |
| rankme                           |  221.6    | 1513       | 0.1464  | -1292       |
| effective_rank                   |   84.43   |  650.7     | 0.1298  |  -566.3     |
| alpha                            |    1.379  |    0.7619  | 1.811   |     0.6175  |
| epps_pulley                      |  227.5    |   52.82    | 4.306   |   174.6     |
| kurt_topeig.worst                |    1.703  |   37.39    | 0.04554 |   -35.69    |



**student.h.gap.L03 vs student.z.dino.bottleneck**


| metric                           |   value_h |   value_z |     tau |     delta |
|:---------------------------------|----------:|----------:|--------:|----------:|
| uniformity                       |   -1.134  |  -3.665   | 0.3095  |   2.531   |
| variance_floor.hinge             |    0.6973 |   0.889   | 0.7843  |  -0.1918  |
| offdiag_redundancy.mean_abs_corr |    0.1487 |   0.08262 | 1.8     |   0.06612 |
| rankme                           |  130.5    | 160.2     | 0.8142  | -29.77    |
| effective_rank                   |   30.74   |  94.16    | 0.3265  | -63.42    |
| alpha                            |    1.72   |   1.661   | 1.036   |   0.05947 |
| epps_pulley                      |  578.5    | 453       | 1.277   | 125.5     |
| kurt_topeig.worst                |    1.114  |  46.94    | 0.02373 | -45.82    |



**student.h.gap.L03 vs student.z.dino.tap1**


| metric                           |   value_h |    value_z |     tau |       delta |
|:---------------------------------|----------:|-----------:|--------:|------------:|
| uniformity                       |   -1.134  |   -3.371   | 0.3365  |     2.237   |
| variance_floor.hinge             |    0.6973 |    0.5857  | 1.19    |     0.1116  |
| offdiag_redundancy.mean_abs_corr |    0.1487 |    0.05546 | 2.682   |     0.09328 |
| rankme                           |  130.5    | 1434       | 0.091   | -1303       |
| effective_rank                   |   30.74   |  374.1     | 0.08217 |  -343.4     |
| alpha                            |    1.72   |    0.9755  | 1.763   |     0.7445  |
| epps_pulley                      |  578.5    |   78.09    | 7.408   |   500.4     |
| kurt_topeig.worst                |    1.114  |    0.8352  | 1.333   |     0.2785  |



**student.h.gap.L03 vs student.z.dino.tap2**


| metric                           |   value_h |    value_z |      tau |      delta |
|:---------------------------------|----------:|-----------:|---------:|-----------:|
| uniformity                       |   -1.134  |   -1.677   |  0.6762  |     0.5431 |
| variance_floor.hinge             |    0.6973 |    0.9068  |  0.7689  |    -0.2096 |
| offdiag_redundancy.mean_abs_corr |    0.1487 |    0.02812 |  5.289   |     0.1206 |
| rankme                           |  130.5    | 1513       |  0.0862  | -1383      |
| effective_rank                   |   30.74   |  650.7     |  0.04724 |  -620      |
| alpha                            |    1.72   |    0.7619  |  2.258   |     0.9581 |
| epps_pulley                      |  578.5    |   52.82    | 10.95    |   525.7    |
| kurt_topeig.worst                |    1.114  |   37.39    |  0.02978 |   -36.28   |



**student.h.gap.L06 vs student.z.dino.bottleneck**


| metric                           |   value_h |   value_z |     tau |     delta |
|:---------------------------------|----------:|----------:|--------:|----------:|
| uniformity                       |   -1.724  |  -3.665   | 0.4705  |   1.941   |
| variance_floor.hinge             |    0.6105 |   0.889   | 0.6867  |  -0.2785  |
| offdiag_redundancy.mean_abs_corr |    0.1231 |   0.08262 | 1.49    |   0.04052 |
| rankme                           |  179.5    | 160.2     | 1.12    |  19.28    |
| effective_rank                   |   55.4    |  94.16    | 0.5883  | -38.76    |
| alpha                            |    1.547  |   1.661   | 0.9315  |  -0.1137  |
| epps_pulley                      |  387      | 453       | 0.8543  | -66.01    |
| kurt_topeig.worst                |    2.146  |  46.94    | 0.04572 | -44.79    |



**student.h.gap.L06 vs student.z.dino.tap1**


| metric                           |   value_h |    value_z |    tau |       delta |
|:---------------------------------|----------:|-----------:|-------:|------------:|
| uniformity                       |   -1.724  |   -3.371   | 0.5115 |     1.647   |
| variance_floor.hinge             |    0.6105 |    0.5857  | 1.042  |     0.02482 |
| offdiag_redundancy.mean_abs_corr |    0.1231 |    0.05546 | 2.22   |     0.06768 |
| rankme                           |  179.5    | 1434       | 0.1252 | -1254       |
| effective_rank                   |   55.4    |  374.1     | 0.1481 |  -318.7     |
| alpha                            |    1.547  |    0.9755  | 1.586  |     0.5713  |
| epps_pulley                      |  387      |   78.09    | 4.955  |   308.9     |
| kurt_topeig.worst                |    2.146  |    0.8352  | 2.569  |     1.311   |



**student.h.gap.L06 vs student.z.dino.tap2**


| metric                           |   value_h |    value_z |     tau |       delta |
|:---------------------------------|----------:|-----------:|--------:|------------:|
| uniformity                       |   -1.724  |   -1.677   | 1.028   |    -0.04678 |
| variance_floor.hinge             |    0.6105 |    0.9068  | 0.6732  |    -0.2963  |
| offdiag_redundancy.mean_abs_corr |    0.1231 |    0.02812 | 4.379   |     0.09501 |
| rankme                           |  179.5    | 1513       | 0.1186  | -1334       |
| effective_rank                   |   55.4    |  650.7     | 0.08514 |  -595.3     |
| alpha                            |    1.547  |    0.7619  | 2.03    |     0.7849  |
| epps_pulley                      |  387      |   52.82    | 7.326   |   334.1     |
| kurt_topeig.worst                |    2.146  |   37.39    | 0.05739 |   -35.25    |



**student.h.gap.L09 vs student.z.dino.bottleneck**


| metric                           |   value_h |   value_z |     tau |     delta |
|:---------------------------------|----------:|----------:|--------:|----------:|
| uniformity                       |   -2.207  |  -3.665   | 0.6022  |    1.458  |
| variance_floor.hinge             |    0.552  |   0.889   | 0.6209  |   -0.337  |
| offdiag_redundancy.mean_abs_corr |    0.1042 |   0.08262 | 1.261   |    0.0216 |
| rankme                           |  220.9    | 160.2     | 1.378   |   60.64   |
| effective_rank                   |   84.49   |  94.16    | 0.8972  |   -9.676  |
| alpha                            |    1.326  |   1.661   | 0.7986  |   -0.3344 |
| epps_pulley                      |  262.7    | 453       | 0.58    | -190.3    |
| kurt_topeig.worst                |    2.218  |  46.94    | 0.04725 |  -44.72   |



**student.h.gap.L09 vs student.z.dino.tap1**


| metric                           |   value_h |    value_z |    tau |       delta |
|:---------------------------------|----------:|-----------:|-------:|------------:|
| uniformity                       |   -2.207  |   -3.371   | 0.6547 |     1.164   |
| variance_floor.hinge             |    0.552  |    0.5857  | 0.9425 |    -0.03368 |
| offdiag_redundancy.mean_abs_corr |    0.1042 |    0.05546 | 1.879  |     0.04876 |
| rankme                           |  220.9    | 1434       | 0.1541 | -1213       |
| effective_rank                   |   84.49   |  374.1     | 0.2258 |  -289.6     |
| alpha                            |    1.326  |    0.9755  | 1.359  |     0.3506  |
| epps_pulley                      |  262.7    |   78.09    | 3.364  |   184.6     |
| kurt_topeig.worst                |    2.218  |    0.8352  | 2.655  |     1.383   |



**student.h.gap.L09 vs student.z.dino.tap2**


| metric                           |   value_h |    value_z |     tau |      delta |
|:---------------------------------|----------:|-----------:|--------:|-----------:|
| uniformity                       |   -2.207  |   -1.677   | 1.316   |    -0.5295 |
| variance_floor.hinge             |    0.552  |    0.9068  | 0.6087  |    -0.3548 |
| offdiag_redundancy.mean_abs_corr |    0.1042 |    0.02812 | 3.706   |     0.0761 |
| rankme                           |  220.9    | 1513       | 0.1459  | -1293      |
| effective_rank                   |   84.49   |  650.7     | 0.1298  |  -566.2    |
| alpha                            |    1.326  |    0.7619  | 1.741   |     0.5643 |
| epps_pulley                      |  262.7    |   52.82    | 4.974   |   209.9    |
| kurt_topeig.worst                |    2.218  |   37.39    | 0.05931 |   -35.17   |



**student.h.gap.L12 vs student.z.dino.bottleneck**


| metric                           |   value_h |   value_z |     tau |      delta |
|:---------------------------------|----------:|----------:|--------:|-----------:|
| uniformity                       |   -2.53   |  -3.665   | 0.6903  |    1.135   |
| variance_floor.hinge             |    0.545  |   0.889   | 0.613   |   -0.3441  |
| offdiag_redundancy.mean_abs_corr |    0.1018 |   0.08262 | 1.232   |    0.01916 |
| rankme                           |  221.6    | 160.2     | 1.383   |   61.35    |
| effective_rank                   |   84.43   |  94.16    | 0.8967  |   -9.731   |
| alpha                            |    1.379  |   1.661   | 0.8307  |   -0.2811  |
| epps_pulley                      |  227.5    | 453       | 0.5021  | -225.5     |
| kurt_topeig.worst                |    1.703  |  46.94    | 0.03628 |  -45.23    |



**student.h.gap.L12 vs student.z.dino.tap1**


| metric                           |   value_h |    value_z |    tau |       delta |
|:---------------------------------|----------:|-----------:|-------:|------------:|
| uniformity                       |   -2.53   |   -3.371   | 0.7505 |     0.8411  |
| variance_floor.hinge             |    0.545  |    0.5857  | 0.9305 |    -0.04072 |
| offdiag_redundancy.mean_abs_corr |    0.1018 |    0.05546 | 1.835  |     0.04631 |
| rankme                           |  221.6    | 1434       | 0.1546 | -1212       |
| effective_rank                   |   84.43   |  374.1     | 0.2257 |  -289.7     |
| alpha                            |    1.379  |    0.9755  | 1.414  |     0.4039  |
| epps_pulley                      |  227.5    |   78.09    | 2.913  |   149.4     |
| kurt_topeig.worst                |    1.703  |    0.8352  | 2.039  |     0.8678  |



**student.h.gap.L12 vs student.z.dino.tap2**


| metric                           |   value_h |    value_z |     tau |       delta |
|:---------------------------------|----------:|-----------:|--------:|------------:|
| uniformity                       |   -2.53   |   -1.677   | 1.508   |    -0.8523  |
| variance_floor.hinge             |    0.545  |    0.9068  | 0.601   |    -0.3618  |
| offdiag_redundancy.mean_abs_corr |    0.1018 |    0.02812 | 3.619   |     0.07365 |
| rankme                           |  221.6    | 1513       | 0.1464  | -1292       |
| effective_rank                   |   84.43   |  650.7     | 0.1298  |  -566.3     |
| alpha                            |    1.379  |    0.7619  | 1.811   |     0.6175  |
| epps_pulley                      |  227.5    |   52.82    | 4.306   |   174.6     |
| kurt_topeig.worst                |    1.703  |   37.39    | 0.04554 |   -35.69    |



**teacher.h.cls vs teacher.z.dino.bottleneck**


| metric                           |    value_h |   value_z |      tau |        delta |
|:---------------------------------|-----------:|----------:|---------:|-------------:|
| uniformity                       |  -3.446    |  -3.662   | 0.9411   |    0.2158    |
| variance_floor.hinge             |   0.008027 |   0.8729  | 0.009196 |   -0.8648    |
| offdiag_redundancy.mean_abs_corr |   0.08209  |   0.08261 | 0.9936   |   -0.0005283 |
| rankme                           | 239.2      | 159.9     | 1.496    |   79.29      |
| effective_rank                   | 113.9      |  94       | 1.212    |   19.92      |
| alpha                            |   1.261    |   1.666   | 0.7571   |   -0.4046    |
| epps_pulley                      | 149.9      | 419.9     | 0.357    | -270         |
| kurt_topeig.worst                |   0.823    |  45.09    | 0.01825  |  -44.26      |



**teacher.h.cls vs teacher.z.dino.tap1**


| metric                           |    value_h |    value_z |     tau |       delta |
|:---------------------------------|-----------:|-----------:|--------:|------------:|
| uniformity                       |  -3.446    |   -3.419   | 1.008   |    -0.02733 |
| variance_floor.hinge             |   0.008027 |    0.5769  | 0.01391 |    -0.5689  |
| offdiag_redundancy.mean_abs_corr |   0.08209  |    0.05596 | 1.467   |     0.02612 |
| rankme                           | 239.2      | 1431       | 0.1672  | -1192       |
| effective_rank                   | 113.9      |  367.8     | 0.3097  |  -253.9     |
| alpha                            |   1.261    |    0.98    | 1.287   |     0.2814  |
| epps_pulley                      | 149.9      |   80.12    | 1.871   |    69.78    |
| kurt_topeig.worst                |   0.823    |    0.8372  | 0.983   |    -0.0142  |



**teacher.h.cls vs teacher.z.dino.tap2**


| metric                           |    value_h |    value_z |      tau |       delta |
|:---------------------------------|-----------:|-----------:|---------:|------------:|
| uniformity                       |  -3.446    |   -1.953   | 1.764    |    -1.493   |
| variance_floor.hinge             |   0.008027 |    0.8978  | 0.008941 |    -0.8898  |
| offdiag_redundancy.mean_abs_corr |   0.08209  |    0.02763 | 2.971    |     0.05446 |
| rankme                           | 239.2      | 1539       | 0.1555   | -1300       |
| effective_rank                   | 113.9      |  662.1     | 0.1721   |  -548.1     |
| alpha                            |   1.261    |    0.7525  | 1.676    |     0.5089  |
| epps_pulley                      | 149.9      |   50.26    | 2.982    |    99.64    |
| kurt_topeig.worst                |   0.823    |   37.03    | 0.02223  |   -36.21    |



**teacher.h.cls.L03 vs teacher.z.dino.bottleneck**


| metric                           |   value_h |   value_z |     tau |    delta |
|:---------------------------------|----------:|----------:|--------:|---------:|
| uniformity                       |   -0.5134 |  -3.662   | 0.1402  |   3.149  |
| variance_floor.hinge             |    0.5468 |   0.8729  | 0.6264  |  -0.3261 |
| offdiag_redundancy.mean_abs_corr |    0.1869 |   0.08261 | 2.262   |   0.1042 |
| rankme                           |   70.02   | 159.9     | 0.4378  | -89.92   |
| effective_rank                   |   25.1    |  94       | 0.267   | -68.9    |
| alpha                            |    2.035  |   1.666   | 1.221   |   0.3687 |
| epps_pulley                      |  772.2    | 419.9     | 1.839   | 352.3    |
| kurt_topeig.worst                |    2.526  |  45.09    | 0.05603 | -42.56   |



**teacher.h.cls.L03 vs teacher.z.dino.tap1**


| metric                           |   value_h |    value_z |     tau |       delta |
|:---------------------------------|----------:|-----------:|--------:|------------:|
| uniformity                       |   -0.5134 |   -3.419   | 0.1502  |     2.906   |
| variance_floor.hinge             |    0.5468 |    0.5769  | 0.9478  |    -0.03012 |
| offdiag_redundancy.mean_abs_corr |    0.1869 |    0.05596 | 3.339   |     0.1309  |
| rankme                           |   70.02   | 1431       | 0.04893 | -1361       |
| effective_rank                   |   25.1    |  367.8     | 0.06825 |  -342.7     |
| alpha                            |    2.035  |    0.98    | 2.076   |     1.055   |
| epps_pulley                      |  772.2    |   80.12    | 9.638   |   692.1     |
| kurt_topeig.worst                |    2.526  |    0.8372  | 3.017   |     1.689   |



**teacher.h.cls.L03 vs teacher.z.dino.tap2**


| metric                           |   value_h |    value_z |      tau |      delta |
|:---------------------------------|----------:|-----------:|---------:|-----------:|
| uniformity                       |   -0.5134 |   -1.953   |  0.2628  |     1.44   |
| variance_floor.hinge             |    0.5468 |    0.8978  |  0.609   |    -0.3511 |
| offdiag_redundancy.mean_abs_corr |    0.1869 |    0.02763 |  6.763   |     0.1592 |
| rankme                           |   70.02   | 1539       |  0.0455  | -1469      |
| effective_rank                   |   25.1    |  662.1     |  0.03792 |  -637      |
| alpha                            |    2.035  |    0.7525  |  2.704   |     1.282  |
| epps_pulley                      |  772.2    |   50.26    | 15.36    |   721.9    |
| kurt_topeig.worst                |    2.526  |   37.03    |  0.06822 |   -34.5    |



**teacher.h.cls.L06 vs teacher.z.dino.bottleneck**


| metric                           |   value_h |   value_z |     tau |      delta |
|:---------------------------------|----------:|----------:|--------:|-----------:|
| uniformity                       |   -1.795  |  -3.662   | 0.49    |    1.868   |
| variance_floor.hinge             |    0.148  |   0.8729  | 0.1696  |   -0.7249  |
| offdiag_redundancy.mean_abs_corr |    0.1182 |   0.08261 | 1.431   |    0.03561 |
| rankme                           |  177.2    | 159.9     | 1.108   |   17.25    |
| effective_rank                   |   63      |  94       | 0.6702  |  -31.01    |
| alpha                            |    1.557  |   1.666   | 0.9345  |   -0.1091  |
| epps_pulley                      |  280.1    | 419.9     | 0.6671  | -139.8     |
| kurt_topeig.worst                |    0.8594 |  45.09    | 0.01906 |  -44.23    |



**teacher.h.cls.L06 vs teacher.z.dino.tap1**


| metric                           |   value_h |    value_z |    tau |       delta |
|:---------------------------------|----------:|-----------:|-------:|------------:|
| uniformity                       |   -1.795  |   -3.419   | 0.5249 |     1.624   |
| variance_floor.hinge             |    0.148  |    0.5769  | 0.2566 |    -0.4289  |
| offdiag_redundancy.mean_abs_corr |    0.1182 |    0.05596 | 2.113  |     0.06226 |
| rankme                           |  177.2    | 1431       | 0.1238 | -1254       |
| effective_rank                   |   63      |  367.8     | 0.1713 |  -304.8     |
| alpha                            |    1.557  |    0.98    | 1.589  |     0.577   |
| epps_pulley                      |  280.1    |   80.12    | 3.496  |   200       |
| kurt_topeig.worst                |    0.8594 |    0.8372  | 1.026  |     0.02216 |



**teacher.h.cls.L06 vs teacher.z.dino.tap2**


| metric                           |   value_h |    value_z |     tau |      delta |
|:---------------------------------|----------:|-----------:|--------:|-----------:|
| uniformity                       |   -1.795  |   -1.953   | 0.9187  |     0.1588 |
| variance_floor.hinge             |    0.148  |    0.8978  | 0.1649  |    -0.7498 |
| offdiag_redundancy.mean_abs_corr |    0.1182 |    0.02763 | 4.279   |     0.0906 |
| rankme                           |  177.2    | 1539       | 0.1151  | -1362      |
| effective_rank                   |   63      |  662.1     | 0.09515 |  -599.1    |
| alpha                            |    1.557  |    0.7525  | 2.069   |     0.8045 |
| epps_pulley                      |  280.1    |   50.26    | 5.573   |   229.8    |
| kurt_topeig.worst                |    0.8594 |   37.03    | 0.02321 |   -36.17   |



**teacher.h.cls.L09 vs teacher.z.dino.bottleneck**


| metric                           |    value_h |   value_z |     tau |       delta |
|:---------------------------------|-----------:|----------:|--------:|------------:|
| uniformity                       |  -2.958    |  -3.662   | 0.8077  |    0.7044   |
| variance_floor.hinge             |   0.009887 |   0.8729  | 0.01133 |   -0.863    |
| offdiag_redundancy.mean_abs_corr |   0.08394  |   0.08261 | 1.016   |    0.001329 |
| rankme                           | 239.7      | 159.9     | 1.499   |   79.73     |
| effective_rank                   | 112.3      |  94       | 1.194   |   18.26     |
| alpha                            |   1.222    |   1.666   | 0.7334  |   -0.4441   |
| epps_pulley                      | 157.2      | 419.9     | 0.3744  | -262.7      |
| kurt_topeig.worst                |   0.9633   |  45.09    | 0.02136 |  -44.12     |



**teacher.h.cls.L09 vs teacher.z.dino.tap1**


| metric                           |    value_h |    value_z |     tau |       delta |
|:---------------------------------|-----------:|-----------:|--------:|------------:|
| uniformity                       |  -2.958    |   -3.419   | 0.8651  |     0.4612  |
| variance_floor.hinge             |   0.009887 |    0.5769  | 0.01714 |    -0.567   |
| offdiag_redundancy.mean_abs_corr |   0.08394  |    0.05596 | 1.5     |     0.02798 |
| rankme                           | 239.7      | 1431       | 0.1675  | -1191       |
| effective_rank                   | 112.3      |  367.8     | 0.3052  |  -255.5     |
| alpha                            |   1.222    |    0.98    | 1.247   |     0.2419  |
| epps_pulley                      | 157.2      |   80.12    | 1.962   |    77.08    |
| kurt_topeig.worst                |   0.9633   |    0.8372  | 1.151   |     0.126   |



**teacher.h.cls.L09 vs teacher.z.dino.tap2**


| metric                           |    value_h |    value_z |     tau |       delta |
|:---------------------------------|-----------:|-----------:|--------:|------------:|
| uniformity                       |  -2.958    |   -1.953   | 1.514   |    -1.004   |
| variance_floor.hinge             |   0.009887 |    0.8978  | 0.01101 |    -0.8879  |
| offdiag_redundancy.mean_abs_corr |   0.08394  |    0.02763 | 3.038   |     0.05632 |
| rankme                           | 239.7      | 1539       | 0.1557  | -1299       |
| effective_rank                   | 112.3      |  662.1     | 0.1696  |  -549.8     |
| alpha                            |   1.222    |    0.7525  | 1.624   |     0.4694  |
| epps_pulley                      | 157.2      |   50.26    | 3.128   |   106.9     |
| kurt_topeig.worst                |   0.9633   |   37.03    | 0.02601 |   -36.07    |



**teacher.h.cls.L12 vs teacher.z.dino.bottleneck**


| metric                           |    value_h |   value_z |      tau |        delta |
|:---------------------------------|-----------:|----------:|---------:|-------------:|
| uniformity                       |  -3.446    |  -3.662   | 0.9411   |    0.2158    |
| variance_floor.hinge             |   0.008027 |   0.8729  | 0.009196 |   -0.8648    |
| offdiag_redundancy.mean_abs_corr |   0.08209  |   0.08261 | 0.9936   |   -0.0005283 |
| rankme                           | 239.2      | 159.9     | 1.496    |   79.29      |
| effective_rank                   | 113.9      |  94       | 1.212    |   19.92      |
| alpha                            |   1.261    |   1.666   | 0.7571   |   -0.4046    |
| epps_pulley                      | 149.9      | 419.9     | 0.357    | -270         |
| kurt_topeig.worst                |   0.823    |  45.09    | 0.01825  |  -44.26      |



**teacher.h.cls.L12 vs teacher.z.dino.tap1**


| metric                           |    value_h |    value_z |     tau |       delta |
|:---------------------------------|-----------:|-----------:|--------:|------------:|
| uniformity                       |  -3.446    |   -3.419   | 1.008   |    -0.02733 |
| variance_floor.hinge             |   0.008027 |    0.5769  | 0.01391 |    -0.5689  |
| offdiag_redundancy.mean_abs_corr |   0.08209  |    0.05596 | 1.467   |     0.02612 |
| rankme                           | 239.2      | 1431       | 0.1672  | -1192       |
| effective_rank                   | 113.9      |  367.8     | 0.3097  |  -253.9     |
| alpha                            |   1.261    |    0.98    | 1.287   |     0.2814  |
| epps_pulley                      | 149.9      |   80.12    | 1.871   |    69.78    |
| kurt_topeig.worst                |   0.823    |    0.8372  | 0.983   |    -0.0142  |



**teacher.h.cls.L12 vs teacher.z.dino.tap2**


| metric                           |    value_h |    value_z |      tau |       delta |
|:---------------------------------|-----------:|-----------:|---------:|------------:|
| uniformity                       |  -3.446    |   -1.953   | 1.764    |    -1.493   |
| variance_floor.hinge             |   0.008027 |    0.8978  | 0.008941 |    -0.8898  |
| offdiag_redundancy.mean_abs_corr |   0.08209  |    0.02763 | 2.971    |     0.05446 |
| rankme                           | 239.2      | 1539       | 0.1555   | -1300       |
| effective_rank                   | 113.9      |  662.1     | 0.1721   |  -548.1     |
| alpha                            |   1.261    |    0.7525  | 1.676    |     0.5089  |
| epps_pulley                      | 149.9      |   50.26    | 2.982    |    99.64    |
| kurt_topeig.worst                |   0.823    |   37.03    | 0.02223  |   -36.21    |



**teacher.h.gap vs teacher.z.dino.bottleneck**


| metric                           |   value_h |   value_z |     tau |      delta |
|:---------------------------------|----------:|----------:|--------:|-----------:|
| uniformity                       |   -2.581  |  -3.662   | 0.7049  |    1.081   |
| variance_floor.hinge             |    0.5455 |   0.8729  | 0.6249  |   -0.3274  |
| offdiag_redundancy.mean_abs_corr |    0.104  |   0.08261 | 1.259   |    0.02143 |
| rankme                           |  220      | 159.9     | 1.376   |   60.07    |
| effective_rank                   |   81.44   |  94       | 0.8663  |  -12.57    |
| alpha                            |    1.4    |   1.666   | 0.84    |   -0.2665  |
| epps_pulley                      |  234.4    | 419.9     | 0.5583  | -185.5     |
| kurt_topeig.worst                |    1.765  |  45.09    | 0.03916 |  -43.32    |



**teacher.h.gap vs teacher.z.dino.tap1**


| metric                           |   value_h |    value_z |    tau |       delta |
|:---------------------------------|----------:|-----------:|-------:|------------:|
| uniformity                       |   -2.581  |   -3.419   | 0.755  |     0.8375  |
| variance_floor.hinge             |    0.5455 |    0.5769  | 0.9456 |    -0.03141 |
| offdiag_redundancy.mean_abs_corr |    0.104  |    0.05596 | 1.859  |     0.04808 |
| rankme                           |  220      | 1431       | 0.1537 | -1211       |
| effective_rank                   |   81.44   |  367.8     | 0.2214 |  -286.4     |
| alpha                            |    1.4    |    0.98    | 1.428  |     0.4195  |
| epps_pulley                      |  234.4    |   80.12    | 2.926  |   154.3     |
| kurt_topeig.worst                |    1.765  |    0.8372  | 2.109  |     0.9282  |



**teacher.h.gap vs teacher.z.dino.tap2**


| metric                           |   value_h |    value_z |     tau |       delta |
|:---------------------------------|----------:|-----------:|--------:|------------:|
| uniformity                       |   -2.581  |   -1.953   | 1.322   |    -0.6282  |
| variance_floor.hinge             |    0.5455 |    0.8978  | 0.6076  |    -0.3523  |
| offdiag_redundancy.mean_abs_corr |    0.104  |    0.02763 | 3.766   |     0.07641 |
| rankme                           |  220      | 1539       | 0.143   | -1319       |
| effective_rank                   |   81.44   |  662.1     | 0.123   |  -580.6     |
| alpha                            |    1.4    |    0.7525  | 1.86    |     0.647   |
| epps_pulley                      |  234.4    |   50.26    | 4.664   |   184.2     |
| kurt_topeig.worst                |    1.765  |   37.03    | 0.04768 |   -35.26    |



**teacher.h.gap.L03 vs teacher.z.dino.bottleneck**


| metric                           |   value_h |   value_z |     tau |     delta |
|:---------------------------------|----------:|----------:|--------:|----------:|
| uniformity                       |   -1.185  |  -3.662   | 0.3237  |   2.477   |
| variance_floor.hinge             |    0.6995 |   0.8729  | 0.8013  |  -0.1734  |
| offdiag_redundancy.mean_abs_corr |    0.1494 |   0.08261 | 1.809   |   0.06681 |
| rankme                           |  131.3    | 159.9     | 0.821   | -28.62    |
| effective_rank                   |   29.93   |  94       | 0.3184  | -64.07    |
| alpha                            |    1.725  |   1.666   | 1.035   |   0.05898 |
| epps_pulley                      |  595.6    | 419.9     | 1.418   | 175.7     |
| kurt_topeig.worst                |    1.282  |  45.09    | 0.02843 | -43.8     |



**teacher.h.gap.L03 vs teacher.z.dino.tap1**


| metric                           |   value_h |    value_z |     tau |       delta |
|:---------------------------------|----------:|-----------:|--------:|------------:|
| uniformity                       |   -1.185  |   -3.419   | 0.3467  |     2.234   |
| variance_floor.hinge             |    0.6995 |    0.5769  | 1.212   |     0.1226  |
| offdiag_redundancy.mean_abs_corr |    0.1494 |    0.05596 | 2.67    |     0.09346 |
| rankme                           |  131.3    | 1431       | 0.09176 | -1300       |
| effective_rank                   |   29.93   |  367.8     | 0.08137 |  -337.9     |
| alpha                            |    1.725  |    0.98    | 1.76    |     0.745   |
| epps_pulley                      |  595.6    |   80.12    | 7.434   |   515.5     |
| kurt_topeig.worst                |    1.282  |    0.8372  | 1.531   |     0.4445  |



**teacher.h.gap.L03 vs teacher.z.dino.tap2**


| metric                           |   value_h |    value_z |      tau |      delta |
|:---------------------------------|----------:|-----------:|---------:|-----------:|
| uniformity                       |   -1.185  |   -1.953   |  0.6068  |     0.7679 |
| variance_floor.hinge             |    0.6995 |    0.8978  |  0.7791  |    -0.1984 |
| offdiag_redundancy.mean_abs_corr |    0.1494 |    0.02763 |  5.408   |     0.1218 |
| rankme                           |  131.3    | 1539       |  0.08533 | -1408      |
| effective_rank                   |   29.93   |  662.1     |  0.04521 |  -632.1    |
| alpha                            |    1.725  |    0.7525  |  2.292   |     0.9725 |
| epps_pulley                      |  595.6    |   50.26    | 11.85    |   545.3    |
| kurt_topeig.worst                |    1.282  |   37.03    |  0.03462 |   -35.75   |



**teacher.h.gap.L06 vs teacher.z.dino.bottleneck**


| metric                           |   value_h |   value_z |     tau |     delta |
|:---------------------------------|----------:|----------:|--------:|----------:|
| uniformity                       |   -1.74   |  -3.662   | 0.4751  |   1.922   |
| variance_floor.hinge             |    0.6127 |   0.8729  | 0.702   |  -0.2601  |
| offdiag_redundancy.mean_abs_corr |    0.124  |   0.08261 | 1.501   |   0.04137 |
| rankme                           |  178      | 159.9     | 1.113   |  18.03    |
| effective_rank                   |   54.05   |  94       | 0.575   | -39.95    |
| alpha                            |    1.56   |   1.666   | 0.9361  |  -0.1065  |
| epps_pulley                      |  393.6    | 419.9     | 0.9374  | -26.3     |
| kurt_topeig.worst                |    2.269  |  45.09    | 0.05033 | -42.82    |



**teacher.h.gap.L06 vs teacher.z.dino.tap1**


| metric                           |   value_h |    value_z |    tau |       delta |
|:---------------------------------|----------:|-----------:|-------:|------------:|
| uniformity                       |   -1.74   |   -3.419   | 0.5089 |     1.679   |
| variance_floor.hinge             |    0.6127 |    0.5769  | 1.062  |     0.03586 |
| offdiag_redundancy.mean_abs_corr |    0.124  |    0.05596 | 2.215  |     0.06802 |
| rankme                           |  178      | 1431       | 0.1244 | -1253       |
| effective_rank                   |   54.05   |  367.8     | 0.147  |  -313.7     |
| alpha                            |    1.56   |    0.98    | 1.591  |     0.5796  |
| epps_pulley                      |  393.6    |   80.12    | 4.913  |   313.5     |
| kurt_topeig.worst                |    2.269  |    0.8372  | 2.71   |     1.432   |



**teacher.h.gap.L06 vs teacher.z.dino.tap2**


| metric                           |   value_h |    value_z |     tau |       delta |
|:---------------------------------|----------:|-----------:|--------:|------------:|
| uniformity                       |   -1.74   |   -1.953   | 0.8908  |     0.2133  |
| variance_floor.hinge             |    0.6127 |    0.8978  | 0.6825  |    -0.2851  |
| offdiag_redundancy.mean_abs_corr |    0.124  |    0.02763 | 4.488   |     0.09635 |
| rankme                           |  178      | 1539       | 0.1156  | -1361       |
| effective_rank                   |   54.05   |  662.1     | 0.08164 |  -608       |
| alpha                            |    1.56   |    0.7525  | 2.073   |     0.8071  |
| epps_pulley                      |  393.6    |   50.26    | 7.831   |   343.3     |
| kurt_topeig.worst                |    2.269  |   37.03    | 0.06128 |   -34.76    |



**teacher.h.gap.L09 vs teacher.z.dino.bottleneck**


| metric                           |   value_h |   value_z |     tau |      delta |
|:---------------------------------|----------:|----------:|--------:|-----------:|
| uniformity                       |   -2.249  |  -3.662   | 0.6142  |    1.413   |
| variance_floor.hinge             |    0.553  |   0.8729  | 0.6336  |   -0.3198  |
| offdiag_redundancy.mean_abs_corr |    0.1064 |   0.08261 | 1.287   |    0.02374 |
| rankme                           |  219.2    | 159.9     | 1.371   |   59.3     |
| effective_rank                   |   81.69   |  94       | 0.869   |  -12.31    |
| alpha                            |    1.341  |   1.666   | 0.8051  |   -0.3247  |
| epps_pulley                      |  274.5    | 419.9     | 0.6537  | -145.4     |
| kurt_topeig.worst                |    2.804  |  45.09    | 0.06219 |  -42.28    |



**teacher.h.gap.L09 vs teacher.z.dino.tap1**


| metric                           |   value_h |    value_z |    tau |       delta |
|:---------------------------------|----------:|-----------:|-------:|------------:|
| uniformity                       |   -2.249  |   -3.419   | 0.6579 |     1.17    |
| variance_floor.hinge             |    0.553  |    0.5769  | 0.9587 |    -0.02385 |
| offdiag_redundancy.mean_abs_corr |    0.1064 |    0.05596 | 1.9    |     0.05039 |
| rankme                           |  219.2    | 1431       | 0.1532 | -1212       |
| effective_rank                   |   81.69   |  367.8     | 0.2221 |  -286.1     |
| alpha                            |    1.341  |    0.98    | 1.369  |     0.3613  |
| epps_pulley                      |  274.5    |   80.12    | 3.426  |   194.4     |
| kurt_topeig.worst                |    2.804  |    0.8372  | 3.349  |     1.967   |



**teacher.h.gap.L09 vs teacher.z.dino.tap2**


| metric                           |   value_h |    value_z |     tau |       delta |
|:---------------------------------|----------:|-----------:|--------:|------------:|
| uniformity                       |   -2.249  |   -1.953   | 1.152   |    -0.2959  |
| variance_floor.hinge             |    0.553  |    0.8978  | 0.616   |    -0.3448  |
| offdiag_redundancy.mean_abs_corr |    0.1064 |    0.02763 | 3.85    |     0.07873 |
| rankme                           |  219.2    | 1539       | 0.1425  | -1320       |
| effective_rank                   |   81.69   |  662.1     | 0.1234  |  -580.4     |
| alpha                            |    1.341  |    0.7525  | 1.783   |     0.5888  |
| epps_pulley                      |  274.5    |   50.26    | 5.461   |   224.2     |
| kurt_topeig.worst                |    2.804  |   37.03    | 0.07572 |   -34.23    |



**teacher.h.gap.L12 vs teacher.z.dino.bottleneck**


| metric                           |   value_h |   value_z |     tau |      delta |
|:---------------------------------|----------:|----------:|--------:|-----------:|
| uniformity                       |   -2.581  |  -3.662   | 0.7049  |    1.081   |
| variance_floor.hinge             |    0.5455 |   0.8729  | 0.6249  |   -0.3274  |
| offdiag_redundancy.mean_abs_corr |    0.104  |   0.08261 | 1.259   |    0.02143 |
| rankme                           |  220      | 159.9     | 1.376   |   60.07    |
| effective_rank                   |   81.44   |  94       | 0.8663  |  -12.57    |
| alpha                            |    1.4    |   1.666   | 0.84    |   -0.2665  |
| epps_pulley                      |  234.4    | 419.9     | 0.5583  | -185.5     |
| kurt_topeig.worst                |    1.765  |  45.09    | 0.03916 |  -43.32    |



**teacher.h.gap.L12 vs teacher.z.dino.tap1**


| metric                           |   value_h |    value_z |    tau |       delta |
|:---------------------------------|----------:|-----------:|-------:|------------:|
| uniformity                       |   -2.581  |   -3.419   | 0.755  |     0.8375  |
| variance_floor.hinge             |    0.5455 |    0.5769  | 0.9456 |    -0.03141 |
| offdiag_redundancy.mean_abs_corr |    0.104  |    0.05596 | 1.859  |     0.04808 |
| rankme                           |  220      | 1431       | 0.1537 | -1211       |
| effective_rank                   |   81.44   |  367.8     | 0.2214 |  -286.4     |
| alpha                            |    1.4    |    0.98    | 1.428  |     0.4195  |
| epps_pulley                      |  234.4    |   80.12    | 2.926  |   154.3     |
| kurt_topeig.worst                |    1.765  |    0.8372  | 2.109  |     0.9282  |



**teacher.h.gap.L12 vs teacher.z.dino.tap2**


| metric                           |   value_h |    value_z |     tau |       delta |
|:---------------------------------|----------:|-----------:|--------:|------------:|
| uniformity                       |   -2.581  |   -1.953   | 1.322   |    -0.6282  |
| variance_floor.hinge             |    0.5455 |    0.8978  | 0.6076  |    -0.3523  |
| offdiag_redundancy.mean_abs_corr |    0.104  |    0.02763 | 3.766   |     0.07641 |
| rankme                           |  220      | 1539       | 0.143   | -1319       |
| effective_rank                   |   81.44   |  662.1     | 0.123   |  -580.6     |
| alpha                            |    1.4    |    0.7525  | 1.86    |     0.647   |
| epps_pulley                      |  234.4    |   50.26    | 4.664   |   184.2     |
| kurt_topeig.worst                |    1.765  |   37.03    | 0.04768 |   -35.26    |



### Pair metrics (alignment ↓ = more view-invariant)


| manifest                   | space                     | metric         |   value |   ci_lo |   ci_hi |
|:---------------------------|:--------------------------|:---------------|--------:|--------:|--------:|
| in100.pairs100.v1@audit_v1 | student.h.cls             | alignment      |  0.6806 |  0.6739 |  0.6862 |
| in100.pairs100.v1@audit_v1 | student.h.cls             | cos_invariance |  0.6597 |  0.6556 |  0.6627 |
| in100.pairs100.v1@audit_v1 | student.h.gap             | alignment      |  0.5526 |  0.5473 |  0.5569 |
| in100.pairs100.v1@audit_v1 | student.h.gap             | cos_invariance |  0.7237 |  0.7212 |  0.7258 |
| in100.pairs100.v1@audit_v1 | student.z.dino.bottleneck | alignment      |  0.5035 |  0.4965 |  0.5092 |
| in100.pairs100.v1@audit_v1 | student.z.dino.bottleneck | cos_invariance |  0.7483 |  0.7439 |  0.7517 |
| in100.pairs100.v1@audit_v1 | student.z.dino.tap1       | alignment      |  0.6366 |  0.6303 |  0.6423 |
| in100.pairs100.v1@audit_v1 | student.z.dino.tap1       | cos_invariance |  0.6817 |  0.6778 |  0.6847 |
| in100.pairs100.v1@audit_v1 | student.z.dino.tap2       | alignment      |  0.2808 |  0.2774 |  0.2834 |
| in100.pairs100.v1@audit_v1 | student.z.dino.tap2       | cos_invariance |  0.8596 |  0.8579 |  0.8611 |
| in100.pairs100.v1@audit_v1 | teacher.h.cls             | alignment      |  0.6801 |  0.6734 |  0.6857 |
| in100.pairs100.v1@audit_v1 | teacher.h.cls             | cos_invariance |  0.66   |  0.6559 |  0.663  |
| in100.pairs100.v1@audit_v1 | teacher.h.gap             | alignment      |  0.5618 |  0.5562 |  0.5662 |
| in100.pairs100.v1@audit_v1 | teacher.h.gap             | cos_invariance |  0.7191 |  0.7166 |  0.7212 |
| in100.pairs100.v1@audit_v1 | teacher.z.dino.bottleneck | alignment      |  0.4981 |  0.491  |  0.5039 |
| in100.pairs100.v1@audit_v1 | teacher.z.dino.bottleneck | cos_invariance |  0.7509 |  0.7466 |  0.7543 |
| in100.pairs100.v1@audit_v1 | teacher.z.dino.tap1       | alignment      |  0.6484 |  0.6421 |  0.6543 |
| in100.pairs100.v1@audit_v1 | teacher.z.dino.tap1       | cos_invariance |  0.6758 |  0.6719 |  0.6789 |
| in100.pairs100.v1@audit_v1 | teacher.z.dino.tap2       | alignment      |  0.3349 |  0.3312 |  0.338  |
| in100.pairs100.v1@audit_v1 | teacher.z.dino.tap2       | cos_invariance |  0.8325 |  0.8305 |  0.8342 |
| in100.pairs100.v1@own_dino | student.h.cls             | alignment      |  0.2952 |  0.2913 |  0.2993 |
| in100.pairs100.v1@own_dino | student.h.cls             | cos_invariance |  0.8524 |  0.8504 |  0.8544 |
| in100.pairs100.v1@own_dino | student.h.gap             | alignment      |  0.2767 |  0.2739 |  0.2794 |
| in100.pairs100.v1@own_dino | student.h.gap             | cos_invariance |  0.8616 |  0.8602 |  0.863  |
| in100.pairs100.v1@own_dino | student.z.dino.bottleneck | alignment      |  0.1674 |  0.1637 |  0.1705 |
| in100.pairs100.v1@own_dino | student.z.dino.bottleneck | cos_invariance |  0.9163 |  0.9146 |  0.9179 |
| in100.pairs100.v1@own_dino | student.z.dino.tap1       | alignment      |  0.2622 |  0.2588 |  0.266  |
| in100.pairs100.v1@own_dino | student.z.dino.tap1       | cos_invariance |  0.8689 |  0.8671 |  0.8708 |
| in100.pairs100.v1@own_dino | student.z.dino.tap2       | alignment      |  0.1203 |  0.1185 |  0.122  |
| in100.pairs100.v1@own_dino | student.z.dino.tap2       | cos_invariance |  0.9398 |  0.9391 |  0.9408 |
| in100.pairs100.v1@own_dino | teacher.h.cls             | alignment      |  0.2947 |  0.2908 |  0.2989 |
| in100.pairs100.v1@own_dino | teacher.h.cls             | cos_invariance |  0.8526 |  0.8506 |  0.8546 |
| in100.pairs100.v1@own_dino | teacher.h.gap             | alignment      |  0.2804 |  0.2777 |  0.2832 |
| in100.pairs100.v1@own_dino | teacher.h.gap             | cos_invariance |  0.8598 |  0.8583 |  0.8611 |
| in100.pairs100.v1@own_dino | teacher.z.dino.bottleneck | alignment      |  0.1657 |  0.1619 |  0.1687 |
| in100.pairs100.v1@own_dino | teacher.z.dino.bottleneck | cos_invariance |  0.9172 |  0.9155 |  0.9188 |
| in100.pairs100.v1@own_dino | teacher.z.dino.tap1       | alignment      |  0.2669 |  0.2634 |  0.2708 |
| in100.pairs100.v1@own_dino | teacher.z.dino.tap1       | cos_invariance |  0.8665 |  0.8648 |  0.8684 |
| in100.pairs100.v1@own_dino | teacher.z.dino.tap2       | alignment      |  0.1426 |  0.1404 |  0.1447 |
| in100.pairs100.v1@own_dino | teacher.z.dino.tap2       | cos_invariance |  0.9287 |  0.9277 |  0.9298 |



### Cross-space similarity (CKA is contested — read as the triple)


| space                                   | metric              |   value |
|:----------------------------------------|:--------------------|--------:|
| teacher.h.cls~teacher.z.dino.bottleneck | cka_linear          |  0.4388 |
| teacher.h.cls~teacher.z.dino.bottleneck | neighbor_jaccard    |  0.2321 |
| teacher.h.cls~teacher.z.dino.bottleneck | procrustes_distance |  0.9176 |
| teacher.h.cls~teacher.z.dino.bottleneck | knn_label_agreement |  0.5804 |
| teacher.h.cls~teacher.z.dino.tap1       | cka_linear          |  0.9156 |
| teacher.h.cls~teacher.z.dino.tap1       | neighbor_jaccard    |  0.5777 |
| teacher.h.cls~teacher.z.dino.tap1       | procrustes_distance |  0.3564 |
| teacher.h.cls~teacher.z.dino.tap1       | knn_label_agreement |  0.7516 |
| teacher.h.cls~teacher.z.dino.tap2       | cka_linear          |  0.5603 |
| teacher.h.cls~teacher.z.dino.tap2       | neighbor_jaccard    |  0.3115 |
| teacher.h.cls~teacher.z.dino.tap2       | procrustes_distance |  0.8726 |
| teacher.h.cls~teacher.z.dino.tap2       | knn_label_agreement |  0.6184 |



### Probes


| space                     |   knn_v1_k20 |   knn_v1_k200 |   linear_house_v1 |   linear_l2_v1 |   linear_raw_v1 |
|:--------------------------|-------------:|--------------:|------------------:|---------------:|----------------:|
| student.h.cls             |       0.6452 |        0.6268 |            0.6982 |         0.6808 |          0.7002 |
| student.h.cls.L03         |       0.2292 |        0.2076 |            0.3066 |         0.2104 |          0.2984 |
| student.h.cls.L06         |       0.3918 |        0.3638 |            0.5026 |         0.4028 |          0.4954 |
| student.h.cls.L09         |       0.5604 |        0.5372 |            0.6348 |         0.5952 |          0.6368 |
| student.h.cls.L12         |       0.6452 |        0.6268 |            0.6982 |         0.6808 |          0.7006 |
| student.h.gap             |       0.5292 |        0.4884 |            0.6584 |         0.5852 |          0.6578 |
| student.h.gap.L03         |       0.2572 |        0.2368 |            0.3696 |         0.2688 |          0.3518 |
| student.h.gap.L06         |       0.3998 |        0.3718 |            0.5230 |         0.4182 |          0.5090 |
| student.h.gap.L09         |       0.5026 |        0.4674 |            0.6278 |         0.5456 |          0.6256 |
| student.h.gap.L12         |       0.5292 |        0.4884 |            0.6584 |         0.5852 |          0.6578 |
| student.z.dino.bottleneck |       0.5986 |        0.5786 |            0.6106 |         0.6016 |          0.6020 |
| student.z.dino.tap1       |       0.6384 |        0.6158 |            0.6914 |         0.6946 |          0.7004 |
| student.z.dino.tap2       |       0.6102 |        0.6026 |            0.6738 |         0.6352 |          0.6806 |
| teacher.h.cls             |       0.6452 |        0.6256 |            0.7010 |         0.6800 |          0.7026 |
| teacher.h.cls.L03         |       0.2274 |        0.2062 |            0.3062 |         0.2080 |          0.2958 |
| teacher.h.cls.L06         |       0.3910 |        0.3628 |            0.4998 |         0.4048 |          0.4956 |
| teacher.h.cls.L09         |       0.5624 |        0.5370 |            0.6380 |         0.5960 |          0.6400 |
| teacher.h.cls.L12         |       0.6452 |        0.6256 |            0.7010 |         0.6800 |          0.7026 |
| teacher.h.gap             |       0.5238 |        0.4880 |            0.6602 |         0.5844 |          0.6582 |
| teacher.h.gap.L03         |       0.2556 |        0.2388 |            0.3708 |         0.2696 |          0.3546 |
| teacher.h.gap.L06         |       0.3954 |        0.3686 |            0.5240 |         0.4184 |          0.5080 |
| teacher.h.gap.L09         |       0.5014 |        0.4634 |            0.6252 |         0.5432 |          0.6234 |
| teacher.h.gap.L12         |       0.5238 |        0.4880 |            0.6602 |         0.5844 |          0.6582 |
| teacher.z.dino.bottleneck |       0.5968 |        0.5820 |            0.6120 |         0.6014 |          0.6078 |
| teacher.z.dino.tap1       |       0.6386 |        0.6158 |            0.6908 |         0.6952 |          0.7016 |
| teacher.z.dino.tap2       |       0.6194 |        0.6074 |            0.6714 |         0.6410 |          0.6810 |



## in100.randinit-s0.ext

### Battery (variant raw|full)

| model.space                                       |   rankme |   effective_rank |   alpha |   epps_pulley |   kurt_topeig.worst |   uniformity |   offdiag_redundancy.mean_abs_corr |   variance_floor.hinge |
|:--------------------------------------------------|---------:|-----------------:|--------:|--------------:|--------------------:|-------------:|-----------------------------------:|-----------------------:|
| in100.randinit-s0.ext · student.h.cls             |    91.52 |            6.936 |   1.977 |          3005 |               1.512 |       -1.884 |                             0.3766 |                 0.2294 |
| in100.randinit-s0.ext · student.h.cls.L03         |    79.61 |            5.8   |   2.016 |          3040 |               2.285 |       -1.84  |                             0.385  |                 0.2385 |
| in100.randinit-s0.ext · student.h.cls.L06         |    84.84 |            6.122 |   1.994 |          3658 |               3     |       -1.853 |                             0.3874 |                 0.2413 |
| in100.randinit-s0.ext · student.h.cls.L09         |    89.5  |            6.856 |   1.986 |          2864 |               2.54  |       -1.909 |                             0.3729 |                 0.2281 |
| in100.randinit-s0.ext · student.h.cls.L12         |    91.52 |            6.936 |   1.977 |          3005 |               1.512 |       -1.884 |                             0.3766 |                 0.2294 |
| in100.randinit-s0.ext · student.h.gap             |    53.05 |            3.532 |   2.142 |          3408 |               5.934 |       -1.678 |                             0.4748 |                 0.5105 |
| in100.randinit-s0.ext · student.h.gap.L03         |    46.36 |            2.83  |   2.109 |          3817 |               9.224 |       -1.764 |                             0.5248 |                 0.5378 |
| in100.randinit-s0.ext · student.h.gap.L06         |    50.08 |            3.204 |   2.123 |          3406 |               9.209 |       -1.742 |                             0.5009 |                 0.5264 |
| in100.randinit-s0.ext · student.h.gap.L09         |    52.47 |            3.43  |   2.129 |          3407 |               6.828 |       -1.715 |                             0.4901 |                 0.5185 |
| in100.randinit-s0.ext · student.h.gap.L12         |    53.05 |            3.532 |   2.142 |          3408 |               5.934 |       -1.678 |                             0.4748 |                 0.5105 |
| in100.randinit-s0.ext · student.z.dino.bottleneck |    82.12 |            9.318 |   1.911 |          2331 |               1.546 |       -1.799 |                             0.333  |                 0.953  |
| in100.randinit-s0.ext · student.z.dino.tap1       |   303.7  |            9.298 |   1.836 |          2530 |               1.523 |       -1.824 |                             0.3325 |                 0.7438 |
| in100.randinit-s0.ext · student.z.dino.tap2       |   301.2  |            9.986 |   1.813 |          2327 |               1.526 |       -1.803 |                             0.3349 |                 0.9212 |



### Transfer ratios τ = value(h) / value(z)


**student.h.cls vs student.z.dino.bottleneck**


| metric                           |   value_h |   value_z |    tau |     delta |
|:---------------------------------|----------:|----------:|-------:|----------:|
| uniformity                       |   -1.884  |    -1.799 | 1.047  |  -0.08499 |
| variance_floor.hinge             |    0.2294 |     0.953 | 0.2407 |  -0.7236  |
| offdiag_redundancy.mean_abs_corr |    0.3766 |     0.333 | 1.131  |   0.04354 |
| rankme                           |   91.52   |    82.12  | 1.114  |   9.402   |
| effective_rank                   |    6.936  |     9.318 | 0.7444 |  -2.382   |
| alpha                            |    1.977  |     1.911 | 1.035  |   0.0664  |
| epps_pulley                      | 3005      |  2331     | 1.289  | 674.3     |
| kurt_topeig.worst                |    1.512  |     1.546 | 0.9777 |  -0.03441 |



**student.h.cls vs student.z.dino.tap1**


| metric                           |   value_h |   value_z |    tau |      delta |
|:---------------------------------|----------:|----------:|-------:|-----------:|
| uniformity                       |   -1.884  |   -1.824  | 1.033  |   -0.05987 |
| variance_floor.hinge             |    0.2294 |    0.7438 | 0.3083 |   -0.5145  |
| offdiag_redundancy.mean_abs_corr |    0.3766 |    0.3325 | 1.133  |    0.04411 |
| rankme                           |   91.52   |  303.7    | 0.3014 | -212.2     |
| effective_rank                   |    6.936  |    9.298  | 0.746  |   -2.362   |
| alpha                            |    1.977  |    1.836  | 1.077  |    0.1414  |
| epps_pulley                      | 3005      | 2530      | 1.188  |  475.5     |
| kurt_topeig.worst                |    1.512  |    1.523  | 0.9924 |   -0.01155 |



**student.h.cls vs student.z.dino.tap2**


| metric                           |   value_h |   value_z |    tau |      delta |
|:---------------------------------|----------:|----------:|-------:|-----------:|
| uniformity                       |   -1.884  |   -1.803  | 1.045  |   -0.08149 |
| variance_floor.hinge             |    0.2294 |    0.9212 | 0.249  |   -0.6918  |
| offdiag_redundancy.mean_abs_corr |    0.3766 |    0.3349 | 1.125  |    0.0417  |
| rankme                           |   91.52   |  301.2    | 0.3038 | -209.7     |
| effective_rank                   |    6.936  |    9.986  | 0.6946 |   -3.049   |
| alpha                            |    1.977  |    1.813  | 1.091  |    0.1643  |
| epps_pulley                      | 3005      | 2327      | 1.291  |  678.2     |
| kurt_topeig.worst                |    1.512  |    1.526  | 0.9907 |   -0.01419 |



**student.h.cls.L03 vs student.z.dino.bottleneck**


| metric                           |   value_h |   value_z |    tau |     delta |
|:---------------------------------|----------:|----------:|-------:|----------:|
| uniformity                       |   -1.84   |    -1.799 | 1.023  |  -0.04096 |
| variance_floor.hinge             |    0.2385 |     0.953 | 0.2502 |  -0.7145  |
| offdiag_redundancy.mean_abs_corr |    0.385  |     0.333 | 1.156  |   0.05191 |
| rankme                           |   79.61   |    82.12  | 0.9694 |  -2.511   |
| effective_rank                   |    5.8    |     9.318 | 0.6224 |  -3.518   |
| alpha                            |    2.016  |     1.911 | 1.055  |   0.1052  |
| epps_pulley                      | 3040      |  2331     | 1.304  | 709.6     |
| kurt_topeig.worst                |    2.285  |     1.546 | 1.478  |   0.7388  |



**student.h.cls.L03 vs student.z.dino.tap1**


| metric                           |   value_h |   value_z |    tau |      delta |
|:---------------------------------|----------:|----------:|-------:|-----------:|
| uniformity                       |   -1.84   |   -1.824  | 1.009  |   -0.01585 |
| variance_floor.hinge             |    0.2385 |    0.7438 | 0.3206 |   -0.5054  |
| offdiag_redundancy.mean_abs_corr |    0.385  |    0.3325 | 1.158  |    0.05249 |
| rankme                           |   79.61   |  303.7    | 0.2621 | -224.1     |
| effective_rank                   |    5.8    |    9.298  | 0.6238 |   -3.498   |
| alpha                            |    2.016  |    1.836  | 1.098  |    0.1801  |
| epps_pulley                      | 3040      | 2530      | 1.202  |  510.8     |
| kurt_topeig.worst                |    2.285  |    1.523  | 1.5    |    0.7616  |



**student.h.cls.L03 vs student.z.dino.tap2**


| metric                           |   value_h |   value_z |    tau |      delta |
|:---------------------------------|----------:|----------:|-------:|-----------:|
| uniformity                       |   -1.84   |   -1.803  | 1.021  |   -0.03746 |
| variance_floor.hinge             |    0.2385 |    0.9212 | 0.2589 |   -0.6827  |
| offdiag_redundancy.mean_abs_corr |    0.385  |    0.3349 | 1.15   |    0.05007 |
| rankme                           |   79.61   |  301.2    | 0.2643 | -221.6     |
| effective_rank                   |    5.8    |    9.986  | 0.5808 |   -4.186   |
| alpha                            |    2.016  |    1.813  | 1.112  |    0.203   |
| epps_pulley                      | 3040      | 2327      | 1.307  |  713.5     |
| kurt_topeig.worst                |    2.285  |    1.526  | 1.497  |    0.759   |



**student.h.cls.L06 vs student.z.dino.bottleneck**


| metric                           |   value_h |   value_z |    tau |      delta |
|:---------------------------------|----------:|----------:|-------:|-----------:|
| uniformity                       |   -1.853  |    -1.799 | 1.03   |   -0.05429 |
| variance_floor.hinge             |    0.2413 |     0.953 | 0.2533 |   -0.7116  |
| offdiag_redundancy.mean_abs_corr |    0.3874 |     0.333 | 1.163  |    0.05434 |
| rankme                           |   84.84   |    82.12  | 1.033  |    2.722   |
| effective_rank                   |    6.122  |     9.318 | 0.657  |   -3.196   |
| alpha                            |    1.994  |     1.911 | 1.044  |    0.0833  |
| epps_pulley                      | 3658      |  2331     | 1.569  | 1327       |
| kurt_topeig.worst                |    3      |     1.546 | 1.94   |    1.453   |



**student.h.cls.L06 vs student.z.dino.tap1**


| metric                           |   value_h |   value_z |    tau |      delta |
|:---------------------------------|----------:|----------:|-------:|-----------:|
| uniformity                       |   -1.853  |   -1.824  | 1.016  |   -0.02917 |
| variance_floor.hinge             |    0.2413 |    0.7438 | 0.3245 |   -0.5025  |
| offdiag_redundancy.mean_abs_corr |    0.3874 |    0.3325 | 1.165  |    0.05492 |
| rankme                           |   84.84   |  303.7    | 0.2794 | -218.8     |
| effective_rank                   |    6.122  |    9.298  | 0.6585 |   -3.175   |
| alpha                            |    1.994  |    1.836  | 1.086  |    0.1583  |
| epps_pulley                      | 3658      | 2530      | 1.446  | 1128       |
| kurt_topeig.worst                |    3      |    1.523  | 1.969  |    1.476   |



**student.h.cls.L06 vs student.z.dino.tap2**


| metric                           |   value_h |   value_z |    tau |      delta |
|:---------------------------------|----------:|----------:|-------:|-----------:|
| uniformity                       |   -1.853  |   -1.803  | 1.028  |   -0.05079 |
| variance_floor.hinge             |    0.2413 |    0.9212 | 0.262  |   -0.6798  |
| offdiag_redundancy.mean_abs_corr |    0.3874 |    0.3349 | 1.157  |    0.0525  |
| rankme                           |   84.84   |  301.2    | 0.2817 | -216.4     |
| effective_rank                   |    6.122  |    9.986  | 0.6131 |   -3.863   |
| alpha                            |    1.994  |    1.813  | 1.1    |    0.1812  |
| epps_pulley                      | 3658      | 2327      | 1.572  | 1331       |
| kurt_topeig.worst                |    3      |    1.526  | 1.966  |    1.474   |



**student.h.cls.L09 vs student.z.dino.bottleneck**


| metric                           |   value_h |   value_z |    tau |     delta |
|:---------------------------------|----------:|----------:|-------:|----------:|
| uniformity                       |   -1.909  |    -1.799 | 1.061  |  -0.1098  |
| variance_floor.hinge             |    0.2281 |     0.953 | 0.2393 |  -0.7249  |
| offdiag_redundancy.mean_abs_corr |    0.3729 |     0.333 | 1.12   |   0.03985 |
| rankme                           |   89.5    |    82.12  | 1.09   |   7.379   |
| effective_rank                   |    6.856  |     9.318 | 0.7358 |  -2.462   |
| alpha                            |    1.986  |     1.911 | 1.04   |   0.07549 |
| epps_pulley                      | 2864      |  2331     | 1.229  | 533.4     |
| kurt_topeig.worst                |    2.54   |     1.546 | 1.643  |   0.9941  |



**student.h.cls.L09 vs student.z.dino.tap1**


| metric                           |   value_h |   value_z |    tau |      delta |
|:---------------------------------|----------:|----------:|-------:|-----------:|
| uniformity                       |   -1.909  |   -1.824  | 1.046  |   -0.0847  |
| variance_floor.hinge             |    0.2281 |    0.7438 | 0.3066 |   -0.5158  |
| offdiag_redundancy.mean_abs_corr |    0.3729 |    0.3325 | 1.122  |    0.04043 |
| rankme                           |   89.5    |  303.7    | 0.2947 | -214.2     |
| effective_rank                   |    6.856  |    9.298  | 0.7374 |   -2.441   |
| alpha                            |    1.986  |    1.836  | 1.082  |    0.1504  |
| epps_pulley                      | 2864      | 2530      | 1.132  |  334.6     |
| kurt_topeig.worst                |    2.54   |    1.523  | 1.668  |    1.017   |



**student.h.cls.L09 vs student.z.dino.tap2**


| metric                           |   value_h |   value_z |    tau |      delta |
|:---------------------------------|----------:|----------:|-------:|-----------:|
| uniformity                       |   -1.909  |   -1.803  | 1.059  |   -0.1063  |
| variance_floor.hinge             |    0.2281 |    0.9212 | 0.2476 |   -0.6931  |
| offdiag_redundancy.mean_abs_corr |    0.3729 |    0.3349 | 1.114  |    0.03801 |
| rankme                           |   89.5    |  301.2    | 0.2971 | -211.7     |
| effective_rank                   |    6.856  |    9.986  | 0.6866 |   -3.129   |
| alpha                            |    1.986  |    1.813  | 1.096  |    0.1734  |
| epps_pulley                      | 2864      | 2327      | 1.231  |  537.3     |
| kurt_topeig.worst                |    2.54   |    1.526  | 1.665  |    1.014   |



**student.h.cls.L12 vs student.z.dino.bottleneck**


| metric                           |   value_h |   value_z |    tau |     delta |
|:---------------------------------|----------:|----------:|-------:|----------:|
| uniformity                       |   -1.884  |    -1.799 | 1.047  |  -0.08499 |
| variance_floor.hinge             |    0.2294 |     0.953 | 0.2407 |  -0.7236  |
| offdiag_redundancy.mean_abs_corr |    0.3766 |     0.333 | 1.131  |   0.04354 |
| rankme                           |   91.52   |    82.12  | 1.114  |   9.402   |
| effective_rank                   |    6.936  |     9.318 | 0.7444 |  -2.382   |
| alpha                            |    1.977  |     1.911 | 1.035  |   0.0664  |
| epps_pulley                      | 3005      |  2331     | 1.289  | 674.3     |
| kurt_topeig.worst                |    1.512  |     1.546 | 0.9777 |  -0.03441 |



**student.h.cls.L12 vs student.z.dino.tap1**


| metric                           |   value_h |   value_z |    tau |      delta |
|:---------------------------------|----------:|----------:|-------:|-----------:|
| uniformity                       |   -1.884  |   -1.824  | 1.033  |   -0.05987 |
| variance_floor.hinge             |    0.2294 |    0.7438 | 0.3083 |   -0.5145  |
| offdiag_redundancy.mean_abs_corr |    0.3766 |    0.3325 | 1.133  |    0.04411 |
| rankme                           |   91.52   |  303.7    | 0.3014 | -212.2     |
| effective_rank                   |    6.936  |    9.298  | 0.746  |   -2.362   |
| alpha                            |    1.977  |    1.836  | 1.077  |    0.1414  |
| epps_pulley                      | 3005      | 2530      | 1.188  |  475.5     |
| kurt_topeig.worst                |    1.512  |    1.523  | 0.9924 |   -0.01155 |



**student.h.cls.L12 vs student.z.dino.tap2**


| metric                           |   value_h |   value_z |    tau |      delta |
|:---------------------------------|----------:|----------:|-------:|-----------:|
| uniformity                       |   -1.884  |   -1.803  | 1.045  |   -0.08149 |
| variance_floor.hinge             |    0.2294 |    0.9212 | 0.249  |   -0.6918  |
| offdiag_redundancy.mean_abs_corr |    0.3766 |    0.3349 | 1.125  |    0.0417  |
| rankme                           |   91.52   |  301.2    | 0.3038 | -209.7     |
| effective_rank                   |    6.936  |    9.986  | 0.6946 |   -3.049   |
| alpha                            |    1.977  |    1.813  | 1.091  |    0.1643  |
| epps_pulley                      | 3005      | 2327      | 1.291  |  678.2     |
| kurt_topeig.worst                |    1.512  |    1.526  | 0.9907 |   -0.01419 |



**student.h.gap vs student.z.dino.bottleneck**


| metric                           |   value_h |   value_z |    tau |     delta |
|:---------------------------------|----------:|----------:|-------:|----------:|
| uniformity                       |   -1.678  |    -1.799 | 0.9326 |    0.1212 |
| variance_floor.hinge             |    0.5105 |     0.953 | 0.5357 |   -0.4424 |
| offdiag_redundancy.mean_abs_corr |    0.4748 |     0.333 | 1.425  |    0.1417 |
| rankme                           |   53.05   |    82.12  | 0.646  |  -29.07   |
| effective_rank                   |    3.532  |     9.318 | 0.3791 |   -5.786  |
| alpha                            |    2.142  |     1.911 | 1.121  |    0.2311 |
| epps_pulley                      | 3408      |  2331     | 1.462  | 1077      |
| kurt_topeig.worst                |    5.934  |     1.546 | 3.838  |    4.388  |



**student.h.gap vs student.z.dino.tap1**


| metric                           |   value_h |   value_z |    tau |     delta |
|:---------------------------------|----------:|----------:|-------:|----------:|
| uniformity                       |   -1.678  |   -1.824  | 0.9198 |    0.1463 |
| variance_floor.hinge             |    0.5105 |    0.7438 | 0.6864 |   -0.2333 |
| offdiag_redundancy.mean_abs_corr |    0.4748 |    0.3325 | 1.428  |    0.1423 |
| rankme                           |   53.05   |  303.7    | 0.1747 | -250.6    |
| effective_rank                   |    3.532  |    9.298  | 0.3799 |   -5.765  |
| alpha                            |    2.142  |    1.836  | 1.167  |    0.3061 |
| epps_pulley                      | 3408      | 2530      | 1.347  |  878.6    |
| kurt_topeig.worst                |    5.934  |    1.523  | 3.896  |    4.411  |



**student.h.gap vs student.z.dino.tap2**


| metric                           |   value_h |   value_z |    tau |     delta |
|:---------------------------------|----------:|----------:|-------:|----------:|
| uniformity                       |   -1.678  |   -1.803  | 0.9308 |    0.1247 |
| variance_floor.hinge             |    0.5105 |    0.9212 | 0.5542 |   -0.4106 |
| offdiag_redundancy.mean_abs_corr |    0.4748 |    0.3349 | 1.418  |    0.1399 |
| rankme                           |   53.05   |  301.2    | 0.1761 | -248.2    |
| effective_rank                   |    3.532  |    9.986  | 0.3537 |   -6.453  |
| alpha                            |    2.142  |    1.813  | 1.181  |    0.329  |
| epps_pulley                      | 3408      | 2327      | 1.465  | 1081      |
| kurt_topeig.worst                |    5.934  |    1.526  | 3.889  |    4.409  |



**student.h.gap.L03 vs student.z.dino.bottleneck**


| metric                           |   value_h |   value_z |    tau |      delta |
|:---------------------------------|----------:|----------:|-------:|-----------:|
| uniformity                       |   -1.764  |    -1.799 | 0.9805 |    0.03517 |
| variance_floor.hinge             |    0.5378 |     0.953 | 0.5643 |   -0.4152  |
| offdiag_redundancy.mean_abs_corr |    0.5248 |     0.333 | 1.576  |    0.1918  |
| rankme                           |   46.36   |    82.12  | 0.5646 |  -35.76    |
| effective_rank                   |    2.83   |     9.318 | 0.3037 |   -6.488   |
| alpha                            |    2.109  |     1.911 | 1.104  |    0.1985  |
| epps_pulley                      | 3817      |  2331     | 1.638  | 1486       |
| kurt_topeig.worst                |    9.224  |     1.546 | 5.966  |    7.678   |



**student.h.gap.L03 vs student.z.dino.tap1**


| metric                           |   value_h |   value_z |    tau |      delta |
|:---------------------------------|----------:|----------:|-------:|-----------:|
| uniformity                       |   -1.764  |   -1.824  | 0.967  |    0.06028 |
| variance_floor.hinge             |    0.5378 |    0.7438 | 0.723  |   -0.2061  |
| offdiag_redundancy.mean_abs_corr |    0.5248 |    0.3325 | 1.578  |    0.1923  |
| rankme                           |   46.36   |  303.7    | 0.1527 | -257.3     |
| effective_rank                   |    2.83   |    9.298  | 0.3044 |   -6.468   |
| alpha                            |    2.109  |    1.836  | 1.149  |    0.2734  |
| epps_pulley                      | 3817      | 2530      | 1.509  | 1287       |
| kurt_topeig.worst                |    9.224  |    1.523  | 6.055  |    7.7     |



**student.h.gap.L03 vs student.z.dino.tap2**


| metric                           |   value_h |   value_z |    tau |      delta |
|:---------------------------------|----------:|----------:|-------:|-----------:|
| uniformity                       |   -1.764  |   -1.803  | 0.9785 |    0.03867 |
| variance_floor.hinge             |    0.5378 |    0.9212 | 0.5838 |   -0.3834  |
| offdiag_redundancy.mean_abs_corr |    0.5248 |    0.3349 | 1.567  |    0.1899  |
| rankme                           |   46.36   |  301.2    | 0.1539 | -254.9     |
| effective_rank                   |    2.83   |    9.986  | 0.2834 |   -7.156   |
| alpha                            |    2.109  |    1.813  | 1.163  |    0.2963  |
| epps_pulley                      | 3817      | 2327      | 1.64   | 1490       |
| kurt_topeig.worst                |    9.224  |    1.526  | 6.045  |    7.698   |



**student.h.gap.L06 vs student.z.dino.bottleneck**


| metric                           |   value_h |   value_z |    tau |      delta |
|:---------------------------------|----------:|----------:|-------:|-----------:|
| uniformity                       |   -1.742  |    -1.799 | 0.9685 |    0.05669 |
| variance_floor.hinge             |    0.5264 |     0.953 | 0.5524 |   -0.4266  |
| offdiag_redundancy.mean_abs_corr |    0.5009 |     0.333 | 1.504  |    0.1678  |
| rankme                           |   50.08   |    82.12  | 0.6099 |  -32.04    |
| effective_rank                   |    3.204  |     9.318 | 0.3438 |   -6.114   |
| alpha                            |    2.123  |     1.911 | 1.111  |    0.2125  |
| epps_pulley                      | 3406      |  2331     | 1.461  | 1075       |
| kurt_topeig.worst                |    9.209  |     1.546 | 5.956  |    7.663   |



**student.h.gap.L06 vs student.z.dino.tap1**


| metric                           |   value_h |   value_z |    tau |      delta |
|:---------------------------------|----------:|----------:|-------:|-----------:|
| uniformity                       |   -1.742  |   -1.824  | 0.9552 |    0.08181 |
| variance_floor.hinge             |    0.5264 |    0.7438 | 0.7077 |   -0.2175  |
| offdiag_redundancy.mean_abs_corr |    0.5009 |    0.3325 | 1.507  |    0.1684  |
| rankme                           |   50.08   |  303.7    | 0.1649 | -253.6     |
| effective_rank                   |    3.204  |    9.298  | 0.3446 |   -6.094   |
| alpha                            |    2.123  |    1.836  | 1.157  |    0.2875  |
| epps_pulley                      | 3406      | 2530      | 1.347  |  876.6     |
| kurt_topeig.worst                |    9.209  |    1.523  | 6.046  |    7.686   |



**student.h.gap.L06 vs student.z.dino.tap2**


| metric                           |   value_h |   value_z |    tau |      delta |
|:---------------------------------|----------:|----------:|-------:|-----------:|
| uniformity                       |   -1.742  |   -1.803  | 0.9666 |    0.06019 |
| variance_floor.hinge             |    0.5264 |    0.9212 | 0.5714 |   -0.3948  |
| offdiag_redundancy.mean_abs_corr |    0.5009 |    0.3349 | 1.496  |    0.166   |
| rankme                           |   50.08   |  301.2    | 0.1663 | -251.1     |
| effective_rank                   |    3.204  |    9.986  | 0.3209 |   -6.782   |
| alpha                            |    2.123  |    1.813  | 1.171  |    0.3104  |
| epps_pulley                      | 3406      | 2327      | 1.464  | 1079       |
| kurt_topeig.worst                |    9.209  |    1.526  | 6.035  |    7.683   |



**student.h.gap.L09 vs student.z.dino.bottleneck**


| metric                           |   value_h |   value_z |    tau |      delta |
|:---------------------------------|----------:|----------:|-------:|-----------:|
| uniformity                       |   -1.715  |    -1.799 | 0.9532 |    0.08421 |
| variance_floor.hinge             |    0.5185 |     0.953 | 0.5441 |   -0.4345  |
| offdiag_redundancy.mean_abs_corr |    0.4901 |     0.333 | 1.472  |    0.1571  |
| rankme                           |   52.47   |    82.12  | 0.639  |  -29.65    |
| effective_rank                   |    3.43   |     9.318 | 0.3681 |   -5.888   |
| alpha                            |    2.129  |     1.911 | 1.114  |    0.2177  |
| epps_pulley                      | 3407      |  2331     | 1.462  | 1077       |
| kurt_topeig.worst                |    6.828  |     1.546 | 4.416  |    5.282   |



**student.h.gap.L09 vs student.z.dino.tap1**


| metric                           |   value_h |   value_z |    tau |     delta |
|:---------------------------------|----------:|----------:|-------:|----------:|
| uniformity                       |   -1.715  |   -1.824  | 0.9401 |    0.1093 |
| variance_floor.hinge             |    0.5185 |    0.7438 | 0.6971 |   -0.2253 |
| offdiag_redundancy.mean_abs_corr |    0.4901 |    0.3325 | 1.474  |    0.1577 |
| rankme                           |   52.47   |  303.7    | 0.1728 | -251.2    |
| effective_rank                   |    3.43   |    9.298  | 0.3689 |   -5.868  |
| alpha                            |    2.129  |    1.836  | 1.159  |    0.2927 |
| epps_pulley                      | 3407      | 2530      | 1.347  |  877.7    |
| kurt_topeig.worst                |    6.828  |    1.523  | 4.482  |    5.304  |



**student.h.gap.L09 vs student.z.dino.tap2**


| metric                           |   value_h |   value_z |    tau |      delta |
|:---------------------------------|----------:|----------:|-------:|-----------:|
| uniformity                       |   -1.715  |   -1.803  | 0.9513 |    0.08771 |
| variance_floor.hinge             |    0.5185 |    0.9212 | 0.5629 |   -0.4026  |
| offdiag_redundancy.mean_abs_corr |    0.4901 |    0.3349 | 1.464  |    0.1552  |
| rankme                           |   52.47   |  301.2    | 0.1742 | -248.8     |
| effective_rank                   |    3.43   |    9.986  | 0.3435 |   -6.556   |
| alpha                            |    2.129  |    1.813  | 1.174  |    0.3156  |
| epps_pulley                      | 3407      | 2327      | 1.464  | 1080       |
| kurt_topeig.worst                |    6.828  |    1.526  | 4.474  |    5.302   |



**student.h.gap.L12 vs student.z.dino.bottleneck**


| metric                           |   value_h |   value_z |    tau |     delta |
|:---------------------------------|----------:|----------:|-------:|----------:|
| uniformity                       |   -1.678  |    -1.799 | 0.9326 |    0.1212 |
| variance_floor.hinge             |    0.5105 |     0.953 | 0.5357 |   -0.4424 |
| offdiag_redundancy.mean_abs_corr |    0.4748 |     0.333 | 1.425  |    0.1417 |
| rankme                           |   53.05   |    82.12  | 0.646  |  -29.07   |
| effective_rank                   |    3.532  |     9.318 | 0.3791 |   -5.786  |
| alpha                            |    2.142  |     1.911 | 1.121  |    0.2311 |
| epps_pulley                      | 3408      |  2331     | 1.462  | 1077      |
| kurt_topeig.worst                |    5.934  |     1.546 | 3.838  |    4.388  |



**student.h.gap.L12 vs student.z.dino.tap1**


| metric                           |   value_h |   value_z |    tau |     delta |
|:---------------------------------|----------:|----------:|-------:|----------:|
| uniformity                       |   -1.678  |   -1.824  | 0.9198 |    0.1463 |
| variance_floor.hinge             |    0.5105 |    0.7438 | 0.6864 |   -0.2333 |
| offdiag_redundancy.mean_abs_corr |    0.4748 |    0.3325 | 1.428  |    0.1423 |
| rankme                           |   53.05   |  303.7    | 0.1747 | -250.6    |
| effective_rank                   |    3.532  |    9.298  | 0.3799 |   -5.765  |
| alpha                            |    2.142  |    1.836  | 1.167  |    0.3061 |
| epps_pulley                      | 3408      | 2530      | 1.347  |  878.6    |
| kurt_topeig.worst                |    5.934  |    1.523  | 3.896  |    4.411  |



**student.h.gap.L12 vs student.z.dino.tap2**


| metric                           |   value_h |   value_z |    tau |     delta |
|:---------------------------------|----------:|----------:|-------:|----------:|
| uniformity                       |   -1.678  |   -1.803  | 0.9308 |    0.1247 |
| variance_floor.hinge             |    0.5105 |    0.9212 | 0.5542 |   -0.4106 |
| offdiag_redundancy.mean_abs_corr |    0.4748 |    0.3349 | 1.418  |    0.1399 |
| rankme                           |   53.05   |  301.2    | 0.1761 | -248.2    |
| effective_rank                   |    3.532  |    9.986  | 0.3537 |   -6.453  |
| alpha                            |    2.142  |    1.813  | 1.181  |    0.329  |
| epps_pulley                      | 3408      | 2327      | 1.465  | 1081      |
| kurt_topeig.worst                |    5.934  |    1.526  | 3.889  |    4.409  |



### Pair metrics (alignment ↓ = more view-invariant)


| manifest                   | space                     | metric         |   value |   ci_lo |   ci_hi |
|:---------------------------|:--------------------------|:---------------|--------:|--------:|--------:|
| in100.pairs100.v1@audit_v1 | student.h.cls             | alignment      |  1.23   |  1.21   |  1.251  |
| in100.pairs100.v1@audit_v1 | student.h.cls             | cos_invariance |  0.3851 |  0.374  |  0.3952 |
| in100.pairs100.v1@audit_v1 | student.h.gap             | alignment      |  1.254  |  1.232  |  1.277  |
| in100.pairs100.v1@audit_v1 | student.h.gap             | cos_invariance |  0.373  |  0.3618 |  0.3834 |
| in100.pairs100.v1@audit_v1 | student.z.dino.bottleneck | alignment      |  1.005  |  0.9895 |  1.021  |
| in100.pairs100.v1@audit_v1 | student.z.dino.bottleneck | cos_invariance |  0.4976 |  0.49   |  0.505  |
| in100.pairs100.v1@audit_v1 | student.z.dino.tap1       | alignment      |  1.066  |  1.049  |  1.083  |
| in100.pairs100.v1@audit_v1 | student.z.dino.tap1       | cos_invariance |  0.4669 |  0.4585 |  0.4753 |
| in100.pairs100.v1@audit_v1 | student.z.dino.tap2       | alignment      |  1.022  |  1.006  |  1.038  |
| in100.pairs100.v1@audit_v1 | student.z.dino.tap2       | cos_invariance |  0.4892 |  0.4813 |  0.4973 |
| in100.pairs100.v1@own_dino | student.h.cls             | alignment      |  0.8486 |  0.8334 |  0.8669 |
| in100.pairs100.v1@own_dino | student.h.cls             | cos_invariance |  0.5757 |  0.5657 |  0.5832 |
| in100.pairs100.v1@own_dino | student.h.gap             | alignment      |  0.7984 |  0.7822 |  0.8158 |
| in100.pairs100.v1@own_dino | student.h.gap             | cos_invariance |  0.6008 |  0.591  |  0.6087 |
| in100.pairs100.v1@own_dino | student.z.dino.bottleneck | alignment      |  0.7309 |  0.7197 |  0.7453 |
| in100.pairs100.v1@own_dino | student.z.dino.bottleneck | cos_invariance |  0.6346 |  0.627  |  0.6405 |
| in100.pairs100.v1@own_dino | student.z.dino.tap1       | alignment      |  0.7677 |  0.7554 |  0.7832 |
| in100.pairs100.v1@own_dino | student.z.dino.tap1       | cos_invariance |  0.6161 |  0.6083 |  0.6226 |
| in100.pairs100.v1@own_dino | student.z.dino.tap2       | alignment      |  0.7398 |  0.7283 |  0.7543 |
| in100.pairs100.v1@own_dino | student.z.dino.tap2       | cos_invariance |  0.6301 |  0.6227 |  0.6362 |



### Cross-space similarity (CKA is contested — read as the triple)


| space                                   | metric              |   value |
|:----------------------------------------|:--------------------|--------:|
| student.h.cls~student.z.dino.bottleneck | cka_linear          |  0.99   |
| student.h.cls~student.z.dino.bottleneck | neighbor_jaccard    |  0.7447 |
| student.h.cls~student.z.dino.bottleneck | procrustes_distance |  0.2    |
| student.h.cls~student.z.dino.bottleneck | knn_label_agreement |  0.6866 |
| student.h.cls~student.z.dino.tap1       | cka_linear          |  0.9972 |
| student.h.cls~student.z.dino.tap1       | neighbor_jaccard    |  0.8743 |
| student.h.cls~student.z.dino.tap1       | procrustes_distance |  0.1334 |
| student.h.cls~student.z.dino.tap1       | knn_label_agreement |  0.8376 |
| student.h.cls~student.z.dino.tap2       | cka_linear          |  0.9965 |
| student.h.cls~student.z.dino.tap2       | neighbor_jaccard    |  0.8471 |
| student.h.cls~student.z.dino.tap2       | procrustes_distance |  0.1568 |
| student.h.cls~student.z.dino.tap2       | knn_label_agreement |  0.8102 |



### Probes


| space                     |   knn_v1_k20 |   knn_v1_k200 |   linear_house_v1 |   linear_l2_v1 |   linear_raw_v1 |
|:--------------------------|-------------:|--------------:|------------------:|---------------:|----------------:|
| student.h.cls             |       0.0540 |        0.0622 |            0.0926 |         0.0638 |          0.0884 |
| student.h.cls.L03         |       0.0550 |        0.0602 |            0.0870 |         0.0624 |          0.0856 |
| student.h.cls.L06         |       0.0578 |        0.0638 |            0.0890 |         0.0636 |          0.0854 |
| student.h.cls.L09         |       0.0578 |        0.0610 |            0.0916 |         0.0636 |          0.0876 |
| student.h.cls.L12         |       0.0540 |        0.0622 |            0.0926 |         0.0638 |          0.0894 |
| student.h.gap             |       0.0640 |        0.0712 |            0.1198 |         0.0742 |          0.1102 |
| student.h.gap.L03         |       0.0578 |        0.0620 |            0.1060 |         0.0664 |          0.0918 |
| student.h.gap.L06         |       0.0638 |        0.0674 |            0.1166 |         0.0722 |          0.1012 |
| student.h.gap.L09         |       0.0636 |        0.0690 |            0.1202 |         0.0734 |          0.1050 |
| student.h.gap.L12         |       0.0640 |        0.0712 |            0.1198 |         0.0742 |          0.1102 |
| student.z.dino.bottleneck |       0.0592 |        0.0652 |            0.0834 |         0.0618 |          0.0610 |
| student.z.dino.tap1       |       0.0590 |        0.0632 |            0.0990 |         0.0712 |          0.0976 |
| student.z.dino.tap2       |       0.0592 |        0.0648 |            0.0990 |         0.0708 |          0.0860 |



## in100.dino-ctrl.ep100.parity30k

### Battery (variant raw|full)

| model.space                                                 |   rankme |   effective_rank |   alpha |   epps_pulley |   kurt_topeig.worst |   uniformity |   offdiag_redundancy.mean_abs_corr |   variance_floor.hinge |
|:------------------------------------------------------------|---------:|-----------------:|--------:|--------------:|--------------------:|-------------:|-----------------------------------:|-----------------------:|
| in100.dino-ctrl.ep100.parity30k · student.h.cls             |    241   |           115.1  |  1.242  |        14.5   |              0.8255 |       -3.42  |                            0.08226 |               0.007297 |
| in100.dino-ctrl.ep100.parity30k · student.h.gap             |    218.4 |            82.87 |  1.388  |        22.88  |              2.787  |       -2.49  |                            0.1024  |               0.5548   |
| in100.dino-ctrl.ep100.parity30k · student.z.dino.bottleneck |    158.7 |            91.38 |  1.684  |        59.99  |             36.09   |       -3.615 |                            0.08543 |               0.9047   |
| in100.dino-ctrl.ep100.parity30k · student.z.dino.tap1       |   1352   |           349.1  |  0.9589 |         8.268 |              0.828  |       -3.342 |                            0.05591 |               0.5954   |
| in100.dino-ctrl.ep100.parity30k · student.z.dino.tap2       |   1403   |           577.2  |  0.7729 |         7.04  |             47.58   |       -1.464 |                            0.03169 |               0.9154   |
| in100.dino-ctrl.ep100.parity30k · teacher.h.cls             |    236.9 |           111.7  |  1.273  |        15.19  |              0.8376 |       -3.419 |                            0.08346 |               0.008181 |
| in100.dino-ctrl.ep100.parity30k · teacher.h.gap             |    217   |            80.16 |  1.407  |        23.47  |              3.672  |       -2.541 |                            0.1044  |               0.5557   |
| in100.dino-ctrl.ep100.parity30k · teacher.z.dino.bottleneck |    158.3 |            91.08 |  1.69   |        55.84  |             28.11   |       -3.614 |                            0.08561 |               0.89     |
| in100.dino-ctrl.ep100.parity30k · teacher.z.dino.tap1       |   1350   |           343    |  0.9639 |         8.472 |              0.829  |       -3.393 |                            0.05643 |               0.5865   |
| in100.dino-ctrl.ep100.parity30k · teacher.z.dino.tap2       |   1430   |           587    |  0.7638 |         6.759 |             55.19   |       -1.73  |                            0.03106 |               0.9069   |



### Transfer ratios τ = value(h) / value(z)


**student.h.cls vs student.z.dino.bottleneck**


| metric                           |    value_h |   value_z |      tau |      delta |
|:---------------------------------|-----------:|----------:|---------:|-----------:|
| uniformity                       |  -3.42     |  -3.615   | 0.9461   |   0.1949   |
| variance_floor.hinge             |   0.007297 |   0.9047  | 0.008065 |  -0.8974   |
| offdiag_redundancy.mean_abs_corr |   0.08226  |   0.08543 | 0.963    |  -0.003161 |
| rankme                           | 241        | 158.7     | 1.519    |  82.34     |
| effective_rank                   | 115.1      |  91.38    | 1.259    |  23.68     |
| alpha                            |   1.242    |   1.684   | 0.7377   |  -0.4417   |
| epps_pulley                      |  14.5      |  59.99    | 0.2418   | -45.48     |
| kurt_topeig.worst                |   0.8255   |  36.09    | 0.02287  | -35.27     |



**student.h.cls vs student.z.dino.tap1**


| metric                           |    value_h |    value_z |     tau |        delta |
|:---------------------------------|-----------:|-----------:|--------:|-------------:|
| uniformity                       |  -3.42     |   -3.342   | 1.023   |    -0.07853  |
| variance_floor.hinge             |   0.007297 |    0.5954  | 0.01226 |    -0.5881   |
| offdiag_redundancy.mean_abs_corr |   0.08226  |    0.05591 | 1.471   |     0.02635  |
| rankme                           | 241        | 1352       | 0.1782  | -1111        |
| effective_rank                   | 115.1      |  349.1     | 0.3295  |  -234.1      |
| alpha                            |   1.242    |    0.9589  | 1.295   |     0.2832   |
| epps_pulley                      |  14.5      |    8.268   | 1.754   |     6.236    |
| kurt_topeig.worst                |   0.8255   |    0.828   | 0.9969  |    -0.002531 |



**student.h.cls vs student.z.dino.tap2**


| metric                           |    value_h |    value_z |      tau |       delta |
|:---------------------------------|-----------:|-----------:|---------:|------------:|
| uniformity                       |  -3.42     |   -1.464   | 2.337    |    -1.957   |
| variance_floor.hinge             |   0.007297 |    0.9154  | 0.007971 |    -0.9081  |
| offdiag_redundancy.mean_abs_corr |   0.08226  |    0.03169 | 2.596    |     0.05058 |
| rankme                           | 241        | 1403       | 0.1718   | -1162       |
| effective_rank                   | 115.1      |  577.2     | 0.1993   |  -462.1     |
| alpha                            |   1.242    |    0.7729  | 1.607    |     0.4691  |
| epps_pulley                      |  14.5      |    7.04    | 2.06     |     7.464   |
| kurt_topeig.worst                |   0.8255   |   47.58    | 0.01735  |   -46.76    |



**student.h.gap vs student.z.dino.bottleneck**


| metric                           |   value_h |   value_z |     tau |     delta |
|:---------------------------------|----------:|----------:|--------:|----------:|
| uniformity                       |   -2.49   |  -3.615   | 0.6888  |   1.125   |
| variance_floor.hinge             |    0.5548 |   0.9047  | 0.6133  |  -0.3499  |
| offdiag_redundancy.mean_abs_corr |    0.1024 |   0.08543 | 1.198   |   0.01696 |
| rankme                           |  218.4    | 158.7     | 1.376   |  59.72    |
| effective_rank                   |   82.87   |  91.38    | 0.9068  |  -8.515   |
| alpha                            |    1.388  |   1.684   | 0.8246  |  -0.2953  |
| epps_pulley                      |   22.88   |  59.99    | 0.3813  | -37.11    |
| kurt_topeig.worst                |    2.787  |  36.09    | 0.07722 | -33.3     |



**student.h.gap vs student.z.dino.tap1**


| metric                           |   value_h |    value_z |    tau |       delta |
|:---------------------------------|----------:|-----------:|-------:|------------:|
| uniformity                       |   -2.49   |   -3.342   | 0.7452 |     0.8516  |
| variance_floor.hinge             |    0.5548 |    0.5954  | 0.9318 |    -0.04058 |
| offdiag_redundancy.mean_abs_corr |    0.1024 |    0.05591 | 1.831  |     0.04647 |
| rankme                           |  218.4    | 1352       | 0.1615 | -1134       |
| effective_rank                   |   82.87   |  349.1     | 0.2373 |  -266.3     |
| alpha                            |    1.388  |    0.9589  | 1.448  |     0.4295  |
| epps_pulley                      |   22.88   |    8.268   | 2.767  |    14.61    |
| kurt_topeig.worst                |    2.787  |    0.828   | 3.366  |     1.959   |



**student.h.gap vs student.z.dino.tap2**


| metric                           |   value_h |    value_z |     tau |       delta |
|:---------------------------------|----------:|-----------:|--------:|------------:|
| uniformity                       |   -2.49   |   -1.464   | 1.701   |    -1.027   |
| variance_floor.hinge             |    0.5548 |    0.9154  | 0.6061  |    -0.3606  |
| offdiag_redundancy.mean_abs_corr |    0.1024 |    0.03169 | 3.231   |     0.07069 |
| rankme                           |  218.4    | 1403       | 0.1557  | -1184       |
| effective_rank                   |   82.87   |  577.2     | 0.1436  |  -494.3     |
| alpha                            |    1.388  |    0.7729  | 1.796   |     0.6155  |
| epps_pulley                      |   22.88   |    7.04    | 3.249   |    15.83    |
| kurt_topeig.worst                |    2.787  |   47.58    | 0.05857 |   -44.8     |



**teacher.h.cls vs teacher.z.dino.bottleneck**


| metric                           |    value_h |   value_z |      tau |      delta |
|:---------------------------------|-----------:|----------:|---------:|-----------:|
| uniformity                       |  -3.419    |  -3.614   | 0.9462   |   0.1946   |
| variance_floor.hinge             |   0.008181 |   0.89    | 0.009193 |  -0.8818   |
| offdiag_redundancy.mean_abs_corr |   0.08346  |   0.08561 | 0.9748   |  -0.002156 |
| rankme                           | 236.9      | 158.3     | 1.496    |  78.6      |
| effective_rank                   | 111.7      |  91.08    | 1.227    |  20.64     |
| alpha                            |   1.273    |   1.69    | 0.7529   |  -0.4176   |
| epps_pulley                      |  15.19     |  55.84    | 0.2719   | -40.66     |
| kurt_topeig.worst                |   0.8376   |  28.11    | 0.02979  | -27.27     |



**teacher.h.cls vs teacher.z.dino.tap1**


| metric                           |    value_h |    value_z |     tau |       delta |
|:---------------------------------|-----------:|-----------:|--------:|------------:|
| uniformity                       |  -3.419    |   -3.393   | 1.008   |    -0.02668 |
| variance_floor.hinge             |   0.008181 |    0.5865  | 0.01395 |    -0.5783  |
| offdiag_redundancy.mean_abs_corr |   0.08346  |    0.05643 | 1.479   |     0.02703 |
| rankme                           | 236.9      | 1350       | 0.1755  | -1113       |
| effective_rank                   | 111.7      |  343       | 0.3257  |  -231.3     |
| alpha                            |   1.273    |    0.9639  | 1.32    |     0.3088  |
| epps_pulley                      |  15.19     |    8.472   | 1.792   |     6.714   |
| kurt_topeig.worst                |   0.8376   |    0.829   | 1.01    |     0.00858 |



**teacher.h.cls vs teacher.z.dino.tap2**


| metric                           |    value_h |    value_z |      tau |      delta |
|:---------------------------------|-----------:|-----------:|---------:|-----------:|
| uniformity                       |  -3.419    |   -1.73    | 1.977    |    -1.69   |
| variance_floor.hinge             |   0.008181 |    0.9069  | 0.009021 |    -0.8988 |
| offdiag_redundancy.mean_abs_corr |   0.08346  |    0.03106 | 2.687    |     0.0524 |
| rankme                           | 236.9      | 1430       | 0.1657   | -1193      |
| effective_rank                   | 111.7      |  587       | 0.1903   |  -475.3    |
| alpha                            |   1.273    |    0.7638  | 1.666    |     0.509  |
| epps_pulley                      |  15.19     |    6.759   | 2.247    |     8.427  |
| kurt_topeig.worst                |   0.8376   |   55.19    | 0.01518  |   -54.35   |



**teacher.h.gap vs teacher.z.dino.bottleneck**


| metric                           |   value_h |   value_z |    tau |     delta |
|:---------------------------------|----------:|----------:|-------:|----------:|
| uniformity                       |   -2.541  |  -3.614   | 0.7032 |   1.073   |
| variance_floor.hinge             |    0.5557 |   0.89    | 0.6244 |  -0.3343  |
| offdiag_redundancy.mean_abs_corr |    0.1044 |   0.08561 | 1.219  |   0.01876 |
| rankme                           |  217      | 158.3     | 1.371  |  58.67    |
| effective_rank                   |   80.16   |  91.08    | 0.8801 | -10.92    |
| alpha                            |    1.407  |   1.69    | 0.8323 |  -0.2835  |
| epps_pulley                      |   23.47   |  55.84    | 0.4202 | -32.38    |
| kurt_topeig.worst                |    3.672  |  28.11    | 0.1306 | -24.44    |



**teacher.h.gap vs teacher.z.dino.tap1**


| metric                           |   value_h |    value_z |    tau |       delta |
|:---------------------------------|----------:|-----------:|-------:|------------:|
| uniformity                       |   -2.541  |   -3.393   | 0.749  |     0.8515  |
| variance_floor.hinge             |    0.5557 |    0.5865  | 0.9475 |    -0.0308  |
| offdiag_redundancy.mean_abs_corr |    0.1044 |    0.05643 | 1.85   |     0.04794 |
| rankme                           |  217      | 1350       | 0.1608 | -1133       |
| effective_rank                   |   80.16   |  343       | 0.2337 |  -262.9     |
| alpha                            |    1.407  |    0.9639  | 1.46   |     0.443   |
| epps_pulley                      |   23.47   |    8.472   | 2.77   |    14.99    |
| kurt_topeig.worst                |    3.672  |    0.829   | 4.43   |     2.843   |



**teacher.h.gap vs teacher.z.dino.tap2**


| metric                           |   value_h |    value_z |     tau |       delta |
|:---------------------------------|----------:|-----------:|--------:|------------:|
| uniformity                       |   -2.541  |   -1.73    | 1.469   |    -0.8114  |
| variance_floor.hinge             |    0.5557 |    0.9069  | 0.6127  |    -0.3513  |
| offdiag_redundancy.mean_abs_corr |    0.1044 |    0.03106 | 3.36    |     0.07331 |
| rankme                           |  217      | 1430       | 0.1517  | -1213       |
| effective_rank                   |   80.16   |  587       | 0.1366  |  -506.8     |
| alpha                            |    1.407  |    0.7638  | 1.842   |     0.6431  |
| epps_pulley                      |   23.47   |    6.759   | 3.472   |    16.71    |
| kurt_topeig.worst                |    3.672  |   55.19    | 0.06654 |   -51.51    |



### Cross-space similarity (CKA is contested — read as the triple)


| space                                   | metric              |   value |
|:----------------------------------------|:--------------------|--------:|
| teacher.h.cls~teacher.z.dino.bottleneck | cka_linear          |  0.4413 |
| teacher.h.cls~teacher.z.dino.bottleneck | neighbor_jaccard    |  0.2091 |
| teacher.h.cls~teacher.z.dino.bottleneck | procrustes_distance |  0.9279 |
| teacher.h.cls~teacher.z.dino.bottleneck | knn_label_agreement |  0.5592 |
| teacher.h.cls~teacher.z.dino.tap1       | cka_linear          |  0.9172 |
| teacher.h.cls~teacher.z.dino.tap1       | neighbor_jaccard    |  0.5578 |
| teacher.h.cls~teacher.z.dino.tap1       | procrustes_distance |  0.3611 |
| teacher.h.cls~teacher.z.dino.tap1       | knn_label_agreement |  0.7374 |
| teacher.h.cls~teacher.z.dino.tap2       | cka_linear          |  0.5776 |
| teacher.h.cls~teacher.z.dino.tap2       | neighbor_jaccard    |  0.2812 |
| teacher.h.cls~teacher.z.dino.tap2       | procrustes_distance |  0.874  |
| teacher.h.cls~teacher.z.dino.tap2       | knn_label_agreement |  0.5836 |



### Probes


| space                     |   knn_v1_k20 |   knn_v1_k200 |   linear_house_v1 |   linear_l2_v1 |   linear_raw_v1 |
|:--------------------------|-------------:|--------------:|------------------:|---------------:|----------------:|
| student.h.cls             |       0.6332 |        0.6160 |            0.6848 |         0.6528 |          0.6882 |
| student.h.gap             |       0.4992 |        0.4684 |            0.6458 |         0.5380 |          0.6422 |
| student.z.dino.bottleneck |       0.5882 |        0.5748 |            0.5996 |         0.5856 |          0.5900 |
| student.z.dino.tap1       |       0.6236 |        0.6056 |            0.6796 |         0.6714 |          0.6936 |
| student.z.dino.tap2       |       0.5898 |        0.5782 |            0.6512 |         0.6084 |          0.6606 |
| teacher.h.cls             |       0.6360 |        0.6142 |            0.6854 |         0.6540 |          0.6872 |
| teacher.h.gap             |       0.4968 |        0.4658 |            0.6442 |         0.5396 |          0.6394 |
| teacher.z.dino.bottleneck |       0.5886 |        0.5752 |            0.6024 |         0.5896 |          0.5940 |
| teacher.z.dino.tap1       |       0.6222 |        0.6034 |            0.6790 |         0.6688 |          0.6926 |
| teacher.z.dino.tap2       |       0.5982 |        0.5852 |            0.6530 |         0.6150 |          0.6626 |



## Parity block

Parity (linear_house_v1 + knn_v1 + RankMe/PR, exact — see docs/HISTORY.md 2026-07-02): 16/16 within ±0.06 pt. Headline probes per D-006v2 (linear_raw_v1 + knn_v1) added to all probe tables 2026-07-02.


## AGREED TAKEAWAY

*(empty — filled only after discussion; see CLAUDE.md)*
