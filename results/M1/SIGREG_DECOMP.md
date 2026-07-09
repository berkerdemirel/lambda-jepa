# SIGReg honesty cells — three-reference decomposition (numbers only)

> Emitted by experiments/sigreg_decomp.py. EP meter: 256 seeded slices, per-dim
> standardized (NOT whitened) — the SIGReg statistic as a meter. References:
> `isotropic_floor` = N(0,I) sample, same (N,d) → pure sampling noise (the oracle if
> SIGReg's target were exactly met); `covmatched_gauss` = N(mean, full covariance of
> the data) → what a perfect GAUSSIAN with this run's covariance would read; the data
> are never transformed. data − covmatched = shape excess (no linear map fixes it);
> covmatched − floor = covariance/anisotropy part (a linear map could fix it).
> Untrained-net reference (matrix null col): EP 569 (embed) / 405 (proj) for the grid
> run. kurt_topeig.worst is affine-invariant → its two references coincide (~noise).

| ckpt                      | space              |    n |   d |   EP_data |   EP_covmatched_gauss |   EP_isotropic_floor |   kurt_data |   kurt_covmatched_gauss |   kurt_isotropic_floor |
|:--------------------------|:-------------------|-----:|----:|----------:|----------------------:|---------------------:|------------:|------------------------:|-----------------------:|
| 150ep grid (house recipe) | student.z.embed    | 9469 | 512 |   203.031 |               157.731 |                0.596 |       3.624 |                   0.091 |                  0.097 |
| 150ep grid (house recipe) | student.z.proj.out | 9469 |  16 |    54.889 |                 7.470 |                0.509 |       1.516 |                   0.145 |                  0.082 |
| 800ep official ckpt (M0)  | student.z.embed    | 9469 | 512 |    89.436 |                69.416 |                0.564 |       0.991 |                   0.083 |                  0.100 |
| 800ep official ckpt (M0)  | student.z.proj.out | 9469 |  16 |    24.930 |                 1.981 |                0.647 |       3.543 |                   0.134 |                  0.124 |

