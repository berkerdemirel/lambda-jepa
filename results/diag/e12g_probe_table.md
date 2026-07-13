# G-wave + lejepa calibration pairs — probe table (Berker request 2026-07-13; raw, no takeaway)

Declared spaces: lejepa `student.z.embed` · vicreg `student.h.gap` · dino `teacher.h.cls`.
All linear v2 probes patience-converged (best_ep ≪ epochs_run) EXCEPT the vicreg-pair `l2_v2`
rows grazing the 1000-ep boundary (gv 910/1000, gvc 985/1000) — D-020 censoring flag applies to
that family only. Monitor = online LayerNorm+Linear on detached h during training.

| pair | probe | arm | control | Δ (arm−ctrl) |
|---|---|---|---|---|
| lejepa f2/c1 | monitor (final) | .6758 | .6338 | **+.0420** |
| | linear_raw_v2 | .6578 (4/125) | .6456 (270/391) | **+.0122** |
| | linear_house_v2 | .6576 (1/122) | .6452 (163/284) | **+.0124** |
| | linear_l2_v2 | .6600 (114/235) | .6382 (931/1000) | **+.0218** |
| | knn_v1_k200 | .6056 | .5306 | **+.0750** |
| vicreg gv/gvc | monitor (final) | .5796 | .5670 | **+.0126** |
| | linear_raw_v2 | .5808 (234/355) | .5966 (412/533) | **−.0158** |
| | linear_house_v2 | .5782 (46/167) | .5934 (148/269) | **−.0152** |
| | linear_l2_v2 † | .5784 (910/1000) | .5862 (985/1000) | −.0078 |
| | knn_v1_k200 | .4614 | .4000 | **+.0614** |
| dino gd/gdc | monitor (final) | .7044 | .7068 | −.0024 |
| | linear_raw_v2 | .6852 (9/130) | .6922 (37/158) | −.0070 |
| | linear_house_v2 | .6816 (9/130) | .6866 (16/137) | −.0050 |
| | linear_l2_v2 | .6874 (104/225) | .6906 (312/433) | −.0032 |
| | knn_v1_k200 | .6342 | .5978 | **+.0364** |

(n/m) = best_ep/epochs_run. † boundary-grazing.

Mechanical notes (discussion next session): (1) kNN200 gains on ALL three pairs (+7.5/+6.1/+3.6).
(2) lejepa: no tax anywhere — arm wins every column. (3) dino: small consistent linear tax
(−.2 to −.7 pts) in monitor AND offline. (4) vicreg: the SIGN FLIPS between instruments —
monitor +1.3 vs converged offline −1.6 (raw and LN'd house agree, both converged ⇒ NOT a
convergence artifact, NOT a LayerNorm artifact; candidate causes for discussion: the monitor
trains on AUGMENTED views and co-trains with the run's schedule).
