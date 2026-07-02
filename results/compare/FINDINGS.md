# Cross-model comparisons (D-009) — numbers + mechanics, no takeaways

Space: `student.h.cls` (384-d), manifest `imagenette.train.v1` (9469 images, same order for all
models). Anchors: A=384 seeded dataset images, same ids for both models; relrep = cosine to anchors
(latentis-canonical: ℓ2+dot, no centering). "direct" = same metrics on the raw feature matrices
(dims match here). See docs/METRICS.md for metric semantics.

| pair | frame | CKA | NN-Jaccard@10 | Procrustes | kNN-label-agree |
|---|---|---|---|---|---|
| self vs self (identity anchor) | both | 1.000 | 1.000 | 0.000 | 1.000 |
| LeJEPA λ=.02 vs InfoNCE (two trained models) | direct | 0.658 | 0.169 | 0.721 | 0.775 |
| | relrep | 0.473 | 0.095 | 0.768 | 0.738 |
| LeJEPA λ=.02 vs random-init (untrained anchor) | direct | 0.195 | 0.013 | 1.157 | 0.316 |
| | relrep | 0.088 | 0.008 | 1.202 | 0.275 |
| LeJEPA λ=.02 vs LeJEPA λ=0 (collapsed anchor) | direct | 0.032 | 0.003 | 1.306 | 0.164 |
| | relrep | 0.025 | 0.002 | 1.325 | 0.148 |

Chance reference for NN-Jaccard@10 at N=9469 ≈ 0.001; label-agreement chance ≈ 0.1 (10 classes).
Missing yardstick (arrives at M1): same method, two seeds — the "how similar is *the same* method
to itself" reference that locates the trained-vs-trained row properly.

Mechanical notes (not conclusions): relrep frame ≤ direct frame on CKA here; since relrep mods out
rotations (verified: rotated copy → CKA 1.0), a rotation-only mismatch would have RAISED relrep
agreement — so the disagreement between these two trained models is not primarily a global
rotation/scale. Caveats: relrep also ℓ2-normalizes (cosine), and A=384 anchor sampling adds noise;
these numbers are toy-scale (Imagenette).
