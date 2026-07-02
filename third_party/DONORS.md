# DONORS — read-only reviewed reference clones

Donor code is NEVER imported. Ports go through review, get a PORT_NOTES section in the method
dossier (donor file, commit, review findings, deviations), and land in `sslgap/` in our style
(DECISIONS L-001).

| donor | path | pinned commit | date | role |
|---|---|---|---|---|
| solo-learn (vturrisi) | `third_party/solo-learn/` | `9187ea39c2c3f43c455ab06664d2b019a9802954` | 2026-04-22 | tuned IN-100 recipes + method impls for SimCLR/BYOL/VICReg/MAE ports; published RN18-IN-100 numbers = port-validation ground truth (M1.5) |
| lejepa (official minimal) | `../lejepa` (sibling repo, not vendored) | working tree | — | LeJEPA recipe ground truth; user's trained Imagenette ckpts = M0 inputs |
| ssl_explore (in-house) | `../ssl_explore` (sibling repo) | working tree | — | harvested: geometry/knn/meters/sigreg (ported into sslgap/metrics, sslgap/probes), DINO IN-100 control recipe + ckpts, SLURM conventions |

Ports so far (kickstart session): `sslgap/metrics/spectra.py` ← sslx/geometry.py + meters.rankme;
`sslgap/metrics/isotropy.py` ← sslx/sigreg.py; `sslgap/probes/knn.py` ← sslx/knn.py (verbatim +
self-test); `sslgap/probes/linear.py` ← sslx/meters.offline_probe (house variant verbatim);
`sslgap/models/heads.py:DINOHead` ← sslx/dinov2.py (verbatim); `sslgap/data.py` transforms ←
sslx/data.py + sslx/dinov2.py `_dino_view` (verbatim).
