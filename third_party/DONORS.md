# DONORS — read-only reviewed reference clones

Donor code is NEVER imported. Ports go through review, get a PORT_NOTES section in the method
dossier (donor file, commit, review findings, deviations), and land in `sslgap/` in our style
(DECISIONS L-001).

| donor | path | pinned commit | date | role |
|---|---|---|---|---|
| solo-learn (vturrisi) | `third_party/solo-learn/` | `9187ea39c2c3f43c455ab06664d2b019a9802954` | 2026-04-22 | tuned IN-100 recipes + method impls for SimCLR/BYOL/VICReg/MAE ports; published RN18-IN-100 numbers = port-validation ground truth (M1.5) |
| latentis (Flegyas) | `third_party/latentis/` | `800699f9fd5a98880adac40590e075aaffb87ab9` | 2026 | relative-representations reference implementation (D-009): cosine relrep semantics verified (no centering by default; optional Centering/StandardScaling abs_transforms) |
| lejepa (official minimal) | `../lejepa` (sibling repo, not vendored) | working tree | — | LeJEPA recipe ground truth; user's trained Imagenette ckpts = M0 inputs |
| ssl_explore (in-house) | `../ssl_explore` (sibling repo) | working tree | — | harvested: geometry/knn/meters/sigreg (ported into sslgap/metrics, sslgap/probes), DINO IN-100 control recipe + ckpts, SLURM conventions |
| visreg (HaiyuWu) | `third_party/visreg/` | `47b1cf4` | 2026-08-10 | the lineage's only published full-multicrop in1k ViT trainer (E27 §(j.4)); faithful-repro runnable copy `~/visreg_repro` (own venv on THEIR pins, THREE declared patches, all data-access/observability: train + eval HF-hub→imagefolder, loader workers + epoch prints — PATCHES.md). PORT_NOTES: house-VISReg trainer = E29 card §(e) (canonical backbone/embed/projector roles, native `_asm_visreg`); ckpt lift = `sslgap/ckpt/adapters.py:from_visreg` (C9); Table-5 transfer-LP evaluator = `experiments/transfer_visreg.py` docstring + E30 card §(c)/(d) (port-validation gate on their printed DINO-B row) |
| lightly (lightly-ai) | `third_party/lightly/` | `f444cf36392621c0b75a9f2a6d90bf7f56caa4ea` | 2026-08-10 | the LeJEPA ViT-S/16 64.0-row benchmark (D-094 zero-diff reproduction); runnable copy `~/lightly_repro` (own venv, ZERO patches, REPRO.md); also the importable ORACLE for exact-match selftests of house replications |
| LeVJEPA (MLO-lab, formerly galilai-group) | `third_party/levjepa/` | `3ea0dda` | 2026-09-02 | LeJEPA on video (arXiv 2608.27395): the video side project's trainer, Lance loader, ViT with token dropping + block-causal attention. Public release = Walking Tours pipeline only (no K710 list, no Kinetics/SSv2 code, no attentive-probe eval). Working fork **`video/levjepa/`** (tracked; "we update theirs", Berker 2026-09-02): the house v6 recipe behind `loss.type=sslgap` + `augmentation.views=6`. PORT_NOTES: `docs/methods/levjepa.md` |

Ports so far (kickstart session): `sslgap/metrics/spectra.py` ← sslx/geometry.py + meters.rankme;
`sslgap/metrics/isotropy.py` ← sslx/sigreg.py; `sslgap/probes/knn.py` ← sslx/knn.py (verbatim +
self-test); `sslgap/probes/linear.py` ← sslx/meters.offline_probe (house variant verbatim);
`sslgap/models/heads.py:DINOHead` ← sslx/dinov2.py (verbatim); `sslgap/data.py` transforms ←
sslx/data.py + sslx/dinov2.py `_dino_view` (verbatim).
