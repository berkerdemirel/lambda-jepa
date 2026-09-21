# E37 — VISReg's own regularizer at the backbone (the E17 arm that was never run)

**Status: PRE-REGISTERED 2026-09-17, launch authorized by Berker the same day ("can you launch a run where
we do visreg too. we will potentially use either lejepa only or visreg and lejepa in case it works the way we
want. you can use h100."). D-row: D-125 (USER-DIRECTED launch; design and predictions = Fable, flagged for veto).**

## Why

The paper's appendix stub *"Each method's own regularizer at the backbone"* rests on E17 (July): each
method's own non-collapse term added at its declared h, z-loss byte-untouched, one dose each (10 % of the
shipped z-term's weighted pull on the first batch). E17 covered LeJEPA (SIGReg at the 512-d embedding: linear
−1.4 / kNN −4.3), VICReg (var+cov at CLS: +0.6 / +4.4 only at 6× the rule dose), SimCLR, BYOL, DINO — and not
VISReg, which joined the zoo in E29 (August). Berker's aim: an empirical justification that SACReg's
particular regularizer at h is the right one, with LeJEPA and VISReg as the important comparators
("we will potentially use either lejepa only or visreg and lejepa in case it works the way we want").

## Frame (anatomy stated loudly)

- Lane: the house VISReg (`sslgap/methods/visreg.py`, E29): IN-100 (CMC split, `~/data/imagenet100`),
  ViT-S/16 @224, 100 epochs, batch 128, seed 0, V = 4 house views, trunk → 512-d embedding → 3-layer
  projector (128-d), loss = 0.9·VISReg(z) + 0.1·inv(z), house hygiene (lr 1e-3, wd 5e-2, warmup 10,
  eta_min 1e-5, grad_clip 1.0). Control = `in100.visreg.s0` (E29, landed, `.extL` stores + probes exist).
- **Arm `in100.visreg.s0.hpull_visreg`** = the control + `h_reg=visreg`: a second instance of VISReg's own
  loss (center + scale + sliced-quantile-shape, the donor's exact code, its own fresh random projections and
  cached quantile target) applied to the per-view 512-d embedding `[V, B, 512]` — the same declared h as
  E29's `visregf` arm (our term at that embedding, λ = .0123). Term key `h_visreg`, weight `h_lamb`.
  z-side loss byte-identical to the control.
- **Dose = the 6 %-share rule at the pilot's epoch-1 state, exactly as `visregf` was dosed**: pilot
  `hpull_visreg_pilot` (2 epochs, provisional h_lamb .03, `share_log_every=1`) → read the ep1 `[share]` line →
  λ* = (T/(1−T)) · Σ_z(w·g) / g_h with T = .06, Σ_z(w·g) = .9·g_visreg + .1·g_inv, g_h = the unit trunk pull of
  `h_visreg`. Like-for-like pair with `visregf`: same tap, same rule, same state. (E17's rule was 10 % of the
  shipped z-term's weighted pull on the first batch; a re-dose of the E17 arms under this rule is a separate
  decision, HANDOVER §1.)
- Compute: one H100 (`gpu100`, `--constraint=H100`), singleton chain `h100-slotA`, 2 × 8 h links (the E29
  chain took 8 h + 2 h on an L40S); resume via `_last.pt`. Landing chain (extract `.extL` h_layers [3,6,9] + o8,
  twospace, audit, probes) = the E29 pipeline, launched at landing.

## Pre-registered predictions (committed before any number exists)

- **P1 (Berker's expectation, the claim at stake):** the own term at h does NOT improve the frozen probes over
  the control at the paper's readers (linear_raw_v2 / knn_v1_k200 at the CLS and at the embedding): linear
  within ±0.5, kNN ≤ +1.0. Reference: our term at the same tap and rule (`visregf`) reads +1.8 linear / +7.7 kNN
  at the CLS.
- **P2 (mechanism, from E17-T3/T6):** VISReg's regularizer is shape-seeing (sliced-quantile term) like SIGReg,
  so the direction is a tax or a null, not a gain: rand-cos at the embedding falls toward 0 (its center term),
  RankMe/d at the CLS rises less than under our term (.28 → .76 for `visregf`), and the class-cosine margin
  does not improve.
- **P3 (null branch, live):** if the arm improves kNN by ≥ +3 (the SimCLR-uniformity pattern, E17 +6.3), the
  paper's claim narrows to LeJEPA and the appendix says so; a dose ladder would then be the next question.
- **P4 (health):** no collapse, no kill-trigger incident; the z-side terms at ep100 within noise of the control.

## Discipline

Card before numbers (this file) · dose from the measured pilot line, recorded below before the chain launch ·
`num_classes=100` on every launch · nothing else launched · numbers land RAW; interpretation only with Berker.

## Launch record

- 2026-09-17: `h_reg=visreg` branch added to `sslgap/methods/visreg.py` (CPU smoke of the own loss on a [4,128,512] tensor: finite, gradient flows). Pilot **66273497** submitted (`e37-pilot`, gpu100/H100, 1 h 30 limit): `method=visreg frame=in100 num_classes=100 +method.h_reg=visreg +method.h_lamb=0.03 tag=hpull_visreg_pilot frame.epochs=2 share_log_every=1`. FAILED in 1 min: wrong Hydra group (`frame=in100`; the frame is `in100_vits16`) — my error, no training happened. Resubmitted as pilot **66273516** with `frame=in100_vits16`, everything else identical. Pilot 66273516 on gpu275 (H100): `[share] ep0 visreg=0.886(g1.365) inv=0.039(g0.534) h_visreg=0.076(g3.494)` (init twin of the visregf pilot: same g_visreg 1.365 / g_inv .53) → `[share] ep1 visreg=0.632(g0.258) inv=0.247(g0.908) h_visreg=0.121(g1.487) | omega_h=3.139 omega_z=3.231 lam=0.986`; no incident lines; ≈5 min per epoch. **λ* = (.06/.94) · (.9·.258 + .1·.908) / 1.487 = .0638 · .3230 / 1.487 = .0139** (self-check: ep1 share .060; visregf's λ was .0123 from Σ_z .3555 / g_h 1.844).
- 2026-09-17: CHAIN LAUNCHED — `in100.visreg.s0.hpull_visreg`, jobs **66273648 → 66273649** (`-J h100-slotA --dependency=singleton`, gpu100/H100, 2 × 8 h, 28 CPUs / 128 G): `method=visreg frame=in100_vits16 num_classes=100 +method.h_reg=visreg +method.h_lamb=0.0139 tag=hpull_visreg share_log_every=1`. Resume via `_last.pt` on the later links; spare third link **66273650** (finds the run complete and exits). No watcher armed.
- 2026-09-17: LANDING CHAIN queued behind the spare link (`afterok`): extract **66273723** (`adapter=native run_id=in100.visreg.s0.hpull_visreg.extL h_layers=[3,6,9] orbit_v=8`, the E29 store layout: train500.v1L, val.v1L, pairs100@audit_v1 + .o8) → probe **66273724** / twospace **66273725** / audit **66273726** (`afterok` the extract). Numbers land in `results/probes/<rid>.csv`, `results/twospace/`, `results/battery/`; nothing else is queued. If a chain link fails instead of completing, the `afterok` dependencies never fire — check `squeue`/`sacct` at the next session start.

## Numbers — RAW

*(filled on landing)*

## AGREED TAKEAWAY

*(empty by contract — filled only after the joint read with Berker)*
