# E29 — house-VISReg ± treatment, and the inv-at-h-alone control (Berker 2026-08-25)

**Status: OPEN — pre-registered before any numbers. Runs on the gpu partition
(A100|L40S|A40); the H100 pool stays with the 400 round.**

## (a) Directives
1. "visreg at in100 with and without treatment ... we should not go out of the house
   with augmentation etc. we follow in100 in house way. just updating the loss as
   visreg." → `sslgap/methods/visreg.py`: lejepa-lane anatomy/frame/aug (V=4 house
   family, 384→512 embed, house projector, house hygiene lr 1e-3/warmup 10/eta_min
   1e-5), loss = the donor's exactly (w_reg=.9 reg + w_inv=.1 view-to-mean inv — their
   lamb=.9 convex mix as explicit weights; reg = center+scale+sliced-quantile-shape,
   ported verbatim; PORT_NOTES in the module docstring; DONORS row owed).
2. The E17-T4 confound (Berker: "kind of confounded with adding both anti collapse +
   invariance at the same time") — CONFIRMED against the card: every E17 ±inv read rode
   a fixed added spread term; the clean cell (control + small h_inv ALONE) was never
   run. → `in100.vicreg.s0.gvinv`: vicreg control + h_inv at the H-wave 10%-pull dose,
   NOTHING else at h.

## (b) Cells
| run | config | dose rule → REALIZED (2026-08-25 late) |
|---|---|---|
| `in100.visreg.s0` (control) | method=visreg, house lane | — (chain 63677740–42 BURNED §(e); relaunched 63678784–86) |
| `in100.visreg.s0.visregf` (treated) | + h_reg=moment at the 512-d embedding | zoo 6%-share: λ* from pilot 63677743's ep1 [share] line, λ* = .06/(1−.06) · Σ(wg_z)/g_h → **pilot 63678562 (post-fix) ep1: Σ(wg_z)=.3555, g_h=1.844 → λ*=.0123** (self-check: share .0600 exact); chain 63678787–89 |
| `in100.vicreg.s0.gvinv` (inv-alone) | vicreg control + h_inv only | H-wave 10%-pull: h_inv* from pilot 63677744's ep1 line, same algebra at τ=.10 → **ep1: Σ(wg_z)=29.56, g_h=.625 → h_inv*=5.25**; chain 63678348–50 |

## (b2a) control/visregf formation (RAW, ep0–3/ep0–2; recorded 2026-08-25 21:03)
- control: visreg .958(g1.365)/inv .042(g.536) ep0 (g's = pilot-ep0 exact, init-twin) →
  .656/.344 ep1 → .561/.439 ep2 → .606/.394 ep3; omega_h 3.63→1.39→.70→.60, lam ~.83–.88.
- visregf (λ=.0123): h_moment_kl share **.049(ep0, g5.348 pilot-exact) → .083(ep1,
  g2.193) → .109(ep2, g2.162)** — realized share drifts ABOVE the 6% design at ep1–2
  because the floor's g holds (~2.2) while visreg/inv g's decline (1.37→.13); the E27
  chain-ep1-miss pattern, one-shot dose stands per that precedent. omega_h ep2: treated
  2.21 vs control .70 (raw). Contrast with gvinv below: the moment floor HOLDS pull,
  h_inv self-quenches — opposite formation behavior of the two h-treatments (raw
  observation, joint read owed).

## (b2) gvinv formation (RAW, chain ep0–3; recorded 2026-08-25 ~20:00)
h_inv share .034(ep0, g.470=pilot-exact) → .014(ep1, g.086) → .006(ep2, g.019) →
.002(ep3, g.004); probe .098→.188. The τ=.10 ep1-algebra target is NOT realized under
the full dose: g_h collapses ~7× by ep1 (the term suppresses its own gradient as
h-invariance rises — dose-curvature of the E27-T1 kind; mechanically the E17-T4
threshold reading, anticipated by P-E29-3). **Raised to Berker: constant-weight h_inv
self-quenches; whether a sustained-pull variant is wanted is his call — run continues
as designed meanwhile.**

## (c) Pre-registered predictions (2026-08-25, no numbers exist)
- **P-E29-1 (visreg treated):** the treated arm shows the fig_treatment_arrows pattern —
  kNN up, Θ_h up, linear ≥ control − 0.5 — a seventh arrow from a 2026 method whose own
  z-term is Gaussian-shape (the closest cousin to our floor: if the floor helps HERE,
  "conditioning belongs at h, not only at z" gains its sharpest instance).
- **P-E29-2 (inv-at-h alone):** does NOT improve — kNN flat-to-down (the E17-T4
  connectivity mechanism predicts cloud-shrink → neighborhood damage), linear ≤ +0.5,
  Θ_h DOWN (the arrow points left in fig_treatment_arrows space, opposite the floor's).
  This is the paper's "we do not want maximal invariance at h" cell, unconfounded.
- **P-E29-3 (null branch, live):** if inv-alone IMPROVES probes, the claim needs
  rewording toward dose-dependence (T4's threshold reading) — reported as-is.

## (b3) gvinv aug-thickness / image-spread check (RAW, Berker-requested 2026-08-25 ~21:00;
## read from the live orbit instrument, wandb run yjxc7te2 — no ckpt job needed)
Orbit decomposition on the fixed share batch (128 img × 4 views, train-aug channel):
W_h = mean within-image view distance² (aug thickness), B_h = debiased between-image
center distance² (image spread), both ABSOLUTE energies.
| entering ep | W_h | B_h | ω_h | B_z |
|---|---|---|---|---|
| 0 | 390 | 188 | 2.08 | 4.7 (init) |
| 5 | 54 | 94 | .58 | 498 |
| 15 | 14.4 | 32 | .45 | 714 |
| 25 | 9.2 | 23 | .40 | 819 |
| 44 | 3.3 | 9.5 | .34 | 1012 |
- BOTH contract at h (W ~120×, B ~20× by ep44; ratio improves 6×) while B_z EXPANDS ~3×
  — contraction is h-specific (var/cov hold spread at z). Whether B_h contraction is
  gvinv-specific or vicreg-h default is UNREADABLE against the July control (predates
  the orbit instrument).
- Probe vs `in100.vicreg.s0` at matched epochs: gvinv +5..+10 pts everywhere (ep43
  .513 vs .431) — **BUT the comparison is provenance-confounded 3 ways (found
  2026-08-25 late): the July-08 control ran bs=256 (gvinv 128 → 2× steps/ep; vicreg
  var/cov batch-sensitive), predates the D-036 GAP→CLS migration (different declared
  probe tap), and 7 weeks of trainer drift.** Internal tension: h_inv share ~.002 from
  ep3 — the delta is UNATTRIBUTED. Discriminator options presented (same-code bs-128
  control twin / offline ep25 orbit on July ckpts / bs-256 gvinv re-pilot); **Berker
  RULED 2026-08-25 ~21:25: HOLD — joint read first, nothing launched.** Mid-run ckpts
  on disk for whenever: gvinv_ep25.pt + July control _ep25/50/75/100. *(Superseded
  2026-08-26 eve — gvinv ckpts deleted on the cell-close directive below; the July
  control ckpts remain.)*

**AGREED READING (Berker 2026-08-25, this check CLOSED — no further runs):** the check
is consistent with his reading; **the ratio is healthy** (ω_h ~.34–.40, nowhere near
0). His worry was that MAX invariance — views collapsing onto image centers relative
to image spread — was what was helping; **it was not the case**. (Scope: this closes
the thickness/spread mechanism question only; the probe-delta provenance/attribution
facts above stand as recorded for the eventual full-cell read.)

## (c2) Landings + the exhibit directive (2026-08-26 morning)
Both visreg arms LANDED: control `in100.visreg.s0` online best **.5590** · treated
`visregf` (λ=.0123) online best **.6564** (RAW, +9.7 online at the embedding probe —
the P-E29-1 read waits on the landing probes). **Berker DIRECTED ("do the same for
visreg treated + untreated" — the D-104 dino pattern):** guillotine landing pipelines
launched for BOTH arms (`in100.visreg.s0.extL` 63705365–68 · `visregf.extL`
63705369–72; h_layers=[3,6,9]+o8, twospace+audit+probe each); the zoo gains a visreg
row (ctrl grey + treated #b3477d); paper exhibits gain the VISReg family (FAMILIES +
C8_PAIRS + fig_treatment_main's row set — the pre-registered "seventh arrow" of
P-E29-1 enters fig_treatment_arrows). Figures regenerate when both pipelines land
(joint with the dino swap regen). gvinv untouched by this directive (its own read
protocol stands).

## (c3) gvinv cell CLOSED without landing (Berker 2026-08-26 eve, verbatim: "ginv
## vicreg is not that necessary you can cancel and clean")
The gvinv landing pipeline (extract 63731881 → twospace/audit/probe 63731882–84,
launched ~17:30 per §(d)) was scancel'd while still PENDING — no feature stores were
created. All gvinv ckpts DELETED (pilot ep1/2/last + ep25/50/75/100 + best + last,
~4 GB; D-103 cleanup pattern: training logs e-gvinv_63678348–50 + the card records
stay). The cell's record = the in-training reads: §(b2) self-quench finding + §(b3)
thickness/spread check (CLOSED, Berker's agreed reading) + the §(b)-block h_inv*=5.25
dose derivation. The §(d) read protocol below no longer applies to gvinv.

## (d) Read protocol
ep100 + house probes (linear_raw_v2, knn_v1_k200) at cls/embed + battery + the two
fig_treatment_arrows coordinates (Θ_h at CLS, probe). Watch: ep1–5 formation lines,
kill-trigger discipline. Extraction: native `_asm_visreg` (canonical backbone/embed/
projector roles; same two-space layout as `_asm_lejepa`, z.embed = declared h).

## (e) Incidents
- 2026-08-25 ~19:07: first build registered the lejepa "encoder" monolith; every launch
  crash-looped at the share logger's `modules["backbone"]` lookup (control chain
  63677740–42 all FAILED in 2 min + treated pilot 63677743; the 3×8h control chain
  burned). Fix: visreg rebuilt on the canonical roles (backbone/embed/projector, the
  base-class vocabulary), the Ω z-path in both trainers taught the documented optional
  embed stage (`modules["embed"](cls) if "embed" in modules else cls` — vicreg's own
  D-043 conduit idiom; inert for embed-less methods), native `_asm_visreg`. Control
  chain + treated pilot relaunched post-fix; vicreg gvinv pilot (63677744) was
  unaffected (ep1 line clean) → gvinv chain launched same evening (63678348–50,
  h_inv*=5.25 from the ep1 algebra: (0.10/0.90)·Σ(w·g_z)/g_h = (1/9)·29.56/0.625).
- Latent condition, reported not fixed: house lejepa itself still registers the
  "encoder" monolith (seed-faithful port construction) — a house-lejepa run through
  `train.py` with `share_log_every>0` would hit the same lookup. Berker-level design
  call (monolith is load-bearing for port parity; renaming breaks shipped ckpts).
