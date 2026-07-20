# E17 — h-pull: each method's OWN desideratum term at h (the desideratum-transfer matrix)

**Status: COMPLETE — AGREED TAKEAWAY T1–T7 USER-APPROVED 2026-07-16 (§bottom; D-039, now in the
DECISIONS archive). 7 arms scored (6 + vicreg varcov@6× closure); reviewer-facing report:
`docs/report/e17_desiderata_at_h_findings.md`. Original pre-registration below unchanged.
Folded to current truth 2026-07-20 (D-053): the mid-flight amendment→refutation→sharpening
layering is now the single §Mid-flight arm evolution block, and the superseded "vicreg — PARKED"
section is folded into §vicreg CLOSURE; pre-fold text in git history.**

*(Original status, 2026-07-14: predictions locked in this file BEFORE any arm produced a number.
Vehicle = the "slight-desideratum-at-h" principle (`docs/theory/CALIBRATION_AT_H_GENERALIZATION.md`),
sharpened by Berker 2026-07-14 to "burden off the MLP + a test of the method's own claim."
GATE: byol + dino arms were DESIGN-FIRST — signed off before code.)*

## Question

E12 established that a **proxy** term at h — the Gaussian moment-KL floor (a pointwise, cluster-blind
regularizer) at small dose — is a method-general kNN-favoring conditioner (E12-T2/T7/T8), and the
H-wave that a generic view-invariance pull at h shrinks the projector's invariance jump (f7:
0.373→0.163). **E17 asks whether this is the proxy's property or a general principle**: does adding a
*small copy of each method's **own** non-collapse term* (and, second arm, its own invariance term) at
the declared projector-input h — on top of the unchanged z-loss — (a) lift burden off the head, and
(b) let the representation carry the property the method asserts for it, at acceptable probe cost?

Two readings (Berker):
- **vicreg / lejepa = a test of the method's own representational CLAIM.** They assert distributional
  properties *for the representation* (var/cov → decorrelated unit-variance; SIGReg → isotropy) yet
  enforce them only at the expander/projector. Pulling the term to h tests whether h can carry the
  claim, and how much the MLP relaxes once it does.
- **simclr / byol / dino = a test of MECHANISM transferability.** Their term is a
  contrastive / predictive / clustering burden, not a distributional claim about h. dino is the
  **falsifier**: an assignment (clustering) loss has no slight-at-h form without instantiating a
  prototype head at h — if clustering-at-h can't help, the principle is bounded to differentiable
  pointwise regularizers.

Adjudicates the guillotine (Bordes: head = protective buffer, desideratum belongs off h — vs the
methods' claims that it belongs at h). Burden-shift is the referee, per method.

## Frame (all arms)

IN-100 canonical CMC split (`~/data/imagenet100`, D-002) · ViT-S/16@224 (`frame=in100_vits16`,
100 ep, D-004). **Each arm = the method's SHIPPED lane (`in100.<method>.s0` config) + a small
copy of its own term(s) at h. Control = the shipped lane itself, byte-identical (no h-term).** No
E12 "package" (spec_norm / embed_calib / lr 3e-4) — that was moment-floor scaffolding; h-pull rides
the shipped recipe so the control is literally the lane and the h-term is the ONLY factor.

- Per-lane grad_clip is INHERITED (byte-identical control): dino 3.0, lejepa `null` (D-011
  port-exact — no clipping), rest = frame default 1.0. House kill-trigger (single-step grad-norm
  >100× running median = INCIDENT, stop & read) stands on every arm regardless.
- h = the declared **projector-input** tap (D-036, PROTOCOL §3): vicreg/simclr/byol → student
  trunk-**CLS**; dino → **student** trunk-CLS (gradients flow through student; audited teacher CLS
  follows by EMA); lejepa → **z.embed** (trunk-CLS→Linear(384→512)).
- seed 0 only (seed replicates deferred). wandb online. Quarter-cadence + best + last ckpts.
- **Dose = SMALL** (E12-T2 lesson: full enforcement scrubs semantics, small dose = net-positive
  conditioner with an interior optimum). Each h-term dosed to **~10% of the shipped z-term's weighted
  pull** (measured on the real first batch via `e12h_pull.py`, the H-wave convention), recorded
  below before launch, launched verbatim (no mid-flight discretion).

## Arms (self-explanatory names `in100.<method>.s0.hpull_<term>`)

| method | reading | arm 1 (non-collapse) | arm 2 (+ own inv) | control | h tap |
|---|---|---|---|---|---|
| **vicreg** | own claim | `hpull_varcov` = +λ_h·(w_var·var(h)+w_cov·cov(h)) | `hpull_varcov_inv` = +inv-MSE(h) | `in100.vicreg.s0` | CLS |
| **lejepa** | own claim | `hpull_sigreg` = +λ_h·SIGReg(embed) | `hpull_sigreg_inv` = +inv-MSE(embed) | `in100.lejepa.s0` | embed |
| **simclr** | mechanism | `hpull_uniform` = +λ_h·uniformity(h) | `hpull_uniform_align` = +alignment(h) | `in100.simclr.s0` | CLS |
| **byol** | mechanism | `hpull_align` = student-h → EMA-teacher-h regress (stop-grad) | — (align IS its only term) | `in100.byol.s0` | CLS |
| **dino** | mechanism (falsifier) | `hpull_protoce` = small linear proto-head @h + center/sharpen/CE | — (CE bundles both) | `in100.dino.s0` | CLS |
| ~~ijepa~~ | — | OUT — loss already targets backbone patch reps (no z to mirror) | | | |

The own-term is the method's **actual** term at h (not the moment-KL proxy): vicreg's var/cov
functions, lejepa's SIGReg module, simclr's uniformity/alignment (Wang–Isola 2020), byol's
predictive regress, dino's prototype-CE. arm 2 exists only where the method has an EXPLICIT
invariance term to add (vicreg inv-MSE; lejepa inv-MSE; simclr alignment). byol/dino have a single
combined term → one arm each.

### Per-method design

**vicreg** (code: `h_reg=varcov` in vicreg.py). h-term = a small copy of vicreg's OWN non-collapse
loss at CLS, keeping its internal 25:1 var:cov ratio: `λ_h·(w_var·[var(ha)+var(hb)] +
w_cov·[cov(ha)+cov(hb)])`, per-view (ha,hb = the two views' CLS), exactly mirroring the z-side
computation. arm 2 adds the existing `h_inv` (view-MSE at CLS, its own inv term). var/cov are
dimension-adaptive (cov normalizes by d=384; var targets unit std) — no rescale needed.

**lejepa** (NO code — already wired). arm 1 = `h_reg=sigreg h_lamb=<dose>` (its OWN SIGReg at
embed); arm 2 = `+ h_inv=<dose>` (view-MSE at embed). NOTE this is DISTINCT from f7, which used the
moment-KL **proxy** (`h_reg=moment`): E17 uses lejepa's own SIGReg, the sharper test (SIGReg reads
higher-order/cluster structure via the anti-CLT channel R4c-6 that moment-KL is blind to by
construction — so it can tax where the proxy helped; that dissociation is the result).

**simclr** (code: `h_uniform` / `h_align` in simclr.py). Wang–Isola functionals on ℓ2-normalized
CLS: `uniformity(X) = log mean_{i≠j} exp(-t‖xi-xj‖²)` (t=2, canonical); `alignment(Xa,Xb) =
mean‖xi_a-xi_b‖²` on positive pairs. uniformity IS simclr's non-collapse mechanism (negatives),
alignment IS its invariance mechanism (positives) — the two-arm split maps exactly. (Running NT-Xent
itself at h bundles both inseparably and can't realize "non-collapse first"; the WI decomposition is
the faithful way to get both arms. t=2 declared, amendable.)

**byol** — DESIGN-FIRST (§Design forks). h-term = predictive regress of student-h onto stop-grad
EMA-teacher-h; the EMA teacher trunk already exists (reused). Fork: predictor@h vs EMA-only.

**dino** — DESIGN-FIRST (§Design forks). h-term = a small LINEAR prototype head at student-CLS
(ℓ2-norm CLS → weight-normed Linear(384→K_small)), softmax + centering + sharpening + CE between
EMA-teacher-h assignment and student-h assignment (mirrors dino_ce). Forks: K_small, temps,
center_m, EMA-teacher-head vs student-only, crop set.

## Design forks (byol + dino — RESOLVED, Berker sign-off 2026-07-14, discipline #1)

**byol · direct alignment, NO projections [CHOSEN — Berker 2026-07-15 refined the fork].** hpull_align
= normalized-MSE(student trunk-CLS view a, **stop-grad EMA-teacher trunk-CLS** view b), symmetrized —
byol's own predictive-regression mechanism (EMA target + stop-grad) at h, but **no predictor/projector
MLP**. Same-dim (384↔384) so no projection is needed; the EMA teacher trunk is reused (no new module,
no extra forward — `ht` is already computed). Collapse-safe: it is a small ADDITIVE term on the
unchanged shipped byol loss, which carries the predictor+EMA anti-collapse. (Supersedes the initial
"predictor@h" pick — the predictor was machinery the additive small-dose regime doesn't need; the
simpler direct form is the cleaner design. Collapse canary = teacher-CLS std, still monitored.)

**dino · faithful mini-DINO@h, K_small=512 [CHOSEN].** ℓ2norm(student-CLS) → weight-normed
Linear(384→512) proto-head (small vs the z-head's 4096, over-clusters IN-100's 100 classes);
**EMA-teacher copy of the h-head** (mirrors dino's teacher-head EMA, reuses the existing EMA update);
reuse dino's temps (t 0.04 / s 0.1) and center_m 0.9 with a **separate center buffer**; CE over the
**2 global crops only** (core clustering signal, minimal machinery — skip the 64 locals for the aux
head). Rejected: K=256 (leaner but chose over-cluster), student-only h-head (breaks teacher-EMA
asymmetry).

## Pre-registered directional predictions (locked before numbers)

**Cross-method PRIMARY (head burden).** For every arm, vs its byte-identical control: the projector
**invariance-jump shrinks** — report the (pos_cos, rand_cos, cos_margin) triple, never the margin
alone (METRICS.md / E12-T9). Operator **head-Lipschitz** (weight σ_max per head layer) drops on the
head segments. Diag-KL ladder shifts conditioning into the trunk. Effect strongest on the arm-2
(+own-inv) variants, which offload invariance directly (f7 mechanism, proxy anchor 0.373→0.163).

**Cross-method SECONDARY (h distribution + probes).** At h: the method's asserted property appears
(vicreg: off-diag corr↓, unit var, effrank↑; lejepa: isotropy/EP↓, effrank↑; simclr: rand_cos↓ /
uniformity↑, effrank↑). Probes: kNN-favoring gain (T7 pattern), converged-linear tax small and local
to the constrained tap. **Interior-optimum check** = head-burden drops AND h-probe holds.

**Per arm:**
- **vicreg-P (hpull_varcov):** vicreg's own conditioning at h — off-diag |corr|↓, var-floor
  satisfied at CLS, effrank↑; kNN@CLS ≥ control, converged-linear within ~1–2 pts. Anchor: the
  moment-floor proxy gave vicreg h effrank 15→113, off-diag 0.128→0.057 (E12-T8); var/cov is
  vicreg's OWN version, cov targets off-diagonals explicitly → expect ≥ that decorrelation.
- **vicreg-P2 (hpull_varcov_inv):** larger inv-jump shrink than varcov-only; watch var-barrier vs
  inv-pull equilibrium (f7 showed floor+inv fight to an equilibrium). Collapse-flavored inv opposed
  by the var floor → kill-trigger standing.
- **lejepa-P (hpull_sigreg):** THE own-claim test. At 10% dose, far gentler than A2 (SIGReg@embed at
  equal-pull → −21.5 lin) but the cluster channel may tax MORE than the moment-KL proxy f2
  (+1.2 lin): predict kNN gain with a converged-linear tax **between** f2's +1.2 and A2's damage —
  the sign of that tax is the datum. If net-positive → lejepa's isotropy claim transfers to h; if
  taxing → the own term touches semantics where the cluster-blind proxy didn't.
- **lejepa-P2 (hpull_sigreg_inv):** inv-jump shrinks (f7 mechanism) with SIGReg as companion instead
  of the moment floor; isotropy+inv may fight harder than floor+inv — watch embed calibration &
  grad-norm.
- **simclr-P (hpull_uniform):** rand_cos↓ (decorrelated negatives → clean cos_margin), effrank↑,
  kNN gain; inv-jump shrink modest (uniformity adds no invariance).
- **simclr-P2 (hpull_uniform_align):** alignment@h offloads projector invariance → largest inv-jump
  shrink of the simclr arms; kNN gain retained.
- **byol-P (hpull_align):** byol's sole mechanism (predictive alignment) offloaded to h → inv-jump
  shrinks; probe gain kNN-favoring. Collapse risk keyed to the fork (EMA-only higher) — teacher-CLS
  std canary + kill-trigger.
- **dino-P (hpull_protoce) — falsifier:** if dino's OWN clustering term at h gives the T7 pattern
  (kNN gain at CLS, small local linear tax) like its moment-floor arms (gd: kNN cls +3.6, gap +3.3),
  the principle extends to assignment losses; if it can't help / needs the full head, the principle
  is bounded to differentiable pointwise regularizers. CE aligns h across views → inv-jump shrinks.

## Kill criteria

- **K1** any ignition/INCIDENT (grad-norm >100× running median) → arm stops; per-step curve
  forensics at onset before any rerun (house default).
- **K2** converged-linear tax > 3 pts vs control at the constrained tap AND at h → dose too high;
  re-read the dose curve before a second arm (E12 f2-vs-A3 lesson).
- **K3 (byol/dino collapse)** teacher-CLS std → 0 (byol) / proto usage → 1 or teacher-entropy floor
  (dino) → the mechanism collapsed h; stop & read.

## Dose measurement record (mechanical fill at launch — not interpretation)

*(Filled from `e12h_pull.py` runs; 10%-of-shipped-weighted-pull rule, launched verbatim.)*

Rule: h-term weighted pull = 10% of the shipped z-term's weighted pull (weight × g_enc), on the
real first batch at seed 0 (`e12h_pull.py`), launched verbatim. Jobs 62341124/125/126 (lejepa/
vicreg/simclr) + 62342249/250 (byol/dino), 2026-07-14/15.

| arm | shipped z-term (weight × g_enc = weighted pull) | h-term (unit g_enc) | launch dose |
|---|---|---|---|
| **lejepa** hpull_sigreg | SIGReg@proj 0.02×321.25 = 6.425 | h_sigreg 170.74 | **h_lamb=0.003763** |
| lejepa hpull_sigreg_inv (+inv) | inv@proj 0.98×1.539 = 1.508 | h_inv 0.4737 | **h_inv=0.3184** |
| **simclr** hpull_uniform | nt_xent 1×9.510 = 9.510 | h_uniform 7.371 | **h_uniform=0.1290** |
| simclr hpull_uniform_align (+align) | nt_xent 1×9.510 = 9.510 | h_align 5.114 | **h_align=0.1860** |
| **vicreg** hpull_varcov | var+cov 25×0.661+1×35.65 = 52.18 | 25×3.845+1×1945.4 = 2041.5 | **h_lamb=0.002556** |
| vicreg hpull_varcov_inv (+inv) | inv 25×0.4172 = 10.43 | h_inv 1.0607 | **h_inv=0.9834** |
| **byol** hpull_align (linear predictor, redesigned) | regress 1×4.259 = 4.259 | h_align 8.017 (through the predictor) | **h_align=0.05313** |
| **dino** hpull_protoce | dino 1×1.5505 = 1.5505 | h_protoce 1.1384 | **h_protoce=0.13620** (K=512) |

Raw observation (not interpretation): vicreg's h-bundle pull is 95% covariance (g_h_cov 1945 ≫
25·g_h_var 96) — at the trunk CLS vicreg's own term is almost pure decorrelation (the CLS is
near-unit-variance already but heavily correlated). lejepa's plain-lane g_sigreg 321 (vs the
package f7's 454 — no embed_calib here) → E17 anchors on the shipped lane it actually rides.

## Discipline (non-negotiable, carried from E12/H-wave)

1. Discussion-first for byol + dino — sign-off BEFORE code. Walls are information; no smuggled machinery.
2. Pre-register directional predictions per arm before any numbers (this file).
3. Dose by MEASURED pull (~10% of shipped z-term pull); record measured pull + launch dose before launching.
4. 2-ep smoke at launch dose → 3×8h chain; per-lane grad_clip; grad-norm kill-trigger standing.
5. Self-explanatory run names (`in100.<method>.s0.hpull_<term>`).
6. Compute: ≤2 H100s (singleton h100-slotA/B, --dependency=singleton); gpu partition UNCONSTRAINED
   for smokes/extraction/probes/battery/figures. Extraction declaration-agnostic (adapter=native
   h_layers=[3,6,9] orbit_v=8 do_eval=true do_pairs=false bs=128, run_id `<run>.ext`).
7. Numbers land raw; NO takeaway without Berker.

## Execution log (in-flight state; not results)

- **2026-07-14/15.** Code landed + CPU-validated (step+backward+term-logging; dino also arch
  round-trip + h_head grad + h_center persistence): vicreg `h_reg=varcov` (own var+cov@CLS, 25:1
  internal), simclr Wang–Isola `uniformity`/`alignment`@CLS, byol direct `h_align`@CLS (no
  predictor), dino `h_protoce` = linear proto-head@CLS K=512 (`DINOLinearHead` + EMA-teacher head +
  own center). lejepa = no code (`h_reg=sigreg`+`h_inv` pre-existing).
- Doses measured + recorded (table above). 8 arms' 2-ep smokes launched at launch dose (gpu
  partition). Controls re-extracted to `in100.<method>.s0.e17c.ext` (declaration-agnostic
  h_layers=[3,6,9] orbit_v=8 do_eval do_pairs=false) — the shipped-lane `.ext` lacked the o8 orbit
  + v1L L-tap stores the head-burden readout needs; fresh run_id avoids clobbering/manifest-shadowing.
- Scoring: `experiments/e17_score.py` (raw head-burden + h-dist + probes, arm vs e17c control) +
  `experiments/e17_figs.py` (per-method control/+non-collapse/+inv panel). Both robust to incremental
  landing.
- **Compute divergence from the goal (flagged to Berker):** chains on the **gpu partition in
  parallel** (F-wave/G-wave precedent), not the 2 H100 slots — 8 arms parallelize far better; f8
  still on an H100 slot.
- (Executed: chains → extraction → probes/battery → the scoring passes below.)

### Mid-flight arm evolution (2026-07-15; folded 2026-07-20 from three stacked blocks: amendment → run-based refutation → sharpened priority)

**Trigger:** Berker read vicreg's `h_inv` panel ≈ 0 ("obviously not necessary"). The control-based
check agreed — pos_cos@h is already 0.86–0.96 for EVERY method (lejepa embed .955, byol cls .934,
vicreg cls .921, simclr cls .858, dino cls .683), the margin held down by rand_cos (vicreg .71,
byol .77, lejepa .58, simclr .48, dino .15) — suggesting the +inv arms were near-null and
decorrelation the whole offloadable burden. **The RUNS refuted that read** (live ep16–20;
matched-ep25 cleaner): under an ACTIVE non-collapse term the picture inverts — lejepa control POS
.955/NEG .584 → hpull_sigreg POS .615/NEG .014 (SIGReg breaks the cone but drags same-image POS
down with it) → hpull_sigreg_inv POS .807/NEG .011 (the inv term RECOVERS the alignment SIGReg
destroyed, at NEG≈0); matched-ep25 control .979/.886 → sigreg .644/.008. simclr (~ep20): control
.858/.483 → uniform .605/.015 → +align .706/.014. vicreg: control .921/.711 → hpull_varcov
.748/.341 — var+cov is a WEAK decorrelator (cone only halved, vs SIGReg/uniformity → ~.015).
**Lesson (stands): judge redundancy on the RUN — non-collapse term active — not the control**,
whose POS is high only because nothing is decorrelating it yet. The +inv is a corrective for the
non-collapse term's alignment cost, tolerable where that term already decorrelates well.

**Final arm dispositions (Berker 2026-07-15):**
- `hpull_varcov_inv` (vicreg) DROPPED/killed (h_inv≈0 in that lane).
- `hpull_sigreg_inv` (lejepa) KEPT ("looks very good"; NEG≈0 satisfies his neg≈0→keep rule; POS
  recovery is real work). Over-dose flag recorded: SIGReg@10% dropping POS to .615 is strong.
- `hpull_align` (byol) REDESIGNED — the direct/no-predictor version was inert (student-CLS ≈
  EMA-teacher-CLS: .999 same-view, gap .132); Berker: "add a linear to its h and do the pull
  there" → small trainable `byol_h_predictor` = Linear(384→384), student-CLS → stop-grad
  EMA-teacher-CLS; re-pulled, re-dosed, relaunched.
- vicreg base resolution (Berker): **strengthen var+cov until it decorrelates, at the SMALLEST
  pull that does so** (small-dose regime, E12-T2) — var+cov-only dose sweep 2×/4×/6× (h_lamb
  .005/.010/.015), neg read at ep15 → the c015 CLOSURE below. Churn owned: the +inv was first
  killed on a wrong "inv hurts" read, the wrong arm boosted, then over-boosted to 8×, before the
  actual ask — always "strengthen the base var+cov's decorrelation, minimally." — was executed
  (`hpull_varcov` itself was never killed, only `hpull_varcov_INV`).
- Kept unchanged: the 4 non-collapse arms + simclr `hpull_uniform_align`.

**E17 deliverable, sharpened (Berker 2026-07-15):** each own non-collapse term's decorrelation
strength at h — SIGReg/uniformity strong (neg→~.015), var+cov weak (neg→.341) — plus the
POS-recovery of the +inv where decorrelation succeeds.

## Numbers land below this line as they arrive; AGREED TAKEAWAY only after joint discussion.

### First scoring pass — 6 arms at ep100 (2026-07-16; RAW + mechanical direction check; `experiments/e17_score.py`; NO takeaway)

**PRIMARY — projector inv-jump (head invariance burden), arm vs byte-identical control.** Predicted
(pre-reg): drops method-agnostically. **Mechanical outcome: DROPS ON EVERY ARM.**

| method | arm | inv-jump ctrl→arm (Δ) | h pos / rand / margin | diag-KL@h (ctrl) | lin Δ | knn200 Δ |
|---|---|---|---|---|---|---|
| lejepa | hpull_sigreg | +.539→**+.145** (−.394) | .724 / **.005** / .719 | .068 (.956) | −1.4 | −4.3 |
| lejepa | hpull_sigreg_inv | +.539→**−.007** (−.546) | .875 / .003 / .872 | .041 (.956) | **−7.2** | **−11.6** |
| simclr | hpull_uniform | +.411→**+.157** (−.253) | .639 / **.002** / .638 | 1.62 (1.25) | +0.3 | **+6.3** |
| simclr | hpull_uniform_align | +.411→**+.010** (−.401) | .787 / .002 / .784 | 1.71 (1.25) | +0.2 | +2.5 |
| byol | hpull_align | +.513→**+.097** (−.416) | .786 / .190 / .595 | 1.32 (1.50) | **+4.1** | **+5.4** |
| dino | hpull_protoce | +.149→**−.006** (−.155) | .792 / .063 / .729 | .754 (.655) | +2.6 | −4.2 |

Raw reads (mechanical, NOT takeaways): (a) **head-burden drop is universal** — the +own-inv/+align
arms drive the projector inv-jump to ≈0 (head does no invariance work). (b) **decorrelation split:**
SIGReg/uniformity crush rand_cos to ~.002–.005 at h; byol align only .765→.190 (no non-collapse
term — align only); dino .146→.063 (already low). (c) **probe split:** simclr `hpull_uniform`
(+6.3 knn, +0.3 lin) and byol `hpull_align` (+5.4 knn, +4.1 lin) = the T7 kNN-favoring gain; dino
lin+2.6/knn−4.2; **lejepa taxes** (sigreg −1.4/−4.3; sigreg_inv **−7.2/−11.6** — the strong SIGReg
decorrelation + inv POS-recovery has a real semantic cost, POS .955→.724→ the inv pulls it to .875).
Monitor bests (T9-biased, not the arbiter): sigreg .511, sigreg_inv .436, uniform .615,
uniform_align .597, align .616, protoce **.727**. **Interpretation owed to Berker.**

### Second scoring pass — verification + decompositions (2026-07-16; RAW + mechanical; NO takeaway)

Session opened with a skeptical re-verification (prior session ran on a different model). (i) Re-run
of `e17_score.py` reproduces the first-pass table exactly, all 6 arms. (ii) Implementation review of
the landed arms against the per-method designs: FAITHFUL (simclr WI functionals on normalized CLS,
squared-distance identity + out-of-place diag mask; byol linear h-predictor on student CLS →
stop-grad EMA-teacher CLS, symmetrized, predictor in param_groups; dino mini-DINO@h with EMA'd
h-head, separate persisted h_center, ep0-freeze extended to the h-head, CE over the 2 globals
cross-view; vicreg per-view var+cov at CLS, 25:1 internal). ONE minor deviation logged: simclr
`uniformity` pools both views into one 2N batch, so ~2% of the pairwise mass is positive pairs
(Wang–Isola reference form is per-view); direction unaffected. (iii) Extraction provenance
spot-checked: ctrl ← shipped-lane ep100, arms ← hpull ep100 (epoch 99, adapter=native).
Figures: `results/figures/e17/e17_{lejepa,simclr,byol,dino}_hz.png` (vicreg auto-skipped, 1/3 arms).

**(a) Jump decomposition** (`e17_score.py` `jump-split`): jump = Δpos(across head) − Δrand(across head).

| run | pos@h | rand@h | pos@z | rand@z | jump | = Δpos | + (−Δrand) |
|---|---|---|---|---|---|---|---|
| lejepa ctrl | .955 | .584 | .908 | −.002 | +.539 | −.047 | +.585 |
| lejepa sigreg | .724 | .005 | .868 | .004 | +.145 | +.144 | +.001 |
| lejepa sigreg_inv | .875 | .003 | .869 | .005 | −.007 | −.006 | −.001 |
| simclr ctrl | .858 | .483 | .788 | .002 | +.411 | −.070 | +.481 |
| simclr uniform | .639 | .002 | .797 | .002 | +.157 | +.158 | −.000 |
| simclr uniform_align | .787 | .002 | .796 | .002 | +.010 | +.009 | +.000 |
| byol ctrl | .934 | .765 | .861 | .179 | +.513 | −.073 | +.586 |
| byol align | .786 | .190 | .870 | .177 | +.097 | +.084 | +.013 |
| dino ctrl | .683 | .146 | .781 | .094 | +.149 | +.098 | +.051 |
| dino protoce | .792 | .063 | .818 | .094 | −.006 | +.026 | −.032 |
| vicreg ctrl | .921 | .711 | .701 | .011 | +.479 | −.220 | +.699 |

Mechanical: every control jump but dino's is ≥94% −Δrand (cone removal); pos DROPS across the head
in all non-dino controls. In the non-collapse arms the residual jump flips to Δpos (the head
re-aligns what the h-term cost). pos/rand @z are near-identical ctrl vs arm for every method.

**(b) Control-anchored battery at declared h + probes** (raw|full; lin = linear_raw_v2, knn = knn_v1_k200, %):

| run | effrank | rankme | EP | kurt_worst | mean\|corr\| | uniformity | lin | knn200 |
|---|---|---|---|---|---|---|---|---|
| lejepa ctrl | 24.5 | 35.0 | 652 | 1.50 | .150 | −1.44 | 60.4 | 52.4 |
| lejepa sigreg | 31.4 | 44.1 | 504 | 2.86 | .148 | −3.48 | 58.9 | 48.1 |
| lejepa sigreg_inv | 14.9 | 21.8 | 1106 | 1.09 | .213 | −3.20 | 53.2 | 40.8 |
| simclr ctrl | 51.0 | 146.3 | 230 | 0.88 | .102 | −1.90 | 60.6 | 49.5 |
| simclr uniform | 134.5 | 219.7 | 142 | 6.56 | .073 | −3.81 | 60.9 | 55.9 |
| simclr uniform_align | 66.0 | 141.1 | 204 | 1.28 | .093 | −3.78 | 60.8 | 52.1 |
| byol ctrl | 45.5 | 92.1 | 386 | 2.09 | .130 | −0.84 | 58.4 | 46.0 |
| byol align | 54.4 | 142.7 | 390 | 4.67 | .121 | −2.96 | 62.4 | 51.4 |
| dino ctrl | 92.4 | 203.6 | 160 | 0.89 | .085 | −3.25 | 68.6 | 59.9 |
| dino protoce | 79.9 | 188.7 | 189 | 1.60 | .091 | −3.55 | 71.2 | 55.7 |
| vicreg ctrl | 66.9 | 148.8 | 238 | 0.86 | .095 | −1.09 | 64.5 | 55.1 |

Mechanical: sigreg_inv effrank 14.9 < ctrl 24.5 (BELOW control); uniform_align 66 vs uniform 134.5
(where a +inv/+align arm exists it halves/reverses the non-collapse arm's rank expansion). EP moves
toward Gaussian only for sigreg (652→504) and uniform (230→142); sigreg_inv 1106 > ctrl. diagKL and
probes dissociate in both directions (simclr diagKL 1.25→1.62 with knn +6.3; lejepa .956→.068 with
knn −4.3).

**(c) h-cone mean decomposition** (`cone@h`, v1L store): rand_cos ≈ mu_share and centered-rand ≈ 0
for EVERY run — ctrl: lejepa .587/.588/.003 · simclr .501/.497/.001 · byol .771/.773/.002 · dino
.161/.161/.000 · vicreg .712/.715/.001; arms likewise. The h-cone is carried by the mean direction.
Code facts (not interpretation): vicreg's var+cov is computed on batch-centered features
(mean-blind); SIGReg/uniformity/moment-KL each see or move the mean. Port facts: our simclr
projector has NO BatchNorm anywhere yet reaches rand@z .002; byol's z keeps rand .18 (no spread term,
BN on hiddens only).

**(d) lejepa trunk-CLS behind the constrained embed** (lin/knn200): ctrl 65.5/51.4 → sigreg
62.2/48.6 → sigreg_inv 61.7/42.8. The sigreg tax is not local to the embed layer. (Control trunk-CLS
lin 65.5 > control embed lin 60.4.)

### Third pass — probe/geometry decompositions for the joint discussion (2026-07-16; RAW, mechanical)

Regenerate: `experiments/e17_centered_probe.py` (f/g), wandb run summaries (e), `results/probes/*.csv`
knn_v1_k20 rows (h). Figures regenerated with effrank/RankMe/mean|corr| rows added
(`results/figures/e17/e17_*_hz.png`, now 11 metric rows — Berker 2026-07-16 request).

**(e) Final per-term z-losses (wandb summary = last logged step; arm vs control):**

| method | z-term | ctrl | +non-collapse | +inv/align |
|---|---|---|---|---|
| lejepa | sigreg@proj | 0.984 | 1.188 (+21%) | 1.406 (+43%) |
| lejepa | inv@proj | 0.053 | 0.062 | 0.076 |
| simclr | nt_xent | 2.417 | 2.475 | 2.520 |
| byol | regress | 0.467 | 0.448 (−4%) | — |
| dino | dino CE | 1.306 | 1.368 | — |

Mechanical: the z-terms end EQUAL-OR-WORSE under every h-pull except byol's (whose h-term is the
same functional as its z-term). h-term finals: h_sigreg 2.30 / 5.13 (sigreg / sigreg_inv — far from
its own z-level ~1), h_uniform −3.90/−3.75, h_align(byol) 0.448-companion 0.590, h_protoce 0.609.

**(f) Cosine-kNN k200 raw vs TRAIN-MEAN-CENTERED (eval-time mean removal; knn_v1 is weighted-cosine
⇒ mean-sensitive) + centered orbit triple (pos_c = view alignment with the cone removed):**

| run | knn200 | knn200 centered | pos_c | rand_c |
|---|---|---|---|---|
| lejepa ctrl | 52.4 | 52.9 | .875 | .003 |
| lejepa sigreg | 48.1 | 48.3 | .723 | .001 |
| lejepa sigreg_inv | 40.8 | 40.6 | .875 | −.000 |
| simclr ctrl | 49.5 | 51.5 | .725 | .002 |
| simclr uniform | 55.9 | 55.9 | .639 | .001 |
| simclr uniform_align | 52.1 | 52.1 | .786 | .002 |
| byol ctrl | 46.0 | 49.4 | .694 | .004 |
| byol align | 51.4 | 51.9 | .726 | .002 |
| dino ctrl | 59.9 | 60.8 | .630 | .001 |
| dino protoce | 55.7 | 55.8 | .777 | .000 |
| vicreg ctrl | 55.1 | 56.5 | .712 | .002 |

Mechanical: centering the CONTROL recovers only +0.5–3.4 knn (largest byol); every gaining arm still
beats its centered control; arms are centering-insensitive (cones already gone). Raw pos was
cone-inflated (byol .934→pos_c .694, vicreg .921→.712, lejepa .955→.875); the +inv arms restore
pos_c to/above control (sigreg_inv .875 = ctrl; protoce .777 vs .630).

**(g) Class-variance share B/T (e12_class_align convention) at h and z.final + cone@z:**

| run | B/T@h | B/T@z | cone@z |
|---|---|---|---|
| lejepa ctrl / sigreg / sigreg_inv | .512 / .400 / .500 | .518 / .471 / .470 | ~0 |
| simclr ctrl / uniform / uniform_align | .293 / .210 / .311 | .293 / .283 / .261 | ~0 |
| byol ctrl / align | .335 / .373 | .425 / .459 | **.171 / .168** |
| dino ctrl / protoce | .237 / **.411** | .383 / .418 | .076 / .082 |
| vicreg ctrl | .313 | **.160** | .009 |

Mechanical: B/T@z > B/T@h only for byol/dino; equal for lejepa/simclr; INVERTED for vicreg (its z
has the lowest class share measured). protoce lifts h's B/T to its z's level. byol's residual z-cone
(.171) is a mean offset (matches its mu_share .168). B/T dissociates from kNN in both directions
(uniform: B/T↓ knn↑; protoce: B/T↑ knn↓ — F-wave f2 precedent: B/T .482→.157 with knn +7.5).

**(h) k-sensitivity of the kNN effect (Δ vs control):** sigreg −4.5(k20)/−4.3(k200) k-flat ·
sigreg_inv −10.4/−11.6 · uniform +3.7/+6.3 (grows with k) · uniform_align +0.6/+2.5 · byol align
+5.8/+5.4 k-flat · protoce −2.4/−4.2 (halves at k20).

**(i) H-wave adjacent:** `e12f8` (shape-dominant f7 variant, ckpt ep100) extraction+probe+battery
chain launched (jobs 62372997/62372999/62373000, gpu partition) — was never extracted.

### Fourth pass — mechanism checks from the joint discussion (2026-07-16; RAW, mechanical)

Berker directions: identify why the moment-KL proxy helps while own-SIGReg hurts at the same tap;
test the MLP hypothesis ("if h already satisfies the losses, the MLP does not have to make crazy
changes to keep them satisfied at z"); centered versions of the mean-sensitive readouts.
Code fact anchoring (j): `MomentFloor` = KL(N(μ_Q,Σ_Q)‖N(0,I))/d′ on a fresh random 128-d slice —
mean + FULL within-slice covariance (logdet barrier), "blind to clusters and all higher-order shape
BY CONSTRUCTION" (its docstring). SIGReg = per-slice CF distance = moments 1–2 PLUS all shape.
The f2-vs-sigreg contrast therefore isolates the higher-order shape channel.

**(j) Class-pair d′ at declared h** (1000 random class pairs, seed 0; projection on μc1−μc2;
d′=|m1−m2|/√((v1+v2)/2); regenerate: session script `e17_mech.py` §1):

| run | mean d′ | p10 d′ | frac>2 | (Δknn200) |
|---|---|---|---|---|
| lejepa ctrl / sigreg / sigreg_inv | 4.41 / **3.65** / 4.03 | 2.61 / 2.54 / **2.31** | .965 / .959 / .938 | — / −4.3 / −11.6 |
| lejepa-E12lane C1 / f2 | 4.44 / 4.27 | 2.71 / **3.12** | .968 / **.994** | — / +7.5 |
| simclr ctrl / uniform / uniform_align | 3.98 / 3.50 / 3.90 | 2.48 / 2.56 / 2.69 | .967 / .983 / .978 | — / +6.3 / +2.5 |
| byol ctrl / align | 3.74 / 3.90 | 2.38 / 2.57 | .955 / .969 | — / +5.4 |
| dino ctrl / protoce | 4.88 / 5.08 | 3.25 / 3.31 | **.993 / .974** | — / −4.2 |
| vicreg ctrl | 4.34 | 2.75 | .980 | — |

Mechanical: own-SIGReg drops mean class-pair separation −17% (the shape channel's gradient
penalizes bimodal projections = separated class pairs); the floor leaves the mean ~flat and LIFTS
THE WORST TAIL (p10 +15%, frac>2 → .994); the probe winners share the tail-lift signature
(uniform/byol-align p10↑); sigreg_inv RECOVERS mean d′ (4.03) yet has the worst kNN — its damage is
within-class/rank (effrank 14.9), invisible to a between-class stat; protoce raises mean d′ while a
small tail falls below 2 (.993→.974) and kNN drops with k-sensitivity (fragmentation signature).
No single stat is monotone with kNN: the dial set is {cone, tail-d′, rank/within-class integrity}.

**(k) Head "gentleness" — h~z.final cross metrics (from landed .cross.csv batteries):**

| run | CKA | nbr-jaccard | procrustes | knn-agree |
|---|---|---|---|---|
| lejepa ctrl / sigreg / sigreg_inv | .774 / .755 / **.955** | .444 / .339 / **.705** | .470 / .576 / **.183** | .675 / .569 / .767 |
| lejepa-E12lane C1 / f2 | .819 / **.460** | .454 / .225 | .412 / .604 | .684 / .571 |
| simclr ctrl / uniform / uniform_align | .488 / .739 / **.877** | .318 / .393 / .623 | .763 / .598 / .393 | .555 / .613 / .736 |
| byol ctrl / align | .907 / .916 | .462 / .503 | .413 / .402 | .629 / .668 |
| dino ctrl / protoce | .389 / .640 | .161 / .389 | .959 / .652 | .522 / .735 |
| vicreg ctrl | .375 | .422 | .842 | .679 |

Mechanical: head-gentleness rises monotonically with how FULLY h satisfies the z-desiderata
(simclr ctrl→uniform→+align; sigreg_inv near-identity; protoce). Counter-cases: sigreg
(decorrelated but dis-aligned h → head wilder than control) and f2 (floor ⊥ z-desideratum, rank-217
input → the WILDEST head measured, CKA .46, on the best-probing representation). Gentle+good
(uniform), gentle+bad (sigreg_inv), wild+good (f2), wild+bad (sigreg) all realized → gentleness and
h-quality are independent axes. Control seg-stretch for reference: lejepa embed→tap1 0.28 (arms
.07/.12), simclr cls→tap1 1.66 / tap1→out 3.88 (uniform 2.35/5.42, +align 3.86/6.45).

**(l) Centered readouts** (`experiments/e17_centered_probe.py` → `results/diag/e17_centered.csv`;
figure `results/figures/e17/e17_centered.png`): kNN with train-mean removed at eval + pos with
pooled-mean removed. Δknn200 raw → centered: uniform +6.3→+4.4 · byol align +5.4→+2.5 ·
uniform_align +2.5→+0.6 · sigreg −4.3→−4.6 · sigreg_inv −11.6→−12.3 · protoce −4.2→−5.0.
**T7-continuity result: C1/f2 are centering-INSENSITIVE** (C1 53.1→53.9, f2 60.5→60.3 — the E12
package's embed_calib had already removed the mean) ⇒ f2's +7.5 kNN gain has no cone component.
Also: f2 pos_c .542 vs C1 .847 — the floor traded true view-alignment for neighborhoods, the same
signature as uniform (.725→.639); the +inv/+align arms trade in the opposite direction everywhere.
byol z-side probes under the arm: lin@z 47.3→52.1, knn@z 44.7→48.7 (only method whose z-loss also
improved). Control seg-stretch completed: byol cls→tap1 2.82/tap1→out 10.31/out→tap1 0.27 · dino
2.13/0.67/1.37 · vicreg 0.84/1.43/5.84.

**(m) Augmentation-cloud overlap at h — the T4 connectivity question (Berker 2026-07-16: does
spread buy the aug-graph overlap the guarantee line asks for?).** Per image on the o8 orbit
(L2-normalized): cloud radius r = mean view→centroid distance; d_intra = same-class centroid-NN
distance; overlap = med(r/d_intra); touch% = frac(2r ≥ d_intra); d_in/d_out = med intra-NN /
inter-NN. Regenerate: session script `e17_overlap.py` → `results/diag/e17_overlap.csv`.

| run | med r | med d_intra | overlap | touch% | d_in/d_out | (knn200) |
|---|---|---|---|---|---|---|
| lejepa ctrl / sigreg / sigreg_inv | .18/.47/.27 | .27/.53/.36 | .66/.90/.78 | 82.5/98.0/87.7 | .88/.92/.93 | 52.4/48.1/40.8 |
| lejepa-E12 C1 / f2 / f8 / f7 | .21/.63/.50/.44 | .32/.74/.70/.66 | .68/**.87/.73/.69** | 87.6/**99.7/95.3/91.2** | .89/.93/.90/.90 | 53.1/**60.5/60.2/57.6** |
| simclr ctrl / uniform / uniform_align | .33/.54/.39 | .47/.71/.59 | .72/.77/**.66** | 93.5/96.9/**81.0** | .95/.92/.90 | 49.5/55.9/52.1 |
| byol ctrl / align | .23/.42 | .28/.55 | .81/.77 | 98.3/96.6 | .96/**.94** | 46.0/51.4 |
| dino ctrl / protoce | .51/.38 | .65/.43 | .79/.88 | 98.2/95.0 | .93/**.85** | 59.9/55.7 |
| vicreg ctrl | .25 | .32 | .78 | 97.4 | .92 | 55.1 |

Mechanical: (i) **within every ±inv pair at fixed spread term, kNN tracks connectivity
monotonically** — f2→f8→f7 overlap .87/.73/.69, touch 99.7/95.3/91.2, knn 60.5/60.2/57.6;
uniform→+align touch 96.9→81.0, knn 55.9→52.1; sigreg→+inv .90→.78, knn 48.1→40.8. (ii) Across
arms no single dial: byol gains at already-saturated overlap via the local margin (d_in/d_out
.964→.935); protoce FATTENS the margin (.932→.849, strongest measured — prototype pulling, the
lin+2.6 side) while touch drops and knn falls (the fragmentation side). (iii) Intra-class NN
distances are 85–96% of inter-class everywhere — neighborhood margins are thin; cloud geometry
decides neighbors. f8 note: knn held (−0.4 vs f2) at touch 95.3 — the connectivity benefit reads
threshold-like (touch ≥ ~95), not linear in overlap.
**Figure: `results/figures/e17/e17_touch_vs_knn.png`** (`e17_overlap_fig.py`; per-family paths
control→+spread→+inv: every +inv leg moves down-left; sigreg's spread leg moves right-but-down —
connectivity gained, shape damage overrides).

**(n) H-wave completion + c015 (P1/P3 execution, 2026-07-16):** full results appended to the E12
card (f8 best-of-both: lin 67.0 = +2.4/c1 with knn held; gv2 settles the gv stake: GAP gains were
tap-local; floor-only gv2 zeroes vicreg's head jump +.474→+.005 — the mean-seeing floor does what
mean-blind var+cov cannot, cross-validating (c)). vicreg c015 extracted+probed (audit running,
job 62378897); gd/gdc re-extracted with fresh run_ids `*.ext2` for the gd2-P4 L-tap val stores
(jobs 62379959–62). Housekeeping flag: `features/in100.vicreg.s0.hpull_varcov.c015.now` = 07-15
churn leftover (o8-only store), purge-list candidate.

### vicreg CLOSURE — c015 (6×) converged scoring (2026-07-16; RAW; chain 62378895–97; `e17_score.py` vicreg arm re-pointed to c015)

| readout | ctrl | hpull_varcov@6× (c015, ep100) |
|---|---|---|
| h-triple pos / rand / margin | .921 / .711 / .210 | .636 / **.035** / .601 |
| cone: rand / centered / mu_share | .712 / .001 / .715 | .055 / −.000 / **.054** |
| inv-jump (jump-split) | +.479 (−.220, +.699) | **+.090** (+.067, +.023) |
| diagKL@h · effrank · kurt_worst · EP | 1.201 · 66.9 · 0.86 · 238 | **.028 · 156.6** · 0.72 · 95 |
| lin / knn200 @h | 64.5 / 55.1 | 65.1 (**+0.6**) / 59.4 (**+4.4**) |
| head σmax/layer | [10.6, 37.4, 11.3] | [14.0, 40.2, 11.2] |

Mechanism decomposition (v1L, `‖μ‖² / tr Σ / E‖x‖²`): ctrl 48.7 / 19.4 / 68.2 → c015 20.8 /
**361.7** / 382.5. The var hinge saturates centered variance (tr → ~d), diluting the mean share to
~.12 alone; the residual mean halves via drift not attributable to the (mean-blind) term's gradient.
**Correction log (owned): the ep15 sweep reads (neg@ep15 — 2×(.005) .464 · 4×(.010) .318 ·
6×(.015) .268; baseline 10%/.0026 → .341; control .711; shallow ~.05/doubling) extrapolated to
"needs 15–20×" were WRONG at convergence — the cone broke at 6× by ep100.** The 1×-dose ep100
behavior is UNKNOWN (baseline-dose run never extracted); the dose curve at convergence has one
point; c005/c010 ep100 ckpts retained for converged scoring if ever wanted. `hpull_varcov_inv`
closed unlaunched (decision). *(The interim "vicreg — PARKED" section that stood below
(Berker 2026-07-15 "we will talk about this later") is folded into this block, 2026-07-20.)*

## AGREED TAKEAWAY (jointly discussed 2026-07-16; Berker: "i approve t1-t7"; mirrored to D-039)

- **E17-T1 — The projector's "invariance jump" is mean-removal.** The h-cone is a mean offset
  (rand_cos ≈ ‖μ‖²/E‖x‖²; centered rand ≈ 0 in every run), the control head's margin jump is
  entirely rand-side — its alignment contribution is NEGATIVE in all four non-dino methods — and
  any mean-seeing term at h takes the job over (SIGReg/uniformity → rand ~0; byol-align partial;
  var+cov cannot — T5). Alignment was never the head's contribution; it is born in the trunk.
- **E17-T2 — h-satisfaction gentles the head, never relieves it; gentleness ⊥ quality.** The
  h→z map becomes more geometry-preserving in proportion to how FULLY the z-desiderata are met at
  h (h~z CKA: simclr .49→.74→.88 along ctrl→uniform→+align; sigreg_inv .955/procrustes .18
  ≈ isometry; partial satisfaction makes it wilder — sigreg; off-desideratum conditioning much
  wilder — f2, CKA .82→.46 on the best representation). Final z-terms end equal-or-worse under
  every h-pull except byol's (whose h-term IS its z-functional; its z-loss and z-probes improve).
  All four {gentle,wild}×{good,bad} cells are realized (uniform / sigreg_inv / f2 / sigreg):
  a gentle MLP is purchasable; it is not the prize.
- **E17-T3 — The transferable component is cone-removal + spread with cluster structure intact**
  (signature: effrank↑, worst-tail class-pair d′↑): simclr uniform +6.3 knn (+4.4 after
  centering), byol-align +4.1 lin/+5.4 knn (both spaces improve), E12-f2 +7.5 knn (cone-free —
  centering-insensitive). Distribution-SHAPE enforcement taxes: own-SIGReg (the one shape-seeing
  arm) drops mean class-pair d′ −17% and taxes both probes trunk-deep. Assignment reshapes:
  protoce fattens the local class margin (d_in/d_out .93→.85, lin +2.6) while fragmenting
  neighborhoods (knn −4.2, tax halves at k=20).
- **E17-T4 — Alignment and neighborhoods trade through augmentation-cloud connectivity.**
  Explicit inv at h shrinks the per-image view-clouds and disconnects the class manifold; spread
  reconnects it. Within every ±inv comparison at fixed spread term, kNN tracks touch%
  monotonically (f2→f8→f7: 99.7/95.3/91.2 vs knn 60.5/60.2/57.6; uniform→+align 96.9→81.0 vs
  55.9→52.1; sigreg→+inv). The benefit is threshold-like (f8 holds f2's kNN at touch 95.3), and
  shape-dominant mixing (f8, floor:inv 4.5:1) buys linear gain with kNN held — zero head-burden
  as a target is pyrrhic (rank collapse: sigreg_inv effrank 14.9 < ctrl 24.5). Figure:
  `results/figures/e17/e17_touch_vs_knn.png`.
- **E17-T5 — vicreg's own term removes the cone only indirectly and dose-hungrily; at 6× it
  joins the winners. [CORRECTED same-day — the bullet first drafted here used ep15
  extrapolations ("can only dilute", "needs 15–20×", rand ".264", probes flat); the CONVERGED
  c015 numbers refute that draft. Corrected wording CONFIRMED by Berker 2026-07-17 ("perfect now
  i agree") with the label: mean-agnostic ⇒ indirect-only removal (dilution + unattributed
  drift), hence dose-hungry.]**
  var+cov is mean-blind (code fact, stands): its cone-removal is INDIRECT — at 6× the var hinge
  recruits centered variance to saturation (tr Σ 19.4→361.7 ≈ d), diluting the mean share to
  ~.12 by itself, and the residual mean additionally halves via un-modeled drift (‖μ‖² 48.7→20.8;
  E‖x‖² NOT conserved 68→383, so not a norm-budget squeeze) → converged rand@h .035, mu_share
  .054. At that point the T7 pattern appears: **knn +4.4, lin +0.6, effrank 66.9→156.6, diagKL
  1.20→.028; inv-jump +.479→+.090**. So: weak PER UNIT DOSE, not incapable — the contrast with
  the mean-seeing floor is dose-efficiency (gv2 zeroes the jump at λ=.02, ~300× less relative
  pull), not possibility. Its z remains the least class-aligned space measured (B/T .16).
  `hpull_varcov_inv` stays closed unlaunched.
- **E17-T6 — "Gaussianity at h" decomposes; the lever = safety × activity.** Safety =
  cluster-blindness (a moments-1–2 objective cannot see, hence cannot spend, class-mixture
  structure; the shape channel can and does). Activity = non-absorbable, rotation-swept spectral
  floor (fresh-frame sliced full-cov KL forces genuine decorrelation the encoder can't fake;
  the axis-aligned version f5 is affine-absorbed and inert at 27× the dose). f5/f2/A3/sigreg
  quadrangulate: blind+absorbable = nothing; blind+non-absorbable at ε-dose = the winner;
  same at destination dose = whitening damage (E12-T1); seeing = damage at any tested dose.
  Gaussian shape is incidental — higher moments stay data-driven under f2 (Varimax coordinates
  remain leptokurtic, kurt .44). Declared-prior fork (finite-ν target) = the open discriminating
  arm, next session.
- **E17-T7 — Guillotine, per method.** lejepa: buffer genuinely protective (two damage channels
  measured: between-class shape-flattening; within-class rank collapse). simclr: the spread half
  of its mechanism belongs at h; the align half is free there anyway. byol: mechanism fully
  portable through one linear map — the anti-buffer datum (both spaces improve). dino: assignment
  reshapes rather than conditions (lin/knn dissociation). vicreg: wrong instrument for its own
  claim. Bordes' buffer survives specifically as the SHAPE-WORK hostel, not as a general
  protective layer.

Scope: IN-100 / ViT-S/16 / seed 0 / one dose point per arm (sigreg over-dose flag open: POS .615
mid-run; lower-dose tax unmapped); f2/f7/f8 comparisons ride the E12 package lane (own control).

### POST-CLOSURE ADDENDUM — T5 loose end: c015 μ-drift cadence (2026-07-17; RAW, mechanical; NO takeaway)

D-039 left the c015 ‖μ‖² halving UNATTRIBUTED. Cadence extracts ep25/50/75 (jobs 62388945–47) +
`experiments/e17_mu_drift.py` → `results/diag/e17_mu_drift.csv` (declared h = student.h.cls,
train500):

| ckpt | ‖μ‖² | tr Σ | E‖x‖² | mu_share | cos(μ, μ_ep100) | cos(μ, μ_ctrl) |
|---|---|---|---|---|---|---|
| ctrl (e17c) | 48.71 | 19.44 | 68.15 | .715 | −.07 | 1.00 |
| c015 ep25 | **81.99** | 334.3 | 416.3 | .197 | .62 | .24 |
| c015 ep50 | 35.63 | 355.0 | 390.6 | .091 | .72 | .11 |
| c015 ep75 | 25.47 | 359.1 | 384.6 | .066 | .89 | −.02 |
| c015 ep100 | 20.78 | 361.7 | 382.5 | .054 | 1.00 | −.07 |

Mechanical: (i) hinge saturation is EARLY — trΣ reaches 334/362 by ep25; after ep25 trΣ and
E‖x‖² are ~static. (ii) ‖μ‖² does not decay from the ctrl value — it first INFLATES to 82
(1.7× ctrl) with the norm explosion, then decays ×3.9 (82→21) against the frozen scale
background, decelerating (per-25-ep factors .43 / .72 / .82). (iii) the mean ROTATES while it
shrinks: cos to the ctrl cone .24 at ep25 → −.07 at ep100 (the converged residual mean is
ORTHOGONAL to the control's cone direction — not the old cone at reduced amplitude); even
within-run, ep25→ep100 cos is only .62. Candidate mechanism (discussion material, NOT a
takeaway): with per-dim variance gradient-defended by the saturated hinge and NOTHING in the
loss defending a nonzero batch mean, weight decay erodes precisely the loss-orphaned mean
component — predicting exactly this signature (decay that decelerates with the cosine-lr
schedule + a residual mean that is a rotating transient, not a persistent offset).

### POST-CLOSURE ADDENDUM 2 — reach v3 sweep read + o32 traversal (2026-07-17b; RAW; NO takeaway)

Figure: `results/figures/e17/e17_reach_3axis.png` (from `results/diag/e17_reach.csv`, 16 runs ×
100 classes, joined with e17_centered.csv — 13 runs have centered probes; f7/f8/c015 reach-only).

- **Run-level margin_max@α\* couples to orbit tightness and anti-tracks kNN.** Median margin_max
  rises near-monotonically with pos_c across the 13 joined runs; the high-kNN spaces (f2 60.5,
  dino 59.9, uniform 55.9) sit at LOW pos_c and LOW margin_max; sigreg_inv (knn 40.8) at the
  top. The α-sweep de-saturates the fixed-radius margin (all runs far above the y=x line vs
  perc_margin@r_mean; spread-orbit spaces rescued from margin≈0), and α\* orders spaces by
  operating point (f2 peaks at α=.62 and dies by α=.87; lejepa ctrl still climbing at α=1).
- **T4 transport check:** within the ±inv family f2→f8→f7, margin_max runs .585→.652→.740 —
  the exact INVERSE of T4's touch% (99.7/95.3/91.2, kNN 60.5/60.2/57.6). margin_max is a
  null-calibrated inverse of touch%: T4's "kNN tracks touch%" transports as "kNN anti-tracks
  margin_max within family", now collapse-safe. Q3 (standing readout + spec) is Berker's call.
- **o32 traversal landed** (job 62396059; ctrl/f2/sigreg_inv on o32 stores, sigreg/C1 fall back
  o8; `e17_traverse.png`). Walk (support distances over full 32-view clouds): f2 purity RISES
  with denser clouds (78%/58% on classes 0/42 vs 71% top at o8) while its hops stay the most
  expensive relative to d_inter (med 1.12–1.21×); sigreg_inv cheapest hops (.51–.63×), impure
  (48%/37%); ctrl 54%/39% at .75–.78×. INSTRUMENT FIX before the aniso read: the cloud
  top-eig-share null was hardcoded to (V=8, D=512) and the share used 8-view chunks — for o32
  rows both understate anisotropy. Fixed same-day (per-run gauss null at the store's actual
  V,D + full-cloud shares); corrected rerun job 62396106:
  - Full-cloud shares vs MATCHED nulls: f2 .180, ctrl .357, sigreg_inv .451 vs V=32 null .049
    (3.7× / 7.3× / 9.2× null) — anisotropy confirmation STRENGTHENS at denser clouds and the
    f2 < ctrl < sigreg_inv ordering sharpens. (o8 rows for sigreg .427 / C1 .439 vs null .170 =
    2.5×/2.6× — different V, ratios not directly comparable across provenance.)
  - Axis alignment at full clouds: ctrl walk-adjacent |cos| .349 vs random-pair .228 — the
    o8-based "class-global not filament-local" reading softens for ctrl (adjacent > random by
    .12); sigreg_inv .323 vs .281 ≈ class-global; f2 low on both (.135/.096). Two pilot
    classes only — wide error bars.

**Cloud-intersection instrument (Berker's organization hypothesis, direct form; 07-17b;
`experiments/e17_intersect.py` → e17_intersect.csv + e17_intersect.png).** Hypothesis as
stated: correct organization ⇔ same-class cloud intersection beats the negs. Iterated v1→v4
in-session; the nulls are findings:

- **v1 (Schilling 1-NN mixing): ω ≈ .003–.02 in every space** — a view's nearest neighbor is
  essentially always an own-cloud sibling; NO interpenetration at within-cloud spacing scale.
- **v2 (view-in-r_mean-ball depth): ≈ 0 everywhere EXCEPT sigreg_inv** (ω_frn .055 > ω_same
  .028, AUC .35): the collapsed-cone space is the only one with literal point interpenetration
  — and it interpenetrates with NEGATIVES. High-d concentration: ball-touching (the
  percolation criterion) almost never implies point-containment.
- **v3 (signed ball-overlap depth on the centroid axis, (r_i+r_j−d)/(r_i+r_j)):** dynamic
  range restored. Global-tail read: the 10 nearest foreign clouds (of ~9900) overlap DEEPER
  than the top-20 same-class (of 99) in EVERY space (AUC .12–.30; e.g. ctrl ω_same −.008 vs
  ω_frn .071; f2 .395 vs .413 — everything touching, foreign deeper at the tail; sigreg_inv
  −.005 vs .255 — same-class balls don't even touch while negs overlap deep). Confound
  identified before interpretation: 100× pool-size order-statistics advantage for the foreign
  side ⇒ this contrast measures the extreme foreign tail kNN actually meets, NOT the fair
  organization claim.
- **v4 = pool-matched AUC (reach-null convention, 99-vs-99 candidates, B=3 draws) — LANDED
  (job 62396575; e17_intersect.png + csv).** The fair "better than the negs": **AUC .854–.975
  in EVERY space** — the hypothesis holds universally at matched pool and is therefore
  necessary-not-discriminating as a run-level scalar. The discriminating object is the ω LEVEL
  structure (panel 2): spaces split into a **touch regime** (ω_same > 0 — typical same-class
  top-20 balls actually overlap: f2 +.397 the deepest; sigreg +.220; uniform +.211; dino ctrl
  +.246; vicreg ctrl +.183; c015 +.261) vs a **gap regime** (ω_same < 0 — same-class typically
  does NOT touch; discrimination is purely metric: lejepa ctrl −.019, C1 −.017, sigreg_inv
  −.011, uniform_align −.041). Matched-pool negs are mostly deep-negative (lejepa ctrl −.60,
  sigreg_inv −.69) while the GLOBAL-tail negs are positive and deeper than same-class
  EVERYWHERE (v3 row above). Within-lejepa, kNN ordering tracks ω_same (f2 .40 > f8 .14 >
  f7 .06 > ctrl/C1/sigreg_inv ≲ 0) with sigreg the exception (+.22, knn 48.1 — overlapping but
  shape-taxed), i.e. touch-regime connectivity reads necessary-not-sufficient, T4's
  threshold-like pattern again. Scope: cross-method ω levels carry the D-004 aug-family caveat
  (per-method view pipelines set r_mean and cloud geometry); within-family reads are clean.
  AUC ceiling compresses differences — read levels, not ranks, going forward.
- **Touch fractions (the path readout; rerun 62396656 + explainer figure
  `e17_intersect_explainer.png`):** fraction of the 20 nearest same-class clouds reachable
  gap-free (ω>0 ⇔ free percolation edge): f2 1.00 · dino ctrl .97 · c015 .97 · uniform/byol
  ctrl .93 · vicreg ctrl/sigreg .89 ‖ gap regime: sigreg_inv .56 · C1 .53 · lejepa ctrl .52 ·
  uniform_align .49 (≈half link ⇒ SPARSE chains — still percolates: 10 linked neighbors per
  instance ⇒ reach's perc_seed .87 for lejepa ctrl is consistent). Tail negs touch .77–1.00
  EVERYWHERE (the 10 globally nearest foreigners essentially always overlap the anchor);
  matched-pool negs touch .03–.08 in tight spaces vs .74–1.00 in open ones (f2 1.00 — in f2
  literally everything touches everything; discrimination is depth-ranking, not touching,
  the reach-anatomy ordering-vs-threshold resolution again).

**Touch CENSUS (Berker's spec verbatim, superseding the nearest-20/pool-draw design;
`experiments/e17_touch_census.py`, job 62396715 → e17_touch_census.csv +
e17_touch_census.png).** Every anchor × every candidate, ω>0 counted: p_pos = P(random
same-class cloud touches you | 99), p_neg = P(random negative touches you | ~9900), enrichment
E = p_pos/p_neg (ratio of class means), purity = deg_pos/(deg_pos+deg_neg) vs base rate .0099.
Aggregation: class rows = means over the class's ~100 anchors; run = median over classes;
α=.75 rides along. RAW:

| touch census α=1 | p_pos | p_neg | E | purity | knn200 |
|---|---|---|---|---|---|
| lejepa/ctrl | .129 | .0069 | 19.1 | .160 | 52.4 |
| e12/C1 | .139 | .0103 | 15.1 | .131 | 53.1 |
| lejepa/sigreg_inv | .165 | .0158 | 9.6 | .088 | 40.8 |
| dino/protoce | .313 | .0388 | 8.5 | .079 | 55.7 |
| simclr/uniform_align | .147 | .0247 | 5.9 | .056 | 52.1 |
| e12/f7 | .273 | .0440 | 5.5 | .052 | 57.6 |
| byol/align | .408 | .0956 | 4.1 | .039 | 51.4 |
| lejepa/sigreg | .465 | .1252 | 3.7 | .035 | 48.1 |
| simclr/ctrl | .372 | .0975 | 3.6 | .035 | 49.5 |
| vicreg/ctrl | .454 | .1290 | 3.3 | .032 | 55.1 |
| byol/ctrl | .517 | .1777 | 2.8 | .028 | 46.0 |
| e12/f8 | .441 | .1711 | 2.5 | .024 | 60.2 |
| dino/ctrl | .708 | .3606 | 1.9 | .019 | 59.9 |
| simclr/uniform | .604 | .3489 | 1.7 | .017 | 55.9 |
| vicreg/varcov_c015 | .743 | .4439 | 1.6 | .016 | 59.4 |
| **e12/f2** | **.995** | **.9182** | **1.1** | **.011** | **60.5** |

Observations on record (no takeaway): (i) E > 1 in all 16 runs — the weak form of the
organization hypothesis is universal. (ii) Within-family, the kNN winner has HIGHER p_pos and
LOWER enrichment/purity in 4 of 5 families (lejepa-E-wave, dino, simclr, vicreg; byol
INVERTS: align 4.1/51.4 vs ctrl 2.8/46.0). (iii) f2 is touch-SATURATED at r_mean (p_pos .995,
p_neg .918, purity .011 ≈ the .0099 base rate): its touching graph carries ~no class signal —
the class signal lives in depth ordering among the touching (the reach-anatomy
ordering-vs-threshold resolution, census form). (iv) α=.75 amplifies all enrichments
(f2 1.8, ctrl 47.1) without changing the ordering. Caveats: o32 stores (ctrl/f2/sigreg_inv)
vs o8 radii differ by the V-estimator bias ≈ 5% (small vs the measured contrasts);
cross-method levels carry the D-004 aug-family scope.
