# `two_space_story.tex` — the lean talk deck (built 2026-08-26)

Beamer deck that tells Berker's three-paragraph story with **exhibits that already exist**.
Built per `SESSION_OPENER_SLIDES.md`; the parallel main-line session owns `docs/paper/`,
the cards and the exhibit generators — this deck only *consumes* artifacts.

```bash
cd docs/slides && pdflatex two_space_story.tex   # twice; no compute, ~3 s
```

Figures are referenced **by path**, so regenerating an exhibit (e.g.
`python experiments/paper_exhibits.py`) flows into the deck on the next compile.

## Slide map

| # | slide | exhibit | status |
|---|---|---|---|
| 1 | title | — | — |
| 2 | ¶1 the gap | `docs/paper/figures/fig_discrepancy.png` | **AGREED** (CHECKLIST C1) |
| 3 | ¶1 the accessibility bridge + rank/dimensionality | `results/figures/paper/C3b_accessibility_profile.png` | **RAW** (C3b-2) + the **AGREED** C3 within-pair Δ |
| 4 | ¶2 the objective | `docs/paper/main.tex` eq.(6.1), Prop 3.2, appendix (slices) | THEORY, quoted verbatim |
| 5 | ¶2 stable center capacity | `assets/depth_capacity.png` (built here by `make_depth_capacity.py`) | **RAW** — depth reading not yet agreed; the AGREED C2 row covers the endpoint only |
| 6 | ¶2/¶3 treatment across the zoo | `docs/paper/figures/fig_treatment_arrows.png` | **AGREED** C8 wording; VISReg arrow unadjudicated (D-105) |
| 7 | ¶3 where the gain comes from + how much variation | `docs/paper/figures/fig_org_vs_sensitivity.png` | **AGREED** (C8, C7) + THEORY (Thm 5.2 i/ii/iii) |
| 8 | competitive standalone | hand-built (not `tab:in1k-main`) — see frame note below | **RAW** — C9 = strategy pending |
| 9 | summary | the three paragraphs verbatim + exhibit pointers | — |

## Notation on the method slide (Berker 2026-08-26)

Slide 4 is written to stand alone for someone who has not read the paper, so it does **not**
use the paper's symbols. The mapping:

| slide | `docs/paper/main.tex` |
|---|---|
| $\widehat\Sigma_s$ ("covariance of the centers") | $\widehat B_{s,V_s}^{\rm train}$ |
| $\mathcal K(\mu,\Sigma)$ | $\mathcal K_{\rm ctr}(\mu,\widetilde B)$ |
| "Cov(centers) = Cov(exact centers) + Cov(views of one image)/V" | $\widetilde B_{s,V}=B_s+A_s/V$, eq. (3.4) |

$V_z$/$V_h$ were dropped (both taps average the same $V$ views in the implementation), and
$S$, $A_s$, $B_s$ never appear — the pooled-center identity is stated in words instead.

## Rules this deck follows

- Every caption carries **AGREED** (usable as a claim, per `docs/paper/CHECKLIST.md`),
  **RAW** (data shown, reading not yet agreed) or **THEORY** (proved in the paper).
- Vocabulary = `docs/GLOSSARY.md` + `docs/paper/main.tex`. Retired words appear nowhere.
  Where a *rendered figure legend* still says "+moment floor", the slide says
  "the h-side moment term" and the path line notes the rename.
- `C11_too_thin.png` is **not** used (ruled out of the paper 2026-08-25). ¶3's "neither too
  little nor too much" rides Thm 5.2(i)+(ii/iii), C7's agreed point and the W/B language.
- No new measurement, no re-plot, no crop: every PNG is included verbatim.

## Slide 8: why the table is hand-built (Berker 2026-08-26)

It deliberately does **not** mirror `tab:in1k-main`. Rows are grouped so each block is
frame-comparable, and a **Views** column declares the training frame:

- **ViT-S block** — LeJEPA (Lightly repro. 64.1), DINO (OK-AI 70.0), iBOT (OK-AI 70.9) all at
  2×224+6×96 (616 tok/sample); **Ours = `d256vm4`** at 4×224 (788 tok, ×1.28). Berker ruled
  vm4 stays: *"it is still comparable although not exact"*. The exactly-matched 616-tok cells
  exist if ever wanted: `e27lmc` 64.55/48.79, `e27lmcse` 65.69/50.08, `e27lmcs5` 63.76/53.50.
- **ViT-B block** — the clean pair: VISReg-B under **their** code (`in1k.visreg.s0.vrb.extL`,
  bench 70.35) vs **Ours** `e27lm4sbe` (71.17/56.04), both 4×224+6×96 (1010 tok), 100 ep,
  one eval protocol. OK-AI's B rows are deliberately absent — they run 616 tok, so putting
  them here would compare our run at ×1.64 their compute.
- VISReg-B repro has **no kNN** (`*.bench_knn.csv` was never produced) — hence the `---`.

## Open items (refresh, don't rewrite)

1. **`s5b` is a ViT-B run, not ViT-S** (`frame.model_name=vit_base_patch16_224`, verified from
   the checkpoint and its wandb config). `HANDOVER.md` and `SESSION_OPENER_SLIDES.md` both call
   it "the S-100 comparison vs OK-AI S externals" — that is wrong, and the correction belongs
   to the main session. Its bench (landing 08-26 evening) is a second B cell, not an S row.
2. **DINO in the depth figure** needs its winner-arm (`e20fwlo`) layer stations; that audit was
   still running on 08-26. Rerun `make_depth_capacity.py` to pick it up.
3. **The L row** fills from the `e27lm4Ls5b` bench (~08-27).
4. Decisions taken with Berker for this deck (2026-08-26): beamer / ~8 slides ·
   C3b **profile** variant only · RAW exhibits allowed **if visibly tagged** · capacity figure
   across layers (trunk → projector → z), not epochs · ViT-S row stays vm4 · ViT-B compared
   against VISReg only.
