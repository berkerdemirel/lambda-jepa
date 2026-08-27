# Slidev port of the two-space deck

`slides.md` is the same 9-slide story as `../two_space_story.tex` (beamer) — same spine
sentences, same exhibits, same captions. The beamer deck stays the reference build; this
is a parallel rendering, not a replacement.

## Run it

Needs Node ≥ 18 **and npm** — this cluster's login node has Node 18.20.4 but **no npm**, so
the deck has never been rendered here. On a machine with npm:

```bash
cd docs/slides/slidev
npm install
npm run dev        # live preview at localhost:3030
npm run export     # PDF (pulls playwright-chromium on first run)
```

Or without installing anything: `npx @slidev/cli slides.md`.
If your Node is older than the pinned CLI wants, `npx @slidev/cli@0.49 slides.md` works too.

## Figures

`public/` holds **symlinks** into the real exhibit locations, so a regenerated figure flows
into the deck on reload — nothing is duplicated:

| file in `public/` | points at |
|---|---|
| `fig_discrepancy.png` | `docs/paper/figures/` |
| `fig_treatment_arrows.png` | `docs/paper/figures/` |
| `fig_org_vs_sensitivity.png` | `docs/paper/figures/` |
| `C3b_accessibility_profile.png` | `results/figures/paper/` |
| `depth_capacity.png` | `docs/slides/assets/` (built by `../make_depth_capacity.py`) |

If a build environment refuses to follow symlinks in `public/`, replace them with copies
(`cp -L`) — but then remember they go stale when an exhibit is regenerated.

## Differences from the beamer build

- Styling lives in `style.css` (Slidev auto-loads it). Palette matches the beamer deck and the
  exhibit generators: ink `#1A1A1A`, muted `#6B6B6B`, control violet `#4A3AA7`, treated green
  `#008300`.
- Math is KaTeX, not LaTeX: `\rm` was rewritten to `\mathrm`, and the display equation uses an
  `aligned` block. Everything else is the same source text.
- `fonts.provider: none` — no Google Fonts fetch, so it renders offline.
- The status discipline of the source deck still applies: claims ride `docs/paper/CHECKLIST.md`
  (AGREED / RAW / THEORY), and the ViT-S rows of the ImageNet-1k table are not exactly
  frame-matched — see `../README_two_space_story.md`.
