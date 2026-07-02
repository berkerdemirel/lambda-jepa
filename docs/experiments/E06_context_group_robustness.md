# E06 — Foreground/background conflict, context swap & group robustness   `STUB — detailed when scheduled (see docs/ROADMAP.md)`

> Derived from docs/report/ssl-projector-gap-report.html (compiled 2026-07-01, quote-verified there). Tags: [E] established · [P] plausible interpretation · [O] open question.

**Status:** stub · phase M5+ (per ROADMAP index) · **Pre-registered:** no

## Hypothesis (from report, to be refined before pre-registration)

Frozen SSL features carry background shortcuts into downstream probes: worst-group accuracy on
Waterbirds ranks DINOv2 ≥ contrastive > MIM at *mean* accuracy but with **inverted worst-group
ordering** (strong semantic features can ride strong context features). Two-space twist: z
(invariance-filtered) may be *more* group-robust than h for view-based methods — the head may
absorb some context sensitivity. [O]

## Protocol sketch

- IN-9 suite (mixed-same / mixed-rand / only-bg-t / only-fg) for all models, both spaces, fixed
  probes.
- Waterbirds + CelebA-blond linear probes with worst-group accuracy (standard group-DRO eval, ERM
  probe).
- Context-swap retrieval: foreground pasted on conflicting background — does h retrieve by fg or
  by bg?

## Datasets / models

IN-9 / Backgrounds Challenge; Waterbirds; CelebA-blond; context-swap sets (CounterAnimal-style if
available, else synthesized fg/bg recombination from IN-9 — §5.0). All §5.0 models, both spaces.

## Expected failure to reveal

BG-gap (mixed-same − mixed-rand) substantial for all SSL, largest for multi-crop models;
worst-group accuracy dramatically below mean for everyone (no SSL objective addresses it).

## Interpretation guide

Fills the named literature hole — no canonical SSL group-robustness audit exists (§4.5;
OPEN_PROBLEMS OP-9). The h-vs-z group-gap tests whether "the projector absorbs context reliance" —
a genuinely open two-space question. Also note §4.5 caveat: the probe layer "itself can be
susceptible to spurious correlations" (Shi et al.) — probe choice is part of the measurement.

## Dependencies (features/datasets/models needed)

- Datasets to stage: IN-9 suite, Waterbirds, CelebA-blond; context-swap set (build from IN-9 if
  CounterAnimal-style unavailable).
- Group labels wired into the probe harness (worst-group reporting).
- Frozen features for these sets in both spaces; fixed probes per PROTOCOL.

## Results

*(empty)*

## AGREED TAKEAWAY

*(empty — never filled unilaterally; see CLAUDE.md)*
