# E30 — Compute-matched us-vs-VISReg SSL-transfer comparison at ViT-B (VISReg Table-5 protocol)

## (a) Directive
Berker 2026-08-26 ~17:30 (HANDOVER §FIRST TASK, executed next session same evening):
build + LAUNCH the compute-matched us-vs-VISReg SSL-transfer comparison at ViT-B —
protocol = VISReg Table-5 verbatim, their code = the oracle (deviations declared),
port-validation on their DINO-B row FIRST, then hand the wheel to Berker for the
paper. This is the first item of the agreed paper build order (transfer/OOD evaluator
→ I-JEPA H/14 GAP lift → data2vec lift → ADE20k seg).

## (b) The matched pair (D-097 anchor principle: frame+FLOP+epoch-matched)
- **ours** = `outputs/in1k.floorssl.s0.e27lm4sbe_ep100.pt` — ViT-B/16, 100 ep,
  lightly_mc 4g+6l@96, bench_linear_v1 **71.17**.
- **VISReg anchor** = `~/visreg_repro/checkpoints/visreg/visreg_vit_b_bs128_lamb0p9_
  lr9em4_projdim256_numproj4096_ng4_nl6/checkpoints/latest.pt` — THEIR code, ViT-B/16,
  100 ep, shipped 4g+6l config; our-bench **70.34** (`in1k.visreg.s0.vrb.extL`),
  their-own-protocol in1k linear **71.99** (job 63597325). Do NOT purge (PURGE doc
  keep-anchor rule).

Same arch, same epochs, same view geometry = the fair-comparison basis — state it
explicitly wherever the numbers land. The PUBLISHED Table-5 VISReg-B row is **400 ep**
(and DINO-B is 400 ep, iBOT-B 400 ep): the compute-matched read is anchor-vs-ours;
published rows enter any exhibit as context rows, never as the matched comparison.

## (c) Protocol (their code = the oracle)
Port of `HaiyuWu/visreg@47b1cf4 downstream/linear_prob/run_evaluation.py` (donor
read-only in `third_party/visreg`; full port record + declared deviations in the
`experiments/transfer_visreg.py` docstring — the PORT_NOTES for this evaluator):
frozen encoder; concat last-4-block CLS (norm on) → 4·D; per-LR heads
BatchNorm(affine)+Linear (N(0,.01)/0); 13 base LRs [1e-4..0.5]·total_bs/256; SGD
m=.9 wd=0; cosine→0 per epoch; bs 32; bf16; RRC(224,(.08,1))+flip train aug,
256→CC224 test; seed 42 per dataset; best head selected ON TEST (their operative
`train_online_and_eval`; the val-split variant is dead code in their `main()`).
**Epochs = 10** (paper Table-5 caption: 10 ep for downstream sets; the 100-ep default
in their script belongs to the Inet1K column — our in1k comparison already exists via
bench + their-lin, so the in1k column is not re-run here).
Datasets = their `DEFAULT_EVAL_DATASETS`: dtd, aircraft, cars, cifar10, cifar100,
flowers, food, pets — split parity with their loaders asserted in-code (canonical
counts); data = the E25 staging (`~/data/ssltransfer`, 19G, already on disk) + the
tanganke cars HF mirror (E25 deviation 3).

## (d) Port-validation gate (pre-registered, committed before any numbers exist)
Cell `tvlp-dinob` = `timm:vit_base_patch16_224.dino` — the official DINO ViT-B/16
(400 ep), exactly what their `create_backbone(pretrained="dino_v1")` loads for their
Table-5 DINO-B row, which the paper marks as run by the authors (no borrowed-value
asterisk). **Target = their printed row: DTD 74.3 · Aircraft 63.6 · Cars 73.9 ·
CIFAR-10 96.5 · CIFAR-100 85.0 · Flowers 94.6 · Food 83.1 · Pets 93.6 · Avg 83.1.**
The evaluator is trusted when per-set deltas look like aug/seed noise (|Δ| ≲ 1 pt,
no systematic sign); anything larger = a protocol divergence to diagnose before the
pair is read. **The pair numbers are not trusted until this gate passes.**

## (e) Cells + jobs (launched 2026-08-26 evening)
Smoke `tvlp-smoke` 63732298 (dtd, 1 ep, native adapter) — gate for the wave; the
three cells ride `--dependency=afterok` on it:
- `tvlp-dinob` **63732742** — port-validation cell (timm DINO-B), 10 ep, 8 sets →
  `in1k.pub.dinob400.visreg_lp.csv`.
- `tvlp-sbe` **63732743** — ours, `native:` on e27lm4sbe_ep100 →
  `in1k.floorssl.s0.e27lm4sbe.visreg_lp.csv`.
- `tvlp-vrb` **63732744** — the anchor, `visreg:` trunk lift (same as the vrb bench) →
  `in1k.visreg.s0.vrb.visreg_lp.csv`.
CSVs → `results/transfer/<tag>.visreg_lp.csv` (dataset-granular resume; wall-safe).

## (d2) Port-validation gate ADJUDICATED: PASS (2026-08-26 ~19:0x; job 63732742,
## 36m49s on gpu287)
ours-eval vs their print, per set (Δ): DTD 74.7/74.3 (+.4) · Aircraft 64.0/63.6
(+.4) · Cars 74.0/73.9 (+.1) · C10 96.5/96.5 (0) · C100 84.9/85.0 (−.1) · Flowers
94.3/94.6 (−.3) · Food 83.1/83.1 (0) · Pets 93.4/93.6 (−.2) · **Avg 83.1/83.1**.
Every |Δ| ≤ .4, mixed signs, avg exact — inside the (d) criterion. **The evaluator
is trusted; the pair read is unblocked.** CSV `in1k.pub.dinob400.visreg_lp.csv`.

## (f) Pre-registered predictions
OWED — the directional-prediction round is joint and happens with Berker at the wheel
(he takes the paper next); nothing is committed unilaterally. The only pre-committed
read is the port gate in (d).

## (h) The pair — LANDED (RAW, 2026-08-26 ~19:25; sbe 34m04s, vrb 51m44s, single GPU each)
| cell | DTD | Aircraft | Cars | C10 | C100 | Flowers | Food | Pets | **Avg** |
|---|---|---|---|---|---|---|---|---|---|
| ours `sbe` (B·100) | 71.4 | 55.2 | 64.8 | 95.3 | 82.0 | 90.4 | 79.1 | 89.0 | **78.4** |
| anchor `vrb` (B·100) | 73.3 | 53.1 | 58.0 | 94.3 | 79.6 | 87.9 | 79.1 | 85.5 | **76.4** |
| Δ (ours−anchor) | −1.9 | +2.2 | +6.8 | +1.0 | +2.3 | +2.5 | +0.0 | +3.5 | **+2.0** |

Matched basis (state wherever quoted): ViT-B/16 · 100 ep · 4g+6l view geometry
(D-097 anchor principle); evaluator = the port-validated Table-5 protocol (gate §d2).
Raw description: ours higher on 7 of 8 sets (food a tie, Δ+.02; cars the largest
gap +6.8), dtd the anchor's set (−1.9). Context rows at a DIFFERENT budget (never
the matched claim): their published 400-ep VISReg-B avg 79.1 (anchor's 100-ep 76.4 →
+2.7 for 4× epochs), DINO-B 400-ep avg 83.1. Takeaway = joint read owed (predictions
§(f) deferred to the joint round by design).

## (i) L-rung extension (Berker 2026-08-26 eve: "we also have vit large checkpoint
## dont we? do ssl transfer on that too." → follow-up: "lets not care about visreg's
## but run ours.")
`tvlp-Ls5b` **63741582** — ours `native:` on e27lm4Ls5b_ep100 (ViT-L/16, 100 ep,
lightly_mc 4g+6l) → `in1k.floorssl.s0.e27lm4Ls5b.visreg_lp.csv`. A public-VISReg-L
cell (`tvlp-vrl` 63741584, pubvit on their released 400-ep L/14) was launched then
CANCELLED 7 s in on the follow-up directive — no rows written; at L the comparison
context = their PRINTED Table-5 L rows only (VISReg-L/14·400 avg 78.5, iBOT-L/16·250
avg 84.5), quoted with the epoch mismatch declared (no 100-ep L anchor exists).

**LANDED (RAW, 43m39s):** ours L·100 — DTD 71.3 · Aircraft 53.3 · Cars 63.6 · C10
96.0 · C100 83.7 · Flowers 89.9 · Food 80.1 · Pets 88.9 · **Avg 78.3**. Raw
observations: avg sits at the B·100 level (78.4 → 78.3); per-set vs our B the larger
sets move up (C100 +1.7, Food +1.0, C10 +0.7) and the fine-grained sets move down
(Aircraft −1.9, Cars −1.2). Vs the printed 400-ep context rows: 0.2 under
VISReg-L/14·400 (78.5) at ¼ the epochs (mismatch declared). Joint read owed.

## (g) Read protocol
(1) Port gate (d) adjudicated first, per-set deltas quoted raw. (2) Then the pair,
row vs row (per-set + avg), RAW until jointly read; caption discipline: the matched
basis (arch/epochs/geometry) stated wherever quoted; published 400-ep rows = context
only. (3) Landing exhibit target: the paper's transfer table block
(`experiments/paper_exhibits.py` `_transfer_seg_tables`) once the read is agreed.
