# Head hypotheses: Lipschitzness, augmentation-orbit overlap, MLP burden

**Status: hypothesis note (Berker, in-conversation 2026-07-13, mid G-wave session). Discussion
material — no numbers, no takeaways. Measurement design happens in-conversation in a later
session; only the feature extraction was pre-launched (record at the bottom). Tags [P]/[O] per
repo convention.**

Companions: `OPEN_PROBLEMS.md` (OP-1 transfer law, OP-2 readout-layer selection, OP-4
nonlinearity mystery) · `LDM_HEAD_COMPOSITION.md` (OP-17 linear-head wedge) ·
`docs/experiments/E02_guillotine_desiderata.md` (locked depth-axis card these hypotheses extend)
· E12 card §G-wave (the result H3 proposes a mechanism for).

## H1 — cosine invariance vs head Lipschitzness [O]

Berker: "at some point we will want to measure the cosine similarity at h and z with respect to
the lipschitzness of the MLP."

Measure view-pair cosine similarity (and its pair_margin-normalized forms) at h and z **jointly
with the head MLP's Lipschitz constant** — per-layer spectral norms from the checkpoint weights,
plus an empirical local-Lipschitz readout on the stored view pairs (‖g(h_A)−g(h_B)‖/‖h_A−h_B‖
over positive pairs). A 1-Lipschitz-normalized head cannot *manufacture* invariance faster than
it contracts everything; how much invariance appears across the head relative to its contraction
budget is a quantitative form of OP-1's transfer law.

Inputs: checkpoint head weights (on disk, no extraction needed) + per-tap view features (the
orbit stores below). No estimator is declared yet — design in-conversation.

## H2 — the overlap-depth rule: "the deepest layer with fair overlap is the most useful" [O]

Theory background (guarantee literature): alignment of positives transfers to class structure
only when augmentation distributions of different samples **overlap** — intra-class connectivity
of the augmentation graph (HaoChen–Wei–Gaidon–Ma, NeurIPS 2021, in `BIBLIOGRAPHY.md`; the
overlap-named variant is Wang et al., ICLR 2022, "Chaos is a Ladder" — *from memory, cite to be
verified before any card cites it*).

Berker's hypothesis: at the loss space z, the per-image augmentation clouds are essentially
**disjoint** — no overlap, so the guarantee machinery is vacuous there. Walking back through the
head and into the trunk, clouds should begin to overlap. **The deepest layer where overlap is
still "fair" is the most useful readout layer.** This is:

- a candidate **label-free rule for OP-2** (readout-layer selection — currently folklore);
- a **mechanism refinement of E02**: the locked E02 card predicts per-metric decay profiles and
  accuracy peaks along h.L03→L06→L09→L12→z-taps; H2 predicts *where and why* the accuracy peak
  sits (at the overlap frontier), i.e. a specific coupling between an unsupervised overlap curve
  and E02's probe curves ("similar to guillotine" — Berker).

Measurement sketch (to be designed, NOT declared): per-image orbit clouds (V views) at every tap;
overlap read as e.g. cloud radius vs distance-to-nearest-other-image-cloud, kNN two-sample
statistics, or class-conditional cloud intersection — estimator choice is exactly the kind of
re-metrization-sensitive decision E12 taught us to normalize within-space (pair_margin lesson).

## H3 — MLP burden: "ViTs are smoother, MLPs can go crazy" [P]

Berker: the head MLP is where feature organization jumps discontinuously; **the E12 moment-KL
floor at h works by taking burden off the MLP's shoulders** — the constraint absorbs (part of)
the re-metrization work the head would otherwise do, leaving a "less crazy" head; the trunk,
under its architectural inductive biases, stays smooth.

Predictions this makes for the extracted pairs (arm vs matched control, per tap):
- the arm's per-layer curves (invariance, probes, geometry stats, H1's Lipschitz constants)
  should show a **smaller discontinuity at the first head layer** than the control's;
- head spectral norms / empirical Lipschitz should be tamer in the arm;
- connects to the G-wave cross-tap datum (`results/diag/e12g_tap_deltas.csv`): the linear tax is
  local to the floored tap while neighboring taps improve — consistent with the constraint doing
  work the head no longer has to.

Links: OP-4 (why nonlinear heads at all), OP-17/`LDM_HEAD_COMPOSITION.md` (a linear head
preserves an affine-readable subspace; a "less crazy" MLP is closer to that wedge).

## Extraction record (launched 2026-07-13; measurement deferred)

Runs: the three calibrated-vs-control pairs at final ckpt — `in100.lejepa.s0.e12f2/.e12c1`,
`in100.vicreg.s0.e12gv/.e12gvc`, `in100.dino.s0.e12gd/.e12gdc` (existing `.ext` run_ids gain new
store keys; landed dirs untouched).

Per run (one `slurm/extract.sbatch` job each on the two H100 singleton slots
`h100-slotA`/`h100-slotB` (Berker: "use h100, we have 2 idle h100 budget"), `adapter=native`,
`bs=128`, `h_layers=[3,6,9]` = E02's declared trunk points below L12≡h):

- `in100.pairs100.v1@<stack>.o8` — **orbit stores**: V=8 views/image, 10k pairs-manifest images,
  every branch × {h.cls, h.gap, h.{cls,gap}.L{03,06,09}, all head z-taps} × view0..7, rows
  image-aligned with labels and with the existing V=2 pair stores (same manifest, no shuffle).
  Stacks: `audit_v1` + method's own (`own_vicreg`, `own_dino`; lejepa's own ≡ audit_v1).
  Asymmetric own_dino alternates its two global-crop transforms across views (meta `view_rule`).
- `in100.train500.v1L` / `in100.val.v1L` — clean eval features with the same per-layer taps, for
  converged per-tap probes (kept separate from the landed `*.v1` dirs so nothing is rewritten).

Budget: ≈ 25 GB fp16 (store was at 61/500 GB, D-005 cap enforced by `store.py`); purge-after-table
applies. Machinery: `OrbitDataset` (`sslgap/data.py`), `extract_views`
(`sslgap/extract/extractor.py`), `orbit_v` in `experiments/configs/extract.yaml` — the same
switches unlock the parked E02 chain for the M2 lanes when pulled.
