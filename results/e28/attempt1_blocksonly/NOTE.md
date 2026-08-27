# attempt 1 — blocks-only ×10 (SUPERSEDED at ~ep10, 2026-08-13)

Job 63437129. The ×10 was applied to the 48 block weight tensors only; `patch_embed.proj`
was left unscaled (the declared reading of the spec). **Berker overruled it the same
afternoon: "i think patch embed proj should be scaled too."** The canonical arm
`in100.floorssl.s0.e28x10` was relaunched with patch_embed included; these files are the
2 epochs of the superseded configuration, kept rather than deleted.

Its init read (ep0, blocks-only): h_raw effrank 180.3 / stable rank 12.6 / rms 61.5.
The same quantity for BOTH variants is measured properly by `+e28.mode=initsweep`
(`results/e28/e28_initsweep.csv`), so nothing here is load-bearing.
