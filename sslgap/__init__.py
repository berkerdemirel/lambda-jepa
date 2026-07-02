"""sslgap — Loss Space != Representation Space: a two-space audit of SSL desiderata.

h = trunk forward_features output (CLS / patch-GAP; per-layer taps) — the probed representation.
z = head taps (projector / predictor / decoder / prototype-bottleneck) — where the loss lives.
Space naming: "<branch>.h.<kind>[.L<layer>]" and "<branch>.z.<role>[.<tap>]" (PROTOCOL.md §1, §3).
"""

__version__ = "0.1.0"
